"""On-screen text (OCR) for images and video frames, using PaddleOCR.

`analyze(path, **opts)` is the pure entry point: it samples frames with ffmpeg (or reads the image, or the
stills it is given), runs one PaddleOCR pipeline per requested language and returns a JSON-serialisable dict.
`OcrEnricher.enrich` calls it for every image/video `Media` and stores the result under `ocr`.
"""

import difflib
import json
import os
import re
import subprocess
import tempfile
import threading
import unicodedata
from collections import Counter

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from auto_archiver.core import Enricher, Metadata
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOL = "paddleocr"

# script groups -> PaddleOCR `lang` codes; any other value is passed to PaddleOCR as a raw lang code
LANGUAGE_GROUPS = {
    "latin": "en",
    "arabic": "ar",
    "chinese": "ch",
    "japanese": "japan",
    "korean": "korean",
}

# scripts each recogniser is trusted to output; letters in any other script are misreads (e.g. Han glyphs
# from the Chinese+English PP-OCRv6 model on a Latin-only frame) and the line is dropped. Digit/symbol-only
# lines have no script and are kept. Raw PaddleOCR lang codes are not filtered.
EXPECTED_SCRIPTS = {
    "latin": {"Latin"},
    "arabic": {"Arabic", "Latin"},
    "chinese": {"Han", "Latin"},
    "japanese": {"Han", "Japanese kana", "Latin"},
    "korean": {"Hangul", "Han", "Latin"},
}

DEFAULTS = {
    "languages": ("latin",),
    "fps": 1.0,
    "min_confidence": 0.6,
    "max_frames": 0,
    "device": "cpu",
    "cpu_threads": 0,
    "ocr_version": None,
    "timeout": 600,
}

_PIPELINES = {}
_PIPELINES_LOCK = threading.Lock()
_SHOWINFO_PTS = re.compile(r"\bn:\s*(\d+)\b.*?\bpts_time:\s*(-?[\d.]+)")


def analyze(path: str, **opts) -> dict:
    """OCR an image or a video file. Options (all optional): languages, fps, min_confidence, max_frames,
    device, cpu_threads, ocr_version, timeout, stills=[(timestamp, image_path), ...] to OCR given stills
    instead of sampling the video. Raises EnricherInputError / EnricherServiceError."""
    unknown = set(opts) - set(DEFAULTS) - {"stills"}
    if unknown:
        raise TypeError(f"analyze() got unknown options: {sorted(unknown)}")
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None}}
    languages = _languages(o["languages"])
    fps = float(o["fps"])
    if fps <= 0:
        raise ValueError("fps must be > 0")

    stills = opts.get("stills")
    if stills is not None:
        media = {"media_type": "stills", "width": None, "height": None, "duration": None}
        frames = [(float(ts), _load_image(p)) for ts, p in _iter_stills(stills)]
    else:
        media, frames = _read_media(path, fps, int(o["max_frames"] or 0), int(o["timeout"]))

    pipelines = {lang: _pipeline(lang, o) for lang in languages}

    per_frame = []
    for ts, image in frames:
        lines = []
        for lang, (ocr, _models) in pipelines.items():
            lines += _ocr_frame(ocr, lang, image, float(o["min_confidence"]))
        per_frame.append({"timestamp": round(ts, 3), "lines": _sorted_lines(_merge_languages(lines))})

    timeline = _timeline(per_frame)
    scripts = Counter(entry["script"] for entry in timeline if entry["script"])
    models = {lang: m for lang, (_ocr, m) in pipelines.items()}
    return {
        "tool": TOOL,
        "tool_version": _version("paddleocr"),
        "paddlepaddle_version": _version("paddle"),
        "model": "; ".join(f"{m['det']}+{m['rec']}" for m in models.values()),
        "models": models,
        **media,
        "sample_fps": fps if media["media_type"] == "video" else None,
        "frame_count": len(per_frame),
        "languages_requested": list(languages),
        "languages_detected": sorted({entry["language"] for entry in timeline}),
        "scripts": dict(scripts),
        "script": scripts.most_common(1)[0][0] if scripts else None,
        "has_text": bool(timeline),
        "empty_reason": None if timeline else "no_text_detected",
        "full_text": "\n".join(entry["text"] for entry in timeline),
        "timeline": timeline,
        "frames": per_frame,
    }


class OcrEnricher(Enricher):
    """Runs `analyze` on every image and video in the item and stores the result on the Media under `ocr`."""

    def enrich(self, to_enrich: Metadata) -> None:
        opts = {k: getattr(self, k, None) for k in DEFAULTS}
        for media in to_enrich.media:
            if not (media.is_video() or media.is_image()):
                continue
            try:
                result = analyze(media.filename, **opts)
            except (EnricherInputError, EnricherServiceError) as e:
                logger.warning(f"OCR failed for {media.filename}: {type(e).__name__} {e.reason}: {e}")
                media.set("ocr_error", {"type": type(e).__name__, "reason": e.reason, "message": str(e)})
                continue
            media.set("ocr", result)
            logger.debug(f"OCR {media.filename}: {len(result['timeline'])} distinct lines")


def _languages(value) -> tuple:
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    langs = tuple(dict.fromkeys(str(v).strip().lower() for v in value if str(v).strip()))
    if not langs:
        raise ValueError("at least one OCR language is required")
    return langs


def _version(module: str):
    try:
        return __import__(module).__version__
    except Exception:
        return None


# ---------- media -> frames ----------


def _read_media(path: str, fps: float, max_frames: int, timeout: int):
    if not path or not os.path.isfile(path):
        raise EnricherInputError(f"no such file: {path}", reason="not_found")
    if os.path.getsize(path) == 0:
        raise EnricherInputError(f"empty file: {path}", reason="empty_file")

    image = _try_still_image(path)
    if image is not None:
        h, w = image.shape[:2]
        return {"media_type": "image", "width": w, "height": h, "duration": None}, [(0.0, image)]

    probe = _ffprobe(path, timeout)
    streams = probe.get("streams") or []
    video = [s for s in streams if s.get("codec_type") == "video" and not _is_cover_art(s)]
    if not video:
        kinds = sorted({s.get("codec_type") for s in streams if s.get("codec_type")})
        if not streams:
            raise EnricherInputError(f"not a readable image or video: {path}", reason="unreadable_media")
        raise EnricherInputError(f"no video stream (streams: {kinds})", reason="no_video_stream")
    if video[0].get("codec_name") in {"ansi", "tty"}:
        raise EnricherInputError("text file, not video", reason="unreadable_media")

    duration = _float((probe.get("format") or {}).get("duration")) or _float(video[0].get("duration"))
    frames = _sample_frames(path, fps, max_frames, timeout)
    if not frames:
        raise EnricherInputError("ffmpeg decoded no frames", reason="no_decodable_frames")
    h, w = frames[0][1].shape[:2]
    return {"media_type": "video", "width": w, "height": h, "duration": duration}, frames


def _try_still_image(path: str):
    """BGR array if PIL reads the file as a single still image, else None (video, animation, not an image)."""
    try:
        with Image.open(path) as img:
            if getattr(img, "n_frames", 1) > 1:
                return None
            img.load()
            return _to_bgr(img)
    except (UnidentifiedImageError, Image.DecompressionBombError):
        return None
    except (OSError, SyntaxError, ValueError) as e:
        raise EnricherInputError(f"corrupt image {path}: {e}", reason="corrupt_image") from e


def _load_image(path: str):
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise EnricherInputError(f"missing or empty still: {path}", reason="not_found")
    try:
        with Image.open(path) as img:
            img.load()
            return _to_bgr(img)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise EnricherInputError(f"unreadable still {path}: {e}", reason="corrupt_image") from e


def _to_bgr(img: Image.Image):
    rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _iter_stills(stills):
    for s in stills:
        if isinstance(s, dict):
            yield s["timestamp"], s["path"]
        else:
            ts, p = s
            yield ts, p


def _is_cover_art(stream: dict) -> bool:
    return bool((stream.get("disposition") or {}).get("attached_pic"))


def _float(v):
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


def _run(cmd: list, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise EnricherServiceError(f"{cmd[0]} is not installed", reason="ffmpeg_missing") from e
    except subprocess.TimeoutExpired as e:
        raise EnricherServiceError(f"{cmd[0]} timed out after {timeout}s", reason="ffmpeg_timeout") from e


def _ffprobe(path: str, timeout: int) -> dict:
    r = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], timeout)
    if r.returncode != 0:
        err = r.stderr.decode(errors="replace").strip().splitlines()
        raise EnricherInputError(
            f"ffprobe can't read {path}: {err[-1] if err else r.returncode}", reason="unreadable_media"
        )
    try:
        return json.loads(r.stdout or b"{}")
    except ValueError as e:
        raise EnricherServiceError("ffprobe returned malformed JSON", reason="ffprobe_bad_output") from e


def _sample_frames(path: str, fps: float, max_frames: int, timeout: int) -> list:
    """[(timestamp_s, bgr array)] sampled at `fps`; timestamps are source pts from showinfo (correct for VFR)."""
    with tempfile.TemporaryDirectory(prefix="ocr_frames_") as tmp:
        # select (not the fps filter, which relabels frames to its output grid): every kept frame is a real
        # decoded frame whose own pts is reported, at least 1/fps after the previous one
        step = 1.0 / fps
        vf = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{step - 1e-3:.6f})',showinfo"
        frames = _extract(path, vf, max_frames, tmp, timeout, step)
        if not frames:
            # clips shorter than one sampling interval (or with a late first pts) still get their first frame
            frames = _extract(path, "showinfo", 1, tmp, timeout, 0.0)
        return [(ts, _load_image(p)) for ts, p in frames]


def _extract(path: str, vf: str, max_frames: int, out_dir: str, timeout: int, step: float) -> list:
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info", "-i", path, "-map", "0:v:0"]
    cmd += ["-an", "-sn", "-dn", "-vf", vf, "-vsync", "vfr"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "image2", os.path.join(out_dir, "%06d.png")]
    r = _run(cmd, timeout)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    stderr = r.stderr.decode(errors="replace")
    if r.returncode != 0 and not files:
        tail = stderr.strip().splitlines()[-1:] or [str(r.returncode)]
        raise EnricherInputError(f"ffmpeg could not decode video frames: {tail[0]}", reason="decode_failed")
    pts = {int(n): float(t) for n, t in _SHOWINFO_PTS.findall(stderr)}
    start = min(pts.values()) if pts else 0.0
    return [(max(0.0, pts.get(i, start + i * step) - start), os.path.join(out_dir, f)) for i, f in enumerate(files)]


# ---------- OCR ----------


def _pipeline(lang: str, o: dict):
    """(PaddleOCR instance, {"det", "rec"} model names), cached per process; a load failure is a service error."""
    key = (lang, o["ocr_version"], o["device"], int(o["cpu_threads"] or 0))
    with _PIPELINES_LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise EnricherServiceError(
                "paddleocr is not installed (see ocr_enricher/requirements.txt)", reason="tool_missing"
            ) from e
        kwargs = {
            "lang": LANGUAGE_GROUPS.get(lang, lang),
            "ocr_version": o["ocr_version"],
            "device": o["device"],
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if o["cpu_threads"]:
            kwargs["cpu_threads"] = int(o["cpu_threads"])
        try:
            ocr = PaddleOCR(**kwargs)
        except ValueError as e:
            if "No models are available" in str(e):
                raise ValueError(f"PaddleOCR has no model for language {lang!r}") from e
            raise EnricherServiceError(f"PaddleOCR failed to load for {lang!r}: {e}", reason="model_load_failed") from e
        except MemoryError as e:
            raise EnricherServiceError("out of memory loading PaddleOCR", reason="out_of_memory") from e
        except Exception as e:
            raise EnricherServiceError(
                f"PaddleOCR models for {lang!r} unavailable: {_root_cause(e)}", reason="model_unavailable"
            ) from e
        sub = ocr._merged_paddlex_config.get("SubModules", {})
        models = {
            "det": sub.get("TextDetection", {}).get("model_name"),
            "rec": sub.get("TextRecognition", {}).get("model_name"),
        }
        _PIPELINES[key] = (ocr, models)
        return _PIPELINES[key]


def _root_cause(e: BaseException) -> str:
    msgs = []
    while e is not None and len(msgs) < 4:
        msgs.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    return " <- ".join(msgs)


def _ocr_frame(ocr, lang: str, image, min_confidence: float) -> list:
    try:
        results = ocr.predict(image)
    except MemoryError as e:
        raise EnricherServiceError("out of memory during OCR", reason="out_of_memory") from e
    except Exception as e:
        raise EnricherServiceError(f"PaddleOCR predict failed: {e}", reason="ocr_failed") from e
    lines = []
    for res in results or []:
        texts, scores, polys = res["rec_texts"], res["rec_scores"], res["rec_polys"]
        for text, score, poly in zip(texts, scores, polys):
            text = unicodedata.normalize("NFC", str(text)).strip()
            if not text or float(score) < min_confidence:
                continue
            script = _script(text)
            if script and lang in EXPECTED_SCRIPTS and script not in EXPECTED_SCRIPTS[lang]:
                continue
            pts = np.asarray(poly).reshape(-1, 2)
            lines.append(
                {
                    "text": text,
                    "confidence": round(float(score), 4),
                    "box": [int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())],
                    "polygon": [[int(x), int(y)] for x, y in pts],
                    "language": lang,
                    "script": script,
                }
            )
    return lines


def _script(text: str):
    counts = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        for script in ("LATIN", "ARABIC", "HANGUL", "HIRAGANA", "KATAKANA", "CJK", "CYRILLIC", "GREEK", "HEBREW"):
            if name.startswith(script) or f" {script} " in f" {name} ":
                counts[
                    {"HIRAGANA": "Japanese kana", "KATAKANA": "Japanese kana", "CJK": "Han"}.get(script, script.title())
                ] += 1
                break
        else:
            counts["Other"] += 1
    return counts.most_common(1)[0][0] if counts else None


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _merge_languages(lines: list) -> list:
    """When several recognisers read the same region, keep the most confident reading."""
    kept = []
    for line in sorted(lines, key=lambda x: -x["confidence"]):
        if all(_iou(line["box"], k["box"]) < 0.5 for k in kept):
            kept.append(line)
    return kept


def _sorted_lines(lines: list) -> list:
    return sorted(lines, key=lambda x: (x["box"][1], x["box"][0]))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def _timeline(frames: list) -> list:
    """Distinct lines across frames. Exact (normalised) repeats merge anywhere; a near-identical reading of the
    same region in the next sampled frame (OCR jitter, e.g. 'GUCCl' vs 'GUCCI') merges into the running entry."""
    entries, by_key = [], {}
    prev_ts = None
    for frame in frames:
        ts = frame["timestamp"]
        for line in frame["lines"]:
            key = _norm(line["text"])
            entry = by_key.get(key)
            if entry is None:
                entry = next(
                    (
                        e
                        for e in entries
                        if e["last_seen"] == prev_ts
                        and _iou(e["box"], line["box"]) >= 0.5
                        and difflib.SequenceMatcher(None, _norm(e["text"]), key).ratio() >= 0.85
                    ),
                    None,
                )
            if entry is None:
                entry = {
                    "text": line["text"],
                    "first_seen": ts,
                    "last_seen": ts,
                    "max_confidence": line["confidence"],
                    "box": line["box"],
                    "frames": 0,
                    "language": line["language"],
                    "script": line["script"],
                }
                entries.append(entry)
            if entry["last_seen"] != ts or entry["frames"] == 0:
                entry["frames"] += 1
            entry["last_seen"] = ts
            if line["confidence"] > entry["max_confidence"]:
                entry.update(
                    text=line["text"],
                    max_confidence=line["confidence"],
                    box=line["box"],
                    language=line["language"],
                    script=line["script"],
                )
            by_key[key] = entry
        prev_ts = ts
    entries.sort(key=lambda e: (e["first_seen"], e["box"][1], e["box"][0]))
    return entries


__all__ = ["analyze", "OcrEnricher", "EnricherInputError", "EnricherServiceError"]
