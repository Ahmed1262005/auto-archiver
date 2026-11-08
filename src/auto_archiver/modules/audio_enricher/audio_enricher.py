"""Audio enricher: EBU R128 loudness, silence and black segments (ffmpeg) and Chromaprint fingerprints (fpcalc).

`analyze(path)` runs ffprobe, one ffmpeg pass (ebur128 + silencedetect on the audio, blackdetect on the video)
and `fpcalc -raw` as subprocesses and returns a JSON-serialisable dict. `fingerprint_similarity(a, b)` compares
two raw fingerprints at every relative offset by bit errors (XOR popcount), the comparison AcoustID describes
for raw Chromaprint fingerprints; Chromaprint ships no maintained Python matcher, so it is done here with numpy.
"""

import json
import re
import subprocess
from functools import lru_cache

import numpy as np

from auto_archiver.core import Enricher, Metadata
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOL = "ffmpeg+chromaprint"
# Chromaprint algorithm 2 (fpcalc default): 11025 Hz, 4096-sample frames, one item every 4096 // 3 samples.
FINGERPRINT_ALGORITHM = 2
ITEM_DURATION_S = (4096 // 3) / 11025
# a stream that is silent for (almost) all of its length counts as no audio
SILENT_FRACTION_NO_AUDIO = 0.99
# ebur128 reports -70 LUFS (the absolute gate) when no block is loud enough to measure
EBUR128_FLOOR_LUFS = -70.0

INPUT_ERROR_MARKERS = (
    "invalid data found when processing input",
    "moov atom not found",
    "could not find codec parameters",
    "no such file or directory",
    "end of file",
    "invalid argument",
    "error while decoding",
    "could not find audio stream",
    "could not open input",
    "error decoding audio",
    "empty fingerprint",
)
SERVICE_ERROR_MARKERS = ("cannot allocate memory", "out of memory", "killed")
STILL_IMAGE_FORMAT = re.compile(r"(^|,)(image2|[a-z0-9]+_pipe|gif|webp)(,|$)")


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a tool; tool problems (missing binary, timeout, killed) become EnricherServiceError."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, errors="replace", timeout=timeout, stdin=subprocess.DEVNULL
        )
    except FileNotFoundError as e:
        raise EnricherServiceError(f"{cmd[0]} is not installed or not on PATH", reason="tool_missing") from e
    except subprocess.TimeoutExpired as e:
        raise EnricherServiceError(f"{cmd[0]} timed out after {timeout}s", reason="timeout") from e
    except OSError as e:
        raise EnricherServiceError(f"{cmd[0]} could not start: {e}", reason="tool_failed") from e
    if proc.returncode < 0 or proc.returncode in (137, 139):
        raise EnricherServiceError(f"{cmd[0]} was killed (exit {proc.returncode})", reason="crashed")
    return proc


def _fail(tool: str, proc: subprocess.CompletedProcess, path: str):
    """Raise the right typed error for a tool that exited non-zero."""
    err = (proc.stderr or "").strip()
    tail = err.splitlines()[-1] if err else f"exit {proc.returncode}"
    low = err.lower()
    if any(m in low for m in SERVICE_ERROR_MARKERS):
        raise EnricherServiceError(f"{tool} ran out of resources on {path}: {tail}", reason="out_of_memory")
    if any(m in low for m in INPUT_ERROR_MARKERS):
        raise EnricherInputError(f"{tool} cannot read {path}: {tail}", reason="unreadable")
    raise EnricherServiceError(f"{tool} failed on {path}: {tail}", reason="tool_failed")


@lru_cache(maxsize=8)
def _version(binary: str, timeout: float = 30) -> str:
    proc = _run([binary, "-version"], timeout)
    out = (proc.stdout or proc.stderr or "").strip()
    m = re.search(r"version\s+(\S+)", out)
    if proc.returncode != 0 or not m:
        raise EnricherServiceError(f"{binary} -version gave no version: {out[:200]!r}", reason="tool_failed")
    return m.group(1)


def tool_versions(ffmpeg_bin: str = "ffmpeg", fpcalc_bin: str = "fpcalc") -> dict:
    return {"ffmpeg": _version(ffmpeg_bin), "fpcalc": _version(fpcalc_bin)}


def _float(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _probe(path: str, ffprobe_bin: str, timeout: float) -> dict:
    proc = _run([ffprobe_bin, "-v", "error", "-show_format", "-show_streams", "-of", "json", path], timeout)
    if proc.returncode != 0:
        _fail("ffprobe", proc, path)
    try:
        probe = json.loads(proc.stdout or "")
    except json.JSONDecodeError as e:
        raise EnricherServiceError(f"ffprobe printed malformed JSON for {path}", reason="malformed_output") from e
    if not isinstance(probe, dict) or not isinstance(probe.get("format"), dict):
        raise EnricherServiceError(f"ffprobe output has no format section for {path}", reason="malformed_output")
    return probe


def _segments(text: str, kind: str, total: float | None) -> list[dict]:
    """Pair `<kind>_start` / `<kind>_end` log lines into segments; an open segment runs to the end."""
    segments, start = [], None
    for m in re.finditer(rf"{kind}_(start|end):\s*(-?[\d.]+(?:e-?\d+)?)", text):
        t = max(0.0, float(m.group(2)))
        if m.group(1) == "start":
            start = t
        elif start is not None:
            segments.append((start, t))
            start = None
    if start is not None and total is not None and total > start:
        segments.append((start, total))
    return [{"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3)} for s, e in segments if e > s]


def _fraction(segments: list[dict], total: float | None):
    if not total:
        return None
    return round(min(1.0, sum(s["duration"] for s in segments) / total), 4)


def _loudness(text: str) -> dict:
    summary = text.rsplit("Summary:", 1)
    if len(summary) != 2:
        raise EnricherServiceError("ffmpeg ebur128 printed no summary", reason="malformed_output")
    s = summary[1]

    def grab(pattern):
        m = re.search(pattern, s)
        return None if not m else _float(m.group(1).replace("inf", "nan"))

    integrated = grab(r"I:\s*(-?[\d.]+|-?inf)\s*LUFS")
    lra = grab(r"LRA:\s*(-?[\d.]+|-?inf)\s*LU\b")
    peak = grab(r"Peak:\s*(-?[\d.]+|-?inf)\s*dB")
    measurable = integrated is not None and integrated > EBUR128_FLOOR_LUFS
    return {
        "standard": "EBU R128",
        "integrated_lufs": integrated if measurable else None,
        "lra_lu": lra if measurable else None,
        "true_peak_dbtp": peak,
    }


def _fingerprint(path: str, fpcalc_bin: str, max_seconds: int, timeout: float) -> dict:
    cmd = [
        fpcalc_bin,
        "-raw",
        "-json",
        "-algorithm",
        str(FINGERPRINT_ALGORITHM),
        "-length",
        str(int(max_seconds)),
        path,
    ]
    proc = _run(cmd, timeout)
    empty = {"algorithm": FINGERPRINT_ALGORITHM, "item_duration_s": ITEM_DURATION_S, "raw": [], "items": 0}
    if proc.returncode != 0:
        if "empty fingerprint" in (proc.stderr or "").lower():
            # valid but too little audio for Chromaprint to produce an item
            return {**empty, "duration_s": None, "status": "too_short"}
        _fail("fpcalc", proc, path)
    try:
        out = json.loads(proc.stdout or "")
        raw = [int(x) for x in out["fingerprint"]]
        duration = _float(out.get("duration"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        raise EnricherServiceError(f"fpcalc printed malformed output for {path}", reason="malformed_output") from e
    if any(x < 0 or x > 0xFFFFFFFF for x in raw):
        raise EnricherServiceError("fpcalc fingerprint items are not unsigned 32-bit", reason="malformed_output")
    status = "ok" if raw else "too_short"
    return {**empty, "raw": raw, "items": len(raw), "duration_s": duration, "status": status}


def analyze(
    path: str,
    *,
    silence_noise_db: float = -50.0,
    silence_min_duration: float = 0.5,
    black_min_duration: float = 0.1,
    black_pixel_threshold: float = 0.10,
    black_picture_threshold: float = 0.98,
    fingerprint_max_seconds: int = 0,
    timeout: float = 600,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    fpcalc_bin: str = "fpcalc",
) -> dict:
    """Loudness, silence, black frames and Chromaprint fingerprint of one media file.

    Raises EnricherInputError when the file can't be analysed (missing, empty, corrupt, no audio or video stream)
    and EnricherServiceError when a tool fails (missing binary, timeout, crash, unparseable output).
    """
    try:
        with open(path, "rb") as f:
            if not f.read(1):
                raise EnricherInputError(f"{path} is empty (0 bytes)", reason="empty_file")
    except FileNotFoundError as e:
        raise EnricherInputError(f"{path} does not exist", reason="not_found") from e
    except IsADirectoryError as e:
        raise EnricherInputError(f"{path} is a directory", reason="not_a_file") from e

    versions = tool_versions(ffmpeg_bin, fpcalc_bin)
    probe = _probe(path, ffprobe_bin, timeout)
    fmt, streams = probe["format"], probe.get("streams") or []
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if audio is None and video is None:
        raise EnricherInputError(f"{path} has no audio or video stream", reason="no_stream")
    still_image = video is not None and audio is None and bool(STILL_IMAGE_FORMAT.search(fmt.get("format_name", "")))
    duration = _float(fmt.get("duration"))
    audio_duration = _float((audio or {}).get("duration")) or duration
    video_duration = _float((video or {}).get("duration")) or duration

    params = {
        "silence_noise_db": silence_noise_db,
        "silence_min_duration": silence_min_duration,
        "black_min_duration": black_min_duration,
        "black_pixel_threshold": black_pixel_threshold,
        "black_picture_threshold": black_picture_threshold,
        "fingerprint_max_seconds": fingerprint_max_seconds,
    }
    result = {
        "tool": TOOL,
        "tool_version": versions,
        "model": None,
        "params": params,
        "duration_s": duration,
        "has_audio_stream": audio is not None,
        "has_video_stream": video is not None,
        "is_still_image": still_image,
        "has_audio": False,
        "empty_reason": None,
        "audio_stream": None,
        "loudness": None,
        "silence": None,
        "black": None,
        "fingerprint": None,
    }
    if audio is not None:
        result["audio_stream"] = {
            "codec": audio.get("codec_name"),
            "sample_rate": int(audio["sample_rate"]) if str(audio.get("sample_rate", "")).isdigit() else None,
            "channels": audio.get("channels"),
        }

    graph, maps = [], []
    if audio is not None:
        graph.append(
            "[0:a:0]ebur128=peak=true:framelog=verbose,"
            f"silencedetect=noise={silence_noise_db}dB:duration={silence_min_duration}[aout]"
        )
        maps += ["-map", "[aout]"]
    if video is not None and not still_image:
        graph.append(
            f"[0:{video['index']}]blackdetect=d={black_min_duration}:pix_th={black_pixel_threshold}"
            f":pic_th={black_picture_threshold}[vout]"
        )
        maps += ["-map", "[vout]"]

    if graph:
        cmd = [ffmpeg_bin, "-hide_banner", "-nostdin", "-nostats", "-v", "info", "-i", path]
        cmd += ["-filter_complex", ";".join(graph), *maps, "-f", "null", "-"]
        proc = _run(cmd, timeout)
        if proc.returncode != 0:
            _fail("ffmpeg", proc, path)
        log = proc.stderr or ""
        if audio is not None:
            result["loudness"] = _loudness(log)
            segs = _segments(log, "silence", audio_duration)
            result["silence"] = {
                "noise_db": silence_noise_db,
                "min_duration_s": silence_min_duration,
                "segments": segs,
                "silent_fraction": _fraction(segs, audio_duration),
            }
        if video is not None and not still_image:
            segs = _segments(log, "black", video_duration)
            result["black"] = {
                "pixel_threshold": black_pixel_threshold,
                "picture_threshold": black_picture_threshold,
                "min_duration_s": black_min_duration,
                "segments": segs,
                "black_fraction": _fraction(segs, video_duration),
            }

    empty_fp = {
        "algorithm": FINGERPRINT_ALGORITHM,
        "item_duration_s": ITEM_DURATION_S,
        "raw": [],
        "items": 0,
        "duration_s": None,
    }
    if audio is None:
        result["empty_reason"] = "still_image" if still_image else "no_audio_stream"
        result["fingerprint"] = {**empty_fp, "status": "no_audio_stream"}
        return result
    silent_fraction = result["silence"]["silent_fraction"]
    if result["loudness"]["integrated_lufs"] is None or (silent_fraction or 0) >= SILENT_FRACTION_NO_AUDIO:
        result["empty_reason"] = "silent_audio"
        result["fingerprint"] = {**empty_fp, "status": "silent_audio"}
        return result
    result["has_audio"] = True
    result["fingerprint"] = _fingerprint(path, fpcalc_bin, fingerprint_max_seconds, timeout)
    if result["fingerprint"]["status"] != "ok":
        result["empty_reason"] = "audio_too_short_to_fingerprint"
    return result


def _raw(fp) -> np.ndarray:
    if isinstance(fp, dict):
        fp = fp.get("raw") if "raw" in fp else (fp.get("fingerprint") or {}).get("raw")
    arr = np.asarray(fp if fp is not None else [], dtype=np.int64).ravel()
    return (arr & 0xFFFFFFFF).astype(np.uint32)  # accepts fpcalc -signed values too


def fingerprint_similarity(
    fp_a,
    fp_b,
    *,
    window_seconds: float = 3.0,
    match_threshold: float = 0.8,
    item_error_threshold: int = 10,
    item_duration_s: float = ITEM_DURATION_S,
) -> dict:
    """Best-offset bit-error similarity of two raw Chromaprint fingerprints.

    `fp_a`/`fp_b` are lists of fpcalc -raw integers (or `analyze()` results / their `fingerprint` dicts).
    Every relative offset where the fingerprints overlap by at least one window is tried; per offset the
    bit errors of the aligned items are averaged over a sliding window and the best window wins. The window
    keeps a short cut-down from being diluted by the rest of the longer ad, and short overlaps at the ends
    from scoring on a handful of items.

    Returns `score` (1 - bit-error rate of the best window; ~0.5 for unrelated audio, ~1.0 for the same),
    `offset_seconds` (where fp_b starts inside fp_a; negative if fp_b starts earlier), `overlap_score`
    (over the whole overlap at that offset), `matched_seconds`/`matched_fraction` (aligned items within
    `item_error_threshold` bits after 1 s smoothing, relative to the shorter fingerprint) and `match`.
    """
    a, b = _raw(fp_a), _raw(fp_b)
    result = {
        "score": 0.0,
        "match": False,
        "offset_items": None,
        "offset_seconds": None,
        "overlap_seconds": 0.0,
        "overlap_score": None,
        "matched_seconds": 0.0,
        "matched_fraction": 0.0,
        "items_a": len(a),
        "items_b": len(b),
        "reason": None,
    }
    if not len(a) or not len(b):
        result["reason"] = "empty_fingerprint"
        return result
    window = max(1, int(round(window_seconds / item_duration_s)))
    shorter = min(len(a), len(b))
    window = min(window, shorter)
    smooth = max(1, int(round(1.0 / item_duration_s)))

    best = None
    for k in range(-(len(b) - window), len(a) - window + 1):
        i0, i1 = max(0, k), min(len(a), len(b) + k)
        errors = np.bitwise_count(a[i0:i1] ^ b[i0 - k : i1 - k]).astype(np.int64)
        csum = np.concatenate(([0], np.cumsum(errors)))
        win_err = (csum[window:] - csum[:-window]).min() / window
        key = (-win_err, len(errors))
        if best is None or key > best[0]:
            best = (key, k, errors)
    (neg_err, _), k, errors = best
    n = len(errors)
    w = min(smooth, n)
    smoothed = np.convolve(errors, np.ones(w) / w, mode="same")
    matched_items = int((smoothed <= item_error_threshold).sum())
    score = 1.0 + neg_err / 32.0
    result.update(
        score=round(float(score), 4),
        match=bool(score >= match_threshold),
        offset_items=int(k),
        offset_seconds=round(k * item_duration_s, 3),
        overlap_seconds=round(n * item_duration_s, 3),
        overlap_score=round(float(1.0 - errors.mean() / 32.0), 4),
        matched_seconds=round(matched_items * item_duration_s, 3),
        matched_fraction=round(matched_items / shorter, 4),
    )
    return result


class AudioEnricher(Enricher):
    """Stores `analyze()` output as `audio` on every video and audio Media; failures are logged and stored."""

    def enrich(self, to_enrich: Metadata) -> None:
        opts = {
            "silence_noise_db": self.silence_noise_db,
            "silence_min_duration": self.silence_min_duration,
            "black_min_duration": self.black_min_duration,
            "black_pixel_threshold": self.black_pixel_threshold,
            "black_picture_threshold": self.black_picture_threshold,
            "fingerprint_max_seconds": self.fingerprint_max_seconds,
            "timeout": self.timeout,
        }
        for media in to_enrich.media:
            if not (media.is_video() or media.is_audio()):
                continue
            try:
                media.set("audio", analyze(media.filename, **opts))
            except (EnricherInputError, EnricherServiceError) as e:
                logger.warning(f"audio_enricher: {media.filename}: {e.reason}: {e}")
                media.set("audio_error", {"type": type(e).__name__, "reason": e.reason, "message": str(e)})
