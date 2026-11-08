"""Video eligibility check and H.264/AAC MP4 rendition ladder.

For every video in a `Metadata`:
  video_check  {"ok", "problems", "probe"}: ffprobe summary plus a full decode pass
  renditions   [Media], one MP4 per ladder rung up to the source size, each probed back before it is kept
  transcode_error  set instead of renditions when encoding or the read-back check fails
"""

import os
import subprocess

import ffmpeg

from auto_archiver.core import Enricher, Media, Metadata
from auto_archiver.utils.custom_logger import logger

REMUX_PROFILES = {"Baseline", "Constrained Baseline", "Main", "High"}
# peak video bitrate per rung, so a noisy source can't make a low rung huge
MAXRATE = {2160: "16M", 1440: "10M", 1080: "6M", 720: "3500k", 480: "1800k", 360: "1000k", 240: "600k"}


def _rotation(stream: dict) -> int:
    rot = (stream.get("tags") or {}).get("rotate")
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    try:
        return int(float(rot or 0)) % 360
    except (TypeError, ValueError):
        return 0


def _fps(rate: str | None) -> float | None:
    try:
        num, den = (rate or "").split("/")
        return round(int(num) / int(den), 3) if int(den) else None
    except ValueError:
        return None


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(probe: dict) -> dict:
    """The fields we act on, from an ffprobe -show_format -show_streams result."""
    streams = probe.get("streams") or []
    v = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = probe.get("format") or {}
    out = {
        "container": fmt.get("format_name"),
        "duration": _num(fmt.get("duration")) or _num((v or {}).get("duration")),
        "bit_rate": int(_num(fmt.get("bit_rate")) or 0) or None,
        "has_video": v is not None,
        "has_audio": a is not None,
    }
    if v:
        w, h, rot = v.get("width") or 0, v.get("height") or 0, _rotation(v)
        dw, dh = (h, w) if rot in (90, 270) else (w, h)
        out.update(
            video_codec=v.get("codec_name"),
            profile=v.get("profile"),
            pix_fmt=v.get("pix_fmt"),
            width=w,
            height=h,
            rotation=rot,
            display_width=dw,
            display_height=dh,
            short_side=min(dw, dh),
            fps=_fps(v.get("avg_frame_rate")) or _fps(v.get("r_frame_rate")),
        )
    if a:
        out.update(audio_codec=a.get("codec_name"), channels=a.get("channels"), sample_rate=_num(a.get("sample_rate")))
    return out


class VideoTranscodeEnricher(Enricher):
    def enrich(self, to_enrich: Metadata) -> None:
        for m in to_enrich.media:
            if not m.is_video():
                continue
            check = self.check(m.filename, m.get("source_duration"))
            m.set("video_check", check)
            if not check["ok"]:
                logger.info(f"video not eligible {m.filename}: {check['problems']}")
                continue
            folder = os.path.join(self.tmp_dir, "renditions", os.path.basename(m.filename))
            try:
                m.set("renditions", self.transcode(m.filename, check["probe"], folder))
            except Exception as e:  # noqa: BLE001 - reported on the media, the original stays as it is
                logger.warning(f"transcode failed for {m.filename}: {e}")
                m.set("transcode_error", str(e)[:2000])

    # ---------------- check ----------------

    def check(self, path: str, source_duration: float | None = None) -> dict:
        problems: list[str] = []
        try:
            info = summarize(ffmpeg.probe(path))
        except ffmpeg.Error as e:
            err = (e.stderr or b"").decode(errors="replace").strip().splitlines()
            return {"ok": False, "problems": [f"unreadable: {err[-1] if err else e}"], "probe": None}
        if not info["has_video"]:
            problems.append("no_video_stream")
        dur = info["duration"]
        if not dur or dur <= 0:
            problems.append("duration_unknown")
        elif dur < self.min_duration:
            problems.append(f"too_short: {dur:.2f}s < {self.min_duration}s")
        if info.get("short_side") is not None and info["short_side"] < self.min_height:
            problems.append(f"too_small: {info['display_width']}x{info['display_height']}")
        expected = _num(source_duration)
        if dur and expected and abs(dur - expected) > max(1.0, 0.05 * expected):
            problems.append(f"duration_mismatch: file {dur:.2f}s, source says {expected:.2f}s")
        if info["has_video"] and self.decode_check:
            errors = self._decode_errors(path)
            if errors:
                problems.append(f"decode_errors: {len(errors)}; first: {errors[0][:300]}")
        return {"ok": not problems, "problems": problems, "probe": info}

    def _decode_errors(self, path: str) -> list[str]:
        r = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-v",
                "error",
                "-i",
                path,
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        lines = [ln for ln in r.stderr.decode(errors="replace").splitlines() if ln.strip()]
        if r.returncode and not lines:
            lines = [f"ffmpeg exited {r.returncode}"]
        return lines

    # ---------------- ladder ----------------

    def ladder(self, info: dict) -> list[int]:
        short = info["short_side"] - info["short_side"] % 2
        rungs = sorted({int(h) for h in self.heights if int(h) <= short}, reverse=True)
        if short < max(int(h) for h in self.heights) and short not in rungs:
            rungs.insert(0, short)  # keep the source's own size when it sits between rungs
        return rungs

    def _can_remux(self, info: dict, rung: int) -> bool:
        return (
            rung == info["short_side"]
            and info.get("video_codec") == "h264"
            and info.get("pix_fmt") == "yuv420p"
            and info.get("profile") in REMUX_PROFILES
            and info.get("rotation") == 0
            and any(c in (info.get("container") or "") for c in ("mp4", "mov"))
            and (not info["has_audio"] or info.get("audio_codec") == "aac")
        )

    def _args(self, src: str, out: str, info: dict, rung: int, remux: bool) -> list[str]:
        args = [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-v",
            "error",
            "-y",
            "-i",
            src,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-map_metadata",
            "-1",
            "-movflags",
            "+faststart",
        ]
        if remux:
            return args + ["-c", "copy", out]
        landscape = info["display_width"] >= info["display_height"]
        scale = f"scale=-2:{rung}" if landscape else f"scale={rung}:-2"
        maxrate = MAXRATE.get(rung) or MAXRATE[min(MAXRATE, key=lambda k: abs(k - rung))]
        bufsize = f"{int(maxrate[:-1]) * 2}{maxrate[-1]}"
        return args + [
            "-vf",
            f"{scale},format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-profile:v",
            "high",
            "-maxrate",
            maxrate,
            "-bufsize",
            bufsize,
            "-c:a",
            "aac",
            "-b:a",
            self.audio_bitrate,
            "-ac",
            "2",
            out,
        ]

    def transcode(self, src: str, info: dict, folder: str) -> list[Media]:
        os.makedirs(folder, exist_ok=True)
        out: list[Media] = []
        for rung in self.ladder(info):
            path = os.path.join(folder, f"{rung}p.mp4")
            remux = self._can_remux(info, rung)
            r = subprocess.run(
                self._args(src, path, info, rung, remux), capture_output=True, timeout=self.timeout, check=False
            )
            if r.returncode or not os.path.exists(path):
                raise RuntimeError(
                    f"ffmpeg {rung}p failed ({r.returncode}): {r.stderr.decode(errors='replace')[-500:]}"
                )
            got = self._verify(path, info, rung)
            size = os.path.getsize(path)
            if out and size >= out[-1].get("bytes"):
                logger.debug(f"dropping {rung}p: {size} bytes is not smaller than {out[-1].get('label')}")
                os.remove(path)
                continue
            media = Media(filename=path).set("id", f"rendition_{rung}p").set("label", f"{rung}p")
            media.mimetype = "video/mp4"
            for k in ("display_width", "display_height", "duration", "bit_rate", "video_codec", "has_audio"):
                media.set(k, got.get(k))
            out.append(media.set("height", rung).set("bytes", size).set("method", "remux" if remux else "encode"))
        if not out:
            raise RuntimeError("no rendition produced")
        return out

    def _verify(self, path: str, src: dict, rung: int) -> dict:
        got = summarize(ffmpeg.probe(path))
        bad = []
        if got.get("video_codec") != "h264":
            bad.append(f"codec {got.get('video_codec')}")
        if got.get("pix_fmt") != "yuv420p":
            bad.append(f"pix_fmt {got.get('pix_fmt')}")
        if got.get("short_side") != rung:
            bad.append(f"short side {got.get('short_side')} != {rung}")
        if (got["display_width"] >= got["display_height"]) != (src["display_width"] >= src["display_height"]):
            bad.append("orientation changed")
        if got["has_audio"] != src["has_audio"]:
            bad.append(f"audio {'missing' if src['has_audio'] else 'added'}")
        if not got["duration"] or abs(got["duration"] - src["duration"]) > max(0.5, 0.01 * src["duration"]):
            bad.append(f"duration {got['duration']} != {src['duration']}")
        if bad:
            raise RuntimeError(f"{rung}p failed its read-back check: {', '.join(bad)}")
        return got
