"""Shot / scene detection for video ads with PySceneDetect.

`analyze(path, **opts)` runs PySceneDetect (AdaptiveDetector by default, ContentDetector as an option)
over the decoded video, saves the middle frame of every shot as a JPEG next to the input, and runs
ffmpeg's `blackdetect` filter (as a subprocess) to find black / faded-to-black segments.
The result is a JSON-serialisable dict. `ShotEnricher` calls it for each video `Media` of an item.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import threading
from fractions import Fraction

from auto_archiver.core import Enricher, Media, Metadata
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOL = "PySceneDetect"
DETECTORS = ("adaptive", "content")
DEFAULTS = {
    "detector": "adaptive",
    "adaptive_threshold": 3.0,
    "min_content_val": 15.0,
    "content_threshold": 27.0,
    "min_scene_len": 0.25,
    "downscale": None,
    "keyframes": True,
    "keyframe_dir": None,
    "keyframe_quality": 90,
    "black_detection": True,
    "black_min_duration": 0.1,
    "black_pixel_threshold": 0.10,
    "black_picture_ratio": 0.98,
    "black_shot_ratio": 0.9,
    "timeout": 600,
    "probe_timeout": 60,
}
# ffprobe format names that are single still images, not videos
STILL_IMAGE_FORMATS = ("image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe")
TRUNCATED_RATIO = 0.9
BLACK_RE = re.compile(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)")


def _tool_version() -> str:
    import scenedetect

    return scenedetect.__version__


def _run(cmd: list, timeout: float, what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
    except FileNotFoundError as e:
        raise EnricherServiceError(f"{what}: {cmd[0]} is not installed", reason="tool_missing") from e
    except subprocess.TimeoutExpired as e:
        raise EnricherServiceError(f"{what} timed out after {timeout}s", reason="timeout") from e


def _ratio(value) -> Fraction | None:
    try:
        r = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None
    return r if r > 0 else None


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None


def _rotation(stream: dict) -> int:
    rotation = None
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            rotation = side["rotation"]
    if rotation is None:
        rotation = (stream.get("tags") or {}).get("rotate", 0)
    try:
        return int(float(rotation)) % 360
    except (TypeError, ValueError):
        return 0


def probe(path: str, timeout: float = 60) -> dict:
    """ffprobe the file and return the first real video stream's facts; raise typed errors for bad input."""
    if not os.path.isfile(path):
        raise EnricherInputError(f"{path} does not exist", reason="file_not_found")
    if os.path.getsize(path) == 0:
        raise EnricherInputError(f"{path} is empty (0 bytes)", reason="empty_file")
    cmd = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path]
    p = _run(cmd, timeout, "ffprobe")
    if p.returncode < 0:
        raise EnricherServiceError(f"ffprobe was killed by signal {-p.returncode}", reason="tool_crashed")
    if p.returncode != 0:
        raise EnricherInputError(f"ffprobe can't read {path}: {p.stderr.strip()[-300:]}", reason="unreadable")
    try:
        info = json.loads(p.stdout)
        streams, fmt = info.get("streams") or [], info.get("format") or {}
    except (ValueError, AttributeError) as e:
        raise EnricherServiceError(f"ffprobe returned malformed JSON for {path}", reason="tool_bad_output") from e
    videos = [
        s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
    ]
    if not videos:
        raise EnricherInputError(f"{path} has no video stream", reason="no_video_stream")
    v = videos[0]
    rotation = _rotation(v)
    width, height = v.get("width"), v.get("height")
    if rotation in (90, 270):
        width, height = height, width
    r_fps, avg_fps = _ratio(v.get("r_frame_rate")), _ratio(v.get("avg_frame_rate"))
    format_names = (fmt.get("format_name") or "").split(",")
    return {
        "codec": v.get("codec_name"),
        "width": width,
        "height": height,
        "rotation": rotation,
        "fps": float(r_fps) if r_fps else None,
        "avg_fps": float(avg_fps) if avg_fps else None,
        "vfr": bool(r_fps and avg_fps and abs(float(r_fps) - float(avg_fps)) > 0.01),
        "duration_s": _float(v.get("duration")) or _float(fmt.get("duration")),
        "nb_frames": int(v["nb_frames"]) if str(v.get("nb_frames", "")).isdigit() else None,
        "still_image": any(n in STILL_IMAGE_FORMATS for n in format_names),
        "format": fmt.get("format_name"),
    }


def black_segments(
    path: str, *, min_duration: float, pixel_threshold: float, picture_ratio: float, timeout: float
) -> list[dict]:
    """Black (or faded-to-black) segments from ffmpeg's blackdetect filter."""
    vf = f"blackdetect=d={min_duration}:pix_th={pixel_threshold}:pic_th={picture_ratio}"
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "info", "-i", path, "-an", "-sn", "-vf", vf, "-f", "null", "-"]
    p = _run(cmd, timeout, "ffmpeg blackdetect")
    if p.returncode < 0:
        raise EnricherServiceError(f"ffmpeg was killed by signal {-p.returncode}", reason="tool_crashed")
    segments = []
    for start, end in BLACK_RE.findall(p.stderr):
        start, end = float(start), float(end)
        segments.append({"start_s": round(start, 3), "end_s": round(end, 3), "duration_s": round(end - start, 3)})
    if p.returncode != 0 and not segments:
        logger.warning(f"ffmpeg blackdetect exited {p.returncode} on {path}: {p.stderr.strip()[-300:]}")
    return segments


def _detector(o: dict):
    from scenedetect import AdaptiveDetector, ContentDetector

    if o["detector"] == "adaptive":
        params = {
            "adaptive_threshold": float(o["adaptive_threshold"]),
            "min_content_val": float(o["min_content_val"]),
            "min_scene_len": o["min_scene_len"],
        }
        return AdaptiveDetector(**params), params
    params = {"threshold": float(o["content_threshold"]), "min_scene_len": o["min_scene_len"]}
    return ContentDetector(**params), params


def _detect(path: str, o: dict):
    """Run the SceneManager; returns (scene_list, frames_read, video). Stops via a timer on timeout."""
    import cv2
    from scenedetect import SceneManager, VideoOpenFailure, open_video

    try:
        video = open_video(path, backend="opencv")
    except (VideoOpenFailure, OSError) as e:
        raise EnricherInputError(f"decoder can't open {path}: {e}", reason="unreadable") from e
    detector, params = _detector(o)
    sm = SceneManager()
    sm.add_detector(detector)
    if o["downscale"]:
        sm.auto_downscale = False
        sm.downscale = int(o["downscale"])
    timed_out = threading.Event()

    def stop():
        timed_out.set()
        sm.stop()

    timer = threading.Timer(float(o["timeout"]), stop)
    timer.daemon = True
    timer.start()
    try:
        frames_read = sm.detect_scenes(video)
    except MemoryError as e:
        raise EnricherServiceError(f"out of memory decoding {path}", reason="out_of_memory") from e
    except cv2.error as e:
        raise EnricherServiceError(f"OpenCV failed on {path}: {e}", reason="decoder_error") from e
    finally:
        timer.cancel()
    if timed_out.is_set():
        raise EnricherServiceError(f"shot detection timed out after {o['timeout']}s", reason="timeout")
    # start_in_scene: a video without cuts is one shot, not an empty list
    return sm.get_scene_list(start_in_scene=True), frames_read, video, params


def _ffmpeg_keyframe(path: str, time_s: float, out_path: str, timeout: float) -> bool:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{time_s:.3f}", "-i", path]
    p = _run(cmd + ["-an", "-sn", "-frames:v", "1", "-q:v", "2", out_path], timeout, "ffmpeg keyframe")
    if p.returncode < 0:
        raise EnricherServiceError(f"ffmpeg was killed by signal {-p.returncode}", reason="tool_crashed")
    return p.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _keyframes(scene_list, video, path: str, o: dict, vfr: bool) -> dict:
    """{shot index: (jpeg path, extractor)} for the middle frame of each shot.

    PySceneDetect's save_images writes them in one pass. Its OpenCV seek lands early or fails on VFR
    input, so VFR videos (and any shot it could not write) are extracted with ffmpeg by timestamp."""
    from scenedetect import save_images

    out_dir = o["keyframe_dir"] or os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    paths = {}
    try:
        os.makedirs(out_dir, exist_ok=True)
        if not vfr:
            saved = save_images(
                scene_list,
                video,
                num_images=1,
                frame_margin=0,
                image_extension="jpg",
                encoder_param=int(o["keyframe_quality"]),
                image_name_template=f"{stem}.shot-$SCENE_NUMBER",
                output_dir=out_dir,
                threading=False,
            )
            for i, names in saved.items():
                full = os.path.abspath(os.path.join(out_dir, names[0])) if names else None
                if full and os.path.isfile(full):
                    paths[i] = (full, "pyscenedetect")
        width = max(3, len(str(len(scene_list))))
        for i, (start, end) in enumerate(scene_list):
            if i in paths:
                continue
            full = os.path.abspath(os.path.join(out_dir, f"{stem}.shot-{i + 1:0{width}d}.jpg"))
            mid = start.seconds + (end.seconds - start.seconds) / 2.0
            if _ffmpeg_keyframe(path, mid, full, float(o["probe_timeout"])):
                paths[i] = (full, "ffmpeg")
    except OSError as e:
        raise EnricherServiceError(f"can't write keyframes to {out_dir}: {e}", reason="keyframe_write_failed") from e
    return paths


def _overlap(a0: float, a1: float, segments: list[dict]) -> float:
    return sum(max(0.0, min(a1, s["end_s"]) - max(a0, s["start_s"])) for s in segments)


def analyze(path: str, **opts) -> dict:
    """Detect the shots of one video file. Raises EnricherInputError / EnricherServiceError.

    Times are in seconds from the first frame; frame numbers are 0-based and `end_frame` is exclusive
    (for VFR input the frame numbers are estimates from the average frame rate, see `frames_approximate`).
    """
    unknown = set(opts) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown shot_enricher options: {sorted(unknown)}")
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None}}
    if o["detector"] not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}, got {o['detector']!r}")

    info = probe(path, timeout=float(o["probe_timeout"]))
    result = {
        "tool": TOOL,
        "tool_version": _tool_version(),
        "model": None,
        "detector": o["detector"],
        "detector_params": None,
        "black_detector": "ffmpeg blackdetect" if o["black_detection"] else None,
        "video": {k: info[k] for k in ("codec", "width", "height", "rotation", "fps", "avg_fps", "vfr", "duration_s")},
        "frames_approximate": info["vfr"],
        "shots": [],
        "shot_count": 0,
        "cut_count": 0,
        "cuts_s": [],
        "cuts_per_minute": 0.0,
        "mean_shot_s": None,
        "median_shot_s": None,
        "black_segments": [],
        "decoded_duration_s": None,
        "frames_decoded": 0,
        "truncated": False,
        "single_shot": False,
        "empty_reason": None,
    }
    if info["still_image"]:
        result["empty_reason"] = "still_image"
        return result

    scene_list, frames_read, video, params = _detect(path, o)
    result["detector_params"] = params
    result["frames_decoded"] = frames_read
    if frames_read <= 0:
        raise EnricherInputError(f"no frame of {path} could be decoded", reason="no_decodable_frames")
    if not scene_list:
        raise EnricherInputError(f"no shots in {frames_read} decoded frames of {path}", reason="no_shots")

    decoded = scene_list[-1][1].seconds
    result["decoded_duration_s"] = round(decoded, 3)
    if info["duration_s"] and decoded < TRUNCATED_RATIO * info["duration_s"] and info["duration_s"] - decoded > 0.5:
        result["truncated"] = True

    if o["black_detection"]:
        result["black_segments"] = black_segments(
            path,
            min_duration=o["black_min_duration"],
            pixel_threshold=o["black_pixel_threshold"],
            picture_ratio=o["black_picture_ratio"],
            timeout=float(o["timeout"]),
        )
    keyframes = _keyframes(scene_list, video, path, o, info["vfr"]) if o["keyframes"] else {}

    for i, (start, end) in enumerate(scene_list):
        start_s, end_s = start.seconds, end.seconds
        dur = end_s - start_s
        mid_s = start_s + dur / 2.0
        shot = {
            "index": i,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(dur, 3),
            "start_frame": start.frame_num,
            "end_frame": end.frame_num,
            "duration_frames": end.frame_num - start.frame_num,
            "black": bool(
                dur > 0 and _overlap(start_s, end_s, result["black_segments"]) >= o["black_shot_ratio"] * dur
            ),
            "keyframe": None,
        }
        if i in keyframes:
            shot["keyframe"] = {"path": keyframes[i][0], "time_s": round(mid_s, 3), "extractor": keyframes[i][1]}
        result["shots"].append(shot)

    durations = [s["duration_s"] for s in result["shots"]]
    result["shot_count"] = len(durations)
    result["cut_count"] = len(durations) - 1
    result["cuts_s"] = [s["start_s"] for s in result["shots"][1:]]
    result["cuts_per_minute"] = round(result["cut_count"] / (decoded / 60.0), 3) if decoded > 0 else 0.0
    result["mean_shot_s"] = round(statistics.fmean(durations), 3)
    result["median_shot_s"] = round(statistics.median(durations), 3)
    result["single_shot"] = result["cut_count"] == 0
    if result["single_shot"]:
        result["empty_reason"] = "no_cuts"
    return result


class ShotEnricher(Enricher):
    """Detects shots in every video Media, stores the analysis as `shots` and the per-shot keyframes as
    `shot_keyframes` (image Media, so pdq_hash_enricher can hash them when it runs after this one)."""

    def setup(self) -> None:
        for binary in ("ffmpeg", "ffprobe"):
            if shutil.which(binary) is None:
                logger.warning(f"shot_enricher: {binary} not found on PATH")

    def options(self) -> dict:
        return {k: getattr(self, k, None) for k in DEFAULTS if k != "keyframe_dir"}

    def enrich(self, to_enrich: Metadata) -> None:
        for media in to_enrich.media:
            if not media.is_video():
                continue
            try:
                result = analyze(media.filename, **self.options())
            except EnricherInputError as e:
                logger.warning(f"shot_enricher: can't analyse {media.filename}: {e.reason}: {e}")
                media.set("shots", {"tool": TOOL, "error": e.reason})
                continue
            media.set("shots", result)
            media.set(
                "shot_keyframes",
                [
                    Media(filename=s["keyframe"]["path"])
                    .set("id", f"shot_keyframe_{s['index']}")
                    .set("timestamp", "%.3fs" % s["keyframe"]["time_s"])
                    for s in result["shots"]
                    if s["keyframe"]
                ],
            )
            logger.debug(f"shot_enricher: {result['shot_count']} shots in {media.filename}")
