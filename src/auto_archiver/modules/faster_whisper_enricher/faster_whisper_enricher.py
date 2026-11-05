"""
Transcript and spoken-language enricher built on faster-whisper (MIT, CTranslate2).

`analyze(path, **opts)` decodes the first audio track with ffmpeg, removes non-speech with the Silero VAD that ships
with faster-whisper, transcribes what is left and returns a JSON-serialisable dict. Media without speech (music-only,
silent, no audio track) is a valid empty result (`speech: false` plus a `reason`), not an error.
"""

import json
import os
import shutil
import subprocess
import threading
from importlib.metadata import version

import numpy as np

from auto_archiver.core import Enricher, Metadata
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOL = "faster-whisper"
SAMPLE_RATE = 16000
TRUNCATION_MARKERS = ("partial file", "Invalid data found when processing input")
DEFAULTS = {
    "model": "small",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 0,
    "language": None,
    "beam_size": 5,
    "word_timestamps": False,
    "download_root": None,
    "local_files_only": False,
    "decode_timeout": 600,
}

_models: dict = {}
_models_lock = threading.Lock()


def _tool_version() -> str:
    return version("faster-whisper")


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise EnricherServiceError(f"{name} is not installed or not on PATH", reason="ffmpeg_missing")
    return path


def _run(cmd: list, timeout: int, what: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise EnricherServiceError(f"{what} timed out after {timeout}s", reason="decode_timeout")
    except OSError as e:
        raise EnricherServiceError(f"{what} could not be started: {e}", reason="ffmpeg_failed")


def _stderr(proc: subprocess.CompletedProcess) -> str:
    return proc.stderr.decode("utf-8", "replace").strip()[-500:]


def _probe(path: str, timeout: int) -> dict:
    proc = _run(
        [_binary("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path], timeout, "ffprobe"
    )
    if proc.returncode != 0:
        raise EnricherInputError(f"ffprobe can't read {path}: {_stderr(proc)}", reason="unreadable_media")
    try:
        probe = json.loads(proc.stdout or b"{}")
    except ValueError:
        raise EnricherInputError(f"ffprobe returned no usable data for {path}", reason="unreadable_media")
    if not probe.get("streams"):
        raise EnricherInputError(f"{path} has no media streams", reason="no_streams")
    return probe


def _decode(path: str, stream_index: int, timeout: int) -> tuple[np.ndarray, str]:
    """Mono 16 kHz float32 samples of one audio stream, plus any non-fatal ffmpeg error output."""
    cmd = [_binary("ffmpeg"), "-nostdin", "-hide_banner", "-v", "error", "-i", path]
    cmd += ["-map", f"0:{stream_index}", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    proc = _run(cmd, timeout, "ffmpeg decode")
    usable = len(proc.stdout) - len(proc.stdout) % 4
    audio = np.frombuffer(proc.stdout[:usable], dtype=np.float32)
    errors = _stderr(proc)
    if audio.size == 0 or proc.returncode != 0:
        raise EnricherInputError(f"no audio could be decoded from {path}: {errors}", reason="audio_decode_failed")
    if any(marker in errors for marker in TRUNCATION_MARKERS):
        # ffmpeg exits 0 on a cut-off file and decodes up to the cut; Whisper then tends to invent a word there
        raise EnricherInputError(f"{path} is truncated: {errors}", reason="truncated_media")
    return audio, errors


def _load_model(opts: dict):
    key = (
        opts["model"],
        opts["device"],
        opts["compute_type"],
        opts["cpu_threads"],
        opts["download_root"],
        opts["local_files_only"],
    )
    with _models_lock:
        if key in _models:
            return _models[key]
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise EnricherServiceError(f"faster-whisper is not installed: {e}", reason="tool_missing")
        try:
            model = WhisperModel(
                opts["model"],
                device=opts["device"],
                compute_type=opts["compute_type"],
                cpu_threads=opts["cpu_threads"],
                download_root=opts["download_root"],
                local_files_only=opts["local_files_only"],
            )
        except MemoryError as e:
            raise EnricherServiceError(f"out of memory loading model {opts['model']}: {e}", reason="out_of_memory")
        except Exception as e:
            # Hub failures surface as LocalEntryNotFoundError (a FileNotFoundError), JSONDecodeError (a ValueError),
            # KeyError on odd payloads, or ctranslate2 RuntimeErrors: none of them say anything about the media.
            raise EnricherServiceError(
                f"could not load model {opts['model']!r}: {type(e).__name__}: {e}", reason="model_unavailable"
            )
        _models[key] = model
        return model


def _empty(result: dict, reason: str) -> dict:
    result.update(
        speech=False,
        reason=reason,
        language=None,
        language_probability=None,
        language_candidates=[],
        text="",
        segments=[],
    )
    return result


def analyze(path: str, **opts) -> dict:
    """Transcribe the first audio track of `path` and detect its spoken language.

    Raises EnricherInputError when the file can't be read as media, EnricherServiceError when ffmpeg or the model fails.
    """
    unknown = set(opts) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown option(s): {', '.join(sorted(unknown))}")
    opts = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None}}

    if not os.path.isfile(path):
        raise EnricherInputError(f"{path} does not exist or is not a file", reason="not_found")
    if os.path.getsize(path) == 0:
        raise EnricherInputError(f"{path} is empty (0 bytes)", reason="empty_file")

    result = {
        "tool": TOOL,
        "tool_version": _tool_version(),
        "model": opts["model"],
        "device": opts["device"],
        "compute_type": opts["compute_type"],
        "vad": "silero",
        "duration": 0.0,
        "speech_duration": 0.0,
    }
    probe = _probe(path, opts["decode_timeout"])
    audio_streams = [s for s in probe["streams"] if s.get("codec_type") == "audio"]
    if not audio_streams:
        return _empty(result, "no_audio_stream")

    audio, decode_errors = _decode(path, audio_streams[0]["index"], opts["decode_timeout"])
    result["duration"] = round(audio.size / SAMPLE_RATE, 3)
    if decode_errors:
        result["decode_errors"] = decode_errors

    model = _load_model(opts)
    try:
        segments, info = model.transcribe(
            audio,
            language=opts["language"],
            beam_size=opts["beam_size"],
            word_timestamps=opts["word_timestamps"],
            vad_filter=True,
            condition_on_previous_text=False,
        )
        segments = [s for s in segments if s.text.strip()]
    except MemoryError as e:
        raise EnricherServiceError(f"out of memory transcribing {path}: {e}", reason="out_of_memory")
    except Exception as e:
        oom = "bad_alloc" in str(e) or "out of memory" in str(e).lower()
        raise EnricherServiceError(
            f"transcription failed for {path}: {type(e).__name__}: {e}",
            reason="out_of_memory" if oom else "transcription_failed",
        )

    result["speech_duration"] = round(float(info.duration_after_vad), 3)
    if not segments:
        return _empty(result, "no_speech_detected" if not info.duration_after_vad else "no_transcribable_speech")

    out_segments = []
    for s in segments:
        seg = {
            "start": round(float(s.start), 3),
            "end": round(float(s.end), 3),
            "text": s.text.strip(),
            "avg_logprob": round(float(s.avg_logprob), 4),
            "no_speech_prob": round(float(s.no_speech_prob), 4),
        }
        if opts["word_timestamps"]:
            seg["words"] = [
                {
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "word": w.word,
                    "probability": round(float(w.probability), 4),
                }
                for w in s.words or []
            ]
        out_segments.append(seg)

    result.update(
        speech=True,
        reason="speech",
        language=info.language,
        language_probability=round(float(info.language_probability), 4),
        language_candidates=[
            {"language": lang, "probability": round(float(p), 4)} for lang, p in (info.all_language_probs or [])[:3]
        ],
        text=" ".join(s["text"] for s in out_segments),
        segments=out_segments,
    )
    return result


class FasterWhisperEnricher(Enricher):
    """Adds a faster-whisper transcript and spoken language to every video/audio Media, under `transcript`."""

    def setup(self) -> None:
        self.opts = {k: getattr(self, k) for k in DEFAULTS}

    def enrich(self, to_enrich: Metadata) -> None:
        for m in to_enrich.media:
            if not (m.is_video() or m.is_audio()):
                continue
            try:
                m.set("transcript", analyze(m.filename, **self.opts))
            except EnricherInputError as e:
                logger.warning(f"faster_whisper_enricher: can't analyse {m.filename} ({e.reason}): {e}")
                m.set("transcript", {"tool": TOOL, "error": e.reason})
