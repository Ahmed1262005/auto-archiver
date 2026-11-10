"""Soundtrack enricher: the music under an ad, separated from the voice-over and described.

Steps, all local and open source: ffmpeg decodes the audio; Demucs v4 (htdemucs, MIT) splits it into vocals and
the rest, the rest being the music bed; the music bed gets a Chromaprint fingerprint (fpcalc, LGPL-2.1+) so the same
track can be found in other ads even when it has no name, music time ranges from its loudness, tempo from beat_this
(MIT), key from a Krumhansl-Schmuckler profile on librosa chroma (ISC), and genre, mood and sung-vs-spoken from LAION
CLAP (larger_clap_music_and_speech, Apache-2.0) zero-shot. AcoustID/MusicBrainz naming is optional and off unless
`acoustid_key` is set; the hosted AcoustID API is for non-commercial use only (self-host acoustid-server or buy a
plan for commercial use; point `acoustid_url` at it).
"""

import importlib.metadata
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

from auto_archiver.core import Enricher, Metadata
from auto_archiver.modules.audio_enricher import audio_enricher as _audio
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOL = "soundtrack"
TOOL_VERSION = "1"
SR = 44100
CLAP_SR = 48000
FRAME_S = 0.5

DEFAULTS = {
    "device": "cpu",
    "cpu_threads": 0,
    "separation_model": "htdemucs",
    "separation_shifts": 0,
    "clap_model": "laion/larger_clap_music_and_speech",
    "clap_window_s": 10.0,
    "beat_model": "final0",
    "music_threshold_db": -42.0,
    "voice_threshold_db": -40.0,
    "min_music_s": 1.0,
    "max_gap_s": 1.0,
    "max_seconds": 600,
    "acoustid_key": os.environ.get("ACOUSTID_API_KEY") or None,
    "acoustid_url": "https://api.acoustid.org/v2/lookup",
    "model_dir": os.environ.get("SOUNDTRACK_MODEL_DIR") or None,
    "keep_stems_dir": None,
    "timeout": 900,
}

GENRES = ("pop", "rock", "hip hop", "electronic dance", "classical orchestral", "jazz", "country", "r&b soul",
          "folk acoustic", "ambient", "cinematic score", "funk", "latin", "bollywood indian", "christmas")  # fmt: skip
MOODS = ("happy upbeat", "sad melancholic", "energetic", "calm relaxing", "romantic", "dark tense",
         "epic triumphant", "playful quirky", "nostalgic")  # fmt: skip
VOICE = ("singing voice, a song with lyrics", "a narrator speaking, voice-over")
VOICE_WINDOW_S = 5.0
# cosine * scale before the softmax; the checkpoint's own logit scale is not loaded by transformers 5
CLAP_SCALE = 20.0
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

_MODELS = {}
_LOCK = threading.Lock()
_ACOUSTID_LOCK = threading.Lock()
_ACOUSTID_LAST = [0.0]


def analyze(path: str, **opts) -> dict:
    """Separate, fingerprint and describe the music of one audio or video file. Options (all optional): device
    (cpu|cuda|auto), cpu_threads, separation_model, separation_shifts, clap_model, clap_window_s, beat_model,
    music_threshold_db, min_music_s, max_gap_s, max_seconds, acoustid_key, acoustid_url, model_dir, keep_stems_dir,
    timeout. Times are seconds from the start of the file. Raises EnricherInputError / EnricherServiceError."""
    unknown = set(opts) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"analyze() got unknown options: {sorted(unknown)}")
    o = {**DEFAULTS, **{k: v for k, v in opts.items() if v is not None}}
    if not os.path.isfile(path):
        raise EnricherInputError(f"{path} does not exist", reason="missing_file")
    probe = _audio._probe(path, "ffprobe", float(o["timeout"]))
    streams = probe.get("streams") or []
    result = {
        "tool": TOOL,
        "tool_version": f"{TOOL_VERSION}+{o['separation_model']}",
        "versions": {d: _version(d) for d in ("demucs", "torch", "beat-this", "librosa", "transformers")},
        "models": {"separation": o["separation_model"], "clap": o["clap_model"], "beats": o["beat_model"]},
        "duration_s": _audio._float((probe.get("format") or {}).get("duration")),
        "status": "ok",
        "music": {"present": False, "ranges": [], "fraction": 0.0},
        "vocals": None,
        "voice_ranges": {"singing": [], "speech": []},
        "bpm": None,
        "beats": 0,
        "key": None,
        "key_confidence": None,
        "genre": [],
        "mood": [],
        "embedding": None,
        "fingerprint": None,
        "acoustid": None,
        "timings": {},
    }
    if not any(s.get("codec_type") == "audio" for s in streams):
        result["status"] = "no_audio_stream"
        return result
    torch = _torch(int(o["cpu_threads"]))
    device = _device(torch, o["device"])
    result["device"] = device
    tmp = tempfile.mkdtemp(prefix="soundtrack-")
    try:
        t = time.time()
        mix = _decode(path, os.path.join(tmp, "mix.f32"), float(o["max_seconds"]), float(o["timeout"]))
        if mix.shape[1] < SR or float(np.abs(mix).max(initial=0)) < 1e-4:
            result["status"] = "silent_audio"
            return result
        result["timings"]["decode_s"] = round(time.time() - t, 2)

        t = time.time()
        vocals, music = _separate(torch, mix, o, device)
        result["timings"]["separate_s"] = round(time.time() - t, 2)

        ranges = _ranges(_db(music), float(o["music_threshold_db"]), float(o["min_music_s"]), float(o["max_gap_s"]))
        total = mix.shape[1] / SR
        result["music"] = {
            "present": bool(ranges),
            "ranges": ranges,
            "fraction": round(sum(r["end"] - r["start"] for r in ranges) / total, 3) if total else 0.0,
        }
        if not ranges:
            result["status"] = "no_music"
            return result

        music_wav = os.path.join(o["keep_stems_dir"] or tmp, "music.wav")
        _write_wav(music_wav, music)
        if o["keep_stems_dir"]:
            _write_wav(os.path.join(o["keep_stems_dir"], "vocals.wav"), vocals)
        t = time.time()
        result["fingerprint"] = _audio._fingerprint(music_wav, "fpcalc", 0, float(o["timeout"]))
        result["timings"]["fingerprint_s"] = round(time.time() - t, 2)

        mono = music.mean(0)
        t = time.time()
        result.update(_tempo(torch, mono, o, device))
        result.update(_key(mono))
        result["timings"]["tempo_key_s"] = round(time.time() - t, 2)

        t = time.time()
        result.update(_clap(torch, mono, vocals.mean(0), ranges, o, device))
        result["timings"]["clap_s"] = round(time.time() - t, 2)

        if o["acoustid_key"]:
            t = time.time()
            result["acoustid"] = _acoustid(music_wav, o)
            result["timings"]["acoustid_s"] = round(time.time() - t, 2)
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class SoundtrackEnricher(Enricher):
    """Runs `analyze` on every video and audio file in the item and stores the result on the Media under
    `soundtrack` (or `soundtrack_error`)."""

    def enrich(self, to_enrich: Metadata) -> None:
        opts = {k: getattr(self, k, None) for k in DEFAULTS}
        for media in to_enrich.media:
            if not (media.is_video() or media.is_audio()):
                continue
            try:
                result = analyze(media.filename, **opts)
            except (EnricherInputError, EnricherServiceError) as e:
                logger.warning(f"soundtrack failed for {media.filename}: {type(e).__name__} {e.reason}: {e}")
                media.set("soundtrack_error", {"type": type(e).__name__, "reason": e.reason, "message": str(e)})
                continue
            media.set("soundtrack", result)
            logger.debug(f"soundtrack {media.filename}: {result['status']} bpm={result['bpm']} key={result['key']}")


def _version(dist: str):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _torch(threads: int):
    try:
        import torch
    except ImportError as e:
        raise EnricherServiceError(
            "torch is not installed (pip install -r soundtrack_enricher/requirements.txt)", reason="tool_missing"
        ) from e
    if threads > 0 and torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    return torch


def _device(torch, want: str) -> str:
    if want == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want not in ("cpu", "cuda") and not str(want).startswith("cuda:"):
        raise ValueError(f"device must be cpu, cuda or auto, got {want!r}")
    if want != "cpu" and not torch.cuda.is_available():
        raise EnricherServiceError(f"device {want} requested but CUDA is not available", reason="device_unavailable")
    return want


def _decode(path: str, out: str, max_seconds: float, timeout: float) -> np.ndarray:
    """Stereo float32 at 44.1 kHz, shape (2, n)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", path, "-vn", "-ac", "2", "-ar", str(SR)]
    if max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    proc = _audio._run([*cmd, "-f", "f32le", out], timeout)
    if proc.returncode != 0:
        _audio._fail("ffmpeg", proc, path)
    data = np.fromfile(out, dtype=np.float32)
    return data[: len(data) // 2 * 2].reshape(-1, 2).T.copy()


def _write_wav(path: str, x: np.ndarray) -> None:
    import soundfile

    soundfile.write(path, np.clip(x.T, -1, 1), SR, subtype="PCM_16")


def _cached(key, load):
    with _LOCK:
        if key not in _MODELS:
            _MODELS[key] = load()
        return _MODELS[key]


def _infer(fn, what: str):
    try:
        return fn()
    except MemoryError as e:
        raise EnricherServiceError(f"{what} ran out of memory", reason="out_of_memory") from e
    except RuntimeError as e:
        reason = "out_of_memory" if "out of memory" in str(e).lower() else "inference_failed"
        raise EnricherServiceError(f"{what} failed: {e}", reason=reason) from e


def _separate(torch, mix: np.ndarray, o: dict, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(vocals, music) stereo stems; music = everything Demucs does not call vocals (drums, bass, other)."""
    try:
        import demucs.api
    except ImportError as e:
        raise EnricherServiceError("demucs is not installed", reason="tool_missing") from e

    def load():
        if o["model_dir"]:
            torch.hub.set_dir(o["model_dir"])
        return demucs.api.Separator(model=o["separation_model"], device=device, shifts=int(o["separation_shifts"]),
                                    overlap=0.25, progress=False)  # fmt: skip

    try:
        sep = _cached(("demucs", o["separation_model"], device, int(o["separation_shifts"]), o["model_dir"]), load)
    except demucs.api.LoadModelError as e:
        raise EnricherServiceError(f"demucs model {o['separation_model']}: {e}", reason="model_unavailable") from e
    except OSError as e:
        raise EnricherServiceError(f"demucs model {o['separation_model']}: {e}", reason="model_unavailable") from e

    def run():
        with torch.inference_mode():
            _, stems = sep.separate_tensor(torch.from_numpy(mix), SR)
        vocals = stems["vocals"].cpu().numpy()
        music = sum(v for k, v in stems.items() if k != "vocals").cpu().numpy()
        return vocals, music

    return _infer(run, "demucs separation")


def _db(x: np.ndarray) -> np.ndarray:
    """RMS level in dBFS per FRAME_S frame of a stereo signal."""
    n = int(SR * FRAME_S)
    m = x.mean(0)
    frames = m[: len(m) // n * n].reshape(-1, n)
    if len(m) % n:
        frames = np.vstack([frames, np.pad(m[len(m) // n * n :], (0, n - len(m) % n))]) if len(frames) else m[None]
    return 20 * np.log10(np.sqrt((frames**2).mean(1)) + 1e-9)


def _ranges(db: np.ndarray, threshold: float, min_len: float, max_gap: float) -> list[dict]:
    on = db > threshold
    out = []
    for i, v in enumerate(on):
        t = i * FRAME_S
        if not v:
            continue
        if out and t - out[-1][1] <= max_gap:
            out[-1][1] = t + FRAME_S
        else:
            out.append([t, t + FRAME_S])
    return [{"start": round(a, 2), "end": round(b, 2)} for a, b in out if b - a >= min_len]


def _tempo(torch, mono: np.ndarray, o: dict, device: str) -> dict:
    try:
        from beat_this.inference import Audio2Beats
    except ImportError as e:
        raise EnricherServiceError("beat_this is not installed", reason="tool_missing") from e
    model = _cached(("beat_this", o["beat_model"], device), lambda: Audio2Beats(checkpoint_path=o["beat_model"],
                                                                                 device=device, dbn=False))  # fmt: skip
    beats, downbeats = _infer(lambda: model(mono, SR), "beat_this")
    ibi = np.diff(np.asarray(beats, dtype=float))
    bpm = round(60.0 / float(np.median(ibi)), 1) if len(ibi) >= 4 else None
    return {"bpm": bpm, "beats": int(len(beats)), "downbeats": int(len(downbeats))}


def _key(mono: np.ndarray) -> dict:
    import librosa

    y = librosa.resample(mono, orig_sr=SR, target_sr=22050)
    chroma = librosa.feature.chroma_cqt(y=y, sr=22050).mean(1)
    if not np.isfinite(chroma).all() or chroma.std() == 0:
        return {"key": None, "key_confidence": None}
    best = max(
        (float(np.corrcoef(np.roll(prof, i), chroma)[0, 1]), f"{PITCHES[i]} {mode}")
        for prof, mode in ((MAJOR, "major"), (MINOR, "minor"))
        for i in range(12)
    )
    return {"key": best[1], "key_confidence": round(best[0], 3)}


def _windows(x: np.ndarray, sr: int, win_s: float) -> list[tuple[float, np.ndarray]]:
    w, hop = int(win_s * sr), int(win_s * sr / 2)
    starts = range(0, max(1, len(x) - w + hop), hop) if len(x) > w else [0]
    return [(i / sr, x[i : i + w]) for i in starts]


def _clap(torch, music: np.ndarray, vocals: np.ndarray, ranges: list[dict], o: dict, device: str) -> dict:
    try:
        import librosa
        from transformers import ClapModel, ClapProcessor
    except ImportError as e:
        raise EnricherServiceError("transformers is not installed", reason="tool_missing") from e

    def load():
        kw = {"cache_dir": o["model_dir"]} if o["model_dir"] else {}
        model = ClapModel.from_pretrained(o["clap_model"], **kw).eval().to(device)
        proc = ClapProcessor.from_pretrained(o["clap_model"], **kw)
        with torch.no_grad():
            texts = {name: _norm(torch, _vec(model.get_text_features(**proc(text=list(labels), return_tensors="pt",
                                                                               padding=True).to(device))))
                     for name, labels in (("genre", [f"This is a {g} song." for g in GENRES]),
                                          ("mood", [f"This music sounds {m}." for m in MOODS]),
                                          ("voice", list(VOICE)))}  # fmt: skip
        return model, proc, texts

    try:
        model, proc, texts = _cached(("clap", o["clap_model"], device, o["model_dir"]), load)
    except OSError as e:
        raise EnricherServiceError(f"CLAP model {o['clap_model']}: {e}", reason="model_unavailable") from e

    def embed(x, win_s):
        wins = _windows(librosa.resample(x, orig_sr=SR, target_sr=CLAP_SR), CLAP_SR, win_s)
        return [t for t, _ in wins], _embed([w for _, w in wins])

    def _embed(wins):
        out = []
        for i in range(0, len(wins), 8):
            batch = proc(audio=wins[i : i + 8], sampling_rate=CLAP_SR, return_tensors="pt")
            with torch.no_grad():
                out.append(_norm(torch, _vec(model.get_audio_features(**batch.to(device)))))
        return torch.cat(out)

    def run():
        win = float(o["clap_window_s"])
        starts, a = embed(music, win)
        keep = [i for i, s in enumerate(starts) if _overlap(s, s + win, ranges) >= win / 2] or list(range(len(starts)))
        mean = _norm(torch, a[keep].mean(0, keepdim=True))
        top = {}
        for name, labels in (("genre", GENRES), ("mood", MOODS)):
            p = (CLAP_SCALE * mean @ texts[name].T).softmax(-1)[0]
            top[name] = [{"label": labels[j], "score": round(float(p[j]), 3)} for j in p.argsort(descending=True)[:3]]
        # only windows where the vocal stem is audible; CLAP then tells singing from a voice-over
        vw = _windows(librosa.resample(vocals, orig_sr=SR, target_sr=CLAP_SR), CLAP_SR, VOICE_WINDOW_S)
        vw = [(t, w) for t, w in vw if 20 * np.log10(np.sqrt((w**2).mean()) + 1e-9) > float(o["voice_threshold_db"])]
        sung, spoken = [], []
        if vw:
            pv = (CLAP_SCALE * _embed([w for _, w in vw]) @ texts["voice"].T).softmax(-1)
            for (t, _), p in zip(vw, pv):
                (sung if float(p[0]) > 0.5 else spoken).append((t, t + VOICE_WINDOW_S))
        sung_in_music = sum(_overlap(a_, b_, ranges) for a_, b_ in _merge(sung))
        music_s = sum(r["end"] - r["start"] for r in ranges)
        return {
            "genre": top["genre"],
            "mood": top["mood"],
            "vocals": "vocal" if music_s and sung_in_music / music_s >= 0.25 else "instrumental",
            "voice_ranges": {"singing": _as_ranges(_merge(sung)), "speech": _as_ranges(_merge(spoken))},
            "embedding": [round(float(x), 5) for x in mean[0].cpu()],
            "embedding_model": o["clap_model"],
        }

    return _infer(run, "CLAP")


def _vec(x):
    return x if hasattr(x, "norm") else x.pooler_output


def _norm(torch, x):
    return torch.nn.functional.normalize(x, dim=-1)


def _overlap(a: float, b: float, ranges: list[dict]) -> float:
    return sum(max(0.0, min(b, r["end"]) - max(a, r["start"])) for r in ranges)


def _merge(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _as_ranges(spans) -> list[dict]:
    return [{"start": round(a, 2), "end": round(b, 2)} for a, b in spans]


def _acoustid(music_wav: str, o: dict) -> dict:
    """AcoustID lookup of the music stem; at most 3 requests per second per process, as the hosted API asks."""
    proc = _audio._run(["fpcalc", "-json", music_wav], float(o["timeout"]))
    if proc.returncode != 0:
        _audio._fail("fpcalc", proc, music_wav)
    fp = json.loads(proc.stdout)
    body = urllib.parse.urlencode({"client": o["acoustid_key"], "duration": int(fp["duration"]),
                                   "fingerprint": fp["fingerprint"], "meta": "recordings releasegroups"}).encode()  # fmt: skip
    with _ACOUSTID_LOCK:
        wait = 0.34 - (time.time() - _ACOUSTID_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _ACOUSTID_LAST[0] = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(o["acoustid_url"], data=body), timeout=30) as r:
            data = json.loads(r.read())
    except (OSError, ValueError) as e:
        raise EnricherServiceError(f"AcoustID lookup failed: {e}", reason="service_unavailable") from e
    if data.get("status") != "ok":
        raise EnricherServiceError(f"AcoustID: {data.get('error')}", reason="service_error")
    matches = []
    for res in data.get("results") or []:
        for rec in res.get("recordings") or [{}]:
            matches.append({
                "score": round(float(res.get("score") or 0), 3),
                "acoustid": res.get("id"),
                "recording_mbid": rec.get("id"),
                "title": rec.get("title"),
                "artists": [a.get("name") for a in rec.get("artists") or []],
            })  # fmt: skip
    return {"matches": matches[:5], "best": matches[0] if matches else None}
