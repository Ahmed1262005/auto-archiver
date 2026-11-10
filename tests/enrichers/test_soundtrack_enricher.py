"""soundtrack_enricher tests: helpers on generated signals, input/option errors on ffmpeg fixtures, the AcoustID
client against a mocked endpoint, and full Demucs/beat_this/CLAP runs when the models are installed.

Real ads are not committed. Point SOUNDTRACK_ENRICHER_SAMPLES at a folder of real ad files to run the real-sample
tests; files named <name>_<seconds>.mp4 that share <name> are treated as cut-downs of one ad.
"""

import glob
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from auto_archiver.core import Media, Metadata
from auto_archiver.modules.audio_enricher import fingerprint_similarity
from auto_archiver.modules.soundtrack_enricher import soundtrack_enricher as st
from auto_archiver.modules.soundtrack_enricher import SoundtrackEnricher, analyze
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOLS = all(shutil.which(b) for b in ("ffmpeg", "ffprobe", "fpcalc"))
needs_tools = pytest.mark.skipif(not TOOLS, reason="ffmpeg, ffprobe and fpcalc are required")
MODELS = all(importlib.util.find_spec(m) for m in ("torch", "demucs", "beat_this", "librosa", "transformers"))
needs_models = pytest.mark.skipif(not (TOOLS and MODELS), reason="pip install -r soundtrack_enricher/requirements.txt")
SAMPLES = os.environ.get("SOUNDTRACK_ENRICHER_SAMPLES", "")
REAL = sorted(glob.glob(os.path.join(SAMPLES, "*.mp4"))) if SAMPLES else []
needs_real = pytest.mark.skipif(not (REAL and MODELS), reason="set SOUNDTRACK_ENRICHER_SAMPLES to real ad files")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("soundtrack")
    if not TOOLS:
        return d
    ffmpeg("-f", "lavfi", "-i", "testsrc=s=64x64:d=2", "-pix_fmt", "yuv420p", str(d / "mute.mp4"))
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "3", str(d / "silent.wav"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=8", "-ac", "2", str(d / "tone.wav"))
    (d / "garbage.mp4").write_bytes(b"\x00not a video" * 100)
    return d


def test_ranges_join_gaps_and_drop_short():
    db = np.array([-80, -30, -30, -80, -30, -30, -30, -80, -80, -80, -30, -80], dtype=float)
    assert st._ranges(db, -42, 1.0, 1.0) == [{"start": 0.5, "end": 3.5}]
    assert st._ranges(db, -42, 0.5, 0.0) == [{"start": 0.5, "end": 1.5}, {"start": 2.0, "end": 3.5},
                                              {"start": 5.0, "end": 5.5}]  # fmt: skip
    assert st._ranges(np.full(4, -90.0), -42, 1.0, 1.0) == []


def test_db_per_half_second():
    x = np.zeros((2, int(st.SR * 1.25)), dtype=np.float32)
    x[:, : st.SR // 2] = 0.5
    db = st._db(x)
    assert len(db) == 3 and abs(db[0] - 20 * np.log10(0.5)) < 0.1 and db[1] < -100


def test_key_of_a_major_triad():
    t = np.arange(st.SR * 4) / st.SR
    y = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.18, 329.63)).astype(np.float32) / 3
    k = st._key(y)
    assert k["key"] in ("A major", "F# minor") and k["key_confidence"] > 0.5


def test_merge_and_overlap():
    assert st._merge([(5, 10), (0, 5), (12, 13)]) == [(0, 10), (12, 13)]
    assert st._overlap(0, 10, [{"start": 2, "end": 4}, {"start": 8, "end": 20}]) == 4


@needs_tools
def test_missing_and_unreadable(fx):
    with pytest.raises(EnricherInputError) as e:
        analyze(str(fx / "nope.mp4"))
    assert e.value.reason == "missing_file"
    with pytest.raises(EnricherInputError):
        analyze(str(fx / "garbage.mp4"))


@needs_tools
def test_video_without_audio_is_not_an_error(fx):
    r = analyze(str(fx / "mute.mp4"))
    assert r["status"] == "no_audio_stream" and r["music"]["present"] is False and r["fingerprint"] is None
    json.dumps(r)


@needs_tools
def test_bad_options(fx):
    with pytest.raises(TypeError):
        analyze(str(fx / "tone.wav"), colour="blue")


@needs_models
def test_device_checks(fx, mocker):
    with pytest.raises(ValueError):
        analyze(str(fx / "tone.wav"), device="tpu")
    import torch

    mocker.patch.object(torch.cuda, "is_available", return_value=False)
    with pytest.raises(EnricherServiceError) as e:
        analyze(str(fx / "tone.wav"), device="cuda")
    assert e.value.reason == "device_unavailable"
    assert st._device(torch, "auto") == "cpu"


@needs_models
def test_silent_audio(fx):
    assert analyze(str(fx / "silent.wav"))["status"] == "silent_audio"


@needs_models
def test_torch_crash_is_service_error(fx, mocker):
    mocker.patch.object(st, "_separate", side_effect=lambda *a: st._infer(_boom, "demucs separation"))
    with pytest.raises(EnricherServiceError) as e:
        analyze(str(fx / "tone.wav"))
    assert e.value.reason == "out_of_memory"


def _boom():
    raise RuntimeError("CUDA out of memory")


@needs_models
def test_tone_end_to_end(fx, tmp_path):
    r = analyze(str(fx / "tone.wav"), cpu_threads=2, keep_stems_dir=str(tmp_path))
    assert r["status"] == "ok", r["status"]
    assert r["music"]["present"] and r["music"]["fraction"] > 0.8
    assert r["fingerprint"]["status"] == "ok" and r["fingerprint"]["items"] > 0
    assert r["vocals"] == "instrumental" and r["acoustid"] is None
    assert len(r["embedding"]) == 512 and {g["label"] for g in r["genre"]} <= set(st.GENRES)
    assert os.path.isfile(tmp_path / "music.wav") and os.path.isfile(tmp_path / "vocals.wav")
    json.dumps(r)


def test_acoustid_parses_and_rate_limits(tmp_path, mocker):
    reply = {"status": "ok", "results": [{"id": "a1", "score": 0.93, "recordings": [
        {"id": "mb1", "title": "Down", "artists": [{"name": "Marian Hill"}]}]}]}  # fmt: skip
    run = mocker.patch.object(st._audio, "_run", return_value=subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"duration": 61.2, "fingerprint": "AQAA"}), stderr=""))  # fmt: skip
    seen = []

    def urlopen(req, timeout):
        seen.append(req.data.decode())
        return io.BytesIO(json.dumps(reply).encode())

    mocker.patch.object(st.urllib.request, "urlopen", side_effect=urlopen)
    sleep = mocker.patch.object(st.time, "sleep")
    o = {**st.DEFAULTS, "acoustid_key": "k"}
    r1, _ = st._acoustid("m.wav", o), st._acoustid("m.wav", o)
    assert r1["best"] == {"score": 0.93, "acoustid": "a1", "recording_mbid": "mb1", "title": "Down",
                          "artists": ["Marian Hill"]}  # fmt: skip
    assert "client=k" in seen[0] and "duration=61" in seen[0] and run.call_count == 2
    assert sleep.called


def test_acoustid_error_is_service_error(mocker):
    mocker.patch.object(st._audio, "_run", return_value=subprocess.CompletedProcess(
        [], 0, stdout=json.dumps({"duration": 5, "fingerprint": "AQAA"}), stderr=""))  # fmt: skip
    mocker.patch.object(st.urllib.request, "urlopen", return_value=io.BytesIO(
        b'{"status": "error", "error": {"code": 4, "message": "invalid API key"}}'))  # fmt: skip
    with pytest.raises(EnricherServiceError):
        st._acoustid("m.wav", {**st.DEFAULTS, "acoustid_key": "bad"})


def test_naming_off_without_key():
    assert st.DEFAULTS["acoustid_key"] == (os.environ.get("ACOUSTID_API_KEY") or None)


@needs_tools
def test_enricher_stores_result_or_error(fx, mocker):
    item = Metadata()
    item.add_media(Media(str(fx / "mute.mp4")))
    item.add_media(Media(str(fx / "garbage.mp4")))
    for m in item.media:
        m.mimetype = "video/mp4"
    e = SoundtrackEnricher.__new__(SoundtrackEnricher)
    e.enrich(item)
    assert item.media[0].get("soundtrack")["status"] == "no_audio_stream"
    assert item.media[1].get("soundtrack_error")["type"] == "EnricherInputError"


def _cutdowns():
    groups = {}
    for f in REAL:
        m = re.match(r"(.+)_(\d+)\.mp4$", os.path.basename(f))
        if m:
            groups.setdefault(m.group(1), []).append(f)
    return [sorted(v) for v in groups.values() if len(v) > 1]


@needs_real
@pytest.mark.parametrize("files", _cutdowns() or [pytest.param([], marks=pytest.mark.skip("no cut-downs"))])
def test_real_cutdowns_share_the_music_fingerprint(files):
    rs = [analyze(f, cpu_threads=4) for f in files]
    for r in rs:
        assert r["status"] == "ok" and r["music"]["present"] and r["bpm"]
    short = min(rs, key=lambda r: r["fingerprint"]["items"])
    for r in rs:
        if r is not short:
            sim = fingerprint_similarity(short["fingerprint"]["raw"], r["fingerprint"]["raw"])
            assert sim["match"], (files, sim)
