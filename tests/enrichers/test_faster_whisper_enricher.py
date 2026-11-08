import glob
import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("faster_whisper", reason="faster-whisper extra not installed")

from auto_archiver.core import Media, Metadata
from auto_archiver.modules.faster_whisper_enricher import faster_whisper_enricher as fwe
from auto_archiver.modules.faster_whisper_enricher import analyze
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

TEST_MODEL = os.environ.get("FASTER_WHISPER_TEST_MODEL", "small")
# directory with real stored ad media (not committed), e.g. the TikTok ads used to develop this module
AD_SAMPLES_DIR = os.environ.get("AD_SAMPLES_DIR")
MBROLA = shutil.which("espeak-ng") and os.path.isdir("/usr/share/mbrola/fr1") and os.path.isdir("/usr/share/mbrola/ar1")

FR_TEXT = (
    "Bonjour à tous. Découvrez la nouvelle collection de printemps, disponible dès aujourd'hui "
    "dans nos boutiques et en ligne. Profitez de la livraison gratuite."
)
AR_TEXT = "مرحبا بكم. اكتشفوا المجموعة الجديدة المتوفرة اليوم في متاجرنا وعلى الإنترنت. التوصيل مجاني."
EN_TEXT = "Discover the new spring collection, available today in our stores and online."


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def espeak(voice, text, out):
    subprocess.run(["espeak-ng", "-v", voice, "-s", "140", "-w", out, text], check=True, capture_output=True)
    return out


@pytest.fixture(scope="session")
def fx(tmp_path_factory):
    """Small generated fixtures, built once per session and never committed."""
    d = tmp_path_factory.mktemp("fw")
    p = lambda name: str(d / name)  # noqa: E731
    lavfi_video = ["-f", "lavfi", "-i", "testsrc=s=320x240:r=25"]
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-f", "lavfi", "-i", "color=c=black:s=320x568:r=25",
           "-t", "5", "-c:v", "libx264", "-c:a", "aac", "-shortest", p("silent.mp4"))  # fmt: skip
    ffmpeg(*lavfi_video, "-t", "3", "-c:v", "libx264", "-an", p("no_audio.mp4"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=0.5", "-f", "lavfi", "-i", "color=s=320x240", "-t", "0.5",
           "-c:v", "libx264", "-c:a", "aac", "-shortest", p("half_second.mp4"))  # fmt: skip
    # synthetic "music": a C-major chord with tremolo over a noise hi-hat and a kick drum, no voice
    music = (
        "aevalsrc='0.2*(sin(2*PI*261.6*t)+sin(2*PI*329.6*t)+sin(2*PI*392*t))*(0.6+0.4*sin(2*PI*2*t))"
        "+0.3*sin(2*PI*55*t)*exp(-12*mod(t,0.5))+0.05*(random(0)-0.5)*exp(-40*mod(t+0.25,0.5))':s=44100:d=20"
    )
    ffmpeg("-f", "lavfi", "-i", music, *lavfi_video, "-t", "20", "-c:v", "libx264", "-c:a", "aac", "-shortest",
           p("music.mp4"))  # fmt: skip
    # a cut-off download: moov atom at the end (ffprobe can't open it) and at the start (decodes up to the cut)
    ffmpeg(*lavfi_video, "-f", "lavfi", "-i", "sine=f=300:d=6", "-t", "6", "-c:v", "libx264", "-c:a", "aac",
           p("moov_end.mp4"))  # fmt: skip
    ffmpeg("-i", p("moov_end.mp4"), "-c", "copy", "-movflags", "+faststart", p("faststart.mp4"))
    for src, dst in (("moov_end.mp4", "cut_moov_end.mp4"), ("faststart.mp4", "cut_faststart.mp4")):
        with open(p(src), "rb") as f:
            data = f.read()
        with open(p(dst), "wb") as f:
            f.write(data[: len(data) // 2])
    with open(p("page.mp4"), "w") as f:
        f.write("<!doctype html><html><head><title>Ad not found</title></head><body>404</body></html>")
    open(p("empty.mp4"), "wb").close()
    ffmpeg("-f", "lavfi", "-i", "testsrc=s=320x240", "-frames:v", "1", p("still.jpg"))
    if shutil.which("espeak-ng"):
        en = espeak("en-us", EN_TEXT, p("en.wav"))
        ffmpeg("-i", en, "-ar", "44100", "-c:a", "aac", p("en_audio_only.m4a"))
        # 3m05s: speech only starts after 3 minutes of silence
        ffmpeg("-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-i", en, "-filter_complex",
               "[1]adelay=180000|180000[s];[0][s]amix=inputs=2:duration=longest[a]", "-map", "[a]", "-t", "185",
               "-c:a", "aac", p("long.m4a"))  # fmt: skip
        # variable frame rate: 30 fps for 2 s then 5 fps, with English speech
        ffmpeg("-f", "lavfi", "-i", "testsrc=s=320x240:r=30:d=6", "-i", en, "-vf",
               "select='lt(t,2)+not(mod(n,6))'", "-vsync", "vfr", "-c:v", "libx264", "-c:a", "aac", "-shortest",
               p("vfr.mp4"))  # fmt: skip
    if MBROLA:
        fr = espeak("mb-fr1", FR_TEXT, p("fr.wav"))
        espeak("mb-ar1", AR_TEXT, p("ar.wav"))
        # 9:16 portrait video with French speech, and the same video tagged as rotated 90 degrees
        ffmpeg("-i", fr, "-f", "lavfi", "-i", "color=s=576x1024:r=25", "-c:v", "libx264", "-c:a", "aac",
               "-shortest", p("portrait_fr.mp4"))  # fmt: skip
        ffmpeg("-i", p("portrait_fr.mp4"), "-c", "copy", "-metadata:s:v:0", "rotate=90", p("rotated_fr.mp4"))
    return p


@pytest.fixture(scope="session")
def model_opts():
    try:
        fwe._load_model({**fwe.DEFAULTS, "model": TEST_MODEL})
    except EnricherServiceError as e:
        pytest.skip(f"faster-whisper model {TEST_MODEL} unavailable: {e}")
    return {"model": TEST_MODEL}


def needs(path):
    if not os.path.exists(path):
        pytest.skip(f"fixture {os.path.basename(path)} needs espeak-ng/mbrola")
    return path


def assert_serialisable(r):
    assert json.loads(json.dumps(r)) == r
    assert r["tool"] == "faster-whisper" and r["tool_version"] and r["model"]


def assert_empty(r, reason):
    assert_serialisable(r)
    assert r["speech"] is False and r["reason"] == reason
    assert r["text"] == "" and r["segments"] == [] and r["language"] is None


# --- input errors: never retry, never need the model ---


@pytest.mark.parametrize(
    "name, reason",
    [
        ("empty.mp4", "empty_file"),
        ("page.mp4", "unreadable_media"),
        ("cut_moov_end.mp4", "unreadable_media"),
        ("cut_faststart.mp4", "truncated_media"),
        ("does_not_exist.mp4", "not_found"),
    ],
)
def test_unreadable_media_raises_input_error(fx, name, reason, mocker):
    load = mocker.patch.object(fwe, "_load_model")
    with pytest.raises(EnricherInputError) as e:
        analyze(fx(name))
    assert e.value.reason == reason
    assert isinstance(e.value, ValueError) and not isinstance(e.value, EnricherServiceError)
    load.assert_not_called()


def test_directory_is_not_media(tmp_path):
    with pytest.raises(EnricherInputError) as e:
        analyze(str(tmp_path))
    assert e.value.reason == "not_found"


@pytest.mark.parametrize("name", ["no_audio.mp4", "still.jpg"])
def test_no_audio_stream_is_empty_result(fx, name, mocker):
    load = mocker.patch.object(fwe, "_load_model")
    assert_empty(analyze(fx(name)), "no_audio_stream")
    load.assert_not_called()


def test_unknown_option_is_a_caller_bug():
    with pytest.raises(TypeError):
        analyze("x.mp4", modle="small")


# --- service errors: ffmpeg / model / runtime ---


def test_missing_ffmpeg(fx, mocker):
    mocker.patch.object(fwe.shutil, "which", return_value=None)
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("silent.mp4"))
    assert e.value.reason == "ffmpeg_missing"


def test_ffprobe_timeout(fx, mocker):
    mocker.patch.object(fwe.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffprobe", 1))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("silent.mp4"), decode_timeout=1)
    assert e.value.reason == "decode_timeout"


class FakeModel:
    def __init__(self, exc):
        self.exc = exc

    def transcribe(self, audio, **kw):
        def gen():
            raise self.exc
            yield

        # faster-whisper decodes lazily: errors surface while iterating the segment generator
        return gen(), None


@pytest.mark.parametrize(
    "exc, reason",
    [
        (RuntimeError("std::bad_alloc"), "out_of_memory"),
        (MemoryError(), "out_of_memory"),
        (RuntimeError("CUDA failed with error unspecified launch failure"), "transcription_failed"),
        (ValueError("operands could not be broadcast together"), "transcription_failed"),
    ],
)
def test_transcription_crash_is_service_error(fx, mocker, exc, reason):
    mocker.patch.object(fwe, "_load_model", return_value=FakeModel(exc))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("silent.mp4"))
    assert e.value.reason == reason


class FakeHub(BaseHTTPRequestHandler):
    """Stands in for huggingface.co with the failure responses a model download can get."""

    mode = "500"
    responses = {
        "500": (500, "application/json", b'{"error":"Internal Error"}'),
        "503": (503, "text/html", b"<html><body>503 Service Unavailable</body></html>"),
        "malformed_json": (200, "application/json", b'{"id": "Systran/faster-whisper-tiny", "siblings": [ '),
        "wrong_content_type": (200, "text/html", b"<!doctype html><html>Sign in to continue</html>"),
        "slow": (200, "application/json", b"{}"),
    }

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.mode == "slow":
            time.sleep(3)
        code, ctype, body = self.responses[self.mode]
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_HEAD = do_GET


@pytest.mark.parametrize("mode", ["500", "503", "malformed_json", "wrong_content_type", "slow", "refused"])
def test_model_download_failures_are_service_errors(mode, tmp_path, monkeypatch):
    from huggingface_hub import constants

    if mode == "refused":
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
    else:
        handler = type("Hub", (FakeHub,), {"mode": mode})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(constants, "ENDPOINT", f"http://127.0.0.1:{port}")
    try:
        with pytest.raises(EnricherServiceError) as e:
            fwe._load_model({**fwe.DEFAULTS, "model": "tiny", "download_root": str(tmp_path)})
        assert e.value.reason == "model_unavailable"
        assert not isinstance(e.value, EnricherInputError)
    finally:
        if mode != "refused":
            server.shutdown()


def test_offline_without_cached_model(tmp_path):
    with pytest.raises(EnricherServiceError) as e:
        fwe._load_model({**fwe.DEFAULTS, "model": "tiny", "download_root": str(tmp_path), "local_files_only": True})
    assert e.value.reason == "model_unavailable"


def test_invalid_model_name(tmp_path):
    with pytest.raises(EnricherServiceError) as e:
        fwe._load_model({**fwe.DEFAULTS, "model": "no-such-size", "download_root": str(tmp_path)})
    assert e.value.reason == "model_unavailable"


# --- real model: hallucination risk ---


@pytest.mark.parametrize(
    "name, reason",
    [
        ("silent.mp4", "no_speech_detected"),
        ("music.mp4", "no_speech_detected"),
        ("half_second.mp4", "no_speech_detected"),
    ],
)
def test_no_text_on_silence_or_music(fx, model_opts, name, reason):
    r = analyze(fx(name), **model_opts)
    assert_empty(r, reason)
    assert r["speech_duration"] == 0.0 and r["duration"] > 0


def test_english_speech(fx, model_opts):
    r = analyze(needs(fx("en.wav")), **model_opts)
    assert_serialisable(r)
    assert r["speech"] is True and r["reason"] == "speech"
    assert r["language"] == "en" and r["language_probability"] > 0.8
    assert "spring collection" in r["text"].lower()
    assert 0 < r["speech_duration"] <= r["duration"]
    seg = r["segments"][0]
    assert set(seg) == {"start", "end", "text", "avg_logprob", "no_speech_prob"}
    assert 0 <= seg["start"] < seg["end"] <= r["duration"] + 0.5


@pytest.mark.parametrize("name, language, words", [("fr.wav", "fr", "collection"), ("ar.wav", "ar", "مرحبا")])
def test_non_english_speech(fx, model_opts, name, language, words):
    r = analyze(needs(fx(name)), **model_opts)
    assert_serialisable(r)
    assert r["language"] == language and r["language_probability"] > 0.8
    assert r["language_candidates"][0]["language"] == language
    assert words in r["text"].lower()


def test_forced_language(fx, model_opts):
    r = analyze(needs(fx("fr.wav")), language="fr", **model_opts)
    assert r["language"] == "fr" and r["language_probability"] == 1


def test_portrait_and_rotated_video(fx, model_opts):
    portrait = analyze(needs(fx("portrait_fr.mp4")), **model_opts)
    rotated = analyze(needs(fx("rotated_fr.mp4")), **model_opts)
    assert portrait["language"] == rotated["language"] == "fr"
    assert portrait["text"] == rotated["text"]


def test_audio_only_file(fx, model_opts):
    r = analyze(needs(fx("en_audio_only.m4a")), **model_opts)
    assert r["speech"] is True and r["language"] == "en"


def test_variable_frame_rate_video(fx, model_opts):
    r = analyze(needs(fx("vfr.mp4")), **model_opts)
    assert r["speech"] is True and "spring collection" in r["text"].lower()


def test_long_clip_timestamps_after_three_minutes(fx, model_opts):
    r = analyze(needs(fx("long.m4a")), **model_opts)
    assert r["duration"] == pytest.approx(185, abs=0.5)
    assert r["speech"] is True and r["speech_duration"] < 15
    assert r["segments"][0]["start"] >= 179


def test_word_timestamps(fx, model_opts):
    r = analyze(needs(fx("en.wav")), word_timestamps=True, **model_opts)
    words = [w for s in r["segments"] for w in s["words"]]
    assert len(words) >= 8
    assert all(w["start"] <= w["end"] for w in words)
    assert [w["start"] for w in words] == sorted(w["start"] for w in words)
    assert_serialisable(r)


# --- real stored ads (set AD_SAMPLES_DIR; media is not committed) ---


def real_samples(pattern):
    if not AD_SAMPLES_DIR:
        return []
    return sorted(glob.glob(os.path.join(AD_SAMPLES_DIR, "**", pattern), recursive=True))


@pytest.mark.skipif(not real_samples("*.mp4"), reason="set AD_SAMPLES_DIR to a folder of real ad videos")
@pytest.mark.parametrize("path", real_samples("*.mp4"), ids=os.path.basename)
def test_real_ad_videos(path, model_opts):
    r = analyze(path, **model_opts)
    assert_serialisable(r)
    assert r["duration"] > 0
    if r["speech"]:
        assert r["text"] and r["language"] and r["segments"]
    else:
        assert r["text"] == "" and r["reason"] in ("no_speech_detected", "no_transcribable_speech")


@pytest.mark.skipif(not real_samples("*.jpg"), reason="set AD_SAMPLES_DIR to a folder with real ad images")
@pytest.mark.parametrize("path", real_samples("*.jpg"), ids=os.path.basename)
def test_real_ad_images_have_no_audio(path):
    assert_empty(analyze(path), "no_audio_stream")


# --- Enricher wrapper ---


@pytest.fixture
def enricher(setup_module):
    return setup_module("faster_whisper_enricher", {"model": "tiny", "word_timestamps": True})


def test_enricher_config_reaches_analyze(enricher, fx, mocker):
    spy = mocker.patch.object(fwe, "analyze", return_value={"tool": "faster-whisper", "speech": False})
    md = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
    md.add_media(Media(filename=fx("silent.mp4")))
    md.add_media(Media(filename=fx("still.jpg")))
    enricher.enrich(md)
    spy.assert_called_once()
    assert spy.call_args.kwargs["model"] == "tiny" and spy.call_args.kwargs["word_timestamps"] is True
    assert md.media[0].get("transcript") == {"tool": "faster-whisper", "speech": False}
    assert md.media[1].get("transcript") is None


def test_enricher_records_input_errors_and_continues(enricher, fx, mocker):
    mocker.patch.object(fwe, "_load_model", return_value=FakeModel(AssertionError("must not transcribe")))
    md = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=2")
    md.add_media(Media(filename=fx("cut_faststart.mp4")))
    md.add_media(Media(filename=fx("no_audio.mp4")))
    enricher.enrich(md)
    assert md.media[0].get("transcript") == {"tool": "faster-whisper", "error": "truncated_media"}
    assert md.media[1].get("transcript")["reason"] == "no_audio_stream"


def test_enricher_propagates_service_errors(enricher, fx, mocker):
    mocker.patch.object(fwe, "_load_model", return_value=FakeModel(RuntimeError("std::bad_alloc")))
    md = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=3")
    md.add_media(Media(filename=fx("silent.mp4")))
    with pytest.raises(EnricherServiceError):
        enricher.enrich(md)
