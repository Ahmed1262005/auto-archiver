"""ocr_enricher: real RapidOCR runs on ffmpeg-generated fixtures and (when present) real stored TikTok ads.

Skipped when rapidocr is not installed (pip install -r src/auto_archiver/modules/ocr_enricher/requirements.txt).
Real ads are read from $OCR_REAL_ADS_DIR (*.mp4, *.jpg); they are never committed.
"""

import glob
import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("rapidocr")
if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)

from auto_archiver.core import Media, Metadata  # noqa: E402
from auto_archiver.modules.ocr_enricher import ocr_enricher  # noqa: E402
from auto_archiver.modules.ocr_enricher import analyze  # noqa: E402
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
REAL_ADS_DIR = os.environ.get("OCR_REAL_ADS_DIR", "")

TEXTS = {
    "en": "Shop the new collection",
    "fr": "Soldes d’été à Paris",
    "ar": "مرحبا بالعالم",
    "price": "Now only £19.99",
    "url": "www.gucci.com/uk",
}


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def drawtext(textfile, size=44, enable=None):
    dt = f"drawtext=fontfile={FONT}:textfile={textfile}:fontsize={size}:fontcolor=black:x=(w-tw)/2:y=(h-th)/2"
    return dt + (f":enable='{enable}'" if enable else "")


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    if not os.path.exists(FONT):
        pytest.skip("DejaVuSans font not installed")
    d = tmp_path_factory.mktemp("ocr_fx")
    p = {}
    for k, text in TEXTS.items():
        (d / f"{k}.txt").write_text(text, encoding="utf-8")
        p[f"{k}_txt"] = str(d / f"{k}.txt")
        p[f"{k}_png"] = str(d / f"{k}.png")
        ff(
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=720x1280:d=1",
            "-vf",
            drawtext(p[f"{k}_txt"], 56),
            "-frames:v",
            "1",
            p[f"{k}_png"],
        )
    p["blank_png"] = str(d / "blank.png")
    ff("-f", "lavfi", "-i", "color=c=white:s=720x1280:d=1", "-frames:v", "1", p["blank_png"])
    x264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    # 9:16 portrait, text on screen from 1 s to 3 s, silent audio track
    p["portrait"] = str(d / "portrait.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=576x1024:r=25:d=5",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-vf",
        drawtext(p["en_txt"], enable="between(t,1,3)"),
        *x264,
        "-c:a",
        "aac",
        "-shortest",
        p["portrait"],
    )
    p["no_audio"] = str(d / "no_audio.mp4")
    ff("-f", "lavfi", "-i", "color=c=white:s=576x1024:r=25:d=2", "-vf", drawtext(p["price_txt"]), *x264, p["no_audio"])
    p["no_text_video"] = str(d / "no_text.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=0x336699:s=576x1024:r=25:d=3",
        "-f",
        "lavfi",
        "-i",
        "sine=f=440:d=3",
        "-vf",
        "noise=alls=30:allf=t",
        *x264,
        "-c:a",
        "aac",
        "-shortest",
        p["no_text_video"],
    )
    p["audio_only"] = str(d / "audio_only.mp4")
    ff("-f", "lavfi", "-i", "sine=f=440:d=2", "-c:a", "aac", p["audio_only"])
    p["arabic_video"] = str(d / "arabic.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=576x1024:r=25:d=2",
        "-vf",
        drawtext(p["ar_txt"], 56),
        *x264,
        p["arabic_video"],
    )
    # stored landscape (text turned sideways), display matrix says rotate 90: only readable if honoured
    p["rotated"] = str(d / "rotated.mp4")
    sideways = str(d / "sideways.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=576x1024:r=25:d=2",
        "-vf",
        drawtext(p["url_txt"]) + ",transpose=clock",
        *x264,
        sideways,
    )
    ff("-i", sideways, "-c", "copy", "-metadata:s:v:0", "rotate=90", p["rotated"])
    p["sideways"] = sideways
    p["half_second"] = str(d / "half_second.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=576x1024:r=25:d=0.5",
        "-vf",
        drawtext(p["fr_txt"]),
        *x264,
        p["half_second"],
    )
    # 3 min 10 s, text only in the last 10 s; low resolution and 15 fps keep it small
    p["long"] = str(d / "long.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=320x568:r=15:d=190",
        "-vf",
        drawtext(p["en_txt"], 22, "gte(t,180)"),
        *x264,
        "-preset",
        "ultrafast",
        p["long"],
    )
    # variable frame rate: 25 fps for 2 s, then frames kept only every 10th
    p["vfr"] = str(d / "vfr.mp4")
    ff(
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=576x1024:r=25:d=4",
        "-vf",
        drawtext(p["en_txt"], enable="gte(t,2)") + ",select='lt(t\\,2)+not(mod(n\\,10))'",
        "-vsync",
        "vfr",
        *x264,
        p["vfr"],
    )
    full = open(p["portrait"], "rb").read()
    p["truncated"] = str(d / "truncated.mp4")  # moov atom at the end is cut off
    open(p["truncated"], "wb").write(full[: len(full) // 2])
    p["faststart"] = str(d / "faststart.mp4")
    ff("-i", p["portrait"], "-c", "copy", "-movflags", "+faststart", p["faststart"])
    fs = open(p["faststart"], "rb").read()
    p["truncated_faststart"] = str(d / "truncated_faststart.mp4")  # index intact, media data cut off
    open(p["truncated_faststart"], "wb").write(fs[: len(fs) // 2])
    p["html"] = str(d / "page.mp4")
    open(p["html"], "w").write(
        "<!doctype html><html><head><title>403 Forbidden</title></head><body>denied</body></html>\n"
    )
    p["empty"] = str(d / "empty.mp4")
    open(p["empty"], "wb").close()
    jpg = str(d / "en.jpg")
    ff("-i", p["en_png"], jpg)
    data = open(jpg, "rb").read()
    p["truncated_jpg"] = str(d / "truncated.jpg")
    open(p["truncated_jpg"], "wb").write(data[: len(data) // 3])
    p["garbage_jpg"] = str(d / "garbage.jpg")
    open(p["garbage_jpg"], "wb").write(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
    return p


def texts(result):
    return [e["text"] for e in result["timeline"]]


def plain(values):
    """Typographic apostrophes folded, so either reading of TEXTS["fr"] matches."""
    return [v.replace("’", "'") for v in values]


def assert_contract(r):
    json.dumps(r)
    for k in ("tool", "tool_version", "model", "frames", "timeline", "full_text", "has_text", "scripts"):
        assert k in r
    assert r["tool"] == "rapidocr" and r["tool_version"] and r["model"]
    for e in r["timeline"]:
        assert e["first_seen"] <= e["last_seen"] and 0 < e["max_confidence"] <= 1 and len(e["box"]) == 4


# ---------- recognised text ----------


@pytest.mark.parametrize("key", ["en", "fr", "price", "url"])
def test_latin_texts_recognised(fx, key):
    r = analyze(fx[f"{key}_png"])
    assert_contract(r)
    assert r["media_type"] == "image" and r["has_text"]
    assert plain(texts(r)) == plain([TEXTS[key]])
    assert r["script"] == "Latin"


def test_arabic_rtl_in_logical_order(fx):
    r = analyze(fx["ar_png"], languages=["latin", "arabic"])
    assert texts(r) == [TEXTS["ar"]]
    assert r["script"] == "Arabic" and r["languages_detected"] == ["arabic"]
    assert "+arabic_PP-OCRv5_rec_mobile" in r["model"]


def test_arabic_not_read_by_latin_only(fx):
    assert analyze(fx["ar_png"], languages=["latin"])["has_text"] is False


def test_price_and_url_kept_with_two_recognisers(fx):
    for key in ("price", "url"):
        assert texts(analyze(fx[f"{key}_png"], languages="latin,arabic")) == [TEXTS[key]]


def test_blank_frame_is_empty_not_error(fx):
    r = analyze(fx["blank_png"])
    assert_contract(r)
    assert r["has_text"] is False and r["empty_reason"] == "no_text_detected"
    assert r["timeline"] == [] and r["full_text"] == "" and r["frames"][0]["lines"] == []


def test_video_timeline_first_last_seen(fx):
    r = analyze(fx["portrait"])
    assert_contract(r)
    assert (r["media_type"], r["width"], r["height"], r["frame_count"]) == ("video", 576, 1024, 5)
    assert [f["timestamp"] for f in r["frames"]] == [0.0, 1.0, 2.0, 3.0, 4.0]
    [entry] = r["timeline"]
    assert entry["text"] == TEXTS["en"] and entry["frames"] == 3
    assert (entry["first_seen"], entry["last_seen"]) == (1.0, 3.0)
    assert r["full_text"] == TEXTS["en"]


def test_video_without_audio(fx):
    assert texts(analyze(fx["no_audio"])) == [TEXTS["price"]]


def test_video_without_text_is_empty(fx):
    r = analyze(fx["no_text_video"])
    assert r["frame_count"] == 3 and r["has_text"] is False and r["empty_reason"] == "no_text_detected"


def test_arabic_video(fx):
    assert texts(analyze(fx["arabic_video"], languages=["arabic"])) == [TEXTS["ar"]]


def test_rotated_video_is_read_upright(fx):
    r = analyze(fx["rotated"])
    assert (r["width"], r["height"]) == (576, 1024)
    assert texts(r) == [TEXTS["url"]]
    sideways = analyze(fx["sideways"])
    assert (sideways["width"], sideways["height"]) == (1024, 576)


def test_half_second_clip_gets_one_frame(fx):
    r = analyze(fx["half_second"])
    assert r["frame_count"] == 1 and plain(texts(r)) == plain([TEXTS["fr"]])


def test_long_clip_timestamps(fx):
    r = analyze(fx["long"], fps=0.1)
    assert r["duration"] > 180 and r["frame_count"] == 19
    [entry] = r["timeline"]
    assert entry["text"] == TEXTS["en"] and entry["first_seen"] == 180.0


def test_variable_frame_rate(fx):
    r = analyze(fx["vfr"])
    ts = [f["timestamp"] for f in r["frames"]]
    assert ts == sorted(ts) and ts[0] == 0.0 and abs(ts[-1] - 3.0) < 0.5
    assert r["timeline"][0]["text"] == TEXTS["en"] and r["timeline"][0]["first_seen"] >= 2.0


def test_max_frames(fx):
    assert analyze(fx["portrait"], max_frames=2)["frame_count"] == 2


def test_stills_passed_in(fx):
    r = analyze("ignored.mp4", stills=[(0.0, fx["blank_png"]), {"timestamp": 1.0, "path": fx["url_png"]}])
    assert r["media_type"] == "stills" and r["frame_count"] == 2
    assert r["timeline"][0]["first_seen"] == 1.0 and texts(r) == [TEXTS["url"]]


def test_truncated_faststart_returns_partial_or_input_error(fx):
    try:
        r = analyze(fx["truncated_faststart"])
    except EnricherInputError as e:
        assert e.reason in {"decode_failed", "no_decodable_frames", "unreadable_media"}
    else:
        assert r["frame_count"] >= 1


# ---------- bad input -> EnricherInputError ----------


@pytest.mark.parametrize(
    "key, reason",
    [
        ("empty", "empty_file"),
        ("html", "unreadable_media"),
        ("truncated", "unreadable_media"),
        ("audio_only", "no_video_stream"),
        ("truncated_jpg", "corrupt_image"),
        ("garbage_jpg", "decode_failed"),
    ],
)
def test_bad_media_raises_input_error(fx, key, reason):
    with pytest.raises(EnricherInputError) as e:
        analyze(fx[key])
    assert e.value.reason == reason


def test_missing_file():
    with pytest.raises(EnricherInputError) as e:
        analyze("/nonexistent/ad.mp4")
    assert e.value.reason == "not_found"


def test_bad_options(fx):
    with pytest.raises(TypeError):
        analyze(fx["blank_png"], language="latin")
    with pytest.raises(ValueError):
        analyze(fx["blank_png"], fps=0)
    with pytest.raises(ValueError, match="no model"):
        analyze(fx["blank_png"], languages=["klingon"])


# ---------- tool / service failures -> EnricherServiceError ----------


def test_ffmpeg_timeout(fx, mocker):
    mocker.patch.object(ocr_enricher.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffprobe", 1))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["portrait"])
    assert e.value.reason == "ffmpeg_timeout"


def test_ffmpeg_missing(fx, mocker):
    mocker.patch.object(ocr_enricher.subprocess, "run", side_effect=FileNotFoundError("ffprobe"))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["portrait"])
    assert e.value.reason == "ffmpeg_missing"


def test_ffprobe_malformed_json(fx, mocker):
    mocker.patch.object(
        ocr_enricher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b'{"streams": [', b"")
    )
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["portrait"])
    assert e.value.reason == "ffprobe_bad_output"


@pytest.mark.parametrize(
    "exc, reason", [(RuntimeError("Segmentation fault in predictor"), "ocr_failed"), (MemoryError(), "out_of_memory")]
)
def test_predict_crash(fx, mocker, exc, reason):
    ocr, models = ocr_enricher._pipeline("latin", ocr_enricher.DEFAULTS)
    key = next(k for k, v in ocr_enricher._PIPELINES.items() if v[0] is ocr)
    mocker.patch.dict(ocr_enricher._PIPELINES, {key: (mocker.Mock(side_effect=exc), models)})
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["en_png"])
    assert e.value.reason == reason


class _StubHoster(BaseHTTPRequestHandler):
    mode = "500"

    def do_GET(self):
        mode = type(self).mode
        if mode == "slow":
            threading.Event().wait(3)
            return
        status, ctype, body = {
            "500": (500, "text/plain", b"Internal Server Error"),
            "503": (503, "text/html", b"<html><body>503 Service Temporarily Unavailable</body></html>"),
            "json": (200, "application/json", b'{"error": "rate limited", "retry'),
            "html": (200, "text/html; charset=utf-8", b"<!doctype html><html><body>Sign in</body></html>"),
        }[mode]
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def stub_model_hoster(tmp_path, mocker):
    """Real RapidOCR download path against a local HTTP stub, with an empty model cache."""
    import requests
    from rapidocr.utils import download_file

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHoster)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mocker.patch.dict(ocr_enricher._PIPELINES, clear=True)
    real_get = requests.get

    def point_at(mode, port=None):
        _StubHoster.mode = mode
        base = f"http://127.0.0.1:{port or server.server_port}/models/"
        mocker.patch.object(
            download_file.requests,
            "get",
            side_effect=lambda url, **k: real_get(base + url.rsplit("/", 1)[-1], **{**k, "timeout": 1}),
        )
        return str(tmp_path / "models")

    yield point_at
    server.shutdown()


@pytest.mark.parametrize("mode", ["slow", "refused", "500", "503", "json", "html"])
def test_model_download_failures(fx, stub_model_hoster, mode):
    if mode == "refused":
        import socket

        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        model_dir = stub_model_hoster(mode, port)
    else:
        model_dir = stub_model_hoster(mode)
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["en_png"], languages=["korean"], model_dir=model_dir)
    assert e.value.reason == "model_unavailable"


def test_rapidocr_not_installed(fx, mocker):
    mocker.patch.dict(ocr_enricher._PIPELINES, clear=True)
    mocker.patch.dict("sys.modules", {"rapidocr": None})
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["en_png"])
    assert e.value.reason == "tool_missing"


# ---------- Enricher wrapper ----------


def test_enrich_stores_result_and_errors(fx, setup_module):
    enricher = setup_module("ocr_enricher", {"languages": ["latin"], "fps": 1.0})
    m = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
    m.add_media(Media(filename=fx["no_audio"]))
    m.add_media(Media(filename=fx["url_png"]))
    m.add_media(Media(filename=fx["html"], _mimetype="video/mp4"))
    m.add_media(Media(filename=fx["en_txt"], _mimetype="text/plain"))
    enricher.enrich(m)
    assert texts(m.media[0].get("ocr")) == [TEXTS["price"]]
    assert texts(m.media[1].get("ocr")) == [TEXTS["url"]]
    assert m.media[2].get("ocr") is None
    assert m.media[2].get("ocr_error") == {
        "type": "EnricherInputError",
        "reason": "unreadable_media",
        "message": m.media[2].get("ocr_error")["message"],
    }
    assert m.media[3].get("ocr") is None and m.media[3].get("ocr_error") is None


# ---------- real stored TikTok ads ----------


def _real(pattern):
    files = sorted(glob.glob(os.path.join(REAL_ADS_DIR, pattern))) if REAL_ADS_DIR else []
    if not files:
        pytest.skip("set OCR_REAL_ADS_DIR to a folder of real ad media")
    return files


def test_real_ad_images():
    for path in _real("*.jpg"):
        r = analyze(path)
        assert_contract(r)
        assert r["media_type"] == "image" and r["frame_count"] == 1
        if r["has_text"]:
            assert r["full_text"].strip()
        else:
            assert r["empty_reason"] == "no_text_detected"


def test_real_ad_videos():
    found = []
    for path in _real("*.mp4"):
        r = analyze(path, fps=0.5)
        assert_contract(r)
        assert r["media_type"] == "video" and r["frame_count"] >= int(r["duration"] * 0.5)
        found += texts(r)
    assert any("GUCCI" in t.upper() for t in found)
