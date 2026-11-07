"""object_detection_enricher: real RF-DETR-S / RT-DETRv2 / OWLv2 / SigLIP 2 runs on ffmpeg fixtures and (when present) real ad frames.

Skipped when torch/transformers are not installed (see object_detection_enricher/requirements.txt). Models download
from the Hugging Face Hub (RF-DETR: Roboflow's CDN) on first run. Real ad keyframes are read from $OD_REAL_FRAMES_DIR (*.jpg, and optional
brand reference logos in $OD_REAL_FRAMES_DIR/brand_refs/<brand>/*); they are never committed (this repo is public).
"""

import glob
import os
import shutil
import subprocess

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("rfdetr")
pytest.importorskip("rapidocr")
if not shutil.which("ffmpeg"):
    pytest.skip("ffmpeg not installed", allow_module_level=True)

from auto_archiver.core import Media, Metadata  # noqa: E402
from auto_archiver.modules.object_detection_enricher import analyze  # noqa: E402
from auto_archiver.modules.object_detection_enricher import object_detection_enricher as od  # noqa: E402
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REAL_DIR = os.environ.get("OD_REAL_FRAMES_DIR", "")
RTDETR = "PekingU/rtdetr_v2_r18vd"


def ff(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("od_fx")
    p = {k: str(d / f) for k, f in {"blank": "blank.png", "clip": "clip.mp4", "wordmark": "wordmark.png"}.items()}
    ff("-f", "lavfi", "-i", "color=c=white:s=540x960:d=1", "-frames:v", "1", p["blank"])
    ff("-f", "lavfi", "-i", "color=c=gray:s=320x568:r=25:d=2.5", "-c:v", "libx264", "-pix_fmt", "yuv420p", p["clip"])
    if os.path.exists(FONT):
        ff("-f", "lavfi", "-i", "color=c=black:s=540x960:d=1", "-vf",
           f"drawtext=fontfile={FONT}:text=SAMSUNG:fontsize=72:fontcolor=white:x=(w-tw)/2:y=(h-th)/2",
           "-frames:v", "1", p["wordmark"])  # fmt: skip
    for k, data in {"empty": b"", "html": b"<!doctype html><html><body>Sign in</body></html>"}.items():
        p[k] = str(d / f"{k}.mp4")
        with open(p[k], "wb") as f:
            f.write(data)
    p["audio_only"] = str(d / "audio.m4a")
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", p["audio_only"])
    return p


def assert_contract(r, layers=("objects", "logos")):
    assert r["tool"] == "transformers" and r["tool_version"] and r["torch_version"]
    assert r["frame_count"] == len(r["frames"]) and r["layers"] == list(layers)
    for layer in layers:
        info = r["models"][layer]
        assert info["id"] and info["architecture"]
    prev = -1.0
    for f in r["frames"]:
        assert f["timestamp"] >= prev
        prev = f["timestamp"]
        for layer in layers:
            for d in f[layer]:
                x0, y0, x1, y1 = d["box"]
                assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
                assert isinstance(d["label"], str) and d["label"]
                assert 0 <= d["score"] <= 1
            if layer == "logos":
                assert all({"brand", "brand_score", "brand_source", "text"} <= set(d) for d in f[layer])


def test_blank_frame_is_empty_not_error(fx):
    r = analyze(fx["blank"])
    assert_contract(r)
    assert r["media_type"] == "image" and r["frame_count"] == 1
    assert r["frames"][0]["objects"] == [] and not r["has_objects"]
    assert r["object_labels"] == {} and r["brands"] == []


def test_video_sampled_like_ocr(fx):
    r = analyze(fx["clip"], fps=1.0, layers=["objects"])
    assert_contract(r, ("objects",))
    assert r["media_type"] == "video" and r["sample_fps"] == 1.0
    assert [f["timestamp"] for f in r["frames"]] == [0.0, 1.0, 2.0]
    assert "logos" not in r["models"] and "logos" not in r["frames"][0]


def test_stills_and_max_frames(fx):
    r = analyze(None, stills=[(0.5, fx["blank"]), {"timestamp": 1.5, "path": fx["blank"]}], layers="objects")
    assert r["media_type"] == "stills" and [f["timestamp"] for f in r["frames"]] == [0.5, 1.5]
    assert analyze(fx["clip"], max_frames=1, layers=["objects"])["frame_count"] == 1


def test_wordmark_logo_named_by_ocr(fx):
    if not os.path.exists(fx["wordmark"]):
        pytest.skip("DejaVuSans-Bold font not installed")
    r = analyze(fx["wordmark"], layers=["logos"], brands=["Samsung", "Gucci"])
    assert_contract(r, ("logos",))
    logos = r["frames"][0]["logos"]
    assert logos and logos[0]["brand"] == "Samsung" and logos[0]["brand_source"] == "ocr"
    assert r["brands"][0]["brand"] == "Samsung" and r["brands"][0]["sources"] == ["ocr"]


@pytest.mark.parametrize(
    "text, brand",
    [("SAMSUNG", "Samsung"), ("GUCCI", "Gucci"), ("L'ORÉAL PARIS", "L'Oréal"), ("GARNiER", "Garnier"),
     ("SAMSUNC", "Samsung"), ("CeraVe", None), ("x", None), ("", None)],
)  # fmt: skip
def test_text_brand(text, brand):
    assert od._text_brand(text, ["Samsung", "Gucci", "L'Oréal", "Garnier"])[0] == brand


def test_nms_and_box_normalisation():
    a = {"box": [0.1, 0.1, 0.5, 0.5]}
    assert od._nms([a, {"box": [0.12, 0.1, 0.5, 0.52]}, {"box": [0.6, 0.6, 0.9, 0.9]}], 0.5) == [
        a,
        {"box": [0.6, 0.6, 0.9, 0.9]},
    ]
    assert od._norm_box([-5, 10, 700, 2000], 540, 960) == [0.0, 0.0104, 1.0, 1.0]


# ---------- bad media / options -> EnricherInputError / TypeError / ValueError ----------


@pytest.mark.parametrize(
    "key, reason", [("empty", "empty_file"), ("html", "unreadable_media"), ("audio_only", "no_video_stream")]
)
def test_bad_media_raises_input_error(fx, key, reason):
    with pytest.raises(EnricherInputError) as e:
        analyze(fx[key])
    assert e.value.reason == reason


def test_missing_file():
    with pytest.raises(EnricherInputError) as e:
        analyze("/nonexistent/ad.mp4")
    assert e.value.reason == "not_found"


def test_bad_options(fx, tmp_path):
    with pytest.raises(TypeError):
        analyze(fx["blank"], model="x")
    with pytest.raises(ValueError):
        analyze(fx["blank"], fps=0)
    with pytest.raises(ValueError, match="layers"):
        analyze(fx["blank"], layers=["faces"])
    with pytest.raises(ValueError, match="logo_prompts"):
        analyze(fx["blank"], logo_prompts=[])
    with pytest.raises(ValueError, match="brand_refs_dir"):
        analyze(fx["blank"], layers=["logos"], brand_refs_dir=str(tmp_path / "missing"))


# ---------- model / runtime failures -> EnricherServiceError ----------


def test_unknown_model_is_model_unavailable(fx):
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"], object_model="sapien-ad-index/no-such-detector")
    assert e.value.reason == "model_unavailable"


def test_offline_cache_miss_is_model_unavailable(fx, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"], object_model=RTDETR, model_dir=str(tmp_path / "empty_cache"))
    assert e.value.reason == "model_unavailable"


def test_wrong_head_is_model_unavailable(fx):
    # a real checkpoint that has no object-detection head
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"], object_model="google/siglip2-base-patch16-224")
    assert e.value.reason == "model_unavailable"


@pytest.mark.parametrize("module", ["torch", "transformers", "rfdetr"])
def test_runtime_not_installed(fx, mocker, module):
    mocker.patch.dict(od._MODELS, clear=True)
    mocker.patch.dict("sys.modules", {module: None})
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"])
    assert e.value.reason == "tool_missing"


@pytest.mark.parametrize(
    "exc, reason",
    [(RuntimeError("[enforce fail at alloc_cpu.cpp:118] DefaultCPUAllocator: can't allocate memory"), "out_of_memory"),
     (MemoryError(), "out_of_memory"), (RuntimeError("shape mismatch in decoder"), "detection_failed"),
     (ImportError("Owlv2ImageProcessor requires the scipy library"), "tool_missing")],
)  # fmt: skip
def test_inference_crash(fx, mocker, exc, reason):
    model, proc = od._load("closed", RTDETR, od.DEFAULTS)
    mocker.patch.object(type(model), "forward", side_effect=exc)
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"], object_model=RTDETR)
    assert e.value.reason == reason


def test_rfdetr_default_and_rtdetr_selectable(fx):
    r = analyze(fx["blank"], layers=["objects"])
    assert r["models"]["objects"]["id"] == "rf-detr-small" and r["models"]["objects"]["architecture"] == "rf-detr"
    assert r["models"]["objects"]["revision"].startswith("md5:") and r["rfdetr_version"] == "1.12.0"
    r = analyze(fx["blank"], layers=["objects"], object_model=RTDETR)
    assert r["models"]["objects"]["architecture"] == "rt_detr_v2" and r["rfdetr_version"] is None


def test_rfdetr_boxes_normalised_and_labelled(mocker):
    import numpy as np
    import supervision as sv
    from PIL import Image

    d = sv.Detections(
        xyxy=np.array([[-5.0, 10.0, 100.0, 300.0], [50.0, 50.0, 50.0, 80.0], [10.0, 20.0, 210.0, 420.0]]),
        confidence=np.array([0.6, 0.9, 0.95]),
        class_id=np.array([1, 77, 3]),
        data={"class_name": np.array(["person", "cell phone", "car"])},
    )
    det = {"kind": "rfdetr", "model": mocker.Mock(predict=mocker.Mock(return_value=d))}
    out = od._detect(od._torch(1), det, Image.new("RGB", (200, 400)), None, 0.5)
    assert out == [{"label": "car", "score": 0.95, "box": [0.05, 0.05, 1.0, 1.0]},
                   {"label": "person", "score": 0.6, "box": [0.0, 0.025, 0.5, 0.75]}]  # fmt: skip


@pytest.mark.parametrize(
    "exc, reason",
    [(OSError("Failed to establish a new connection"), "model_unavailable"),
     (ValueError("MD5 mismatch for rf-detr-small.pth"), "model_unavailable"),
     (MemoryError(), "out_of_memory")],
)  # fmt: skip
def test_rfdetr_load_failure(fx, mocker, exc, reason):
    import rfdetr

    mocker.patch.dict(od._MODELS, clear=True)
    mocker.patch.object(rfdetr, "RFDETRSmall", side_effect=exc)
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"])
    assert e.value.reason == reason


@pytest.mark.parametrize(
    "exc, reason",
    [(RuntimeError("DefaultCPUAllocator: can't allocate memory"), "out_of_memory"),
     (RuntimeError("shape mismatch in decoder"), "detection_failed")],
)  # fmt: skip
def test_rfdetr_inference_crash(fx, mocker, exc, reason):
    det = od._rfdetr("rf-detr-small", od.DEFAULTS)
    mocker.patch.object(det["model"], "predict", side_effect=exc)
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["blank"], layers=["objects"])
    assert e.value.reason == reason


def test_ffmpeg_timeout(fx, mocker):
    from auto_archiver.modules.ocr_enricher import ocr_enricher

    mocker.patch.object(ocr_enricher.subprocess, "run", side_effect=subprocess.TimeoutExpired("ffprobe", 1))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["clip"])
    assert e.value.reason == "ffmpeg_timeout"


# ---------- Enricher wrapper ----------


def test_enrich_stores_result_and_errors(fx, setup_module):
    enricher = setup_module("object_detection_enricher", {"layers": ["objects"], "fps": 1.0})
    m = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
    m.add_media(Media(filename=fx["clip"]))
    m.add_media(Media(filename=fx["html"], _mimetype="video/mp4"))
    m.add_media(Media(filename=fx["audio_only"], _mimetype="audio/mp4"))
    enricher.enrich(m)
    assert m.media[0].get("object_detection")["frame_count"] == 3
    assert m.media[1].get("object_detection") is None
    assert m.media[1].get("object_detection_error")["reason"] == "unreadable_media"
    assert m.media[2].get("object_detection") is None and m.media[2].get("object_detection_error") is None


# ---------- real ad keyframes ----------


def _real():
    files = sorted(glob.glob(os.path.join(REAL_DIR, "*.jpg"))) if REAL_DIR else []
    if not files:
        pytest.skip("set OD_REAL_FRAMES_DIR to a folder of real ad keyframes")
    return files


def test_real_ad_frames():
    refs = os.path.join(REAL_DIR, "brand_refs")
    opts = {"brand_refs_dir": refs} if os.path.isdir(refs) else {}
    found, logos = set(), 0
    for path in _real():
        r = analyze(path, **opts)
        assert_contract(r)
        found |= set(r["object_labels"])
        logos += sum(len(f["logos"]) for f in r["frames"])
    assert "person" in found  # every real ad sample we hold has people in it
    assert logos > 0


def test_wordmarks_name_or_add_logos():
    owl = {"label": "logo", "score": 0.3, "box": [0.1, 0.1, 0.5, 0.3], "brand": None, "brand_score": None,
           "brand_source": None, "text": None}  # fmt: skip
    inside = {"label": "wordmark", "score": 0.9, "box": [0.15, 0.15, 0.45, 0.25], "brand": "Samsung",
              "brand_score": 1.0, "brand_source": "ocr", "text": "SAMSUNG"}  # fmt: skip
    apart = dict(inside, box=[0.2, 0.8, 0.6, 0.9], brand="Gucci", text="GUCCI")
    out = od._merge_wordmarks([dict(owl)], [inside, apart])
    assert [(d["label"], d["brand"]) for d in out] == [("logo", "Samsung"), ("wordmark", "Gucci")]


def test_wordmarks_skip_captions_that_mention_the_brand(mocker):
    lines = [{"text": "Samsung completely touchscreen tablet", "confidence": 0.99, "box": [78, 496, 473, 527]},
             {"text": "SAMSUNG", "confidence": 0.95, "box": [200, 900, 380, 940]},
             {"text": "SAMSUND", "confidence": 0.8, "box": [200, 100, 380, 140]}]  # fmt: skip
    mocker.patch.object(od._ocr, "_ocr_frame", return_value=lines)
    found = od._wordmarks(None, od.np.zeros((1024, 576, 3), od.np.uint8), ["Samsung", "Gucci"])
    assert [(d["text"], d["brand"]) for d in found] == [("SAMSUNG", "Samsung"), ("SAMSUND", "Samsung")]
    assert found[0]["box"] == [round(200 / 576, 4), round(900 / 1024, 4), round(380 / 576, 4), round(940 / 1024, 4)]
