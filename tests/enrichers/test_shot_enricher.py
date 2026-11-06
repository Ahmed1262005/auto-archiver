"""shot_enricher on real ffmpeg-generated videos (colour/testsrc segments of known length), broken files,
mocked tool failures, and (when AA_REAL_AD_SAMPLES points at a folder of stored ads) real ad media."""

import glob
import json
import os
import shutil
import subprocess

import pytest
from PIL import Image

from auto_archiver.core import Media, Metadata
from auto_archiver.modules.shot_enricher import analyze
from auto_archiver.modules.shot_enricher import shot_enricher as se
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg and ffprobe"
)

FRAME_25 = 1 / 25
X264 = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]


def ff(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True, capture_output=True)


def concat(path, segments, size="160x120", rate=25, extra=()):
    """One video from lavfi sources [(source, seconds), ...] joined with hard cuts."""
    inputs = []
    for src, dur in segments:
        sep = ":" if "=" in src else "="
        inputs += ["-f", "lavfi", "-i", f"{src}{sep}size={size}:rate={rate}:duration={dur}"]
    n = len(segments)
    graph = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0"
    ff(*inputs, "-filter_complex", graph, *X264, *extra, path)
    return path


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    d = tmp_path_factory.mktemp("shot_fixtures")
    p = lambda name: str(d / name)  # noqa: E731
    files = {}
    files["cuts"] = concat(
        p("cuts.mp4"), [("color=c=red", 1), ("color=c=green", 1.52), ("color=c=blue", 0.8), ("testsrc", 2)]
    )
    files["static"] = concat(p("static.mp4"), [("color=c=gray", 4)])
    files["rapid"] = concat(
        p("rapid.mp4"), [(f"color=c={c}", 0.4) for c in ("red", "green", "blue", "yellow", "white")]
    )
    # fade to black (1 s), 1 s of black, fade in from black: hard edges at 3.0 s and 4.0 s
    ff("-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3", "-vf", "fade=t=out:st=2:d=1", *X264, p("a.mp4"))
    ff("-f", "lavfi", "-i", "color=c=black:size=320x240:rate=25:duration=1", *X264, p("b.mp4"))
    ff("-f", "lavfi", "-i", "smptebars=size=320x240:rate=25:duration=3", "-vf", "fade=t=in:st=0:d=1", *X264, p("c.mp4"))
    with open(p("list.txt"), "w") as f:
        f.write("file a.mp4\nfile b.mp4\nfile c.mp4\n")
    ff("-f", "concat", "-i", p("list.txt"), "-c", "copy", p("fade.mp4"))
    files["fade"] = p("fade.mp4")
    # variable frame rate: 2 s at 30 fps then 2 s at 10 fps
    concat(p("v30.mp4"), [("testsrc", 2)], rate=30)
    concat(p("v10.mp4"), [("smptebars", 2)], rate=10)
    ff(
        "-i", p("v30.mp4"), "-i", p("v10.mp4"),
        "-filter_complex",
        "[0:v]settb=1/90000,setpts=PTS-STARTPTS[a];[1:v]settb=1/90000,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1:a=0",
        "-vsync", "vfr", *X264, p("vfr.mp4"),
    )  # fmt: skip
    files["vfr"] = p("vfr.mp4")
    concat(p("land.mp4"), [("testsrc", 3), ("smptebars", 3)], size="320x180")
    ff("-i", p("land.mp4"), "-c", "copy", "-metadata:s:v:0", "rotate=90", p("rotated.mp4"))
    files["rotated"] = p("rotated.mp4")
    files["portrait"] = concat(p("portrait.mp4"), [("testsrc", 2), ("smptebars", 2)], size="180x320")
    files["short"] = concat(p("short.mp4"), [("testsrc", 0.5)])
    files["oneframe"] = concat(p("oneframe.mp4"), [("testsrc", 0.04)])
    files["long"] = concat(p("long.mp4"), [("testsrc", 200), ("smptebars", 1)], size="96x64", rate=10)
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", p("audio_only.mp4"))
    files["audio_only"] = p("audio_only.mp4")
    ff(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=4",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", *X264, "-c:a", "aac", "-t", "4", p("silent.mp4"),
    )  # fmt: skip
    files["silent"] = p("silent.mp4")
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=6", *X264, "-movflags", "+faststart", p("fs.mp4"))
    with open(p("fs.mp4"), "rb") as f:
        data = f.read()
    with open(p("truncated_tail.mp4"), "wb") as f:
        f.write(data[: len(data) // 3])  # moov is up front, the media data is cut off
    files["truncated_tail"] = p("truncated_tail.mp4")
    with open(p("silent.mp4"), "rb") as f:
        head = f.read(4000)
    with open(p("truncated_head.mp4"), "wb") as f:
        f.write(head)  # moov at the end was never written
    files["truncated_head"] = p("truncated_head.mp4")
    with open(p("html.mp4"), "w") as f:
        f.write("<!doctype html><html><body><h1>403 Forbidden</h1></body></html>")
    files["html"] = p("html.mp4")
    open(p("zero.mp4"), "wb").close()
    files["zero"] = p("zero.mp4")
    ff("-f", "lavfi", "-i", "testsrc=size=160x120:rate=1:duration=1", "-frames:v", "1", p("still.jpg"))
    files["still"] = p("still.jpg")
    return files


@pytest.fixture
def copy(tmp_path):
    """Copy a fixture into this test's folder, so keyframes written next to it don't leak between tests."""

    def _copy(src):
        dst = str(tmp_path / os.path.basename(src))
        shutil.copy(src, dst)
        return dst

    return _copy


def near(values, expected, tol):
    return len(values) == len(expected) and all(abs(a - b) <= tol for a, b in zip(values, expected))


# --- generated fixtures ------------------------------------------------------------------------------------------


def test_known_cuts_within_one_frame(fx, copy):
    r = analyze(copy(fx["cuts"]))
    assert near(r["cuts_s"], [1.0, 2.52, 3.32], FRAME_25), r["cuts_s"]
    assert r["cut_count"] == 3 and r["shot_count"] == 4
    assert [s["start_frame"] for s in r["shots"]] == [0, 25, 63, 83]
    assert [s["duration_frames"] for s in r["shots"]] == [25, 38, 20, 50]
    assert near([s["duration_s"] for s in r["shots"]], [1.0, 1.52, 0.8, 2.0], FRAME_25)
    assert r["shots"][-1]["end_s"] == pytest.approx(5.32, abs=FRAME_25)
    assert r["cuts_per_minute"] == pytest.approx(3 / (5.32 / 60), rel=0.01)
    assert r["mean_shot_s"] == pytest.approx(5.32 / 4, abs=0.01)
    assert r["median_shot_s"] == pytest.approx((1.0 + 1.52) / 2, abs=FRAME_25)
    assert r["empty_reason"] is None and not r["single_shot"] and not r["truncated"]
    json.dumps(r)


def test_result_carries_tool_and_version(fx, copy):
    import scenedetect

    r = analyze(copy(fx["cuts"]))
    assert r["tool"] == "PySceneDetect" and r["tool_version"] == scenedetect.__version__ == "0.7.1"
    assert "model" in r and r["model"] is None
    assert r["detector"] == "adaptive" and r["detector_params"]["adaptive_threshold"] == 3.0


def test_keyframe_is_the_middle_frame_of_each_shot_saved_next_to_the_input(fx, copy):
    path = copy(fx["cuts"])
    r = analyze(path)
    colours = []
    for shot in r["shots"]:
        kf = shot["keyframe"]
        assert os.path.dirname(kf["path"]) == os.path.dirname(os.path.abspath(path))
        assert kf["path"].endswith(f"cuts.shot-{shot['index'] + 1:03d}.jpg")
        assert kf["time_s"] == pytest.approx((shot["start_s"] + shot["end_s"]) / 2, abs=0.001)
        with Image.open(kf["path"]) as im:
            assert im.format == "JPEG" and im.size == (160, 120)
            colours.append(im.convert("RGB").resize((1, 1)).getpixel((0, 0)))
    red, green, blue = colours[:3]
    assert red[0] > 200 and red[1] < 50 and green[1] > 100 and green[0] < 50 and blue[2] > 200 and blue[0] < 50


def test_single_static_shot_has_zero_cuts_and_says_why(fx, copy):
    r = analyze(copy(fx["static"]))
    assert r["cut_count"] == 0 and r["shot_count"] == 1 and r["cuts_s"] == []
    assert r["single_shot"] and r["empty_reason"] == "no_cuts"
    assert r["cuts_per_minute"] == 0.0
    assert r["mean_shot_s"] == r["median_shot_s"] == pytest.approx(4.0, abs=FRAME_25)
    assert r["shots"][0]["keyframe"] is not None


def test_fade_to_black_is_reported_as_a_black_segment(fx, copy):
    r = analyze(copy(fx["fade"]))
    assert len(r["black_segments"]) == 1
    seg = r["black_segments"][0]
    # 1 s of pure black at 3.0-4.0 s, plus the darkest frames of the fades either side
    assert 2.6 <= seg["start_s"] <= 3.0 and 4.0 <= seg["end_s"] <= 4.4
    black = [s for s in r["shots"] if s["black"]]
    assert len(black) == 1 and black[0]["start_s"] == pytest.approx(3.0, abs=FRAME_25)
    assert r["black_detector"] == "ffmpeg blackdetect"


def test_black_detection_can_be_switched_off(fx, copy):
    r = analyze(copy(fx["fade"]), black_detection=False)
    assert r["black_segments"] == [] and r["black_detector"] is None and not any(s["black"] for s in r["shots"])


def test_rapid_cuts_under_half_a_second(fx, copy):
    r = analyze(copy(fx["rapid"]))
    assert near(r["cuts_s"], [0.4, 0.8, 1.2, 1.6], FRAME_25), r["cuts_s"]
    assert all(s["duration_frames"] == 10 for s in r["shots"])
    assert r["cuts_per_minute"] == pytest.approx(120, rel=0.01)


def test_min_scene_len_merges_shots_shorter_than_it(fx, copy):
    # PySceneDetect's own default (15 frames = 0.6 s here) would drop every other cut of a fast-cut ad
    r = analyze(copy(fx["rapid"]), min_scene_len=15)
    assert r["cut_count"] < 4


def test_content_detector_option(fx, copy):
    r = analyze(copy(fx["cuts"]), detector="content", content_threshold=27.0)
    assert r["detector"] == "content" and r["detector_params"] == {"threshold": 27.0, "min_scene_len": 0.25}
    assert near(r["cuts_s"], [1.0, 2.52, 3.32], FRAME_25)


def test_variable_frame_rate_times_come_from_timestamps(fx, copy):
    r = analyze(copy(fx["vfr"]))
    assert r["video"]["vfr"] and r["frames_approximate"]
    assert near(r["cuts_s"], [2.0], 1 / 30), r["cuts_s"]
    # OpenCV seeking is unreliable on VFR, so every keyframe comes from ffmpeg by timestamp
    assert [s["keyframe"]["extractor"] for s in r["shots"]] == ["ffmpeg", "ffmpeg"]
    with Image.open(r["shots"][1]["keyframe"]["path"]) as im:
        assert im.size == (160, 120)


def test_rotated_landscape_is_measured_and_keyframed_upright(fx, copy):
    r = analyze(copy(fx["rotated"]))
    assert r["video"]["rotation"] == 90
    assert (r["video"]["width"], r["video"]["height"]) == (180, 320)
    assert near(r["cuts_s"], [3.0], FRAME_25)
    for shot in r["shots"]:
        with Image.open(shot["keyframe"]["path"]) as im:
            assert im.size == (180, 320)


def test_portrait_9_16(fx, copy):
    r = analyze(copy(fx["portrait"]))
    assert (r["video"]["width"], r["video"]["height"]) == (180, 320) and r["video"]["rotation"] == 0
    assert near(r["cuts_s"], [2.0], FRAME_25)
    with Image.open(r["shots"][0]["keyframe"]["path"]) as im:
        assert im.size == (180, 320)


def test_half_second_clip_is_one_shot(fx, copy):
    r = analyze(copy(fx["short"]))
    assert r["shot_count"] == 1 and r["empty_reason"] == "no_cuts"
    assert r["decoded_duration_s"] == pytest.approx(0.5, abs=2 * FRAME_25)


def test_one_frame_video(fx, copy):
    r = analyze(copy(fx["oneframe"]))
    assert r["shot_count"] == 1 and r["shots"][0]["duration_frames"] == 1 and r["frames_decoded"] == 1


def test_clip_longer_than_three_minutes(fx, copy):
    r = analyze(copy(fx["long"]), keyframes=False)
    assert near(r["cuts_s"], [200.0], 0.1) and r["decoded_duration_s"] == pytest.approx(201.0, abs=0.1)
    assert all(s["keyframe"] is None for s in r["shots"])


@pytest.mark.parametrize("name", ["silent", "static"])
def test_silent_track_or_no_audio_stream_makes_no_difference(fx, copy, name):
    r = analyze(copy(fx[name]))
    assert r["shot_count"] == 1 and r["frames_decoded"] == 100


def test_cut_off_mp4_with_moov_up_front_is_analysed_and_flagged_truncated(fx, copy):
    r = analyze(copy(fx["truncated_tail"]))
    assert r["truncated"] and r["video"]["duration_s"] == pytest.approx(6.0, abs=0.1)
    assert 0 < r["decoded_duration_s"] < 6.0 * se.TRUNCATED_RATIO


@pytest.mark.parametrize(
    "name, reason",
    [
        ("audio_only", "no_video_stream"),
        ("truncated_head", "unreadable"),
        ("html", "unreadable"),
        ("zero", "empty_file"),
    ],
)
def test_bad_input_raises_input_error(fx, copy, name, reason):
    with pytest.raises(EnricherInputError) as e:
        analyze(copy(fx[name]))
    assert e.value.reason == reason


def test_missing_file(tmp_path):
    with pytest.raises(EnricherInputError) as e:
        analyze(str(tmp_path / "nope.mp4"))
    assert e.value.reason == "file_not_found"


def test_still_image_returns_an_empty_result(fx, copy):
    r = analyze(copy(fx["still"]))
    assert r["empty_reason"] == "still_image" and r["shots"] == [] and r["shot_count"] == 0
    json.dumps(r)


def test_bad_options(fx):
    with pytest.raises(ValueError):
        analyze(fx["cuts"], detector="hash")
    with pytest.raises(TypeError):
        analyze(fx["cuts"], threshold=3)


# --- tool failures (mocked) --------------------------------------------------------------------------------------


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize(
    "side_effect, error, reason",
    [
        (FileNotFoundError("ffprobe"), EnricherServiceError, "tool_missing"),
        (subprocess.TimeoutExpired("ffprobe", 60), EnricherServiceError, "timeout"),
        (completed(-9), EnricherServiceError, "tool_crashed"),
        (completed(0, stdout='{"streams": [{"codec_type": "vid'), EnricherServiceError, "tool_bad_output"),
        (completed(0, stdout="<html>502 Bad Gateway</html>"), EnricherServiceError, "tool_bad_output"),
        (completed(0, stdout='["not", "an", "object"]'), EnricherServiceError, "tool_bad_output"),
        (completed(1, stderr="Invalid data found when processing input"), EnricherInputError, "unreadable"),
        (completed(0, stdout='{"streams": [], "format": {}}'), EnricherInputError, "no_video_stream"),
    ],
)
def test_ffprobe_failures(fx, mocker, side_effect, error, reason):
    mocker.patch.object(se.subprocess, "run", side_effect=[side_effect] if isinstance(side_effect, Exception) else None)
    if not isinstance(side_effect, Exception):
        se.subprocess.run.return_value = side_effect
    with pytest.raises(error) as e:
        analyze(fx["cuts"])
    assert e.value.reason == reason


def test_cover_art_is_not_a_video_stream(fx, mocker):
    probe = {"streams": [{"codec_type": "audio"}, {"codec_type": "video", "disposition": {"attached_pic": 1}}]}
    mocker.patch.object(se.subprocess, "run", return_value=completed(0, stdout=json.dumps(probe)))
    with pytest.raises(EnricherInputError) as e:
        analyze(fx["cuts"])
    assert e.value.reason == "no_video_stream"


def test_detection_timeout_stops_the_scene_manager(fx, copy):
    with pytest.raises(EnricherServiceError) as e:
        analyze(copy(fx["long"]), timeout=0.01)
    assert e.value.reason == "timeout"


@pytest.mark.parametrize(
    "exc, reason",
    [(MemoryError(), "out_of_memory"), (__import__("cv2").error("decode failed"), "decoder_error")],
)
def test_detector_crash_is_a_service_error(fx, copy, mocker, exc, reason):
    mocker.patch("scenedetect.SceneManager.detect_scenes", side_effect=exc)
    with pytest.raises(EnricherServiceError) as e:
        analyze(copy(fx["cuts"]))
    assert e.value.reason == reason


def test_unopenable_by_decoder_is_an_input_error(fx, copy, mocker):
    from scenedetect import VideoOpenFailure

    mocker.patch("scenedetect.open_video", side_effect=VideoOpenFailure("no decoder"))
    with pytest.raises(EnricherInputError) as e:
        analyze(copy(fx["cuts"]))
    assert e.value.reason == "unreadable"


def test_blackdetect_timeout_is_a_service_error(fx, copy, mocker):
    real_run = subprocess.run

    def run(cmd, **kw):
        if "blackdetect" in " ".join(cmd):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return real_run(cmd, **kw)

    mocker.patch.object(se.subprocess, "run", side_effect=run)
    with pytest.raises(EnricherServiceError) as e:
        analyze(copy(fx["cuts"]))
    assert e.value.reason == "timeout"


def test_unwritable_keyframe_dir_is_a_service_error(fx, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx["cuts"], keyframe_dir=str(blocker / "sub"))
    assert e.value.reason == "keyframe_write_failed"


# --- Enricher wrapper -------------------------------------------------------------------------------------------


@pytest.fixture
def enricher(setup_module):
    return setup_module("shot_enricher", {})


def test_enrich_stores_shots_and_keyframe_media_that_pdq_can_hash(enricher, setup_module, fx, copy):
    video, image = Media(filename=copy(fx["cuts"])), Media(filename=copy(fx["still"]))
    item = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
    item.add_media(video)
    item.add_media(image)
    enricher.enrich(item)

    shots = item.media[0].get("shots")
    assert shots["cut_count"] == 3 and item.media[1].get("shots") is None
    keyframes = item.media[0].get("shot_keyframes")
    assert [k.get("id") for k in keyframes] == [f"shot_keyframe_{i}" for i in range(4)]
    assert all(k.is_image() and os.path.isfile(k.filename) for k in keyframes)

    setup_module("pdq_hash_enricher", {}).enrich(item)
    assert all(len(k.get("pdq_hash") or "") > 0 for k in keyframes)


def test_enrich_records_input_errors_and_keeps_going(enricher, fx, copy):
    bad, good = Media(filename=copy(fx["html"])), Media(filename=copy(fx["static"]))
    item = Metadata().set_url("https://example.com/ad")
    item.add_media(bad)
    item.add_media(good)
    enricher.enrich(item)
    assert item.media[0].get("shots") == {"tool": "PySceneDetect", "error": "unreadable"}
    assert item.media[1].get("shots")["shot_count"] == 1


def test_enrich_lets_service_errors_through(enricher, fx, copy, mocker):
    mocker.patch.object(se.subprocess, "run", side_effect=FileNotFoundError("ffprobe"))
    item = Metadata().set_url("https://example.com/ad")
    item.add_media(Media(filename=copy(fx["cuts"])))
    with pytest.raises(EnricherServiceError):
        enricher.enrich(item)


def test_enrich_passes_config_through(setup_module, fx, copy):
    e = setup_module("shot_enricher", {"detector": "content", "keyframes": False, "black_detection": False})
    item = Metadata().set_url("https://example.com/ad")
    item.add_media(Media(filename=copy(fx["cuts"])))
    e.enrich(item)
    assert item.media[0].get("shots")["detector"] == "content"
    assert item.media[0].get("shot_keyframes") == []


# --- real stored ads (set AA_REAL_AD_SAMPLES to a folder of downloaded ad files; not committed) --------------------

REAL_DIR = os.environ.get("AA_REAL_AD_SAMPLES", "")
REAL_VIDEOS = sorted(glob.glob(os.path.join(REAL_DIR, "*.mp4"))) if REAL_DIR else []
REAL_IMAGES = sorted(glob.glob(os.path.join(REAL_DIR, "*.jpg"))) if REAL_DIR else []


@pytest.mark.skipif(not REAL_VIDEOS, reason="set AA_REAL_AD_SAMPLES to a folder with real ad videos")
@pytest.mark.parametrize("src", REAL_VIDEOS, ids=os.path.basename)
def test_real_ad_video(src, copy):
    r = analyze(copy(src))
    json.dumps(r)
    assert r["shot_count"] >= 1 and not r["truncated"]
    assert sum(s["duration_s"] for s in r["shots"]) == pytest.approx(r["decoded_duration_s"], abs=0.05)
    assert r["decoded_duration_s"] == pytest.approx(r["video"]["duration_s"], abs=0.1)
    for prev, cur in zip(r["shots"], r["shots"][1:]):
        assert cur["start_s"] == prev["end_s"] and cur["start_frame"] == prev["end_frame"]
    for shot in r["shots"]:
        with Image.open(shot["keyframe"]["path"]) as im:
            assert im.size == (r["video"]["width"], r["video"]["height"])
    if os.path.basename(src).startswith("tiktok_video_6731f3cf68"):  # the attached 100 s Gucci ad
        assert r["shot_count"] == 48 and r["cuts_per_minute"] == pytest.approx(28.2, abs=0.1)


@pytest.mark.skipif(not REAL_IMAGES, reason="set AA_REAL_AD_SAMPLES to a folder with real ad images")
@pytest.mark.parametrize("src", REAL_IMAGES, ids=os.path.basename)
def test_real_ad_image_is_an_empty_result(src, copy):
    r = analyze(copy(src))
    assert r["empty_reason"] == "still_image" and r["shots"] == []
