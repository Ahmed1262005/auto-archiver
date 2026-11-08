"""Real ffmpeg on small synthetic videos (lavfi testsrc + sine), including broken and borderline files."""

import os
import shutil
import subprocess

import ffmpeg
import pytest

from auto_archiver.core import Media, Metadata
from auto_archiver.modules.video_transcode_enricher.video_transcode_enricher import summarize

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

FAST = {"preset": "ultrafast", "heights": [720, 480, 360]}


def make(path, w=640, h=360, dur=2.0, vcodec="libx264", audio=True, extra=(), pix="yuv420p", rate=25):
    args = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={w}x{h}:rate={rate}:duration={dur}",
    ]
    if audio:
        args += [
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={dur}",
            "-c:a",
            "aac" if path.endswith((".mp4", ".mov")) else "libopus",
        ]
    args += ["-c:v", vcodec, "-pix_fmt", pix, *extra, path]
    subprocess.run(args, check=True, capture_output=True)
    return path


@pytest.fixture
def enricher(setup_module):
    return setup_module("video_transcode_enricher", dict(FAST))


def run(enricher, path, **props):
    m = Media(filename=path)
    m.mimetype = "video/mp4"
    for k, v in props.items():
        m.set(k, v)
    item = Metadata().set_url("https://example.com/ad")
    item.add_media(m)
    enricher.enrich(item)
    return item.media[0]


def labels(m):
    return [(r.get("label"), r.get("method")) for r in m.get("renditions") or []]


def test_h264_aac_source_is_remuxed_at_its_size_and_encoded_below(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "a.mp4")))
    assert m.get("video_check")["ok"], m.get("video_check")
    assert labels(m) == [("360p", "remux")]  # 640x360: only its own rung, nothing to downscale to


def test_ladder_stops_at_the_source_size_and_never_upscales(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "a.mp4"), w=1280, h=720))
    assert labels(m) == [("720p", "remux"), ("480p", "encode"), ("360p", "encode")]
    for r in m.get("renditions"):
        info = summarize(ffmpeg.probe(r.filename))
        assert info["short_side"] == r.get("height") and info["video_codec"] == "h264" and info["has_audio"]
        assert r.get("bytes") == os.path.getsize(r.filename)
    sizes = [r.get("bytes") for r in m.get("renditions")]
    assert sizes == sorted(sizes, reverse=True)


def test_source_between_rungs_keeps_its_own_size(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "a.webm"), w=960, h=540, vcodec="libvpx-vp9", extra=("-b:v", "300k")))
    assert labels(m)[0] == ("540p", "encode")  # vp9/opus: no remux
    assert enricher.ladder(m.get("video_check")["probe"]) == [540, 480, 360]
    assert {lbl for lbl, _ in labels(m)[1:]} <= {"480p", "360p"}  # a rung that saves nothing is dropped


def test_portrait_rungs_use_the_short_side(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "p.mp4"), w=720, h=1280))
    top = m.get("renditions")[0]
    assert top.get("label") == "720p" and (top.get("display_width"), top.get("display_height")) == (720, 1280)
    low = summarize(ffmpeg.probe(m.get("renditions")[-1].filename))
    assert low["display_width"] == 360 and low["display_height"] == 640


def test_rotation_metadata_is_applied_not_remuxed(enricher, tmp_path):
    src = make(str(tmp_path / "r.mp4"), w=640, h=360)
    rot = str(tmp_path / "rot.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", src, "-c", "copy", "-metadata:s:v:0", "rotate=90", rot], check=True
    )
    m = run(enricher, rot)
    assert m.get("video_check")["probe"]["rotation"] == 90
    r = m.get("renditions")[0]
    assert r.get("method") == "encode"
    info = summarize(ffmpeg.probe(r.filename))
    assert (info["display_width"], info["display_height"], info["rotation"]) == (360, 640, 0)


def test_silent_video_stays_silent(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "s.mp4"), w=854, h=480, audio=False))
    assert m.get("video_check")["probe"]["has_audio"] is False
    assert labels(m)[0] == ("480p", "remux")
    assert all(not r.get("has_audio") for r in m.get("renditions"))


def test_10bit_h264_is_encoded_to_yuv420p(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "t.mp4"), w=640, h=360, pix="yuv422p"))
    assert labels(m) == [("360p", "encode")]
    assert summarize(ffmpeg.probe(m.get("renditions")[0].filename))["pix_fmt"] == "yuv420p"


@pytest.mark.parametrize(
    "kw,problem",
    [
        ({"dur": 0.4}, "too_short"),
        ({"w": 320, "h": 180}, "too_small"),
    ],
)
def test_ineligible_videos_get_reasons_and_no_renditions(enricher, tmp_path, kw, problem):
    m = run(enricher, make(str(tmp_path / "x.mp4"), **kw))
    check = m.get("video_check")
    assert not check["ok"] and any(p.startswith(problem) for p in check["problems"])
    assert m.get("renditions") is None and m.get("transcode_error") is None


def test_duration_that_differs_from_the_source_is_flagged(enricher, tmp_path):
    m = run(enricher, make(str(tmp_path / "x.mp4"), dur=2.0), source_duration=30)
    assert any(p.startswith("duration_mismatch") for p in m.get("video_check")["problems"])
    assert run(enricher, make(str(tmp_path / "y.mp4"), dur=2.0), source_duration=2.4).get("video_check")["ok"]


def test_not_a_video_and_cut_off_files_are_unreadable(enricher, tmp_path):
    html = tmp_path / "h.mp4"
    html.write_text("<html><body>Access denied</body></html>")
    assert run(enricher, str(html)).get("video_check")["problems"][0].startswith("unreadable")
    src = make(str(tmp_path / "full.mp4"), extra=("-movflags", "+faststart"))
    cut = tmp_path / "cut.mp4"
    data = open(src, "rb").read()  # noqa: SIM115
    cut.write_bytes(data[: len(data) // 2])
    check = run(enricher, str(cut)).get("video_check")
    assert not check["ok"], check
    assert any(p.startswith(("unreadable", "decode_errors", "duration")) for p in check["problems"])


def test_audio_only_file_has_no_video_stream(enricher, tmp_path):
    p = str(tmp_path / "a.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=2", "-c:a", "aac", p], check=True
    )
    assert "no_video_stream" in run(enricher, p).get("video_check")["problems"]


def test_corrupt_frames_fail_the_decode_pass(enricher, tmp_path):
    src = make(str(tmp_path / "c.ts"), w=640, h=360, dur=3, audio=False, extra=("-f", "mpegts"))
    data = bytearray(open(src, "rb").read())  # noqa: SIM115
    for off in range(len(data) // 3, 2 * len(data) // 3, 997):  # scribble over the middle third
        data[off : off + 40] = b"\xff" * 40
    bad = tmp_path / "bad.ts"
    bad.write_bytes(bytes(data))
    check = run(enricher, str(bad)).get("video_check")
    assert any(p.startswith("decode_errors") for p in check["problems"]), check


def test_decode_check_can_be_turned_off(setup_module, tmp_path):
    e = setup_module("video_transcode_enricher", dict(FAST, decode_check=False))
    assert e.check(make(str(tmp_path / "a.mp4")))["ok"]


def test_failed_read_back_reports_an_error_and_no_renditions(enricher, tmp_path, mocker):
    src = make(str(tmp_path / "a.mp4"), w=1280, h=720)
    real = summarize

    def lying(probe):
        out = real(probe)
        if probe["format"]["filename"].endswith("480p.mp4"):
            out["duration"] = 0.5
        return out

    mocker.patch("auto_archiver.modules.video_transcode_enricher.video_transcode_enricher.summarize", side_effect=lying)
    m = run(enricher, src)
    assert m.get("renditions") is None
    assert "480p failed its read-back check: duration" in m.get("transcode_error")


def test_ffmpeg_failure_is_reported_not_raised(enricher, tmp_path, mocker):
    src = make(str(tmp_path / "a.mp4"), w=1280, h=720)
    mocker.patch(
        "subprocess.run",
        side_effect=[
            subprocess.CompletedProcess([], 0, b"", b""),
            subprocess.CompletedProcess([], 1, b"", b"Conversion failed!"),
        ],
    )
    m = run(enricher, src)
    assert m.get("video_check")["ok"] and "ffmpeg 720p failed (1): Conversion failed!" in m.get("transcode_error")


def test_rung_not_smaller_than_the_one_above_is_dropped(enricher, tmp_path):
    # a tiny-bitrate 720p source: its remux is smaller than a crf-23 480p encode
    m = run(enricher, make(str(tmp_path / "a.mp4"), w=1280, h=720, extra=("-b:v", "40k")))
    sizes = [r.get("bytes") for r in m.get("renditions")]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == len(sizes)


def test_images_are_ignored(enricher, tmp_path):
    item = Metadata().set_url("https://example.com/ad")
    img = Media(filename=str(tmp_path / "x.jpg"))
    img.mimetype = "image/jpeg"
    item.add_media(img)
    enricher.enrich(item)
    assert img.get("video_check") is None
