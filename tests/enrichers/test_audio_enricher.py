"""audio_enricher tests: real ffmpeg/fpcalc runs on generated fixtures (and real ads when available),
plus mocked tool failures.

Real stored ads are not committed. Point AUDIO_ENRICHER_SAMPLES at a folder of real ad files (*.mp4, *.jpg)
to run the real-sample tests; they are skipped otherwise.
"""

import glob
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from auto_archiver.core import Media, Metadata
from auto_archiver.modules.audio_enricher import audio_enricher as ae
from auto_archiver.modules.audio_enricher import analyze, fingerprint_similarity
from auto_archiver.utils.enricher_errors import EnricherInputError, EnricherServiceError

TOOLS = all(shutil.which(b) for b in ("ffmpeg", "ffprobe", "fpcalc"))
needs_tools = pytest.mark.skipif(not TOOLS, reason="ffmpeg, ffprobe and fpcalc (libchromaprint-tools) are required")

# Pseudo-random two-voice melody: a note every 0.25 s / 0.5 s picked by a sin hash, so each SEED is a different "track".
MELODY = (
    "aevalsrc=exprs='0.3*sin(2*PI*t*220*pow(2\\,floor(12*mod(abs(sin(floor(t*4)*12.9898+{seed}))*43758.5453\\,1))/12))"
    "+0.2*sin(2*PI*t*330*pow(2\\,floor(12*mod(abs(sin(floor(t*2)*78.233+{seed}))*43758.5453\\,1))/12))':s=44100:d={d}"
)
SAMPLES = os.environ.get("AUDIO_ENRICHER_SAMPLES", "")
REAL_VIDEOS = sorted(glob.glob(os.path.join(SAMPLES, "*.mp4"))) if SAMPLES else []
REAL_IMAGES = sorted(glob.glob(os.path.join(SAMPLES, "*.jpg"))) if SAMPLES else []
needs_real = pytest.mark.skipif(not REAL_VIDEOS, reason="set AUDIO_ENRICHER_SAMPLES to a folder of real ad files")
REAL_RUN = subprocess.run  # captured before any test patches it


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    if not TOOLS:
        pytest.skip("tools missing")
    d = tmp_path_factory.mktemp("audio_fx")
    p = lambda name: str(d / name)  # noqa: E731
    gray = "color=c=gray:s=320x568:r=25:d=30"  # portrait 9:16
    ffmpeg("-f", "lavfi", "-i", MELODY.format(seed=1, d=30), "-f", "lavfi", "-i", gray, "-shortest",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", p("ad.mp4"))  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", MELODY.format(seed=7, d=30), "-c:a", "aac", "-b:a", "128k", p("other.m4a"))
    ffmpeg("-i", p("ad.mp4"), "-vn", "-c:a", "libmp3lame", "-b:a", "64k", p("ad.mp3"))
    ffmpeg("-i", p("ad.mp4"), "-vn", "-c:a", "libopus", "-b:a", "32k", p("ad.opus"))
    ffmpeg("-ss", "11.3", "-t", "6", "-i", p("ad.mp4"), "-c:v", "libx264", "-c:a", "aac", "-b:a", "96k", p("cut6.mp4"))
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=25",
           "-t", "5", "-c:v", "libx264", "-c:a", "aac", p("silent.mp4"))  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25", "-t", "3", "-c:v", "libx264", p("noaudio.mp4"))
    # mono 1 kHz sine at -20 dBFS peak: BS.1770 gives -20 - 3.01 = -23.0 LUFS
    ffmpeg("-f", "lavfi", "-i", "aevalsrc='0.1*sin(2*PI*1000*t)':s=48000:d=10", "-c:a", "aac", "-b:a", "128k",
           p("quiet.m4a"))  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "aevalsrc='clip(1.6*sin(2*PI*997*t)\\,-1\\,1)':s=48000:d=5", "-c:a", "flac",
           p("loud.flac"))  # fmt: skip
    ffmpeg("-i", p("ad.mp4"), "-c", "copy", "-movflags", "+faststart", p("ad_faststart.mp4"))
    with open(p("ad.mp4"), "rb") as f:
        open(p("truncated.mp4"), "wb").write(f.read(60_000))  # moov at the end: cut off before it
    with open(p("ad_faststart.mp4"), "rb") as f:
        open(p("truncated_faststart.mp4"), "wb").write(f.read(200_000))  # moov up front, media cut off
    open(p("html.mp4"), "w").write(
        "<!doctype html><html><head><title>Access Denied</title></head><body>403</body></html>"
    )
    open(p("empty.mp4"), "wb").close()
    ffmpeg("-ss", "3", "-t", "0.5", "-i", p("ad.mp4"), "-c:v", "libx264", "-c:a", "aac", p("short.mp4"))
    ffmpeg("-f", "lavfi", "-i", MELODY.format(seed=3, d=200), "-f", "lavfi", "-i", "color=c=red:s=160x120:r=10",
           "-t", "200", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", p("long.mp4"))  # fmt: skip
    ffmpeg("-i", p("ad.mp4"), "-c", "copy", "-metadata:s:v:0", "rotate=90", p("rotated.mp4"))
    ffmpeg("-i", p("ad.mp4"), "-vf", "setpts='if(lt(N,100),PTS,PTS*1.7)'", "-vsync", "vfr", "-c:v", "libx264",
           "-c:a", "copy", p("vfr.mp4"))  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x240:r=25:d=1", "-f", "lavfi", "-i",
           "color=c=white:s=320x240:r=25:d=3", "-f", "lavfi", "-i", MELODY.format(seed=1, d=4), "-filter_complex",
           "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-t", "4", "-c:v", "libx264",
           "-c:a", "aac", p("black_start.mp4"))  # fmt: skip
    ffmpeg("-f", "lavfi", "-i", "color=c=green:s=64x64", "-frames:v", "1", p("still.jpg"))
    return p


# ---------- real stored ads ----------


@needs_tools
@needs_real
@pytest.mark.parametrize("path", REAL_VIDEOS, ids=os.path.basename)
def test_real_ad_video(path):
    r = analyze(path)
    json.dumps(r)
    assert r["tool"] == "ffmpeg+chromaprint" and set(r["tool_version"]) == {"ffmpeg", "fpcalc"}
    assert r["has_audio"] and r["has_video_stream"] and r["empty_reason"] is None
    assert -40 < r["loudness"]["integrated_lufs"] < 0
    assert r["loudness"]["true_peak_dbtp"] < 6
    assert 0 <= r["silence"]["silent_fraction"] <= 1 and 0 <= r["black"]["black_fraction"] <= 1
    fp = r["fingerprint"]
    assert fp["status"] == "ok" and fp["items"] > 0
    assert fp["duration_s"] == pytest.approx(r["duration_s"], abs=0.5)
    assert fingerprint_similarity(fp, fp)["score"] == 1.0


@needs_tools
@pytest.mark.skipif(not REAL_IMAGES, reason="set AUDIO_ENRICHER_SAMPLES to a folder of real ad files")
@pytest.mark.parametrize("path", REAL_IMAGES, ids=os.path.basename)
def test_real_ad_image_is_empty_not_error(path):
    r = analyze(path)
    assert r["is_still_image"] and not r["has_audio"] and r["empty_reason"] == "still_image"
    assert r["fingerprint"]["raw"] == [] and r["black"] is None and r["loudness"] is None


@needs_tools
@pytest.mark.skipif(len(REAL_VIDEOS) < 2, reason="needs two or more real ads")
def test_real_ads_with_different_audio_do_not_match():
    fps = [analyze(p)["fingerprint"] for p in REAL_VIDEOS]
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if REAL_VIDEOS[i].rsplit("_", 1)[-1] == REAL_VIDEOS[j].rsplit("_", 1)[-1]:
                continue  # same file stored twice
            assert not fingerprint_similarity(fps[i], fps[j])["match"], (REAL_VIDEOS[i], REAL_VIDEOS[j])


@needs_tools
@needs_real
def test_real_ad_reencode_and_cutdown_match(tmp_path):
    src = max(REAL_VIDEOS, key=lambda p: analyze(p)["duration_s"])
    full = analyze(src)
    ffmpeg("-i", src, "-vn", "-c:a", "libopus", "-b:a", "24k", str(tmp_path / "re.opus"))
    start = min(37.3, full["duration_s"] / 2)
    ffmpeg("-ss", str(start), "-t", "6", "-i", src, "-c:v", "libx264", "-c:a", "aac", str(tmp_path / "cut.mp4"))
    re = fingerprint_similarity(full, analyze(str(tmp_path / "re.opus")))
    assert re["match"] and re["score"] > 0.95 and re["offset_seconds"] == pytest.approx(0, abs=0.13)
    cut = fingerprint_similarity(full, analyze(str(tmp_path / "cut.mp4")))
    assert cut["match"] and cut["offset_seconds"] == pytest.approx(start, abs=0.25)


# ---------- generated fixtures: matching ----------


@needs_tools
@pytest.mark.parametrize("name", ["ad.mp3", "ad.opus", "rotated.mp4", "vfr.mp4", "ad_faststart.mp4"])
def test_same_audio_reencoded_matches_strongly(fx, name):
    s = fingerprint_similarity(analyze(fx("ad.mp4")), analyze(fx(name)))
    assert s["match"] and s["score"] >= 0.95 and s["overlap_score"] >= 0.95
    assert s["offset_items"] == 0 and s["matched_fraction"] >= 0.95


@needs_tools
def test_trimmed_6s_excerpt_matches_at_offset(fx):
    full, cut = analyze(fx("ad.mp4")), analyze(fx("cut6.mp4"))
    s = fingerprint_similarity(full, cut)
    assert s["match"] and s["score"] >= 0.9
    assert s["offset_seconds"] == pytest.approx(11.3, abs=0.25)
    # argument order flips the sign of the offset
    r = fingerprint_similarity(cut, full)
    assert r["match"] and r["offset_seconds"] == pytest.approx(-11.3, abs=0.25)


@needs_tools
@pytest.mark.parametrize("name", ["other.m4a", "long.mp4"])
def test_different_audio_does_not_match(fx, name):
    s = fingerprint_similarity(analyze(fx("ad.mp4")), analyze(fx(name)))
    assert not s["match"] and s["score"] < 0.75 and s["matched_fraction"] < 0.3


@needs_tools
@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng generates the voice-over fixtures")
def test_language_versions_sharing_a_soundtrack_match(fx, tmp_path):
    lines = {
        "en": "Introducing the new collection. Designed in Florence, made for the city. Available now.",
        "fr": "Découvrez la nouvelle collection. Conçue à Florence, faite pour la ville. Disponible maintenant.",
        "ar": "اكتشفوا المجموعة الجديدة. صممت في فلورنسا من أجل المدينة. متوفرة الآن.",
    }
    fps = {}
    for lang, text in lines.items():
        wav, out = str(tmp_path / f"{lang}.wav"), str(tmp_path / f"{lang}.m4a")
        subprocess.run(["espeak-ng", "-v", lang, "-w", wav, text], check=True, capture_output=True)
        ffmpeg("-i", fx("ad.mp4"), "-i", wav, "-filter_complex",
               "[0:a]volume=0.5[m];[1:a]aresample=44100,apad[s];[m][s]amix=inputs=2:duration=first:normalize=0",
               "-c:a", "aac", "-b:a", "96k", out)  # fmt: skip
        fps[lang] = analyze(out)
    assert fingerprint_similarity(fps["en"], fps["fr"])["match"]
    assert fingerprint_similarity(fps["en"], fps["ar"])["match"]
    assert not fingerprint_similarity(fps["fr"], analyze(fx("other.m4a")))["match"]


# ---------- generated fixtures: analysis results ----------


@needs_tools
def test_portrait_ad_full_result(fx):
    r = analyze(fx("ad.mp4"))
    json.dumps(r)
    assert r["has_audio"] and r["has_video_stream"] and r["has_audio_stream"]
    assert r["audio_stream"] == {"codec": "aac", "sample_rate": 44100, "channels": 1}
    assert r["silence"]["segments"] == [] and r["black"]["segments"] == []
    assert r["fingerprint"]["algorithm"] == 2 and r["fingerprint"]["duration_s"] == pytest.approx(30, abs=0.1)
    assert all(0 <= x <= 0xFFFFFFFF for x in r["fingerprint"]["raw"])


@needs_tools
def test_quiet_tone_lufs(fx):
    r = analyze(fx("quiet.m4a"))
    assert r["loudness"]["integrated_lufs"] == pytest.approx(-23.0, abs=1.0)
    assert r["loudness"]["true_peak_dbtp"] == pytest.approx(-20.0, abs=1.0)
    assert r["black"] is None and not r["has_video_stream"]


@needs_tools
def test_loud_clipped_tone_true_peak_near_zero(fx):
    r = analyze(fx("loud.flac"))
    assert r["loudness"]["true_peak_dbtp"] == pytest.approx(0.0, abs=1.0)
    assert r["loudness"]["integrated_lufs"] > -3


@needs_tools
def test_silent_audio_is_empty_not_error(fx):
    r = analyze(fx("silent.mp4"))
    assert r["has_audio_stream"] and not r["has_audio"] and r["empty_reason"] == "silent_audio"
    assert r["loudness"]["integrated_lufs"] is None and r["silence"]["silent_fraction"] == 1.0
    assert r["fingerprint"] == {**r["fingerprint"], "raw": [], "items": 0, "status": "silent_audio"}
    assert r["black"]["black_fraction"] > 0.95
    assert fingerprint_similarity(r, analyze(fx("ad.mp4")))["reason"] == "empty_fingerprint"


@needs_tools
def test_no_audio_stream_is_empty_not_error(fx):
    r = analyze(fx("noaudio.mp4"))
    assert not r["has_audio_stream"] and not r["has_audio"] and r["empty_reason"] == "no_audio_stream"
    assert r["loudness"] is None and r["silence"] is None and r["fingerprint"]["status"] == "no_audio_stream"
    assert r["black"]["black_fraction"] == 0.0


@needs_tools
def test_still_image(fx):
    r = analyze(fx("still.jpg"))
    assert r["is_still_image"] and r["empty_reason"] == "still_image" and r["fingerprint"]["raw"] == []


@needs_tools
def test_half_second_clip_measures_but_cannot_fingerprint(fx):
    r = analyze(fx("short.mp4"))
    assert r["has_audio"] and r["loudness"]["integrated_lufs"] is not None
    assert r["fingerprint"]["status"] == "too_short" and r["empty_reason"] == "audio_too_short_to_fingerprint"


@needs_tools
def test_long_clip_fingerprints_the_whole_stream(fx):
    r = analyze(fx("long.mp4"))
    assert r["fingerprint"]["duration_s"] == pytest.approx(200, abs=0.5)
    assert r["fingerprint"]["items"] > 180 / ae.ITEM_DURATION_S
    capped = analyze(fx("long.mp4"), fingerprint_max_seconds=60)
    assert capped["fingerprint"]["items"] < 60 / ae.ITEM_DURATION_S + 1


@needs_tools
def test_black_segment_detected(fx):
    r = analyze(fx("black_start.mp4"))
    (seg,) = r["black"]["segments"]
    assert seg["start"] == 0.0 and seg["end"] == pytest.approx(1.0, abs=0.05)
    assert r["black"]["black_fraction"] == pytest.approx(0.25, abs=0.02)


@needs_tools
def test_silence_segment_detected(fx, tmp_path):
    out = str(tmp_path / "gap.m4a")
    ffmpeg("-f", "lavfi", "-i", MELODY.format(seed=2, d=6), "-filter_complex",
           "[0:a]volume=enable='between(t,2,4)':volume=0[a]", "-map", "[a]", "-c:a", "aac", out)  # fmt: skip
    r = analyze(out)
    (seg,) = r["silence"]["segments"]
    assert seg["start"] == pytest.approx(2.0, abs=0.1) and seg["end"] == pytest.approx(4.0, abs=0.1)
    assert r["silence"]["silent_fraction"] == pytest.approx(2 / 6, abs=0.03) and r["has_audio"]


# ---------- input errors (retry won't help) ----------


@needs_tools
@pytest.mark.parametrize(
    "name, reason",
    [
        ("empty.mp4", "empty_file"),
        ("html.mp4", "unreadable"),
        ("truncated.mp4", "unreadable"),
        ("truncated_faststart.mp4", "unreadable"),
        ("does-not-exist.mp4", "not_found"),
    ],
)
def test_bad_media_raises_input_error(fx, name, reason):
    with pytest.raises(EnricherInputError) as e:
        analyze(fx(name))
    assert e.value.reason == reason


def test_directory_raises_input_error(tmp_path):
    with pytest.raises(EnricherInputError) as e:
        analyze(str(tmp_path))
    assert e.value.reason == "not_a_file"


@needs_tools
def test_data_only_file_raises_no_stream(tmp_path):
    path = str(tmp_path / "subs.mkv")
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n")
    ffmpeg("-i", str(srt), "-c:s", "srt", path)
    with pytest.raises(EnricherInputError) as e:
        analyze(path)
    assert e.value.reason == "no_stream"


# ---------- tool failures (retry may help), mocked ----------


@pytest.fixture
def fake_tools(mocker):
    """Patch subprocess.run in the module: `behaviours[binary]` is an exception to raise or a CompletedProcess
    to return; other binaries run for real."""
    ae._version.cache_clear()
    behaviours = {}

    def run(cmd, *args, **kwargs):
        b = behaviours.get(os.path.basename(cmd[0]))
        if isinstance(b, BaseException):
            raise b
        if b is not None:
            return subprocess.CompletedProcess(cmd, *b) if isinstance(b, tuple) else b(cmd)
        return REAL_RUN(cmd, *args, **kwargs)

    mocker.patch.object(ae.subprocess, "run", side_effect=run)
    yield behaviours
    ae._version.cache_clear()


@needs_tools
@pytest.mark.parametrize(
    "binary, behaviour, reason",
    [
        ("ffmpeg", FileNotFoundError(2, "No such file or directory: 'ffmpeg'"), "tool_missing"),
        ("fpcalc", FileNotFoundError(2, "No such file or directory: 'fpcalc'"), "tool_missing"),
        ("ffprobe", subprocess.TimeoutExpired("ffprobe", 600), "timeout"),
        ("fpcalc", PermissionError(13, "Permission denied"), "tool_failed"),
        ("ffprobe", (-9, "", ""), "crashed"),
        ("ffprobe", (0, "{not json", ""), "malformed_output"),
        ("ffprobe", (0, '{"streams": []}', ""), "malformed_output"),
        ("ffprobe", (0, "<html><body>502 Bad Gateway</body></html>", ""), "malformed_output"),
        ("ffprobe", (1, "", "Cannot allocate memory"), "out_of_memory"),
        ("ffprobe", (1, "", "Segfault in libavcodec"), "tool_failed"),
    ],
)
def test_tool_failures_raise_service_error(fx, fake_tools, binary, behaviour, reason):
    fake_tools[binary] = behaviour
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("ad.mp4"))
    assert e.value.reason == reason


@needs_tools
def test_ffmpeg_timeout_and_oom_during_analysis(fx, fake_tools):
    def ffmpeg_behaviour(exc_or_result):
        def run(cmd):
            if "-version" in cmd:
                return REAL_RUN(cmd, capture_output=True, text=True)
            if isinstance(exc_or_result, BaseException):
                raise exc_or_result
            return subprocess.CompletedProcess(cmd, *exc_or_result)

        return run

    fake_tools["ffmpeg"] = ffmpeg_behaviour(subprocess.TimeoutExpired("ffmpeg", 600))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("ad.mp4"))
    assert e.value.reason == "timeout"
    fake_tools["ffmpeg"] = ffmpeg_behaviour((137, "", ""))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("ad.mp4"))
    assert e.value.reason == "crashed"
    fake_tools["ffmpeg"] = ffmpeg_behaviour((0, "", "nothing useful"))
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("ad.mp4"))
    assert e.value.reason == "malformed_output"


@needs_tools
@pytest.mark.parametrize(
    "stdout",
    ['{"duration": 30.0}', '{"duration": 30, "fingerprint": [1, -5]}', "garbage", '{"fingerprint": "AQAA"}'],
)
def test_fpcalc_malformed_output(fx, fake_tools, stdout):
    fake_tools["fpcalc"] = lambda cmd: (
        REAL_RUN(cmd, capture_output=True, text=True)
        if "-version" in cmd
        else subprocess.CompletedProcess(cmd, 0, stdout, "")
    )
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("ad.mp4"))
    assert e.value.reason == "malformed_output"


@needs_tools
def test_fpcalc_version_garbage(fx, fake_tools):
    fake_tools["fpcalc"] = (0, "", "")
    with pytest.raises(EnricherServiceError) as e:
        analyze(fx("ad.mp4"))
    assert e.value.reason == "tool_failed"


# ---------- fingerprint_similarity (pure) ----------


def random_fp(n, seed):
    return np.random.default_rng(seed).integers(0, 2**32, size=n, dtype=np.uint64).tolist()


def test_similarity_identical_and_shifted():
    a = random_fp(240, 1)
    assert fingerprint_similarity(a, a)["score"] == 1.0
    s = fingerprint_similarity(a, a[100:148])
    assert s["score"] == 1.0 and s["offset_items"] == 100 and s["matched_fraction"] == 1.0
    assert s["offset_seconds"] == pytest.approx(100 * ae.ITEM_DURATION_S, abs=1e-3)


def test_similarity_unrelated_random():
    s = fingerprint_similarity(random_fp(240, 1), random_fp(240, 2))
    assert not s["match"] and 0.4 < s["score"] < 0.7


def test_similarity_noisy_copy_matches():
    a = np.array(random_fp(240, 3), dtype=np.uint32)
    flips = np.random.default_rng(4).random((240, 32)) < 0.08  # 8% bit errors, about what a lossy re-encode costs
    noise = (flips * (1 << np.arange(32, dtype=np.uint64))).sum(1).astype(np.uint32)
    s = fingerprint_similarity(a.tolist(), (a ^ noise)[50:].tolist())
    assert s["match"] and s["offset_items"] == 50 and s["score"] > 0.85


def test_similarity_accepts_signed_and_result_dicts():
    a = random_fp(100, 5)
    signed = [x - 2**32 if x >= 2**31 else x for x in a]
    assert fingerprint_similarity(a, signed)["score"] == 1.0
    assert fingerprint_similarity({"fingerprint": {"raw": a}}, {"raw": a})["score"] == 1.0


@pytest.mark.parametrize("a, b", [([], [1, 2, 3]), ([1, 2], []), (None, [1]), ({"raw": []}, {"raw": []})])
def test_similarity_empty(a, b):
    s = fingerprint_similarity(a, b)
    assert s["reason"] == "empty_fingerprint" and s["score"] == 0.0 and not s["match"]


def test_similarity_short_fingerprints():
    a = random_fp(5, 6)
    s = fingerprint_similarity(a, a[1:3])
    assert s["score"] == 1.0 and s["offset_items"] == 1


# ---------- Enricher wrapper ----------


@needs_tools
def test_enricher_sets_audio_on_media(fx, setup_module):
    enricher = setup_module("audio_enricher", {})
    m = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
    m.add_media(Media(filename=fx("ad.mp4")))
    m.add_media(Media(filename=fx("still.jpg")))
    m.add_media(Media(filename=fx("html.mp4")))
    enricher.enrich(m)
    assert m.media[0].get("audio")["has_audio"] and m.media[0].get("audio")["fingerprint"]["status"] == "ok"
    assert m.media[1].get("audio") is None
    assert m.media[2].get("audio") is None
    assert m.media[2].get("audio_error") == {**m.media[2].get("audio_error"), "type": "EnricherInputError"}
