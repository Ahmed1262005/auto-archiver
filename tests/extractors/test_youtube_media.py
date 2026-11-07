import os

import pytest
import yt_dlp

from auto_archiver.modules.direct_media_extractor.direct_media_extractor import (
    EMPTY,
    TOO_LARGE,
    TRANSIENT,
    MediaFetchError,
)
from auto_archiver.modules.youtube_feeder import youtube_media as ym

URL = "https://www.youtube.com/watch?v=q2-Blj7xvgw"


class FakeYDL:
    def __init__(self, info=None, error=None, write=b"\x00\x00\x00\x18ftypmp42"):
        self.info, self.error, self.write = info or {"id": "q2-Blj7xvgw"}, error, write

    def __call__(self, opts):
        self.opts = opts
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download):
        assert download is False
        if self.error:
            raise yt_dlp.utils.DownloadError(self.error)
        return dict(self.info)

    def process_ie_result(self, info, download):
        path = os.path.join(os.path.dirname(self.opts["outtmpl"]["default"]), f"{info['id']}.mp4")
        if self.write is not None:
            with open(path, "wb") as f:
                f.write(self.write)
        return info | {"requested_downloads": [{"filepath": path}], "format_id": "398+140"}


def test_downloads_into_out_dir(tmp_path):
    fake = FakeYDL()
    m = ym.fetch_media(URL, str(tmp_path), ydl_factory=fake)
    assert m.filename == str(tmp_path / "q2-Blj7xvgw.mp4")
    assert m.get("source_content_type") == "video/mp4" and m.get("source_url") == URL
    assert fake.opts["format"] == ym.FORMAT and fake.opts["merge_output_format"] == "mp4"
    assert fake.opts["noplaylist"] is True and fake.opts["ignoreerrors"] is False


@pytest.mark.parametrize(
    "message,reason",
    [
        ("ERROR: [youtube] xxxxxxxxxxx: This video is unavailable", ym.UNAVAILABLE),  # recorded 2026-11-06
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access to this video", ym.UNAVAILABLE),
        ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot. Use --cookies-from-browser", ym.BLOCKED),
        ("ERROR: unable to download video data: HTTP Error 429: Too Many Requests", ym.BLOCKED),
        ("ERROR: unable to download video data: HTTP Error 403: Forbidden", ym.BLOCKED),  # recorded 2026-11-07
        ("ERROR: unable to download video data: HTTP Error 503: Service Unavailable", TRANSIENT),
        ("ERROR: [youtube] abc: Read timed out.", TRANSIENT),
        ("ERROR: [youtube] abc: Requested format is not available", ym.UNAVAILABLE),
        ("ERROR: [youtube] abc: nsig extraction failed: Some formats may be missing", ym.FAILED),
    ],
)
def test_errors_are_typed(tmp_path, message, reason):
    with pytest.raises(MediaFetchError) as e:
        ym.fetch_media(URL, str(tmp_path), ydl_factory=FakeYDL(error=message))
    assert e.value.reason == reason


@pytest.mark.parametrize("status", ["is_live", "is_upcoming"])
def test_live_is_unavailable(tmp_path, status):
    with pytest.raises(MediaFetchError) as e:
        ym.fetch_media(URL, str(tmp_path), ydl_factory=FakeYDL(info={"id": "x", "live_status": status}))
    assert e.value.reason == ym.UNAVAILABLE


def test_too_large_is_refused_before_download(tmp_path):
    info = {"id": "x", "requested_formats": [{"filesize": 600}, {"filesize_approx": 500}]}
    with pytest.raises(MediaFetchError) as e:
        ym.fetch_media(URL, str(tmp_path), max_bytes=1000, ydl_factory=FakeYDL(info=info))
    assert e.value.reason == TOO_LARGE and not os.listdir(tmp_path)


def test_no_file_is_empty(tmp_path):
    with pytest.raises(MediaFetchError) as e:
        ym.fetch_media(URL, str(tmp_path), ydl_factory=FakeYDL(write=None))
    assert e.value.reason == EMPTY
