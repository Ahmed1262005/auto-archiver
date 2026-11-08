import os
from typing import ClassVar

import pytest
import requests

from auto_archiver.core.metadata import Metadata
from tests.extractors.test_extractor_base import TestExtractorBase

SIGNED_URL = "https://cdn.example.com/v/abc.mp4?" + "x-signature=" + "a" * 500


class FakeResponse:
    def __init__(self, content_type="video/mp4", body=b"data", status=200, length=None, broken=False):
        self.headers = {"Content-Type": content_type}
        if length is not None:
            self.headers["Content-Length"] = str(length)
        self.body = body
        self.status = self.status_code = status
        self.broken = broken
        self.closed = False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def iter_content(self, _size):
        yield self.body
        if self.broken:
            raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")

    def close(self):
        self.closed = True


class TestDirectMediaExtractor(TestExtractorBase):
    extractor_module = "direct_media_extractor"
    config: ClassVar[dict] = {"allowed_mimetypes": ["video/", "image/", "audio/"], "timeout": 5}

    @pytest.fixture(autouse=True)
    def tmp_dir(self, tmp_path):
        self.extractor.tmp_dir = str(tmp_path)

    def patch_get(self, mocker, response):
        return mocker.patch("requests.get", return_value=response)

    def test_downloads_media_with_short_filename(self, mocker):
        self.patch_get(mocker, FakeResponse())
        result = self.extractor.download(Metadata().set_url(SIGNED_URL))
        assert result.is_success()
        media = result.media[0]
        assert media.get("source_content_type") == "video/mp4"
        assert os.path.getsize(media.filename) == 4
        assert len(os.path.basename(media.filename)) < 40

    def test_non_media_falls_through(self, mocker):
        resp = FakeResponse(content_type="text/html; charset=utf-8")
        self.patch_get(mocker, resp)
        assert self.extractor.download(Metadata().set_url("https://example.com/page")) is False
        assert resp.closed

    def test_empty_body_fails(self, mocker):
        self.patch_get(mocker, FakeResponse(body=b""))
        assert self.extractor.download(Metadata().set_url(SIGNED_URL)) is False

    def test_http_error_fails(self, mocker):
        self.patch_get(mocker, FakeResponse(status=403))
        assert self.extractor.download(Metadata().set_url(SIGNED_URL)) is False

    def test_request_exception_fails(self, mocker):
        mocker.patch("requests.get", side_effect=requests.ConnectionError("boom"))
        assert self.extractor.download(Metadata().set_url(SIGNED_URL)) is False

    def test_downloads_all_media_urls(self, mocker):
        get = self.patch_get(mocker, FakeResponse())
        get.side_effect = [FakeResponse(), FakeResponse(content_type="image/jpeg")]
        item = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
        item.set("media_urls", ["https://cdn/a.mp4", "https://cdn/b.jpg"])
        result = self.extractor.download(item)
        assert result.is_success()
        assert [m.get("source_url") for m in result.media] == ["https://cdn/a.mp4", "https://cdn/b.jpg"]

    def test_partial_media_urls_still_succeed(self, mocker):
        mocker.patch("requests.get", side_effect=[FakeResponse(status=403), FakeResponse()])
        item = Metadata().set_url("https://library.tiktok.com/ads/detail/?ad_id=1")
        item.set("media_urls", ["https://cdn/a.mp4", "https://cdn/b.mp4"])
        result = self.extractor.download(item)
        assert len(result.media) == 1
        assert [(f["url"], f["reason"], f["status"]) for f in result.get("media_failures")] == [
            ("https://cdn/a.mp4", "expired", 403)
        ]

    def test_explicit_empty_media_urls_is_metadata_only(self, mocker):
        get = self.patch_get(mocker, FakeResponse())
        item = Metadata().set_url("https://www.linkedin.com/ad-library/detail/1").set("media_urls", [])
        result = self.extractor.download(item)
        assert result.is_success()
        assert result.status == "metadata_only: success"
        assert not result.media
        get.assert_not_called()

    @pytest.mark.parametrize(
        "response,reason",
        [
            (FakeResponse(status=403), "expired"),
            (FakeResponse(status=410), "expired"),
            (FakeResponse(status=404), "not_found"),
            (FakeResponse(status=429), "transient"),
            (FakeResponse(status=503), "transient"),
            (FakeResponse(status=400), "http_error"),
            (FakeResponse(content_type="text/html"), "not_media"),
            (FakeResponse(content_type="application/json", body=b'{"error":"expired"}'), "not_media"),
            (FakeResponse(content_type=""), "not_media"),
            (FakeResponse(body=b""), "empty"),
            (FakeResponse(body=b"da", length=4), "partial"),
            (FakeResponse(broken=True), "partial"),
            (FakeResponse(body=b"x" * 11), "too_large"),
            (FakeResponse(length=11), "too_large"),
        ],
    )
    def test_fetch_failure_reasons(self, mocker, response, reason, tmp_path):
        from auto_archiver.modules.direct_media_extractor.direct_media_extractor import MediaFetchError

        self.extractor.max_bytes = 10
        self.patch_get(mocker, response)
        with pytest.raises(MediaFetchError) as e:
            self.extractor.fetch(SIGNED_URL)
        assert e.value.reason == reason
        assert e.value.retryable == (reason in ("transient", "partial"))
        assert response.closed
        assert list(tmp_path.iterdir()) == []  # no half-written file is left behind

    @pytest.mark.parametrize("exc", [requests.ConnectionError("reset"), requests.Timeout("slow")])
    def test_network_errors_are_transient(self, mocker, exc):
        from auto_archiver.modules.direct_media_extractor.direct_media_extractor import MediaFetchError

        mocker.patch("requests.get", side_effect=exc)
        with pytest.raises(MediaFetchError) as e:
            self.extractor.fetch(SIGNED_URL)
        assert e.value.reason == "transient"

    def test_matching_content_length_succeeds(self, mocker):
        self.patch_get(mocker, FakeResponse(length=4))
        assert os.path.getsize(self.extractor.fetch(SIGNED_URL).filename) == 4
