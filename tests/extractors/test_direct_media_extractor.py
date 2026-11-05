import os
from typing import ClassVar

import pytest
import requests

from auto_archiver.core.metadata import Metadata
from tests.extractors.test_extractor_base import TestExtractorBase

SIGNED_URL = "https://cdn.example.com/v/abc.mp4?" + "x-signature=" + "a" * 500


class FakeResponse:
    def __init__(self, content_type="video/mp4", body=b"data", status=200):
        self.headers = {"Content-Type": content_type}
        self.body = body
        self.status = status
        self.closed = False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status}")

    def iter_content(self, _size):
        yield self.body

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
