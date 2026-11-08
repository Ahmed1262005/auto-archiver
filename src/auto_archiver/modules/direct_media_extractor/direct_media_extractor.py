import hashlib
import mimetypes
import os

import requests

from auto_archiver.core import Extractor, Media, Metadata
from auto_archiver.utils.custom_logger import logger

# why a media URL produced no file; callers decide what is worth retrying
EXPIRED = "expired"  # 401/403/410: signed CDN links answer this once their signature lapses
NOT_FOUND = "not_found"
TRANSIENT = "transient"  # 408/429/5xx, connection errors, timeouts
HTTP_ERROR = "http_error"
NOT_MEDIA = "not_media"  # the URL served HTML/JSON/other non-media content
EMPTY = "empty"
TOO_LARGE = "too_large"
PARTIAL = "partial"  # the body ended before Content-Length, or the stream broke mid-file
RETRYABLE = frozenset({TRANSIENT, PARTIAL})


class MediaFetchError(Exception):
    def __init__(self, reason: str, message: str, status: int | None = None):
        super().__init__(f"{reason}: {message}")
        self.reason, self.status = reason, status

    @property
    def retryable(self) -> bool:
        return self.reason in RETRYABLE


def _status_reason(status: int) -> str:
    if status in (401, 403, 410):
        return EXPIRED
    if status == 404:
        return NOT_FOUND
    if status in (408, 425, 429) or status >= 500:
        return TRANSIENT
    return HTTP_ERROR


class DirectMediaExtractor(Extractor):
    def download(self, item: Metadata) -> Metadata | bool:
        self.failures: list[dict] = []
        urls = item.get("media_urls")
        if urls == []:
            # the feeder states the source exposes no media; the record itself is the archive
            return Metadata().success("metadata_only")
        urls = urls or [item.get_url()]
        result = Metadata()
        for url in urls:
            if media := self._fetch(url):
                result.add_media(media)
        if not result.media:
            return False
        if self.failures:
            logger.warning(f"direct media: fetched {len(result.media)} of {len(urls)} media urls")
            result.set("media_failures", self.failures)
        return result.success("direct_media")

    def _fetch(self, url: str) -> Media | None:
        try:
            return self.fetch(url)
        except MediaFetchError as e:
            logger.debug(f"direct media failed: {e}")
            self.failures.append({"url": url, "reason": e.reason, "status": e.status, "error": str(e)})
            return None

    def fetch(self, url: str) -> Media:
        """Download one media URL or raise MediaFetchError with the reason it failed."""
        try:
            r = requests.get(url, stream=True, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
        except (requests.ConnectionError, requests.Timeout) as e:
            raise MediaFetchError(TRANSIENT, str(e)) from e
        except requests.RequestException as e:
            raise MediaFetchError(HTTP_ERROR, str(e)) from e
        try:
            return self._save(url, r)
        finally:
            r.close()

    def _save(self, url: str, r) -> Media:
        if r.status_code >= 400:
            raise MediaFetchError(_status_reason(r.status_code), f"HTTP {r.status_code}", r.status_code)
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not any(ctype.startswith(p) for p in self.allowed_mimetypes):
            raise MediaFetchError(NOT_MEDIA, f"content-type {ctype or 'missing'!r}", r.status_code)
        expected = r.headers.get("Content-Length")
        expected = int(expected) if expected and expected.isdigit() else None
        if self.max_bytes and expected and expected > self.max_bytes:
            raise MediaFetchError(TOO_LARGE, f"{expected} bytes > {self.max_bytes}", r.status_code)
        ext = mimetypes.guess_extension(ctype) or ""
        filename = os.path.join(self.tmp_dir, hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
        size = 0
        try:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    size += len(chunk)
                    if self.max_bytes and size > self.max_bytes:
                        raise MediaFetchError(TOO_LARGE, f"more than {self.max_bytes} bytes", r.status_code)
                    f.write(chunk)
        except (requests.RequestException, OSError) as e:
            os.remove(filename)
            raise MediaFetchError(PARTIAL, f"stream broke after {size} bytes: {e}", r.status_code) from e
        except MediaFetchError:
            os.remove(filename)
            raise
        if size == 0:
            os.remove(filename)
            raise MediaFetchError(EMPTY, "empty body", r.status_code)
        if expected is not None and size != expected:
            os.remove(filename)
            raise MediaFetchError(PARTIAL, f"got {size} of {expected} bytes", r.status_code)
        return Media(filename=filename, properties={"source_content_type": ctype, "source_url": url})
