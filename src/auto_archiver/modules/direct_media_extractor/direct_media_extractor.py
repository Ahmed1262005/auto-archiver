import hashlib
import mimetypes
import os

import requests

from auto_archiver.core import Extractor, Media, Metadata
from auto_archiver.utils.custom_logger import logger


class DirectMediaExtractor(Extractor):
    def download(self, item: Metadata) -> Metadata | bool:
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
        if len(result.media) < len(urls):
            logger.warning(f"direct media: fetched {len(result.media)} of {len(urls)} media urls")
        return result.success("direct_media")

    def _fetch(self, url: str) -> Media | None:
        try:
            r = requests.get(url, stream=True, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except requests.RequestException as e:
            logger.debug(f"direct media request failed: {e}")
            return None
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not any(ctype.startswith(p) for p in self.allowed_mimetypes):
            r.close()
            return None
        ext = mimetypes.guess_extension(ctype) or ""
        filename = os.path.join(self.tmp_dir, hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
        with open(filename, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
        if os.path.getsize(filename) == 0:
            logger.warning("direct media download was empty")
            return None
        return Media(filename=filename, properties={"source_content_type": ctype, "source_url": url})
