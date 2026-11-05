import hashlib
import mimetypes
import os

import requests

from auto_archiver.core import Extractor, Media, Metadata
from auto_archiver.utils.custom_logger import logger


class DirectMediaExtractor(Extractor):
    def download(self, item: Metadata) -> Metadata | bool:
        url = item.get_url()
        try:
            r = requests.get(url, stream=True, timeout=self.timeout, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except requests.RequestException as e:
            logger.debug(f"direct media request failed: {e}")
            return False
        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not any(ctype.startswith(p) for p in self.allowed_mimetypes):
            r.close()
            return False
        ext = mimetypes.guess_extension(ctype) or ""
        filename = os.path.join(self.tmp_dir, hashlib.sha1(url.encode()).hexdigest()[:16] + ext)
        with open(filename, "wb") as f:
            f.writelines(r.iter_content(1 << 20))
        if os.path.getsize(filename) == 0:
            logger.warning("direct media download was empty")
            return False
        result = Metadata()
        result.add_media(Media(filename=filename, properties={"source_content_type": ctype}))
        return result.success("direct_media")
