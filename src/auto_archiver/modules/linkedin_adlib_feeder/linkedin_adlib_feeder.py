import time
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

import requests

from auto_archiver.core import Feeder, Metadata
from auto_archiver.core.consts import SetupError
from auto_archiver.utils.custom_logger import logger

AD_LIBRARY_URL = "https://api.linkedin.com/rest/adLibrary"


def restli_date(d: date) -> str:
    return f"(year:{d.year},month:{d.month},day:{d.day})"


class LinkedInAdLibFeeder(Feeder):
    def setup(self) -> None:
        if not self.advertisers:
            raise SetupError("linkedin_adlib_feeder needs at least one advertiser name")
        if not self.access_token:
            raise SetupError("linkedin_adlib_feeder needs an access_token")

    def _get(self, advertiser: str, start: date, end: date, offset: int) -> dict:
        # dateRange is Rest.li syntax and must not be percent-encoded, so the query string is built by hand
        url = (
            f"{AD_LIBRARY_URL}?q=criteria&advertiser={quote(advertiser)}"
            f"&dateRange=(start:{restli_date(start)},end:{restli_date(end)})&start={offset}&count={self.count}"
        )
        r = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "LinkedIn-Version": str(self.linkedin_version),
                "X-Restli-Protocol-Version": "2.0.0",
            },
            timeout=self.timeout,
        )
        if r.status_code != 200:
            raise RuntimeError(f"LinkedIn ad library query failed: HTTP {r.status_code} {r.text[:200]}")
        return r.json()

    def _ads(self, advertiser: str, start: date, end: date):
        offset = 0
        for page in range(self.max_pages):
            if page:
                time.sleep(self.pause_seconds)
            data = self._get(advertiser, start, end, offset)
            elements = data.get("elements") or []
            yield from elements
            offset += len(elements)
            if not elements or offset >= (data.get("paging") or {}).get("total", 0):
                return
        logger.warning(f"linkedin_adlib_feeder hit max_pages for advertiser {advertiser}")

    @staticmethod
    def to_metadata(record: dict, query: str) -> Metadata:
        details = record.get("details") or {}
        advertiser = details.get("advertiser") or {}
        stats = details.get("adStatistics") or {}
        ad_url = record["adUrl"]
        m = Metadata().set_url(ad_url)
        m.set("source", "linkedin_adlib")
        m.set("source_ad_id", ad_url.rstrip("/").rsplit("/", 1)[-1])
        m.set("advertiser_id", (advertiser.get("advertiserUrl") or "").rstrip("/").rsplit("/", 1)[-1] or None)
        m.set("advertiser_name", advertiser.get("advertiserName"))
        m.set("advertiser_query", query)
        m.set("ad_payer", advertiser.get("adPayer"))
        m.set("ad_type", details.get("type"))
        m.set("first_shown", stats.get("firstImpressionAt"))
        m.set("last_shown", stats.get("latestImpressionAt"))
        m.set("media_urls", [])
        m.set("raw", record)
        m.set_title(f"{advertiser.get('advertiserName')} LinkedIn ad {m.get('source_ad_id')}")
        return m

    def __iter__(self) -> Metadata:
        # the API returns nothing for windows that end today
        end = datetime.now(UTC).date() - timedelta(days=1)
        start = end - timedelta(days=self.days_back - 1)
        for i, advertiser in enumerate(self.advertisers):
            if i:
                time.sleep(self.pause_seconds)
            count = 0
            for record in self._ads(advertiser, start, end):
                count += 1
                yield self.to_metadata(record, advertiser)
            logger.info(f"linkedin_adlib_feeder: {count} ads for advertiser {advertiser!r} ({start}..{end})")
