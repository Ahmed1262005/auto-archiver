import time
from datetime import UTC, date, datetime, timedelta
from urllib.parse import quote

import requests

from auto_archiver.core import Feeder, Metadata
from auto_archiver.core.consts import SetupError
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.provider_errors import ProviderError

AD_LIBRARY_URL = "https://api.linkedin.com/rest/adLibrary"


def restli_date(d: date) -> str:
    return f"(year:{d.year},month:{d.month},day:{d.day})"


def effective_window(start: date, end: date) -> tuple[date, date]:
    """The live API (2026-11) returns zero ads for one-day ranges, so a range is widened to at least two days."""
    return min(start, end - timedelta(days=1)), end


def request_url(advertiser: str, start: date, end: date, offset: int, count: int) -> str:
    # dateRange is Rest.li syntax and must not be percent-encoded, so the query string is built by hand
    start, end = effective_window(start, end)
    return (
        f"{AD_LIBRARY_URL}?q=criteria&advertiser={quote(advertiser)}"
        f"&dateRange=(start:{restli_date(start)},end:{restli_date(end)})&start={offset}&count={count}"
    )


def parse_page(payload, status: int = 200, offset: int = 0) -> tuple[list, int | None, int]:
    """(elements, next offset or None, paging.total) for one adLibrary response."""
    if status != 200:
        body = payload if isinstance(payload, dict) else {}
        raise ProviderError(
            f"LinkedIn ad library query failed: HTTP {status} {body.get('code')} {body.get('message')}",
            status=status,
            code=body.get("code") or (str(body["serviceErrorCode"]) if "serviceErrorCode" in body else None),
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise ProviderError("LinkedIn ad library: elements missing or not a list", status=status, schema=True)
    paging = payload.get("paging")
    if not isinstance(paging, dict) or not isinstance(paging.get("total"), int):
        raise ProviderError("LinkedIn ad library: paging.total missing", status=status, schema=True)
    elements = payload["elements"]
    nxt = offset + len(elements)
    return elements, (nxt if elements and nxt < paging["total"] else None), paging["total"]


def impressions_range(stats: dict) -> dict | None:
    """totalImpressions as {lower, upper}; the top band comes back as {from: 1000000, to: 0}, meaning 1M+."""
    total = stats.get("totalImpressions")
    if not isinstance(total, dict):
        return None
    lower, upper = total.get("from"), total.get("to")
    if upper == 0 and (lower or 0) > 0:
        upper = None
    return {"lower": lower, "upper": upper}


class LinkedInAdLibFeeder(Feeder):
    def setup(self) -> None:
        if not self.advertisers:
            raise SetupError("linkedin_adlib_feeder needs at least one advertiser name")
        if not self.access_token:
            raise SetupError("linkedin_adlib_feeder needs an access_token")

    def _get(self, advertiser: str, start: date, end: date, offset: int) -> tuple[list, int | None, int]:
        r = requests.get(
            request_url(advertiser, start, end, offset, self.count),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "LinkedIn-Version": str(self.linkedin_version),
                "X-Restli-Protocol-Version": "2.0.0",
            },
            timeout=self.timeout,
        )
        try:
            payload = r.json()
        except ValueError:
            raise ProviderError(
                f"LinkedIn ad library query: HTTP {r.status_code} body is not JSON", status=r.status_code
            )
        return parse_page(payload, r.status_code, offset)

    def _ads(self, advertiser: str, start: date, end: date):
        offset = 0
        for page in range(self.max_pages):
            if page:
                time.sleep(self.pause_seconds)
            elements, offset, _total = self._get(advertiser, start, end, offset)
            yield from elements
            if offset is None:
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
        m.set("impressions", impressions_range(stats))
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
