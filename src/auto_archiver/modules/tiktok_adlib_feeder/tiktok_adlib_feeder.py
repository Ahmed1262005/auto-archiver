import time
from datetime import UTC, date, datetime, timedelta

import requests

from auto_archiver.core import Feeder, Metadata
from auto_archiver.core.consts import SetupError
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.provider_errors import ProviderError

TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
AD_QUERY_URL = "https://open.tiktokapis.com/v2/research/adlib/ad/query/"
AD_FIELDS = (
    "ad.id,ad.first_shown_date,ad.last_shown_date,ad.status,ad.videos,ad.image_urls,ad.reach,"
    "advertiser.business_id,advertiser.business_name,advertiser.paid_for_by"
)
AD_PAGE_URL = "https://library.tiktok.com/ads/detail/?ad_id={}"


def query_body(business_id: int, start: date, end: date, country_code: str, max_count: int, search_id=None) -> dict:
    """Request body for one page. The next page is addressed by the previous page's search_id alone:
    the live API (2026-11) returns no cursor, and sending one as well returns a different result set."""
    body = {
        "filters": {
            "advertiser_business_ids": [int(business_id)],
            "ad_published_date_range": {"min": start.strftime("%Y%m%d"), "max": end.strftime("%Y%m%d")},
            "country_code": country_code,
        },
        "max_count": max_count,
    }
    if search_id:
        body["search_id"] = search_id
    return body


def parse_page(payload, status: int = 200) -> tuple[list, str | None]:
    """(ads, next search_id or None) from one ad/query response; raises ProviderError on error envelopes."""
    if not isinstance(payload, dict):
        raise ProviderError("TikTok ad query: response is not a JSON object", status=status, schema=True)
    error = payload.get("error") or {}
    if not isinstance(error, dict):
        error = {"code": str(error), "message": payload.get("error_description")}
    if status != 200 or error.get("code") not in (None, "ok"):
        raise ProviderError(
            f"TikTok ad query failed: HTTP {status} {error.get('code')} {error.get('message')}",
            status=status,
            code=error.get("code"),
        )
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("ads", []), list):
        raise ProviderError("TikTok ad query: data.ads missing or not a list", status=status, schema=True)
    if not data.get("has_more"):
        return data.get("ads") or [], None
    if not data.get("search_id"):
        raise ProviderError("TikTok ad query: has_more without search_id", status=status, schema=True)
    return data.get("ads") or [], data["search_id"]


class TikTokAdLibFeeder(Feeder):
    def setup(self) -> None:
        if not self.advertiser_business_ids:
            raise SetupError("tiktok_adlib_feeder needs at least one advertiser_business_id")
        if not self.access_token and not (self.client_key and self.client_secret):
            raise SetupError("tiktok_adlib_feeder needs access_token or client_key and client_secret")

    def _token(self) -> str:
        if not self.access_token:
            r = requests.post(
                TOKEN_URL,
                data={
                    "client_key": self.client_key,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            self.access_token = r.json()["access_token"]
        return self.access_token

    def _query(self, body: dict) -> tuple[list, str | None]:
        r = requests.post(
            AD_QUERY_URL,
            params={"fields": AD_FIELDS},
            json=body,
            headers={"Authorization": f"Bearer {self._token()}"},
            timeout=self.timeout,
        )
        try:
            payload = r.json() if r.content else {}
        except ValueError:
            raise ProviderError(f"TikTok ad query: HTTP {r.status_code} body is not JSON", status=r.status_code)
        return parse_page(payload, r.status_code)

    def _ads(self, business_id: int, start: date, end: date):
        search_id = None
        for _ in range(self.max_pages):
            body = query_body(business_id, start, end, self.country_code, self.max_count, search_id)
            ads, search_id = self._query(body)
            yield from ads
            if not search_id:
                return
            time.sleep(0.5)
        logger.warning(f"tiktok_adlib_feeder hit max_pages for advertiser {business_id}")

    @staticmethod
    def to_metadata(record: dict) -> Metadata:
        ad, advertiser = record.get("ad") or {}, record.get("advertiser") or {}
        media_urls = [v["url"] for v in ad.get("videos") or [] if v.get("url")]
        media_urls += [u for u in ad.get("image_urls") or [] if u]
        m = Metadata().set_url(AD_PAGE_URL.format(ad["id"]))
        m.set("source", "tiktok_adlib")
        m.set("source_ad_id", str(ad["id"]))
        m.set("advertiser_id", str(advertiser.get("business_id")))
        m.set("advertiser_name", advertiser.get("business_name"))
        m.set("first_shown", ad.get("first_shown_date"))
        m.set("last_shown", ad.get("last_shown_date"))
        m.set("media_urls", media_urls)
        m.set("raw", record)
        m.set_title(f"{advertiser.get('business_name')} TikTok ad {ad['id']}")
        return m

    def __iter__(self) -> Metadata:
        # the API rejects windows that end today
        end = datetime.now(UTC).date() - timedelta(days=1)
        start = end - timedelta(days=self.days_back)
        for business_id in self.advertiser_business_ids:
            count = 0
            for record in self._ads(business_id, start, end):
                count += 1
                yield self.to_metadata(record)
            logger.info(f"tiktok_adlib_feeder: {count} ads for advertiser {business_id} ({start}..{end})")
