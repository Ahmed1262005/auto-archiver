import json
import re
import time
from datetime import UTC, date, datetime, timedelta

import requests

from auto_archiver.core import Feeder, Metadata
from auto_archiver.core.consts import SetupError
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.provider_errors import ProviderError

GRAPH_URL = "https://graph.facebook.com/{version}/ads_archive"
AD_PAGE_URL = "https://www.facebook.com/ads/library/?id={}"
MAX_PAGE_IDS = 10  # search_page_ids takes at most ten IDs per request
# every field documented for archived ads (graph-api/reference/archived-ad, v27.0); fields Meta only fills for
# UK/EU or political ads come back missing for other ads, which is not an error
_FIELDS = (
    "id",
    "ad_creation_time",
    "ad_creative_bodies",
    "ad_creative_link_captions",
    "ad_creative_link_descriptions",
    "ad_creative_link_titles",
    "ad_delivery_start_time",
    "ad_delivery_stop_time",
    "ad_snapshot_url",
    "age_country_gender_reach_breakdown",
    "beneficiary_payers",
    "br_total_reach",
    "bylines",
    "currency",
    "delivery_by_region",
    "demographic_distribution",
    "estimated_audience_size",
    "eu_total_reach",
    "impressions",
    "languages",
    "page_id",
    "page_name",
    "publisher_platforms",
    "spend",
    "target_ages",
    "target_gender",
    "target_locations",
    "total_reach_by_location",
)
AD_FIELDS = ",".join(_FIELDS)
MEDIA_TYPES = ("ALL", "IMAGE", "MEME", "VIDEO", "NONE")
_TOKEN = re.compile(r"access_token=[^&\"'\s\\]+")


def request_params(
    page_ids,
    countries,
    start: date,
    end: date,
    *,
    limit: int = 100,
    after: str | None = None,
    media_type: str = "ALL",
    active_status: str = "ALL",
    ad_type: str = "ALL",
) -> dict:
    """Query string for one /ads_archive page. Array parameters are JSON arrays, as in Meta's examples."""
    page_ids = [str(p) for p in page_ids]
    if not 1 <= len(page_ids) <= MAX_PAGE_IDS:
        raise ValueError(f"search_page_ids takes 1..{MAX_PAGE_IDS} ids, got {len(page_ids)}")
    if media_type not in MEDIA_TYPES:
        raise ValueError(f"media_type must be one of {MEDIA_TYPES}")
    p = {
        "search_page_ids": json.dumps(page_ids),
        "ad_reached_countries": json.dumps(list(countries)),
        "ad_delivery_date_min": start.isoformat(),
        "ad_delivery_date_max": end.isoformat(),
        "ad_active_status": active_status,
        "ad_type": ad_type,
        "media_type": media_type,
        "fields": AD_FIELDS,
        "limit": limit,
    }
    if after:
        p["after"] = after
    return p


def redact(text: str) -> str:
    """Drop access tokens Meta echoes back in ad_snapshot_url and paging.next before anything is stored."""
    return _TOKEN.sub("access_token=REDACTED", text)


def graph_error(payload) -> dict | None:
    err = payload.get("error") if isinstance(payload, dict) else None
    return err if isinstance(err, dict) else None


def parse_page(payload, status: int = 200) -> tuple[list, str | None]:
    """(ads, next `after` cursor or None) for one /ads_archive response.

    Ends on an empty `data` page or a page without `paging.next`, as the Graph API documents."""
    err = graph_error(payload)
    if status != 200 or err:
        err = err or {}
        raise ProviderError(
            f"Meta ads_archive query failed: HTTP {status} code {err.get('code')}/{err.get('error_subcode')} "
            f"{err.get('type')}: {err.get('message')}",
            status=status,
            code=None if err.get("code") is None else f"{err.get('code')}/{err.get('error_subcode') or ''}".rstrip("/"),
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ProviderError("Meta ads_archive: data missing or not a list", status=status, schema=True)
    data = payload["data"]
    for ad in data:
        if not isinstance(ad, dict) or not ad.get("id") or not ad.get("page_id"):
            raise ProviderError("Meta ads_archive: ad without id or page_id", status=status, schema=True)
    paging = payload.get("paging") or {}
    if not isinstance(paging, dict):
        raise ProviderError("Meta ads_archive: paging is not an object", status=status, schema=True)
    after = (paging.get("cursors") or {}).get("after")
    if not data or not paging.get("next"):
        return data, None
    if not after:
        raise ProviderError("Meta ads_archive: paging.next without cursors.after", status=status, schema=True)
    return data, after


def _day(v: str | None) -> str | None:
    return v[:10] if v else None


class MetaAdLibFeeder(Feeder):
    def setup(self) -> None:
        if not self.page_ids:
            raise SetupError("meta_adlib_feeder needs at least one Facebook Page ID")
        if not self.access_token:
            raise SetupError("meta_adlib_feeder needs an access_token")
        if self.media_type not in MEDIA_TYPES:
            raise SetupError(f"meta_adlib_feeder media_type must be one of {MEDIA_TYPES}")

    def _get(self, params: dict) -> tuple[list, str | None]:
        r = requests.get(
            GRAPH_URL.format(version=self.graph_version),
            params=params,
            headers={"Authorization": f"Bearer {self.access_token}"},  # never in the URL
            timeout=self.timeout,
        )
        try:
            payload = json.loads(redact(r.text))
        except ValueError:
            raise ProviderError(f"Meta ads_archive: HTTP {r.status_code} body is not JSON", status=r.status_code)
        return parse_page(payload, r.status_code)

    def _ads(self, page_ids: list, start: date, end: date):
        after = None
        for page in range(self.max_pages):
            if page:
                time.sleep(self.pause_seconds)
            params = request_params(
                page_ids,
                self.countries,
                start,
                end,
                limit=self.limit,
                after=after,
                media_type=self.media_type,
                active_status=self.active_status,
            )
            ads, after = self._get(params)
            yield from ads
            if after is None:
                return
        logger.warning(f"meta_adlib_feeder hit max_pages for pages {page_ids}")

    @staticmethod
    def to_metadata(record: dict, media_type: str = "ALL") -> Metadata:
        record = json.loads(redact(json.dumps(record)))
        m = Metadata().set_url(AD_PAGE_URL.format(record["id"]))
        m.set("source", "meta_adlib")
        m.set("source_ad_id", str(record["id"]))
        m.set("advertiser_id", str(record["page_id"]))
        m.set("advertiser_name", record.get("page_name"))
        m.set("media_type_query", media_type)
        m.set("first_shown", _day(record.get("ad_delivery_start_time")))
        m.set("last_shown", _day(record.get("ad_delivery_stop_time")))
        m.set("snapshot_url", record.get("ad_snapshot_url"))
        m.set("publisher_platforms", record.get("publisher_platforms") or [])
        m.set("media_urls", [])
        m.set("raw", record)
        m.set_title(f"{record.get('page_name')} Meta ad {record['id']}")
        return m

    def __iter__(self) -> Metadata:
        end = datetime.now(UTC).date()
        start = end - timedelta(days=self.days_back - 1)
        ids = [str(p) for p in self.page_ids]
        for i in range(0, len(ids), MAX_PAGE_IDS):
            group = ids[i : i + MAX_PAGE_IDS]
            count = 0
            for record in self._ads(group, start, end):
                count += 1
                yield self.to_metadata(record, self.media_type)
            logger.info(f"meta_adlib_feeder: {count} ads for pages {group} ({start}..{end})")
