import json
import os
from collections import Counter
from datetime import UTC, date, datetime, timedelta

from google.cloud import bigquery
from google.oauth2 import service_account

from auto_archiver.core import Feeder, Metadata
from auto_archiver.core.consts import SetupError
from auto_archiver.utils.custom_logger import logger

TABLE = "bigquery-public-data.google_ads_transparency_center.creative_stats"
QUERY = f"""
SELECT creative_id, advertiser_id, advertiser_disclosed_name, advertiser_legal_name, ad_format_type,
  creative_page_url,
  MIN(SAFE_CAST(rs.first_shown AS DATE)) AS first_shown,
  MAX(SAFE_CAST(rs.last_shown AS DATE)) AS last_shown,
  ARRAY_AGG(STRUCT(rs.region_code, rs.first_shown, rs.last_shown,
    rs.times_shown_lower_bound, rs.times_shown_upper_bound)) AS regions
FROM `{TABLE}`, UNNEST(region_stats) rs
WHERE advertiser_id IN UNNEST(@advertiser_ids) AND SAFE_CAST(rs.last_shown AS DATE) >= @since
GROUP BY creative_id, advertiser_id, advertiser_disclosed_name, advertiser_legal_name, ad_format_type,
  creative_page_url
QUALIFY ROW_NUMBER() OVER (PARTITION BY advertiser_id ORDER BY last_shown DESC) <= @max_rows_per_advertiser
ORDER BY advertiser_id, last_shown DESC
"""


class GoogleATCFeeder(Feeder):
    def setup(self) -> None:
        if not self.advertiser_ids:
            raise SetupError("google_atc_feeder needs at least one advertiser_id")

    def _client(self) -> bigquery.Client:
        if not self.credentials_json:
            return bigquery.Client(project=self.project)
        info = self.credentials_json
        if os.path.isfile(info):
            with open(info) as f:
                info = f.read()
        creds = service_account.Credentials.from_service_account_info(json.loads(info))
        return bigquery.Client(credentials=creds, project=self.project or creds.project_id)

    def _rows(self, since: date):
        config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("advertiser_ids", "STRING", list(self.advertiser_ids)),
                bigquery.ScalarQueryParameter("since", "DATE", since),
                bigquery.ScalarQueryParameter("max_rows_per_advertiser", "INT64", self.max_rows_per_advertiser),
            ],
            maximum_bytes_billed=self.maximum_gb_billed * 10**9,
        )
        job = self._client().query(QUERY, job_config=config)
        rows = [dict(r) for r in job.result()]
        logger.info(f"google_atc_feeder: {len(rows)} creatives, {(job.total_bytes_billed or 0) / 1e9:.1f} GB billed")
        for advertiser_id, n in Counter(r["advertiser_id"] for r in rows).items():
            if n >= self.max_rows_per_advertiser:
                logger.warning(f"google_atc_feeder hit max_rows_per_advertiser for {advertiser_id}")
        return rows

    @staticmethod
    def to_metadata(row: dict) -> Metadata:
        row = json.loads(json.dumps(row, default=str))
        m = Metadata().set_url(row["creative_page_url"])
        m.set("source", "google_atc")
        m.set("source_ad_id", row["creative_id"])
        m.set("advertiser_id", row["advertiser_id"])
        m.set("advertiser_name", row.get("advertiser_disclosed_name") or row.get("advertiser_legal_name"))
        m.set("ad_type", row.get("ad_format_type"))
        m.set("first_shown", row.get("first_shown"))
        m.set("last_shown", row.get("last_shown"))
        m.set("regions", row.get("regions") or [])
        m.set("media_urls", [])
        m.set("raw", row)
        m.set_title(f"{m.get('advertiser_name')} Google ad {row['creative_id']}")
        return m

    def __iter__(self) -> Metadata:
        since = datetime.now(UTC).date() - timedelta(days=self.days_back)
        for row in self._rows(since):
            yield self.to_metadata(row)
