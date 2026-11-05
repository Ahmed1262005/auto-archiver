{
    "name": "Google Ads Transparency Center Feeder",
    "type": ["feeder"],
    "entry_point": "google_atc_feeder::GoogleATCFeeder",
    "requires_setup": True,
    "dependencies": {"python": ["google.cloud.bigquery", "google.oauth2", "loguru"]},
    "configs": {
        "advertiser_ids": {
            "default": [],
            "help": "Google Ads Transparency advertiser ids (AR...) from the targets phase",
        },
        "credentials_json": {
            "default": None,
            "help": "service account JSON (string or file path); application default credentials when unset",
        },
        "project": {"default": None, "help": "GCP project billed for the query; defaults to the credentials' project"},
        "days_back": {"default": 7, "type": "int", "help": "creatives last shown within this many days"},
        "max_rows_per_advertiser": {
            "default": 2000,
            "type": "int",
            "help": "most recent creatives kept per advertiser, so a large advertiser cannot crowd out the others",
        },
        "maximum_gb_billed": {
            "default": 80,
            "type": "int",
            "help": "hard cap on bytes scanned; the query fails instead of exceeding it",
        },
    },
    "description": """
Feeds one item per creative from Google's official Ads Transparency Center public BigQuery dataset
(`bigquery-public-data.google_ads_transparency_center.creative_stats`). The table is ~140 GB and unpartitioned,
so every query scans most of it: this feeder runs ONE query for all configured advertiser ids per run and caps the
bytes billed. Each item's URL is the creative's Transparency Center page; per-region stats are kept under `regions`
and the row under `raw`. The dataset has no direct media links, so items carry metadata only.
""",
}
