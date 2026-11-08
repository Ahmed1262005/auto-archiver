{
    "name": "TikTok Ad Library Feeder",
    "type": ["feeder"],
    "entry_point": "tiktok_adlib_feeder::TikTokAdLibFeeder",
    "requires_setup": True,
    "dependencies": {"python": ["requests", "loguru"]},
    "configs": {
        "client_key": {"default": None, "help": "TikTok Commercial Content API client key"},
        "client_secret": {"default": None, "help": "TikTok Commercial Content API client secret"},
        "access_token": {"default": None, "help": "client token; fetched from client_key/client_secret when unset"},
        "advertiser_business_ids": {
            "default": [],
            "help": "approved TikTok advertiser business_ids (from the targets phase)",
        },
        "days_back": {"default": 7, "type": "int", "help": "size of the published-date window ending yesterday"},
        "country_code": {"default": "ALL", "help": "country filter for the ad query"},
        "max_count": {"default": 50, "type": "int", "help": "ads per page (API max 50)"},
        "max_pages": {"default": 20, "type": "int", "help": "page cap per advertiser"},
        "timeout": {"default": 90, "type": "int", "help": "request timeout in seconds"},
    },
    "description": """
Feeds one item per ad from TikTok's official Commercial Content (Ad Library) API for a list of advertiser
business_ids over a recent date window. Each item's URL is the ad's public library page; the provider record is
kept under `raw`, and the ad's direct video/image links under `media_urls` for `direct_media_extractor`.
A failed API call raises instead of yielding nothing, so an outage is never recorded as "no ads".
""",
}
