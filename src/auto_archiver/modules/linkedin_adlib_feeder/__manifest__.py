{
    "name": "LinkedIn Ad Library Feeder",
    "type": ["feeder"],
    "entry_point": "linkedin_adlib_feeder::LinkedInAdLibFeeder",
    "requires_setup": True,
    "dependencies": {"python": ["requests", "loguru"]},
    "configs": {
        "access_token": {"default": None, "help": "3-legged member OAuth token for the LinkedIn Ad Library API"},
        "advertisers": {
            "default": [],
            "help": "advertiser names to query (from the targets phase), e.g. Kering, PepsiCo",
        },
        "linkedin_version": {"default": "202610", "help": "LinkedIn-Version header (YYYYMM)"},
        "days_back": {"default": 7, "type": "int", "help": "size of the date window ending yesterday"},
        "count": {"default": 25, "type": "int", "help": "ads per page"},
        "max_pages": {"default": 40, "type": "int", "help": "page cap per advertiser"},
        "pause_seconds": {"default": 8, "type": "int", "help": "pause between calls (the API throttles per minute)"},
        "timeout": {"default": 60, "type": "int", "help": "request timeout in seconds"},
    },
    "description": """
Feeds one item per ad from LinkedIn's official Ad Library API (`/rest/adLibrary?q=criteria&advertiser=...`)
for a list of advertiser names over a recent date window. It queries by advertiser rather than keyword, because
keyword search mostly returns other companies' ads that mention the term. Each item's URL is the ad's public
library page and the provider record is kept under `raw`. The API returns no creative media links, so these items
carry metadata only. A failed API call raises instead of yielding nothing.
""",
}
