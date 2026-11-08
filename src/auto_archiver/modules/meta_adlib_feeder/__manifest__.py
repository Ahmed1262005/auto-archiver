{
    "name": "Meta Ad Library Feeder",
    "type": ["feeder"],
    "entry_point": "meta_adlib_feeder::MetaAdLibFeeder",
    "requires_setup": True,
    "dependencies": {"python": ["requests", "loguru"]},
    "configs": {
        "access_token": {"default": None, "help": "user access token of an identity-confirmed Meta developer account"},
        "page_ids": {"default": [], "help": "Facebook Page IDs to query (from the targets phase), 10 per request"},
        "countries": {"default": ["GB"], "help": "ad_reached_countries; non-political ads only return for UK/EU"},
        "media_type": {"default": "ALL", "help": "ALL, IMAGE, MEME, VIDEO or NONE"},
        "active_status": {"default": "ALL", "help": "ACTIVE, INACTIVE or ALL"},
        "graph_version": {"default": "v27.0", "help": "Graph API version"},
        "days_back": {"default": 7, "type": "int", "help": "size of the delivery-date window ending today"},
        "limit": {"default": 100, "type": "int", "help": "ads per page"},
        "max_pages": {"default": 50, "type": "int", "help": "page cap per request group"},
        "pause_seconds": {"default": 2, "type": "int", "help": "pause between pages"},
        "timeout": {"default": 60, "type": "int", "help": "request timeout in seconds"},
    },
    "description": """
Feeds one item per ad from Meta's official Ad Library API (Graph API `/ads_archive`) for a list of Facebook Page
IDs, ad_reached_countries and a delivery-date window. Pages are followed by `paging.cursors.after` until a page
comes back empty or without a next link. Each item's URL is the ad's public Ad Library page and the provider
record is kept under `raw`, with the access token removed from `ad_snapshot_url`. The API returns no direct media
links, only the snapshot page, so these items carry metadata only. API errors raise a ProviderError carrying the
Graph error code and subcode.
""",
}
