{
    "name": "YouTube Feeder",
    "type": ["feeder"],
    "entry_point": "youtube_feeder::YouTubeFeeder",
    "requires_setup": True,
    "dependencies": {"python": ["requests", "loguru"]},
    "configs": {
        "api_key": {"default": None, "help": "YouTube Data API v3 key (sent as the X-Goog-Api-Key header)"},
        "channel_ids": {"default": [], "help": "brand YouTube channel ids (UC...) from the targets phase"},
        "days_back": {"default": 7, "type": "int", "help": "keep uploads published in the last N days"},
        "max_results": {"default": 50, "type": "int", "help": "uploads per page (API max 50)"},
        "max_pages": {"default": 20, "type": "int", "help": "page cap per channel"},
        "timeout": {"default": 30, "type": "int", "help": "request timeout in seconds"},
    },
    "description": """
Feeds one item per video uploaded by a list of brand YouTube channels, through the official YouTube Data API v3
(channels.list -> uploads playlist -> playlistItems.list -> videos.list; 1 quota unit per call, search.list is not
used). Each item's URL is the public watch page; the API video resource is kept under `raw`, with `embed_url` and
`thumbnails`. An organic upload is not evidence of advertising: items carry `ad_evidence="none"` unless a caller
links the video to an ad-library record. No media is downloaded: YouTube's Terms of Service and API Developer
Policies forbid downloading or storing copies of audiovisual content without YouTube's written permission, so
`media_urls` is always empty. A failed API call raises a typed YouTubeAPIError instead of yielding nothing.
""",
}
