{
    "name": "Direct Media Extractor",
    "type": ["extractor"],
    "entry_point": "direct_media_extractor::DirectMediaExtractor",
    "requires_setup": False,
    "dependencies": {"python": ["requests", "loguru"]},
    "configs": {
        "allowed_mimetypes": {
            "default": ["video/", "image/", "audio/"],
            "help": "content-type prefixes accepted as direct media",
        },
        "timeout": {"default": 60, "type": "int", "help": "request timeout in seconds"},
        "max_bytes": {"default": 2_000_000_000, "type": "int", "help": "largest file accepted; 0 for no limit"},
    },
    "description": """
Downloads a URL that points straight at a media file (e.g. provider CDN links returned by ad library APIs).
It checks the response content-type and only succeeds when a non-empty media file was saved, so a page URL
falls through to the next extractor. Signed CDN links expire, so put this extractor first.
Each failed URL is recorded with a reason (expired, not_found, transient, http_error, not_media, empty,
too_large, partial) under `media_failures`; `fetch(url)` raises `MediaFetchError` with the same reason.
""",
}
