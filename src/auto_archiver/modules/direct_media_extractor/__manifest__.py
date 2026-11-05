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
    },
    "description": """
Downloads a URL that points straight at a media file (e.g. provider CDN links returned by ad library APIs).
It checks the response content-type and only succeeds when a non-empty media file was saved, so a page URL
falls through to the next extractor. Signed CDN links expire, so put this extractor first.
""",
}
