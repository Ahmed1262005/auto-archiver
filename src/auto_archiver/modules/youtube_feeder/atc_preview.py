"""What Google's Ads Transparency Center shows for one creative: the video file and/or YouTube video behind it.

The creative page (adstransparency.google.com/advertiser/AR.../creative/CR...) loads its data from the
LookupService/GetCreativeById RPC; each variation links a preview script (displayads-formats.googleusercontent.com
/ads/preview/content.js). For video ads that script carries either a VAST 2.0 document whose <MediaFile> is a
googlevideo.com mp4 (H.264, up to 720p; the link is bound to the caller's IP and expires within hours) or a
YouTube thumbnail (i.ytimg.com/vi/<id>/) naming the YouTube video that ran as the ad, often an unlisted one.

These are undocumented endpoints of a public Google page (adstransparency.google.com has no robots.txt). Google
can change them at any time: parse failures raise AtcPreviewError(kind="schema")."""

import json
import re
import time
from collections.abc import Callable

from auto_archiver.utils.provider_errors import ProviderError

RPC_URL = "https://adstransparency.google.com/anji/_/rpc/LookupService/GetCreativeById"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://adstransparency.google.com/"}
ADVERTISER = re.compile(r"^AR\d{20}$")
CREATIVE = re.compile(r"^CR\d{20}$")
PAGE_URL = re.compile(r"adstransparency\.google\.com/advertiser/(AR\d{20})/creative/(CR\d{20})")
_MEDIA_FILE = re.compile(r"<MediaFile([^>]*)>\s*<!\[CDATA\[\s*(https://[^\]\s]+)\s*\]\]>\s*</MediaFile>")
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_YOUTUBE_ID = re.compile(
    r"(?:ytimg\.com/vi(?:_webp)?/|youtube\.com/(?:watch\?v=|embed/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)


class AtcPreviewError(ProviderError):
    """`kind`: not_found, rate_limited, transient, bad_request, schema."""

    def __init__(self, message: str, *, kind: str, status: int | None = None):
        super().__init__(message, status=status, code=kind, schema=kind == "schema")
        self.kind = kind


def ids_from_url(url: str) -> tuple[str, str] | None:
    """(advertiser_id, creative_id) from an ATC creative page URL."""
    m = PAGE_URL.search(url or "")
    return (m.group(1), m.group(2)) if m else None


def creative_form(advertiser_id: str, creative_id: str) -> dict:
    if not ADVERTISER.match(advertiser_id or "") or not CREATIVE.match(creative_id or ""):
        raise AtcPreviewError(
            f"not an ATC advertiser/creative id: {advertiser_id!r} {creative_id!r}", kind="bad_request"
        )
    return {"f.req": json.dumps({"1": advertiser_id, "2": creative_id, "5": {"1": 1}})}


def parse_creative(payload) -> list[str]:
    """Preview script URLs of every variation; an unknown or withdrawn creative answers {}."""
    if not isinstance(payload, dict):
        raise AtcPreviewError("GetCreativeById: response is not a JSON object", kind="schema")
    body = payload.get("1")
    if body is None:
        raise AtcPreviewError("GetCreativeById: creative not found", kind="not_found")
    variations = body.get("5") if isinstance(body, dict) else None
    if not isinstance(variations, list):
        raise AtcPreviewError("GetCreativeById: no variation list", kind="schema")
    urls = []
    for v in variations:
        u = ((v or {}).get("1") or {}).get("4") if isinstance(v, dict) else None
        if isinstance(u, str) and u.startswith("https://"):
            urls.append(u)
    return urls


def _decode(js: str) -> str:
    # the script embeds HTML/XML with \xNN and \uNNNN escapes
    return re.sub(r"\\x([0-9a-fA-F]{2})|\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1) or m.group(2), 16)), js)


def parse_preview(js: str) -> dict:
    """{"media_files": [{url, type, width, height, bitrate}], "youtube_ids": [...]} from one preview script."""
    text = _decode(js)
    files = []
    for attrs, url in _MEDIA_FILE.findall(text):
        a = dict(_ATTR.findall(attrs))
        files.append(
            {
                "url": url.replace("&amp;", "&"),
                "type": a.get("type"),
                "width": int(a["width"]) if a.get("width", "").isdigit() else None,
                "height": int(a["height"]) if a.get("height", "").isdigit() else None,
                "bitrate": int(a["bitrate"]) if a.get("bitrate", "").isdigit() else None,
            }
        )
    ids = list(dict.fromkeys(_YOUTUBE_ID.findall(text)))
    return {"media_files": files, "youtube_ids": ids}


Post = Callable[[str, dict], object]  # (url, form) -> parsed JSON
Get = Callable[[str], str]  # url -> text


def lookup(post: Post, get: Get, advertiser_id: str, creative_id: str) -> dict:
    """Every media file and YouTube id the creative's previews show, plus the raw responses for archiving."""
    payload = post(RPC_URL, creative_form(advertiser_id, creative_id))
    previews = parse_creative(payload)
    files, ids, scripts = [], [], []
    for u in previews:
        js = get(u)
        scripts.append(js)
        p = parse_preview(js)
        files += [f for f in p["media_files"] if f["url"] not in {x["url"] for x in files}]
        ids += [i for i in p["youtube_ids"] if i not in ids]
    return {
        "advertiser_id": advertiser_id,
        "creative_id": creative_id,
        "media_files": files,
        "youtube_ids": ids,
        "creative": payload,
        "previews": len(previews),
    }


_last_call = [0.0]


def _pace(min_interval: float) -> None:
    # ATC answers 429 to bursts (seen after ~10 calls 0.2 s apart); one call per interval per process stays under it
    wait = _last_call[0] + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def requests_transport(timeout: int = 30, min_interval: float = 2.0):
    """(post, get) over requests, raising AtcPreviewError with a kind on HTTP failures."""
    import requests

    def _check(r):
        if r.status_code == 429 or "/sorry/" in r.url:
            raise AtcPreviewError(f"ATC: HTTP {r.status_code} rate limited", kind="rate_limited", status=r.status_code)
        if r.status_code >= 500:
            raise AtcPreviewError(f"ATC: HTTP {r.status_code}", kind="transient", status=r.status_code)
        if r.status_code == 404:
            raise AtcPreviewError("ATC: HTTP 404", kind="not_found", status=404)
        if r.status_code >= 400:
            raise AtcPreviewError(f"ATC: HTTP {r.status_code}", kind="bad_request", status=r.status_code)
        return r

    def post(url, form):
        _pace(min_interval)
        try:
            r = _check(requests.post(url, params={"authuser": "0"}, data=form, headers=HEADERS, timeout=timeout))
        except requests.RequestException as e:
            raise AtcPreviewError(f"ATC: {e}", kind="transient") from e
        try:
            return r.json()
        except ValueError as e:
            raise AtcPreviewError("ATC: response is not JSON", kind="schema", status=r.status_code) from e

    def get(url):
        _pace(min_interval)
        try:
            return _check(requests.get(url, headers=HEADERS, timeout=timeout)).text
        except requests.RequestException as e:
            raise AtcPreviewError(f"ATC preview: {e}", kind="transient") from e

    return post, get
