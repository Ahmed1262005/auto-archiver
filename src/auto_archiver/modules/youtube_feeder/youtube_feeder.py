import re
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta

import requests

from auto_archiver.core import Feeder, Metadata
from auto_archiver.core.consts import SetupError
from auto_archiver.utils.custom_logger import logger
from auto_archiver.utils.provider_errors import ProviderError

API = "https://www.googleapis.com/youtube/v3"
CHANNELS_URL = f"{API}/channels"
PLAYLIST_ITEMS_URL = f"{API}/playlistItems"
VIDEOS_URL = f"{API}/videos"
# https://developers.google.com/youtube/v3/determine_quota_cost (default project quota: 10,000 units/day)
PLAYLISTS_URL = f"{API}/playlists"
QUOTA_UNITS = {CHANNELS_URL: 1, PLAYLIST_ITEMS_URL: 1, VIDEOS_URL: 1, PLAYLISTS_URL: 1}
VIDEO_PARTS = "snippet,contentDetails,status,statistics,topicDetails"
WATCH_URL = "https://www.youtube.com/watch?v={}"
EMBED_URL = "https://www.youtube.com/embed/{}"
CHANNEL_URL = "https://www.youtube.com/channel/{}"

AD_EVIDENCE_NONE = "none"  # a brand's own upload; says nothing about whether it ran as an ad
AD_EVIDENCE_AD_LIBRARY = "ad_library"  # the video is linked from an ad-library record

# error reason -> kind; reasons come from error.details[].reason (ErrorInfo) or error.errors[].reason
ERROR_KINDS = {
    "quotaExceeded": "quota",
    "dailyLimitExceeded": "quota",
    "RATE_LIMIT_EXCEEDED": "rate_limited",
    "rateLimitExceeded": "rate_limited",
    "userRateLimitExceeded": "rate_limited",
    "keyInvalid": "auth",
    "keyExpired": "auth",
    "API_KEY_INVALID": "auth",
    "API_KEY_EXPIRED": "auth",
    "API_KEY_SERVICE_BLOCKED": "auth",
    "SERVICE_DISABLED": "auth",
    "accessNotConfigured": "auth",
    "forbidden": "auth",
    "ipRefererBlocked": "auth",
    "channelNotFound": "not_found",
    "playlistNotFound": "not_found",
    "videoNotFound": "not_found",
    "backendError": "transient",
    "internalError": "transient",
}


class YouTubeAPIError(ProviderError):
    """`kind` is one of quota, rate_limited, auth, not_found, transient, bad_request, schema."""

    def __init__(self, message: str, *, kind: str, status: int | None = None, code: str | None = None):
        super().__init__(message, status=status, code=code, schema=kind == "schema")
        self.kind = kind


def error_reason(payload) -> str | None:
    err = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(err, dict):
        return None
    for d in err.get("details") or []:
        if isinstance(d, dict) and d.get("reason"):
            return d["reason"]
    for e in err.get("errors") or []:
        if isinstance(e, dict) and e.get("reason"):
            return e["reason"]
    return err.get("status")


def error_kind(status: int | None, reason: str | None) -> str:
    if reason in ERROR_KINDS:
        return ERROR_KINDS[reason]
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status is None or status >= 500:
        return "transient"
    return "bad_request"


def check(payload, status: int = 200, what: str = "YouTube API") -> dict:
    """The payload of a successful call; raises YouTubeAPIError on an error envelope or a non-object body."""
    if isinstance(payload, dict) and "error" in payload or status >= 400:
        reason = error_reason(payload)
        err = payload.get("error") if isinstance(payload, dict) else None
        msg = err.get("message") if isinstance(err, dict) else None
        kind = error_kind(status, reason)
        raise YouTubeAPIError(f"{what}: HTTP {status} {reason}: {msg}", kind=kind, status=status, code=reason)
    if not isinstance(payload, dict):
        raise YouTubeAPIError(f"{what}: response is not a JSON object", kind="schema", status=status)
    return payload


def _items(payload, what: str) -> list:
    items = check(payload, what=what).get("items", [])
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise YouTubeAPIError(f"{what}: items is not a list of objects", kind="schema")
    return items


def channel_params(channel_id: str) -> dict:
    return {"part": "snippet,contentDetails", "id": channel_id}


def playlist_params(playlist_id: str, page_token: str | None = None, max_results: int = 50) -> dict:
    p = {"part": "snippet,contentDetails,status", "playlistId": playlist_id, "maxResults": max_results}
    return p | ({"pageToken": page_token} if page_token else {})


def playlists_params(channel_id: str, page_token: str | None = None, max_results: int = 50) -> dict:
    p = {"part": "snippet,contentDetails", "channelId": channel_id, "maxResults": max_results}
    return p | ({"pageToken": page_token} if page_token else {})


def parse_playlists(payload) -> tuple[list[str], str | None]:
    """(playlist ids, next page token or None) from playlists.list. Brand playlists are where unlisted videos
    (often the ones that ran as ads) are linked; the uploads playlist only lists public uploads."""
    items = _items(payload, "playlists.list")
    ids = [i["id"] for i in items if isinstance(i.get("id"), str)]
    token = payload.get("nextPageToken")
    return ids, token if isinstance(token, str) and token else None


def videos_params(ids: list[str]) -> dict:
    if not 0 < len(ids) <= 50:
        raise ValueError("videos.list takes 1-50 ids")
    return {"part": VIDEO_PARTS, "id": ",".join(ids), "maxResults": 50}


def parse_channel(payload) -> dict:
    """{channel_id, title, uploads} for a channels.list response; an unknown channel id answers 200 with no items."""
    items = _items(payload, "channels.list")
    if not items:
        raise YouTubeAPIError("channels.list: channel not found", kind="not_found", status=200, code="channelNotFound")
    ch = items[0]
    uploads = ((ch.get("contentDetails") or {}).get("relatedPlaylists") or {}).get("uploads")
    if not isinstance(uploads, str) or not uploads or not isinstance(ch.get("id"), str):
        raise YouTubeAPIError("channels.list: channel has no id or uploads playlist", kind="schema")
    return {"channel_id": ch["id"], "title": (ch.get("snippet") or {}).get("title"), "uploads": uploads}


def parse_playlist_page(payload) -> tuple[list, str | None]:
    """(playlist items, next page token or None); every item must name its video."""
    items = _items(payload, "playlistItems.list")
    for i in items:
        if not isinstance((i.get("contentDetails") or {}).get("videoId"), str):
            raise YouTubeAPIError("playlistItems.list: item without contentDetails.videoId", kind="schema")
    token = payload.get("nextPageToken")
    if token is not None and not isinstance(token, str):
        raise YouTubeAPIError("playlistItems.list: nextPageToken is not a string", kind="schema")
    return items, token or None


def parse_videos(payload) -> list:
    """Video resources from videos.list; each needs an id, snippet.publishedAt and snippet.channelId."""
    items = _items(payload, "videos.list")
    for v in items:
        sn = v.get("snippet")
        if (
            not isinstance(v.get("id"), str)
            or not isinstance(sn, dict)
            or not isinstance(sn.get("publishedAt"), str)
            or not isinstance(sn.get("channelId"), str)
        ):
            raise YouTubeAPIError("videos.list: video without id / snippet.publishedAt / channelId", kind="schema")
    return items


Get = Callable[[str, dict], dict]


def fetch_page(get: Get, channel_id: str, cursor: dict | None = None, max_results: int = 50) -> dict:
    """One page of a channel's uploads as the raw API responses it took, plus the quota units spent.

    `get(url, params)` performs one call and returns the checked payload. The first page also resolves the channel's
    uploads playlist; later pages carry it in the cursor.
    """
    cursor = cursor or {}
    bundle = {"channel_id": channel_id, "page_token": cursor.get("page_token"), "units": 0}
    uploads = cursor.get("uploads")
    if not uploads:
        bundle["channel"] = get(CHANNELS_URL, channel_params(channel_id))
        bundle["units"] += QUOTA_UNITS[CHANNELS_URL]
        uploads = parse_channel(bundle["channel"])["uploads"]
    bundle["uploads"] = uploads
    bundle["playlist_items"] = get(PLAYLIST_ITEMS_URL, playlist_params(uploads, cursor.get("page_token"), max_results))
    bundle["units"] += QUOTA_UNITS[PLAYLIST_ITEMS_URL]
    items, _ = parse_playlist_page(bundle["playlist_items"])
    ids = [i["contentDetails"]["videoId"] for i in items]
    bundle["videos"] = {"items": []}
    if ids:
        bundle["videos"] = get(VIDEOS_URL, videos_params(ids))
        bundle["units"] += QUOTA_UNITS[VIDEOS_URL]
    return bundle


def parse_bundle(bundle: dict, start: date | None = None, end: date | None = None) -> tuple[list, dict | None]:
    """(videos published in [start, end], next cursor or None) from one fetch_page bundle.

    Uploads are listed newest first, so the walk stops after the page whose oldest upload predates `start`.
    Private or deleted uploads have no video resource and are skipped.
    """
    if not isinstance(bundle, dict) or not isinstance(bundle.get("uploads"), str):
        raise YouTubeAPIError("youtube page: bundle without uploads playlist", kind="schema")
    items, token = parse_playlist_page(bundle.get("playlist_items"))
    by_id = {v["id"]: v for v in parse_videos(bundle.get("videos"))}
    out, passed_start = [], False
    for i in items:
        added = (i.get("snippet") or {}).get("publishedAt") or (i.get("contentDetails") or {}).get("videoPublishedAt")
        if start and isinstance(added, str) and added[:10] < start.isoformat():
            passed_start = True
        v = by_id.get(i["contentDetails"]["videoId"])
        if v is None:
            continue
        day = v["snippet"]["publishedAt"][:10]
        if (start and day < start.isoformat()) or (end and day > end.isoformat()):
            continue
        out.append(v)
    nxt = {"uploads": bundle["uploads"], "page_token": token} if token and not passed_start else None
    return out, nxt


_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$")


def duration_seconds(iso: str | None) -> float | None:
    """Seconds in an ISO 8601 duration such as PT1M5S (videos.list contentDetails.duration); None if unparseable."""
    m = _DURATION.match(iso or "")
    if not m or not any(m.groups()):
        return None
    d, h, mi, s = m.groups()
    return int(d or 0) * 86400 + int(h or 0) * 3600 + int(mi or 0) * 60 + float(s or 0)


def thumbnails(video: dict) -> list[dict]:
    """Thumbnail links, largest first."""
    th = (video.get("snippet") or {}).get("thumbnails") or {}
    out = [{"name": k} | v for k, v in th.items() if isinstance(v, dict) and v.get("url")]
    return sorted(out, key=lambda t: -(t.get("width") or 0) * (t.get("height") or 0))


class YouTubeFeeder(Feeder):
    def setup(self) -> None:
        if not self.channel_ids:
            raise SetupError("youtube_feeder needs at least one channel_id")
        if not self.api_key:
            raise SetupError("youtube_feeder needs api_key")

    def _get(self, url: str, params: dict) -> dict:
        # the key goes in a header so it never appears in a URL, log line or exception message
        r = requests.get(url, params=params, headers={"X-Goog-Api-Key": self.api_key}, timeout=self.timeout)
        try:
            payload = r.json() if r.content else {}
        except ValueError:
            kind = "transient" if r.status_code >= 500 else "schema"
            raise YouTubeAPIError(
                f"YouTube API: HTTP {r.status_code} body is not JSON", kind=kind, status=r.status_code
            )
        return check(payload, r.status_code)

    @staticmethod
    def to_metadata(video: dict, ad_evidence: str = AD_EVIDENCE_NONE) -> Metadata:
        sn = video["snippet"]
        m = Metadata().set_url(WATCH_URL.format(video["id"]))
        m.set("source", "youtube")
        m.set("source_ad_id", video["id"])
        m.set("advertiser_id", sn["channelId"])
        m.set("advertiser_name", sn.get("channelTitle"))
        m.set("published_at", sn["publishedAt"])
        m.set("ad_evidence", ad_evidence)
        m.set("embed_url", EMBED_URL.format(video["id"]))
        m.set("thumbnails", thumbnails(video))
        m.set("media_urls", [])
        m.set("raw", video)
        m.set_title(sn.get("title") or f"YouTube video {video['id']}")
        return m

    def _videos(self, channel_id: str, start: date):
        cursor = None
        for _ in range(self.max_pages):
            bundle = fetch_page(self._get, channel_id, cursor, self.max_results)
            videos, cursor = parse_bundle(bundle, start)
            yield from videos
            if not cursor:
                return
        logger.warning(f"youtube_feeder hit max_pages for channel {channel_id}")

    def __iter__(self) -> Metadata:
        start = datetime.now(UTC).date() - timedelta(days=self.days_back)
        for channel_id in self.channel_ids:
            count = 0
            for video in self._videos(channel_id, start):
                count += 1
                yield self.to_metadata(video)
            logger.info(f"youtube_feeder: {count} uploads for channel {channel_id} since {start}")
