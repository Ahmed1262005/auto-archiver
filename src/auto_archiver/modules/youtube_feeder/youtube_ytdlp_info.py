"""YouTube listing and metadata through yt-dlp, for when the Data API key is missing or out of quota.

`to_resource` maps a yt-dlp info dict onto the Data API videos.list resource shape (id, snippet, contentDetails,
status, statistics), so callers can reuse one normaliser. Fields yt-dlp does not expose are left out."""

from datetime import UTC, datetime

import yt_dlp

from auto_archiver.modules.youtube_feeder import youtube_media as ym

CHANNEL_TABS = ("videos", "shorts")
CHANNEL_TAB_URL = "https://www.youtube.com/channel/{}/{}"
_PRIVACY = {"public": "public", "unlisted": "unlisted", "private": "private", "needs_auth": "private"}


def _opts(timeout: int, cookiefile: str | None, extra: dict | None = None) -> dict:
    o = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "socket_timeout": timeout,
        "js_runtimes": {"node": {}},
        "ignoreerrors": False,
    }
    if cookiefile:
        o["cookiefile"] = cookiefile
    return o | (extra or {})


def _iso_duration(seconds) -> str:
    s = int(seconds or 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return "PT" + (f"{h}H" if h else "") + (f"{m}M" if m else "") + f"{sec}S"


def _published(info: dict) -> str | None:
    ts = info.get("timestamp") or info.get("release_timestamp")
    if ts:
        return datetime.fromtimestamp(int(ts), UTC).isoformat().replace("+00:00", "Z")
    d = info.get("upload_date")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}T00:00:00Z" if isinstance(d, str) and len(d) == 8 else None


def to_resource(info: dict) -> dict:
    """A videos.list-shaped resource from one yt-dlp video info dict."""
    stats = {
        k: info[f]
        for k, f in (("viewCount", "view_count"), ("likeCount", "like_count"), ("commentCount", "comment_count"))
        if info.get(f) is not None
    }
    sn = {
        "publishedAt": _published(info),
        "channelId": info.get("channel_id"),
        "channelTitle": info.get("channel") or info.get("uploader") or "",
        "title": info.get("title") or "",
        "description": info.get("description") or "",
        "tags": list(info.get("tags") or []),
    }
    if info.get("language"):
        sn["defaultAudioLanguage"] = info["language"]
    thumbs = [t for t in info.get("thumbnails") or [] if t.get("url") and t.get("width")]
    if thumbs:
        best = max(thumbs, key=lambda t: (t.get("width") or 0) * (t.get("height") or 0))
        sn["thumbnails"] = {"maxres": {k: best[k] for k in ("url", "width", "height") if best.get(k)}}
    return {
        "kind": "youtube#video",
        "id": info.get("id"),
        "snippet": sn,
        "contentDetails": {"duration": _iso_duration(info.get("duration"))},
        "status": {"privacyStatus": _PRIVACY.get(info.get("availability") or "public", "public")},
        "statistics": stats,
    }


def channel_video_ids(
    channel_id: str, *, limit: int = 50, timeout: int = 60, cookiefile: str | None = None, ydl_factory=yt_dlp.YoutubeDL
) -> list[str]:
    """Newest video ids on a channel's Videos and Shorts tabs (public uploads only), without Data API quota."""
    ids: list[str] = []
    opts = _opts(timeout, cookiefile, {"extract_flat": "in_playlist", "playlistend": limit})
    for tab in CHANNEL_TABS:
        try:
            with ydl_factory(opts) as ydl:
                info = ydl.extract_info(CHANNEL_TAB_URL.format(channel_id, tab), download=False)
        except yt_dlp.utils.DownloadError as e:
            if "does not have a" in str(e):  # channel without a Shorts tab
                continue
            raise ym.MediaFetchError(ym.media_reason(str(e)), str(e)) from e
        ids += [e["id"] for e in (info or {}).get("entries") or [] if e and e.get("id") and e["id"] not in ids]
    return ids


def video_info(url: str, *, timeout: int = 60, cookiefile: str | None = None, ydl_factory=yt_dlp.YoutubeDL) -> dict:
    """yt-dlp's info dict for one video, or MediaFetchError with a youtube_media reason."""
    try:
        with ydl_factory(_opts(timeout, cookiefile)) as ydl:
            return ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise ym.MediaFetchError(ym.media_reason(str(e)), str(e)) from e
