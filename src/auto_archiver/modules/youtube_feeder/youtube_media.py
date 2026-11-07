"""The video file behind a YouTube watch URL, through yt-dlp.

YouTube's Terms of Service do not authorise downloading (only "as expressly authorized by the Service" or "with
prior written permission"); the project owner chose to download anyway. Failures are typed with
direct_media_extractor's MediaFetchError so callers retry, back off or dead-letter them like any other media fetch."""

import mimetypes
import os
import re

import yt_dlp

from auto_archiver.core import Media
from auto_archiver.modules.direct_media_extractor.direct_media_extractor import (
    EMPTY,
    TOO_LARGE,
    TRANSIENT,
    MediaFetchError,
)

UNAVAILABLE = "unavailable"  # removed, private, age/members-gated, live or upcoming: retrying won't help
BLOCKED = "blocked"  # YouTube's bot check or HTTP 429: back off
FAILED = "failed"  # anything else, usually yt-dlp out of date with a YouTube change

# 720p is enough for every enricher; mp4/m4a first so the merge is a remux, not a re-encode
FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b"

REASONS = [
    (BLOCKED, r"not a bot|HTTP Error 429|Too Many Requests"),
    (TRANSIENT, r"HTTP Error 5\d\d|timed out|Connection|Temporary failure|IncompleteRead|Remote end closed"),
    (
        UNAVAILABLE,
        r"unavailable|Private video|has been removed|is not available|members|confirm your age|"
        r"account .*terminated|copyright|live event|Premieres in",
    ),
]


def media_reason(message: str) -> str:
    for reason, pattern in REASONS:
        if re.search(pattern, message, re.I):
            return reason
    return FAILED


def ydl_options(out_dir: str, timeout: int = 120) -> dict:
    args = [
        "-o",
        os.path.join(out_dir, "%(id)s.%(ext)s"),
        "--quiet",
        "--no-warnings",
        "--no-progress",
        "--no-playlist",
        "--no-write-subs",
        "-f",
        FORMAT,
        "--merge-output-format",
        "mp4",
        "--postprocessor-args",
        "ffmpeg:-bitexact",  # the same video merges to the same bytes, so its SHA-256 is stable
        "--js-runtimes",
        "node",
        "--socket-timeout",
        str(timeout),
    ]
    *_, opts = yt_dlp.parse_options(args)
    return opts | {"ignoreerrors": False}  # the CLI default swallows extraction errors into a None result


def _size(info: dict) -> int:
    return sum(f.get("filesize") or f.get("filesize_approx") or 0 for f in info.get("requested_formats") or [info])


def fetch_media(
    url: str, out_dir: str, *, max_bytes: int = 500 * 2**20, timeout: int = 120, ydl_factory=yt_dlp.YoutubeDL
) -> Media:
    """Download one video (<=720p mp4) into out_dir, or raise MediaFetchError with a reason."""
    try:
        with ydl_factory(ydl_options(out_dir, timeout)) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("live_status") in ("is_live", "is_upcoming", "post_live"):
                raise MediaFetchError(UNAVAILABLE, f"{url}: {info['live_status']}")
            if (size := _size(info)) > max_bytes:
                raise MediaFetchError(TOO_LARGE, f"{url}: {size} bytes > {max_bytes}")
            info = ydl.process_ie_result(info, download=True)
    except (yt_dlp.utils.DownloadError, yt_dlp.utils.ExtractorError) as e:
        raise MediaFetchError(media_reason(str(e)), f"{url}: {e}"[:500]) from e
    path = info.get("filepath") or next((d.get("filepath") for d in info.get("requested_downloads") or []), None)
    if not path or not os.path.exists(path) or not os.path.getsize(path):
        raise MediaFetchError(EMPTY, f"{url}: yt-dlp produced no file")
    mime = mimetypes.guess_type(path)[0] or "video/mp4"
    return Media(
        filename=path,
        properties={"source_content_type": mime, "source_url": url, "format_id": info.get("format_id")},
    )
