import json
from datetime import date
from pathlib import Path

import pytest

from auto_archiver.core.consts import SetupError
from auto_archiver.modules.youtube_feeder import youtube_feeder as yf

DATA = Path(__file__).parent.parent / "data" / "youtube"
CHANNEL = "UCosXctaTYxN4YPIvI5Fpcrw"  # Coca-Cola, recorded 2026-11-06


def load(name):
    return json.loads((DATA / name).read_text())


class FakeResponse:
    def __init__(self, payload, status=200, content=None):
        self._payload, self.status_code = payload, status
        self.content = content if content is not None else json.dumps(payload).encode()

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def fake_get(calls):
    def get(url, params):
        calls.append((url, params))
        if url == yf.CHANNELS_URL:
            return load("channels_cocacola.json")
        if url == yf.PLAYLIST_ITEMS_URL:
            return load("playlist_items_cocacola.json")
        return load("videos_cocacola.json")

    return get


def test_recorded_responses_parse():
    ch = yf.parse_channel(load("channels_cocacola.json"))
    assert ch == {"channel_id": CHANNEL, "title": "Coca-Cola", "uploads": "UUosXctaTYxN4YPIvI5Fpcrw"}
    items, token = yf.parse_playlist_page(load("playlist_items_cocacola.json"))
    assert len(items) == 12 and token
    videos = yf.parse_videos(load("videos_cocacola.json"))
    assert {v["id"] for v in videos} == {i["contentDetails"]["videoId"] for i in items}


def test_first_page_resolves_uploads_and_counts_units():
    calls = []
    bundle = yf.fetch_page(fake_get(calls), CHANNEL)
    assert [u for u, _ in calls] == [yf.CHANNELS_URL, yf.PLAYLIST_ITEMS_URL, yf.VIDEOS_URL]
    assert bundle["units"] == 3 and bundle["uploads"] == "UUosXctaTYxN4YPIvI5Fpcrw"
    assert len(calls[2][1]["id"].split(",")) == 12
    videos, cursor = yf.parse_bundle(bundle)
    assert len(videos) == 12
    assert cursor == {"uploads": "UUosXctaTYxN4YPIvI5Fpcrw", "page_token": bundle["playlist_items"]["nextPageToken"]}


def test_next_page_reuses_cursor_without_channel_call():
    calls = []
    bundle = yf.fetch_page(fake_get(calls), CHANNEL, {"uploads": "UUx", "page_token": "T"})
    assert [u for u, _ in calls] == [yf.PLAYLIST_ITEMS_URL, yf.VIDEOS_URL]
    assert calls[0][1]["pageToken"] == "T" and bundle["units"] == 2


def test_window_filters_and_stops_past_start():
    bundle = yf.fetch_page(fake_get([]), CHANNEL)
    videos, cursor = yf.parse_bundle(bundle, start=date(2026, 10, 20), end=date(2026, 10, 31))
    assert {v["snippet"]["publishedAt"][:10] for v in videos} == {"2026-10-26", "2026-10-23"}
    assert cursor is None  # the page reached uploads older than the window
    _, cursor = yf.parse_bundle(bundle, start=date(2026, 9, 1))
    assert cursor is not None


def test_upload_without_video_resource_is_skipped():
    bundle = yf.fetch_page(fake_get([]), CHANNEL)
    bundle["videos"]["items"] = bundle["videos"]["items"][1:]
    videos, _ = yf.parse_bundle(bundle)
    assert len(videos) == 11


@pytest.mark.parametrize(
    "name,status,kind,code",
    [
        ("error_api_key_invalid.json", 400, "auth", "API_KEY_INVALID"),
        ("error_quota_exceeded_doc_shape.json", 403, "quota", "quotaExceeded"),
        ("error_forbidden_no_key.json", 403, "auth", "forbidden"),
        ("error_invalid_value.json", 400, "bad_request", "invalid"),
    ],
)
def test_error_envelopes_are_typed(name, status, kind, code):
    with pytest.raises(yf.YouTubeAPIError) as e:
        yf.check(load(name), status)
    assert (e.value.kind, e.value.code, e.value.status) == (kind, code, status)


def test_legacy_key_invalid_reason_is_auth():
    payload = {"error": {"code": 400, "errors": [{"domain": "usageLimits", "reason": "keyInvalid"}]}}
    with pytest.raises(yf.YouTubeAPIError) as e:
        yf.check(payload, 400)
    assert e.value.kind == "auth"


def test_unknown_channel_is_not_found():
    with pytest.raises(yf.YouTubeAPIError) as e:
        yf.parse_channel(load("channels_empty.json"))
    assert e.value.kind == "not_found"


@pytest.mark.parametrize(
    "status,kind", [(500, "transient"), (503, "transient"), (429, "rate_limited"), (404, "not_found")]
)
def test_status_fallbacks(status, kind):
    assert yf.error_kind(status, None) == kind


def test_schema_breaks():
    with pytest.raises(yf.YouTubeAPIError) as e:
        yf.parse_videos({"items": [{"id": "x", "snippet": {"title": "no date"}}]})
    assert e.value.kind == "schema" and e.value.schema
    with pytest.raises(yf.YouTubeAPIError, match="items"):
        yf.parse_playlist_page({"items": {"0": {}}})
    with pytest.raises(yf.YouTubeAPIError, match="videoId"):
        yf.parse_playlist_page({"items": [{"contentDetails": {}}]})
    with pytest.raises(yf.YouTubeAPIError, match="nextPageToken"):
        yf.parse_playlist_page({"items": [], "nextPageToken": 5})
    with pytest.raises(yf.YouTubeAPIError, match="uploads"):
        yf.parse_channel({"items": [{"id": CHANNEL, "contentDetails": {}}]})
    with pytest.raises(yf.YouTubeAPIError):
        yf.check(["not", "an", "object"])


@pytest.mark.parametrize(
    "iso,s", [("PT15S", 15), ("PT1M5S", 65), ("PT2H", 7200), ("P1DT1S", 86401), ("P0D", 0), ("", None), ("x", None)]
)
def test_duration(iso, s):
    assert yf.duration_seconds(iso) == s


@pytest.fixture
def feeder(setup_module):
    return setup_module("youtube_feeder", {"api_key": "k", "channel_ids": [CHANNEL], "days_back": 3650})


def test_setup_requires_channels_and_key(setup_module):
    with pytest.raises(SetupError):
        setup_module("youtube_feeder", {"api_key": "k"})
    with pytest.raises(SetupError):
        setup_module("youtube_feeder", {"channel_ids": [CHANNEL]})


def test_feeder_yields_metadata_without_media(feeder, mocker):
    pl = load("playlist_items_cocacola.json")
    pl.pop("nextPageToken")
    get = mocker.patch(
        "requests.get",
        side_effect=[
            FakeResponse(load("channels_cocacola.json")),
            FakeResponse(pl),
            FakeResponse(load("videos_cocacola.json")),
        ],
    )
    items = list(feeder)
    assert len(items) == 12
    first = items[0]
    assert first.get_url() == "https://www.youtube.com/watch?v=q2-Blj7xvgw"
    assert first.get("source") == "youtube" and first.get("advertiser_id") == CHANNEL
    assert first.get("ad_evidence") == "none" and first.get("media_urls") == []
    assert first.get("embed_url") == "https://www.youtube.com/embed/q2-Blj7xvgw"
    assert first.get("thumbnails")[0]["name"] == "fhd"
    for c in get.call_args_list:
        assert c.kwargs["headers"] == {"X-Goog-Api-Key": "k"} and "key" not in c.kwargs["params"]


def test_feeder_raises_on_quota_instead_of_yielding_nothing(feeder, mocker):
    mocker.patch("requests.get", return_value=FakeResponse(load("error_quota_exceeded_doc_shape.json"), 403))
    with pytest.raises(yf.YouTubeAPIError) as e:
        list(feeder)
    assert e.value.kind == "quota"


def test_non_json_5xx_is_transient(feeder, mocker):
    mocker.patch("requests.get", return_value=FakeResponse(None, 502, content=b"<html>bad gateway</html>"))
    with pytest.raises(yf.YouTubeAPIError) as e:
        list(feeder)
    assert e.value.kind == "transient"
