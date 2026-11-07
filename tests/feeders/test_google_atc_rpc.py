import json
from pathlib import Path

import pytest
import requests

from auto_archiver.modules.google_atc_feeder import atc_rpc

DATA = Path(__file__).resolve().parents[1] / "data" / "google_atc"  # recorded 2026-11-07


def _resp(status=200, body="", url=atc_rpc.BASE, ctype="application/json"):
    r = requests.Response()
    r.status_code, r._content, r.url = status, body.encode(), url
    r.headers["content-type"] = ctype
    return r


class Session:
    def __init__(self, route):
        self.route, self.calls, self.headers = route, [], {}

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if method == "GET" and url == f"{atc_rpc.BASE}/":
            return _resp(ctype="text/html")
        return self.route(method, url, kw)


def test_search_creatives_filters_and_parses():
    body = (DATA / "search_video.json").read_text()
    s = Session(lambda m, u, kw: _resp(body=body))
    from datetime import date

    res = atc_rpc.Client(session=s, min_interval=0).search_creatives(
        "AR09597646619583447041", start=date(2026, 10, 1), end=date(2026, 11, 6), region="US", fmt="VIDEO"
    )
    sent = json.loads(s.calls[1][2]["data"]["f.req"])
    assert (
        sent["3"]["4"] == 3 and sent["3"]["6"] == 20261001 and sent["3"]["7"] == 20261106 and sent["3"]["8"] == [2840]
    )
    assert res["total"] == (3, 3) and not res["truncated"]
    item = res["items"][0]
    assert (item["creative_id"], item["format"], item["first_shown"], item["last_shown"]) == (
        "CR17871033171156402177",
        "VIDEO",
        "2025-12-14",
        "2025-12-29",
    )
    assert atc_rpc.decode_overlay(item["preview_url"]) == {"externalVideoId": "-DT_zyuTmCs"}


def test_get_creative_regions_and_preview():
    detail = atc_rpc.parse_creative(json.loads((DATA / "creative_video.json").read_text()))
    assert detail["format"] == "VIDEO" and detail["regions"] == [{"region": "US", "last_shown": "2025-12-28"}]
    assert atc_rpc.parse_preview_body((DATA / "content_video.js").read_text()) == {"youtube_video_ids": ["-DT_zyuTmCs"]}


def test_sorry_redirect_is_blocked():
    s = Session(lambda m, u, kw: _resp(429, "", url="https://www.google.com/sorry/index"))
    with pytest.raises(atc_rpc.AtcError) as e:
        atc_rpc.Client(session=s, min_interval=0).get_creative("AR1", "CR1")
    assert e.value.kind == atc_rpc.BLOCKED


def test_throttles_between_requests():
    t = {"now": 0.0}
    sleeps = []
    s = Session(lambda m, u, kw: _resp(body="{}"))
    c = atc_rpc.Client(session=s, min_interval=1.5, clock=lambda: t["now"], sleep=sleeps.append)
    c.rpc(atc_rpc.SEARCH, {})
    c.rpc(atc_rpc.SEARCH, {})
    assert sleeps == [1.5, 1.5]
