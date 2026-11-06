import json
from datetime import date

import pytest

from auto_archiver.core.consts import SetupError
from auto_archiver.modules.meta_adlib_feeder import meta_adlib_feeder as maf
from auto_archiver.utils.provider_errors import ProviderError

TOKEN = "EAAtesttoken123"


def ad(n, page="20531316728"):
    return {
        "id": str(n),
        "page_id": page,
        "page_name": "Nike",
        "ad_creation_time": "2026-10-30T09:00:00+0000",
        "ad_delivery_start_time": "2026-10-31T00:00:00+0000",
        "ad_snapshot_url": f"https://www.facebook.com/ads/archive/render_ad/?id={n}&access_token={TOKEN}",
        "ad_creative_bodies": ["Just do it."],
        "publisher_platforms": ["facebook", "instagram"],
        "eu_total_reach": 12000,
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self.status_code = status
        self.text = payload if isinstance(payload, str) else json.dumps(payload)


def page(ads, after=None):
    paging = {"cursors": {"before": "B", "after": after}} if after else {}
    if after:
        paging["next"] = f"https://graph.facebook.com/v27.0/ads_archive?access_token={TOKEN}&after={after}"
    return FakeResponse({"data": ads, "paging": paging})


@pytest.fixture
def feeder(setup_module, mocker):
    mocker.patch("time.sleep")
    return setup_module("meta_adlib_feeder", {"access_token": TOKEN, "page_ids": ["20531316728"], "limit": 2})


def test_setup_requires_pages_token_and_valid_media_type(setup_module):
    with pytest.raises(SetupError):
        setup_module("meta_adlib_feeder", {"access_token": "t"})
    with pytest.raises(SetupError):
        setup_module("meta_adlib_feeder", {"page_ids": ["1"]})
    with pytest.raises(SetupError):
        setup_module("meta_adlib_feeder", {"access_token": "t", "page_ids": ["1"], "media_type": "GIF"})


def test_request_params_match_documented_shape():
    p = maf.request_params(
        ["1", 2], ["GB", "FR"], date(2026, 11, 1), date(2026, 11, 3), limit=50, after="X", media_type="VIDEO"
    )
    assert json.loads(p["search_page_ids"]) == ["1", "2"] and json.loads(p["ad_reached_countries"]) == ["GB", "FR"]
    assert p["ad_delivery_date_min"] == "2026-11-01" and p["ad_delivery_date_max"] == "2026-11-03"
    assert p["media_type"] == "VIDEO" and p["ad_active_status"] == "ALL" and p["after"] == "X" and p["limit"] == 50
    assert "eu_total_reach" in p["fields"].split(",") and "access_token" not in p
    with pytest.raises(ValueError):
        maf.request_params([str(i) for i in range(11)], ["GB"], date.today(), date.today())


def test_follows_after_cursor_until_no_next(feeder, mocker):
    get = mocker.patch("requests.get", side_effect=[page([ad(1), ad(2)], after="C1"), page([ad(3)])])
    assert [i.get("source_ad_id") for i in feeder] == ["1", "2", "3"]
    first, second = (c.kwargs["params"] for c in get.call_args_list)
    assert "after" not in first and second["after"] == "C1" and first["access_token"] == TOKEN


def test_empty_page_ends(feeder, mocker):
    mocker.patch("requests.get", return_value=FakeResponse({"data": [], "paging": {"next": "x"}}))
    assert list(feeder) == []


def test_item_fields_and_token_redacted(feeder, mocker):
    mocker.patch("requests.get", return_value=page([ad(7)]))
    (item,) = list(feeder)
    assert item.get_url() == "https://www.facebook.com/ads/library/?id=7"
    assert item.get("advertiser_id") == "20531316728" and item.get("first_shown") == "2026-10-31"
    assert TOKEN not in json.dumps(item.get("raw")) and "access_token=REDACTED" in item.get("snapshot_url")
    assert item.get("media_urls") == []


def test_page_id_groups_of_ten(setup_module, mocker):
    mocker.patch("time.sleep")
    f = setup_module("meta_adlib_feeder", {"access_token": "t", "page_ids": [str(i) for i in range(23)]})
    get = mocker.patch("requests.get", return_value=page([]))
    list(f)
    assert [len(json.loads(c.kwargs["params"]["search_page_ids"])) for c in get.call_args_list] == [10, 10, 3]


@pytest.mark.parametrize(
    "status,err,code",
    [
        (
            400,
            {
                "code": 10,
                "error_subcode": 2332002,
                "type": "OAuthException",
                "message": "Application does not have permission for this action",
            },
            "10/2332002",
        ),
        (400, {"code": 190, "type": "OAuthException", "message": "Invalid OAuth access token"}, "190"),
        (400, {"code": 613, "message": "Calls to this api have exceeded the rate limit"}, "613"),
        (400, {"code": 100, "message": "Invalid parameter"}, "100"),
        (500, {"code": 1, "message": "An unknown error occurred", "is_transient": True}, "1"),
    ],
)
def test_graph_errors_carry_code(status, err, code):
    with pytest.raises(ProviderError) as e:
        maf.parse_page({"error": err}, status)
    assert e.value.code == code and e.value.status == status and not e.value.schema


@pytest.mark.parametrize(
    "payload",
    [
        {"data": {"id": 1}},
        {"paging": {}},
        {"data": [{"page_id": "1"}]},
        {"data": [{"id": "1", "page_id": "2"}], "paging": {"next": "x", "cursors": {}}},
        [1, 2],
    ],
)
def test_schema_breaks(payload):
    with pytest.raises(ProviderError) as e:
        maf.parse_page(payload)
    assert e.value.schema


def test_non_json_body(feeder, mocker):
    mocker.patch("requests.get", return_value=FakeResponse("<html>", 502))
    with pytest.raises(ProviderError) as e:
        list(feeder)
    assert e.value.status == 502
