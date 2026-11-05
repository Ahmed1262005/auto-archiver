from datetime import datetime, timezone

import pytest

from auto_archiver.core.consts import SetupError

AD = {
    "advertiser": {"business_id": 111, "business_name": "GUCCIO GUCCI SPA"},
    "ad": {
        "id": 42,
        "status": "active",
        "first_shown_date": "20261102",
        "last_shown_date": "20261104",
        "videos": [{"url": "https://v77.tiktokcdn.com/v.mp4", "cover_image_url": "https://p16/c.jpg"}],
        "image_urls": ["https://p16/i.jpeg"],
    },
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.content = b"x"

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def page(ads, has_more=False, cursor=None):
    return FakeResponse(
        {"data": {"ads": ads, "has_more": has_more, "search_id": "s1", "cursor": cursor}, "error": {"code": "ok"}}
    )


@pytest.fixture
def feeder(setup_module, mocker):
    mocker.patch("time.sleep")
    return setup_module("tiktok_adlib_feeder", {"access_token": "t", "advertiser_business_ids": [111]})


def test_setup_requires_ids_and_credentials(setup_module):
    with pytest.raises(SetupError):
        setup_module("tiktok_adlib_feeder", {"access_token": "t"})
    with pytest.raises(SetupError):
        setup_module("tiktok_adlib_feeder", {"advertiser_business_ids": [1]})


def test_yields_one_item_per_ad_across_pages(feeder, mocker):
    post = mocker.patch("requests.post", side_effect=[page([AD], True, 50), page([AD | {"ad": AD["ad"] | {"id": 43}}])])
    items = list(feeder)
    assert [i.get("source_ad_id") for i in items] == ["42", "43"]
    second_body = post.call_args_list[1].kwargs["json"]
    assert second_body["search_id"] == "s1" and "cursor" not in second_body
    assert second_body["filters"]["advertiser_business_ids"] == [111]
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    assert second_body["filters"]["ad_published_date_range"]["max"] < today


def test_item_fields(feeder, mocker):
    mocker.patch("requests.post", return_value=page([AD]))
    (item,) = list(feeder)
    assert item.get_url() == "https://library.tiktok.com/ads/detail/?ad_id=42"
    assert item.get("advertiser_id") == "111"
    assert item.get("advertiser_name") == "GUCCIO GUCCI SPA"
    assert item.get("media_urls") == ["https://v77.tiktokcdn.com/v.mp4", "https://p16/i.jpeg"]
    assert item.get("raw") == AD


def test_api_error_raises_not_empty(feeder, mocker):
    mocker.patch("requests.post", return_value=FakeResponse({"error": {"code": "rate_limit_exceeded"}}, status=429))
    with pytest.raises(RuntimeError, match="429"):
        list(feeder)


def test_fetches_token_when_missing(setup_module, mocker):
    f = setup_module("tiktok_adlib_feeder", {"client_key": "k", "client_secret": "s", "advertiser_business_ids": [111]})
    post = mocker.patch("requests.post", side_effect=[FakeResponse({"access_token": "fresh"}), page([])])
    assert list(f) == []
    assert post.call_args_list[1].kwargs["headers"]["Authorization"] == "Bearer fresh"


def test_has_more_without_search_id_is_a_schema_error(feeder, mocker):
    from auto_archiver.utils.provider_errors import ProviderError

    bad = FakeResponse({"data": {"ads": [AD], "has_more": True}, "error": {"code": "ok"}})
    mocker.patch("requests.post", return_value=bad)
    with pytest.raises(ProviderError) as e:
        list(feeder)
    assert e.value.schema


@pytest.mark.parametrize(
    "payload,status,schema,code",
    [
        ({"error": {"code": "internal_error", "message": "x"}}, 500, False, "internal_error"),
        ({"error": {"code": "access_token_invalid"}}, 401, False, "access_token_invalid"),
        ({"data": {}, "error": {"code": "invalid_params", "message": "date"}}, 200, False, "invalid_params"),
        ({"error": "invalid_client", "error_description": "bad key"}, 401, False, "invalid_client"),
        ({"data": {"ads": "nope"}, "error": {"code": "ok"}}, 200, True, None),
        (["not", "an", "object"], 200, True, None),
    ],
)
def test_parse_page_errors(payload, status, schema, code):
    from auto_archiver.modules.tiktok_adlib_feeder.tiktok_adlib_feeder import parse_page
    from auto_archiver.utils.provider_errors import ProviderError

    with pytest.raises(ProviderError) as e:
        parse_page(payload, status)
    assert (e.value.schema, e.value.code) == (schema, code)


def test_parse_page_last_page_and_body():
    from datetime import date

    from auto_archiver.modules.tiktok_adlib_feeder.tiktok_adlib_feeder import parse_page, query_body

    assert parse_page({"data": {"ads": [AD], "has_more": False, "search_id": "s9"}, "error": {"code": "ok"}}) == (
        [AD],
        None,
    )
    body = query_body(111, date(2026, 11, 1), date(2026, 11, 4), "FR", 50, "s9")
    assert body["search_id"] == "s9" and body["filters"]["country_code"] == "FR" and "cursor" not in body


def test_non_json_body_raises(feeder, mocker):
    from auto_archiver.utils.provider_errors import ProviderError

    r = FakeResponse(None, status=502)
    r.json = lambda: (_ for _ in ()).throw(ValueError("html"))
    mocker.patch("requests.post", return_value=r)
    with pytest.raises(ProviderError, match="502"):
        list(feeder)
