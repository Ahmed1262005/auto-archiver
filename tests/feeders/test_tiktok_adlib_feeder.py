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
    assert (second_body["search_id"], second_body["cursor"]) == ("s1", 50)
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
