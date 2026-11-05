import pytest

from auto_archiver.core.consts import SetupError


def ad(n, name="PepsiCo Away from Home"):
    return {
        "isRestricted": False,
        "adUrl": f"https://www.linkedin.com/ad-library/detail/{n}",
        "details": {
            "type": "SPONSORED_VIDEO",
            "advertiser": {
                "adPayer": "Omne Agency",
                "advertiserName": name,
                "advertiserUrl": "https://www.linkedin.com/company/77820429",
            },
            "adStatistics": {"firstImpressionAt": 1793000000000, "latestImpressionAt": None},
        },
    }


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


def page(elements, total):
    return FakeResponse({"elements": elements, "paging": {"start": 0, "count": len(elements), "total": total}})


@pytest.fixture
def feeder(setup_module, mocker):
    mocker.patch("time.sleep")
    return setup_module("linkedin_adlib_feeder", {"access_token": "t", "advertisers": ["PepsiCo"], "count": 2})


def test_setup_requires_advertisers_and_token(setup_module):
    with pytest.raises(SetupError):
        setup_module("linkedin_adlib_feeder", {"access_token": "t"})
    with pytest.raises(SetupError):
        setup_module("linkedin_adlib_feeder", {"advertisers": ["PepsiCo"]})


def test_pages_until_total(feeder, mocker):
    get = mocker.patch("requests.get", side_effect=[page([ad(1), ad(2)], 3), page([ad(3)], 3)])
    assert [i.get("source_ad_id") for i in feeder] == ["1", "2", "3"]
    first, second = (c.args[0] for c in get.call_args_list)
    assert "advertiser=PepsiCo" in first and "&start=0&count=2" in first
    assert "&start=2&count=2" in second
    assert "dateRange=(start:(year:" in first


def test_item_fields(feeder, mocker):
    mocker.patch("requests.get", return_value=page([ad(1624254)], 1))
    (item,) = list(feeder)
    assert item.get_url() == "https://www.linkedin.com/ad-library/detail/1624254"
    assert item.get("advertiser_id") == "77820429"
    assert item.get("advertiser_name") == "PepsiCo Away from Home"
    assert item.get("advertiser_query") == "PepsiCo"
    assert item.get("ad_payer") == "Omne Agency"
    assert item.get("media_urls") == []
    assert item.get("raw") == ad(1624254)


def test_api_error_raises_not_empty(feeder, mocker):
    mocker.patch("requests.get", return_value=FakeResponse({"message": "throttled"}, status=429))
    with pytest.raises(RuntimeError, match="429"):
        list(feeder)
