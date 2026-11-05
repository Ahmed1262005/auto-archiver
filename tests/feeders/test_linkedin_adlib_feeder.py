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


def test_one_day_window_is_widened_to_two():
    from datetime import date

    from auto_archiver.modules.linkedin_adlib_feeder.linkedin_adlib_feeder import effective_window, request_url

    d = date(2026, 11, 4)
    assert effective_window(d, d) == (date(2026, 11, 3), d)
    assert effective_window(date(2026, 10, 1), d) == (date(2026, 10, 1), d)
    assert "start:(year:2026,month:11,day:3)" in request_url("Kering", d, d, 0, 25)


def test_one_day_feeder_window_queries_two_days(setup_module, mocker):
    mocker.patch("time.sleep")
    f = setup_module("linkedin_adlib_feeder", {"access_token": "t", "advertisers": ["Kering"], "days_back": 1})
    get = mocker.patch("requests.get", return_value=page([], 0))
    list(f)
    url = get.call_args.args[0]
    start, end = url.split("dateRange=(start:")[1].split(",end:")
    assert start != end.split(")&")[0] + ")"


@pytest.mark.parametrize(
    "total,expected",
    [
        ({"from": 1000000, "to": 0}, {"lower": 1000000, "upper": None}),
        ({"from": 1000, "to": 5000}, {"lower": 1000, "upper": 5000}),
        ({"from": 0, "to": 0}, {"lower": 0, "upper": 0}),
        (None, None),
    ],
)
def test_impressions_range(total, expected):
    from auto_archiver.modules.linkedin_adlib_feeder.linkedin_adlib_feeder import impressions_range

    assert impressions_range({"totalImpressions": total} if total else {}) == expected


def test_item_carries_open_ended_impressions():
    from auto_archiver.modules.linkedin_adlib_feeder.linkedin_adlib_feeder import LinkedInAdLibFeeder

    rec = ad(9)
    rec["details"]["adStatistics"]["totalImpressions"] = {"from": 1000000, "to": 0}
    assert LinkedInAdLibFeeder.to_metadata(rec, "PepsiCo").get("impressions") == {"lower": 1000000, "upper": None}


@pytest.mark.parametrize(
    "payload,status,schema,code",
    [
        (
            {"status": 429, "code": "TOO_MANY_REQUESTS", "message": "Resource level throttle FINDER"},
            429,
            False,
            "TOO_MANY_REQUESTS",
        ),
        ({"serviceErrorCode": 65601, "message": "The token used in the request has been revoked"}, 401, False, "65601"),
        (
            {"code": "NONEXISTENT_VERSION", "message": "Requested version 202001 is not active"},
            426,
            False,
            "NONEXISTENT_VERSION",
        ),
        ({"elements": "x", "paging": {"total": 1}}, 200, True, None),
        ({"elements": [], "paging": {}}, 200, True, None),
        ([], 200, True, None),
    ],
)
def test_parse_page_errors(payload, status, schema, code):
    from auto_archiver.modules.linkedin_adlib_feeder.linkedin_adlib_feeder import parse_page
    from auto_archiver.utils.provider_errors import ProviderError

    with pytest.raises(ProviderError) as e:
        parse_page(payload, status)
    assert (e.value.schema, e.value.code, e.value.status) == (schema, code, status)


def test_parse_page_offsets():
    from auto_archiver.modules.linkedin_adlib_feeder.linkedin_adlib_feeder import parse_page

    assert parse_page({"elements": [ad(1), ad(2)], "paging": {"total": 3}}, 200, 0)[1:] == (2, 3)
    assert parse_page({"elements": [ad(3)], "paging": {"total": 3}}, 200, 2)[1:] == (None, 3)
    # an empty page stops even when the reported total says otherwise (no infinite loop)
    assert parse_page({"elements": [], "paging": {"total": 9}}, 200, 4)[1:] == (None, 9)
