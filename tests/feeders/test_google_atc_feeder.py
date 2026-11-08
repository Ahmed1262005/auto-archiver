from datetime import date

import pytest

from auto_archiver.core.consts import SetupError

ROW = {
    "creative_id": "CR123",
    "advertiser_id": "AR10307956454253920257",
    "advertiser_disclosed_name": "Coca-Cola Europacific Partners",
    "advertiser_legal_name": "COCA-COLA EUROPACIFIC PARTNERS GREAT BRITAIN LIMITED",
    "ad_format_type": "VIDEO",
    "creative_page_url": "https://adstransparency.google.com/advertiser/AR10307956454253920257/creative/CR123",
    "first_shown": date(2026, 10, 1),
    "last_shown": date(2026, 11, 4),
    "regions": [{"region_code": "GB", "first_shown": "2026-10-01", "last_shown": "2026-11-04"}],
}


@pytest.fixture
def bq(mocker):
    client = mocker.MagicMock()
    client.query.return_value.result.return_value = [ROW]
    client.query.return_value.total_bytes_billed = 0
    mocker.patch("auto_archiver.modules.google_atc_feeder.google_atc_feeder.bigquery.Client", return_value=client)
    return client


def test_setup_requires_advertiser_ids(setup_module):
    with pytest.raises(SetupError):
        setup_module("google_atc_feeder", {})


def test_one_query_for_all_advertisers_with_byte_cap(setup_module, bq):
    feeder = setup_module("google_atc_feeder", {"advertiser_ids": ["AR1", "AR2"], "maximum_gb_billed": 60})
    items = list(feeder)
    assert len(items) == 1
    bq.query.assert_called_once()
    config = bq.query.call_args.kwargs["job_config"]
    assert config.maximum_bytes_billed == 60 * 10**9
    params = {p.name: p for p in config.query_parameters}
    assert params["advertiser_ids"].values == ["AR1", "AR2"]
    assert params["max_rows_per_advertiser"].value == 2000
    assert "PARTITION BY advertiser_id" in bq.query.call_args.args[0]


def test_item_fields(setup_module, bq):
    (item,) = list(setup_module("google_atc_feeder", {"advertiser_ids": ["AR1"]}))
    assert item.get_url() == ROW["creative_page_url"]
    assert item.get("source_ad_id") == "CR123"
    assert item.get("advertiser_name") == "Coca-Cola Europacific Partners"
    assert item.get("last_shown") == "2026-11-04"
    assert item.get("regions")[0]["region_code"] == "GB"
    assert item.get("media_urls") == []


def test_fetch_page_window_and_capped(bq):
    from auto_archiver.modules.google_atc_feeder.google_atc_feeder import fetch_page

    bq.query.return_value.result.return_value = [ROW, ROW | {"creative_id": "CR2"}, ROW | {"advertiser_id": "AR2"}]
    bq.query.return_value.total_bytes_billed = 55 * 10**9
    page = fetch_page(bq, ["AR10307956454253920257", "AR2"], date(2026, 11, 4), date(2026, 11, 4), 2, 60 * 10**9)
    assert page["capped"] == ["AR10307956454253920257"]
    assert page["bytes_billed"] == 55 * 10**9 and len(page["rows"]) == 3
    params = {
        p.name: p.value for p in bq.query.call_args.kwargs["job_config"].query_parameters if p.name != "advertiser_ids"
    }
    assert params == {"start": date(2026, 11, 4), "end": date(2026, 11, 4), "max_rows_per_advertiser": 2}
    assert "@start" in bq.query.call_args.args[0] and "@end" in bq.query.call_args.args[0]


def test_quota_error_propagates(bq):
    from google.api_core.exceptions import Forbidden

    from auto_archiver.modules.google_atc_feeder.google_atc_feeder import fetch_page

    bq.query.return_value.result.side_effect = Forbidden("Custom quota exceeded: QueryUsagePerDay")
    with pytest.raises(Forbidden):
        fetch_page(bq, ["AR1"], date(2026, 11, 4), date(2026, 11, 4), 2000, 60 * 10**9)
