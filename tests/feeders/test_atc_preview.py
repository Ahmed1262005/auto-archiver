import json
import pathlib

import pytest

from auto_archiver.modules.youtube_feeder import atc_preview as atc

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
ADV, CR = "AR10307956454253920257", "CR07855827554723168257"  # recorded 2026-11-07


def test_recorded_creative_and_vast():
    previews = atc.parse_creative(json.loads((DATA / "atc_creative_CR07855827554723168257.json").read_text()))
    assert previews and previews[0].startswith("https://displayads-formats.googleusercontent.com/")
    p = atc.parse_preview((DATA / "atc_preview_vast.js").read_text())
    assert [f["type"] for f in p["media_files"]] == ["video/mp4"]
    assert "googlevideo.com/videoplayback" in p["media_files"][0]["url"]


def test_lookup_merges_variations():
    js_id = r'"https:\x2F\x2Fi.ytimg.com\x2Fvi\x2FK5Xwi6M-hyQ\x2Fhqdefault.jpg"'
    two = {"1": {"5": [{"1": {"4": "https://displayads-formats.googleusercontent.com/a"}}] * 2}}
    out = atc.lookup(lambda url, form: two, lambda url: js_id, ADV, CR)
    assert out["youtube_ids"] == ["K5Xwi6M-hyQ"] and out["previews"] == 2 and out["media_files"] == []


@pytest.mark.parametrize("payload,kind", [({}, "not_found"), ({"1": {}}, "schema"), ([], "schema")])
def test_bad_creatives(payload, kind):
    with pytest.raises(atc.AtcPreviewError) as e:
        atc.parse_creative(payload)
    assert e.value.kind == kind


def test_ids():
    url = f"https://adstransparency.google.com/advertiser/{ADV}/creative/{CR}?region=GB"
    assert atc.ids_from_url(url) == (ADV, CR)
    with pytest.raises(atc.AtcPreviewError):
        atc.creative_form("AR1", CR)
