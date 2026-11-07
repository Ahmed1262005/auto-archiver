import json
import pathlib

from auto_archiver.modules.youtube_feeder import youtube_ytdlp_info as yi

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def test_to_resource_recorded_info():
    info = json.loads((DATA / "ytdlp_info_q2-Blj7xvgw.json").read_text())  # yt-dlp 2026.10.26
    r = yi.to_resource(info)
    assert r["id"] == "q2-Blj7xvgw" and r["snippet"]["channelId"] == "UCosXctaTYxN4YPIvI5Fpcrw"
    assert r["contentDetails"]["duration"] == "PT15S" and r["status"]["privacyStatus"] == "public"


def test_iso_duration():
    assert yi._iso_duration(3725) == "PT1H2M5S" and yi._iso_duration(None) == "PT0S"
