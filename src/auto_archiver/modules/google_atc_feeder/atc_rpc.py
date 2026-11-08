"""Google Ads Transparency Center web RPCs: the JSON endpoints adstransparency.google.com itself calls
(`/anji/_/rpc/SearchService/SearchCreatives`, `/anji/_/rpc/LookupService/GetCreativeById`) and the creative
previews they link to (`displayads-formats.googleusercontent.com/ads/preview/content.js`, `tpc.googlesyndication.com`
images).

There is no official API for creatives: the BigQuery dataset has metadata only. These endpoints are undocumented and
can change or be throttled at any time; Google answers a bursty client with HTTP 429 or a redirect to /sorry/ (a
CAPTCHA page), which is reported as BLOCKED and never solved. Field numbers follow two open-source clients:
faniAhmed/GoogleAdsTransparencyScraper and block-town/google-ads-transparency-mcp (MIT, Copyright (c) 2025 Sam Town;
REGION_IDS and the overlay decoder are adapted from it).

Terms: Google's Terms of Service forbid "using automated means to access content from any of our services in violation
of the machine-readable instructions on our web pages (for example, robots.txt files ...)" and "bypassing our systems
or protective measures". adstransparency.google.com, displayads-formats.googleusercontent.com and
tpc.googlesyndication.com serve no robots.txt (HTTP 404, checked 2026-11-07), so the client stays polite (one request
per second by default) and stops on a block instead of working around it.
"""

import base64
import gzip
import html
import json
import re
import time
from datetime import UTC, date, datetime
from urllib.parse import parse_qs, urlparse

import requests

BASE = "https://adstransparency.google.com"
SEARCH = "SearchService/SearchCreatives"
LOOKUP = "LookupService/GetCreativeById"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

FORMATS = {1: "TEXT", 2: "IMAGE", 3: "VIDEO"}
FORMAT_CODES = {v: k for k, v in FORMATS.items()}

# ISO 3166-1 alpha-2 -> Google Ads geo criterion id, as the RPCs use them (2000 + ISO numeric for countries)
REGION_IDS = {
    "AL": 2008,
    "DZ": 2012,
    "AS": 2016,
    "AD": 2020,
    "AO": 2024,
    "AQ": 2010,
    "AG": 2028,
    "AR": 2032,
    "AM": 2051,
    "AU": 2036,
    "AT": 2040,
    "AZ": 2031,
    "BH": 2048,
    "BD": 2050,
    "BB": 2052,
    "BY": 2112,
    "BE": 2056,
    "BZ": 2084,
    "BJ": 2204,
    "BT": 2064,
    "BO": 2068,
    "BA": 2070,
    "BW": 2072,
    "BR": 2076,
    "BN": 2096,
    "BG": 2100,
    "BF": 2854,
    "BI": 2108,
    "KH": 2116,
    "CM": 2120,
    "CA": 2124,
    "CV": 2132,
    "BQ": 2535,
    "CF": 2140,
    "TD": 2148,
    "CL": 2152,
    "CN": 2156,
    "CX": 2162,
    "CC": 2166,
    "CO": 2170,
    "KM": 2174,
    "CK": 2184,
    "CR": 2188,
    "CI": 2384,
    "HR": 2191,
    "CW": 2531,
    "CY": 2196,
    "CZ": 2203,
    "CD": 2180,
    "DK": 2208,
    "DJ": 2262,
    "DM": 2212,
    "DO": 2214,
    "EC": 2218,
    "EG": 2818,
    "SV": 2222,
    "GQ": 2226,
    "ER": 2232,
    "EE": 2233,
    "SZ": 2748,
    "ET": 2231,
    "FM": 2583,
    "FJ": 2242,
    "FI": 2246,
    "FR": 2250,
    "PF": 2258,
    "TF": 2260,
    "GA": 2266,
    "GE": 2268,
    "DE": 2276,
    "GH": 2288,
    "GR": 2300,
    "GD": 2308,
    "GU": 2316,
    "GT": 2320,
    "GG": 2831,
    "GN": 2324,
    "GW": 2624,
    "GY": 2328,
    "HT": 2332,
    "HM": 2334,
    "HN": 2340,
    "HU": 2348,
    "IS": 2352,
    "IN": 2356,
    "ID": 2360,
    "IQ": 2368,
    "IE": 2372,
    "IL": 2376,
    "IT": 2380,
    "JM": 2388,
    "JP": 2392,
    "JE": 2832,
    "JO": 2400,
    "KZ": 2398,
    "KE": 2404,
    "KI": 2296,
    "KW": 2414,
    "KG": 2417,
    "LA": 2418,
    "LV": 2428,
    "LB": 2422,
    "LS": 2426,
    "LR": 2430,
    "LY": 2434,
    "LI": 2438,
    "LT": 2440,
    "LU": 2442,
    "MG": 2450,
    "MW": 2454,
    "MY": 2458,
    "MV": 2462,
    "MT": 2470,
    "ML": 2466,
    "MH": 2584,
    "MR": 2478,
    "MU": 2480,
    "MX": 2484,
    "MD": 2498,
    "MC": 2492,
    "MN": 2496,
    "ME": 2499,
    "MA": 2504,
    "MZ": 2508,
    "MM": 2104,
    "NA": 2516,
    "NR": 2520,
    "NP": 2524,
    "NL": 2528,
    "NC": 2540,
    "NZ": 2554,
    "NI": 2558,
    "NE": 2562,
    "NG": 2566,
    "NU": 2570,
    "NF": 2574,
    "MK": 2807,
    "MP": 2580,
    "NO": 2578,
    "OM": 2512,
    "PK": 2586,
    "PW": 2585,
    "PA": 2591,
    "PG": 2598,
    "PY": 2600,
    "PE": 2604,
    "PH": 2608,
    "PN": 2612,
    "PL": 2616,
    "PT": 2620,
    "QA": 2634,
    "CG": 2178,
    "RO": 2642,
    "RW": 2646,
    "SH": 2654,
    "KN": 2659,
    "LC": 2662,
    "PM": 2666,
    "VC": 2670,
    "WS": 2882,
    "SM": 2674,
    "ST": 2678,
    "SA": 2682,
    "SN": 2686,
    "SC": 2690,
    "SL": 2694,
    "SG": 2702,
    "SX": 2534,
    "SK": 2703,
    "SI": 2705,
    "SB": 2090,
    "SO": 2706,
    "ZA": 2710,
    "GS": 2239,
    "KR": 2410,
    "ES": 2724,
    "LK": 2144,
    "SR": 2740,
    "CH": 2756,
    "SE": 2752,
    "TJ": 2762,
    "TZ": 2834,
    "TH": 2764,
    "BS": 2044,
    "GM": 2270,
    "TL": 2626,
    "TG": 2768,
    "TK": 2772,
    "TO": 2776,
    "TT": 2780,
    "TN": 2788,
    "TR": 2792,
    "TM": 2795,
    "TV": 2798,
    "UG": 2800,
    "UA": 2804,
    "AE": 2784,
    "GB": 2826,
    "US": 2840,
    "UY": 2858,
    "UZ": 2860,
    "VU": 2548,
    "VA": 2336,
    "VE": 2862,
    "VN": 2704,
    "YE": 2887,
    "ZM": 2894,
    "ZW": 2716,
    "BM": 2060,
    "BV": 2074,
    "IO": 2086,
    "VG": 2092,
    "KY": 2136,
    "TW": 2158,
    "YT": 2175,
    "FO": 2234,
    "FK": 2238,
    "GF": 2254,
    "PS": 2275,
    "GI": 2292,
    "GL": 2304,
    "GP": 2312,
    "HK": 2344,
    "IM": 20340,
    "MO": 2446,
    "MQ": 2474,
    "MS": 2500,
    "AW": 2533,
    "PR": 2630,
    "RE": 2638,
    "AI": 2660,
    "EH": 2732,
    "SJ": 2744,
    "TC": 2796,
    "UM": 2581,
    "VI": 2850,
    "WF": 2876,
}
GEO_REGIONS = {v: k for k, v in REGION_IDS.items()}

BLOCKED = "blocked"  # 429 or the /sorry/ CAPTCHA page: back off, never solve it
NOT_FOUND = "not_found"  # the creative or asset is gone
SCHEMA = "schema"  # the response no longer has the shape this module parses
TRANSIENT = "transient"  # network error or 5xx
HTTP_ERROR = "http_error"  # any other 4xx


class AtcError(Exception):
    def __init__(self, kind: str, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind, self.status, self.retry_after = kind, status, retry_after


def creative_page_url(advertiser_id: str, creative_id: str) -> str:
    return f"{BASE}/advertiser/{advertiser_id}/creative/{creative_id}"


def parse_page_url(url: str) -> tuple[str, str] | None:
    """(advertiser_id, creative_id) from a Transparency Center creative page URL, e.g. BigQuery's creative_page_url."""
    m = re.search(r"/advertiser/(AR\d+)/creative/(CR\d+)", url or "")
    return (m.group(1), m.group(2)) if m else None


def _date_from_ts(obj) -> str | None:
    try:
        return datetime.fromtimestamp(int(obj["1"]), UTC).date().isoformat()
    except (KeyError, TypeError, ValueError):
        return None


def _date_from_int(n) -> str | None:
    try:
        s = str(int(n))
        return date(int(s[:4]), int(s[4:6]), int(s[6:8])).isoformat()
    except (TypeError, ValueError):
        return None


def _yyyymmdd(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def preview_link(variant: dict) -> str:
    """The preview of one creative variant: an <img> (image ads and rendered text ads) or a content.js renderer."""
    if not isinstance(variant, dict):
        return ""
    raw = (variant.get("3") or {}).get("2") if isinstance(variant.get("3"), dict) else None
    if isinstance(raw, str) and raw:
        m = re.search(r'src="([^"]+)"', raw)
        return html.unescape(m.group(1)) if m else raw
    for path in (("1", "4"), ("2", "4"), ("4",)):
        obj = variant
        for k in path:
            obj = obj.get(k) if isinstance(obj, dict) else None
        if isinstance(obj, str) and obj:
            return obj
    return ""


def parse_search_item(item: dict) -> dict:
    if not isinstance(item, dict) or not str(item.get("2", "")).startswith("CR"):
        raise AtcError(SCHEMA, f"search item without a creative id: {str(item)[:200]}")
    adv, cr = item.get("1", ""), item["2"]
    return {
        "advertiser_id": adv,
        "creative_id": cr,
        "advertiser_name": item.get("12") or "",
        "format": FORMATS.get(item.get("4"), "UNKNOWN"),
        "first_shown": _date_from_ts(item.get("6")),
        "last_shown": _date_from_ts(item.get("7")),
        "preview_url": preview_link(item.get("3") or {}),
        "creative_page_url": creative_page_url(adv, cr),
    }


def parse_creative(body: dict) -> dict:
    """A GetCreativeById answer: format, last shown, every variant's preview and the last-shown date per region."""
    ad = (body or {}).get("1")
    if not isinstance(ad, dict):
        raise AtcError(NOT_FOUND, "GetCreativeById returned no creative")
    adv, cr = ad.get("1", ""), ad.get("2", "")
    regions = []
    for r in ad.get("17") or []:
        gid = r.get("1")
        regions.append({"region": GEO_REGIONS.get(gid, f"geo:{gid}"), "last_shown": _date_from_int(r.get("5"))})
    return {
        "advertiser_id": adv,
        "creative_id": cr,
        "advertiser_name": (ad.get("22") or {}).get("1") or "",
        "format": FORMATS.get(ad.get("8"), "UNKNOWN"),
        "last_shown": _date_from_ts(ad.get("4")),
        "previews": [p for p in (preview_link(v) for v in ad.get("5") or []) if p],
        "regions": regions,
        "creative_page_url": creative_page_url(adv, cr),
    }


_DATA_P_COPY = re.compile(r'data-p="%\.@\.\[1,\[&quot;(.*?)&quot;,&quot;(.*?)&quot;,&quot;(.*?)&quot;')
_YT_ID = r"([A-Za-z0-9_-]{11})"
_YT_PATTERNS = [
    re.compile(r"ytimg\.com/vi(?:_webp)?/" + _YT_ID),
    re.compile(r"youtube(?:-nocookie)?\.com/(?:embed/|watch\?v=)" + _YT_ID),
    re.compile(r"['\"]video_?id['\"]\s*:\s*['\"]" + _YT_ID, re.I),
]
_GOOGLEVIDEO = re.compile(r"https://[^\s'\"\]]*googlevideo\.com/videoplayback[^\s'\"\]]*")


def parse_preview_body(body: str) -> dict:
    """Assets inside a content.js renderer: YouTube video ids, simgad image ids, a direct video stream, ad copy."""
    text = re.sub(r"\\+x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), body)
    text = re.sub(r"\\+u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    out: dict = {}
    ids: list[str] = []
    for pat in _YT_PATTERNS:
        for vid in pat.findall(text):
            if vid not in ids:
                ids.append(vid)
    if ids:
        out["youtube_video_ids"] = ids
    simgad = list(dict.fromkeys(re.findall(r"simgad/(\d+)", text)))
    if simgad:
        out["simgad_ids"] = simgad
    stream = _GOOGLEVIDEO.search(text)
    if stream:
        out["video_url"] = stream.group(0).replace("\\/", "/")
    copy = _DATA_P_COPY.search(text)
    if copy:
        headline, description, visurl = (html.unescape(g) for g in copy.groups())
        out.update(
            {k: v for k, v in (("headline", headline), ("description", description), ("display_url", visurl)) if v}
        )
    return out


def _varint(data: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while i < len(data):
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7
    raise ValueError("truncated varint")


def _fields(data: bytes):
    i = 0
    while i < len(data):
        key, i = _varint(data, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(data, i)
        elif wire == 2:
            n, i = _varint(data, i)
            if i + n > len(data):
                raise ValueError("truncated field")
            value, i = data[i : i + n], i + n
        elif wire == 5:
            value, i = data[i : i + 4], i + 4
        elif wire == 1:
            value, i = data[i : i + 8], i + 8
        else:
            raise ValueError(f"wire type {wire}")
        yield field, wire, value


def decode_overlay(url: str) -> dict:
    """The `overlay=` parameter of a content.js URL: urlsafe base64 -> gzip -> protobuf whose repeated field 9 holds
    {name, {value}} pairs (e.g. headline, description, visurl). {} when absent or undecodable."""
    try:
        blob = (parse_qs(urlparse(url).query).get("overlay") or [""])[0].lstrip("=")
        if not blob:
            return {}
        data = gzip.decompress(base64.urlsafe_b64decode(blob + "=" * (-len(blob) % 4)))
        out = {}
        for field, wire, value in _fields(data):
            if field != 9 or wire != 2:
                continue
            name = val = ""
            for f2, w2, v2 in _fields(value):
                if w2 != 2:
                    continue
                if f2 == 1:
                    name = v2.decode("utf-8", "replace")
                elif f2 == 2:
                    val = next((v3.decode("utf-8", "replace") for f3, w3, v3 in _fields(v2) if f3 == 1 and w3 == 2), "")
            if name:
                out[name] = val
        return out
    except (ValueError, OSError, EOFError):
        return {}


class Client:
    """One cookie session against the Transparency Center, at most one request per `min_interval` seconds.
    `on_request(kind)` is called before every request ("home", "rpc", "preview", "asset") so callers can meter them."""

    def __init__(
        self,
        session: requests.Session | None = None,
        *,
        min_interval: float = 1.0,
        timeout: float = 20,
        on_request=None,
        sleep=time.sleep,
        clock=time.monotonic,
    ):
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval, self.timeout, self.on_request = min_interval, timeout, on_request
        self._sleep, self._clock, self._last, self._ready = sleep, clock, None, False

    def _send(self, kind: str, method: str, url: str, **kw) -> requests.Response:
        if self._last is not None:
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        if self.on_request:
            self.on_request(kind)
        self._last = self._clock()
        try:
            r = self.session.request(method, url, timeout=self.timeout, **kw)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise AtcError(TRANSIENT, f"{url[:120]}: {e}") from e
        except requests.RequestException as e:
            raise AtcError(HTTP_ERROR, f"{url[:120]}: {e}") from e
        if _blocked(r):
            ra = r.headers.get("retry-after")
            raise AtcError(
                BLOCKED,
                f"HTTP {r.status_code} {r.url[:120]}",
                r.status_code,
                float(ra) if ra and ra.isdigit() else None,
            )
        if r.status_code in (404, 410):
            raise AtcError(NOT_FOUND, f"HTTP {r.status_code} {url[:120]}", r.status_code)
        if r.status_code >= 500 or r.status_code == 408:
            raise AtcError(TRANSIENT, f"HTTP {r.status_code} {url[:120]}", r.status_code)
        if r.status_code >= 400:
            raise AtcError(HTTP_ERROR, f"HTTP {r.status_code} {url[:120]}", r.status_code)
        return r

    def _cookies(self) -> None:
        if not self._ready:
            self._send("home", "GET", f"{BASE}/", params={"region": "anywhere"})
            self._ready = True

    def rpc(self, method: str, body: dict, authuser: str = "") -> dict:
        self._cookies()
        r = self._send(
            "rpc",
            "POST",
            f"{BASE}/anji/_/rpc/{method}",
            params={"authuser": authuser},
            data={"f.req": json.dumps(body)},
        )
        try:
            out = r.json()
        except ValueError as e:
            raise AtcError(SCHEMA, f"{method} answered non-JSON: {r.text[:200]!r}", r.status_code) from e
        if not isinstance(out, dict):
            raise AtcError(SCHEMA, f"{method} answered {type(out).__name__}, not an object")
        return out

    def search_creatives(
        self,
        advertiser_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
        region: str | None = None,
        fmt: str | None = None,
        page_size: int = 100,
        max_items: int | None = None,
        max_pages: int = 50,
    ) -> dict:
        """Creatives of one advertiser shown in [start, end] (whole days, UTC), newest first, following page tokens.
        `total` is the site's own (lower, upper) estimate; `truncated` is True when max_items/max_pages stopped paging."""
        q: dict = {"12": {"1": "", "2": True}, "13": {"1": [advertiser_id]}}
        if start:
            q["6"] = _yyyymmdd(start)
        if end:
            q["7"] = _yyyymmdd(end)
        if region:
            if region not in REGION_IDS:
                raise AtcError(HTTP_ERROR, f"unknown region {region}")
            q["8"] = [REGION_IDS[region]]
        if fmt:
            q["4"] = FORMAT_CODES[fmt]
        items, token, pages, total = [], None, 0, None
        while True:
            body = {"2": min(page_size, 100), "3": q, "7": {"1": 1}}
            if token:
                body["4"] = token
            res = self.rpc(SEARCH, body)
            pages += 1
            if total is None and ("4" in res or "5" in res):
                total = (int(res.get("4") or 0), int(res.get("5") or res.get("4") or 0))
            items.extend(parse_search_item(i) for i in res.get("1") or [])
            token = res.get("2")
            if not token or not res.get("1"):
                return {"items": items, "total": total, "pages": pages, "truncated": False}
            if (max_items and len(items) >= max_items) or pages >= max_pages:
                return {"items": items[: max_items or None], "total": total, "pages": pages, "truncated": True}

    def get_creative(self, advertiser_id: str, creative_id: str) -> dict:
        return parse_creative(self.rpc(LOOKUP, {"1": advertiser_id, "2": creative_id, "5": {"1": 1}}, authuser="0"))

    def preview(self, url: str) -> dict:
        """What a preview link holds: image_url for an image, youtube_video_ids / video_url / copy for content.js."""
        if "content.js" not in url:
            return {"image_url": url} if url else {}
        out = parse_preview_body(self._send("preview", "GET", url).text)
        overlay = decode_overlay(url)
        if overlay:
            out["overlay"] = overlay
        ids = out.pop("simgad_ids", [])
        if ids:
            img = self.simgad_url(ids[0])
            if img:
                out["image_url"] = img
        return out

    def simgad_url(self, simgad_id: str) -> str | None:
        """Google files images under /simgad/<id> or /archive/simgad/<id> and 404s the other one."""
        for form in ("archive/simgad", "simgad"):
            url = f"https://tpc.googlesyndication.com/{form}/{simgad_id}"
            try:
                self._send("asset", "HEAD", url, allow_redirects=True)
                return url
            except AtcError as e:
                if e.kind != NOT_FOUND:
                    raise
        return None


def _blocked(r: requests.Response) -> bool:
    if r.status_code == 429 or "/sorry/" in r.url:
        return True
    if any("/sorry/" in h.headers.get("location", "") for h in r.history):
        return True
    ctype = r.headers.get("content-type", "")
    return "html" in ctype and "recaptcha" in r.text[:4000].lower()
