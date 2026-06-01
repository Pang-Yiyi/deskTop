"""Optional external API integrations for weather and routing.

All functions are safe for demos: they return structured fallback results when
API keys, network access, or external services are unavailable.
"""

from __future__ import annotations

import json
import os
import ssl
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

# Windows 常見 SSL 根憑證問題：建立不驗證憑證的 context（僅限本機展示）
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

from env_loader import load_local_env


load_local_env()


DISTRICT_CENTERS = {
    "中區": (24.1417, 120.6806),
    "東區": (24.1372, 120.6970),
    "南區": (24.1211, 120.6657),
    "西區": (24.1436, 120.6634),
    "北區": (24.1587, 120.6820),
    "西屯區": (24.1816, 120.6466),
    "南屯區": (24.1408, 120.6177),
    "北屯區": (24.1822, 120.6864),
    "豐原區": (24.2521, 120.7224),
    "大里區": (24.0994, 120.6778),
    "太平區": (24.1268, 120.7187),
    "烏日區": (24.1045, 120.6233),
    "霧峰區": (24.0471, 120.7002),
    "后里區": (24.3096, 120.7100),
    "石岡區": (24.2761, 120.7808),
    "東勢區": (24.2587, 120.8309),
    "和平區": (24.1744, 121.1402),
    "新社區": (24.2341, 120.8096),
    "潭子區": (24.2118, 120.7031),
    "大雅區": (24.2252, 120.6478),
    "神岡區": (24.2578, 120.6612),
    "大肚區": (24.1535, 120.5434),
    "沙鹿區": (24.2370, 120.5619),
    "龍井區": (24.2006, 120.5280),
    "梧棲區": (24.2556, 120.5313),
    "清水區": (24.2684, 120.5740),
    "大甲區": (24.3472, 120.6244),
    "外埔區": (24.3350, 120.6547),
    "大安區": (24.3651, 120.5870),
}


def fetch_cwa_weather_tool(district: str | None = None) -> dict[str, Any]:
    """Fetch current CWA weather observation when API key is configured."""
    api_key = os.getenv("CWA_API_KEY") or os.getenv("CWA_AUTHORIZATION")
    if not api_key:
        return {
            "source": "fallback",
            "available": False,
            "district": district,
            "summary": "尚未設定中央氣象署 API key，因此目前使用歷史天候欄位進行風險分析。",
            "data_gap": "設定 CWA_API_KEY 後可顯示台中即時天氣觀測。",
        }

    params = {
        "Authorization": api_key,
        "format": "JSON",
        "StationName": "臺中",
    }
    url = "https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001?" + urlencode(params)
    try:
        payload = _get_json(url)
        station = _first_station(payload)
    except Exception as exc:
        return {
            "source": "cwa",
            "available": False,
            "district": district,
            "summary": f"中央氣象署 API 暫時無法取得資料：{exc}",
            "data_gap": "展示時可先使用歷史天候欄位；網路或 API key 正常後會自動顯示即時天氣。",
        }

    weather = _extract_weather(station)
    return {
        "source": "cwa",
        "available": True,
        "district": district,
        "station": station.get("StationName") or station.get("stationName"),
        "weather": weather,
        "summary": _format_weather_summary(weather),
    }


def fetch_osrm_route_tool(
    origin_district: str | None,
    destination_district: str | None,
) -> dict[str, Any]:
    """Fetch a route summary from public OSRM using district center points."""
    if not origin_district or not destination_district:
        return {
            "source": "fallback",
            "available": False,
            "summary": "需指定起點與終點行政區後，才能查詢 OSRM 路線。",
        }

    origin = DISTRICT_CENTERS.get(origin_district)
    destination = DISTRICT_CENTERS.get(destination_district)
    if not origin or not destination:
        return {
            "source": "fallback",
            "available": False,
            "summary": "目前沒有此行政區中心點座標，無法查詢 OSRM 路線。",
        }

    url = _build_osrm_url(origin, destination)
    try:
        payload = _get_json(url, timeout=8)
        route = (payload.get("routes") or [])[0]
    except Exception as exc:
        return {
            "source": "osrm",
            "available": False,
            "origin_district": origin_district,
            "destination_district": destination_district,
            "summary": f"OSRM 路線服務暫時無法取得資料：{exc}",
            "data_gap": "目前仍可使用起訖行政區事故資料進行出行風險提醒。",
        }

    distance_km = round(float(route.get("distance", 0)) / 1000, 2)
    duration_min = round(float(route.get("duration", 0)) / 60, 1)
    coordinates = route.get("geometry", {}).get("coordinates", [])
    route_points = [{"lon": lon, "lat": lat} for lon, lat in coordinates]

    return {
        "source": "osrm",
        "available": True,
        "origin_district": origin_district,
        "destination_district": destination_district,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "route_points": route_points,
        "summary": f"{origin_district}到{destination_district}參考路線約 {distance_km} 公里，預估 {duration_min} 分鐘。",
    }


def _build_osrm_url(origin: tuple[float, float], destination: tuple[float, float]) -> str:
    origin_lat, origin_lon = origin
    dest_lat, dest_lon = destination
    coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    return (
        "https://router.project-osrm.org/route/v1/driving/"
        + quote(coords, safe=",;")
        + "?overview=full&geometries=geojson&steps=false"
    )


def _get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "traffic-risk-demo/1.0"})
    try:
        with urlopen(request, timeout=timeout, context=_SSL_CTX) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    except TimeoutError as exc:
        raise RuntimeError("request timed out") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid JSON response") from exc


def _first_station(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records") or {}
    stations = records.get("Station") or records.get("location") or []
    if not stations:
        raise RuntimeError("no station data")
    return stations[0]


def _extract_weather(station: dict[str, Any]) -> dict[str, Any]:
    weather_element = station.get("WeatherElement") or station.get("weatherElement") or {}
    geo_info = station.get("GeoInfo") or {}
    parameters = station.get("parameter") or []
    fallback_county = parameters[0].get("parameterValue") if parameters and isinstance(parameters[0], dict) else None
    return {
        "weather": weather_element.get("Weather") or weather_element.get("weather"),
        "temperature": weather_element.get("AirTemperature") or weather_element.get("TEMP"),
        "humidity": weather_element.get("RelativeHumidity") or weather_element.get("HUMD"),
        "wind_speed": weather_element.get("WindSpeed") or weather_element.get("WDSD"),
        "observed_at": station.get("ObsTime", {}).get("DateTime") if isinstance(station.get("ObsTime"), dict) else station.get("time", {}).get("obsTime"),
        "county": geo_info.get("CountyName") or fallback_county,
    }


def _format_weather_summary(weather: dict[str, Any]) -> str:
    parts = []
    if weather.get("weather"):
        parts.append(f"天氣 {weather['weather']}")
    if weather.get("temperature") not in (None, ""):
        parts.append(f"氣溫 {weather['temperature']}°C")
    if weather.get("humidity") not in (None, ""):
        parts.append(f"相對濕度 {weather['humidity']}%")
    if weather.get("wind_speed") not in (None, ""):
        parts.append(f"風速 {weather['wind_speed']} m/s")
    return "中央氣象署即時觀測：" + "，".join(parts) + "。" if parts else "中央氣象署即時觀測資料已取得。"


# ─────────────────────────────────────────────────────────
# TDX 即時車流（運輸資料流通服務平台）
# ─────────────────────────────────────────────────────────

_TDX_TOKEN_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
# 即時車輛偵測器（VD）資料：含各車道即時車速
_TDX_VD_URL = "https://tdx.transportdata.tw/api/basic/v2/Road/Traffic/Live/VD/City/Taichung?%24format=JSON"
# VD 靜態資料：含偵測器座標（用於對應行政區）
_TDX_VD_STATIC_URL = "https://tdx.transportdata.tw/api/basic/v2/Road/Traffic/VD/City/Taichung?%24format=JSON"
_tdx_token_cache: dict[str, Any] = {}
_tdx_data_cache: dict[str, Any] = {}
_tdx_vdmap_cache: dict[str, Any] = {}   # VDID → 行政區對照（靜態，快取較久）


def _get_tdx_token() -> str | None:
    """取得 TDX OAuth2 access token（快取至過期）。"""
    import time
    now = time.time()
    cached = _tdx_token_cache
    if cached.get("token") and now < cached.get("expires_at", 0) - 30:
        return cached["token"]

    client_id     = os.getenv("CLIENT_ID", "").strip()
    client_secret = os.getenv("CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        return None

    body = urlencode({
        "grant_type":    "client_credentials",
        "client_id":     client_id,
        "client_secret": client_secret,
    }).encode("utf-8")
    req = Request(
        _TDX_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 1800))
        cached["token"] = token
        cached["expires_at"] = now + expires_in
        return token
    except Exception:
        return None


def _extract_vd_speeds(vd_records: list[dict[str, Any]]) -> list[float]:
    """從 TDX VD 巢狀結構提取有效車速（km/h）。

    結構：VD → LinkFlows[] → Lanes[] → Speed
    濾除無效值（-99、0 代表偵測器異常或無車流）。
    """
    speeds: list[float] = []
    for vd in vd_records:
        for link in vd.get("LinkFlows", []) or []:
            for lane in link.get("Lanes", []) or []:
                spd = lane.get("Speed")
                if spd is not None and 0 < spd <= 120:  # 合理車速範圍
                    speeds.append(float(spd))
    return speeds


def _nearest_district(lat: float, lon: float) -> str | None:
    """以最近行政區中心點判定座標所屬行政區。"""
    best, best_dist = None, float("inf")
    for district, (dlat, dlon) in DISTRICT_CENTERS.items():
        d = (lat - dlat) ** 2 + (lon - dlon) ** 2
        if d < best_dist:
            best_dist, best = d, district
    return best


def _get_vdid_district_map(token: str) -> dict[str, str]:
    """取得 VDID → 行政區對照（VD 靜態座標，快取 1 小時）。"""
    import time
    cache = _tdx_vdmap_cache
    if cache.get("map") and time.time() - cache.get("ts", 0) < 3600:
        return cache["map"]

    req = Request(
        _TDX_VD_STATIC_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=12, context=_SSL_CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return cache.get("map", {})

    records = data if isinstance(data, list) else (
        next((v for v in data.values() if isinstance(v, list)), [])
    )
    vdid_map: dict[str, str] = {}
    for vd in records:
        lat, lon = vd.get("PositionLat"), vd.get("PositionLon")
        vdid = vd.get("VDID")
        if vdid and lat and lon:
            district = _nearest_district(float(lat), float(lon))
            if district:
                vdid_map[vdid] = district

    cache["map"] = vdid_map
    cache["ts"] = time.time()
    return vdid_map


def _district_speeds(
    vd_live: list[dict[str, Any]],
    vdid_map: dict[str, str],
) -> dict[str, list[float]]:
    """將即時 VD 車速按行政區分組。"""
    from collections import defaultdict
    result: dict[str, list[float]] = defaultdict(list)
    for vd in vd_live:
        district = vdid_map.get(vd.get("VDID"))
        if not district:
            continue
        for link in vd.get("LinkFlows", []) or []:
            for lane in link.get("Lanes", []) or []:
                spd = lane.get("Speed")
                if spd is not None and 0 < spd <= 120:
                    result[district].append(float(spd))
    return dict(result)


def _speed_to_level(avg_speed: float) -> str:
    if avg_speed >= 55:
        return "順暢"
    if avg_speed >= 40:
        return "車多"
    if avg_speed >= 25:
        return "壅塞"
    return "嚴重壅塞"


def _vd_age_minutes(vd_records: list[dict[str, Any]]) -> float | None:
    """回傳 VD 資料採集時間距今的分鐘數；無法解析時回傳 None。

    用於驗證即時車流的新鮮度——TDX 免費層的台中 VD 可能回傳過期快照，
    過期資料不應被當成「現在路況」使用。
    """
    from datetime import datetime
    for vd in vd_records:
        ct = vd.get("DataCollectTime")
        if not ct:
            continue
        try:
            collected = datetime.fromisoformat(ct)
            now = datetime.now(collected.tzinfo)
            return (now - collected).total_seconds() / 60.0
        except Exception:
            return None
    return None


_hourly_traffic_cache: dict[str, Any] = {}


def _hourly_congestion_profile() -> dict[int, float]:
    """以歷史事故的時段分布推估各小時的相對車流／壅塞程度（快取）。

    邏輯：事故件數越多的時段，通常車流越密集、壅塞風險越高。
    以各小時事故數正規化為 0-1 的「壅塞指數」。
    """
    if _hourly_traffic_cache:
        return _hourly_traffic_cache
    try:
        from data_loader import load_accident_data
        df = load_accident_data()
        counts = df["hour"].value_counts()
        max_c = int(counts.max()) if not counts.empty else 1
        profile = {int(h): round(int(c) / max_c, 3) for h, c in counts.items()}
        _hourly_traffic_cache.update(profile)
    except Exception:
        pass
    return _hourly_traffic_cache


def estimate_hourly_traffic(hour: int | None) -> dict[str, Any]:
    """即時車流不可用時，用歷史時段密度推估此時段的車流狀況。

    回傳：available, source="歷史推估", hour, level, congestion_index, summary
    """
    if hour is None:
        from datetime import datetime
        hour = datetime.now().hour

    profile = _hourly_congestion_profile()
    idx = profile.get(int(hour), 0.0)

    if idx >= 0.8:
        level, desc = "車流高峰", "通勤尖峰，車多易壅塞"
    elif idx >= 0.5:
        level, desc = "車流偏多", "車流量中等偏高"
    elif idx >= 0.25:
        level, desc = "車流普通", "車流量普通"
    else:
        level, desc = "車流稀少", "離峰時段，車流量少"

    return {
        "available": True,
        "source": "歷史推估",
        "hour": int(hour),
        "level": level,
        "congestion_index": idx,
        "summary": f"{hour}時通常為「{level}」（{desc}，依歷史事故時段分布推估）",
    }


def fetch_tdx_traffic_tool(
    district: str | None = None,
    districts: list[str] | None = None,
) -> dict[str, Any]:
    """取得台中市即時車流資料，彙整為全市與分區壅塞摘要。

    資料來源：TDX Road/Traffic/Live/VD（每分鐘更新）+ VD 靜態座標（分區用）
    結果快取 3 分鐘，避免頻繁呼叫觸發 API 流量限制。

    Parameters
    ----------
    district  : 單一查詢行政區（回傳該區車速）
    districts : 多個關注行政區（如起訖點，回傳各區車速供比較）

    回傳：
        available        : bool
        avg_speed        : 全市平均車速（km/h）
        level            : 全市壅塞程度
        vd_count         : 有效車道偵測點數
        congested_ratio  : 全市壅塞比例（speed < 20）
        district_traffic : {行政區: {avg_speed, level, count}}（分區資料）
        summary          : 自然語言摘要（含關注行政區）
    """
    import time
    cache = _tdx_data_cache
    cached = cache.get("data") if cache.get("data") and time.time() - cache.get("ts", 0) < 180 else None

    token = _get_tdx_token()
    if not token:
        return {
            "available": False,
            "district": district,
            "summary": "尚未設定 TDX API 憑證，無法取得即時車流資料。",
            "data_gap": "設定 CLIENT_ID 與 CLIENT_SECRET 後可顯示台中即時車流。",
        }

    # 全市資料用快取，分區摘要每次依 districts 重組
    if cached:
        base = cached
    else:
        req = Request(
            _TDX_VD_URL,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        try:
            with urlopen(req, timeout=12, context=_SSL_CTX) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return {
                "available": False,
                "district": district,
                "summary": f"TDX 車流 API 暫時無法取得資料：{exc}",
                "data_gap": "網路或 API 正常後將自動顯示即時車流。",
            }

        vd_records = data if isinstance(data, list) else (
            next((v for v in data.values() if isinstance(v, list)), [])
        )

        # 新鮮度驗證：TDX 免費層可能回傳過期快照，超過 30 分鐘不當即時車流用
        age_min = _vd_age_minutes(vd_records)
        if age_min is not None and age_min > 30:
            result = {
                "available": False,
                "district": district,
                "data_age_minutes": round(age_min),
                "summary": "目前無法取得即時車流（資料來源未即時更新）。",
                "data_gap": "TDX 車流資料源更新延遲，本次以歷史與天氣資料為主。",
            }
            cache["data"] = result
            cache["ts"] = time.time()
            return result

        speeds = _extract_vd_speeds(vd_records)
        if not speeds:
            result = {
                "available": False,
                "district": district,
                "summary": "目前台中市偵測器無有效車速資料（可能為離峰或偵測器維護）。",
            }
            cache["data"] = result
            cache["ts"] = time.time()
            return result

        avg_speed = round(sum(speeds) / len(speeds), 1)
        congested_ratio = round(sum(1 for s in speeds if s < 20) / len(speeds) * 100, 1)
        level = _speed_to_level(avg_speed)

        # 分區聚合
        vdid_map = _get_vdid_district_map(token)
        d_speeds = _district_speeds(vd_records, vdid_map)
        district_traffic = {
            d: {
                "avg_speed": round(sum(sp) / len(sp), 1),
                "level": _speed_to_level(sum(sp) / len(sp)),
                "count": len(sp),
            }
            for d, sp in d_speeds.items() if sp
        }

        base = {
            "available": True,
            "avg_speed": avg_speed,
            "level": level,
            "vd_count": len(speeds),
            "congested_ratio": congested_ratio,
            "district_traffic": district_traffic,
        }
        cache["data"] = base
        cache["ts"] = time.time()

    # 依關注行政區組裝摘要（去重保序）
    dt = base.get("district_traffic", {})
    _raw_focus = list(districts) if districts else ([district] if district else [])
    focus = list(dict.fromkeys([d for d in _raw_focus if d]))

    parts = [f"台中市整體平均車速 {base['avg_speed']} km/h（{base['level']}）"]
    focus_lines = []
    for d in focus:
        info = dt.get(d)
        if info:
            focus_lines.append(f"{d} {info['avg_speed']} km/h（{info['level']}）")
    if focus_lines:
        parts.append("；".join(focus_lines))
    else:
        parts.append(f"全市 {base['congested_ratio']}% 路段車速低於 20 km/h")

    return {
        **base,
        "district": district,
        "summary": "即時路況：" + "；".join(parts) + "。",
    }
