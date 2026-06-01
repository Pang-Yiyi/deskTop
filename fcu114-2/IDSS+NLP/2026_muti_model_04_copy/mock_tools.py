"""Mock analysis tools for the LLM/Agent demo.

These functions keep the same interface expected from A's future analysis
modules. Replace this module with real tools after A provides them.
"""

from __future__ import annotations

from typing import Any


DISTRICT_STATS = [
    {"district": "西屯區", "count": 19818, "rank": 1},
    {"district": "北屯區", "count": 15475, "rank": 2},
    {"district": "南屯區", "count": 11350, "rank": 3},
    {"district": "北區", "count": 9737, "rank": 4},
    {"district": "大里區", "count": 9043, "rank": 5},
]

HOUR_STATS = [
    {"hour": 17, "count": 14381, "rank": 1},
    {"hour": 18, "count": 11361, "rank": 2},
    {"hour": 7, "count": 11303, "rank": 3},
    {"hour": 8, "count": 10724, "rank": 4},
    {"hour": 16, "count": 9638, "rank": 5},
]

CAUSE_STATS = [
    {"code": "84", "description": "其他不當駕車行為", "count": 17023},
    {"code": "07", "description": "未保持行車安全距離", "count": 10713},
    {"code": "59", "description": "恍神、緊張、心不在焉分心駕駛", "count": 6884},
    {"code": "11", "description": "有號誌路口，轉彎車未讓直行車先行", "count": 4905},
    {"code": "51", "description": "起步時未注意安全", "count": 4720},
]

WEATHER_LOOKUP = {
    "風": {"code": "1", "count": 105},
    "風沙": {"code": "2", "count": 1},
    "霧": {"code": "3", "count": 18},
    "煙": {"code": "3", "count": 18},
    "雪": {"code": "4", "count": 4},
    "雨": {"code": "5", "count": 9084},
    "陰": {"code": "6", "count": 6631},
    "晴": {"code": "7", "count": 129722},
}

CODE_LOOKUP = {
    "07": "未保持行車安全距離",
    "08": "未保持行車安全間隔",
    "11": "有號誌路口，轉彎車未讓直行車先行",
    "14": "無號誌路口，轉彎車未讓直行車先行",
    "20": "其他未依規定讓車",
    "40": "變換車道不當",
    "51": "起步時未注意安全",
    "59": "恍神、緊張、心不在焉分心駕駛",
    "84": "其他不當駕車行為",
}


def _find_district(district: str | None) -> dict[str, Any] | None:
    if not district:
        return None
    return next((item for item in DISTRICT_STATS if item["district"] == district), None)


def _find_hour(hour: int | str | None) -> dict[str, Any] | None:
    if hour is None or hour == "":
        return None
    try:
        hour_int = int(str(hour).replace(":00", ""))
    except ValueError:
        return None
    return next((item for item in HOUR_STATS if item["hour"] == hour_int), None)


def _normalize_weather(weather: str | None) -> str | None:
    if not weather:
        return None
    for key in WEATHER_LOOKUP:
        if key in weather:
            return key
    return weather


def accident_query_tool(
    district: str | None = None,
    hour: int | str | None = None,
    weekday: str | None = None,
    month: int | str | None = None,
    weather: str | None = None,
) -> dict[str, Any]:
    """Return mock accident statistics for a query."""
    district_stat = _find_district(district)
    hour_stat = _find_hour(hour)
    normalized_weather = _normalize_weather(weather)
    weather_stat = WEATHER_LOOKUP.get(normalized_weather or "")

    return {
        "tool": "accident_query_tool",
        "query": {
            "district": district,
            "hour": int(str(hour).replace(":00", "")) if hour not in (None, "") else None,
            "weekday": weekday,
            "month": month,
            "weather": normalized_weather,
        },
        "result": {
            "district": district_stat,
            "hour": hour_stat,
            "weather": weather_stat,
            "top_districts": DISTRICT_STATS,
            "top_hours": HOUR_STATS,
        },
        "summary": _build_accident_summary(district_stat, hour_stat, normalized_weather),
        "source": "mock statistics based on current EDA outputs",
    }


def risk_score_tool(
    district: str | None = None,
    hour: int | str | None = None,
    weekday: str | None = None,
    month: int | str | None = None,
    weather: str | None = None,
) -> dict[str, Any]:
    """Return a rule-based mock risk score."""
    district_stat = _find_district(district)
    hour_stat = _find_hour(hour)
    normalized_weather = _normalize_weather(weather)

    district_score = 30 if district_stat and district_stat["rank"] <= 3 else 15 if district_stat else 8
    hour_score = 30 if hour_stat and hour_stat["rank"] <= 3 else 15 if hour_stat else 8
    weather_score = 20 if normalized_weather in {"雨", "霧", "煙"} else 10 if normalized_weather == "陰" else 5
    weekday_score = 10 if weekday in {"星期五", "週五", "禮拜五"} else 6 if weekday else 4
    score = min(100, district_score + hour_score + weather_score + weekday_score)

    if score >= 70:
        level = "高風險"
    elif score >= 40:
        level = "中風險"
    else:
        level = "低風險"

    reasons = []
    if district_stat:
        reasons.append(f"{district_stat['district']}在目前資料中事故數排名第 {district_stat['rank']}，共 {district_stat['count']} 筆。")
    if hour_stat:
        reasons.append(f"{hour_stat['hour']} 時為事故高峰時段之一，排名第 {hour_stat['rank']}，共 {hour_stat['count']} 筆。")
    if normalized_weather:
        reasons.append(f"{normalized_weather}天會影響視線、路面摩擦或車距判斷，需提高注意。")
    if weekday:
        reasons.append(f"{weekday}可能受通勤與下班車流影響，需搭配歷史時段分布判斷。")

    return {
        "tool": "risk_score_tool",
        "query": {
            "district": district,
            "hour": int(str(hour).replace(":00", "")) if hour not in (None, "") else None,
            "weekday": weekday,
            "month": month,
            "weather": normalized_weather,
        },
        "risk": {"score": score, "level": level},
        "score_breakdown": {
            "district_score": district_score,
            "hour_score": hour_score,
            "weather_score": weather_score,
            "weekday_score": weekday_score,
        },
        "evidence": {
            "district_rank": district_stat["rank"] if district_stat else None,
            "district_count": district_stat["count"] if district_stat else None,
            "hour_rank": hour_stat["rank"] if hour_stat else None,
            "hour_count": hour_stat["count"] if hour_stat else None,
        },
        "reasons": reasons,
        "source": "mock rule score; replace with A's risk model later",
    }


def cause_analysis_tool(district: str | None = None, top_n: int = 5) -> dict[str, Any]:
    """Return mock top accident causes."""
    causes = CAUSE_STATS[:top_n]
    scope = f"{district}事故資料" if district else "台中市事故資料"
    return {
        "tool": "cause_analysis_tool",
        "query": {"district": district, "top_n": top_n},
        "causes": causes,
        "summary": f"根據目前 {scope} 的 mock 摘要，主要肇事因素以不當駕車、未保持安全距離與分心駕駛為主。",
        "source": "mock cause ranking based on current EDA outputs",
    }


def weather_time_heatmap_tool(
    weather: str | None = None,
    hour: int | str | None = None,
) -> dict[str, Any]:
    """Return a compact mock weather-time risk summary."""
    normalized_weather = _normalize_weather(weather)
    hour_stat = _find_hour(hour)
    high_hours = [item["hour"] for item in HOUR_STATS[:3]]
    return {
        "tool": "weather_time_heatmap_tool",
        "query": {
            "weather": normalized_weather,
            "hour": int(str(hour).replace(":00", "")) if hour not in (None, "") else None,
        },
        "result": {
            "weather": WEATHER_LOOKUP.get(normalized_weather or ""),
            "high_risk_hours": high_hours,
            "queried_hour_rank": hour_stat["rank"] if hour_stat else None,
        },
        "summary": f"{normalized_weather or '指定天候'}與時段交互分析中，17、18、7 時屬於需要優先注意的事故高峰時段。",
        "source": "mock heatmap summary",
    }


def recommendation_tool(
    risk_level: str,
    causes: list[dict[str, Any]] | None = None,
    district: str | None = None,
    hour: int | str | None = None,
    weather: str | None = None,
) -> dict[str, Any]:
    """Return recommendations for users and agencies."""
    normalized_weather = _normalize_weather(weather)
    recommendations = {
        "民眾": ["降低車速並保持安全距離", "避免急煞、搶快與頻繁變換車道"],
        "交通管理單位": ["針對事故高峰時段加強路口巡邏與號誌管理", "優先盤點高事故行政區的主要路口"],
        "執法單位": ["加強取締未保持安全距離、違規變換車道與轉彎未讓直行車"],
    }

    if normalized_weather == "雨":
        recommendations["民眾"].append("雨天應增加跟車距離並提早煞車")
    if district:
        recommendations["交通管理單位"].append(f"針對{district}事故熱點安排道路安全改善")
    if hour not in (None, ""):
        recommendations["交通管理單位"].append(f"於 {hour} 時前後加強尖峰疏導")

    return {
        "tool": "recommendation_tool",
        "query": {
            "risk_level": risk_level,
            "district": district,
            "hour": hour,
            "weather": normalized_weather,
        },
        "recommendations": recommendations,
        "basis": causes or CAUSE_STATS[:3],
        "summary": f"此查詢判定為{risk_level}，建議同時從駕駛行為、熱點路口與尖峰時段管理著手。",
    }


def citizen_route_advice_tool(
    origin_district: str | None = None,
    destination_district: str | None = None,
    hour: int | None = None,
    weather: str | None = None,
    transport_mode: str = "機車",
) -> dict[str, Any]:
    """Mock stub for citizen route advice."""
    normalized_weather = _normalize_weather(weather)
    return {
        "tool": "citizen_route_advice_tool",
        "query": {
            "origin_district": origin_district,
            "destination_district": destination_district,
            "hour": hour,
            "weather": normalized_weather,
            "transport_mode": transport_mode,
        },
        "risk": {"score": 50, "level": "中風險", "breakdown": {}},
        "factors": ["此為 mock 資料，請確認 analysis_tools 已正確載入。"],
        "advice": ["請確認系統已正確載入分析工具。"],
        "limitation": "Mock 模式，不含真實事故資料分析。",
    }


def rag_lookup_tool(keyword: str) -> dict[str, Any]:
    """Return mock RAG lookup for codes and field definitions."""
    normalized = keyword.strip().upper()
    code = None
    for token in normalized.replace("代碼", " ").split():
        if token.isdigit():
            code = token.zfill(2)
            break
    if code and code in CODE_LOOKUP:
        answer = f"肇事因素代碼 {code} 代表「{CODE_LOOKUP[code]}」。"
    elif "天候" in keyword:
        answer = "天候欄位代表事故發生時的天氣狀態，常見代碼包含 5=雨、6=陰、7=晴。"
    elif "資料" in keyword or "限制" in keyword:
        answer = "目前系統根據歷史事故資料分析，尚未串接即時天氣、即時車流或未來預測資料。"
    else:
        answer = "目前知識庫尚未找到完全對應的條目，建議補充欄位名稱或代碼。"

    return {
        "tool": "rag_lookup_tool",
        "query": {"keyword": keyword},
        "answer": answer,
        "source": "mock knowledge base",
    }


def _build_accident_summary(
    district_stat: dict[str, Any] | None,
    hour_stat: dict[str, Any] | None,
    weather: str | None,
) -> str:
    parts = []
    if district_stat:
        parts.append(f"{district_stat['district']}事故數排名第 {district_stat['rank']}，共 {district_stat['count']} 筆")
    else:
        parts.append("目前查詢會回傳台中市事故熱點摘要")
    if hour_stat:
        parts.append(f"{hour_stat['hour']} 時事故數排名第 {hour_stat['rank']}，共 {hour_stat['count']} 筆")
    if weather:
        parts.append(f"天候條件為{weather}")
    return "；".join(parts) + "。"
