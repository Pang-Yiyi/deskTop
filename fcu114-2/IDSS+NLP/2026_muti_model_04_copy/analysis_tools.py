"""
analysis_tools.py
提供可供 B（LLM/Agent）直接呼叫的事故查詢工具函式。
所有函式回傳 dict（JSON 相容格式）。
"""

from __future__ import annotations

import re
import pandas as pd
from data_loader import load_accident_data, WEEKDAY_MAP, WEATHER_MAPPING, load_code_dict

# 全域快取（避免每次呼叫重新讀檔）
_df: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = load_accident_data()
    return _df


def _filter(
    df: pd.DataFrame,
    district: str | None = None,
    hour: int | None = None,
    weekday: str | None = None,
    month: int | None = None,
    weather: str | None = None,
) -> pd.DataFrame:
    """依條件篩選 DataFrame。"""
    if district:
        df = df[df["區"] == district]
    if hour is not None:
        df = df[df["hour"] == hour]
    if weekday:
        df = df[df["weekday"] == weekday]
    if month is not None:
        df = df[df["month"] == month]
    if weather:
        df = df[df["天候_str"] == weather]
    return df


# ─────────────────────────────────────────────────────────
# accident_query_tool
# ─────────────────────────────────────────────────────────

def accident_query_tool(
    district: str | None = None,
    hour: int | None = None,
    weekday: str | None = None,
    month: int | None = None,
    weather: str | None = None,
) -> dict:
    """
    查詢符合條件的事故統計。

    Parameters
    ----------
    district : 行政區（例如「西屯區」）
    hour     : 小時（0-23）
    weekday  : 星期（例如「星期五」）
    month    : 月份（1-12）
    weather  : 天候中文（例如「雨」）

    Returns
    -------
    dict with keys: query, result, summary
    """
    df = _get_df()
    filtered = _filter(df, district, hour, weekday, month, weather)
    accident_count = len(filtered)

    result: dict = {"accident_count": accident_count}

    # 行政區排名
    if district:
        district_counts = df["區"].value_counts()
        result["district_total"] = int(district_counts.get(district, 0))
        result["district_rank"] = int(
            (district_counts > district_counts.get(district, 0)).sum() + 1
        )

    # 小時排名
    if hour is not None:
        hour_counts = df["hour"].value_counts()
        result["hour_total"] = int(hour_counts.get(hour, 0))
        result["hour_rank"] = int(
            (hour_counts > hour_counts.get(hour, 0)).sum() + 1
        )

    # 生成摘要文字
    parts = []
    if district:
        rank = result.get("district_rank", "?")
        total = result.get("district_total", 0)
        parts.append(f"{district}為本資料集中事故數第 {rank} 名的行政區（共 {total:,} 件）")
    if hour is not None:
        rank = result.get("hour_rank", "?")
        total = result.get("hour_total", 0)
        parts.append(f"{hour} 時為事故第 {rank} 多的時段（共 {total:,} 件）")
    if weather:
        parts.append(f"天候「{weather}」")
    if weekday:
        parts.append(f"{weekday}")
    if month:
        parts.append(f"{month} 月")

    condition_str = "、".join(parts) if parts else "全資料集"
    summary = f"{condition_str}，符合條件事故共 {accident_count:,} 件。"

    return {
        "query": {
            "district": district,
            "hour": hour,
            "weekday": weekday,
            "month": month,
            "weather": weather,
        },
        "result": result,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────
# cause_analysis_tool
# ─────────────────────────────────────────────────────────

def cause_analysis_tool(
    district: str | None = None,
    top_n: int = 5,
) -> dict:
    """
    分析主要肇事因素排名，同時統計次要因素，並合併計算複合因素總覽。

    Returns
    -------
    dict with keys: query, causes, secondary_causes, combined_causes, summary
    """
    df = _get_df()
    if district:
        df = df[df["區"] == district]

    if "肇事因素主要_str" not in df.columns:
        return {
            "query": {"district": district, "top_n": top_n},
            "causes": [],
            "secondary_causes": [],
            "combined_causes": [],
            "summary": "無肇事因素資料",
        }

    code_dict = load_code_dict()

    def _extract_cause_counts(col: str) -> list[dict]:
        if col not in df.columns:
            return []
        raw = df[col].apply(
            lambda x: str(int(float(x))).zfill(2) if pd.notna(x) else None
        ).dropna()
        # 排除「無」代碼（通常為 "00" 或 "99"）
        raw = raw[~raw.isin(["00", "99"])]
        counts = raw.value_counts().head(top_n)
        return [
            {
                "code": code,
                "description": code_dict.get(code, f"{code}(未知代碼)"),
                "count": int(cnt),
            }
            for code, cnt in counts.items()
        ]

    causes = _extract_cause_counts("肇事因素主要")
    secondary_causes = _extract_cause_counts("肇事因素次要") if "肇事因素次要" in df.columns else []

    # 合併主因與次因，累計各代碼出現次數
    combined: dict[str, int] = {}
    for item in causes:
        combined[item["code"]] = combined.get(item["code"], 0) + item["count"]
    for item in secondary_causes:
        combined[item["code"]] = combined.get(item["code"], 0) + item["count"]
    combined_causes = sorted(
        [
            {"code": code, "description": code_dict.get(code, f"{code}(未知代碼)"), "total_count": cnt}
            for code, cnt in combined.items()
        ],
        key=lambda x: x["total_count"],
        reverse=True,
    )[:top_n]

    top_names = "、".join(c["description"] for c in causes[:3])
    suffix = f"（以{top_names}為主）" if top_names else ""
    label = "全資料集" if not district else district
    secondary_note = (
        f"；次要因素前三名：{'、'.join(c['description'] for c in secondary_causes[:3])}"
        if secondary_causes
        else ""
    )
    summary = f"{label}主要肇事因素前 {top_n} 名{suffix}{secondary_note}。"

    return {
        "query": {"district": district, "top_n": top_n},
        "causes": causes,
        "secondary_causes": secondary_causes,
        "combined_causes": combined_causes,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────
# weather_time_heatmap_tool
# ─────────────────────────────────────────────────────────

def weather_time_heatmap_tool(
    weather: str | None = None,
    hour: int | None = None,
) -> dict:
    """
    查詢天候 × 時段事故分布（交叉統計）。

    Returns
    -------
    dict with keys: query, heatmap, summary
    """
    df = _get_df()

    if "天候_str" not in df.columns:
        return {"query": {"weather": weather, "hour": hour}, "heatmap": {}, "summary": "無天候資料"}

    cross = pd.crosstab(df["天候_str"], df["hour"])

    heatmap: dict = {}
    for w in cross.index:
        heatmap[w] = {int(h): int(cross.loc[w, h]) for h in cross.columns if h in cross.columns}

    # 特定天候或時段的摘要
    summary_parts = []
    if weather and weather in cross.index:
        peak_hour = int(cross.loc[weather].idxmax())
        peak_count = int(cross.loc[weather].max())
        summary_parts.append(f"天候「{weather}」在 {peak_hour} 時事故最多（{peak_count:,} 件）")
    if hour is not None and hour in cross.columns:
        peak_weather = cross[hour].idxmax()
        peak_count = int(cross[hour].max())
        summary_parts.append(f"{hour} 時以天候「{peak_weather}」事故最多（{peak_count:,} 件）")

    summary = "；".join(summary_parts) if summary_parts else "天候 × 時段事故分布如 heatmap 所示。"

    return {
        "query": {"weather": weather, "hour": hour},
        "heatmap": heatmap,
        "summary": summary,
    }


# ─────────────────────────────────────────────────────────
# recommendation_tool
# ─────────────────────────────────────────────────────────

_RECOMMENDATIONS = {
    "高風險": {
        "public": [
            "強烈建議避開此時段或行政區行車",
            "若必須出行，請降低車速至少 20%",
            "保持足夠安全距離，避免急煞急轉",
        ],
        "authority": [
            "建議加派警力於事故熱點路段",
            "可評估於尖峰時段調整號誌週期",
            "建議設立可變速限標誌",
        ],
    },
    "中風險": {
        "public": [
            "行車前確認車況與燈光",
            "保持安全距離，減少變換車道",
            "惡劣天候請延後出行或改走替代道路",
        ],
        "authority": [
            "可加強路口標線與警告標誌維護",
            "定期稽查超速與酒駕",
        ],
    },
    "低風險": {
        "public": [
            "維持正常行車注意事項",
            "注意行人與機車動向",
        ],
        "authority": [
            "持續監測事故趨勢",
        ],
    },
}


def recommendation_tool(
    risk_level: str,
    causes: list | None = None,
    district: str | None = None,
    hour: int | None = None,
    weather: str | None = None,
) -> dict:
    """
    根據風險等級與條件產生建議。

    Parameters
    ----------
    risk_level : 「高風險」/「中風險」/「低風險」
    causes     : 主要肇事因素說明清單（可選）
    district, hour, weather : 同其他工具

    Returns
    -------
    dict with keys: risk_level, recommendations, context
    """
    base = _RECOMMENDATIONS.get(risk_level, _RECOMMENDATIONS["低風險"])

    public_recs = list(base["public"])
    authority_recs = list(base["authority"])

    # 依肇因加入針對性建議
    normalized_causes = _normalize_causes(causes)
    if normalized_causes:
        for cause in normalized_causes:
            if "安全距離" in cause:
                public_recs.append("特別注意與前車保持安全距離")
                authority_recs.append("可強化車距宣導與取締")
            if "超速" in cause or "減速" in cause:
                public_recs.append("遵守速限，勿超速")
                authority_recs.append("建議加強測速執法")
            if "酒" in cause:
                public_recs.append("切勿酒後駕車，可改搭大眾交通工具")
                authority_recs.append("加強酒測路檢頻率")

    if weather in ("雨", "霧或煙", "陰"):
        public_recs.append(f"天候「{weather}」時視線不佳，請開啟霧燈並降速行駛")

    context_parts = []
    if district:
        context_parts.append(f"行政區：{district}")
    if hour is not None:
        context_parts.append(f"時段：{hour} 時")
    if weather:
        context_parts.append(f"天候：{weather}")

    decision_support = _build_decision_support(
        risk_level=risk_level,
        causes=normalized_causes,
        district=district,
        hour=hour,
        weather=weather,
    )

    return {
        "risk_level": risk_level,
        "context": context_parts,
        "decision_support": decision_support,
        "recommendations": {
            "民眾建議": list(dict.fromkeys(public_recs)),
            "交通管理單位建議": list(dict.fromkeys(authority_recs)),
        },
    }


def citizen_route_advice_tool(
    origin_district: str | None = None,
    destination_district: str | None = None,
    hour: int | None = None,
    weather: str | None = None,
    transport_mode: str = "機車",
) -> dict:
    """
    民眾出行風險輔助。

    目前不計算真實導航路線，而是根據起訖行政區、時段、天候與交通工具，
    用歷史事故資料提供出行風險提醒。
    """
    df = _get_df()
    district_counts = df["區"].value_counts()
    hour_counts = df["hour"].value_counts()
    max_district_count = int(district_counts.max()) if not district_counts.empty else 1
    max_hour_count = int(hour_counts.max()) if not hour_counts.empty else 1

    route_districts = [d for d in [origin_district, destination_district] if d]
    route_districts = list(dict.fromkeys(route_districts))

    district_details = []
    district_score = 0
    for district in route_districts:
        count = int(district_counts.get(district, 0))
        rank = int((district_counts > count).sum() + 1) if count else None
        score = round(count / max_district_count * 35) if max_district_count else 0
        district_score = max(district_score, score)
        district_details.append(
            {
                "district": district,
                "accident_count": count,
                "rank": rank,
                "score": score,
            }
        )

    time_score = 15
    hour_detail = None
    if hour is not None:
        hour_count = int(hour_counts.get(hour, 0))
        hour_rank = int((hour_counts > hour_count).sum() + 1) if hour_count else None
        time_score = round(hour_count / max_hour_count * 25) if max_hour_count else 0
        hour_detail = {"hour": hour, "accident_count": hour_count, "rank": hour_rank, "score": time_score}

    weather_score = {
        "雨": 18,
        "霧": 20,
        "霧或煙": 20,
        "風沙": 18,
        "雪": 20,
        "陰": 10,
        "風": 8,
        "晴": 3,
    }.get(weather, 5)

    mode_score = {
        "機車": 15,
        "自行車": 14,
        "行人": 12,
        "汽車": 8,
        "大眾運輸": 3,
    }.get(transport_mode, 8)

    total_score = min(100, district_score + time_score + weather_score + mode_score)
    if total_score >= 70:
        level = "高風險"
    elif total_score >= 40:
        level = "中風險"
    else:
        level = "低風險"

    factors = []
    if district_details:
        hot = sorted(district_details, key=lambda item: item["accident_count"], reverse=True)[0]
        factors.append(f"{hot['district']}在本資料集中事故數排名第 {hot['rank']}，屬於出行需注意區域")
    if hour_detail and hour_detail.get("rank") and hour_detail["rank"] <= 5:
        factors.append(f"{hour} 時為事故高峰時段之一，可能受通勤或下班車流影響")
    if weather in ("雨", "霧", "霧或煙", "風沙", "雪"):
        factors.append(f"天候「{weather}」可能降低視線或路面安全性")
    if transport_mode in ("機車", "自行車"):
        factors.append(f"{transport_mode}對天候、路面與大型車流較敏感，需提高防禦駕駛")
    if transport_mode == "行人":
        factors.append("行人需特別注意路口穿越、轉彎車與夜間能見度")

    advice = []
    if level == "高風險":
        advice.append("建議延後或提前出發，避開尖峰時段。")
        advice.append("若可選擇路線，優先避開事故熱點行政區或大型路口。")
    elif level == "中風險":
        advice.append("建議預留更多通勤時間，降低趕路造成的風險。")
    else:
        advice.append("維持正常注意事項，仍需留意路口與車距。")

    if weather in ("雨", "霧", "霧或煙", "風沙", "雪"):
        advice.append("不良天候下請降速、增加安全距離，機車族建議穿戴醒目雨具。")
    if transport_mode == "大眾運輸":
        advice.append("若天候不佳或尖峰風險較高，可優先考慮大眾運輸。")
    if transport_mode in ("機車", "自行車"):
        advice.append("避免貼近大型車，通過路口前先確認轉彎車動向。")

    return {
        "query": {
            "origin_district": origin_district,
            "destination_district": destination_district,
            "hour": hour,
            "weather": weather,
            "transport_mode": transport_mode,
        },
        "risk": {
            "score": total_score,
            "level": level,
            "breakdown": {
                "地區風險": district_score,
                "時段風險": time_score,
                "天候風險": weather_score,
                "交通工具風險": mode_score,
            },
        },
        "district_details": district_details,
        "hour_detail": hour_detail,
        "factors": factors,
        "advice": advice,
        "limitation": "目前尚未串接真實導航路網與即時車流，因此此功能是出行風險提醒，不是完整路線規劃。",
    }


def _normalize_causes(causes: list | None) -> list[str]:
    if not causes:
        return []

    normalized = []
    for cause in causes:
        if isinstance(cause, dict):
            description = cause.get("description")
            if description:
                normalized.append(str(description))
        elif cause:
            normalized.append(str(cause))
    return normalized


def _build_decision_support(
    risk_level: str,
    causes: list[str],
    district: str | None,
    hour: int | None,
    weather: str | None,
) -> dict:
    target_user = "一般民眾與交通管理單位"
    if district or causes:
        target_user = "交通管理單位與執法單位"

    if risk_level == "高風險":
        priority = "高，建議優先排入勤務、宣導或道路安全檢討。"
    elif risk_level == "中風險":
        priority = "中，建議持續監測並針對明確風險條件採取預防措施。"
    else:
        priority = "低，建議維持例行監測。"

    basis = []
    if district:
        basis.append(f"{district}事故分布")
    if hour is not None:
        basis.append(f"{hour}時時段風險")
    if weather:
        basis.append(f"{weather}天候條件")
    if causes:
        basis.append("主要肇事因素：" + "、".join(causes[:3]))

    next_actions = []
    if district:
        next_actions.append("檢視該行政區高事故路口或路段")
    if hour is not None:
        next_actions.append("評估尖峰時段警力、號誌或宣導配置")
    if weather:
        next_actions.append("於不良天候加強提醒與速限管理")
    if causes:
        next_actions.append("依主要肇因設計宣導與執法重點")

    return {
        "target_user": target_user,
        "priority": priority,
        "decision_basis": "、".join(basis) if basis else "全資料集事故分布與主要肇因排序",
        "external_factors": _infer_external_factors(district=district, hour=hour, weather=weather, causes=causes),
        "data_gap": _external_factor_data_gap(district=district, hour=hour, weather=weather, causes=causes),
        "next_action": "；".join(next_actions) if next_actions else "先確認事故熱點、時段與肇因排序，再決定管理資源配置。",
    }


def _infer_external_factors(
    district: str | None,
    hour: int | None,
    weather: str | None,
    causes: list[str],
) -> str:
    factors = []
    if district:
        factors.append(f"{district}事故較多可能與通勤路線、商圈活動、主要幹道或路口密度較高有關")
    if hour is not None:
        if 7 <= hour <= 9:
            factors.append("早上時段可能受到上班、上課與接送車流集中影響")
        elif 17 <= hour <= 19:
            factors.append("傍晚時段可能受到下班車流、疲勞駕駛與路口壅塞影響")
        elif 22 <= hour or hour <= 5:
            factors.append("深夜時段可能與視線不足、疲勞駕駛或車速較快有關")
    if weather in ("雨", "霧", "霧或煙", "風沙", "雪"):
        factors.append("不良天候可能降低視線、路面摩擦力與駕駛反應時間")
    if any("安全距離" in cause for cause in causes):
        factors.append("未保持安全距離可能在車流密集或走走停停的尖峰路段更容易發生")
    if any("分心" in cause or "恍神" in cause for cause in causes):
        factors.append("分心或恍神可能與通勤疲勞、長時間駕駛或路況複雜有關")

    return "；".join(factors) if factors else "目前可從時間、地區、天候與肇因推測外在風險，但尚無車流與道路環境資料可直接驗證。"


def _external_factor_data_gap(
    district: str | None,
    hour: int | None,
    weather: str | None,
    causes: list[str],
) -> str:
    gaps = ["分時車流量"]
    if district:
        gaps.extend(["道路型態", "路口密度", "商圈、學校、工業區或交流道位置"])
    if hour is not None:
        gaps.extend(["通勤人口", "尖峰時段平均速率"])
    if weather:
        gaps.append("即時或逐時天氣資料")
    if causes:
        gaps.append("事故現場型態與違規紀錄")
    return "、".join(dict.fromkeys(gaps))


# ─────────────────────────────────────────────────────────
# rag_lookup_tool（含同義詞擴展）
# ─────────────────────────────────────────────────────────

# 同義詞表：key 為使用者可能輸入的詞，value 為搜尋時額外展開的詞
_SYNONYM_MAP: dict[str, list[str]] = {
    "未保持距離": ["安全距離", "車距"],
    "車距": ["安全距離", "未保持距離"],
    "安全距離": ["車距", "未保持距離"],
    "闖紅燈": ["號誌", "違反號誌", "紅燈"],
    "紅燈": ["號誌", "違反號誌"],
    "違規": ["違反", "超速", "逆向"],
    "超速": ["速限", "減速"],
    "酒駕": ["酒", "飲酒"],
    "酒後": ["酒駕", "飲酒"],
    "飲酒": ["酒駕", "酒後"],
    "轉彎": ["轉向", "迴轉"],
    "迴轉": ["轉彎", "轉向"],
    "分心": ["恍神", "注意力"],
    "恍神": ["分心", "注意力"],
    "疲勞": ["打瞌睡", "疲憊"],
    "打瞌睡": ["疲勞", "疲憊"],
    "行人": ["步行", "穿越"],
    "機車": ["摩托車", "重機"],
    "夜間": ["深夜", "晚上"],
    "視線": ["視距", "能見度", "霧"],
    "路面": ["路況", "濕滑"],
    "超車": ["變換車道", "換道"],
}


def _expand_keywords(keyword: str) -> list[str]:
    """將關鍵字展開為含同義詞的搜尋詞組。"""
    terms = [keyword]
    for key, synonyms in _SYNONYM_MAP.items():
        if key in keyword:
            terms.extend(synonyms)
    return list(dict.fromkeys(terms))  # 去重保序


def rag_lookup_tool(keyword: str) -> dict:
    """
    查詢代碼對照表或欄位說明知識庫。

    檢索流程（Week 8 BM25 + Week 2 jieba）：
      1. jieba 斷詞 → BM25 稀疏檢索（rank_bm25）取 top-5 相關條目
      2. 同義詞擴展後對代碼字典做精確 / 模糊比對（補充保底）
      3. 合併去重後回傳

    Parameters
    ----------
    keyword : 查詢關鍵字，例如「07」、「天候 5」、「雨」、「闖紅燈」

    Returns
    -------
    dict with keys: keyword, retrieval_method, expanded_terms, matches, answer, summary
    """
    from bm25_rag import get_rag
    keyword = keyword.strip()

    # ── Step 1：BM25 檢索 ─────────────────────────────────
    rag = get_rag()
    bm25_hits = rag.search(keyword, top_k=5, min_score=0.3)

    retrieval_method = "BM25 + jieba" if rag.ready else "substring fallback"
    matches: list[dict] = []
    seen: set[str] = set()

    for hit in bm25_hits:
        key = f"{hit['type']}:{hit['id']}"
        if key not in seen:
            seen.add(key)
            matches.append({
                "type": hit["type"],
                "code": hit.get("code", ""),
                "description": hit["content"],
                "title": hit["title"],
                "score": hit.get("score", 0),
            })

    # ── Step 2：同義詞擴展 + 精確代碼比對（保底）──────────
    code_dict = load_code_dict()
    code_match = re.search(r"\d+", keyword)
    normalized_code = code_match.group(0).zfill(2) if code_match else None
    search_terms = _expand_keywords(keyword)

    def _add_match(item_type: str, code: str, description: str) -> None:
        key = f"{item_type}:{code}"
        if key not in seen:
            seen.add(key)
            matches.append({"type": item_type, "code": code, "description": description,
                             "title": f"{item_type} {code}", "score": 0})

    # 直接代碼精確比對
    padded = normalized_code or (keyword.zfill(2) if keyword.isdigit() else keyword)
    if padded in code_dict:
        _add_match("肇事因素代碼", padded, code_dict[padded])

    for term in search_terms:
        for code, desc in code_dict.items():
            if term in desc:
                _add_match("肇事因素代碼", code, desc)
        for code, desc in WEATHER_MAPPING.items():
            if term in code or term in desc:
                _add_match("天候代碼", code, desc)

    # ── Step 3：組合回傳 ──────────────────────────────────
    expanded_note = (
        f"（同義詞擴展：{'、'.join(search_terms[1:])}）" if len(search_terms) > 1 else ""
    )
    # 最相關的 BM25 hit 直接作為主要答案
    answer = matches[0]["description"] if matches else None

    summary = (
        f"找到 {len(matches)} 筆與「{keyword}」相關的說明{expanded_note}。"
        if matches
        else f"找不到與「{keyword}」相關的說明，請確認關鍵字。"
    )

    return {
        "keyword": keyword,
        "retrieval_method": retrieval_method,
        "expanded_terms": search_terms,
        "matches": matches[:5],  # 最多回傳前5筆
        "answer": answer,
        "summary": summary,
    }


if __name__ == "__main__":
    print("=== accident_query_tool ===")
    print(accident_query_tool(district="西屯區", hour=18))

    print("\n=== cause_analysis_tool ===")
    print(cause_analysis_tool(district="西屯區", top_n=5))

    print("\n=== rag_lookup_tool ===")
    print(rag_lookup_tool("07"))
