"""Rule-based multi-agent pipeline for the traffic accident demo."""

from __future__ import annotations

import re
from typing import Any

from llm_orchestrator import generate_response
from nlu_parser import parse_user_query_with_llm
try:
    from analysis_tools import (
        accident_query_tool,
        cause_analysis_tool,
        citizen_route_advice_tool,
        rag_lookup_tool,
        recommendation_tool,
        weather_time_heatmap_tool,
    )
    from risk_model import risk_score_tool
except Exception:
    from mock_tools import (
        accident_query_tool,
        cause_analysis_tool,
        citizen_route_advice_tool,
        rag_lookup_tool,
        recommendation_tool,
        risk_score_tool,
        weather_time_heatmap_tool,
    )
from prompts import CLARIFICATION_MESSAGE, DATA_LIMITATION_MESSAGE, OUT_OF_SCOPE_MESSAGE
from text_preprocessor import (
    extract_origin_destination,
    extract_transport_mode,
    resolve_district,
    tokenize,
)


INTENT_HOTSPOT_QUERY = "事故熱點查詢"
INTENT_TIME_QUERY = "時段查詢"
INTENT_CAUSE_QUERY = "肇因分析"
INTENT_RISK_PREDICTION = "風險預測"
INTENT_POLICY_RECOMMENDATION = "政策建議"
INTENT_CODE_LOOKUP = "代碼說明"
INTENT_CITIZEN_ROUTE = "民眾出行建議"
INTENT_OUT_OF_SCOPE = "超出資料範圍"
INTENT_CLARIFICATION_NEEDED = "需要補充條件"

TAICHUNG_DISTRICTS = [
    "中區",
    "東區",
    "南區",
    "西區",
    "北區",
    "西屯區",
    "南屯區",
    "北屯區",
    "豐原區",
    "大里區",
    "太平區",
    "烏日區",
    "霧峰區",
    "后里區",
    "石岡區",
    "東勢區",
    "和平區",
    "新社區",
    "潭子區",
    "大雅區",
    "神岡區",
    "大肚區",
    "沙鹿區",
    "龍井區",
    "梧棲區",
    "清水區",
    "大甲區",
    "外埔區",
    "大安區",
]


def detect_intent(user_input: str) -> dict[str, Any]:
    """Detect query intent with deterministic rules for a stable demo."""
    text = user_input.strip()
    extracted = extract_query_entities(text)

    # ── 地理範圍外（其他縣市，且沒有同時提到台中）──────────
    if any(city in text for city in ["台北", "新北", "桃園", "高雄", "台南"]) and "台中" not in text:
        return _intent(INTENT_OUT_OF_SCOPE, extracted, "目前資料範圍只支援台中市。")

    # ── 明確未來/即時事件：需精準比對，避免「今天是星期五…」被誤判 ──
    # 「明天」「後天」「未來」「即時」「現在會不會」= 真的在問未來
    # 「今天」單獨出現可能只是口語語境，不在此封鎖
    _future_patterns = [
        r"明天", r"後天", r"未來.*事故", r"即時.*事故",
        r"現在會不會", r"等一下會", r"接下來會",
    ]
    if any(re.search(p, text) for p in _future_patterns):
        return _intent(INTENT_OUT_OF_SCOPE, extracted, "目前沒有即時或未來事故資料。")

    # ── 代碼查詢 ──────────────────────────────────────────
    if re.search(r"(代碼|code)\s*\d+", text, flags=re.IGNORECASE) or "欄位" in text:
        return _intent(INTENT_CODE_LOOKUP, extracted, "使用者詢問欄位或代碼說明。")

    # ── 政策建議 ──────────────────────────────────────────
    if any(term in text for term in ["交通局", "政策", "優先改善", "改善哪些", "管理單位", "管理措施", "採取什麼", "應採取"]):
        return _intent(INTENT_POLICY_RECOMMENDATION, extracted, "使用者需要政策或管理建議。")

    # ── 民眾出行建議（路線判斷需在風險詞之前）────────────────
    _route_terms = ["從", "到", "出發", "前往", "去", "騎車", "開車", "走路", "搭車", "路線"]
    _transport_terms = ["機車", "汽車", "行人", "自行車", "大眾運輸", "捷運", "公車"]
    has_route = any(t in text for t in _route_terms)
    has_transport = any(t in text for t in _transport_terms)
    has_from_to = bool(re.search(r"從.{1,8}[到去]", text))
    district_count = sum(1 for d in TAICHUNG_DISTRICTS if d in text)

    if district_count >= 3:
        # 三個以上行政區 → 比較查詢，落入熱點判斷
        pass
    elif district_count >= 2 and (has_route or has_transport):
        return _intent(INTENT_CITIZEN_ROUTE, extracted, "使用者詢問出行路線風險。")
    elif has_from_to and has_route:
        # 「從X到Y」無論有無明確行政區 → 出行建議
        return _intent(INTENT_CITIZEN_ROUTE, extracted, "使用者詢問出行路線風險。")
    elif has_transport and extracted.get("district"):
        # 單一行政區 + 指定交通工具 → 出行建議
        return _intent(INTENT_CITIZEN_ROUTE, extracted, "使用者詢問出行路線風險。")

    # ── 「哪個時段比較安全/出門」→ 時段查詢（問的是找出最佳時段）
    # 僅匹配「比較安全/最安全/出門」等安全建議語境，排除「風險最高」問法
    if re.search(r"哪個時段.*[安全出門]|哪個時間.*[安全出門]|幾點.*最安全", text):
        return _intent(INTENT_TIME_QUERY, extracted, "使用者詢問哪個時段比較安全。")

    # ── 風險預測優先於時段查詢 ────────────────────────────
    # 「容易發生」移除：「雨天比較容易發生事故嗎」屬統計查詢應歸熱點
    _risk_terms = ["危險", "風險", "高嗎", "安全嗎", "會發生", "機率"]
    _time_terms = ["時間", "時段", "幾點", "什麼時候", "哪個時間"]
    has_risk = any(t in text for t in _risk_terms)
    has_time = any(t in text for t in _time_terms)

    # 含「現在/目前/此刻」等即時詞 → 即時資料可補全條件，直接走風險預測
    _realtime_kw = ["現在", "目前", "此刻", "等等", "待會", "出門", "上路"]
    has_realtime = any(kw in text for kw in _realtime_kw)

    # 即時詞 + 「注意/小心/留意」也算即時風險詢問（即使沒有明確風險詞）
    if has_realtime and any(kw in text for kw in ["注意", "小心", "留意", "好嗎", "如何"]):
        has_risk = True

    # 短問句 + 無任何實體 + 非即時問句 → 條件不足
    has_any_entity = any([
        extracted.get("district"),
        extracted.get("hour") is not None,
        extracted.get("weekday"),
        extracted.get("weather"),
    ])
    if has_risk and not has_any_entity and not has_realtime and len(text) <= 7:
        return _intent(INTENT_CLARIFICATION_NEEDED, extracted, "問題太短且缺少查詢條件。")

    if has_risk:
        return _intent(INTENT_RISK_PREDICTION, extracted, "使用者要求風險判斷。")

    if has_time:
        return _intent(INTENT_TIME_QUERY, extracted, "使用者詢問事故高峰時段。")

    # ── 肇因分析 ──────────────────────────────────────────
    if any(term in text for term in ["肇事", "原因", "成因", "外在因素", "車流", "上班族", "通勤", "怎麼改善", "為什麼", "酒駕", "飲酒", "比例"]):
        return _intent(INTENT_CAUSE_QUERY, extracted, "使用者詢問肇事原因。")

    # ── 事故熱點查詢 ──────────────────────────────────────
    if district_count >= 3:
        return _intent(INTENT_HOTSPOT_QUERY, extracted, "使用者比較多個行政區事故。")
    if any(term in text for term in ["哪個區", "行政區", "最多", "熱點", "排名", "最集中", "集中", "比較容易"]):
        return _intent(INTENT_HOTSPOT_QUERY, extracted, "使用者詢問事故熱點或排名。")

    return _intent(INTENT_CLARIFICATION_NEEDED, extracted, "問題缺少明確查詢目的。")


def extract_query_entities(user_input: str) -> dict[str, Any]:
    """Extract entities via jieba tokenization + rule-based matching.

    Fields: district, hour, weekday, month, weather, keyword,
            transport_mode, origin_district, destination_district
    """
    text = user_input.strip()

    # jieba 斷詞（Week 2 技術）：供交通工具提取使用
    tokens = tokenize(text)

    # 基本欄位（行政區：完整名 → 別名 → 地標）
    district = next((name for name in TAICHUNG_DISTRICTS if name in text), None)
    if not district:
        district = resolve_district(text)
    weekday = _extract_weekday(text)
    hour = _extract_hour(text)
    month = _extract_month(text)
    weather = _extract_weather(text)
    code_match = re.search(r"(?:代碼|code)\s*(\d+)", text, flags=re.IGNORECASE)
    keyword = f"代碼 {code_match.group(1)}" if code_match else text

    # 新增：交通工具（jieba token 優先，substring fallback）
    transport_mode = extract_transport_mode(tokens, text)

    # 新增：起訖行政區（從「從X到Y / 從X去Y」模式提取）
    origin_district, destination_district = extract_origin_destination(text, TAICHUNG_DISTRICTS)

    return {
        "district": district,
        "hour": hour,
        "weekday": weekday,
        "month": month,
        "weather": weather,
        "keyword": keyword,
        "transport_mode": transport_mode,
        "origin_district": origin_district,
        "destination_district": destination_district,
    }


def build_tool_plan(intent_result: dict[str, Any]) -> list[str]:
    """Build the ordered tool plan for an intent."""
    intent = intent_result["intent"]
    if intent == INTENT_RISK_PREDICTION:
        return ["risk_score_tool", "accident_query_tool", "weather_time_heatmap_tool", "recommendation_tool"]
    if intent == INTENT_CITIZEN_ROUTE:
        # 加入時段分布資料，讓 LLM 能回答「哪個時段比較安全」
        return ["citizen_route_advice_tool", "accident_query_tool", "weather_time_heatmap_tool"]
    if intent == INTENT_HOTSPOT_QUERY:
        return ["accident_query_tool"]
    if intent == INTENT_TIME_QUERY:
        return ["accident_query_tool", "weather_time_heatmap_tool"]
    if intent == INTENT_CAUSE_QUERY:
        return ["accident_query_tool", "cause_analysis_tool", "recommendation_tool"]
    if intent == INTENT_POLICY_RECOMMENDATION:
        return ["accident_query_tool", "cause_analysis_tool", "recommendation_tool"]
    if intent == INTENT_CODE_LOOKUP:
        return ["rag_lookup_tool"]
    return []


def run_tools(tool_plan: list[str], query: dict[str, Any]) -> dict[str, Any]:
    """Run tools in order and preserve their output by tool name."""
    results: dict[str, Any] = {}
    risk_level = "中風險"
    causes = None

    for tool_name in tool_plan:
        if tool_name == "risk_score_tool":
            result = risk_score_tool(**_tool_query(query))
            risk_level = result["risk"]["level"]
        elif tool_name == "accident_query_tool":
            result = accident_query_tool(**_tool_query(query))
        elif tool_name == "weather_time_heatmap_tool":
            result = weather_time_heatmap_tool(weather=query.get("weather"), hour=query.get("hour"))
        elif tool_name == "cause_analysis_tool":
            result = cause_analysis_tool(district=query.get("district"), top_n=5)
            causes = result.get("causes")
        elif tool_name == "recommendation_tool":
            if "risk_score_tool" in results:
                risk_level = results["risk_score_tool"]["risk"]["level"]
            if "cause_analysis_tool" in results:
                causes = results["cause_analysis_tool"].get("causes")
            result = recommendation_tool(
                risk_level=risk_level,
                causes=causes,
                district=query.get("district"),
                hour=query.get("hour"),
                weather=query.get("weather"),
            )
        elif tool_name == "citizen_route_advice_tool":
            result = citizen_route_advice_tool(
                origin_district=query.get("origin_district") or query.get("district"),
                destination_district=query.get("destination_district"),
                hour=query.get("hour"),
                weather=query.get("weather"),
                transport_mode=query.get("transport_mode", "機車"),
            )
        elif tool_name == "rag_lookup_tool":
            result = rag_lookup_tool(query.get("keyword") or "")
        else:
            result = {"tool": tool_name, "error": "unknown tool"}

        results[tool_name] = result

    return results


def _map_realtime_weather(weather_condition: str) -> str | None:
    """CWA 即時天氣描述 → 系統天候類別（對應資料集代碼）。"""
    if not weather_condition:
        return None
    w = weather_condition
    if any(x in w for x in ["雨", "雷", "陣雨", "豪雨"]):
        return "雨"
    if any(x in w for x in ["霧", "靄", "煙", "煙霧"]):
        return "霧或煙"
    if any(x in w for x in ["雪", "冰"]):
        return "雪"
    if any(x in w for x in ["風沙", "沙塵"]):
        return "風沙"
    if "陰" in w:
        return "陰"
    if any(x in w for x in ["晴", "多雲", "雲"]):
        return "晴"
    return None


def _refresh_traffic_for_districts(
    rt: dict[str, Any],
    query: dict[str, Any],
) -> dict[str, Any]:
    """依查詢的行政區（含起訖點）重新取得分區即時車流摘要。

    讓「從西屯去中區」能拿到「西屯區 35 km/h、中區 43 km/h」的分區車速，
    而非只有全市平均。失敗時保留原本的全市摘要。
    """
    # 起訖點優先；無起訖點才用 district。去重保序
    raw = [query.get("origin_district"), query.get("destination_district")]
    if not any(raw):
        raw = [query.get("district")]
    focus = list(dict.fromkeys([d for d in raw if d]))
    if not focus:
        return rt

    try:
        from external_api_tools import fetch_tdx_traffic_tool, estimate_hourly_traffic
        traffic = fetch_tdx_traffic_tool(districts=focus)
        if traffic.get("available") and traffic.get("summary"):
            # 即時車流可用
            rt = {**rt,
                  "traffic_available": True,
                  "traffic_summary": traffic["summary"],
                  "traffic_avg_speed": traffic.get("avg_speed"),
                  "traffic_level": traffic.get("level", "")}
        else:
            # 即時不可用 → 用歷史時段密度推估
            est = estimate_hourly_traffic(query.get("hour"))
            if est.get("available"):
                rt = {**rt,
                      "traffic_available": True,
                      "traffic_summary": est["summary"],
                      "traffic_level": est.get("level", ""),
                      "traffic_is_estimate": True}
    except Exception:
        pass
    return rt


def _inject_realtime_query(
    query: dict[str, Any],
    rt: dict[str, Any],
    user_input: str = "",
) -> None:
    """將即時時間/天候補入 query（僅在使用者未指定時補入）。

    目的：讓「北區現在上路很危險嗎」自動帶入當前時段與天候，
    使 RF 模型預測「當前條件」而非「無條件平均」。

    注意：若問題是「哪個時段比較安全」等比較型問句，
    不補入當前時段（讓 LLM 以全時段視角回答）。
    """
    # 比較型問句不補時段（使用者要的是「哪個時段」比較，非現在這個時段）
    _time_compare_kw = ["哪個時段", "哪個時間", "什麼時段", "幾點最", "哪個時候"]
    asking_which_time = any(kw in user_input for kw in _time_compare_kw)

    # 補入當前時段（使用者未指定 hour 且非比較型問句）
    if query.get("hour") is None and rt.get("hour") is not None and not asking_which_time:
        query["hour"] = rt["hour"]

    # 補入當前星期（使用者未指定 weekday 時）
    if not query.get("weekday") and rt.get("weekday"):
        query["weekday"] = rt["weekday"]

    # 補入當前月份
    if query.get("month") is None:
        try:
            from datetime import datetime
            query["month"] = datetime.now().month
        except Exception:
            pass

    # 補入即時天候（使用者未指定 weather，且 CWA 資料可用時）
    if not query.get("weather") and rt.get("weather_available"):
        mapped = _map_realtime_weather(rt.get("weather_condition", ""))
        if mapped:
            query["weather"] = mapped


def run_agent_pipeline(
    user_input: str,
    form_input: dict[str, Any] | None = None,
    user_role: str = "一般民眾",
    realtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full B-side agent pipeline."""
    rule_intent_result = detect_intent(user_input)
    llm_parse = parse_user_query_with_llm(user_input, rule_result=rule_intent_result)
    intent_result = _choose_intent_result(rule_intent_result, llm_parse)
    query = {**intent_result.get("entities", {})}
    if form_input:
        query.update({key: value for key, value in form_input.items() if value not in (None, "", "不指定")})
        intent_result["entities"] = query

    # ── 即時背景資訊補入 query ─────────────────────────────
    # 規則：使用者有填條件（時段/星期/天候）就照條件；沒填就用「現在」。
    _REALTIME_INTENTS = {INTENT_RISK_PREDICTION, INTENT_CITIZEN_ROUTE}
    ent = intent_result.get("entities", {})
    user_specified_time = bool(
        ent.get("hour") is not None or ent.get("weekday") or ent.get("weather")
    )
    use_realtime = (
        realtime_context is not None
        and intent_result["intent"] in _REALTIME_INTENTS
        and not user_specified_time   # 沒指定任何時間/天候條件 → 用現在
    )
    if use_realtime:
        _inject_realtime_query(query, realtime_context, user_input=user_input)
        intent_result["entities"] = query
        realtime_context = _refresh_traffic_for_districts(realtime_context, query)
    else:
        # 使用者已指定條件 → 不注入即時背景，完全照使用者情境回答
        realtime_context = None

    # ── 問題中提到角色時自動採用對應角色 ─────────────────────
    _ROLE_KEYWORDS = {
        "交通局": "交通局",
        "警察": "警察/交通大隊",
        "交通大隊": "警察/交通大隊",
        "道路工程": "道路工程單位",
        "工程單位": "道路工程單位",
    }
    for kw, role in _ROLE_KEYWORDS.items():
        if kw in user_input:
            user_role = role
            break

    if intent_result["intent"] == INTENT_CLARIFICATION_NEEDED and _has_enough_form_query(query):
        intent_result = _intent(INTENT_RISK_PREDICTION, query, "表單提供足夠條件，執行風險預測。")

    tool_plan = build_tool_plan(intent_result)
    tool_results = run_tools(tool_plan, query)

    pipeline_result = {
        "user_input": user_input,
        "intent": intent_result,
        "query": query,
        "tool_plan": tool_plan,
        "tool_results": tool_results,
        "user_role": user_role,
        "realtime_context": realtime_context or {},
        "nlu": {
            "rule_intent": rule_intent_result,
            "llm_parse": _safe_nlu_debug(llm_parse),
            "selected_source": intent_result.get("source", "rules"),
        },
        "agent_steps": build_agent_steps(intent_result, tool_plan),
        "clarification_message": CLARIFICATION_MESSAGE,
    }
    response = build_response(pipeline_result)
    critic = critic_check(response, tool_results, intent_result)
    pipeline_result["response"] = response
    pipeline_result["critic"] = critic
    return pipeline_result


def build_response(pipeline_result: dict[str, Any]) -> str:
    """Build final answer through the response generation layer."""
    response, metadata = generate_response(pipeline_result)
    pipeline_result["response_generation"] = metadata
    pipeline_result["llm_thinking"] = metadata.get("thinking", "")
    return response


def critic_check(
    response: str,
    tool_results: dict[str, Any],
    intent_result: dict[str, Any],
) -> dict[str, Any]:
    """Check the final answer for grounding and scope."""
    issues = []
    if intent_result["intent"] not in {INTENT_OUT_OF_SCOPE, INTENT_CLARIFICATION_NEEDED} and not tool_results:
        issues.append("沒有工具結果可支持回答。")
    if "根據本資料集" not in response and intent_result["intent"] not in {INTENT_CLARIFICATION_NEEDED, INTENT_OUT_OF_SCOPE}:
        issues.append("回答未明確標示資料依據。")
    if "資料限制" not in response and intent_result["intent"] != INTENT_CLARIFICATION_NEEDED:
        issues.append("回答未說明資料限制。")
    if any(term in response for term in ["一定會發生", "保證會", "保證不會"]):
        issues.append("回答可能把歷史趨勢誤述為未來確定事件。")

    return {
        "agent": "Critic Agent",
        "passed": len(issues) == 0,
        "issues": issues,
        "message": "回答已通過基本資料依據檢查。" if not issues else "回答仍有需要修正的地方。",
    }


def build_agent_steps(intent_result: dict[str, Any], tool_plan: list[str]) -> list[dict[str, Any]]:
    """Return display-friendly agent progress."""
    source = intent_result.get("source", "rules")
    source_label = "本機 LLM NLU" if source == "local_llm" else "規則式 NLU"
    steps = [
        {
            "agent": "NLU Parser",
            "status": "完成",
            "detail": f"使用{source_label}解析自然語言問題。",
        },
        {
            "agent": "Intent Agent",
            "status": "完成",
            "detail": f"判斷為「{intent_result['intent']}」：{intent_result.get('reason')}",
        }
    ]
    tool_agent_map = {
        "accident_query_tool": "Data Agent",
        "risk_score_tool": "Prediction Agent",
        "weather_time_heatmap_tool": "Data Agent",
        "cause_analysis_tool": "Data Agent",
        "rag_lookup_tool": "RAG Agent",
        "recommendation_tool": "Decision Agent",
        "citizen_route_advice_tool": "Route Agent",
    }
    for tool_name in tool_plan:
        steps.append(
            {
                "agent": tool_agent_map.get(tool_name, "Tool Agent"),
                "status": "完成",
                "detail": f"呼叫 {tool_name}",
            }
        )
    steps.append({"agent": "Response Agent", "status": "完成", "detail": "整合工具結果並產生自然語言回答。"})
    steps.append({"agent": "Critic Agent", "status": "完成", "detail": "檢查回答是否根據工具結果與資料範圍。"})
    return steps


def _intent(intent: str, entities: dict[str, Any], reason: str) -> dict[str, Any]:
    return {"intent": intent, "entities": entities, "reason": reason}


def _choose_intent_result(rule_result: dict[str, Any], llm_parse: dict[str, Any]) -> dict[str, Any]:
    """Prefer valid local LLM NLU, but keep deterministic safety boundaries."""
    # 規則強制保護的 intent：LLM 不可覆蓋
    _RULE_PROTECTED = {
        INTENT_OUT_OF_SCOPE,    # 其他縣市、未來預測
        INTENT_CITIZEN_ROUTE,   # 明確路線模式（從X到Y）
    }
    if rule_result["intent"] in _RULE_PROTECTED:
        return {**rule_result, "source": "rules"}

    if not llm_parse.get("ok"):
        return {**rule_result, "source": "rules"}

    if llm_parse.get("confidence", 0) < 0.55:
        return {**rule_result, "source": "rules"}

    intent = llm_parse.get("intent")
    if intent == "民眾出行建議":
        intent = INTENT_CITIZEN_ROUTE

    # 規則式已判定為可回答的 intent，LLM 不應降級為「需要補充條件」
    # （即時問句如「現在開車危險嗎」有即時資料可補全，不需再問使用者）
    if (intent == INTENT_CLARIFICATION_NEEDED
            and rule_result["intent"] != INTENT_CLARIFICATION_NEEDED):
        return {**rule_result, "source": "rules"}

    entities = {**rule_result.get("entities", {}), **llm_parse.get("entities", {})}
    return {
        "intent": intent,
        "entities": entities,
        "reason": f"{llm_parse.get('reason', '本機 LLM 完成解析')}（confidence={llm_parse.get('confidence'):.2f}）",
        "source": "local_llm",
    }


def _safe_nlu_debug(llm_parse: dict[str, Any]) -> dict[str, Any]:
    debug = {key: value for key, value in llm_parse.items() if key != "raw"}
    if "raw" in llm_parse:
        raw = str(llm_parse["raw"])
        debug["raw_preview"] = raw[:500]
    return debug


def _tool_query(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "district": query.get("district"),
        "hour": query.get("hour"),
        "weekday": query.get("weekday"),
        "month": query.get("month"),
        "weather": query.get("weather"),
    }


def _has_enough_form_query(query: dict[str, Any]) -> bool:
    return bool(query.get("district") or query.get("hour") or query.get("weather"))


def _extract_weekday(text: str) -> str | None:
    mapping = {
        "星期一": "星期一",
        "週一": "星期一",
        "禮拜一": "星期一",
        "星期二": "星期二",
        "週二": "星期二",
        "禮拜二": "星期二",
        "星期三": "星期三",
        "週三": "星期三",
        "禮拜三": "星期三",
        "星期四": "星期四",
        "週四": "星期四",
        "禮拜四": "星期四",
        "星期五": "星期五",
        "週五": "星期五",
        "禮拜五": "星期五",
        "星期六": "星期六",
        "週六": "星期六",
        "禮拜六": "星期六",
        "星期日": "星期日",
        "星期天": "星期日",
        "週日": "星期日",
        "週末": "星期六",
    }
    return next((value for key, value in mapping.items() if key in text), None)


def _extract_hour(text: str) -> int | None:
    # 時段口語關鍵詞（優先，在數字解析前）
    _period_map = [
        (["凌晨", "半夜", "深夜兩點", "深夜三點"], 2),
        (["清晨", "天亮"], 6),
        (["早上上班", "上班尖峰", "早上通勤"], 8),
        (["中午", "午休"], 12),
        (["下午兩點", "午後"], 14),
        (["下班", "傍晚", "晚上下班", "放學", "下班尖峰"], 18),
        (["深夜", "半夜"], 23),
    ]
    for keywords, default_hour in _period_map:
        if any(kw in text for kw in keywords):
            # 若有更精確的數字在同一段，優先用數字解析，否則用預設
            break

    # 帶前綴的精確時刻（優先解析）
    _prefix_patterns = [
        (r"凌晨\s*(\d{1,2})\s*點",   0,   False),
        (r"清晨\s*(\d{1,2})\s*點",   0,   False),
        (r"早上\s*(\d{1,2})\s*點",   0,   False),
        (r"上午\s*(\d{1,2})\s*點",   0,   False),
        (r"中午\s*(\d{1,2})\s*點",   0,   False),
        (r"下午\s*(\d{1,2})\s*點",   12,  True),
        (r"傍晚\s*(\d{1,2})\s*點",   12,  True),
        (r"晚上\s*(\d{1,2})\s*點",   12,  True),
        (r"深夜\s*(\d{1,2})\s*點",   0,   False),
        (r"(\d{1,2})\s*[:：]\s*\d{2}", 0, False),  # HH:MM
        (r"(\d{1,2})\s*點",           0,   False),  # 純數字時
    ]
    for pattern, add_if_lt12, do_add in _prefix_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        hour = int(match.group(1))
        if do_add and hour < 12:
            hour += add_if_lt12
        if 0 <= hour <= 23:
            return hour

    # 純口語時段（無數字）
    _keyword_hours = [
        (["凌晨", "半夜"], 2),
        (["清晨", "天未亮"], 5),
        (["早上", "上午", "上班尖峰", "早尖峰"], 8),
        (["中午", "午休"], 12),
        (["下午"], 14),
        (["傍晚", "黃昏", "下班", "晚尖峰", "下班時間", "放學"], 18),
        (["晚上"], 20),
        (["深夜", "夜間"], 23),
    ]
    for keywords, hour in _keyword_hours:
        if any(kw in text for kw in keywords):
            return hour

    return None


_CHINESE_NUM = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}


def _extract_month(text: str) -> int | None:
    # 阿拉伯數字：3月、03月
    match = re.search(r"(\d{1,2})\s*月", text)
    if match:
        month = int(match.group(1))
        return month if 1 <= month <= 12 else None
    # 中文數字：三月、十一月
    for zh, num in sorted(_CHINESE_NUM.items(), key=lambda x: -len(x[0])):
        if f"{zh}月" in text:
            return num
    return None


def _extract_weather(text: str) -> str | None:
    # 順序重要：長字串（霧或煙、風沙）必須在子字串（霧、煙、風）之前檢查
    # 資料值對照：WEATHER_MAPPING = {"3": "霧或煙", "5": "雨", "6": "陰", "7": "晴", ...}
    WEATHER_PATTERNS = [
        ("霧或煙",  ["霧或煙", "起霧", "有霧", "有煙", "煙霧", "霧天", "霧大"]),
        ("風沙",    ["風沙"]),
        ("雨",      ["雨", "下雨", "雨天", "雨勢", "降雨"]),
        ("晴",      ["晴", "晴天", "放晴", "好天氣"]),
        ("陰",      ["陰天", "陰", "多雲"]),
        ("雪",      ["雪", "下雪", "積雪"]),
        ("風",      ["風", "有風", "強風", "颳風"]),
    ]
    for data_value, keywords in WEATHER_PATTERNS:
        if any(kw in text for kw in keywords):
            return data_value
    return None
