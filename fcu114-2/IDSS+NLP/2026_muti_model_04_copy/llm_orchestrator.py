"""Response generation layer.

The demo defaults to deterministic rule-based generation so it remains stable
without API keys. A future LLM integration can call the same public function.
"""

from __future__ import annotations

import json
import re
from typing import Any

from local_llm_client import LocalLLMClient, LocalLLMError
from prompts import (
    DATA_LIMITATION_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    RESPONSE_AGENT_SYSTEM_PROMPT,
    RESPONSE_TEMPLATE,
)


def _parse_thinking(raw: str) -> tuple[str, str]:
    """從 Qwen3 回應中分離 <think>...</think> 與最終答案。

    Returns:
        (thinking_text, response_text)
    """
    match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        response = raw[match.end():].strip()
        return thinking, response
    return "", raw.strip()


def generate_response(pipeline_result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Generate the final response with local LLM first, then rule fallback."""
    try:
        response, thinking = generate_response_with_llm(pipeline_result)
        return response, {"mode": "local_llm", "fallback": False, "thinking": thinking}
    except LocalLLMError as exc:
        response = generate_response_with_rules(pipeline_result)
        return response, {"mode": "rules", "fallback": True, "error": str(exc), "thinking": ""}


def generate_response_with_rules(pipeline_result: dict[str, Any]) -> str:
    """Build a grounded final response from tool outputs."""
    intent = pipeline_result.get("intent", {})
    tool_results = pipeline_result.get("tool_results", {})

    if intent.get("intent") == "超出資料範圍":
        return (
            f"{OUT_OF_SCOPE_MESSAGE}\n\n"
            f"資料限制：\n{DATA_LIMITATION_MESSAGE}"
        )

    if intent.get("intent") == "需要補充條件":
        return pipeline_result.get("clarification_message", "請補充更多查詢條件。")

    if "rag_lookup_tool" in tool_results and len(tool_results) == 1:
        rag = tool_results["rag_lookup_tool"]
        answer = rag.get("answer") or _format_rag_matches(rag)
        return f"根據本資料集與代碼對照資料：\n\n{answer}\n\n資料限制：\n{DATA_LIMITATION_MESSAGE}"

    query = _first_query(tool_results)
    query_lines = _format_query(query)
    main_result = _format_main_result(tool_results)
    evidence_lines = _format_evidence(tool_results)
    reason_lines = _format_reasons(tool_results)
    decision_lines = _format_decision_support(tool_results, intent.get("intent"))
    recommendation_lines = _format_recommendations(tool_results)

    return RESPONSE_TEMPLATE.format(
        query_lines=query_lines,
        main_result=main_result,
        evidence_lines=evidence_lines,
        reason_lines=reason_lines,
        decision_lines=decision_lines,
        recommendation_lines=recommendation_lines,
        limitation=DATA_LIMITATION_MESSAGE,
    ).strip()


def generate_response_with_llm(pipeline_result: dict[str, Any]) -> tuple[str, str]:
    """Generate a grounded Chinese response through the configured local LLM.

    Flow: 程式組裝事實卡片 → 精簡 prompt → LLM 口語化 → 輸出清洗
    Returns: (response_text, thinking_text)
    """
    intent = pipeline_result.get("intent", {})
    if intent.get("intent") == "需要補充條件":
        clarification = pipeline_result.get("clarification_message", "請補充更多查詢條件。")
        return clarification, ""

    prompt = _build_grounded_response_prompt(pipeline_result)
    messages = [
        {"role": "system", "content": RESPONSE_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    # 關閉 think（4B 思考品質不穩且拖慢）；低溫求穩定；事實卡片已限制內容
    # max_tokens 給較大空間：推理模型（deepseek-r1）需要思考 token，
    # 一般模型不會用滿。思考內容由 _parse_thinking / _clean_response 剝除。
    raw = LocalLLMClient().chat(messages, temperature=0.3, max_tokens=1200, think=False).strip()
    # 殘留 <think> 仍清掉，再做輸出清洗
    _, raw = _parse_thinking(raw)
    response = _clean_response(raw)
    return response, ""


# 需清除的 meta 內容樣式（模型偶爾會把指令執行確認寫進正文）
_META_PATTERNS = [
    re.compile(r"[（(]\s*字數[：:].*?[）)]"),         # （字數：58，符合120字內要求）
    re.compile(r"[（(]\s*共?\s*\d+\s*字[）)]"),        # （58字）（共58字）
    re.compile(r"[（(]\s*說明[：:].*?[）)]", re.DOTALL),  # （說明：根據JSON...）
    re.compile(r"[（(]\s*註[：:].*?[）)]", re.DOTALL),     # （註：...）
    re.compile(r"[（(]\s*備註[：:].*?[）)]", re.DOTALL),
    re.compile(r"符合\s*\d+\s*字.*?要求[。．]?"),
    re.compile(r"<think>.*?</think>", re.DOTALL),
    re.compile(r"</?think>"),
]

# 開頭贅詞
_LEADING_FILLER = re.compile(
    r"^(好的[，,、]?|您好[！!]?|根據(您的|你的|提供的).{0,12}?[，,：:]|"
    r"以下是.{0,10}?[：:]|回答如下[：:]?)\s*"
)


def _clean_response(text: str) -> str:
    """清除模型誤吐的 meta 內容、字數標注、開頭贅詞、過多語助詞。"""
    if not text:
        return text
    for pat in _META_PATTERNS:
        text = pat.sub("", text)
    text = _LEADING_FILLER.sub("", text.strip())
    # 收斂語助詞：
    # ① 開頭的「喔！」「喔，」「欸，」等
    text = re.sub(r"^[喔啦哦呢嘛囉欸][～~]?[！，、。]?\s*", "", text.strip())
    # ② 語助詞夾在中文字之間當停頓（行為啦～開車）→ 換成逗號
    text = re.sub(r"(?<=[一-鿿])[喔啦哦呢嘛囉][～~]+(?=[一-鿿])", "，", text)
    # ③ 其餘帶波浪號的語助詞無條件移除
    text = re.sub(r"[喔啦哦呢嘛囉][～~]+", "", text)
    # ③ 語助詞後接標點或結尾 → 移除語助詞
    text = re.sub(r"[喔啦哦呢嘛囉](?=[，。！、？])", "", text)
    text = re.sub(r"[喔啦哦呢嘛囉]$", "", text.rstrip())
    # ④ 殘留波浪號、重複驚嘆號
    text = re.sub(r"[～~]+", "", text)
    text = re.sub(r"！+", "！", text)
    # 若清完開頭變成標點，去掉
    text = re.sub(r"^[，、。！？]+", "", text.strip())
    # 移除多餘空行與行首尾空白
    lines = [ln.strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    text = text.strip()
    # 句尾補標點
    if text and text[-1] not in "。！？，、：；":
        text += "。"
    return text


def _retrieve_rag_context(user_input: str, query: dict[str, Any]) -> str:
    """BM25 RAG 擷取與問題相關的背景知識，供 LLM 回答時參考。

    對應課程：NLP Week 8（BM25 稀疏檢索 + jieba 斷詞）
    與工具結果的分工：
      - 工具結果  = 真實事故數字（由 analysis_tools 查詢 CSV 計算）
      - RAG 知識  = 欄位定義、代碼解釋、天候說明等背景脈絡
    """
    try:
        from bm25_rag import get_rag
        rag = get_rag()

        # 搜尋詞：使用者原始問題 + 關鍵條件詞（天候、行政區）
        parts = [user_input]
        if query.get("weather"):
            parts.append(query["weather"])
        if query.get("district"):
            parts.append(query["district"])

        hits = rag.search(" ".join(parts), top_k=3, min_score=0.3)
        if not hits:
            return ""

        lines = [
            f"- {h['title']}：{h['content'][:90].rstrip('。')}。"
            for h in hits
        ]
        return "\n".join(lines)
    except Exception:
        return ""


_ROLE_GUIDANCE: dict[str, str] = {
    "一般民眾":                     "請以一般民眾角度回答，重點是出行安全建議（何時出發、如何避開風險）。",
    "公家單位（交通局/警察/工程）":   "請以公家管理單位角度回答，重點是政策方向、勤務佈署、資源配置與道路改善。",
    "交通局":                        "請以交通管理單位角度回答，重點是政策方向、資源配置與路口管理。",
    "警察/交通大隊":                  "請以執法單位角度回答，重點是勤務佈署、執法時段與取締重點。",
    "道路工程單位":                   "請以工程單位角度回答，重點是路口設計、號誌標線與硬體改善。",
}


_top_stats_cache: dict[str, Any] = {}


def _get_top_stats() -> dict[str, Any]:
    """從事故資料計算 top 時段與 top 行政區（快取）。"""
    if _top_stats_cache:
        return _top_stats_cache
    try:
        from data_loader import load_accident_data
        df = load_accident_data()
        hours = df["hour"].value_counts().head(5)
        dists = df["區"].value_counts().head(5)
        _top_stats_cache["hours"] = [(int(h), int(c)) for h, c in hours.items()]
        _top_stats_cache["districts"] = [(str(d), int(c)) for d, c in dists.items()]
    except Exception:
        _top_stats_cache["hours"] = []
        _top_stats_cache["districts"] = []
    return _top_stats_cache


def _build_fact_card(pipeline_result: dict[str, Any]) -> str:
    """從 tool_results 抽取關鍵事實，組成人類可讀的卡片（取代丟整包 JSON）。

    目的：LLM 只能用卡片內的事實，無法自行挖 JSON 或編造數字。
    """
    intent_name = (pipeline_result.get("intent") or {}).get("intent", "")
    query = pipeline_result.get("query") or {}
    tr = pipeline_result.get("tool_results") or {}
    rt = pipeline_result.get("realtime_context") or {}
    use_realtime = bool(rt and rt.get("datetime_str"))
    lines: list[str] = []

    # ── 情境（查詢條件）──────────────────────────────────
    cond = []
    if query.get("district"):
        cond.append(query["district"])
    if query.get("origin_district") and query.get("destination_district"):
        cond.append(f"{query['origin_district']}→{query['destination_district']}")
    if query.get("transport_mode"):
        cond.append(query["transport_mode"])
    if query.get("weekday"):
        cond.append(query["weekday"])
    if query.get("hour") is not None:
        cond.append(f"{query['hour']}時")
    if query.get("weather"):
        cond.append(f"{query['weather']}天")
    if cond:
        tag = "（即時）" if use_realtime else "（指定情境）"
        lines.append(f"查詢情境{tag}：{'、'.join(cond)}")

    # ── 風險分數 ─────────────────────────────────────────
    risk = (tr.get("risk_score_tool") or {}).get("risk") or {}
    if risk.get("level"):
        lines.append(f"風險評估：{risk['level']}（{risk.get('score','?')}分，由 Random Forest 模型預測）")
    sev = (tr.get("risk_score_tool") or {}).get("severity") or {}
    if sev.get("accident_count"):
        lines.append(f"此條件歷史事故：{sev['accident_count']}件、死亡{sev.get('deaths',0)}人、受傷{sev.get('injuries',0)}人")

    # ── 出行建議：起訖區各用同一個 RF 模型評分（與風險預測一致）──
    if intent_name == "民眾出行建議":
        try:
            from risk_model import risk_score_tool
            seen_d = []
            for d in [query.get("origin_district"), query.get("destination_district"), query.get("district")]:
                if d and d not in seen_d:
                    seen_d.append(d)
            for d in seen_d:
                rr = risk_score_tool(
                    district=d, hour=query.get("hour"),
                    weekday=query.get("weekday"), weather=query.get("weather"),
                ).get("risk", {})
                if rr.get("level"):
                    lines.append(f"{d} 風險：{rr['level']}（{rr.get('score','?')}分，RF 模型）")
            if query.get("transport_mode"):
                lines.append(f"交通工具：{query['transport_mode']}（機車/自行車對天候、路況較敏感）")
        except Exception:
            pass

    # ── 行政區/時段排名依據 ──────────────────────────────
    ev = (tr.get("risk_score_tool") or {}).get("evidence") or {}
    if ev.get("district_rank"):
        lines.append(f"{query.get('district','該區')}事故全市排名第{ev['district_rank']}（{ev.get('district_count','?')}件）")
    if ev.get("hour_rank"):
        lines.append(f"{query.get('hour','?')}時事故時段排名第{ev['hour_rank']}")

    # ── 即時背景（時間/天氣/車流）────────────────────────
    if use_realtime:
        bg = [f"現在 {rt['datetime_str']}（{rt.get('weekday','')} {rt.get('hour_label','')}）"]
        if rt.get("weather_available") and rt.get("weather_condition"):
            t = f"{rt.get('temperature')}°C" if rt.get('temperature') else ""
            bg.append(f"天氣{rt['weather_condition']}{t}")
        if rt.get("traffic_available") and rt.get("traffic_summary"):
            bg.append(rt["traffic_summary"].replace("即時路況：", ""))
        lines.append("即時背景：" + "；".join(bg))

    # ── 各 intent 專屬事實（從資料算 top N）──────────────
    if intent_name == "時段查詢":
        hrs = _get_top_stats().get("hours") or []
        if hrs:
            top = "、".join(f"{h}時({c:,}件)" for h, c in hrs[:3])
            least = hrs[-1] if len(hrs) >= 5 else None
            lines.append(f"事故最多時段：{top}")
            lines.append("（高峰多為上下班通勤時段，凌晨事故最少）")
    if intent_name == "事故熱點查詢":
        dts = _get_top_stats().get("districts") or []
        if dts:
            top = "、".join(f"{d}({c:,}件)" for d, c in dts[:5])
            lines.append(f"事故最多行政區（前5名）：{top}")
    if intent_name in ("肇因分析", "政策建議"):
        causes = (tr.get("cause_analysis_tool") or {}).get("causes") or []
        for c in causes[:3]:
            lines.append(f"肇因 代碼{c['code']} {c['description']}：{c['count']:,}件")

    # ── RAG 知識（僅代碼說明類查詢使用，避免無關肇因代碼干擾風險判斷）──
    rag = tr.get("rag_lookup_tool") or {}
    if rag.get("answer"):
        lines.append(f"知識庫：{rag['answer']}")
    elif intent_name == "代碼說明":
        rag_ctx = _retrieve_rag_context(pipeline_result.get("user_input", ""), query)
        if rag_ctx:
            lines.append("背景參考：\n" + rag_ctx)

    return "\n".join(f"- {ln}" for ln in lines) if lines else "- （無可用數據，請說明資料不足）"


# ── 各 intent 的回答口吻指引（精簡版）─────────────────────
_INTENT_TONE = {
    "風險預測":     "先講風險等級和分數，用白話說一句為什麼，最後給1個現在能做的小提醒。",
    "民眾出行建議": "先講起訖區風險，若有各區車速就點出哪區比較塞，給1個出發建議（早點走/改道/放慢）。",
    "時段查詢":     "直接說哪個時段事故最多、哪個最少，講出具體時間和件數。",
    "事故熱點查詢": "直接說事故最多的行政區和件數，可順帶提前三名。",
    "肇因分析":     "列出前三大肇事原因（說明+件數），語氣像在跟人解釋。",
    "政策建議":     "針對對象給2-3個具體可做的建議，務實不空泛。",
    "代碼說明":     "用一兩句把代碼或欄位的意思講清楚。",
}


def _build_grounded_response_prompt(pipeline_result: dict[str, Any]) -> str:
    """精簡 prompt：角色 + 事實卡片 + 口吻指引。不丟 JSON，避免小模型混亂。"""
    user_input = pipeline_result.get("user_input", "")
    user_role = pipeline_result.get("user_role", "一般民眾")
    intent_name = (pipeline_result.get("intent") or {}).get("intent", "")

    is_public = "民眾" in user_role
    fact_card = _build_fact_card(pipeline_result)
    tone = _INTENT_TONE.get(intent_name, "先給核心結論，再補一句說明。")

    if is_public:
        role_label = "一般民眾"
        style_line = (
            "請用自然、清楚的語氣回答，像在跟人講話但不囉嗦。"
            "【重要】整段最多只能用一個語助詞，禁止連續使用「喔」「啦」「哦」「呢」「啦～」這類詞，"
            "尤其不要每句結尾都加語助詞，保持乾淨俐落。"
        )
        length_line = "- 60 字內，自然口語但簡潔，不要寫成報告或條列。"
    else:
        role_label = "公家單位（交通局／警察／工程單位）"
        style_line = (
            "請用專業、正式、務實的語氣回答，像對機關承辦人員做簡報。"
            "不要用「嘿」「朋友」「喔」「啦」等輕鬆口語詞，直接切入重點與建議。"
        )
        length_line = "- 100 字內，條理清楚、語氣正式，可分點列出具體措施。"

    return f"""你是台中市交通事故風險決策支援系統，回答對象是「{role_label}」。
{style_line}

使用者問：{user_input}

可用事實（只能根據這些回答，不可自己新增或估算任何數字）：
{fact_card}

回答方式：{tone}

注意：
{length_line}
- 只陳述事實與建議，不要解釋你的思考過程，不要寫字數，不要加括號補充說明。
- 不用每次聲明「歷史資料」，除非使用者問「保證/一定會不會出事」。""".strip()


def _first_query(tool_results: dict[str, Any]) -> dict[str, Any]:
    for result in tool_results.values():
        query = result.get("query")
        if query:
            return query
    return {}


def _format_query(query: dict[str, Any]) -> str:
    labels = {
        "district": "行政區",
        "hour": "時段",
        "weekday": "星期",
        "month": "月份",
        "weather": "天候",
    }
    lines = []
    for key, label in labels.items():
        value = query.get(key)
        if value not in (None, ""):
            suffix = " 時" if key == "hour" and isinstance(value, int) else ""
            lines.append(f"- {label}: {value}{suffix}")
    return "\n".join(lines) if lines else "- 未指定特定條件"


def _format_main_result(tool_results: dict[str, Any]) -> str:
    risk_result = tool_results.get("risk_score_tool")
    if risk_result:
        risk = risk_result.get("risk", {})
        return f"風險等級：{risk.get('level', '未判定')}，風險分數：{risk.get('score', '無資料')} / 100"

    accident_result = tool_results.get("accident_query_tool")
    if accident_result:
        return f"分析結果：{accident_result.get('summary')}"

    cause_result = tool_results.get("cause_analysis_tool")
    if cause_result:
        return f"分析結果：{cause_result.get('summary')}"

    return "分析結果：已完成查詢。"


def _format_evidence(tool_results: dict[str, Any]) -> str:
    lines = []
    risk_result = tool_results.get("risk_score_tool")
    if risk_result:
        evidence = risk_result.get("evidence", {})
        if evidence.get("district_rank"):
            lines.append(f"- 行政區排名第 {evidence['district_rank']}，事故數 {evidence['district_count']} 筆")
        if evidence.get("hour_rank"):
            lines.append(f"- 時段排名第 {evidence['hour_rank']}，事故數 {evidence['hour_count']} 筆")
        breakdown = risk_result.get("risk", {}).get("breakdown", {})
        if breakdown:
            bonus = breakdown.get("複合風險加成", 0)
            bonus_str = f"、複合加成 {bonus}" if bonus > 0 else ""
            lines.append(
                "- 評分組成：行政區 {district}、時段 {hour}、天候 {weather}、星期 {weekday}、月份 {month}{bonus}".format(
                    district=breakdown.get("行政區分數", 0),
                    hour=breakdown.get("時段分數", 0),
                    weather=breakdown.get("天候分數", 0),
                    weekday=breakdown.get("星期分數", 0),
                    month=breakdown.get("月份分數", 0),
                    bonus=bonus_str,
                )
            )
        severity = risk_result.get("severity", {})
        if severity.get("accident_count"):
            lines.append(
                f"- 此條件下事故 {severity['accident_count']} 件，"
                f"死亡 {severity['deaths']} 人、受傷 {severity['injuries']} 人"
                f"（{severity.get('severity_label', '')}）"
            )

    accident_result = tool_results.get("accident_query_tool")
    if accident_result:
        summary = accident_result.get("summary")
        if summary:
            lines.append(f"- {summary}")

    cause_result = tool_results.get("cause_analysis_tool")
    if cause_result:
        for cause in cause_result.get("causes", [])[:3]:
            lines.append(f"- 主因 {cause['code']}「{cause['description']}」: {cause['count']} 筆")
        for cause in cause_result.get("secondary_causes", [])[:2]:
            lines.append(f"- 次因 {cause['code']}「{cause['description']}」: {cause['count']} 筆")

    heatmap_result = tool_results.get("weather_time_heatmap_tool")
    if heatmap_result:
        lines.append(f"- {heatmap_result.get('summary')}")

    return "\n".join(lines) if lines else "- 目前工具未回傳可引用的數據。"


def _format_reasons(tool_results: dict[str, Any]) -> str:
    reasons = []
    risk_result = tool_results.get("risk_score_tool")
    if risk_result:
        reasons.extend(risk_result.get("reasons", []))
    cause_result = tool_results.get("cause_analysis_tool")
    if cause_result:
        reasons.append(cause_result.get("summary"))
    if not reasons:
        accident_result = tool_results.get("accident_query_tool")
        if accident_result:
            reasons.append(accident_result.get("summary"))

    return "\n".join(f"{index}. {reason}" for index, reason in enumerate(reasons, start=1) if reason) or "1. 目前沒有足夠原因資料。"


def _format_decision_support(tool_results: dict[str, Any], intent: str | None) -> str:
    rec_result = tool_results.get("recommendation_tool", {})
    decision = rec_result.get("decision_support", {})
    lines = []

    if decision.get("target_user"):
        lines.append(f"- 使用者：{decision['target_user']}")
    if decision.get("priority"):
        lines.append(f"- 處理優先順序：{decision['priority']}")
    if decision.get("decision_basis"):
        lines.append(f"- 判斷依據：{decision['decision_basis']}")
    if decision.get("external_factors"):
        lines.append(f"- 可能外在因素：{decision['external_factors']}")
    if decision.get("data_gap"):
        lines.append(f"- 尚需補充資料：{decision['data_gap']}")
    if decision.get("next_action"):
        lines.append(f"- 後續行動：{decision['next_action']}")

    if lines:
        return "\n".join(lines)

    if intent == "事故熱點查詢":
        return "- 使用者：交通管理單位\n- 可能外在因素：事故熱點可能與通勤車流、商圈活動、主要幹道或路口密度有關。\n- 尚需補充資料：車流量、道路型態、路口位置與土地使用資料。\n- 後續行動：將事故數較高的行政區列為巡邏、工程檢討或宣導優先名單。"
    if intent == "時段查詢":
        return "- 使用者：交通管理單位與一般民眾\n- 可能外在因素：高事故時段可能與上班、下班、接送或商業活動造成的車流集中有關。\n- 尚需補充資料：分時車流量、通勤人口與大眾運輸轉乘資料。\n- 後續行動：針對事故高峰時段調整出行、勤務或號誌管理。"
    if intent == "肇因分析":
        return "- 使用者：交通管理單位與執法單位\n- 可能外在因素：主要肇因可能受到車流密度、道路複雜度、駕駛通勤疲勞或尖峰壓力影響。\n- 尚需補充資料：車流量、道路設計、路口號誌與事故現場型態資料。\n- 後續行動：將主要肇因轉為宣導、取締或道路改善重點。"

    return "- 本次結果可作為人工判讀前的初步排序依據，仍需搭配現場條件確認。"


def _format_recommendations(tool_results: dict[str, Any]) -> str:
    rec_result = tool_results.get("recommendation_tool")
    if not rec_result:
        return "- 目前沒有產生改善建議。"

    lines = []
    for owner, values in rec_result.get("recommendations", {}).items():
        lines.append(f"- {owner}: {'；'.join(values)}")
    return "\n".join(lines)


def _format_rag_matches(rag_result: dict[str, Any]) -> str:
    matches = rag_result.get("matches", [])
    if not matches:
        return rag_result.get("summary", "目前知識庫沒有找到對應資料。")

    lines = [rag_result.get("summary", "查詢結果如下：")]
    for item in matches[:5]:
        code = item.get("code")
        description = item.get("description")
        item_type = item.get("type", "資料")
        if code and description:
            lines.append(f"- {item_type} {code}: {description}")
    return "\n".join(lines)
