"""LLM-assisted natural-language understanding for traffic accident queries."""

from __future__ import annotations

import json
import re
from typing import Any

from local_llm_client import LocalLLMClient, LocalLLMError


ALLOWED_INTENTS = {
    "事故熱點查詢",
    "時段查詢",
    "肇因分析",
    "風險預測",
    "政策建議",
    "代碼說明",
    "民眾出行建議",
    "超出資料範圍",
    "需要補充條件",
}

ENTITY_KEYS = {
    "district",
    "hour",
    "weekday",
    "month",
    "weather",
    "keyword",
    "role",
    "transport_mode",
    "origin_district",
    "destination_district",
}

# ── Few-shot 範例（覆蓋所有 intent，幫助小型本機 LLM 校準）────────────
_FEW_SHOT_EXAMPLES = """
範例 1（風險預測）：
問題：西屯區星期五晚上六點雨天危險嗎？
輸出：{"intent":"風險預測","entities":{"district":"西屯區","weekday":"星期五","hour":18,"weather":"雨"},"confidence":0.97,"needs_clarification":false,"reason":"詢問特定條件下的危險程度，屬風險預測。"}

範例 2（事故熱點查詢）：
問題：台中市哪個行政區事故最多？
輸出：{"intent":"事故熱點查詢","entities":{},"confidence":0.95,"needs_clarification":false,"reason":"詢問事故最多的行政區，屬熱點查詢。"}

範例 3（時段查詢）：
問題：什麼時間最容易發生交通事故？
輸出：{"intent":"時段查詢","entities":{},"confidence":0.93,"needs_clarification":false,"reason":"詢問事故高峰時段，無明確風險判斷需求。"}

範例 4（肇因分析）：
問題：最常見肇事原因是什麼？
輸出：{"intent":"肇因分析","entities":{},"confidence":0.94,"needs_clarification":false,"reason":"詢問肇事原因排名。"}

範例 5（政策建議）：
問題：如果我是交通局，應該優先改善哪些問題？
輸出：{"intent":"政策建議","entities":{"role":"交通局"},"confidence":0.96,"needs_clarification":false,"reason":"詢問管理單位的優先改善方向，屬政策建議。"}

範例 6（民眾出行建議）：
問題：我要從西屯區騎機車去大里區，晚上六點下雨，危不危險？
輸出：{"intent":"民眾出行建議","entities":{"origin_district":"西屯區","destination_district":"大里區","transport_mode":"機車","hour":18,"weather":"雨"},"confidence":0.96,"needs_clarification":false,"reason":"有起訖點與交通工具，屬民眾出行建議。"}

範例 7（代碼說明）：
問題：肇事因素代碼 07 是什麼？
輸出：{"intent":"代碼說明","entities":{"keyword":"代碼 07"},"confidence":0.98,"needs_clarification":false,"reason":"詢問代碼說明。"}

範例 8（超出資料範圍 - 其他縣市）：
問題：台北市事故熱點如何？
輸出：{"intent":"超出資料範圍","entities":{},"confidence":0.99,"needs_clarification":false,"reason":"台北市不在台中市資料範圍內。"}

範例 9（超出資料範圍 - 未來事件）：
問題：明天早上會不會出車禍？
輸出：{"intent":"超出資料範圍","entities":{},"confidence":0.99,"needs_clarification":false,"reason":"詢問未來即時事故，系統只有歷史資料。"}

範例 10（風險預測含「今天」口語語境）：
問題：今天是星期五下班時間騎機車安不安全？
輸出：{"intent":"風險預測","entities":{"weekday":"星期五","hour":18,"transport_mode":"機車"},"confidence":0.88,"needs_clarification":false,"reason":"今天是口語語境，本質上是問星期五傍晚的歷史風險。"}

範例 11（風險預測含時段詞）：
問題：西屯區什麼時段風險最高？
輸出：{"intent":"風險預測","entities":{"district":"西屯區"},"confidence":0.90,"needs_clarification":false,"reason":"詢問特定區域的風險高峰時段，以風險預測為主。"}

範例 12（需要補充條件）：
問題：這樣危險嗎？
輸出：{"intent":"需要補充條件","entities":{},"confidence":0.85,"needs_clarification":true,"reason":"條件不明確，需補充行政區、時段或天候。"}
""".strip()


def parse_user_query_with_llm(
    user_input: str,
    *,
    rule_result: dict[str, Any] | None = None,
    client: LocalLLMClient | None = None,
) -> dict[str, Any]:
    """Parse a user query into intent/entities with a local LLM.

    Returns a metadata-rich dict. When parsing fails, ``ok`` is false and the
    caller should keep the deterministic rule result.
    """
    rule_entities = (rule_result or {}).get("entities", {})
    messages = [
        {
            "role": "system",
            "content": (
                "你是台中市交通事故風險決策支援系統的 NLU Parser。"
                "你只能輸出 JSON，不要輸出 Markdown 或任何說明文字。"
                "任務是把使用者問題轉成 intent 與 entities JSON。"
            ),
        },
        {
            "role": "user",
            "content": _build_prompt(user_input, rule_entities),
        },
    ]

    try:
        raw = (client or LocalLLMClient()).chat(
            messages,
            temperature=0,
            max_tokens=500,
            format_json=True,
        )
        parsed = _extract_json_object(raw)
        normalized = _normalize_parse(parsed, rule_entities)
    except (LocalLLMError, ValueError, TypeError) as exc:
        return {"ok": False, "source": "local_llm", "error": str(exc)}

    return {
        "ok": True,
        "source": "local_llm",
        "raw": raw,
        **normalized,
    }


def _build_prompt(user_input: str, rule_entities: dict[str, Any]) -> str:
    return f"""
請將使用者問題解析為下列 JSON schema，只輸出 JSON，不要有任何說明文字：
{{
  "intent": "事故熱點查詢 | 時段查詢 | 肇因分析 | 風險預測 | 政策建議 | 代碼說明 | 民眾出行建議 | 超出資料範圍 | 需要補充條件",
  "entities": {{
    "district": "台中行政區或 null",
    "hour": "0-23 的整數或 null",
    "weekday": "星期一到星期日或 null",
    "month": "1-12 的整數或 null",
    "weather": "晴、雨、陰、霧或煙、雪、風沙、風 或 null（霧天/起霧/煙霧請填「霧或煙」）",
    "keyword": "代碼或查詢關鍵字或 null",
    "role": "一般民眾、交通局、警察/交通大隊、道路工程單位 或 null",
    "transport_mode": "機車、汽車、行人、自行車、大眾運輸 或 null",
    "origin_district": "起點行政區或 null",
    "destination_district": "終點行政區或 null"
  }},
  "confidence": "0.0 到 1.0",
  "needs_clarification": "true 或 false",
  "reason": "一句話說明判斷原因"
}}

判斷規則（按優先順序）：
1. 其他縣市（台北、新北、高雄等）→「超出資料範圍」
2. 明天/後天/未來即時事故 →「超出資料範圍」（「今天是星期五…」是口語，屬風險預測）
3. 問「危險嗎、風險高嗎、安全嗎、容易發生嗎、什麼時段風險最高」→「風險預測」（優先於時段查詢）
4. 問「什麼時間、幾點、事故高峰時段」（無風險語義）→「時段查詢」
5. 有起訖點或交通工具 →「民眾出行建議」
6. 問「哪個區最多、熱點、排名」→「事故熱點查詢」
7. 問「原因、肇事、為什麼、外在因素」→「肇因分析」
8. 問「交通局、政策、優先改善」→「政策建議」
9. 問「代碼 07、欄位說明」→「代碼說明」
10. 條件不明確 →「需要補充條件」

--- 參考範例 ---
{_FEW_SHOT_EXAMPLES}
---

規則式已抽取到的條件（可作為參考，不要盲目照搬）：
{json.dumps(rule_entities, ensure_ascii=False)}

使用者問題：
{user_input}
""".strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response.")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON is not an object.")
    return value


def _normalize_parse(parsed: dict[str, Any], rule_entities: dict[str, Any]) -> dict[str, Any]:
    intent = str(parsed.get("intent", "")).strip()
    if intent not in ALLOWED_INTENTS:
        raise ValueError(f"Unsupported intent from LLM: {intent}")

    raw_entities = parsed.get("entities") or {}
    if not isinstance(raw_entities, dict):
        raw_entities = {}

    entities = {key: raw_entities.get(key) for key in ENTITY_KEYS if key in raw_entities}
    entities = _drop_empty_values(entities)

    # 規則式提取通常對在地欄位更精準，補入 LLM 遺漏的值
    for key, value in rule_entities.items():
        if key not in entities and value not in (None, "", "不指定"):
            entities[key] = value

    if "hour" in entities:
        entities["hour"] = _coerce_int_range(entities["hour"], 0, 23)
    if "month" in entities:
        entities["month"] = _coerce_int_range(entities["month"], 1, 12)

    # weather 正規化：確保是資料庫內的合法值
    if "weather" in entities:
        entities["weather"] = _normalize_weather(entities["weather"])
        if entities["weather"] is None:
            del entities["weather"]

    confidence = parsed.get("confidence", 0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    needs_clarification = bool(parsed.get("needs_clarification", intent == "需要補充條件"))
    reason = str(parsed.get("reason") or "本機 LLM 完成自然語言解析。")

    return {
        "intent": intent,
        "entities": entities,
        "confidence": confidence,
        "needs_clarification": needs_clarification,
        "reason": reason,
    }


# 合法資料值集合（來自 WEATHER_MAPPING 對應結果）
_VALID_WEATHER = {"晴", "雨", "陰", "霧或煙", "雪", "風沙", "風"}
# LLM 可能回傳的非標準值 → 正確資料值
_WEATHER_ALIAS = {
    "霧": "霧或煙",
    "煙": "霧或煙",
    "霧或煙": "霧或煙",
    "起霧": "霧或煙",
    "煙霧": "霧或煙",
}


def _normalize_weather(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if value in _VALID_WEATHER:
        return value
    return _WEATHER_ALIAS.get(value)


def _drop_empty_values(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in values.items()
        if value not in (None, "", "null", "None", "不指定")
    }


def _coerce_int_range(value: Any, low: int, high: int) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None
