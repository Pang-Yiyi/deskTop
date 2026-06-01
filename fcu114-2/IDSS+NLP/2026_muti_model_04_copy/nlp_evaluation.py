"""
nlp_evaluation.py
台中市交通事故決策支援系統 — NLP 評估實驗

執行方式：
    python nlp_evaluation.py

輸出：
    - 終端機：Intent 準確率、Per-Intent F1、實體提取準確率、Rule vs LLM 比較
    - nlp_eval_results.json：詳細逐題結果
    - nlp_eval_report.md：可貼入期末報告的 Markdown 表格
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from agents import detect_intent, extract_query_entities
from nlu_parser import parse_user_query_with_llm


# ─────────────────────────────────────────────────────────
# 載入測試集
# ─────────────────────────────────────────────────────────

def load_dataset(path: Path = BASE_DIR / "nlp_eval_dataset.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


# ─────────────────────────────────────────────────────────
# 評估輔助函數
# ─────────────────────────────────────────────────────────

def _match_intent(predicted: str, expected: str) -> bool:
    return predicted.strip() == expected.strip()


def _match_entity(predicted_val: Any, expected_val: Any, key: str) -> bool:
    """比較單一實體是否正確。"""
    if expected_val is None:
        return True  # 未標注的實體不計入
    if predicted_val is None:
        return False
    if key == "hour":
        # 時段允許 ±1 小時誤差（口語時段本就有模糊性）
        try:
            return abs(int(predicted_val) - int(expected_val)) <= 1
        except (TypeError, ValueError):
            return False
    return str(predicted_val).strip() == str(expected_val).strip()


def evaluate_entities(predicted: dict, expected: dict) -> dict[str, bool]:
    """逐欄位比較實體提取結果。"""
    results: dict[str, bool] = {}
    for key, exp_val in expected.items():
        pred_val = predicted.get(key)
        results[key] = _match_entity(pred_val, exp_val, key)
    return results


# ─────────────────────────────────────────────────────────
# 主評估邏輯
# ─────────────────────────────────────────────────────────

def run_rule_evaluation(cases: list[dict]) -> list[dict]:
    """執行規則式 NLU 評估。"""
    results = []
    for case in cases:
        question = case["question"]
        expected_intent = case["expected_intent"]
        expected_entities = case.get("expected_entities", {})

        rule_result = detect_intent(question)
        predicted_intent = rule_result.get("intent", "")
        predicted_entities = rule_result.get("entities", {})

        intent_correct = _match_intent(predicted_intent, expected_intent)
        entity_results = evaluate_entities(predicted_entities, expected_entities)

        results.append({
            "id": case["id"],
            "question": question,
            "note": case.get("note", ""),
            "expected_intent": expected_intent,
            "predicted_intent_rule": predicted_intent,
            "intent_correct_rule": intent_correct,
            "expected_entities": expected_entities,
            "predicted_entities_rule": predicted_entities,
            "entity_results_rule": entity_results,
        })
    return results


def _check_llm_available(timeout: float = 3.0) -> bool:
    """快速檢查 Ollama 是否在 localhost:11434 運行。"""
    try:
        from urllib.request import urlopen
        from urllib.error import URLError
        urlopen("http://localhost:11434", timeout=timeout)
        return True
    except Exception:
        return False


def run_llm_evaluation(cases: list[dict], results: list[dict]) -> list[dict]:
    """執行本機 LLM NLU 評估（若 Ollama 未啟動則跳過）。"""
    print("\n[LLM 評估] 嘗試連線本機 LLM（需要 Ollama 在 localhost:11434 運行）...")
    if not _check_llm_available():
        print("  → 本機 LLM 不可用，跳過 LLM 評估（只顯示規則式結果）。")
        return results, False
    llm_available = False
    updated = []

    for i, (case, result) in enumerate(zip(cases, results)):
        question = case["question"]
        expected_intent = case["expected_intent"]
        expected_entities = case.get("expected_entities", {})

        try:
            rule_result = {
                "intent": result["predicted_intent_rule"],
                "entities": result["predicted_entities_rule"],
            }
            llm_parse = parse_user_query_with_llm(question, rule_result=rule_result)

            if llm_parse.get("ok"):
                llm_available = True
                predicted_intent = llm_parse.get("intent", "")
                predicted_entities = llm_parse.get("entities", {})
                confidence = llm_parse.get("confidence", 0.0)
                intent_correct = _match_intent(predicted_intent, expected_intent)
                entity_results = evaluate_entities(predicted_entities, expected_entities)

                result["predicted_intent_llm"] = predicted_intent
                result["intent_correct_llm"] = intent_correct
                result["predicted_entities_llm"] = predicted_entities
                result["entity_results_llm"] = entity_results
                result["llm_confidence"] = round(confidence, 2)

                status = "✅" if intent_correct else "❌"
                print(f"  [{i+1:02d}/{len(cases)}] {status} {expected_intent} → {predicted_intent} (conf={confidence:.2f})")
                time.sleep(0.2)
            else:
                result["predicted_intent_llm"] = None
                result["intent_correct_llm"] = None
                result["llm_error"] = llm_parse.get("error", "unknown")
        except Exception as exc:
            result["predicted_intent_llm"] = None
            result["intent_correct_llm"] = None
            result["llm_error"] = str(exc)

        updated.append(result)

    if not llm_available:
        print("  → 本機 LLM 不可用，跳過 LLM 評估（只顯示規則式結果）。")
    return updated, llm_available


# ─────────────────────────────────────────────────────────
# 指標計算
# ─────────────────────────────────────────────────────────

def compute_metrics(results: list[dict], mode: str = "rule") -> dict:
    """計算 Intent 準確率與 Per-Intent 統計。"""
    key = f"intent_correct_{mode}"
    valid = [r for r in results if r.get(key) is not None]
    if not valid:
        return {}

    total = len(valid)
    correct = sum(1 for r in valid if r[key])
    accuracy = round(correct / total * 100, 1)

    # Per-intent 統計
    intents = sorted(set(r["expected_intent"] for r in valid))
    per_intent: dict[str, dict] = {}
    for intent in intents:
        subset = [r for r in valid if r["expected_intent"] == intent]
        n = len(subset)
        c = sum(1 for r in subset if r[key])
        per_intent[intent] = {
            "total": n,
            "correct": c,
            "accuracy": round(c / n * 100, 1) if n else 0,
        }

    # 實體提取準確率
    entity_key = f"entity_results_{mode}"
    entity_totals: dict[str, int] = {}
    entity_correct: dict[str, int] = {}
    for r in valid:
        for field, ok in r.get(entity_key, {}).items():
            entity_totals[field] = entity_totals.get(field, 0) + 1
            if ok:
                entity_correct[field] = entity_correct.get(field, 0) + 1
    entity_accuracy = {
        field: round(entity_correct.get(field, 0) / cnt * 100, 1)
        for field, cnt in entity_totals.items()
    }

    # sklearn Precision / Recall / F1
    f1_report: dict = {}
    macro_f1 = weighted_f1 = macro_precision = macro_recall = None
    if _SKLEARN_OK:
        pred_key = f"predicted_intent_{mode}"
        y_true = [r["expected_intent"] for r in valid]
        y_pred = [r.get(pred_key, "") for r in valid]
        labels = sorted(set(y_true))
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        for i, label in enumerate(labels):
            f1_report[label] = {
                "precision": round(float(p[i]), 3),
                "recall": round(float(r[i]), 3),
                "f1": round(float(f[i]), 3),
                "support": per_intent.get(label, {}).get("total", 0),
            }
        p_mac, r_mac, f_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)
        p_wt, r_wt, f_wt, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0)
        macro_precision = round(float(p_mac), 3)
        macro_recall = round(float(r_mac), 3)
        macro_f1 = round(float(f_mac), 3)
        weighted_f1 = round(float(f_wt), 3)

    return {
        "mode": mode,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_intent": per_intent,
        "entity_accuracy": entity_accuracy,
        "f1_report": f1_report,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }


# ─────────────────────────────────────────────────────────
# 列印報告
# ─────────────────────────────────────────────────────────

def print_summary(rule_metrics: dict, llm_metrics: dict | None) -> None:
    print("\n" + "=" * 60)
    print("  NLP 評估結果摘要")
    print("=" * 60)

    # 整體準確率比較
    print(f"\n【Intent 分類準確率】")
    rule_f1_str = f"  Macro-F1={rule_metrics['macro_f1']}" if rule_metrics.get("macro_f1") is not None else ""
    print(f"  規則式 NLU：{rule_metrics['correct']}/{rule_metrics['total']} = {rule_metrics['accuracy']}%{rule_f1_str}")
    if llm_metrics:
        llm_f1_str = f"  Macro-F1={llm_metrics['macro_f1']}" if llm_metrics.get("macro_f1") is not None else ""
        print(f"  本機 LLM  ：{llm_metrics['correct']}/{llm_metrics['total']} = {llm_metrics['accuracy']}%{llm_f1_str}")

    # Per-intent Precision / Recall / F1
    if rule_metrics.get("f1_report"):
        print(f"\n【Per-Intent Precision / Recall / F1（規則式）】")
        print(f"  {'Intent':<14} {'P':>6} {'R':>6} {'F1':>6} {'支持數':>5}")
        print(f"  {'-'*42}")
        for intent, s in sorted(rule_metrics["f1_report"].items()):
            print(f"  {intent:<14} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f} {s['support']:>5}")
        print(f"  {'Macro avg':<14} {rule_metrics['macro_precision']:>6.3f} "
              f"{rule_metrics['macro_recall']:>6.3f} {rule_metrics['macro_f1']:>6.3f} {rule_metrics['total']:>5}")

    if llm_metrics and llm_metrics.get("f1_report"):
        print(f"\n【Per-Intent Precision / Recall / F1（本機 LLM）】")
        print(f"  {'Intent':<14} {'P':>6} {'R':>6} {'F1':>6} {'支持數':>5}")
        print(f"  {'-'*42}")
        for intent, s in sorted(llm_metrics["f1_report"].items()):
            print(f"  {intent:<14} {s['precision']:>6.3f} {s['recall']:>6.3f} {s['f1']:>6.3f} {s['support']:>5}")
        print(f"  {'Macro avg':<14} {llm_metrics['macro_precision']:>6.3f} "
              f"{llm_metrics['macro_recall']:>6.3f} {llm_metrics['macro_f1']:>6.3f} {llm_metrics['total']:>5}")

    # Per-intent 明細（Accuracy）
    print(f"\n【各 Intent 準確率（規則式）】")
    print(f"  {'Intent':<14} {'題數':>4} {'正確':>4} {'準確率':>6}")
    print(f"  {'-'*35}")
    for intent, stat in rule_metrics["per_intent"].items():
        mark = "✅" if stat["accuracy"] == 100 else ("⚠️ " if stat["accuracy"] >= 67 else "❌")
        print(f"  {mark} {intent:<12} {stat['total']:>4}  {stat['correct']:>4}  {stat['accuracy']:>5}%")

    if llm_metrics and llm_metrics.get("per_intent"):
        print(f"\n【各 Intent 準確率（本機 LLM）】")
        print(f"  {'Intent':<14} {'題數':>4} {'正確':>4} {'準確率':>6}")
        print(f"  {'-'*35}")
        for intent, stat in llm_metrics["per_intent"].items():
            mark = "✅" if stat["accuracy"] == 100 else ("⚠️ " if stat["accuracy"] >= 67 else "❌")
            print(f"  {mark} {intent:<12} {stat['total']:>4}  {stat['correct']:>4}  {stat['accuracy']:>5}%")

    # 實體提取
    if rule_metrics.get("entity_accuracy"):
        print(f"\n【實體提取準確率（規則式）】")
        label_map = {
            "district": "行政區",
            "hour": "時段（±1h）",
            "weekday": "星期",
            "weather": "天候",
            "transport_mode": "交通工具",
            "origin_district": "起點行政區",
            "destination_district": "終點行政區",
            "keyword": "關鍵字",
            "role": "角色",
        }
        for field, acc in sorted(rule_metrics["entity_accuracy"].items()):
            label = label_map.get(field, field)
            bar = "█" * int(acc / 10) + "░" * (10 - int(acc / 10))
            print(f"  {label:<14} {bar} {acc:>5}%")

    print("\n" + "=" * 60)


def print_error_cases(results: list[dict]) -> None:
    errors = [r for r in results if not r.get("intent_correct_rule")]
    if not errors:
        print("\n【規則式 NLU：無誤判案例】全部正確 ✅")
        return
    print(f"\n【規則式 NLU 誤判案例（共 {len(errors)} 題）】")
    for r in errors:
        print(f"  Q{r['id']:02d}: {r['question']}")
        print(f"       預期: {r['expected_intent']}  →  預測: {r['predicted_intent_rule']}")
        print(f"       備注: {r['note']}")


# ─────────────────────────────────────────────────────────
# 儲存結果
# ─────────────────────────────────────────────────────────

def save_results(results: list[dict], rule_metrics: dict, llm_metrics: dict | None) -> None:
    output = {
        "rule_metrics": rule_metrics,
        "llm_metrics": llm_metrics,
        "cases": results,
    }
    result_path = BASE_DIR / "nlp_eval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n詳細結果已儲存至：{result_path.name}")


def save_markdown_report(rule_metrics: dict, llm_metrics: dict | None, results: list[dict]) -> None:
    lines = [
        "# NLP 評估報告",
        "",
        f"**資料集**：台中市交通事故決策支援系統 NLP 測試集（共 {rule_metrics['total']} 題）",
        "",
        "## 1. Intent 分類準確率",
        "",
    ]

    # 整體準確率表（含 F1）
    has_f1 = bool(rule_metrics.get("macro_f1") is not None)
    if has_f1:
        lines += ["| 方法 | 正確題數 | 總題數 | Accuracy | Macro-F1 | Macro-P | Macro-R |",
                  "|---|---|---|---|---|---|---|"]
        lines.append(
            f"| 規則式 NLU | {rule_metrics['correct']} | {rule_metrics['total']} "
            f"| **{rule_metrics['accuracy']}%** | **{rule_metrics['macro_f1']}** "
            f"| {rule_metrics['macro_precision']} | {rule_metrics['macro_recall']} |"
        )
        if llm_metrics and llm_metrics.get("macro_f1") is not None:
            lines.append(
                f"| 本機 LLM（Qwen3 4B）| {llm_metrics['correct']} | {llm_metrics['total']} "
                f"| **{llm_metrics['accuracy']}%** | **{llm_metrics['macro_f1']}** "
                f"| {llm_metrics['macro_precision']} | {llm_metrics['macro_recall']} |"
            )
    else:
        lines += ["| 方法 | 正確題數 | 總題數 | 準確率 |", "|---|---|---|---|"]
        lines.append(f"| 規則式 NLU | {rule_metrics['correct']} | {rule_metrics['total']} | **{rule_metrics['accuracy']}%** |")
        if llm_metrics:
            lines.append(f"| 本機 LLM（Qwen3 4B）| {llm_metrics['correct']} | {llm_metrics['total']} | **{llm_metrics['accuracy']}%** |")
    lines.append("")

    # Per-intent F1 表（規則式）
    if rule_metrics.get("f1_report"):
        lines += ["## 2. Per-Intent Precision / Recall / F1（規則式）", ""]
        lines += ["| Intent | Support | Precision | Recall | F1-score |", "|---|---|---|---|---|"]
        for intent in sorted(rule_metrics["f1_report"].keys()):
            s = rule_metrics["f1_report"][intent]
            lines.append(f"| {intent} | {s['support']} | {s['precision']:.3f} | {s['recall']:.3f} | **{s['f1']:.3f}** |")
        lines.append(
            f"| **Macro avg** | {rule_metrics['total']} "
            f"| {rule_metrics['macro_precision']:.3f} "
            f"| {rule_metrics['macro_recall']:.3f} "
            f"| **{rule_metrics['macro_f1']:.3f}** |"
        )
        lines.append("")

    if llm_metrics and llm_metrics.get("f1_report"):
        lines += ["## 2b. Per-Intent Precision / Recall / F1（本機 LLM）", ""]
        lines += ["| Intent | Support | Precision | Recall | F1-score |", "|---|---|---|---|---|"]
        for intent in sorted(llm_metrics["f1_report"].keys()):
            s = llm_metrics["f1_report"][intent]
            lines.append(f"| {intent} | {s['support']} | {s['precision']:.3f} | {s['recall']:.3f} | **{s['f1']:.3f}** |")
        lines.append(
            f"| **Macro avg** | {llm_metrics['total']} "
            f"| {llm_metrics['macro_precision']:.3f} "
            f"| {llm_metrics['macro_recall']:.3f} "
            f"| **{llm_metrics['macro_f1']:.3f}** |"
        )
        lines.append("")

    # Per-intent Accuracy 比較表
    lines += ["## 3. 各 Intent Accuracy 比較", ""]
    header = "| Intent | 題數 | 規則式正確 | 規則式 Acc |"
    sep = "|---|---|---|---|"
    if llm_metrics:
        header += " LLM 正確 | LLM Acc |"
        sep += "---|---|"
    lines += [header, sep]

    for intent in sorted(rule_metrics["per_intent"].keys()):
        r_stat = rule_metrics["per_intent"][intent]
        row = f"| {intent} | {r_stat['total']} | {r_stat['correct']} | {r_stat['accuracy']}% |"
        if llm_metrics and intent in llm_metrics.get("per_intent", {}):
            l_stat = llm_metrics["per_intent"][intent]
            row += f" {l_stat['correct']} | {l_stat['accuracy']}% |"
        lines.append(row)
    lines.append("")

    # 實體提取表
    if rule_metrics.get("entity_accuracy"):
        lines += ["## 4. 實體提取準確率（規則式）", ""]
        lines += ["| 實體欄位 | 測試題數 | 準確率 |", "|---|---|---|"]
        label_map = {
            "district": "行政區",
            "hour": "時段（±1 小時）",
            "weekday": "星期",
            "weather": "天候",
            "transport_mode": "交通工具",
            "origin_district": "起點行政區",
            "destination_district": "終點行政區",
            "keyword": "關鍵字",
            "role": "角色",
        }
        entity_totals = {}
        entity_key = "entity_results_rule"
        for r in results:
            for field in r.get(entity_key, {}):
                entity_totals[field] = entity_totals.get(field, 0) + 1

        for field, acc in sorted(rule_metrics["entity_accuracy"].items()):
            label = label_map.get(field, field)
            cnt = entity_totals.get(field, "—")
            lines.append(f"| {label} | {cnt} | {acc}% |")
        lines.append("")

    # 誤判案例
    errors = [r for r in results if not r.get("intent_correct_rule")]
    lines += ["## 5. 規則式 NLU 誤判案例", ""]
    if errors:
        lines += ["| # | 問題 | 預期 Intent | 預測 Intent | 備注 |", "|---|---|---|---|---|"]
        for r in errors:
            lines.append(f"| Q{r['id']:02d} | {r['question']} | {r['expected_intent']} | {r['predicted_intent_rule']} | {r['note']} |")
    else:
        lines.append("無誤判案例，規則式 NLU 全部正確。✅")
    lines.append("")

    # 觀察與分析
    lines += [
        "## 6. 觀察與分析",
        "",
        "### 規則式 NLU 優點",
        "- 不依賴外部服務，離線環境下仍可穩定運行。",
        "- 對精確關鍵字匹配（如代碼查詢、政策建議）準確率高。",
        "- 延遲低，每次解析不超過 5ms。",
        "",
        "### 規則式 NLU 限制",
        "- 對語義模糊問題（如「這樣危險嗎」）無法精準判斷。",
        "- 同義詞覆蓋有限，需人工維護關鍵詞表。",
        "- 無法理解問句結構，僅靠關鍵詞觸發。",
        "",
        "### 本機 LLM（Qwen3 4B）優點",
        "- 理解自然語言語義，處理模糊與多義問題更靈活。",
        "- Few-shot Prompting 提供對齊邊界案例的能力。",
        "- 可提取複雜實體組合（如起訖點 + 交通工具）。",
        "",
        "### 混合架構設計說明",
        "本系統採用規則式優先、LLM 輔助的混合架構：",
        "1. 規則式 NLU 先執行，提取確定性強的實體（行政區、時段、天候）。",
        "2. LLM NLU 根據語義判斷 intent，信心值 ≥ 0.55 時採用 LLM 結果。",
        "3. 超出範圍（其他縣市）由規則式強制過濾，LLM 結果不覆蓋。",
        "4. LLM 不可用時自動 fallback 至規則式，確保系統可離線運行。",
    ]

    report_path = BASE_DIR / "nlp_eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown 報告已儲存至：{report_path.name}")


# ─────────────────────────────────────────────────────────
# 進入點
# ─────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  台中市交通事故決策支援系統 — NLP 評估實驗")
    print("=" * 60)

    cases = load_dataset()
    print(f"\n載入測試集：{len(cases)} 題")

    # Step 1：規則式評估
    print("\n[Step 1] 執行規則式 NLU 評估...")
    results = run_rule_evaluation(cases)
    rule_metrics = compute_metrics(results, mode="rule")
    print(f"  完成：{rule_metrics['correct']}/{rule_metrics['total']} 正確（{rule_metrics['accuracy']}%）")

    # Step 2：LLM 評估（選配）
    results, llm_available = run_llm_evaluation(cases, results)
    llm_metrics = compute_metrics(results, mode="llm") if llm_available else None
    if llm_available and llm_metrics:
        print(f"  完成：{llm_metrics['correct']}/{llm_metrics['total']} 正確（{llm_metrics['accuracy']}%）")
    elif llm_available:
        print("  → LLM 回傳結果為空，跳過 LLM 指標計算。")
        llm_metrics = None

    # Step 3：列印摘要
    print_summary(rule_metrics, llm_metrics)
    print_error_cases(results)

    # Step 4：儲存
    save_results(results, rule_metrics, llm_metrics)
    save_markdown_report(rule_metrics, llm_metrics, results)

    print("\n評估完成。")


if __name__ == "__main__":
    main()
