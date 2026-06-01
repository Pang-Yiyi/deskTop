"""
risk_model.py
交通事故風險評分模型。

優先使用訓練好的 Random Forest 模型（rf_risk_model.pkl）；
若模型不存在或載入失敗，自動 fallback 至規則式評分。

RF 模型由 train_risk_model.py 訓練，特徵重要性：
  時段 > 行政區 > 月份 > 星期 > 天候
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_accident_data

_df: pd.DataFrame | None = None
_weights: dict | None = None

# ── ML 模型快取 ───────────────────────────────────────────
_ML_MODEL_PATH = Path(__file__).resolve().parent / "rf_risk_model.pkl"
_ml_model_data: dict | None = None
_ml_loaded: bool = False   # 是否已嘗試載入（避免重複 import）


def _load_ml_model() -> dict | None:
    """嘗試載入已訓練的 RF 模型，失敗時回傳 None。"""
    global _ml_model_data, _ml_loaded
    if _ml_loaded:
        return _ml_model_data
    _ml_loaded = True
    if not _ML_MODEL_PATH.exists():
        return None
    try:
        import joblib
        _ml_model_data = joblib.load(_ML_MODEL_PATH)
        return _ml_model_data
    except Exception:
        return None


def _ml_risk_score(
    district: str | None,
    hour: int | None,
    weekday: str | None,
    month: int | None,
    weather: str | None,
) -> tuple[int, str, dict] | None:
    """
    使用 RF 模型預測風險分數。
    回傳 (score: int, level: str, breakdown: dict) 或 None（模型不可用時）。
    """
    md = _load_ml_model()
    if md is None:
        return None

    try:
        import pandas as pd

        rf       = md["model"]
        encoders = md["encoders"]
        fi       = md.get("feature_importance", {})

        # 建立單筆輸入
        row = {
            "district":  str(district)  if district  else "西屯區",
            "hour_int":  int(hour)       if hour is not None else 12,
            "weekday":   str(weekday)    if weekday   else "星期三",
            "month_int": int(month)      if month is not None else 6,
            "weather":   str(weather)    if weather   else "晴",
        }
        df_row = pd.DataFrame([row])

        # Encode（處理訓練集外的標籤）
        for col in ["district", "weekday", "weather"]:
            le = encoders[col]
            val = str(row[col])
            if val not in set(le.classes_):
                val = le.classes_[0]
            df_row[col] = le.transform([val])

        # 預測機率
        proba = rf.predict_proba(df_row)[0]
        class_idx = {c: i for i, c in enumerate(rf.classes_)}

        p_high = proba[class_idx.get("高風險", 0)]
        p_mid  = proba[class_idx.get("中風險", 1 if len(rf.classes_) > 1 else 0)]
        p_low  = proba[class_idx.get("低風險", 2 if len(rf.classes_) > 2 else 0)]

        # 分數映射：高風險 → 70-100，中風險 → 35-70，低風險保留空間
        # p_high=1 → 85, p_mid=1 → 35, 50/50 → 60
        score = p_high * 85 + p_mid * 35 + p_low * 15
        score = min(100, max(0, int(score)))
        level = _risk_level(score)

        # 分解（特徵重要性作為各維度貢獻說明）
        breakdown = {
            "行政區分數（RF）":  round(fi.get("district",  0) * score),
            "時段分數（RF）":    round(fi.get("hour_int",  0) * score),
            "星期分數（RF）":    round(fi.get("weekday",   0) * score),
            "天候分數（RF）":    round(fi.get("weather",   0) * score),
            "月份分數（RF）":    round(fi.get("month_int", 0) * score),
            "P(高風險)":         round(float(p_high), 3),
            "P(中風險)":         round(float(p_mid),  3),
        }
        return score, level, breakdown

    except Exception:
        return None


def _severity_weighted_series(df: pd.DataFrame, col: str) -> pd.Series:
    """依死亡/受傷件數計算各類別的嚴重度加權總分。"""
    work = df[[col]].copy()
    if "死亡數量" in df.columns:
        work["_deaths"] = pd.to_numeric(df["死亡數量"], errors="coerce").fillna(0)
    else:
        work["_deaths"] = 0
    if "受傷數量" in df.columns:
        work["_injuries"] = pd.to_numeric(df["受傷數量"], errors="coerce").fillna(0)
    else:
        work["_injuries"] = 0
    # 死亡權重 3 倍、受傷 1 倍、每件事故基礎 1 分
    work["_weight"] = work["_deaths"] * 3 + work["_injuries"] + 1
    return work.groupby(col)["_weight"].sum().dropna()


def _get_weights() -> dict:
    """依全年資料計算各維度的百分位分數表。"""
    global _df, _weights
    if _weights is not None:
        return _weights

    df = load_accident_data()
    _df = df

    def _pct_score(series: pd.Series, max_score: int = 20) -> dict:
        """將計數/加權值轉成 0~max_score 的分數（線性縮放）。"""
        min_c, max_c = series.min(), series.max()
        if max_c == min_c:
            return {k: max_score // 2 for k in series.index}
        return {
            k: round((v - min_c) / (max_c - min_c) * max_score)
            for k, v in series.items()
        }

    _weights = {
        # 行政區與時段使用嚴重度加權，其餘使用件數
        "district": _pct_score(_severity_weighted_series(df, "區")),
        "hour": _pct_score(_severity_weighted_series(df, "hour")),
        "weekday": _pct_score(df["weekday"].value_counts()),
        "weather": _pct_score(df["天候_str"].value_counts()),
        "month": _pct_score(df["month"].value_counts()),
    }
    return _weights


def _risk_level(score: int) -> str:
    if score >= 70:
        return "高風險"
    if score >= 40:
        return "中風險"
    return "低風險"


def _weather_score(weather: str | None, weights: dict) -> int:
    """Combine historical frequency with driving hazard severity."""
    if not weather:
        return 5

    historical_score = weights["weather"].get(weather, 5)
    hazard_floor = {
        "雨": 12,
        "陰": 8,
        "霧": 16,
        "霧或煙": 16,
        "風": 10,
        "風沙": 14,
        "雪": 16,
    }.get(weather, 0)
    return max(historical_score, hazard_floor)


def _get_severity_stats(
    df: pd.DataFrame,
    district: str | None,
    hour: int | None,
    weekday: str | None,
    month: int | None,
    weather: str | None,
) -> dict:
    """計算篩選條件下的死亡與受傷統計，以及相對於全資料集的嚴重度比值。"""
    filtered = df
    if district:
        filtered = filtered[filtered["區"] == district]
    if hour is not None:
        filtered = filtered[filtered["hour"] == hour]
    if weekday:
        filtered = filtered[filtered["weekday"] == weekday]
    if month is not None:
        filtered = filtered[filtered["month"] == month]
    if weather:
        filtered = filtered[filtered["天候_str"] == weather]

    total_count = len(filtered)
    if total_count == 0:
        return {"deaths": 0, "injuries": 0, "severity_index": 0.0, "severity_label": "無資料"}

    deaths = int(pd.to_numeric(filtered.get("死亡數量", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    injuries = int(pd.to_numeric(filtered.get("受傷數量", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())

    # 嚴重度指數：每件事故平均（死亡×3 + 受傷）
    severity_index = round((deaths * 3 + injuries) / total_count, 2)

    # 全資料集基準
    global_deaths = int(pd.to_numeric(df.get("死亡數量", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    global_injuries = int(pd.to_numeric(df.get("受傷數量", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    global_index = (global_deaths * 3 + global_injuries) / len(df) if len(df) > 0 else 1

    severity_ratio = round(severity_index / global_index, 2) if global_index > 0 else 1.0
    if severity_ratio >= 1.5:
        severity_label = "嚴重度偏高"
    elif severity_ratio >= 1.1:
        severity_label = "嚴重度略高於平均"
    elif severity_ratio >= 0.9:
        severity_label = "嚴重度接近平均"
    else:
        severity_label = "嚴重度低於平均"

    return {
        "deaths": deaths,
        "injuries": injuries,
        "accident_count": total_count,
        "severity_index": severity_index,
        "severity_ratio": severity_ratio,
        "severity_label": severity_label,
    }


def _compute_interaction_bonus(
    district: str | None,
    hour: int | None,
    weekday: str | None,
    weather: str | None,
    score_breakdown: dict[str, int],
) -> int:
    """
    計算多重不利條件同時成立時的交互加成分數（0-10）。

    單一條件的風險是線性疊加；但高風險區 + 尖峰時段 + 惡劣天候
    的組合危險性遠高於個別加總，此函數提供額外加成。
    """
    high_risk_flags: list[bool] = [
        score_breakdown.get("行政區分數", 0) >= 15,
        score_breakdown.get("時段分數", 0) >= 15,
        weather in ("雨", "霧", "霧或煙", "風沙", "雪"),
        weekday in ("星期五", "星期六", "星期日"),
        hour is not None and (17 <= hour <= 20 or 7 <= hour <= 9),
    ]
    count = sum(high_risk_flags)

    # 2 個以上不利條件同時觸發才給加成
    if count < 2:
        return 0
    if count == 2:
        return 3
    if count == 3:
        return 6
    return 10  # 4 個以上


def risk_score_tool(
    district: str | None = None,
    hour: int | None = None,
    weekday: str | None = None,
    month: int | None = None,
    weather: str | None = None,
) -> dict:
    """
    計算指定條件下的風險分數。

    Returns
    -------
    dict with keys: query, risk, evidence, severity, reasons, recommendations
    """
    # ── 優先使用 RF 模型，fallback 至規則式 ──────────────
    ml_result = _ml_risk_score(district, hour, weekday, month, weather)
    if ml_result is not None:
        total_score, level, score_breakdown = ml_result
        score_breakdown["模型來源"] = "Random Forest"
    else:
        # 規則式 fallback
        weights = _get_weights()
        score_breakdown: dict[str, int] = {
            "行政區分數": weights["district"].get(district, 10) if district else 10,
            "時段分數":   weights["hour"].get(hour, 10) if hour is not None else 10,
            "星期分數":   weights["weekday"].get(weekday, 10) if weekday else 10,
            "天候分數":   _weather_score(weather, weights),
            "月份分數":   weights["month"].get(month, 10) if month else 10,
        }
        interaction_bonus = _compute_interaction_bonus(
            district=district, hour=hour, weekday=weekday,
            weather=weather, score_breakdown=score_breakdown,
        )
        if interaction_bonus > 0:
            score_breakdown["複合風險加成"] = interaction_bonus
        total_score = min(100, max(0, sum(
            v for k, v in score_breakdown.items() if isinstance(v, int)
        )))
        level = _risk_level(total_score)
        score_breakdown["模型來源"] = "規則式"

    # ── 確保 _df 已載入（用於後續佐證統計）──────────────
    if _df is None:
        _get_weights()
    df = _df

    # ── 佐證資訊 ─────────────────────────────────────────
    evidence: dict = {}
    if district:
        district_counts = df["區"].value_counts()
        evidence["district_rank"] = int(
            (district_counts > district_counts.get(district, 0)).sum() + 1
        )
        evidence["district_count"] = int(district_counts.get(district, 0))

    if hour is not None:
        hour_counts = df["hour"].value_counts()
        evidence["hour_rank"] = int(
            (hour_counts > hour_counts.get(hour, 0)).sum() + 1
        )
        evidence["hour_count"] = int(hour_counts.get(hour, 0))

    # ── 傷亡嚴重度資訊 ────────────────────────────────────
    severity = _get_severity_stats(df, district, hour, weekday, month, weather)

    # ── 原因說明 ─────────────────────────────────────────
    reasons: list[str] = []
    if district and evidence.get("district_rank", 99) <= 3:
        reasons.append(f"{district}為本資料集中事故數前 {evidence['district_rank']} 名行政區")
    if hour is not None and evidence.get("hour_rank", 99) <= 5:
        reasons.append(f"{hour} 時屬於事故高峰時段（排名第 {evidence['hour_rank']} 名）")
    if weather in ("雨", "霧或煙", "風沙", "雪"):
        reasons.append(f"天候「{weather}」會增加視線與路面風險")
    if weekday in ("星期五", "星期六"):
        reasons.append(f"{weekday}為週末前後，交通量較大")
    if severity.get("severity_label") == "嚴重度偏高":
        reasons.append(
            f"此條件下每件事故平均傷亡嚴重度為全資料集平均的 {severity['severity_ratio']} 倍"
        )
    if score_breakdown.get("複合風險加成", 0) > 0:
        reasons.append(
            f"多重不利條件同時成立（加成 +{score_breakdown['複合風險加成']} 分），"
            "實際風險高於各條件單獨評估"
        )

    # ── 建議（簡化版，完整版由 recommendation_tool 提供）──
    from analysis_tools import recommendation_tool
    recs = recommendation_tool(level, district=district, hour=hour, weather=weather)
    recommendations = (
        recs["recommendations"]["民眾建議"][:2]
        + recs["recommendations"]["交通管理單位建議"][:1]
    )

    return {
        "query": {
            "district": district,
            "hour": hour,
            "weekday": weekday,
            "month": month,
            "weather": weather,
        },
        "risk": {
            "score": total_score,
            "level": level,
            "breakdown": score_breakdown,
        },
        "evidence": evidence,
        "severity": severity,
        "reasons": reasons,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    result = risk_score_tool(district="西屯區", hour=18, weekday="星期五", weather="雨")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))
