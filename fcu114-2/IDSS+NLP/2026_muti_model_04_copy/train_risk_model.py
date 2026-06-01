"""
train_risk_model.py
Random Forest 交通事故風險預測模型訓練與比較腳本

特徵：行政區、時段、星期、月份、天候（5 維）
目標：事故嚴重度類別（低風險 / 中風險 / 高風險）
      低：deaths*3 + injuries = 0-1
      中：deaths*3 + injuries = 2-5
      高：deaths*3 + injuries ≥ 6

訓練後若 RF Macro-F1 > 規則式，自動儲存模型。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import joblib
    _JOBLIB = True
except ImportError:
    import pickle as joblib
    _JOBLIB = False

from data_loader import load_accident_data
from risk_model import _get_weights, _risk_level, _weather_score

FEATURE_COLS = ["district", "hour_int", "weekday", "month_int", "weather"]
CAT_COLS     = ["district", "weekday", "weather"]
TARGET_COL   = "severity_class"
MODEL_PATH   = BASE_DIR / "rf_risk_model.pkl"


# ─────────────────────────────────────────────────────────
# 資料準備
# ─────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    目標變數：條件組合的事故頻率（高/中/低）
    - 以 (district, hour_int, weekday, month_int, weather) 為 key 聚合事故數
    - 以三分位數切分高/中/低：出事越多的條件組合 → 高風險
    - 每筆原始記錄繼承其所屬條件組合的標籤
    """
    df = df.copy()
    df["hour_int"]  = pd.to_numeric(df["hour"],  errors="coerce")
    df["month_int"] = pd.to_numeric(df["month"], errors="coerce")
    df = df.rename(columns={"區": "district", "天候_str": "weather"})
    df = df.dropna(subset=["district", "hour_int", "weekday", "month_int", "weather"])
    df["hour_int"]  = df["hour_int"].astype(int)
    df["month_int"] = df["month_int"].astype(int)

    # 聚合：每個條件組合的事故件數
    agg = (
        df.groupby(FEATURE_COLS)
        .size()
        .reset_index(name="accident_count")
    )

    # 三分位數 → 風險標籤（確保三類都有樣本）
    q33 = agg["accident_count"].quantile(0.33)
    q67 = agg["accident_count"].quantile(0.67)

    def risk_label(cnt: int) -> str:
        if cnt >= q67:
            return "高風險"
        if cnt >= q33:
            return "中風險"
        return "低風險"

    agg[TARGET_COL] = agg["accident_count"].apply(risk_label)
    print(f"    分位數：q33={q33:.0f}件  q67={q67:.0f}件")

    # 將標籤 merge 回原始記錄
    df = df.merge(agg[FEATURE_COLS + [TARGET_COL]], on=FEATURE_COLS, how="left")
    df[TARGET_COL] = df[TARGET_COL].fillna("低風險")
    return df[FEATURE_COLS + [TARGET_COL]].reset_index(drop=True)


def encode_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    if fit:
        encoders = {}
    for col in CAT_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known else le.classes_[0]
            )
            df[col] = le.transform(df[col])
    return df, encoders


# ─────────────────────────────────────────────────────────
# 規則式模型批次預測（向量化，速度快）
# ─────────────────────────────────────────────────────────

def rule_predict_batch(df_test: pd.DataFrame) -> list[str]:
    """用現有規則式模型批次預測，供比較用。"""
    weights = _get_weights()
    preds = []
    for _, row in df_test.iterrows():
        s = (
            weights["district"].get(row["district"], 10)
            + weights["hour"].get(row["hour_int"], 10)
            + weights["weekday"].get(row["weekday"], 10)
            + _weather_score(row["weather"], weights)
            + weights["month"].get(row["month_int"], 10)
        )
        preds.append(_risk_level(min(100, max(0, s))))
    return preds


# ─────────────────────────────────────────────────────────
# 主訓練流程
# ─────────────────────────────────────────────────────────

def train() -> bool:
    print("=" * 55)
    print("  Random Forest 風險模型訓練與比較")
    print("=" * 55)

    # 1. 載入資料
    print("\n[1] 載入事故資料...")
    df = prepare_data(load_accident_data())
    print(f"    有效記錄：{len(df):,} 筆")
    dist = df[TARGET_COL].value_counts()
    for cls, cnt in dist.items():
        print(f"    {cls}: {cnt:,} 筆 ({cnt/len(df)*100:.1f}%)")

    # 2. 切分資料集
    X, y = df[FEATURE_COLS], df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n    訓練：{len(X_train):,} / 測試：{len(X_test):,}")

    # 3. 訓練 RF
    print("\n[2] 訓練 Random Forest（n_estimators=200）...")
    X_train_enc, encoders = encode_features(X_train, fit=True)
    X_test_enc,  _        = encode_features(X_test,  encoders=encoders, fit=False)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_enc, y_train)

    # 4. 評估 RF
    y_pred_rf = rf.predict(X_test_enc)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf  = f1_score(y_test, y_pred_rf, average="macro")

    print(f"\n[3] Random Forest 評估（{len(X_test):,} 筆）")
    print(f"    Accuracy = {acc_rf:.4f}   Macro-F1 = {f1_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    # 5. 評估規則式（抽樣 5000 筆加速）
    print("\n[4] 規則式模型評估（抽樣 5,000 筆）...")
    sample_idx = np.random.RandomState(42).choice(len(X_test), min(5000, len(X_test)), replace=False)
    X_sample   = X_test.iloc[sample_idx]
    y_sample   = y_test.iloc[sample_idx]
    y_pred_rule = rule_predict_batch(X_sample)
    acc_rule = accuracy_score(y_sample, y_pred_rule)
    f1_rule  = f1_score(y_sample, y_pred_rule, average="macro", zero_division=0)
    print(f"    Accuracy = {acc_rule:.4f}   Macro-F1 = {f1_rule:.4f}")

    # 6. 比較
    print("\n[5] 比較結果")
    print(f"    {'方法':<18} {'Accuracy':>10} {'Macro-F1':>10}")
    print(f"    {'-'*40}")
    print(f"    {'Random Forest':<18} {acc_rf:>10.4f} {f1_rf:>10.4f}")
    print(f"    {'規則式模型':<18} {acc_rule:>10.4f} {f1_rule:>10.4f}")
    winner = "Random Forest ✅" if f1_rf > f1_rule else "規則式模型 ✅"
    print(f"\n    → 較佳模型：{winner}")

    # 7. 儲存（僅在 RF 較佳時）
    if f1_rf > f1_rule:
        fi = dict(zip(FEATURE_COLS, rf.feature_importances_))
        model_data = {
            "model":              rf,
            "encoders":           encoders,
            "feature_cols":       FEATURE_COLS,
            "cat_cols":           CAT_COLS,
            "classes":            list(rf.classes_),
            "feature_importance": fi,
            "metrics": {
                "rf_accuracy":    round(acc_rf,   4),
                "rf_macro_f1":    round(f1_rf,    4),
                "rule_accuracy":  round(acc_rule, 4),
                "rule_macro_f1":  round(f1_rule,  4),
            },
        }
        joblib.dump(model_data, MODEL_PATH)
        print(f"\n[6] 模型已儲存：{MODEL_PATH.name}")

        print("\n    特徵重要性：")
        for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(imp * 40)
            print(f"    {feat:<14} {bar:<40} {imp:.4f}")

        return True

    print("\n[6] 規則式較好或相當，不取代現有模型。")
    return False


if __name__ == "__main__":
    train()
