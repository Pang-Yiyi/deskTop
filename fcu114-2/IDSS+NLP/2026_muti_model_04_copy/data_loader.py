"""
data_loader.py
負責讀取與清理 12 個月交通事故資料，並建立完整分析用欄位。
"""

import os
import pandas as pd
import numpy as np

# ── 路徑設定 ──────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_script_dir, "DataSets", "2025_Big_Data_Analytics_DataBase")
CODE_CHANGE_PATH = os.path.join(DATA_DIR, "code_change.csv")

# ── 代碼對照表 ────────────────────────────────────────────
WEATHER_MAPPING = {
    "1": "風",
    "2": "風沙",
    "3": "霧或煙",
    "4": "雪",
    "5": "雨",
    "6": "陰",
    "7": "晴",
}

ALCOHOL_MAPPING = {
    "1": "未飲酒",
    "2": "檢測無酒精",
    "3": "未超標",
    "4": "輕度超標",
    "5": "中輕度超標",
    "6": "中度超標",
    "7": "中重度超標",
    "8": "重度超標",
    "9": "駕駛人無法測",
    "10": "非駕駛人未測",
    "11": "駕駛人不明",
}

WEEKDAY_MAP = {
    0: "星期一",
    1: "星期二",
    2: "星期三",
    3: "星期四",
    4: "星期五",
    5: "星期六",
    6: "星期日",
}


def _to_int_str(x):
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return None


def _to_padded_code(x):
    try:
        return str(int(float(x))).zfill(2)
    except (ValueError, TypeError):
        return None


def load_code_dict() -> dict:
    """載入肇事因素代碼對照表，回傳 {代碼: 說明} dict。"""
    if not os.path.exists(CODE_CHANGE_PATH):
        return {}
    df = pd.read_csv(CODE_CHANGE_PATH, encoding="utf-8", dtype=str)
    col_code, col_desc = df.columns[0], df.columns[1]
    return dict(zip(df[col_code].str.strip(), df[col_desc].str.strip()))


def _read_month(month: int) -> pd.DataFrame | None:
    candidates = [
        os.path.join(DATA_DIR, f"OpenData_1130{month}.csv"),
        os.path.join(DATA_DIR, f"OpenData_1130{month}.csv.csv"),
    ]
    filename = next((path for path in candidates if os.path.exists(path)), None)
    if filename is None:
        return None

    for enc in ["utf-8", "big5", "cp950"]:
        try:
            df = pd.read_csv(filename, encoding=enc)
            if df.shape[1] > 1:
                df["month"] = month
                return df
        except Exception:
            continue
    return None


def load_accident_data() -> pd.DataFrame:
    """
    讀取並清理全年 12 個月事故資料。

    回傳的 DataFrame 包含以下額外欄位：
        datetime  : pd.Timestamp（事故發生時間）
        date      : pd.Timestamp（日期，不含時分）
        weekday   : str（星期幾，中文，例如「星期五」）
        hour      : int（0-23）
        天候_str  : str（天候中文說明）
        飲酒情形_str : str（飲酒情形中文說明）
        肇事因素主要_str : str（主要肇事因素中文說明）
    """
    all_data = []
    for month in range(1, 13):
        df = _read_month(month)
        if df is not None:
            all_data.append(df)

    if not all_data:
        raise RuntimeError("找不到任何 CSV 資料，請確認 DataSets 資料夾。")

    df = pd.concat(all_data, ignore_index=True)

    # 移除完全重複列
    df = df.drop_duplicates()

    # 欄位名稱標準化（保留中文）
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
    )

    # ── 時間欄位 ──────────────────────────────────────────
    df["datetime"] = pd.to_datetime(
        df["年"].astype(str)
        + df["月"].astype(str).str.zfill(2)
        + df["日"].astype(str).str.zfill(2)
        + df["時"].astype(str).str.zfill(2)
        + df["分"].astype(str).str.zfill(2),
        format="%Y%m%d%H%M",
        errors="coerce",
    )

    df["date"] = pd.to_datetime(
        df["年"].astype(str)
        + "-"
        + df["月"].astype(str).str.zfill(2)
        + "-"
        + df["日"].astype(str).str.zfill(2),
        errors="coerce",
    )

    df["weekday"] = df["date"].dt.weekday.map(WEEKDAY_MAP)
    df["hour"] = pd.to_numeric(df["時"], errors="coerce").astype("Int64")

    # ── 代碼轉中文 ────────────────────────────────────────
    code_dict = load_code_dict()

    if "天候" in df.columns:
        df["天候_str"] = df["天候"].apply(_to_int_str).map(
            lambda x: WEATHER_MAPPING.get(x, x) if x else None
        )

    if "飲酒情形" in df.columns:
        df["飲酒情形_str"] = df["飲酒情形"].apply(_to_int_str).map(
            lambda x: ALCOHOL_MAPPING.get(x, x) if x else None
        )

    if "肇事因素主要" in df.columns:
        df["肇事因素主要_str"] = df["肇事因素主要"].apply(_to_padded_code).map(
            lambda x: code_dict.get(x, x) if x else None
        )

    return df


def get_data_summary() -> dict:
    """
    回傳資料集的基本摘要，包含時間範圍與總筆數。

    Returns
    -------
    dict with keys: total_count, date_min, date_max, date_range_str, year_range_str
    """
    df = load_accident_data()
    total_count = len(df)

    date_min = df["date"].dropna().min() if "date" in df.columns else None
    date_max = df["date"].dropna().max() if "date" in df.columns else None

    if date_min is not None and date_max is not None:
        date_range_str = f"{date_min.strftime('%Y-%m-%d')} 至 {date_max.strftime('%Y-%m-%d')}"
        # 轉民國年顯示
        roc_start = date_min.year - 1911
        roc_end = date_max.year - 1911
        year_range_str = (
            f"民國 {roc_start} 年" if roc_start == roc_end
            else f"民國 {roc_start}–{roc_end} 年"
        )
    else:
        date_range_str = "日期資料不完整"
        year_range_str = "年份不明"

    return {
        "total_count": total_count,
        "date_min": str(date_min) if date_min is not None else None,
        "date_max": str(date_max) if date_max is not None else None,
        "date_range_str": date_range_str,
        "year_range_str": year_range_str,
    }


if __name__ == "__main__":
    df = load_accident_data()
    print(f"總筆數: {len(df)}")
    print(f"欄位: {df.columns.tolist()}")
    print(df[["date", "weekday", "hour", "天候_str", "肇事因素主要_str"]].head())
