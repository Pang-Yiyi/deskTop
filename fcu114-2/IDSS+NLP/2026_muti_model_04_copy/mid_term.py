# =========================
# setup
# =========================

import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import pandas as pd
import numpy as np
import datetime
import matplotlib.ticker as mtick
import matplotlib.cm as cm
from sklearn.linear_model import LinearRegression

# 先設定 seaborn 主題，再設定字體（避免 sns.set_theme 覆蓋字體設定）
sns.set_theme(style="whitegrid")

def setup_chinese_font():
    font_paths = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode MS.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            font_prop = font_manager.FontProperties(fname=path)
            font_name = font_prop.get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["font.sans-serif"] = [font_name]
            print("✅ 成功載入字體:", font_name)
            break
    else:
        plt.rcParams["font.family"] = "Arial Unicode MS"
        plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
        print("⚠️ 使用 fallback 字體: Arial Unicode MS")

    plt.rcParams["axes.unicode_minus"] = False

# 必須在 sns.set_theme() 之後呼叫，否則字體設定會被覆蓋
setup_chinese_font()

def to_int_str(x):
    """'7.0' → '7'，過濾 nan"""
    try:
        return str(int(float(x)))
    except (ValueError, TypeError):
        return None

def to_padded_code(x):
    """'7.0' → '07'，對應 code_change.csv 的兩位零補齊代碼"""
    try:
        return str(int(float(x))).zfill(2)
    except (ValueError, TypeError):
        return None

plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# =========================
# 路徑設定
# =========================

# 以腳本位置為基準，自動找到 DataSets 資料夾
_script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_script_dir, "DataSets", "2025_Big_Data_Analytics_DataBase")
code_change_path = os.path.join(data_dir, "code_change.csv")

if not os.path.exists(data_dir):
    raise Exception(f"資料夾不存在，請確認路徑：{data_dir}")

print("資料夾中的檔案：")
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
print(files)

if len(files) == 0:
    raise Exception("找不到任何CSV檔案")

# =========================
# 載入肇事代碼對照表
# =========================

if os.path.exists(code_change_path):
    code_change_df = pd.read_csv(code_change_path, encoding="utf-8", dtype=str)
    # 欄位名稱：代碼, 肇事原因說明
    col_code = code_change_df.columns[0]
    col_desc = code_change_df.columns[1]
    code_dict = dict(zip(code_change_df[col_code].str.strip(), code_change_df[col_desc].str.strip()))
    print(f"✅ 載入代碼對照表，共 {len(code_dict)} 筆")
else:
    code_dict = {}
    print("⚠️ 找不到 code_change.csv，代碼將不會轉換")

# =========================
# 讀取函數
# =========================

def read_accident_data(month):
    filename = os.path.join(data_dir, f"OpenData_1130{month}.csv.csv")

    if not os.path.exists(filename):
        print(f"檔案不存在: {filename}")
        return None

    print(f"\n嘗試讀取檔案: {filename}")

    for enc in ["big5", "utf-8", "cp950"]:
        try:
            df = pd.read_csv(filename, encoding=enc)
            if df.shape[1] > 1:
                print(f"成功使用 {enc}，欄位數: {df.shape[1]}")
                df["month"] = month
                return df
        except Exception:
            continue

    print(f"❌ 無法讀取: {filename}")
    return None

# =========================
# 讀取 1~12 月
# =========================

all_data = []
successful_reads = 0

for month in range(1, 13):
    print(f"\n=== 正在讀取第 {month} 月 ===")
    data = read_accident_data(month)
    if data is not None:
        all_data.append(data)
        successful_reads += 1
        print(f"第 {month} 月成功，筆數: {len(data)}")
    else:
        print(f"第 {month} 月失敗")

print(f"\n成功讀取 {successful_reads} 個月")

if successful_reads == 0:
    raise Exception("沒有成功讀取任何資料")

# =========================
# 合併資料
# =========================

accident_data = pd.concat(all_data, ignore_index=True)

print("\n每月資料筆數：")
print(accident_data["month"].value_counts().sort_index())
print("\n=== 資料讀取完成 ===")
print("總筆數:", len(accident_data))
print("欄位數:", accident_data.shape[1])
print("欄位名稱:")
print(accident_data.columns.tolist())
print("\n資料預覽：")
print(accident_data.head())

# =========================
# 資料清理
# =========================

missing_values = accident_data.isna().sum()
print("\n各欄位缺失值數量：")
print(missing_values[missing_values > 0])

accident_data_cleaned = accident_data.drop_duplicates()

# 欄位名稱標準化（只影響 ASCII，中文欄位名稱保持不變）
accident_data_cleaned = accident_data_cleaned.copy()
accident_data_cleaned.columns = (
    accident_data_cleaned.columns
    .str.lower()
    .str.replace(r"[^\w]", "_", regex=True)
    .str.replace(r"\s+", "_", regex=True)
)

print("\n清理後欄位名稱:")
print(accident_data_cleaned.columns.tolist())
print("\n資料結構：")
print(accident_data_cleaned.info())

# =========================
# datetime 建立
# =========================

accident_data_cleaned["datetime"] = pd.to_datetime(
    accident_data_cleaned["年"].astype(str) +
    accident_data_cleaned["月"].astype(str).str.zfill(2) +
    accident_data_cleaned["日"].astype(str).str.zfill(2) +
    accident_data_cleaned["時"].astype(str).str.zfill(2) +
    accident_data_cleaned["分"].astype(str).str.zfill(2),
    format="%Y%m%d%H%M",
    errors="coerce"
)

accident_data_cleaned["date"] = pd.to_datetime(
    accident_data_cleaned["年"].astype(str) + "-" +
    accident_data_cleaned["月"].astype(str).str.zfill(2) + "-" +
    accident_data_cleaned["日"].astype(str).str.zfill(2),
    errors="coerce"
)

# =========================
# EDA：小時分析
# =========================

hourly = accident_data_cleaned["時"].value_counts().sort_index()

plt.figure()
colors = cm.viridis(np.linspace(0, 1, 24))
plt.bar(hourly.index, hourly.values, color=colors)
plt.title("每小時事故發生次數")
plt.xlabel("小時")
plt.ylabel("事故次數")
plt.tight_layout()
plt.show()

# =========================
# EDA：星期分析
# =========================

weekday_map = {
    0: "星期一", 1: "星期二", 2: "星期三",
    3: "星期四", 4: "星期五", 5: "星期六", 6: "星期日"
}
weekday_order = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]

weekdays_zh = accident_data_cleaned["date"].dt.weekday.map(weekday_map)
weekly = weekdays_zh.value_counts().reindex(weekday_order)

plt.figure()
colors = cm.viridis(np.linspace(0, 1, 7))
plt.bar(weekly.index, weekly.values, color=colors)
plt.title("每週事故發生次數")
plt.xlabel("星期")
plt.ylabel("事故次數")
plt.tight_layout()
plt.show()

# =========================
# EDA：月份分析
# =========================

print("原始月份資料：")
print(accident_data_cleaned["month"].value_counts().sort_index())

actual_counts = accident_data_cleaned["month"].value_counts()
month_data = pd.DataFrame({
    "month": range(1, 13),
    "count": [actual_counts.get(m, 0) for m in range(1, 13)]
})

plt.figure()
colors = cm.viridis(np.linspace(0, 1, 12))
bars = plt.bar(month_data["month"], month_data["count"], color=colors)
plt.title("每月事故發生次數")
plt.xlabel("月份")
plt.ylabel("事故次數")
plt.xticks(month_data["month"], [f"{m}月" for m in month_data["month"]])

for i, v in enumerate(month_data["count"]):
    plt.text(month_data["month"].iloc[i], v, str(int(v)),
             ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# =========================
# 3.2 主要肇事因素分析
# =========================

weather_mapping = {
    "1": "風",
    "2": "風沙",
    "3": "霧或煙",
    "4": "雪",
    "5": "雨",
    "6": "陰",
    "7": "晴"
}

if "肇事因素主要" in accident_data_cleaned.columns:
    cause_counts = (
        accident_data_cleaned["肇事因素主要"]
        .apply(to_padded_code)
        .dropna()
        .value_counts()
        .head(15)
    )

    print("\n前15個最常見的肇事因素代碼：")
    print(cause_counts)

    cause_df = pd.DataFrame({
        "code": cause_counts.index,
        "count": cause_counts.values
    })

    cause_df["description"] = cause_df["code"].map(code_dict)
    cause_df["description"] = cause_df["description"].fillna(
        cause_df["code"] + "(未知代碼)"
    )

    print("\n前15個最常見的肇事因素及說明：")
    print(cause_df)

    plt.figure(figsize=(12, 5))
    colors = cm.viridis(np.linspace(0, 1, len(cause_df)))
    plt.bar(cause_df["description"], cause_df["count"], color=colors)
    plt.title("主要肇事原因")
    plt.xticks(rotation=90)
    plt.ylabel("次數")
    plt.tight_layout()
    plt.show()

if "肇事因素次要" in accident_data_cleaned.columns:
    sec_counts = (
        accident_data_cleaned["肇事因素次要"]
        .apply(to_padded_code)
        .dropna()
        .value_counts()
        .head(15)
    )

    print("\n前15個最常見的次要肇事因素代碼：")
    print(sec_counts)

    sec_df = pd.DataFrame({
        "code": sec_counts.index,
        "count": sec_counts.values
    })

    sec_df["description"] = sec_df["code"].map(code_dict)
    sec_df["description"] = sec_df["description"].fillna(
        sec_df["code"] + "(未知代碼)"
    )

    print("\n前15個最常見的次要肇事因素及說明：")
    print(sec_df)

    plt.figure(figsize=(12, 5))
    colors = cm.viridis(np.linspace(0, 1, len(sec_df)))
    plt.bar(sec_df["description"], sec_df["count"], color=colors)
    plt.title("次要肇事原因")
    plt.xticks(rotation=90)
    plt.ylabel("次數")
    plt.tight_layout()
    plt.show()

# =========================
# 3.3 區域分析
# =========================

if "區" in accident_data_cleaned.columns:
    district = (
        accident_data_cleaned["區"]
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 5))
    colors = cm.viridis(np.linspace(0, 1, len(district)))
    plt.bar(district.index, district.values, color=colors)
    plt.title("各區事故發生次數")
    plt.ylabel("事故數")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


# =========================
# 天候與事故關係
# =========================

if "天候" in accident_data_cleaned.columns:
    weather_counts = (
        accident_data_cleaned["天候"]
        .apply(to_int_str)
        .dropna()
        .value_counts()
    )

    weather_labels = weather_counts.index.map(
        lambda x: weather_mapping.get(x, f"{x}(未知)")
    )

    plt.figure()
    colors = cm.viridis(np.linspace(0, 1, len(weather_counts)))
    plt.bar(weather_labels, weather_counts.values, color=colors)
    plt.title("天候與事故關係")
    plt.ylabel("事故數")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# =========================
# 飲酒情形分析
# =========================

alcohol_mapping = {
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
    "11": "駕駛人不明"
}

if "飲酒情形" in accident_data_cleaned.columns:
    alcohol_counts = (
        accident_data_cleaned["飲酒情形"]
        .apply(to_int_str)
        .dropna()
        .value_counts()
    )

    labels = [
        alcohol_mapping.get(code, f"{code}(未知)")
        for code in alcohol_counts.index
    ]

    plt.figure()
    colors = cm.viridis(np.linspace(0, 1, len(alcohol_counts)))
    bars = plt.bar(labels, alcohol_counts.values, color=colors)
    plt.title("飲酒情形與事故關係")
    plt.xticks(rotation=90, fontsize=8)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    y_max = alcohol_counts.max()
    ax.set_yticks(np.linspace(0, y_max, 10))
    plt.ylabel("事故數量（件）")
    plt.tight_layout()
    plt.show()

# =========================
# 回歸分析
# =========================

hourly_counts = (
    accident_data_cleaned["時"]
    .value_counts()
    .sort_index()
)

hourly_data = pd.DataFrame({
    "hour": hourly_counts.index.astype(int),
    "accidents": hourly_counts.values
})

X = np.column_stack([
    hourly_data["hour"],
    hourly_data["hour"] ** 2
])
y = hourly_data["accidents"]

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure()
plt.scatter(hourly_data["hour"], y, label="實際值")
plt.plot(hourly_data["hour"], y_pred, color="red", label="預測曲線")
plt.title("事故數量與時段的關係")
plt.xlabel("小時")
plt.ylabel("事故數量")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Heatmap：天候 × 時段
# =========================

_weather_code = accident_data_cleaned["天候"].apply(to_int_str)
heatmap_data = pd.crosstab(
    _weather_code[_weather_code.notna()],
    accident_data_cleaned.loc[_weather_code.notna(), "時"]
)

heatmap_data.index = [
    weather_mapping.get(code, f"{code}(未知)")
    for code in heatmap_data.index
]

plt.figure(figsize=(14, 5))
sns.heatmap(
    heatmap_data,
    cmap=sns.color_palette(["yellow", "orange", "red", "darkred"], as_cmap=True),
    linewidths=0.5
)
plt.title("天候與時段的事故分布")
plt.xlabel("小時")
plt.ylabel("天候")
plt.tight_layout()
plt.show()
