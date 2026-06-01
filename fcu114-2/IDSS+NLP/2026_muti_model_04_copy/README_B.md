# 台中市交通事故風險預測與決策支援系統

> **課程**：智慧決策支援系統（IDSS）× 自然語言處理（NLP）期末專題  
> **資料夾路徑**：`fcu114-2/IDSS+NLP/2026_muti_model_04_copy/`

---

## 系統簡介

以**自然語言問答**為介面的交通事故風險決策平台。使用者用口語提問（例如「西屯區現在開車危險嗎」），系統自動完成意圖辨識、資料查詢、RF 模型預測，並生成口語化回答。

整合三類即時資料：**歷史事故（14.5 萬筆）× 即時天氣（CWA）× 即時車流（TDX）**。

---

## 快速啟動

### 前置需求

```bash
pip install streamlit pandas numpy scikit-learn rank_bm25 jieba joblib
```

### 安裝並啟動本機 LLM（必要）

```bash
# 安裝 Ollama（https://ollama.com）後執行：
ollama pull qwen3:4b
ollama serve          # 背景執行
```

### 啟動系統

```bash
# 在 2026_muti_model_04_copy/ 資料夾內執行
streamlit run app.py
```

瀏覽器開啟 **http://localhost:8501** 即可使用。

### API 金鑰設定（選配，已有 `.env` 可略過）

`.env` 檔已放好，內容如下（不需要再動）：

```env
CWA_API_KEY=CWA-E445582F-9CBF-41B2-A8B8-21928EE8F640
CLIENT_ID=d1245810-04bac901-641a-4f63       # TDX 車流
CLIENT_SECRET=3a0c44ce-7a05-4ff5-84b5-9b575b725b60
ENABLE_LOCAL_LLM=1
LLM_BACKEND=ollama
LLM_MODEL=qwen3:4b
```

---

## 資料夾結構說明

```
2026_muti_model_04_copy/
│
├── 📁 DataSets/                    原始事故資料
│   └── 2025_Big_Data_Analytics_DataBase/
│       ├── *.csv                   台中市113年事故記錄（A1/A2類）
│       └── code_change.csv         肇事因素代碼對照表
│
├── 📁 charts/                      EDA 圖表（決策分析頁用）
│   ├── hourly_accidents.png        每小時事故分布
│   ├── district_accidents.png      行政區排名
│   ├── main_causes.png             主要肇因 Top 10
│   ├── weather_accidents.png       天候分析
│   ├── weekly_accidents.png        每週分布
│   ├── monthly_accidents.png       每月分布
│   ├── alcohol_accidents.png       飲酒分析
│   ├── hourly_regression.png       時段迴歸
│   └── weather_hour_heatmap.png    天候×時段熱力圖
│
├── 📁 __pycache__/                 Python 編譯快取（自動生成，可刪）
│
├── ─── 核心系統程式 ────────────────────────────────────────
│
├── app.py                          Streamlit 主程式（UI 入口）
├── agents.py                       多代理人管線主控
│                                   - 規則式 NLU（意圖/實體辨識）
│                                   - 地標→行政區解析
│                                   - 工具計畫與執行
│                                   - 即時資料注入邏輯
│
├── ─── NLP 核心模組 ────────────────────────────────────────
│
├── text_preprocessor.py            中文前處理（Week 2）
│                                   - jieba 斷詞 + 自訂詞典
│                                   - 台中地標對照表（40+ 條）
│                                   - 交通工具/起訖點擷取
│
├── nlu_parser.py                   本機 LLM 語義解析（Week 6-7）
│                                   - Few-shot Prompting（12 個範例）
│                                   - 輸出 intent + entities JSON
│
├── bm25_rag.py                     BM25 稀疏檢索知識庫（Week 8）
│                                   - 132 條知識條目
│                                   - jieba 斷詞 + rank_bm25
│
├── llm_orchestrator.py             回答生成層
│                                   - 事實卡片組裝（防幻覺）
│                                   - LLM 口語化（Qwen3 4B）
│                                   - 輸出清洗（正則去 meta）
│
├── local_llm_client.py             Ollama/llama.cpp 客戶端
├── prompts.py                      系統提示與回答規則
│
├── ─── 資料與分析工具 ──────────────────────────────────────
│
├── data_loader.py                  CSV 讀取與前處理
├── analysis_tools.py               事故查詢、肇因分析、出行建議、RAG
├── risk_model.py                   風險評分模型
│                                   - 優先使用 RF 預測模型
│                                   - fallback 規則式評分
├── advanced_analysis_tools.py      GPS 熱點聚合、Markdown 報告匯出
├── external_api_tools.py           CWA 天氣 + TDX 車流 API
├── mock_tools.py                   備援模擬工具（真實工具失敗時用）
│
├── ─── 機器學習模型 ────────────────────────────────────────
│
├── rf_risk_model.pkl               訓練好的 Random Forest 模型
│                                   Accuracy 88.8%，Macro-F1 0.855
│                                   （訓練資料：145,565 筆事故）
│
├── train_risk_model.py             RF 模型訓練腳本
│                                   執行: python train_risk_model.py
│                                   自動比較 RF vs 規則式，較好才存檔
│
├── ─── NLP 評估實驗 ─────────────────────────────────────────
│
├── nlp_evaluation.py               NLP 評估腳本（跑實驗用）
├── nlp_eval_dataset.json           30 題標註測試集（⚠️ 勿刪）
│                                   涵蓋 9 種意圖 + 邊界案例
│
├── ─── 設定檔 ───────────────────────────────────────────────
│
├── .env                            API 金鑰（⚠️ 勿上傳 Git）
├── .env.example                    範本
├── env_loader.py                   讀取 .env 的工具
│
├── ─── 期末報告 ─────────────────────────────────────────────
│
├── NLP期末專題報告.md               NLP 課程用報告
├── IDSS期末專題報告.md              IDSS 課程用報告
│
└── ─── 其他 ─────────────────────────────────────────────────
    ├── run_streamlit.bat            Windows 一鍵啟動腳本
    ├── mid_term.py                  期中作業（系統已不引用，可刪）
    └── Figure_1~9.png               EDA 補充圖表（決策分析頁顯示）
```

---

## 各模組分工說明

### 一次問答的完整流程

```
使用者輸入「西屯區現在開車危險嗎」
    │
    ├─ agents.py → 規則式 NLU
    │   text_preprocessor.py → jieba 斷詞
    │   → intent=風險預測, district=西屯區
    │
    ├─ agents.py → 即時條件注入
    │   外部 API: 現在 14:30 晴天，車流稀少（推估）
    │   → query: {district:西屯, hour:14, weather:晴, weekday:星期六}
    │
    ├─ agents.py → 工具執行
    │   risk_model.py → RF 模型預測 → 中風險 55 分
    │   analysis_tools.py → 事故查詢、肇因、天候熱力圖
    │
    ├─ llm_orchestrator.py → 事實卡片組裝
    │   bm25_rag.py → BM25 檢索相關知識（可選）
    │   → 送給 Qwen3 4B 口語化
    │
    └─ app.py → 顯示在聊天介面
        「西屯區開車現在中風險55分，下午車流普通，開車時別分心。」
```

---

## 報告所需內容位置

### NLP 課程報告用

| 報告章節 | 資料位置 |
|---|---|
| jieba 斷詞實作 | `text_preprocessor.py` 第 1–80 行 |
| Few-shot Prompting 範例 | `nlu_parser.py` `_FEW_SHOT_EXAMPLES` |
| BM25 RAG 架構 | `bm25_rag.py` |
| 意圖分類準確率表 | `nlp_eval_report.md` |
| 實體擷取準確率 | `nlp_eval_report.md` |
| 評估測試集 | `nlp_eval_dataset.json` |
| 重新跑評估 | `python nlp_evaluation.py` |

### IDSS 課程報告用

| 報告章節 | 資料位置 |
|---|---|
| RF 模型訓練與評估 | `train_risk_model.py` |
| 特徵重要性數據 | `rf_risk_model.pkl`（跑 `train_risk_model.py` 的輸出）|
| 描述層：EDA 圖表 | `charts/` 資料夾 |
| 預測層：風險評分 | `risk_model.py` |
| 處方層：勤務覆蓋率 | `app.py` `_render_patrol_coverage()` |
| 即時資料整合 | `external_api_tools.py` |
| 系統截圖 | 啟動後自行截圖 http://localhost:8501 |

---

## 展示 Demo 建議問題

| 意圖類型 | 建議問法 |
|---|---|
| 🔴 風險預測（即時） | 西屯區現在開車危險嗎 |
| 🔴 風險預測（情境） | 西屯區星期五晚上六點雨天危險嗎？ |
| 🚗 出行建議 | 現在從逢甲騎機車去火車站危險嗎 |
| ⏰ 時段查詢 | 什麼時間最容易發生交通事故 |
| 📍 熱點查詢 | 台中市哪個行政區事故最多 |
| 🔍 肇因分析 | 最常見的肇事原因是什麼 |
| 📋 政策建議（公家單位） | 如果我是交通局應該優先改善什麼 |
| 📖 代碼說明 | 肇事因素代碼07是什麼 |
| 🚫 超出範圍（展示邊界） | 台北市事故最多哪個行政區 |

---

## 常見問題

**Q：系統回答很慢？**  
A：Qwen3 4B 每題約 5–10 秒，正常。第一題較慢（模型載入），之後較快。

**Q：出現「LLM 不可用」或退回規則式回答？**  
A：確認 Ollama 是否在背景執行（`ollama serve`），且已下載 `qwen3:4b`。

**Q：車流顯示「歷史推估」？**  
A：TDX 免費層資料更新頻率不穩定，系統自動以歷史時段密度推估，屬正常行為。

**Q：想重新訓練 RF 模型？**  
A：執行 `python train_risk_model.py`，會自動比較新模型與規則式，較好才覆蓋 `rf_risk_model.pkl`。

**Q：想重新跑 NLP 評估？**  
A：執行 `python nlp_evaluation.py`，結果會更新 `nlp_eval_report.md` 與 `nlp_eval_results.json`。

---

## 注意事項

- ⚠️ `.env` 含 API 金鑰，**不要上傳 Git**
- ⚠️ `nlp_eval_dataset.json` 是評估測試集，**不要刪除**
- ⚠️ `rf_risk_model.pkl` 是訓練好的模型，**不要刪除**（刪了風險預測退回規則式）
- ✅ `__pycache__/` 可以刪，會自動重生
- ✅ `*.log`、`streamlit_*.log` 可以刪，是舊的執行記錄
