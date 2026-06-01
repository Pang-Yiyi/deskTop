"""BM25 稀疏檢索引擎：台中市交通事故知識庫。

對應課程：NLP Week 8（BM25 稀疏檢索 + jieba 斷詞）

架構：
  KnowledgeBase    — 知識條目結構化建置（代碼說明 + 欄位說明 + 系統限制）
  BM25RAG          — BM25Okapi 檢索引擎，jieba 斷詞作為 tokenizer
  search()         — 統一查詢介面，支援 top-k 結果與最低分數門檻

使用方式：
    rag = BM25RAG()
    results = rag.search("天候欄位代碼怎麼看", top_k=3)
    results = rag.search("未保持安全距離代碼07", top_k=5)
"""

from __future__ import annotations

from typing import Any

try:
    from rank_bm25 import BM25Okapi
    _BM25_OK = True
except ImportError:
    _BM25_OK = False

from text_preprocessor import tokenize


# ─────────────────────────────────────────────────────────
# 靜態知識條目（欄位說明 / 系統資訊）
# ─────────────────────────────────────────────────────────

_STATIC_ENTRIES: list[dict[str, str]] = [
    # 天候欄位
    {
        "id": "field_weather",
        "title": "天候欄位說明",
        "content": (
            "天候欄位記錄事故發生當時的天氣狀況。"
            "代碼對應：1=風，2=風沙，3=霧或煙，4=雪，5=雨，6=陰，7=晴。"
            "台中市事故以晴天（代碼7）最多，其次為陰天（6）與雨天（5）。"
        ),
        "type": "field_desc",
        "code": "",
    },
    # 各天候代碼
    {
        "id": "weather_1",
        "title": "天候代碼 1（風）",
        "content": "天候代碼 1 代表「風」，指大風或強風天候，可能影響騎車穩定性。",
        "type": "weather_code",
        "code": "1",
    },
    {
        "id": "weather_2",
        "title": "天候代碼 2（風沙）",
        "content": "天候代碼 2 代表「風沙」，視線受沙塵影響，較少見於台中市。",
        "type": "weather_code",
        "code": "2",
    },
    {
        "id": "weather_3",
        "title": "天候代碼 3（霧或煙）",
        "content": (
            "天候代碼 3 代表「霧或煙」，包含起霧、濃霧、煙霧等能見度低的天候。"
            "霧天視線受阻，駕駛人反應時間縮短，需降速並保持安全距離。"
        ),
        "type": "weather_code",
        "code": "3",
    },
    {
        "id": "weather_4",
        "title": "天候代碼 4（雪）",
        "content": "天候代碼 4 代表「雪」，路面打滑，煞車距離大幅增加。台中市極少見。",
        "type": "weather_code",
        "code": "4",
    },
    {
        "id": "weather_5",
        "title": "天候代碼 5（雨）",
        "content": (
            "天候代碼 5 代表「雨」，包含小雨至大雨。"
            "雨天路面濕滑，輪胎與路面摩擦係數降低，煞車距離增加，能見度下降。"
            "台中市雨天事故佔全部事故約 5-7%。"
        ),
        "type": "weather_code",
        "code": "5",
    },
    {
        "id": "weather_6",
        "title": "天候代碼 6（陰）",
        "content": "天候代碼 6 代表「陰」，無直接日曬但視線尚可。台中市陰天事故件數次於晴天。",
        "type": "weather_code",
        "code": "6",
    },
    {
        "id": "weather_7",
        "title": "天候代碼 7（晴）",
        "content": (
            "天候代碼 7 代表「晴」，為台中市最常見天候。"
            "晴天事故件數最多，主要因為晴天出行量大，絕對件數高。"
        ),
        "type": "weather_code",
        "code": "7",
    },
    # 行政區欄位
    {
        "id": "field_district",
        "title": "行政區欄位說明",
        "content": (
            "行政區欄位記錄事故發生的台中市行政區，共 29 個行政區。"
            "事故數前五名：西屯區、北屯區、南屯區、北區、大里區。"
            "西屯區因商業活動密集、幹道多，長期居事故排名第一。"
        ),
        "type": "field_desc",
        "code": "",
    },
    # 時段欄位
    {
        "id": "field_hour",
        "title": "時段欄位說明",
        "content": (
            "時段（小時）欄位記錄事故發生的整點時刻（0-23 時）。"
            "事故高峰時段為下午 17-18 時（下班）與早上 7-8 時（上班）。"
            "凌晨 2-4 時事故件數最少，但嚴重度相對較高（疲勞駕駛、車速快）。"
        ),
        "type": "field_desc",
        "code": "",
    },
    # 資料集說明
    {
        "id": "data_scope",
        "title": "資料集範圍說明",
        "content": (
            "本系統使用台中市民國 113 年（西元 2024 年）1 至 12 月交通事故開放資料，"
            "共約 15 萬筆事故記錄。"
            "資料由台中市政府警察局提供，包含事故時間、地點、天候、肇事因素等欄位。"
            "本資料不包含即時天氣、即時車流或其他縣市資料。"
        ),
        "type": "data_info",
        "code": "",
    },
    # 飲酒欄位
    {
        "id": "field_alcohol",
        "title": "飲酒情形欄位說明",
        "content": (
            "飲酒情形欄位記錄事故相關人員是否有飲酒。"
            "代碼：0=無，1=飲酒（未超標），2=酒測值超標。"
            "酒後駕車（酒駕）會嚴重影響反應時間、視野與判斷力，是高嚴重度事故的重要肇因。"
        ),
        "type": "field_desc",
        "code": "",
    },
    # 酒駕專項說明
    {
        "id": "cause_alcohol",
        "title": "酒駕肇因說明",
        "content": (
            "酒駕（飲酒後駕車）是指駕駛人在飲酒後操作車輛。"
            "酒精會降低反應速度、視野縮小並影響判斷，導致事故嚴重度顯著上升。"
            "「飲酒情形」欄位代碼 2 表示酒測超標；相關肇事因素通常標記於肇因代碼欄位。"
        ),
        "type": "cause_info",
        "code": "",
    },
]


# ─────────────────────────────────────────────────────────
# BM25 RAG 引擎
# ─────────────────────────────────────────────────────────

class BM25RAG:
    """BM25 稀疏檢索引擎。

    初始化時從 data_loader 載入代碼字典，並與靜態條目合併建立索引。
    使用 jieba 斷詞作為 tokenizer（Week 2 技術整合）。
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, str]] = []
        self._bm25: Any = None
        self._ready = False
        self._build_index()

    def _build_index(self) -> None:
        """建立 BM25 索引。"""
        entries = list(_STATIC_ENTRIES)

        # 動態載入肇事因素代碼（從 CSV）
        try:
            from data_loader import load_code_dict
            code_dict = load_code_dict()
            for code, desc in code_dict.items():
                entries.append({
                    "id": f"cause_{code}",
                    "title": f"肇事因素代碼 {code}",
                    "content": (
                        f"肇事因素代碼 {code} 代表「{desc}」。"
                        f"此代碼用於標記造成事故的主要或次要肇因。"
                    ),
                    "type": "cause_code",
                    "code": code,
                })
        except Exception:
            pass  # 資料不可用時僅用靜態條目

        self._docs = entries

        if not _BM25_OK or not entries:
            return

        # jieba 斷詞後建立 BM25 索引
        # 搜尋欄位 = title + content（加權 title 兩倍，增強代碼匹配）
        corpus = [
            tokenize(doc["title"] + " " + doc["title"] + " " + doc["content"])
            for doc in entries
        ]
        self._bm25 = BM25Okapi(corpus)
        self._ready = True

    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> list[dict]:
        """BM25 檢索，回傳最相關的 top_k 條目。

        Parameters
        ----------
        query     : 查詢字串
        top_k     : 最多回傳幾筆
        min_score : 最低 BM25 分數門檻（過濾不相關結果）

        Returns
        -------
        list of dicts with keys: id, title, content, type, code, score
        """
        if not self._ready or not self._bm25:
            return self._fallback_search(query, top_k)

        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # 取 top_k 最高分、且分數超過門檻
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )
        results = []
        for idx, score in ranked[:top_k * 2]:  # 多取再過濾
            if score >= min_score:
                doc = dict(self._docs[idx])
                doc["score"] = round(float(score), 4)
                results.append(doc)
            if len(results) >= top_k:
                break

        return results if results else self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 不可用或無結果時，回退到 substring 搜尋。"""
        results = []
        query_lower = query.lower()
        for doc in self._docs:
            searchable = (doc["title"] + doc["content"] + doc.get("code", "")).lower()
            if any(term in searchable for term in query_lower.split()):
                doc_copy = dict(doc)
                doc_copy["score"] = 0.0
                results.append(doc_copy)
                if len(results) >= top_k:
                    break
        return results

    @property
    def ready(self) -> bool:
        return self._ready


# 模組級單例（避免重複建索引）
_rag_instance: BM25RAG | None = None


def get_rag() -> BM25RAG:
    """回傳全域 BM25RAG 單例。"""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = BM25RAG()
    return _rag_instance
