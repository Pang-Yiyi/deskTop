"""中文文本前處理：jieba 分詞、行政區別名解析、實體輔助提取。

對應課程：NLP Week 2（中文分詞 / jieba）
用途：
  - tokenize()              jieba 斷詞，供後續關鍵詞匹配與 BM25 使用
  - extract_transport_mode()  從 token 清單提取交通工具
  - extract_origin_destination()  從「從X到Y / 從X去Y」模式提取起訖行政區
"""

from __future__ import annotations

import re
from typing import Any

try:
    import jieba
    jieba.setLogLevel(20)  # 關閉 DEBUG 輸出
    _JIEBA_OK = True
except ImportError:
    _JIEBA_OK = False


# ── 行政區短別名對照（不含「區」後綴的口語說法）────────────────
# 目的：識別「開車從豐原到西屯」中的「豐原」、「西屯」
DISTRICT_ALIASES: dict[str, str] = {
    "豐原": "豐原區",
    "西屯": "西屯區",
    "北屯": "北屯區",
    "南屯": "南屯區",
    "大里": "大里區",
    "太平": "太平區",
    "烏日": "烏日區",
    "霧峰": "霧峰區",
    "后里": "后里區",
    "石岡": "石岡區",
    "東勢": "東勢區",
    "和平": "和平區",
    "新社": "新社區",
    "潭子": "潭子區",
    "大雅": "大雅區",
    "神岡": "神岡區",
    "大肚": "大肚區",
    "沙鹿": "沙鹿區",
    "龍井": "龍井區",
    "梧棲": "梧棲區",
    "清水": "清水區",
    "大甲": "大甲區",
    "外埔": "外埔區",
    "大安": "大安區",
}

# ── 台中知名地標 → 行政區對照 ───────────────────────────────
# 讓「逢甲」「火車站」「一中」等口語地標能對應到實際行政區
LANDMARK_TO_DISTRICT: dict[str, str] = {
    "逢甲": "西屯區", "逢甲大學": "西屯區", "逢甲夜市": "西屯區",
    "火車站": "中區", "台中車站": "中區", "台中火車站": "中區",
    "高鐵": "烏日區", "高鐵站": "烏日區", "高鐵台中站": "烏日區",
    "一中": "北區", "一中街": "北區", "中友": "北區",
    "中friday": "北區", "中國醫": "北區",
    "科博館": "北區", "植物園": "北區",
    "東海": "龍井區", "東海大學": "龍井區",
    "中科": "西屯區", "台中工業區": "南屯區",
    "勤美": "西區", "草悟道": "西區", "美術館": "西區", "審計新村": "西區",
    "市政府": "西屯區", "市府": "西屯區", "新市政": "西屯區",
    "七期": "西屯區", "老虎城": "西屯區", "tigercity": "西屯區",
    "大遠百": "西屯區", "新光三越": "西屯區",
    "中清路": "北屯區", "崇德": "北屯區",
    "中山醫": "南區", "文心森林公園": "南屯區",
    "干城": "東區", "台中公園": "中區", "宮原眼科": "中區",
    "彩虹眷村": "南屯區", "麻園頭": "西區",
    "新時代": "東區", "秀泰": "東區",
    "大里夜市": "大里區", "國光花市": "大里區",
    "豐原廟東": "豐原區", "廟東夜市": "豐原區",
    "三井": "梧棲區", "三井outlet": "梧棲區", "港區": "梧棲區",
    "麗寶": "后里區", "麗寶樂園": "后里區", "outlet": "后里區",
    "中正紀念": "中區",
}

# ── 交通工具詞典 ────────────────────────────────────────────
TRANSPORT_VOCAB: set[str] = {
    "機車", "汽車", "行人", "自行車", "腳踏車",
    "大眾運輸", "捷運", "公車", "開車", "騎車", "騎機車",
}

# 口語 → 標準詞（正規化）
TRANSPORT_NORMALIZE: dict[str, str] = {
    "腳踏車": "自行車",
    "開車": "汽車",
    "騎車": "機車",
    "騎機車": "機車",
}

# 加入自訂詞：交通工具與行政區別名，確保 jieba 不拆分這些詞
if _JIEBA_OK:
    for _term in TRANSPORT_VOCAB:
        jieba.add_word(_term)
    for _alias in DISTRICT_ALIASES:
        jieba.add_word(_alias)


def tokenize(text: str) -> list[str]:
    """jieba 斷詞（含自訂詞典）；jieba 不可用時回傳逐字列表作為 fallback。"""
    if _JIEBA_OK:
        return list(jieba.lcut(text))
    return list(text)


def extract_transport_mode(tokens: list[str], text: str = "") -> str | None:
    """從 jieba token 清單提取交通工具，並正規化。

    優先使用 token 比對（精確）；若 jieba 斷詞不理想，再對原文做
    substring fallback（長詞優先）。

    Example:
        tokens = ["我", "要", "騎", "機車", "去", ...]  → "機車"
    """
    # token 比對
    for token in tokens:
        if token in TRANSPORT_VOCAB:
            return TRANSPORT_NORMALIZE.get(token, token)
    # substring fallback（長詞優先，避免「機車」被「腳踏車」覆蓋）
    if text:
        for term in sorted(TRANSPORT_VOCAB, key=len, reverse=True):
            if term in text:
                return TRANSPORT_NORMALIZE.get(term, term)
    return None


def resolve_district(text: str) -> str | None:
    """從文字辨識單一行政區：完整名 → 別名 → 地標，取第一個出現的。"""
    candidates: list[tuple[int, str]] = []
    # 地標（長詞優先比對，例如「逢甲夜市」優於「逢甲」）
    for landmark in sorted(LANDMARK_TO_DISTRICT, key=len, reverse=True):
        idx = text.find(landmark)
        if idx >= 0:
            candidates.append((idx, LANDMARK_TO_DISTRICT[landmark]))
            break
    # 別名
    for alias, full in DISTRICT_ALIASES.items():
        idx = text.find(alias)
        if idx >= 0:
            candidates.append((idx, full))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def extract_origin_destination(
    text: str,
    districts: list[str],
) -> tuple[str | None, str | None]:
    """從「從X到Y」或「從X去Y」模式提取起點與終點行政區。

    支援完整名稱（西屯區）、短別名（西屯、豐原）與地標（逢甲、火車站）。

    Returns:
        (origin_district, destination_district)
        若只找到一個 → (origin, None)
        若無路線模式  → (None, None)
    """
    # 收集文中出現的所有地點（行政區/別名/地標），依文字位置排序
    found: list[tuple[int, str]] = []
    seen_pos: set[int] = set()

    def _add(idx: int, name: str) -> None:
        if idx >= 0:
            found.append((idx, name))

    for d in districts:
        _add(text.find(d), d)
    for alias, full_name in DISTRICT_ALIASES.items():
        _add(text.find(alias), full_name)
    for landmark, full_name in LANDMARK_TO_DISTRICT.items():
        _add(text.find(landmark), full_name)

    found.sort(key=lambda x: x[0])
    # 依出現位置去重（同位置只留一個；不同位置的同行政區保留，因起訖可能同區）
    deduped: list[tuple[int, str]] = []
    for idx, name in found:
        if idx not in seen_pos:
            seen_pos.add(idx)
            deduped.append((idx, name))

    # 僅在出現「從...到」或「從...去」路線模式時提取起訖點
    has_route_pattern = bool(re.search(r"從.{1,12}[到去往]", text))
    if not has_route_pattern:
        return None, None

    if len(deduped) >= 2:
        return deduped[0][1], deduped[1][1]
    if len(deduped) == 1:
        return deduped[0][1], None

    return None, None


def jieba_available() -> bool:
    """回傳 jieba 是否已安裝。"""
    return _JIEBA_OK
