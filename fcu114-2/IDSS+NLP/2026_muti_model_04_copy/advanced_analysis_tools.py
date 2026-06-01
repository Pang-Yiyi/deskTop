"""Advanced local analysis tools: segment aggregation and report export."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def build_segment_hotspots(
    map_df: pd.DataFrame,
    *,
    precision: int = 3,
    top_n: int = 12,
) -> dict[str, Any]:
    """Approximate road/intersection hotspots by clustering GPS points in grid cells."""
    if map_df.empty or "lat" not in map_df.columns or "lon" not in map_df.columns:
        return {"segments": [], "summary": "目前沒有可用 GPS 點位可進行路段級聚合。"}

    work = map_df.dropna(subset=["lat", "lon"]).copy()
    if work.empty:
        return {"segments": [], "summary": "目前沒有有效 GPS 點位可進行路段級聚合。"}

    work["segment_lat"] = work["lat"].round(precision)
    work["segment_lon"] = work["lon"].round(precision)
    grouped = (
        work.groupby(["segment_lat", "segment_lon"], dropna=True)
        .agg(
            accident_count=("lat", "size"),
            top_district=("區", _mode_or_empty),
            top_weather=("天候_str", _mode_or_empty),
            peak_hour=("hour", _mode_or_empty),
        )
        .reset_index()
        .sort_values("accident_count", ascending=False)
        .head(top_n)
    )

    segments = []
    max_count = int(grouped["accident_count"].max()) if not grouped.empty else 1
    for index, row in enumerate(grouped.itertuples(index=False), start=1):
        count = int(row.accident_count)
        risk_score = round(count / max_count * 100) if max_count else 0
        if risk_score >= 70:
            level = "高風險熱點"
        elif risk_score >= 40:
            level = "中風險熱點"
        else:
            level = "低風險熱點"
        segments.append(
            {
                "rank": index,
                "lat": float(row.segment_lat),
                "lon": float(row.segment_lon),
                "accident_count": count,
                "district": row.top_district,
                "peak_hour": row.peak_hour,
                "top_weather": row.top_weather,
                "risk_score": risk_score,
                "risk_level": level,
            }
        )

    summary = (
        f"已依 GPS 約略座標聚合出前 {len(segments)} 個路口/路段熱點。"
        "此為原型用空間聚合，尚未套疊正式道路中心線。"
    )
    return {"segments": segments, "summary": summary}


def export_markdown_report(
    *,
    title: str,
    query: dict[str, Any] | None = None,
    pipeline_result: dict[str, Any] | None = None,
    route_result: dict[str, Any] | None = None,
    weather_result: dict[str, Any] | None = None,
    segment_result: dict[str, Any] | None = None,
) -> str:
    """Build a Markdown report from the current dashboard state."""
    lines = [
        f"# {title}",
        "",
        f"產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 查詢條件",
    ]
    query = query or {}
    if query:
        for key, value in query.items():
            if value not in (None, "", "不指定"):
                lines.append(f"- {key}: {value}")
    else:
        lines.append("- 未指定")

    if pipeline_result:
        lines.extend(["", "## 智慧問答結果", ""])
        lines.append(pipeline_result.get("response", "目前沒有智慧問答結果。"))

    if weather_result:
        lines.extend(["", "## 即時天氣", ""])
        lines.append(weather_result.get("summary", "目前沒有即時天氣摘要。"))

    if route_result:
        lines.extend(["", "## 路線參考", ""])
        lines.append(route_result.get("summary", "目前沒有路線摘要。"))

    if segment_result:
        lines.extend(["", "## 路口/路段熱點", ""])
        lines.append(segment_result.get("summary", "目前沒有路段熱點摘要。"))
        segments = segment_result.get("segments") or []
        if segments:
            lines.extend(["", "| 排名 | 行政區 | 約略座標 | 事故數 | 高峰時段 | 風險 |", "|---|---|---|---|---|---|"])
            for item in segments[:10]:
                lines.append(
                    "| {rank} | {district} | {lat}, {lon} | {count} | {hour} | {level} |".format(
                        rank=item.get("rank"),
                        district=item.get("district", ""),
                        lat=item.get("lat"),
                        lon=item.get("lon"),
                        count=item.get("accident_count"),
                        hour=item.get("peak_hour", ""),
                        level=item.get("risk_level", ""),
                    )
                )

    lines.extend(
        [
            "",
            "## 資料限制",
            "- 本報告主要依據歷史事故資料、可用即時 API 與系統工具輸出產生。",
            "- 若未設定中央氣象署 API key 或 OSRM 服務無法連線，相關欄位會以本機歷史資料或 fallback 說明呈現。",
            "- 路口/路段熱點為 GPS 約略座標聚合，尚未等同正式道路工程路段分析。",
        ]
    )
    return "\n".join(lines)


def _mode_or_empty(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return ""
    return values.mode().iloc[0]
