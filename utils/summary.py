"""Helpers for converting numeric indicators into natural language."""

from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd


def _format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "―"
    return f"{value * 100:.1f}%"


def _format_currency(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "―"
    return f"{value:,.0f}円"


def generate_summary(
    metrics: Mapping[str, float],
    timeframe: str,
    highlights: Optional[pd.DataFrame] = None,
    anomalies: Optional[pd.DataFrame] = None,
) -> str:
    """Create a narrative summary for the analysed dataset.

    Parameters
    ----------
    metrics:
        Dictionary containing KPI values such as ``{"売上": 123}`` or
        ``{"yoy": 0.05}``.
    timeframe:
        Text label describing the period the metrics correspond to.
    highlights:
        Optional dataframe with notable rows (e.g. top products).  The
        first three rows are mentioned in the summary.
    anomalies:
        Optional dataframe with detected anomalies.  The summary notes
        the number of anomalies and the first entry if provided.
    """

    yoy = metrics.get("yoy")
    delta = metrics.get("delta")
    revenue = metrics.get("total") or metrics.get("売上") or metrics.get("revenue")
    slope = metrics.get("slope")

    lines = [f"{timeframe}の年計売上は{_format_currency(revenue)}です。"]
    if yoy is not None:
        direction = "増加" if yoy >= 0 else "減少"
        lines.append(f"前年同月比は{_format_percent(yoy)}で{direction}しました。")
    if delta is not None:
        trend = "プラス" if delta >= 0 else "マイナス"
        lines.append(f"前月比は{delta:,.0f}円で{trend}寄与です。")
    if slope is not None:
        slope_text = "上昇傾向" if slope >= 0 else "下降傾向"
        lines.append(f"直近トレンドの傾きは{slope:.2f}で{slope_text}です。")

    if highlights is not None and not highlights.empty:
        head = highlights.head(3)
        bullets = []
        for _, row in head.iterrows():
            label = row.get("product_name") or row.get("product_code") or "不明"
            value = row.get("amount")
            if value is None or pd.isna(value):
                value = row.get("year_sum", 0)
            bullets.append(f"{label}: {float(value):,.0f}")
        lines.append("注目SKUは" + "、".join(bullets) + "。")

    if anomalies is not None and not anomalies.empty:
        total = len(anomalies)
        first = anomalies.iloc[0]
        name = first.get("product_name") or first.get("product_code") or "対象"
        lines.append(f"異常検知では{total}件のシグナルを確認。代表例は{name}です。")

    summary = "".join(lines)

    # Try to upgrade the summary using the optional LLM helper when
    # available.  The import is deferred to keep module import time fast
    # and avoid hard dependency on the transformers stack.
    try:  # pragma: no cover - optional dependency path
        from ai_features import explain_analysis

        ai_text = explain_analysis(dict(metrics))
        if timeframe:
            summary = f"{timeframe}｜{ai_text}"
        else:
            summary = ai_text
    except Exception:
        pass
    return summary
