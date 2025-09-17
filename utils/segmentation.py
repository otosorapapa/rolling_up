"""Segmentation and descriptive analytics helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class RFMResult:
    """Container for RFM scores to ease testing."""

    dataframe: pd.DataFrame
    quantiles: pd.DataFrame


def rfm_analysis(
    df: pd.DataFrame,
    customer_col: str,
    date_col: str,
    amount_col: str,
    reference_date: Optional[pd.Timestamp] = None,
) -> RFMResult:
    """Compute Recency, Frequency, Monetary scores for the dataset."""

    if df.empty:
        empty = pd.DataFrame(columns=["R", "F", "M", "R_score", "F_score", "M_score", "RFM"])
        return RFMResult(empty, pd.DataFrame())

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    reference_date = reference_date or data[date_col].max() + pd.Timedelta(days=1)

    grouped = data.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (reference_date - x.max()).days),
        Frequency=(date_col, "count"),
        Monetary=(amount_col, "sum"),
    )
    quantiles = grouped.quantile(q=[0.25, 0.5, 0.75])

    def score_column(col: str, reverse: bool = False) -> pd.Series:
        bins = quantiles[col]
        if reverse:
            return grouped[col].apply(
                lambda x: 4 - min(np.searchsorted(bins.values, x, side="right"), 3)
            )
        return grouped[col].apply(
            lambda x: min(np.searchsorted(bins.values, x, side="right"), 3) + 1
        )

    grouped["R_score"] = score_column("Recency", reverse=True)
    grouped["F_score"] = score_column("Frequency")
    grouped["M_score"] = score_column("Monetary")
    grouped["RFM"] = grouped["R_score"].astype(str) + grouped["F_score"].astype(str) + grouped["M_score"].astype(str)

    grouped = grouped.reset_index()
    grouped["R_score"] = grouped["R_score"].astype(int)
    grouped["F_score"] = grouped["F_score"].astype(int)
    grouped["M_score"] = grouped["M_score"].astype(int)
    return RFMResult(grouped, quantiles)


def abc_analysis(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    """Perform ABC classification by cumulative contribution."""

    agg = df.groupby(group_col, as_index=False)[value_col].sum()
    agg = agg.sort_values(value_col, ascending=False)
    total = agg[value_col].sum()
    if total == 0:
        agg["class"] = "C"
        return agg
    agg["share"] = agg[value_col] / total
    agg["cum_share"] = agg["share"].cumsum()
    agg["class"] = np.where(
        agg["cum_share"] <= 0.8,
        "A",
        np.where(agg["cum_share"] <= 0.95, "B", "C"),
    )
    return agg


def correlation_heatmap(df: pd.DataFrame, columns: Iterable[str]) -> go.Figure:
    """Create a correlation heatmap for the provided columns."""

    subset = df[list(columns)].dropna()
    corr = subset.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=30, r=30, t=20, b=20))
    return fig


def scatter_matrix(df: pd.DataFrame, columns: Iterable[str]) -> go.Figure:
    """Return a responsive scatter matrix figure."""

    fig = px.scatter_matrix(df, dimensions=list(columns))
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(height=500)
    return fig
