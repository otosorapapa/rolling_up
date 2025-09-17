import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def apply_elegant_theme(fig: go.Figure, theme: str = "dark") -> go.Figure:
    """Apply subdued, elegant styling to Plotly figures when enabled."""
    if not st.session_state.get("elegant_on", True):
        return fig
    if theme == "dark":
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0F1117",
            plot_bgcolor="#0F1117",
            font=dict(
                family="Noto Sans JP, Meiryo, Arial",
                size=12,
                color="#E9F1FF",
            ),
            legend=dict(
                bgcolor="rgba(17,22,29,.70)",
                bordercolor="rgba(255,255,255,.14)",
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="rgba(28,39,51,0.92)",
                bordercolor="rgba(233,241,255,0.18)",
                font=dict(color="#E9F1FF"),
            ),
        )
        grid = "#2A3240"
        axisline = "rgba(255,255,255,.28)"
        marker_border = "rgba(233,241,255,0.85)"
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(
                family="Noto Sans JP, Meiryo, Arial",
                size=12,
                color="#0B1324",
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,.85)",
                bordercolor="rgba(11,19,36,.14)",
                borderwidth=1,
            ),
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.98)",
                bordercolor="rgba(11,19,36,0.12)",
                font=dict(color="#0B1324"),
            ),
        )
        grid = "rgba(11,19,36,.10)"
        axisline = "rgba(11,19,36,.30)"
        marker_border = "rgba(11,19,36,0.24)"
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        linecolor=axisline,
        ticks="outside",
        ticklen=4,
        tickcolor=axisline,
        showline=True,
        linewidth=1,
        title_standoff=14,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        linecolor=axisline,
        ticks="outside",
        ticklen=4,
        tickcolor=axisline,
        showline=True,
        linewidth=1,
        title_standoff=16,
    )
    fig.update_traces(
        selector=lambda t: "markers" in getattr(t, "mode", ""),
        marker=dict(size=6, line=dict(width=1.2, color=marker_border)),
    )
    return fig


def _plot_area_height(fig: go.Figure) -> int:
    h = fig.layout.height or 520
    m = fig.layout.margin or {}
    t = getattr(m, "t", 40) or 40
    b = getattr(m, "b", 60) or 60
    return max(120, int(h - t - b))


def _y_to_px(y, y0, y1, plot_h):
    if y1 == y0:
        y1 = y0 + 1.0
    return float((1 - (y - y0) / (y1 - y0)) * plot_h)


def add_latest_labels_no_overlap(
    fig: go.Figure,
    df_long: pd.DataFrame,
    label_col: str = "display_name",
    x_col: str = "month",
    y_col: str = "year_sum",
    max_labels: int = 12,
    min_gap_px: int = 12,
    alternate_side: bool = True,
    xpad_px: int = 8,
    font_size: int = 11,
):
    last = df_long.sort_values(x_col).groupby(label_col, as_index=False).tail(1)
    if len(last) == 0:
        return
    cand = last.sort_values(y_col, ascending=False).head(max_labels).copy()
    yaxis = fig.layout.yaxis
    if getattr(yaxis, "range", None):
        y0, y1 = yaxis.range
    else:
        y0, y1 = float(df_long[y_col].min()), float(df_long[y_col].max())
    plot_h = _plot_area_height(fig)
    cand["y_px"] = cand[y_col].apply(lambda v: _y_to_px(v, y0, y1, plot_h))
    cand = cand.sort_values("y_px")
    placed = []
    for _, r in cand.iterrows():
        base = r["y_px"]
        if placed and base <= placed[-1] + min_gap_px:
            base = placed[-1] + min_gap_px
        base = float(np.clip(base, 0 + 6, plot_h - 6))
        placed.append(base)
        yshift = -(base - r["y_px"])
        xshift = xpad_px if (not alternate_side or (len(placed) % 2 == 1)) else -xpad_px
        fig.add_annotation(
            x=r[x_col],
            y=r[y_col],
            text=f"{r[label_col]}：{r[y_col]:,.0f}（{pd.to_datetime(r[x_col]).strftime('%Y-%m')}）",
            showarrow=False,
            xanchor="left" if xshift >= 0 else "right",
            yanchor="middle",
            xshift=xshift,
            yshift=yshift,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=font_size),
        )
