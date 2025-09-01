import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import json


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


DEFAULT_STYLE = {
    "theme": "light",
    "bgcolor": "#ffffff",
    "line_color": "#003a70",
    "marker_color": "#003a70",
    "legend": True,
    "grid": True,
}


def styled_plotly_chart(
    fig: go.Figure,
    *,
    key: str,
    config: dict | None = None,
    **kwargs,
):
    """Render Plotly figure with an attached style panel."""

    global_style = st.session_state.setdefault("global_style", DEFAULT_STYLE.copy())
    style_map = st.session_state.setdefault("plot_styles", {})
    style = style_map.get(key, global_style.copy())

    col_ctrl, col_chart = st.columns([0.06, 0.94])
    with col_ctrl:
        with st.popover("🎛", key=f"{key}_popover"):
            theme_idx = 1 if style.get("theme") == "dark" else 0
            style["theme"] = st.radio("テーマ", ["light", "dark"], index=theme_idx, key=f"{key}_theme")
            style["bgcolor"] = st.color_picker(
                "背景色", value=style.get("bgcolor", "#ffffff"), key=f"{key}_bg"
            )
            style["line_color"] = st.color_picker(
                "線色", value=style.get("line_color", "#003a70"), key=f"{key}_line"
            )
            style["marker_color"] = st.color_picker(
                "ノード色", value=style.get("marker_color", "#003a70"), key=f"{key}_marker"
            )
            style["legend"] = st.checkbox(
                "凡例", value=style.get("legend", True), key=f"{key}_legend"
            )
            style["grid"] = st.checkbox(
                "グリッド", value=style.get("grid", True), key=f"{key}_grid"
            )
            st.download_button(
                "JSON保存",
                data=json.dumps(style, ensure_ascii=False).encode("utf-8"),
                file_name=f"{key}_style.json",
                mime="application/json",
                key=f"{key}_save",
            )
            uploaded = st.file_uploader("JSON読込", type="json", key=f"{key}_load")
            if uploaded is not None:
                try:
                    loaded = json.load(uploaded)
                    for k in DEFAULT_STYLE:
                        if k in loaded:
                            style[k] = loaded[k]
                except Exception:
                    st.warning("JSONの読み込みに失敗しました")

    style_map[key] = style
    st.session_state.plot_styles = style_map

    template = "plotly_dark" if style["theme"] == "dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor=style["bgcolor"],
        plot_bgcolor=style["bgcolor"],
        showlegend=style["legend"],
    )
    fig.update_xaxes(showgrid=style["grid"])
    fig.update_yaxes(showgrid=style["grid"])
    fig.update_traces(line_color=style["line_color"])
    fig.update_traces(marker=dict(color=style["marker_color"]))

    with col_chart:
        st.plotly_chart(fig, config=config, **kwargs)
