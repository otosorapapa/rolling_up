import json
from typing import Dict

import streamlit as st

DEFAULT_PALETTES = {
    "Default": None,  # Plotly既定
    "Contrast": ["#1f77b4","#d62728","#2ca02c","#ff7f0e","#9467bd","#17becf","#7f7f7f","#bcbd22"],
    "Pastel":   ["#8ecae6","#ffb5a7","#c1d3fe","#bde0fe","#caffbf","#ffd6a5","#d0bdf4","#ffc8dd"],
    "Colorblind": ["#0072B2","#D55E00","#009E73","#CC79A7","#F0E442","#56B4E9","#E69F00","#000000"],
}


def get_style_prefs(page_key: str) -> Dict:
    """グローバル or ページ専用のスタイル設定を取得"""
    pg = st.session_state.get("style_prefs_page", {}).get(page_key, {})
    gl = st.session_state.get("style_prefs", {})
    return {**gl, **pg}  # ページが優先


def apply_user_style(fig, prefs: Dict):
    """ユーザー指定のスタイルを最終適用（全グラフ共通）"""
    if not prefs:
        return fig

    paper = prefs.get("paper_bgcolor")
    plot = prefs.get("plot_bgcolor")
    textc = prefs.get("text_color")
    gridc = prefs.get("grid_color")
    accent = prefs.get("accent")
    pal = prefs.get("palette")
    if pal in DEFAULT_PALETTES and DEFAULT_PALETTES[pal]:
        fig.update_layout(colorway=DEFAULT_PALETTES[pal])
    fig.update_layout(
        paper_bgcolor=paper or fig.layout.paper_bgcolor,
        plot_bgcolor=plot or fig.layout.plot_bgcolor,
        font=dict(color=textc) if textc else fig.layout.font,
        legend=(
            dict(y=1, x=1.02, yanchor="top", xanchor="left")
            if prefs.get("legend_pos", "右") == "右"
            else (
                dict(orientation="h", y=1.08, x=0, yanchor="bottom")
                if prefs.get("legend_pos") == "上"
                else (
                    dict(orientation="h", y=-0.2, x=0)
                    if prefs.get("legend_pos") == "下"
                    else dict(y=1, x=-0.02, yanchor="top", xanchor="right")
                )
            )
        ),
    )
    # 軸・グリッド
    show_grid = prefs.get("show_grid", True)
    fig.update_xaxes(
        showgrid=show_grid,
        gridcolor=gridc or "rgba(255,255,255,.08)",
        linecolor="rgba(255,255,255,.25)" if prefs.get("bold_axis") else None,
    )
    fig.update_yaxes(
        showgrid=show_grid,
        gridcolor=gridc or "rgba(255,255,255,.08)",
        linecolor="rgba(255,255,255,.25)" if prefs.get("bold_axis") else None,
    )

    # 線・ノード
    lw = float(prefs.get("line_width", 2.2))
    dash_map = {"実線": None, "点線": "dot", "破線": "dash", "点破線": "dashdot"}
    dash = dash_map.get(prefs.get("line_dash", "実線"))
    ms = int(prefs.get("marker_size", 6))
    if ms <= 0:
        fig.update_traces(mode="lines")
    else:
        fig.update_traces(mode="lines+markers")
    fig.update_traces(line=dict(width=lw, dash=dash), selector=dict(mode="lines"))
    marker_line = dict(
        width=float(prefs.get("marker_edge_w", 1)),
        color=prefs.get("marker_edge_c", "#FFFFFF"),
    )
    fig.update_traces(
        selector=lambda t: "markers" in (t.mode or ""),
        marker=dict(size=ms, line=marker_line, symbol=prefs.get("marker_symbol", "circle")),
    )

    # 予測帯などの塗り透明度
    fill_alpha = float(prefs.get("band_alpha", 0.18))
    if accent and accent.startswith("rgba"):
        for tr in fig.data:
            if getattr(tr, "fill", None) in ("tonexty", "tozeroy"):
                fig.update_traces(
                    selector=lambda t, uid=tr.uid: t.uid == uid,
                    fillcolor=accent.replace(")", f",{fill_alpha})"),
                )

    # 系列色の手動指定
    manual = prefs.get("series_colors", {})
    if manual:
        for tr in fig.data:
            name = (tr.name or "").split("：")[0]
            if name in manual and manual[name]:
                fig.update_traces(
                    selector=lambda t, uid=tr.uid: t.uid == uid,
                    line=dict(color=manual[name]),
                    marker=dict(color=manual[name]),
                )
    return fig
