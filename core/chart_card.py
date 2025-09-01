import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from core.style_prefs import get_style_prefs, apply_user_style

from services import (
    slopes_snapshot,
    shape_flags,
    top_growth_codes,
    forecast_linear_band,
    forecast_holt_linear,
    band_from_moving_stats,
    detect_linear_anomalies,
)
from core.plot_utils import add_latest_labels_no_overlap

UNIT_SCALE = {"円": 1, "千円": 1_000, "百万円": 1_000_000}


def marker_step(dates, target_points=24):
    n = len(pd.unique(dates))
    return max(1, round(n / target_points))


def _ensure_css():
    if st.session_state.get("_chart_css_injected"):
        return
    st.markdown(
        """
<style>
.chart-card { position: relative; margin:.25rem 0 1rem; border-radius:12px;
  border:1px solid var(--color-primary); background:var(--card-bg,#fff); }
.chart-toolbar { position: sticky; top: -1px; z-index: 5;
  display:flex; gap:.6rem; flex-wrap:wrap; align-items:center;
  padding:.35rem .6rem; background: linear-gradient(180deg, rgba(0,58,112,.08), rgba(0,58,112,.02));
  border-bottom:1px solid var(--color-primary); }
.chart-toolbar .stRadio, .chart-toolbar .stSelectbox, .chart-toolbar .stSlider,
.chart-toolbar .stMultiSelect, .chart-toolbar .stCheckbox { margin-bottom:0 !important; }
.chart-toolbar .stRadio > label, .chart-toolbar .stCheckbox > label { color:#003a70; }
.chart-toolbar .stSlider label { color:#003a70; }
.chart-body { padding:.15rem .4rem .4rem; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.session_state["_chart_css_injected"] = True


def toolbar_sku_detail(multi_mode: bool):
    _ensure_css()
    ui = st.session_state.setdefault("ui", {})

    a, b, c, d, e = st.columns([1.1, 1.6, 1.1, 1.0, 0.9])
    with a:
        period_opts = ["12ヶ月", "24ヶ月", "36ヶ月"]
        period = st.radio(
            "期間",
            period_opts,
            horizontal=True,
            index=period_opts.index(ui.get("period", "24ヶ月")),
        )
        ui["period"] = period
    with b:
        node_opts = ["自動", "主要ノードのみ", "すべて", "非表示"]
        node_mode = st.radio(
            "ノード表示",
            node_opts,
            horizontal=True,
            index=node_opts.index(ui.get("node_mode", "自動")),
        )
        ui["node_mode"] = node_mode
    with c:
        hover_opts = ["個別", "同月まとめ"]
        hover_mode = st.radio(
            "ホバー",
            hover_opts,
            horizontal=True,
            index=hover_opts.index(ui.get("hover_mode", "個別")),
        )
        ui["hover_mode"] = hover_mode
    with d:
        op_opts = ["パン", "ズーム", "選択"]
        op_mode = st.radio(
            "操作",
            op_opts,
            horizontal=True,
            index=op_opts.index(ui.get("op_mode", "パン")),
        )
        ui["op_mode"] = op_mode
    with e:
        peak_on = st.checkbox("ピーク表示", value=ui.get("peak_on", False))
        ui["peak_on"] = peak_on

    f, g, h, i = st.columns([1.0, 1.5, 1.4, 1.4])
    with f:
        unit_opts = ["円", "千円", "百万円"]
        unit = st.radio(
            "単位",
            unit_opts,
            horizontal=True,
            index=unit_opts.index(ui.get("unit", "千円")),
        )
        ui["unit"] = unit
    with g:
        enable_avoid = st.checkbox("ラベル衝突回避", value=ui.get("enable_avoid", True))
        ui["enable_avoid"] = enable_avoid
        gap_px = st.slider("ラベル最小間隔(px)", 8, 24, ui.get("gap_px", 12))
        ui["gap_px"] = gap_px
    with h:
        max_labels = st.slider("ラベル最大件数", 5, 20, ui.get("max_labels", 12))
        ui["max_labels"] = max_labels
    with i:
        alt_side = st.checkbox("ラベル左右交互配置", value=ui.get("alt_side", True))
        ui["alt_side"] = alt_side

    slope_conf = None
    if multi_mode:
        j, k, l, m = st.columns([1.2, 1.6, 1.2, 1.6])
        with j:
            quick_opts = ["なし", "Top5", "Top10", "最新YoY上位", "直近6M伸長上位"]
            quick = st.radio(
                "クイック絞り込み",
                quick_opts,
                horizontal=True,
                index=quick_opts.index(ui.get("quick", "なし")),
            )
            ui["quick"] = quick
        with k:
            n_win = st.slider(
                "傾きウィンドウ（月）",
                0,
                12,
                ui.get("n_win", 6),
                1,
                help="0=自動（系列の全期間で判定）",
            )
            ui["n_win"] = n_win
            cmp_opts = ["以上", "未満"]
            cmp_mode = st.radio(
                "傾き条件",
                cmp_opts,
                horizontal=True,
                index=cmp_opts.index(ui.get("cmp_mode", "以上")),
            )
            ui["cmp_mode"] = cmp_mode
        with l:
            thr_opts = ["円/月", "%/月", "zスコア"]
            thr_type = st.radio(
                "しきい値の種類",
                thr_opts,
                horizontal=True,
                index=thr_opts.index(ui.get("thr_type", "円/月")),
            )
            ui["thr_type"] = thr_type
        with m:
            thr_val = st.number_input("しきい値", value=float(ui.get("thr_val", 100000.0)), step=10000.0)
            ui["thr_val"] = float(thr_val)
        s1, s2 = st.columns([1.2, 1.2])
        with s1:
            shape_opts = ["（なし）", "急勾配", "山（への字）", "谷（逆への字）"]
            shape_pick = st.radio(
                "形状抽出",
                shape_opts,
                horizontal=True,
                index=shape_opts.index(ui.get("shape_pick", "（なし）")),
            )
            ui["shape_pick"] = shape_pick
        with s2:
            sens = st.slider("形状抽出の感度", 0.0, 1.0, ui.get("sens", 0.5), 0.05)
            ui["sens"] = sens
        slope_conf = dict(
            n_win=n_win,
            cmp_mode=cmp_mode,
            thr_type=thr_type,
            thr_val=float(thr_val),
            shape_pick=shape_pick,
            sens=sens,
            quick=quick,
        )
    p1, p2, p3, p4, p5 = st.columns([1.4, 0.9, 0.9, 1.2, 0.9])
    with p1:
        method_opts = ["なし", "ローカル線形±kσ", "Holt線形", "移動平均±kσ", "移動平均±MAD"]
        f_method = st.selectbox("予測帯", method_opts, index=method_opts.index(ui.get("f_method", "なし")))
        ui["f_method"] = f_method
    with p2:
        f_win = st.selectbox("学習窓幅", [6, 9, 12], index=[6, 9, 12].index(ui.get("f_win", 12)))
        ui["f_win"] = f_win
    with p3:
        f_h = st.selectbox("先の予測ステップ", [3, 6, 12], index=[3, 6, 12].index(ui.get("f_h", 6)))
        ui["f_h"] = f_h
    with p4:
        f_k = st.slider("バンド幅k", 1.5, 3.0, ui.get("f_k", 2.0), 0.1)
        ui["f_k"] = f_k
    with p5:
        f_robust = st.checkbox("ロバスト(MAD)", value=ui.get("f_robust", False))
        ui["f_robust"] = f_robust
    anom_opts = ["OFF", "z≥2.5", "MAD≥3.5"]
    anomaly = st.selectbox("異常検知", anom_opts, index=anom_opts.index(ui.get("anomaly", "OFF")))
    ui["anomaly"] = anomaly
    st.session_state["ui"] = ui
    return dict(
        period=period,
        node_mode=node_mode,
        hover_mode=hover_mode,
        op_mode=op_mode,
        peak_on=peak_on,
        unit=unit,
        enable_avoid=enable_avoid,
        gap_px=gap_px,
        max_labels=max_labels,
        alt_side=alt_side,
        slope_conf=slope_conf,
        forecast_method=ui.get("f_method", "なし"),
        forecast_window=ui.get("f_win", 12),
        forecast_horizon=ui.get("f_h", 6),
        forecast_k=ui.get("f_k", 2.0),
        forecast_robust=ui.get("f_robust", False),
        anomaly=ui.get("anomaly", "OFF"),
    )


def build_chart_card(df_long, selected_codes, multi_mode, tb, band_range=None):
    months = {"12ヶ月": 12, "24ヶ月": 24, "36ヶ月": 36}[tb["period"]]
    dfp = df_long.sort_values("month").groupby("product_code").tail(months)
    if selected_codes:
        dfp = dfp[dfp["product_code"].isin(selected_codes)].copy()

    scale = UNIT_SCALE[tb["unit"]]
    dfp["year_sum_disp"] = dfp["year_sum"] / scale

    if multi_mode and tb.get("slope_conf"):
        sc = tb["slope_conf"]
        snap = slopes_snapshot(dfp, n=sc["n_win"])
        key = {"円/月": "slope_yen", "%/月": "slope_ratio", "zスコア": "slope_z"}[sc["thr_type"]]
        mask = (snap[key] >= sc["thr_val"]) if sc["cmp_mode"] == "以上" else (snap[key] <= sc["thr_val"])
        codes_by_slope = set(snap.loc[mask, "product_code"])
        if sc.get("quick") and sc["quick"] != "なし":
            snapshot = dfp.sort_values("month").groupby("product_code").tail(1)
            if sc["quick"] == "Top5":
                quick_codes = snapshot.nlargest(5, "year_sum")["product_code"]
            elif sc["quick"] == "Top10":
                quick_codes = snapshot.nlargest(10, "year_sum")["product_code"]
            elif sc["quick"] == "最新YoY上位":
                quick_codes = snapshot.dropna(subset=["yoy"]).sort_values("yoy", ascending=False).head(10)["product_code"]
            elif sc["quick"] == "直近6M伸長上位":
                quick_codes = top_growth_codes(dfp, dfp["month"].max(), window=6, top=10)
            else:
                quick_codes = snapshot["product_code"]
            codes_by_slope = codes_by_slope & set(quick_codes)
        if sc["shape_pick"] != "（なし）":
            eff_n = sc["n_win"] if sc["n_win"] > 0 else 12
            sh = shape_flags(
                dfp,
                window=max(6, eff_n * 2),
                alpha_ratio=0.02 * (1.0 - sc["sens"]),
                amp_ratio=0.06 * (1.0 - sc["sens"]),
            )
            pick_map = {
                "急勾配": snap.loc[snap["slope_z"].abs() >= 1.5, "product_code"],
                "山（への字）": sh.loc[sh["is_mountain"], "product_code"],
                "谷（逆への字）": sh.loc[sh["is_valley"], "product_code"],
            }
            pick = pick_map.get(sc["shape_pick"])
            dfp = dfp[dfp["product_code"].isin(set(pick).intersection(codes_by_slope))]
        else:
            dfp = dfp[dfp["product_code"].isin(codes_by_slope)]

    fig = px.line(dfp, x="month", y="year_sum_disp", color="display_name", custom_data=["display_name"])
    fig.update_yaxes(title_text=f"売上 年計（{tb['unit']}）", tickformat="~,d")
    fig.update_traces(
        mode="lines+markers",
        hovertemplate="<b>%{customdata[0]}</b><br>月：%{x|%Y-%m}<br>年計：%{y:,.0f} {tb['unit']}<extra></extra>",
    )
    if band_range:
        low, high = band_range
        fig.add_hrect(y0=low / scale, y1=high / scale, fillcolor="green", opacity=0.12, line_width=0)

    fig.update_layout(
        dragmode={"パン": "pan", "ズーム": "zoom", "選択": "select"}[tb["op_mode"]],
        hovermode="closest" if tb["hover_mode"] == "個別" else "x unified",
    )

    if tb.get("forecast_method") and tb["forecast_method"] != "なし":
        method = tb["forecast_method"]
        win = tb.get("forecast_window", 12)
        horizon = tb.get("forecast_horizon", 6)
        k = tb.get("forecast_k", 2.0)
        robust = tb.get("forecast_robust", False)
        for name, d in dfp.groupby("display_name"):
            s = d.sort_values("month").set_index("month")["year_sum"]
            if method == "ローカル線形±kσ":
                f, lo, hi = forecast_linear_band(s, window=win, horizon=horizon, k=k, robust=robust)
            elif method == "Holt線形":
                f = forecast_holt_linear(s, horizon=horizon)
                f2, lo, hi = forecast_linear_band(s, window=win, horizon=horizon, k=k, robust=robust)
                lo, hi = f - (f2 - lo), f + (hi - f2)
            elif method == "移動平均±kσ":
                f, lo, hi = band_from_moving_stats(s, window=win, horizon=horizon, k=k, robust=False)
            else:
                f, lo, hi = band_from_moving_stats(s, window=win, horizon=horizon, k=k, robust=True)
            if len(f) == 0:
                continue
            last_t = pd.to_datetime(d["month"].max())
            future_idx = pd.period_range(last_t.to_period("M"), periods=horizon, freq="M").to_timestamp() + pd.offsets.MonthBegin(1)
            fig.add_scatter(x=future_idx, y=f/scale, mode="lines", name=f"{name}予測", line=dict(dash="dash"), showlegend=False)
            fig.add_scatter(x=future_idx, y=hi/scale, mode="lines", line=dict(width=0), showlegend=False)
            fig.add_scatter(x=future_idx, y=lo/scale, mode="lines", fill="tonexty", line=dict(width=0), fillcolor="rgba(113,178,255,.18)", showlegend=False)

    if tb.get("anomaly") and tb["anomaly"] != "OFF":
        robust = tb["anomaly"].startswith("MAD")
        thr = 3.5 if robust else 2.5
        for name, d in dfp.groupby("display_name"):
            s = d.sort_values("month").set_index("month")["year_sum"]
            res = detect_linear_anomalies(s, window=tb.get("forecast_window", 12), threshold=thr, robust=robust)
            if res.empty:
                continue
            fig.add_scatter(
                x=pd.to_datetime(res["month"]),
                y=res["value"] / scale,
                mode="markers",
                name=f"{name}異常",
                marker=dict(symbol="triangle-up", color="red", size=10),
                showlegend=False,
                customdata=np.stack([res["score"]], axis=-1),
                hovertemplate=f"<b>{name}</b><br>月：%{{x|%Y-%m}}<br>値：%{{y:,.0f}} {tb['unit']}<br>スコア：%{{customdata[0]:.2f}}<extra></extra>",
            )

    theme_is_dark = st.get_option("theme.base") == "dark"
    halo = "#ffffff" if theme_is_dark else "#222222"
    if tb["node_mode"] == "自動":
        step = marker_step(dfp["month"])
        df_nodes = (
            dfp.sort_values("month")
            .assign(_idx=dfp.sort_values("month").groupby("display_name").cumcount())
            .query("(_idx % @step) == 0")
        )
    elif tb["node_mode"] == "主要ノードのみ":
        g = dfp.sort_values("month").groupby("display_name")
        latest = g.tail(1)
        idxmax = dfp.loc[g["year_sum"].idxmax().dropna()]
        idxmin = dfp.loc[g["year_sum"].idxmin().dropna()]
        ystart = g.head(1)
        df_nodes = pd.concat([latest, idxmax, idxmin, ystart]).drop_duplicates(["display_name", "month"])
    elif tb["node_mode"] == "すべて":
        df_nodes = dfp.copy()
    else:
        df_nodes = dfp.iloc[0:0].copy()

    for name, d in df_nodes.groupby("display_name"):
        fig.add_scatter(
            x=d["month"],
            y=d["year_sum_disp"],
            mode="markers",
            name=name,
            legendgroup=name,
            showlegend=False,
            marker=dict(size=6, symbol="circle", line=dict(color=halo, width=2), opacity=0.95),
            customdata=np.stack([d["display_name"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>月：%{x|%Y-%m}<br>年計：%{y:,.0f} {tb['unit']}<extra></extra>",
        )

    if tb["enable_avoid"]:
        add_latest_labels_no_overlap(
            fig,
            dfp,
            label_col="display_name",
            x_col="month",
            y_col="year_sum_disp",
            max_labels=tb["max_labels"],
            min_gap_px=tb["gap_px"],
            alternate_side=tb["alt_side"],
        )
    prefs = get_style_prefs(st.session_state.get("current_page", "default"))
    fig = apply_user_style(fig, prefs)
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True, "doubleClick": "reset"},
    )
    return fig
