
import io
import json
import math
from datetime import datetime
from typing import Optional, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_CONFIG = {
    "locale": "ja",
    "displaylogo": False,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "autoScale2d",
        "resetViewMapbox",
        "toggleSpikelines",
        "select2d",
        "lasso2d",
        "zoom3d",
        "orbitRotation",
        "tableRotation",
    ],
    "toImageButtonOptions": {"format": "png", "filename": "年計比較"},
}

from services import (
    parse_uploaded_table,
    fill_missing_months,
    compute_year_rolling,
    compute_slopes,
    abc_classification,
    compute_hhi,
    build_alerts,
    aggregate_overview,
    build_indexed_series,
    latest_yearsum_snapshot,
    resolve_band,
    filter_products_by_band,
    get_yearly_series,
    top_growth_codes,
    trend_last6,
    slopes_snapshot,
    shape_flags,
)

APP_TITLE = "売上年計（12カ月移動累計）ダッシュボード"
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# ---------------- Session State ----------------
if "data_monthly" not in st.session_state:
    st.session_state.data_monthly = None  # long-form DF
if "data_year" not in st.session_state:
    st.session_state.data_year = None
if "settings" not in st.session_state:
    st.session_state.settings = {
        "window": 12,
        "last_n": 12,
        "missing_policy": "zero_fill",
        "yoy_threshold": -0.10,
        "delta_threshold": -300000.0,
        "slope_threshold": -1.0,
        "currency_unit": "円",
    }
if "notes" not in st.session_state:
    st.session_state.notes = {}  # product_code -> str
if "tags" not in st.session_state:
    st.session_state.tags = {}  # product_code -> List[str]
if "saved_views" not in st.session_state:
    st.session_state.saved_views = {}  # name -> dict
if "compare_params" not in st.session_state:
    st.session_state.compare_params = {}
if "compare_results" not in st.session_state:
    st.session_state.compare_results = None


# ---------------- Helpers ----------------
def require_data():
    if st.session_state.data_year is None or st.session_state.data_monthly is None:
        st.info("データが未取り込みです。左メニュー「データ取込」からアップロードしてください。")
        st.stop()


def month_options(df: pd.DataFrame) -> List[str]:
    return sorted(df["month"].dropna().unique().tolist())


def end_month_selector(df: pd.DataFrame, key="end_month"):
    mopts = month_options(df)
    default = mopts[-1] if mopts else None
    return st.selectbox("終端月（年計の計算対象）", mopts, index=(len(mopts)-1) if mopts else 0, key=key)


def download_excel(df: pd.DataFrame, filename: str) -> bytes:
    import xlsxwriter  # noqa
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return output.getvalue()


def download_pdf_overview(kpi: dict, top_df: pd.DataFrame, filename: str) -> bytes:
    # Minimal PDF using reportlab (text only)
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "年計KPIサマリー")
    y -= 24
    c.setFont("Helvetica", 11)
    for k, v in kpi.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 14
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "TOP10（年計）")
    y -= 18
    c.setFont("Helvetica", 10)
    cols = ["product_code", "product_name", "year_sum"]
    for _, row in top_df[cols].head(10).iterrows():
        c.drawString(40, y, f"{row['product_code']}  {row['product_name']}  {int(row['year_sum']):,}")
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def marker_step(dates, target_points=24):
    n = len(pd.unique(dates))
    return max(1, round(n / target_points))


# ---- Correlation Helpers & Label Overlap Avoidance ----


def fisher_ci(r: float, n: int, zcrit: float = 1.96):
    r = np.clip(r, -0.999999, 0.999999)
    if n <= 3:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def corr_table(df: pd.DataFrame, cols, method: str = "pearson"):
    sub = df[cols].dropna()
    n = len(sub)
    c = sub.corr(method=method)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = c.loc[a, b]
            lo, hi = fisher_ci(r, n)
            sig = "有意(95%)" if (lo > 0 or hi < 0) else "n.s."
            rows.append(
                {
                    "pair": f"{a}×{b}",
                    "r": r,
                    "n": n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "sig": sig,
                }
            )
    return pd.DataFrame(rows).sort_values("r", ascending=False)


def winsorize_frame(df, cols, p: float = 0.01):
    out = df.copy()
    for col in cols:
        x = out[col]
        lo, hi = x.quantile(p), x.quantile(1 - p)
        out[col] = x.clip(lo, hi)
    return out


def maybe_log1p(df, cols, enable: bool):
    if not enable:
        return df
    out = df.copy()
    for col in cols:
        if (out[col] >= 0).all():
            out[col] = np.log1p(out[col])
    return out


def narrate_top_insights(tbl: pd.DataFrame, name_map: dict, k: int = 3):
    pos = tbl[tbl["r"] > 0].nlargest(k, "r")
    neg = tbl[tbl["r"] < 0].nsmallest(k, "r")
    lines = []

    def jp(pair):
        a, b = pair.split("×")
        return f"「{name_map.get(a, a)}」と「{name_map.get(b, b)}」"

    for _, r in pos.iterrows():
        lines.append(
            f"{jp(r['pair'])} は **正の相関** (r={r['r']:.2f}, 95%CI [{r['ci_low']:.2f},{r['ci_high']:.2f}], n={r['n']})。"
        )
    for _, r in neg.iterrows():
        lines.append(
            f"{jp(r['pair'])} は **負の相関** (r={r['r']:.2f}, 95%CI [{r['ci_low']:.2f},{r['ci_high']:.2f}], n={r['n']})。"
        )
    return lines


def fit_line(x, y):
    x = x.values.astype(float)
    y = y.values.astype(float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return m, b, r2


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


NAME_MAP = {
    "year_sum": "年計（12ヶ月累計）",
    "yoy": "YoY（前年同月比）",
    "delta": "Δ（前月差）",
    "slope6m": "直近6ヶ月の傾き",
    "std6m": "直近6ヶ月の変動",
    "slope_beta": "直近Nの傾き",
    "hhi_share": "HHI寄与度",
}


# ---------------- Sidebar ----------------
st.sidebar.title(APP_TITLE)
page = st.sidebar.radio(
    "メニュー",
    [
        "ダッシュボード",
        "ランキング",
        "比較ビュー",
        "SKU詳細",
        "相関分析",
        "データ取込",
        "アラート",
        "設定",
        "保存ビュー",
    ],
)

# ---------------- Pages ----------------

# 1) データ取込
if page == "データ取込":
    st.header("データ取込 / マッピング / 品質チェック")

    st.markdown("**Excel(.xlsx) / CSV をアップロードしてください。** "
                "列に `YYYY-MM`（または日付系）形式の月度が含まれている必要があります。")

    col_u1, col_u2 = st.columns([2,1])
    with col_u1:
        file = st.file_uploader("ファイル選択", type=["xlsx","csv"])
    with col_u2:
        st.session_state.settings["missing_policy"] = st.selectbox("欠測月ポリシー",
            options=["zero_fill","mark_missing"],
            format_func=lambda x: "ゼロ補完(推奨)" if x=="zero_fill" else "欠測含む窓は非計上",
            index=0)

    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(file)
            else:
                df_raw = pd.read_excel(file, engine="openpyxl")
        except Exception as e:
            st.error(f"読込エラー: {e}")
            st.stop()

        st.caption("アップロードプレビュー（先頭100行）")
        st.dataframe(df_raw.head(100), use_container_width=True)

        cols = df_raw.columns.tolist()
        product_name_col = st.selectbox("商品名列の選択", options=cols, index=0)
        product_code_col = st.selectbox("商品コード列の選択（任意）", options=["<なし>"] + cols, index=0)
        code_col = None if product_code_col == "<なし>" else product_code_col

        if st.button("変換＆取込", type="primary"):
            try:
                long_df = parse_uploaded_table(df_raw, product_name_col=product_name_col, product_code_col=code_col)
                long_df = fill_missing_months(long_df, policy=st.session_state.settings["missing_policy"])
                # Compute year rolling & slopes
                year_df = compute_year_rolling(long_df, window=st.session_state.settings["window"],
                                               policy=st.session_state.settings["missing_policy"])
                year_df = compute_slopes(year_df, last_n=st.session_state.settings["last_n"])

                st.session_state.data_monthly = long_df
                st.session_state.data_year = year_df
                st.success("取込完了。ダッシュボードへ移動して可視化を確認してください。")

                st.subheader("品質チェック（欠測月/非数値/重複）")
                # 欠測月
                miss_rate = (long_df["is_missing"].sum(), len(long_df))
                st.write(f"- 欠測セル数: {miss_rate[0]:,} / {miss_rate[1]:,}")
                # 月レンジ
                st.write(f"- データ期間: {long_df['month'].min()} 〜 {long_df['month'].max()}")
                # SKU数
                st.write(f"- SKU数: {long_df['product_code'].nunique():,}")
                st.write(f"- レコード数: {len(long_df):,}")

                st.download_button("年計テーブルをCSVでダウンロード", data=st.session_state.data_year.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="year_rolling.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)

# 2) ダッシュボード
elif page == "ダッシュボード":
    require_data()
    st.header("ダッシュボード")

    end_m = end_month_selector(st.session_state.data_year, key="end_month_dash")

    # KPI
    kpi = aggregate_overview(st.session_state.data_year, end_m)
    hhi = compute_hhi(st.session_state.data_year, end_m)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年計総額", f"{int(kpi['total_year_sum']):,} 円" if kpi["total_year_sum"] is not None else "-")
    c2.metric("年計YoY", f"{kpi['yoy']*100:.1f} %" if kpi["yoy"] is not None else "—")
    c3.metric("前月差(Δ)", f"{int(kpi['delta']):,} 円" if kpi["delta"] is not None else "—")
    c4.metric("HHI(集中度)", f"{hhi:.3f}")

    # 総合年計トレンド（全SKU合計）
    totals = st.session_state.data_year.groupby("month", as_index=False)["year_sum"].sum()
    fig = px.line(totals, x="month", y="year_sum", title="総合 年計トレンド", markers=True)
    fig.update_layout(height=350, margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # ランキング（年計）
    snap = st.session_state.data_year[st.session_state.data_year["month"] == end_m].dropna(subset=["year_sum"]).copy()
    snap = snap.sort_values("year_sum", ascending=False)
    st.subheader(f"ランキング（{end_m} 時点 年計）")
    st.dataframe(snap[["product_code","product_name","year_sum","yoy","delta"]].head(20), use_container_width=True)
    st.download_button("この表をCSVでダウンロード", data=snap.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"ranking_{end_m}.csv", mime="text/csv")

    # PDF出力（KPI + TOP10）
    pdf_bytes = download_pdf_overview({"total_year_sum": int(kpi["total_year_sum"]),
                                       "yoy": round(kpi["yoy"],4) if kpi["yoy"] is not None else None,
                                       "delta": int(kpi["delta"]) if kpi["delta"] is not None else None},
                                      snap, filename=f"overview_{end_m}.pdf")
    st.download_button("会議用PDF（KPI+Top10）を出力", data=pdf_bytes, file_name=f"overview_{end_m}.pdf", mime="application/pdf")

# 3) ランキング
elif page == "ランキング":
    require_data()
    st.header("ランキング / ワースト")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_rank")
    metric = st.selectbox("指標", options=["year_sum","yoy","delta","slope_beta"], index=0)
    order = st.radio("並び順", options=["desc","asc"], horizontal=True)
    hide_zero = st.checkbox("年計ゼロを除外", value=True)

    snap = st.session_state.data_year[st.session_state.data_year["month"] == end_m].copy()
    total = len(snap)
    zero_cnt = int((snap["year_sum"] == 0).sum())
    if hide_zero:
        snap = snap[snap["year_sum"] > 0]
    snap = snap.dropna(subset=[metric])
    snap = snap.sort_values(metric, ascending=(order == "asc"))
    st.caption(f"除外 {zero_cnt} 件 / 全 {total} 件")

    fig_bar = px.bar(snap.head(20), x="product_name", y=metric)
    st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    st.dataframe(
        snap[["product_code", "product_name", "year_sum", "yoy", "delta", "slope_beta"]].head(100),
        use_container_width=True,
    )

    st.download_button(
        "CSVダウンロード",
        data=snap.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"ranking_{metric}_{end_m}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Excelダウンロード",
        data=download_excel(snap, f"ranking_{metric}_{end_m}.xlsx"),
        file_name=f"ranking_{metric}_{end_m}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


    # 4) 比較ビュー（マルチ商品バンド）
elif page == "比較ビュー":
    require_data()
    st.header("マルチ商品比較")
    params = st.session_state.compare_params
    year_df = st.session_state.data_year
    end_m = end_month_selector(year_df, key="compare_end_month")

    snapshot = latest_yearsum_snapshot(year_df, end_m)
    snapshot["display_name"] = snapshot["product_name"].fillna(snapshot["product_code"])

    search = st.text_input("検索ボックス", "")
    if search:
        snapshot = snapshot[snapshot["display_name"].str.contains(search, case=False, na=False)]
    # ---- 操作バー＋グラフ密着カード ----

    band_params = params.get("band_params", {})
    max_amount = float(snapshot["year_sum"].max()) if not snapshot.empty else 0.0
    low0 = band_params.get("low_amount", float(snapshot["year_sum"].min()) if not snapshot.empty else 0.0)
    high0 = band_params.get("high_amount", max_amount)

    st.markdown(
        """
<style>
.chart-card { position: relative; margin:.25rem 0 1rem; border-radius:12px;
  border:1px solid rgba(113,178,255,.25); background:var(--background-color,#0f172a); }
.chart-toolbar { position: sticky; top: -1px; z-index: 5;
  display:flex; gap:.6rem; flex-wrap:wrap; align-items:center;
  padding:.35rem .6rem; background: linear-gradient(180deg, rgba(113,178,255,.20), rgba(113,178,255,.10));
  border-bottom:1px solid rgba(113,178,255,.35); }
/* Streamlit標準の下マージンを除去（ここが距離の主因） */
.chart-toolbar .stRadio, .chart-toolbar .stSelectbox, .chart-toolbar .stSlider,
.chart-toolbar .stMultiSelect, .chart-toolbar .stCheckbox { margin-bottom:0 !important; }
.chart-toolbar .stRadio > label, .chart-toolbar .stCheckbox > label { color:#e6f2ff; }
.chart-toolbar .stSlider label { color:#e6f2ff; }
.chart-body { padding:.15rem .4rem .4rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<section class="chart-card" id="line-compare">', unsafe_allow_html=True)

    st.markdown('<div class="chart-toolbar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 1.1, 1.0, 0.9])
    with c1:
        period = st.radio("期間", ["12ヶ月", "24ヶ月", "36ヶ月"], horizontal=True, index=1)
    with c2:
        node_mode = st.radio("ノード表示", ["自動", "主要ノードのみ", "すべて", "非表示"], horizontal=True, index=0)
    with c3:
        hover_mode = st.radio("ホバー", ["個別", "同月まとめ"], horizontal=True, index=0)
    with c4:
        op_mode = st.radio("操作", ["パン", "ズーム", "選択"], horizontal=True, index=0)
    with c5:
        peak_on = st.checkbox("ピーク表示", value=False)

    c6, c7, c8 = st.columns([2.0, 1.9, 1.6])
    with c6:
        band_mode = st.radio(
            "バンド",
            ["金額指定", "商品指定(2)", "パーセンタイル", "順位帯", "ターゲット近傍"],
            horizontal=True,
            index=["金額指定", "商品指定(2)", "パーセンタイル", "順位帯", "ターゲット近傍"].index(
                params.get("band_mode", "金額指定")
            ),
        )
    with c7:
        band_params = params.get("band_params", {})
        if band_mode == "金額指定" and not snapshot.empty:
            low, high = st.slider(
                "金額レンジ",
                min_value=0.0,
                max_value=max_amount,
                value=(
                    band_params.get("low_amount", low0),
                    band_params.get("high_amount", high0),
                ),
                step=max(max_amount / 100, 1.0),
            )
            band_params = {"low_amount": low, "high_amount": high}
        elif band_mode == "商品指定(2)" and not snapshot.empty:
            opts = (
                snapshot["product_code"].fillna("") + " | " + snapshot["display_name"].fillna("")
            ).tolist()
            opts = [o for o in opts if o.strip() != "|"]
            prod_a = st.selectbox("商品A", opts, index=0)
            prod_b = st.selectbox("商品B", opts, index=1 if len(opts) > 1 else 0)
            band_params = {"prod_a": prod_a.split(" | ")[0], "prod_b": prod_b.split(" | ")[0]}
        elif band_mode == "パーセンタイル" and not snapshot.empty:
            p_low, p_high = band_params.get("p_low", 0), band_params.get("p_high", 100)
            p_low, p_high = st.slider("百分位(%)", 0, 100, (int(p_low), int(p_high)))
            band_params = {"p_low": p_low, "p_high": p_high}
        elif band_mode == "順位帯" and not snapshot.empty:
            max_rank = int(snapshot["rank"].max()) if not snapshot.empty else 1
            r_low, r_high = band_params.get("r_low", 1), band_params.get("r_high", max_rank)
            r_low, r_high = st.slider("順位", 1, max_rank, (int(r_low), int(r_high)))
            band_params = {"r_low": r_low, "r_high": r_high}
        else:
            opts = (snapshot["product_code"] + " | " + snapshot["display_name"]).tolist()
            tlabel = st.selectbox("基準商品", opts, index=0) if opts else ""
            tcode = tlabel.split(" | ")[0] if tlabel else ""
            by = st.radio("幅指定", ["金額", "%"], horizontal=True)
            width_default = 100000.0 if by == "金額" else 0.1
            width = st.number_input(
                "幅", value=float(band_params.get("width", width_default)), step=width_default / 10
            )
            band_params = {"target_code": tcode, "by": "amt" if by == "金額" else "pct", "width": width}
    with c8:
        quick = st.radio(
            "クイック絞り込み",
            ["なし", "Top5", "Top10", "最新YoY上位", "直近6M伸長上位"],
            horizontal=True,
            index=0,
        )
    c9, c10, c11, c12 = st.columns([1.2, 1.5, 1.5, 1.5])
    with c9:
        enable_label_avoid = st.checkbox("ラベル衝突回避", value=True)
    with c10:
        label_gap_px = st.slider("ラベル最小間隔(px)", 8, 24, 12)
    with c11:
        label_max = st.slider("ラベル最大件数", 5, 20, 12)
    with c12:
        alternate_side = st.checkbox("ラベル左右交互配置", value=True)
    c13, c14, c15, c16, c17 = st.columns([1.0, 1.4, 1.2, 1.2, 1.2])
    with c13:
        unit = st.radio("単位", ["円", "千円", "百万円"], horizontal=True, index=1)
    with c14:
        n_win = st.slider("傾きウィンドウ（月）", 3, 12, 6, 1)
    with c15:
        cmp_mode = st.radio("傾き条件", ["以上", "未満"], horizontal=True)
    with c16:
        thr_type = st.radio("しきい値の種類", ["円/月", "%/月", "zスコア"], horizontal=True)
    with c17:
        default_thr = 0.03 if thr_type == "%/月" else (1.5 if thr_type == "zスコア" else 100000.0)
        thr_val = st.number_input("しきい値", value=float(default_thr))
    c18, c19, c20 = st.columns([1.6, 1.2, 1.8])
    with c18:
        sens = st.slider("形状抽出の感度", 0.0, 1.0, 0.5, 0.05)
    with c19:
        z_thr = st.slider("急勾配 zスコア", 0.5, 3.0, 1.5, 0.1)
    with c20:
        shape_pick = st.radio("形状抽出", ["（なし）", "急勾配", "山（への字）", "谷（逆への字）"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    params = {
        "end_month": end_m,
        "band_mode": band_mode,
        "band_params": band_params,
        "quick": quick,
    }
    st.session_state.compare_params = params

    mode_map = {
        "金額指定": "amount",
        "商品指定(2)": "two_products",
        "パーセンタイル": "percentile",
        "順位帯": "rank",
        "ターゲット近傍": "target_near",
    }
    low, high = resolve_band(snapshot, mode_map[band_mode], band_params)
    codes = filter_products_by_band(snapshot, low, high)

    if quick == "Top5":
        codes = snapshot.nlargest(5, "year_sum")["product_code"].tolist()
    elif quick == "Top10":
        codes = snapshot.nlargest(10, "year_sum")["product_code"].tolist()
    elif quick == "最新YoY上位":
        codes = (
            snapshot.dropna(subset=["yoy"]).sort_values("yoy", ascending=False).head(10)["product_code"].tolist()
        )
    elif quick == "直近6M伸長上位":
        codes = top_growth_codes(year_df, end_m, window=6, top=10)

    snap = slopes_snapshot(year_df, n=n_win)
    if   thr_type == "円/月":
        key, v = "slope_yen", float(thr_val)
    elif thr_type == "%/月":
        key, v = "slope_ratio", float(thr_val)
    else:
        key, v = "slope_z", float(thr_val)
    mask = (snap[key] >= v) if cmp_mode == "以上" else (snap[key] <= v)
    codes_by_slope = set(snap.loc[mask, "product_code"])

    shape_df = shape_flags(year_df, window=max(8, n_win * 2),
                            alpha_ratio=0.02 * (1.0 - sens), amp_ratio=0.06 * (1.0 - sens))
    codes_steep = set(snap.loc[snap["slope_z"].abs() >= z_thr, "product_code"])
    codes_mtn = set(shape_df.loc[shape_df["is_mountain"], "product_code"])
    codes_val = set(shape_df.loc[shape_df["is_valley"], "product_code"])
    shape_map = {"（なし）": None, "急勾配": codes_steep, "山（への字）": codes_mtn, "谷（逆への字）": codes_val}
    codes_by_shape = shape_map[shape_pick] or set(snap["product_code"])

    codes_from_band = set(codes)
    target_codes = list(codes_from_band & codes_by_slope & codes_by_shape)

    scale = {"円": 1, "千円": 1_000, "百万円": 1_000_000}[unit]
    snapshot_disp = snapshot.copy()
    snapshot_disp["year_sum_disp"] = snapshot_disp["year_sum"] / scale
    hist_fig = px.histogram(snapshot_disp, x="year_sum_disp")
    hist_fig.update_xaxes(title_text=f"年計（{unit}）")

    df_long, _ = get_yearly_series(year_df, target_codes)
    df_long["month"] = pd.to_datetime(df_long["month"])
    df_long["display_name"] = df_long["product_name"].fillna(df_long["product_code"])
    df_long["year_sum_disp"] = df_long["year_sum"] / scale

    months_back = {"12ヶ月": 12, "24ヶ月": 24, "36ヶ月": 36}[period]
    max_month = df_long["month"].max()
    if pd.notna(max_month):
        start_date = max_month - pd.DateOffset(months=months_back - 1)
        df_long = df_long[df_long["month"] >= start_date]

    # TopN 制限
    main_codes = target_codes
    max_lines = 15
    if len(main_codes) > max_lines:
        top_order = (
            snapshot[snapshot["product_code"].isin(main_codes)]
            .sort_values("year_sum", ascending=False)["product_code"].tolist()
        )
        main_codes = top_order[:max_lines]

    df_main = df_long[df_long["product_code"].isin(main_codes)]

    fig = px.line(
        df_main,
        x="month",
        y="year_sum_disp",
        color="display_name",
        custom_data=["display_name"],
    )
    fig.add_hrect(y0=low / scale, y1=high / scale, fillcolor="green", opacity=0.12, line_width=0)

    fig.update_traces(
        mode="lines",
        line=dict(width=1.5),
        opacity=0.45,
        selector=dict(mode="lines"),
        hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
    )
    fig.update_layout(hoverlabel=dict(bgcolor="rgba(30,30,30,0.92)", font=dict(color="#fff", size=12)))

    theme_is_dark = st.get_option("theme.base") == "dark"
    HALO = "#ffffff" if theme_is_dark else "#222222"
    SZ = 6
    if node_mode == "自動":
        step = marker_step(df_main["month"])
        df_nodes = (
            df_main.sort_values("month")
            .assign(_idx=df_main.sort_values("month").groupby("display_name").cumcount())
            .query("(_idx % @step) == 0")
        )
    elif node_mode == "主要ノードのみ":
        g = df_main.sort_values("month").groupby("display_name")
        latest = g.tail(1)
        idxmax = df_main.loc[g["year_sum"].idxmax().dropna()]
        idxmin = df_main.loc[g["year_sum"].idxmin().dropna()]
        ystart = g.head(1)
        df_nodes = pd.concat([latest, idxmax, idxmin, ystart]).drop_duplicates(["display_name", "month"])
    elif node_mode == "すべて":
        df_nodes = df_main.copy()
    else:
        df_nodes = df_main.iloc[0:0].copy()

    for name, d in df_nodes.groupby("display_name"):
        fig.add_scatter(
            x=d["month"],
            y=d["year_sum_disp"],
            mode="markers",
            name=name,
            legendgroup=name,
            showlegend=False,
            marker=dict(size=SZ, symbol="circle", line=dict(color=HALO, width=2), opacity=0.95),
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
            customdata=np.stack([d["display_name"]], axis=-1),
        )

    if node_mode != "非表示":
        if enable_label_avoid:
            add_latest_labels_no_overlap(
                fig,
                df_main,
                label_col="display_name",
                x_col="month",
                y_col="year_sum_disp",
                max_labels=label_max,
                min_gap_px=label_gap_px,
                alternate_side=alternate_side,
            )
        else:
            for _, r in df_nodes.groupby("display_name").tail(1).iterrows():
                fig.add_annotation(
                    x=r["month"],
                    y=r["year_sum_disp"],
                    text=f"{r['year_sum_disp']:,.0f} {unit}（{r['month']:%Y-%m}）",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=8,
                    ay=-12,
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    font=dict(size=11),
                )

    n_months = (
        df_main["month"].max().year * 12
        + df_main["month"].max().month
        - df_main["month"].min().year * 12
        - df_main["month"].min().month
        + 1
    ) if not df_main.empty else 0
    dtick = "M1" if n_months <= 18 else ("M3" if n_months <= 48 else "M6")
    fig.update_xaxes(
        dtick=dtick,
        tickformat="%Y-%m",
        tickangle=-45,
        showgrid=True,
        rangeslider_visible=True,
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        title_text="月（YYYY-MM）",
    )
    fig.update_yaxes(
        tickformat="~,d",
        title_text=f"売上 年計（{unit}）",
        zeroline=True,
        showgrid=True,
    )
    fig.update_layout(
        legend_title_text="商品名",
        legend=dict(y=1, x=1.02, yanchor="top", xanchor="left", tracegroupgap=8, itemsizing="constant"),
        margin=dict(l=60, r=160, t=40, b=70),
        template="plotly_dark",
        height=500,
        font=dict(family="Noto Sans JP, Meiryo, Arial", size=12),
    )

    drag = {"ズーム": "zoom", "パン": "pan", "選択": "select"}[op_mode]
    fig.update_layout(dragmode=drag)
    if hover_mode == "個別":
        fig.update_layout(hovermode="closest")
    else:
        fig.update_layout(hovermode="x unified", hoverlabel=dict(align="left"))

    if peak_on:
        for name, grp in df_main.groupby("display_name"):
            max_row = grp.loc[grp["year_sum"].idxmax()]
            min_row = grp.loc[grp["year_sum"].idxmin()]
            fig.add_annotation(
                x=max_row["month"],
                y=max_row["year_sum_disp"],
                text=f"{max_row['year_sum_disp']:,.0f} {unit} ({max_row['month'].strftime('%Y-%m')})",
                showarrow=False,
                yanchor="bottom",
                font=dict(size=9),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            )
            fig.add_annotation(
                x=min_row["month"],
                y=min_row["year_sum_disp"],
                text=f"{min_row['year_sum_disp']:,.0f} {unit} ({min_row['month'].strftime('%Y-%m')})",
                showarrow=False,
                yanchor="top",
                font=dict(size=9),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)",
                borderwidth=0,
            )

    st.markdown('<div class="chart-body">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)

    st.caption("凡例クリックで表示切替、ダブルクリックで単独表示。ドラッグでズーム/パン、右上メニューからPNG/CSV取得可。")
    st.markdown("""
傾き（円/月）：直近 n ヶ月の回帰直線の傾き。+は上昇、−は下降。

%/月：傾き÷平均年計。規模によらず比較可能。

zスコア：全SKUの傾き分布に対する標準化。|z|≥1.5で急勾配の目安。

山/谷：前半と後半の平均変化率の符号が**＋→−（山）／−→＋（谷）かつ振幅が十分**。
""")

    snap_export = snapshot[snapshot["product_code"].isin(main_codes)].copy()
    snap_export[f"year_sum_{unit}"] = snap_export["year_sum"] / scale
    snap_export = snap_export.drop(columns=["year_sum"]) 
    st.download_button(
        "CSVエクスポート",
        data=snap_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"band_snapshot_{end_m}.csv",
        mime="text/csv",
    )
    try:
        png_bytes = fig.to_image(format="png")
        st.download_button("PNGエクスポート", data=png_bytes, file_name=f"band_overlay_{end_m}.png", mime="image/png")
    except Exception:
        pass

    with st.expander("分布（オプション）", expanded=False):
        st.plotly_chart(hist_fig, use_container_width=True)

    # ---- Small Multiples ----
    st.subheader("スモールマルチプル")
    share_y = st.checkbox("Y軸共有", value=False)
    show_keynode_labels = st.checkbox("キーノードラベル表示", value=False)
    per_page = st.radio("1ページ表示枚数", [8, 12], horizontal=True, index=0)
    total_pages = max(1, math.ceil(len(main_codes) / per_page))
    page_idx = st.number_input("ページ", min_value=1, max_value=total_pages, value=1)
    start = (page_idx - 1) * per_page
    page_codes = main_codes[start : start + per_page]
    col_count = 4
    cols = st.columns(col_count)
    ymax = df_long[df_long["product_code"].isin(main_codes)]["year_sum_disp"].max() if share_y else None
    for i, code in enumerate(page_codes):
        g = df_long[df_long["product_code"] == code]
        disp = g["display_name"].iloc[0] if not g.empty else code
        fig_s = px.line(
            g,
            x="month",
            y="year_sum_disp",
            color_discrete_sequence=[fig.layout.colorway[i % len(fig.layout.colorway)]],
            custom_data=["display_name"],
        )
        fig_s.update_traces(
            mode="lines",
            line=dict(width=1.5),
            opacity=0.8,
            showlegend=False,
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
        )
        g_nodes = df_nodes[df_nodes["product_code"] == code]
        if not g_nodes.empty:
            fig_s.add_scatter(
                x=g_nodes["month"],
                y=g_nodes["year_sum_disp"],
                mode="markers",
                marker=dict(size=SZ, symbol="circle", line=dict(color=HALO, width=2), opacity=0.95),
                showlegend=False,
                hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
                customdata=np.stack([g_nodes["display_name"]], axis=-1),
            )
            if show_keynode_labels and node_mode != "非表示":
                last_r = g_nodes.sort_values("month").iloc[-1]
                fig_s.add_annotation(
                    x=last_r["month"],
                    y=last_r["year_sum_disp"],
                    text=f"{last_r['year_sum_disp']:,.0f} {unit}（{last_r['month']:%Y-%m}）",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=8,
                    ay=-12,
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    font=dict(size=10),
                )
        fig_s.update_xaxes(tickformat="%Y-%m", dtick=dtick, title_text="月（YYYY-MM）")
        fig_s.update_yaxes(tickformat="~,d", range=[0, ymax] if ymax else None, title_text=f"売上 年計（{unit}）")
        fig_s.update_layout(font=dict(family="Noto Sans JP, Meiryo, Arial", size=12))
        fig_s.update_layout(hoverlabel=dict(bgcolor="rgba(30,30,30,0.92)", font=dict(color="#fff", size=12)))
        fig_s.update_layout(dragmode=drag)
        if hover_mode == "個別":
            fig_s.update_layout(hovermode="closest")
        else:
            fig_s.update_layout(hovermode="x unified", hoverlabel=dict(align="left"))
        last_val = g.sort_values("month")["year_sum_disp"].iloc[-1] if not g.empty else np.nan
        with cols[i % col_count]:
            st.metric(disp, f"{last_val:,.0f} {unit}" if not np.isnan(last_val) else "—")
            st.plotly_chart(
                fig_s,
                use_container_width=True,
                height=150,
                config=PLOTLY_CONFIG,
            )

    # 5) SKU詳細
elif page == "SKU詳細":
    require_data()
    st.header("SKU 詳細")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_detail")
    prods = st.session_state.data_year[["product_code", "product_name"]].drop_duplicates().sort_values("product_code")
    mode = st.radio("表示モード", ["単品", "複数比較"], horizontal=True)
    if mode == "単品":
        prod_label = st.selectbox("SKU選択", options=prods["product_code"] + " | " + prods["product_name"])
        code = prod_label.split(" | ")[0]

        g_m = st.session_state.data_monthly[st.session_state.data_monthly["product_code"] == code].sort_values("month")
        g_y = st.session_state.data_year[st.session_state.data_year["product_code"] == code].sort_values("month")

        c1, c2 = st.columns(2)
        with c1:
            fig1 = px.line(g_m, x="month", y="sales_amount_jpy", title="単月 売上推移", markers=True)
            st.plotly_chart(fig1, use_container_width=True, height=350, config=PLOTLY_CONFIG)
        with c2:
            fig2 = px.line(g_y, x="month", y="year_sum", title="年計 推移", markers=True)
            st.plotly_chart(fig2, use_container_width=True, height=350, config=PLOTLY_CONFIG)

        row = g_y[g_y["month"] == end_m]
        if not row.empty:
            rr = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("年計", f"{int(rr['year_sum']) if not pd.isna(rr['year_sum']) else '—'}")
            c2.metric("YoY", f"{rr['yoy']*100:.1f} %" if not pd.isna(rr["yoy"]) else "—")
            c3.metric("Δ", f"{int(rr['delta'])}" if not pd.isna(rr["delta"]) else "—")

        st.subheader("メモ / タグ")
        note = st.text_area("メモ（保存で保持）", value=st.session_state.notes.get(code, ""), height=100)
        tags_str = st.text_input("タグ（カンマ区切り）", value=",".join(st.session_state.tags.get(code, [])))
        c1, c2 = st.columns([1, 1])
        if c1.button("保存"):
            st.session_state.notes[code] = note
            st.session_state.tags[code] = [t.strip() for t in tags_str.split(",") if t.strip()]
            st.success("保存しました")
        if c2.button("CSVでエクスポート"):
            meta = pd.DataFrame(
                [
                    {
                        "product_code": code,
                        "note": st.session_state.notes.get(code, ""),
                        "tags": ",".join(st.session_state.tags.get(code, [])),
                    }
                ]
            )
            st.download_button(
                "ダウンロード",
                data=meta.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"notes_{code}.csv",
                mime="text/csv",
            )
    else:
        opts = (prods["product_code"] + " | " + prods["product_name"]).tolist()
        sel = st.multiselect("SKU選択（最大12件）", options=opts, max_selections=12)
        codes = [s.split(" | ")[0] for s in sel]
        if codes:
            df_long, _ = get_yearly_series(st.session_state.data_year, codes)
            df_long["display_name"] = df_long["product_name"].fillna(df_long["product_code"])
            fig = px.line(df_long, x="month", y="year_sum", color="display_name")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True, height=350, config=PLOTLY_CONFIG)

            fig_sm = px.line(
                df_long,
                x="month",
                y="year_sum",
                color="display_name",
                facet_col="display_name",
                facet_col_wrap=3,
            )
            fig_sm.update_layout(showlegend=False)
            st.plotly_chart(fig_sm, use_container_width=True, height=350, config=PLOTLY_CONFIG)

            snap = latest_yearsum_snapshot(st.session_state.data_year, end_m)
            snap = snap[snap["product_code"].isin(codes)]
            st.dataframe(
                snap[["product_code", "product_name", "year_sum", "yoy", "delta"]],
                use_container_width=True,
            )
            st.download_button(
                "CSVダウンロード",
                data=snap.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"sku_multi_{end_m}.csv",
                mime="text/csv",
            )
        else:
            st.info("SKUを選択してください。")

# 5) 相関分析
elif page == "相関分析":
    require_data()
    st.header("相関分析")
    end_m = end_month_selector(st.session_state.data_year, key="corr_end_month")
    snapshot = latest_yearsum_snapshot(st.session_state.data_year, end_m)

    metric_opts = ["year_sum", "yoy", "delta", "slope_beta", "slope6m", "std6m", "hhi_share"]
    metrics = st.multiselect(
        "指標",
        [m for m in metric_opts if m in snapshot.columns],
        default=[m for m in ["year_sum", "yoy", "delta", "slope_beta"] if m in snapshot.columns],
    )
    method = st.radio(
        "相関の種類",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda x: "Pearson" if x == "pearson" else "Spearman",
    )
    winsor_pct = st.slider("外れ値丸め(%)", 0.0, 5.0, 1.0)
    log_enable = st.checkbox("ログ変換", value=False)

    if metrics:
        df_plot = snapshot.copy()
        df_plot = winsorize_frame(df_plot, metrics, p=winsor_pct / 100)
        df_plot = maybe_log1p(df_plot, metrics, log_enable)
        tbl = corr_table(df_plot, metrics, method=method)

        st.subheader("相関の要点")
        for line in narrate_top_insights(tbl, NAME_MAP):
            st.write("・", line)
        sig_cnt = int((tbl["sig"] == "有意(95%)").sum())
        weak_cnt = int((tbl["r"].abs() < 0.2).sum())
        st.write(f"統計的に有意な相関: {sig_cnt} 組")
        st.write(f"|r|<0.2 の組み合わせ: {weak_cnt} 組")

        st.subheader("相関ヒートマップ")
        st.caption("右上=強い正、左下=強い負、白=関係薄")
        corr = df_plot[metrics].corr(method=method)
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True)
        st.plotly_chart(fig_corr, use_container_width=True, config=PLOTLY_CONFIG)

        st.subheader("ペア・エクスプローラ")
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("指標X", metrics, index=0)
        with c2:
            y_col = st.selectbox("指標Y", metrics, index=1 if len(metrics) > 1 else 0)
        df_xy = df_plot[[x_col, y_col, "product_name", "product_code"]].dropna()
        if not df_xy.empty:
            m, b, r2 = fit_line(df_xy[x_col], df_xy[y_col])
            r = df_xy[x_col].corr(df_xy[y_col], method=method)
            lo, hi = fisher_ci(r, len(df_xy))
            fig_sc = px.scatter(df_xy, x=x_col, y=y_col, hover_data=["product_code", "product_name"])
            xs = np.linspace(df_xy[x_col].min(), df_xy[x_col].max(), 100)
            fig_sc.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="回帰"))
            fig_sc.add_annotation(
                x=0.99,
                y=0.01,
                xref="paper",
                yref="paper",
                xanchor="right",
                yanchor="bottom",
                text=f"r={r:.2f} (95%CI [{lo:.2f},{hi:.2f}])<br>R²={r2:.2f}",
                showarrow=False,
                align="right",
                bgcolor="rgba(255,255,255,0.6)",
            )
            resid = np.abs(df_xy[y_col] - (m * df_xy[x_col] + b))
            outliers = df_xy.loc[resid.nlargest(3).index]
            for _, row in outliers.iterrows():
                label = row["product_name"] or row["product_code"]
                fig_sc.add_annotation(x=row[x_col], y=row[y_col], text=label, showarrow=True, arrowhead=1)
            st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("rは -1〜+1。0は関連が薄い。CIに0を含まなければ有意。")
            st.caption("散布図の点が右上・左下に伸びれば正、右下・左上なら負。")
    else:
        st.info("指標を選択してください。")

    with st.expander("相関の読み方"):
        st.write("正の相関：片方が大きいほどもう片方も大きい")
        st.write("負の相関：片方が大きいほどもう片方は小さい")
        st.write("|r|<0.2は弱い、0.2-0.5はややあり、0.5-0.8は中~強、>0.8は非常に強い（目安）")

# 6) アラート
elif page == "アラート":
    require_data()
    st.header("アラート")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_alert")
    s = st.session_state.settings
    alerts = build_alerts(st.session_state.data_year, end_month=end_m,
                          yoy_threshold=s["yoy_threshold"],
                          delta_threshold=s["delta_threshold"],
                          slope_threshold=s["slope_threshold"])
    if alerts.empty:
        st.success("閾値に該当するアラートはありません。")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button("CSVダウンロード", data=alerts.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"alerts_{end_m}.csv", mime="text/csv")

# 6) 設定
elif page == "設定":
    st.header("設定")
    s = st.session_state.settings
    c1, c2, c3 = st.columns(3)
    with c1:
        s["window"] = st.number_input("年計ウィンドウ（月）", min_value=3, max_value=24, value=int(s["window"]), step=1)
        s["last_n"] = st.number_input("傾き算出の対象点数", min_value=3, max_value=36, value=int(s["last_n"]), step=1)
    with c2:
        s["yoy_threshold"] = st.number_input("YoY 閾値（<=）", value=float(s["yoy_threshold"]), step=0.01, format="%.2f")
        s["delta_threshold"] = st.number_input("Δ 閾値（<= 円）", value=float(s["delta_threshold"]), step=10000.0, format="%.0f")
    with c3:
        s["slope_threshold"] = st.number_input("傾き 閾値（<=）", value=float(s["slope_threshold"]), step=0.1, format="%.2f")
        s["currency_unit"] = st.selectbox("通貨単位表記", options=["円","千円","百万円"], index=["円","千円","百万円"].index(s["currency_unit"]))

    st.caption("※ 設定変更後は再計算が必要です。")
    if st.button("年計の再計算を実行", type="primary"):
        if st.session_state.data_monthly is None:
            st.warning("先にデータを取り込んでください。")
        else:
            long_df = st.session_state.data_monthly
            year_df = compute_year_rolling(long_df, window=s["window"], policy=s["missing_policy"])
            year_df = compute_slopes(year_df, last_n=s["last_n"])
            st.session_state.data_year = year_df
            st.success("再計算が完了しました。")

# 7) 保存ビュー
elif page == "保存ビュー":
    st.header("保存ビュー / ブックマーク")
    s = st.session_state.settings
    cparams = st.session_state.compare_params
    st.write("現在の設定・選択（閾値、ウィンドウ、単位など）を名前を付けて保存します。")

    name = st.text_input("ビュー名")
    if st.button("保存"):
        if not name:
            st.warning("ビュー名を入力してください。")
        else:
            st.session_state.saved_views[name] = {"settings": dict(s), "compare": dict(cparams)}
            st.success(f"ビュー「{name}」を保存しました。")

    st.subheader("保存済みビュー")
    if not st.session_state.saved_views:
        st.info("保存済みビューはありません。")
    else:
        for k, v in st.session_state.saved_views.items():
            st.write(f"**{k}**: {json.dumps(v, ensure_ascii=False)}")
            if st.button(f"適用: {k}"):
                st.session_state.settings.update(v.get("settings", {}))
                st.session_state.compare_params = v.get("compare", {})
                st.session_state.compare_results = None
                st.success(f"ビュー「{k}」を適用しました。")
