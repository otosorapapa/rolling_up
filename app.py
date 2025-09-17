import io
import json
import math
import textwrap
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ai_features import (
    summarize_dataframe,
    generate_comment,
    explain_analysis,
    generate_actions,
    answer_question,
    generate_anomaly_brief,
)

# McKinsey inspired pastel palette
MCKINSEY_PALETTE = [
    "#123a5f",  # deep navy
    "#2d6f8e",  # steel blue
    "#4f9ab8",  # aqua accent
    "#71b7d4",  # sky blue
    "#a9d0e7",  # frost blue
    "#dbe8f5",  # airy pastel
]
# Apply palette across figures
px.defaults.color_discrete_sequence = MCKINSEY_PALETTE

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


@st.cache_data(ttl=600)
def _ai_sum_df(df: pd.DataFrame) -> str:
    return summarize_dataframe(df)


@st.cache_data(ttl=600)
def _ai_explain(d: dict) -> str:
    return explain_analysis(d)


@st.cache_data(ttl=600)
def _ai_comment(t: str) -> str:
    return generate_comment(t)


@st.cache_data(ttl=600)
def _ai_actions(metrics: Dict[str, float], focus: str) -> str:
    return generate_actions(metrics, focus)


@st.cache_data(ttl=600)
def _ai_answer(question: str, context: str) -> str:
    return answer_question(question, context)


@st.cache_data(ttl=600)
def _ai_anomaly_report(df: pd.DataFrame) -> str:
    return generate_anomaly_brief(df)


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
    detect_linear_anomalies,
)
from core.chart_card import toolbar_sku_detail, build_chart_card
from core.plot_utils import apply_elegant_theme

APP_TITLE = "売上年計（12カ月移動累計）ダッシュボード"
st.set_page_config(
    page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded"
)

# McKinsey inspired light theme
st.markdown(
    """
<style>
:root{
  --bg:#edf3f9;
  --panel:#ffffff;
  --text:#1f2a36;
  --accent:#123a5f;
  --accent-soft:#2d6f8e;
  --accent-light:#71b7d4;
  --muted:#4f6274;
}
body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--text) !important; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b2f4c 0%,#123a5f 100%); color:#fff; padding-top:1rem; }
[data-testid="stSidebar"] *{ color:#fff !important; }
[data-testid="stSidebar"] .stButton>button{ background:rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.25); color:#fff; }
h1,h2,h3{ color:var(--accent); font-weight:800; letter-spacing:.4px; }
p,li,span,div{ color:var(--text); }
[data-testid="stMetric"]{ background:var(--panel); border:1px solid rgba(11,41,70,0.08); border-radius:12px; padding:0.75rem 0.9rem; box-shadow:0 6px 18px rgba(17,49,82,0.08); }
[data-testid="stMetricValue"]{ color:var(--accent); font-variant-numeric:tabular-nums; font-weight:700; }
[data-testid="stMetricLabel"]{ color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.08em; }
.mck-sidebar-summary{ background:rgba(255,255,255,0.10); border-radius:12px; padding:0.85rem; margin-bottom:1.2rem; font-size:0.88rem; line-height:1.5; }
.mck-sidebar-summary strong{ color:#fff; }
.mck-hero{ background:linear-gradient(135deg, rgba(18,58,95,0.94) 0%, rgba(47,111,142,0.86) 100%); color:#fff; padding:1.8rem 2rem; border-radius:18px; margin-bottom:1.2rem; box-shadow:0 18px 38px rgba(11,44,74,0.25); position:relative; overflow:hidden; }
.mck-hero::after{ content:""; position:absolute; inset:auto -18% -32% auto; width:220px; height:220px; background:rgba(255,255,255,0.12); border-radius:50%; }
.mck-hero h1{ color:#fff; margin-bottom:0.5rem; font-size:1.9rem; }
.mck-hero p{ color:rgba(255,255,255,0.82); font-size:1rem; margin-bottom:0; }
.mck-hero__eyebrow{ text-transform:uppercase; letter-spacing:.16em; font-size:0.75rem; font-weight:600; color:rgba(255,255,255,0.82); margin-bottom:0.6rem; display:inline-flex; align-items:center; gap:0.5rem; }
.mck-hero__eyebrow:before{ content:"◦"; font-size:0.9rem; }
.mck-section-header{ display:flex; align-items:flex-start; gap:0.85rem; margin:0.8rem 0 0.6rem; }
.mck-section-header h2{ margin:0; font-size:1.35rem; line-height:1.2; }
.mck-section-subtitle{ margin:0.25rem 0 0; font-size:0.92rem; color:var(--muted); }
.mck-section-icon{ width:42px; height:42px; display:inline-flex; align-items:center; justify-content:center; border-radius:50%;background:rgba(18,58,95,0.12); color:var(--accent); font-size:1.2rem; flex-shrink:0; margin-top:0.1rem; }
.mck-ai-answer{ background:var(--panel); border-radius:12px; border:1px solid rgba(11,41,70,0.08); padding:0.75rem 0.9rem; box-shadow:0 12px 26px rgba(11,44,74,0.10); margin-top:0.75rem; }
.mck-ai-answer strong{ color:var(--accent); }
.stTabs [data-baseweb="tab-list"]{ gap:0.6rem; }
.stTabs [data-baseweb="tab"]{ background:var(--panel); padding:0.6rem 1rem; border-radius:999px; border:1px solid rgba(18,58,95,0.18); color:var(--muted); font-weight:600; }
.stTabs [data-baseweb="tab"]:hover{ border-color:rgba(47,111,142,0.42); color:var(--accent); }
.stTabs [data-baseweb="tab"]:focus{ outline:none; }
.stTabs [aria-selected="true"]{ background:#123a5f; color:#fff; border-color:#123a5f; }
.stDataFrame{ border-radius:14px !important; }
.stButton>button{ border-radius:999px; padding:0.45rem 1.2rem; font-weight:600; border:1px solid rgba(18,58,95,0.35); color:var(--accent); background:rgba(18,58,95,0.06); }
.stButton>button:hover{ background:rgba(18,58,95,0.12); border-color:#2d6f8e; color:#123a5f; }
.chart-card{ background:var(--panel); border:1px solid rgba(11,41,70,0.08); border-radius:14px; box-shadow:0 12px 26px rgba(11,44,74,0.06); }
.chart-toolbar{ background:linear-gradient(180deg, rgba(18,58,95,0.04), rgba(18,58,95,0.01)); border-bottom:1px solid rgba(18,58,95,0.18); }
</style>
    """,
    unsafe_allow_html=True,
)

# ===== Elegant（品格）UI ON/OFF（ヘッダに設置） =====
elegant_on = st.toggle(
    "品格UI",
    value=True,
    help="上品で読みやすい配色・余白・タイポグラフィを適用",
)
st.session_state["elegant_on"] = elegant_on

# ===== 品格UI CSS（配色/余白/フォント/境界の見直し） =====
if elegant_on:
    st.markdown(
        """
    <style>
      :root{
        --ink:#1c2733;
        --bg:#eef2f8;
        --panel:#ffffff;
        --line:rgba(18,58,95,.12);
        --accent:#123a5f;
        --muted:#526274;
      }
      body, .stApp, [data-testid="stAppViewContainer"]{ background:var(--bg) !important; color:var(--ink) !important; }
      h1,h2,h3{ letter-spacing:.3px; font-weight:800; color:var(--accent); }
      p,li,div,span{ font-variant-numeric: tabular-nums; color:var(--ink); }
      .chart-card, .stTabs, .stDataFrame, .element-container{
        border-radius:16px; box-shadow:0 16px 32px rgba(18,38,67,.08);
        border:1px solid var(--line); background:var(--panel);
      }
      .chart-toolbar{
        background:linear-gradient(180deg, rgba(18,58,95,.08), rgba(18,58,95,.02));
        border-bottom:1px solid rgba(18,58,95,.18);
      }
      .stButton>button, .stRadio label, .stCheckbox label, .stSelectbox label{ border-radius:999px; font-weight:600; color:var(--accent); }
      .stButton>button{ border:1px solid rgba(18,58,95,0.35); background:rgba(18,58,95,0.06); }
      .stButton>button:hover{ background:rgba(18,58,95,0.12); border-color:#2d6f8e; color:#123a5f; }
      [data-testid="stSidebar"]{ background:linear-gradient(180deg,#0b2f4c 0%,#123a5f 100%); color:#fff; }
      [data-testid="stSidebar"] *{ color:#fff !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

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
if "copilot_answer" not in st.session_state:
    st.session_state.copilot_answer = ""
if "copilot_context" not in st.session_state:
    st.session_state.copilot_context = ""
if "copilot_focus" not in st.session_state:
    st.session_state.copilot_focus = "全体サマリー"

# track user interactions and global filters
if "click_log" not in st.session_state:
    st.session_state.click_log = {}
if "filters" not in st.session_state:
    st.session_state.filters = {}

# currency unit scaling factors
UNIT_MAP = {"円": 1, "千円": 1_000, "百万円": 1_000_000}


def log_click(name: str):
    """Increment click count for command bar actions."""
    st.session_state.click_log[name] = st.session_state.click_log.get(name, 0) + 1


def render_app_hero():
    st.markdown(
        f"""
        <div class=\"mck-hero\">
            <div class=\"mck-hero__eyebrow\">Growth Intelligence Workspace</div>
            <h1>{APP_TITLE}</h1>
            <p>12カ月移動累計で成長ドライバーとリスクを直感的に把握し、次の一手を素早く導きます。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(
    title: str, subtitle: Optional[str] = None, icon: Optional[str] = None
):
    icon_html = f"<span class='mck-section-icon'>{icon}</span>" if icon else ""
    subtitle_html = (
        f"<p class='mck-section-subtitle'>{subtitle}</p>" if subtitle else ""
    )
    st.markdown(
        f"""
        <div class=\"mck-section-header\">
            {icon_html}
            <div>
                <h2>{title}</h2>
                {subtitle_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clip_text(value: str, width: int = 220) -> str:
    if not value:
        return ""
    return textwrap.shorten(value, width=width, placeholder="…")


# ---------------- Helpers ----------------
def require_data():
    if st.session_state.data_year is None or st.session_state.data_monthly is None:
        st.info(
            "データが未取り込みです。左メニュー「データ取込」からアップロードしてください。"
        )
        st.stop()


def month_options(df: pd.DataFrame) -> List[str]:
    return sorted(df["month"].dropna().unique().tolist())


def end_month_selector(df: pd.DataFrame, key="end_month"):
    mopts = month_options(df)
    default = mopts[-1] if mopts else None
    return st.selectbox(
        "終端月（年計の計算対象）",
        mopts,
        index=(len(mopts) - 1) if mopts else 0,
        key=key,
    )


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
        c.drawString(
            40,
            y,
            f"{row['product_code']}  {row['product_name']}  {int(row['year_sum']):,}",
        )
        y -= 12
        if y < 60:
            c.showPage()
            y = h - 50
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def format_amount(val: Optional[float], unit: str) -> str:
    """Format a numeric value according to currency unit."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    scale = UNIT_MAP.get(unit, 1)
    return f"{format_int(val / scale)} {unit}".strip()


def format_int(val: float | int) -> str:
    """Format a number with commas and no decimal part."""
    try:
        return f"{int(round(val)):,}"
    except (TypeError, ValueError):
        return "0"


def nice_slider_step(max_value: int, target_steps: int = 40) -> int:
    """Return an intuitive step size so sliders move in round increments."""
    if max_value <= 0:
        return 1

    raw_step = max_value / target_steps
    if raw_step <= 1:
        return 1

    exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
    base = raw_step / (10 ** exponent) if raw_step > 0 else 1

    for nice in (1, 2, 5, 10):
        if base <= nice:
            step = nice * (10 ** exponent)
            return int(step) if step >= 1 else 1

    return int(10 ** (exponent + 1))


def choose_amount_slider_unit(max_amount: int) -> tuple[int, str]:
    """Choose a unit so the slider operates in easy-to-understand scales."""
    units = [
        (1, "円"),
        (1_000, "千円"),
        (10_000, "万円"),
        (1_000_000, "百万円"),
        (100_000_000, "億円"),
    ]

    if max_amount <= 0:
        return units[0]

    for scale, label in units:
        if max_amount / scale <= 300:
            return scale, label

    return units[-1]


def int_input(label: str, value: int) -> int:
    """Text input for integer values displayed with thousands separators."""
    text = st.text_input(label, format_int(value))
    try:
        return int(text.replace(",", ""))
    except ValueError:
        return value


def render_sidebar_summary() -> Optional[str]:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        st.sidebar.caption("データを取り込むと最新サマリーが表示されます。")
        return None

    months = month_options(year_df)
    if not months:
        st.sidebar.caption("月次データが存在しません。")
        return None

    end_m = months[-1]
    unit = st.session_state.settings.get("currency_unit", "円")
    kpi = aggregate_overview(year_df, end_m)
    hhi_val = compute_hhi(year_df, end_m)
    sku_cnt = int(year_df["product_code"].nunique())
    rec_cnt = int(len(year_df))

    total_txt = format_amount(kpi.get("total_year_sum"), unit)
    yoy_val = kpi.get("yoy")
    yoy_txt = f"{yoy_val * 100:.1f}%" if yoy_val is not None else "—"
    delta_txt = format_amount(kpi.get("delta"), unit)
    hhi_txt = f"{hhi_val:.3f}" if hhi_val is not None else "—"

    st.sidebar.markdown(
        f"""
        <div class=\"mck-sidebar-summary\">
            <strong>最新月:</strong> {end_m}<br>
            <strong>年計総額:</strong> {total_txt}<br>
            <strong>YoY:</strong> {yoy_txt}<br>
            <strong>Δ:</strong> {delta_txt}<br>
            <strong>HHI:</strong> {hhi_txt}<br>
            <strong>SKU数:</strong> {sku_cnt:,}<br>
            <strong>レコード:</strong> {rec_cnt:,}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return end_m


def build_copilot_context(
    focus: str, end_month: Optional[str] = None, top_n: int = 5
) -> str:
    year_df = st.session_state.get("data_year")
    if year_df is None or year_df.empty:
        return "データが取り込まれていません。"

    months = month_options(year_df)
    if not months:
        return "月度情報が存在しません。"

    end_m = end_month or months[-1]
    snap = (
        year_df[year_df["month"] == end_m]
        .dropna(subset=["year_sum"])
        .copy()
    )
    if snap.empty:
        return f"{end_m}の年計スナップショットが空です。"

    kpi = aggregate_overview(year_df, end_m)
    hhi_val = compute_hhi(year_df, end_m)

    def fmt_amt(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "—"
        return f"{format_int(val)}円"

    def fmt_pct(val: Optional[float]) -> str:
        if val is None or pd.isna(val):
            return "—"
        return f"{val * 100:.1f}%"

    lines = [
        f"対象月: {end_m}",
        f"年計総額: {fmt_amt(kpi.get('total_year_sum'))}",
        f"年計YoY: {fmt_pct(kpi.get('yoy'))}",
        f"前月差Δ: {fmt_amt(kpi.get('delta'))}",
        f"SKU数: {snap['product_code'].nunique():,}",
    ]
    if hhi_val is not None:
        lines.append(f"HHI: {hhi_val:.3f}")

    if focus == "伸びているSKU":
        subset = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(top_n)
        )
        label = "伸長SKU"
    elif focus == "苦戦しているSKU":
        subset = (
            snap.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=True)
            .head(top_n)
        )
        label = "苦戦SKU"
    else:
        subset = snap.sort_values("year_sum", ascending=False).head(top_n)
        label = "主要SKU"

    if not subset.empty:
        bullets = []
        for _, row in subset.iterrows():
            name = row.get("product_name") or row.get("product_code")
            yoy_txt = fmt_pct(row.get("yoy"))
            delta_txt = fmt_amt(row.get("delta"))
            bullets.append(
                f"{name} (年計 {fmt_amt(row.get('year_sum'))}, YoY {yoy_txt}, Δ {delta_txt})"
            )
        lines.append(f"{label}: " + " / ".join(bullets))

    worst = (
        snap.dropna(subset=["yoy"])
        .sort_values("yoy", ascending=True)
        .head(1)
    )
    best = (
        snap.dropna(subset=["yoy"])
        .sort_values("yoy", ascending=False)
        .head(1)
    )
    if not best.empty:
        b = best.iloc[0]
        lines.append(
            f"YoY最高: {(b['product_name'] or b['product_code'])} ({fmt_pct(b['yoy'])})"
        )
    if not worst.empty:
        w = worst.iloc[0]
        lines.append(
            f"YoY最低: {(w['product_name'] or w['product_code'])} ({fmt_pct(w['yoy'])})"
        )

    return " ｜ ".join(lines)


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
st.sidebar.markdown(
    f"""
    <div style="font-weight:700; font-size:1.05rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:0.75rem;">
        {APP_TITLE}
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "メニュー",
    [
        "ダッシュボード",
        "ランキング",
        "比較ビュー",
        "SKU詳細",
        "異常検知",
        "相関分析",
        "データ取込",
        "アラート",
        "設定",
        "保存ビュー",
    ],
)
latest_month = render_sidebar_summary()
st.sidebar.divider()

with st.sidebar.expander("AIコパイロット", expanded=False):
    st.caption("最新の年計スナップショットを使って質問できます。")
    st.text_area(
        "聞きたいこと",
        key="copilot_question",
        height=90,
        placeholder="例：前年同月比が高いSKUや、下落しているSKUを教えて",
    )
    focus = st.selectbox(
        "フォーカス",
        ["全体サマリー", "伸びているSKU", "苦戦しているSKU"],
        key="copilot_focus",
    )
    if st.button("AIに質問", key="ask_ai", use_container_width=True):
        question = st.session_state.get("copilot_question", "").strip()
        if not question:
            st.warning("質問を入力してください。")
        else:
            context = build_copilot_context(focus, end_month=latest_month)
            answer = _ai_answer(question, context)
            st.session_state.copilot_answer = answer
            st.session_state.copilot_context = context
    if st.session_state.copilot_answer:
        st.markdown(
            f"<div class='mck-ai-answer'><strong>AI回答</strong><br>{st.session_state.copilot_answer}</div>",
            unsafe_allow_html=True,
        )
        if st.session_state.copilot_context:
            st.caption("コンテキスト: " + clip_text(st.session_state.copilot_context, 220))
st.sidebar.divider()

render_app_hero()

# ---------------- Pages ----------------

# 1) データ取込
if page == "データ取込":
    section_header(
        "データ取込", "ファイルのマッピングと品質チェックを行います。", icon="📥"
    )

    st.markdown(
        "**Excel(.xlsx) / CSV をアップロードしてください。** "
        "列に `YYYY-MM`（または日付系）形式の月度が含まれている必要があります。"
    )

    col_u1, col_u2 = st.columns([2, 1])
    with col_u1:
        file = st.file_uploader("ファイル選択", type=["xlsx", "csv"])
    with col_u2:
        st.session_state.settings["missing_policy"] = st.selectbox(
            "欠測月ポリシー",
            options=["zero_fill", "mark_missing"],
            format_func=lambda x: (
                "ゼロ補完(推奨)" if x == "zero_fill" else "欠測含む窓は非計上"
            ),
            index=0,
        )

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
        product_code_col = st.selectbox(
            "商品コード列の選択（任意）", options=["<なし>"] + cols, index=0
        )
        code_col = None if product_code_col == "<なし>" else product_code_col

        if st.button("変換＆取込", type="primary"):
            try:
                long_df = parse_uploaded_table(
                    df_raw, product_name_col=product_name_col, product_code_col=code_col
                )
                long_df = fill_missing_months(
                    long_df, policy=st.session_state.settings["missing_policy"]
                )
                # Compute year rolling & slopes
                year_df = compute_year_rolling(
                    long_df,
                    window=st.session_state.settings["window"],
                    policy=st.session_state.settings["missing_policy"],
                )
                year_df = compute_slopes(
                    year_df, last_n=st.session_state.settings["last_n"]
                )

                st.session_state.data_monthly = long_df
                st.session_state.data_year = year_df
                st.success(
                    "取込完了。ダッシュボードへ移動して可視化を確認してください。"
                )

                st.subheader("品質チェック（欠測月/非数値/重複）")
                # 欠測月
                miss_rate = (long_df["is_missing"].sum(), len(long_df))
                st.write(f"- 欠測セル数: {miss_rate[0]:,} / {miss_rate[1]:,}")
                # 月レンジ
                st.write(
                    f"- データ期間: {long_df['month'].min()} 〜 {long_df['month'].max()}"
                )
                # SKU数
                st.write(f"- SKU数: {long_df['product_code'].nunique():,}")
                st.write(f"- レコード数: {len(long_df):,}")

                st.download_button(
                    "年計テーブルをCSVでダウンロード",
                    data=st.session_state.data_year.to_csv(index=False).encode(
                        "utf-8-sig"
                    ),
                    file_name="year_rolling.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.exception(e)

# 2) ダッシュボード
elif page == "ダッシュボード":
    require_data()
    section_header("ダッシュボード", "年計KPIと成長トレンドを俯瞰します。", icon="📈")

    # Command bar (期間/単位)
    with st.container():
        col_p, col_u = st.columns([1, 1])
        with col_p:
            st.selectbox(
                "期間",
                options=[12, 24, 36],
                index=[12, 24, 36].index(st.session_state.settings.get("window", 12)),
                key="cmd_period",
                on_change=lambda: log_click("期間"),
            )
        with col_u:
            st.selectbox(
                "単位",
                options=list(UNIT_MAP.keys()),
                index=list(UNIT_MAP.keys()).index(
                    st.session_state.settings.get("currency_unit", "円")
                ),
                key="cmd_unit",
                on_change=lambda: log_click("単位"),
            )

    # update settings and filter log
    st.session_state.settings["window"] = st.session_state.cmd_period
    st.session_state.settings["currency_unit"] = st.session_state.cmd_unit
    st.session_state.filters.update(
        {
            "period": st.session_state.cmd_period,
            "currency_unit": st.session_state.cmd_unit,
        }
    )

    end_m = end_month_selector(st.session_state.data_year, key="end_month_dash")

    # KPI
    kpi = aggregate_overview(st.session_state.data_year, end_m)
    hhi = compute_hhi(st.session_state.data_year, end_m)
    unit = st.session_state.settings["currency_unit"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("年計総額", format_amount(kpi["total_year_sum"], unit))
    c2.metric("年計YoY", f"{kpi['yoy']*100:.1f} %" if kpi["yoy"] is not None else "—")
    c3.metric("前月差(Δ)", format_amount(kpi["delta"], unit))
    c4.metric("HHI(集中度)", f"{hhi:.3f}")

    snap = (
        st.session_state.data_year[st.session_state.data_year["month"] == end_m]
        .dropna(subset=["year_sum"])
        .copy()
        .sort_values("year_sum", ascending=False)
    )

    totals = st.session_state.data_year.groupby("month", as_index=False)[
        "year_sum"
    ].sum()
    totals["year_sum_disp"] = totals["year_sum"] / UNIT_MAP[unit]

    tab_highlight, tab_ranking = st.tabs(["ハイライト", "ランキング / エクスポート"])

    with tab_highlight:
        ai_on = st.toggle(
            "AIサマリー",
            value=False,
            help="要約・コメント・自動説明を表示（オンデマンド計算）",
            key="dash_ai_summary",
        )
        if ai_on:
            with st.spinner("AI要約を生成中…"):
                kpi_text = _ai_explain(
                    {
                        "年計総額": kpi["total_year_sum"],
                        "年計YoY": kpi["yoy"],
                        "前月差Δ": kpi["delta"],
                    }
                )
                snap_ai = snap[["year_sum", "yoy", "delta"]].head(100)
                stat_text = _ai_sum_df(snap_ai)
                st.info(f"**AI説明**：{kpi_text}\n\n**AI要約**：{stat_text}")
                actions = _ai_actions(
                    {
                        "total_year_sum": float(kpi.get("total_year_sum") or 0.0),
                        "yoy": float(kpi.get("yoy") or 0.0),
                        "delta": float(kpi.get("delta") or 0.0),
                        "hhi": float(hhi or 0.0),
                    },
                    focus=end_m,
                )
                st.success(f"**AI推奨アクション**：{actions}")
                st.caption(_ai_comment("直近の年計トレンドと上位SKUの動向"))

        fig = px.line(
            totals, x="month", y="year_sum_disp", title="総合 年計トレンド", markers=True
        )
        fig.update_yaxes(title=f"年計({unit})", tickformat="~,d")
        fig.update_layout(height=525, margin=dict(l=10, r=10, t=50, b=10))
        fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "dark"))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        st.caption("凡例クリックで系列の表示切替、ダブルクリックで単独表示。")

    with tab_ranking:
        st.markdown(f"#### ランキング（{end_m} 時点 年計）")
        snap_disp = snap.copy()
        snap_disp["year_sum"] = snap_disp["year_sum"] / UNIT_MAP[unit]
        st.dataframe(
            snap_disp[["product_code", "product_name", "year_sum", "yoy", "delta"]].head(
                20
            ),
            use_container_width=True,
        )
        st.download_button(
            "この表をCSVでダウンロード",
            data=snap.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"ranking_{end_m}.csv",
            mime="text/csv",
        )

        pdf_bytes = download_pdf_overview(
            {
                "total_year_sum": int(kpi["total_year_sum"])
                if kpi["total_year_sum"] is not None
                else 0,
                "yoy": round(kpi["yoy"], 4) if kpi["yoy"] is not None else None,
                "delta": int(kpi["delta"]) if kpi["delta"] is not None else None,
            },
            snap,
            filename=f"overview_{end_m}.pdf",
        )
        st.download_button(
            "会議用PDF（KPI+Top10）を出力",
            data=pdf_bytes,
            file_name=f"overview_{end_m}.pdf",
            mime="application/pdf",
        )

# 3) ランキング
elif page == "ランキング":
    require_data()
    section_header("ランキング", "上位と下位のSKUを瞬時に把握します。", icon="🏆")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_rank")
    metric = st.selectbox(
        "指標", options=["year_sum", "yoy", "delta", "slope_beta"], index=0
    )
    order = st.radio("並び順", options=["desc", "asc"], horizontal=True)
    hide_zero = st.checkbox("年計ゼロを除外", value=True)

    ai_on = st.toggle(
        "AIサマリー",
        value=False,
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    snap = st.session_state.data_year[
        st.session_state.data_year["month"] == end_m
    ].copy()
    total = len(snap)
    zero_cnt = int((snap["year_sum"] == 0).sum())
    if hide_zero:
        snap = snap[snap["year_sum"] > 0]
    snap = snap.dropna(subset=[metric])
    snap = snap.sort_values(metric, ascending=(order == "asc"))
    st.caption(f"除外 {zero_cnt} 件 / 全 {total} 件")

    fig_bar = px.bar(snap.head(20), x="product_name", y=metric)
    fig_bar = apply_elegant_theme(
        fig_bar, theme=st.session_state.get("ui_theme", "dark")
    )
    st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    if ai_on and not snap.empty:
        st.info(_ai_sum_df(snap[["year_sum", "yoy", "delta"]].head(200)))
        st.caption(_ai_comment("上位と下位の入替やYoYの極端値に注意"))

    st.dataframe(
        snap[
            ["product_code", "product_name", "year_sum", "yoy", "delta", "slope_beta"]
        ].head(100),
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
    section_header("マルチ商品比較", "条件を柔軟に切り替えてSKUを重ね合わせます。", icon="🔍")
    params = st.session_state.compare_params
    year_df = st.session_state.data_year
    end_m = end_month_selector(year_df, key="compare_end_month")

    snapshot = latest_yearsum_snapshot(year_df, end_m)
    snapshot["display_name"] = snapshot["product_name"].fillna(snapshot["product_code"])

    search = st.text_input("検索ボックス", "")
    if search:
        snapshot = snapshot[
            snapshot["display_name"].str.contains(search, case=False, na=False)
        ]
    # ---- 操作バー＋グラフ密着カード ----

    band_params = params.get("band_params", {})
    max_amount = int(snapshot["year_sum"].max()) if not snapshot.empty else 0
    low0 = int(
        band_params.get(
            "low_amount", int(snapshot["year_sum"].min()) if not snapshot.empty else 0
        )
    )
    high0 = int(band_params.get("high_amount", max_amount))

    st.markdown(
        """
<style>
.chart-card { position: relative; margin:.25rem 0 1rem; border-radius:12px;
  border:1px solid var(--color-primary); background:var(--card-bg,#fff); }
.chart-toolbar { position: sticky; top: -1px; z-index: 5;
  display:flex; gap:.6rem; flex-wrap:wrap; align-items:center;
  padding:.35rem .6rem; background: linear-gradient(180deg, rgba(0,58,112,.08), rgba(0,58,112,.02));
  border-bottom:1px solid var(--color-primary); }
/* Streamlit標準の下マージンを除去（ここが距離の主因） */
.chart-toolbar .stRadio, .chart-toolbar .stSelectbox, .chart-toolbar .stSlider,
.chart-toolbar .stMultiSelect, .chart-toolbar .stCheckbox { margin-bottom:0 !important; }
.chart-toolbar .stRadio > label, .chart-toolbar .stCheckbox > label { color:#003a70; }
.chart-toolbar .stSlider label { color:#003a70; }
.chart-body { padding:.15rem .4rem .4rem; }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<section class="chart-card" id="line-compare">', unsafe_allow_html=True
    )

    st.markdown('<div class="chart-toolbar">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.6, 1.1, 1.0, 0.9])
    with c1:
        period = st.radio(
            "期間", ["12ヶ月", "24ヶ月", "36ヶ月"], horizontal=True, index=1
        )
    with c2:
        node_mode = st.radio(
            "ノード表示",
            ["自動", "主要ノードのみ", "すべて", "非表示"],
            horizontal=True,
            index=0,
        )
    with c3:
        hover_mode = st.radio(
            "ホバー", ["個別", "同月まとめ"], horizontal=True, index=0
        )
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
            index=[
                "金額指定",
                "商品指定(2)",
                "パーセンタイル",
                "順位帯",
                "ターゲット近傍",
            ].index(params.get("band_mode", "金額指定")),
        )
    with c7:
        band_params = params.get("band_params", {})
        if band_mode == "金額指定" and not snapshot.empty:
            unit_scale, unit_label = choose_amount_slider_unit(max_amount)
            slider_max = int(
                math.ceil(max(max_amount, band_params.get("high_amount", high0)) / unit_scale)
            )
            slider_max = max(slider_max, 1)

            default_low = int(round(band_params.get("low_amount", low0) / unit_scale))
            default_high = int(round(band_params.get("high_amount", high0) / unit_scale))
            default_low = max(0, min(default_low, slider_max))
            default_high = max(default_low, min(default_high, slider_max))

            step = nice_slider_step(slider_max)

            low_scaled, high_scaled = st.slider(
                f"金額レンジ（{unit_label}単位）",
                min_value=0,
                max_value=slider_max,
                value=(default_low, default_high),
                step=step,
            )

            low = int(low_scaled * unit_scale)
            high = int(high_scaled * unit_scale)
            high = min(high, max_amount)
            low = min(low, high)

            st.caption(
                f"選択中: {format_int(low)}円 〜 {format_int(high)}円"
            )

            band_params = {"low_amount": low, "high_amount": high}
        elif band_mode == "商品指定(2)" and not snapshot.empty:
            opts = (
                snapshot["product_code"].fillna("")
                + " | "
                + snapshot["display_name"].fillna("")
            ).tolist()
            opts = [o for o in opts if o.strip() != "|"]
            prod_a = st.selectbox("商品A", opts, index=0)
            prod_b = st.selectbox("商品B", opts, index=1 if len(opts) > 1 else 0)
            band_params = {
                "prod_a": prod_a.split(" | ")[0],
                "prod_b": prod_b.split(" | ")[0],
            }
        elif band_mode == "パーセンタイル" and not snapshot.empty:
            p_low, p_high = band_params.get("p_low", 0), band_params.get("p_high", 100)
            p_low, p_high = st.slider("百分位(%)", 0, 100, (int(p_low), int(p_high)))
            band_params = {"p_low": p_low, "p_high": p_high}
        elif band_mode == "順位帯" and not snapshot.empty:
            max_rank = int(snapshot["rank"].max()) if not snapshot.empty else 1
            r_low, r_high = band_params.get("r_low", 1), band_params.get(
                "r_high", max_rank
            )
            r_low, r_high = st.slider("順位", 1, max_rank, (int(r_low), int(r_high)))
            band_params = {"r_low": r_low, "r_high": r_high}
        else:
            opts = (
                snapshot["product_code"] + " | " + snapshot["display_name"]
            ).tolist()
            tlabel = st.selectbox("基準商品", opts, index=0) if opts else ""
            tcode = tlabel.split(" | ")[0] if tlabel else ""
            by = st.radio("幅指定", ["金額", "%"], horizontal=True)
            if by == "金額":
                width_default = 100000
                width = int_input("幅", int(band_params.get("width", width_default)))
                band_params = {"target_code": tcode, "by": "amt", "width": int(width)}
            else:
                width_default = 0.1
                width = st.number_input(
                    "幅",
                    value=float(band_params.get("width", width_default)),
                    step=width_default / 10,
                )
                band_params = {"target_code": tcode, "by": "pct", "width": width}
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
        n_win = st.slider(
            "傾きウィンドウ（月）",
            0,
            12,
            6,
            1,
            help="0=自動（系列の全期間で判定）",
        )
    with c15:
        cmp_mode = st.radio("傾き条件", ["以上", "未満"], horizontal=True)
    with c16:
        thr_type = st.radio(
            "しきい値の種類", ["円/月", "%/月", "zスコア"], horizontal=True
        )
    with c17:
        if thr_type == "円/月":
            thr_val = int_input("しきい値", 0)
        else:
            thr_val = st.number_input("しきい値", value=0.0, step=0.01, format="%.2f")
    c18, c19, c20 = st.columns([1.6, 1.2, 1.8])
    with c18:
        sens = st.slider("形状抽出の感度", 0.0, 1.0, 0.5, 0.05)
    with c19:
        z_thr = st.slider("急勾配 zスコア", 0.0, 3.0, 0.0, 0.1)
    with c20:
        shape_pick = st.radio(
            "形状抽出",
            ["（なし）", "急勾配", "山（への字）", "谷（逆への字）"],
            horizontal=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

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
            snapshot.dropna(subset=["yoy"])
            .sort_values("yoy", ascending=False)
            .head(10)["product_code"]
            .tolist()
        )
    elif quick == "直近6M伸長上位":
        codes = top_growth_codes(year_df, end_m, window=6, top=10)

    snap = slopes_snapshot(year_df, n=n_win)
    if thr_type == "円/月":
        key, v = "slope_yen", float(thr_val)
    elif thr_type == "%/月":
        key, v = "slope_ratio", float(thr_val)
    else:
        key, v = "slope_z", float(thr_val)
    mask = (snap[key] >= v) if cmp_mode == "以上" else (snap[key] <= v)
    codes_by_slope = set(snap.loc[mask, "product_code"])

    eff_n = n_win if n_win > 0 else 12
    shape_df = shape_flags(
        year_df,
        window=max(6, eff_n * 2),
        alpha_ratio=0.02 * (1.0 - sens),
        amp_ratio=0.06 * (1.0 - sens),
    )
    codes_steep = set(snap.loc[snap["slope_z"].abs() >= z_thr, "product_code"])
    codes_mtn = set(shape_df.loc[shape_df["is_mountain"], "product_code"])
    codes_val = set(shape_df.loc[shape_df["is_valley"], "product_code"])
    shape_map = {
        "（なし）": None,
        "急勾配": codes_steep,
        "山（への字）": codes_mtn,
        "谷（逆への字）": codes_val,
    }
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

    main_codes = target_codes
    max_lines = 30
    if len(main_codes) > max_lines:
        top_order = (
            snapshot[snapshot["product_code"].isin(main_codes)]
            .sort_values("year_sum", ascending=False)["product_code"]
            .tolist()
        )
        main_codes = top_order[:max_lines]

    df_main = df_long[df_long["product_code"].isin(main_codes)]
    ai_on = st.toggle(
        "AIサマリー",
        value=False,
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )
    if ai_on and not df_main.empty:
        pos = len(codes_steep)
        mtn = len(codes_mtn & set(main_codes))
        val = len(codes_val & set(main_codes))
        explain = _ai_explain(
            {
                "対象SKU数": len(main_codes),
                "中央値(年計)": float(
                    snapshot_disp.loc[
                        snapshot_disp["product_code"].isin(main_codes), "year_sum_disp"
                    ].median()
                ),
                "急勾配数": pos,
                "山数": mtn,
                "谷数": val,
            }
        )
        st.info(f"**AI比較コメント**：{explain}")

    tb_common = dict(
        period=period,
        node_mode=node_mode,
        hover_mode=hover_mode,
        op_mode=op_mode,
        peak_on=peak_on,
        unit=unit,
        enable_avoid=enable_label_avoid,
        gap_px=label_gap_px,
        max_labels=label_max,
        alt_side=alternate_side,
        slope_conf=None,
        forecast_method="なし",
        forecast_window=12,
        forecast_horizon=6,
        forecast_k=2.0,
        forecast_robust=False,
        anomaly="OFF",
    )

    st.markdown('<div class="chart-body">', unsafe_allow_html=True)
    fig = build_chart_card(
        df_main,
        selected_codes=None,
        multi_mode=True,
        tb=tb_common,
        band_range=(low, high),
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</section>", unsafe_allow_html=True)

    st.caption(
        "凡例クリックで表示切替、ダブルクリックで単独表示。ドラッグでズーム/パン、右上メニューからPNG/CSV取得可。"
    )
    st.markdown(
        """
傾き（円/月）：直近 n ヶ月の回帰直線の傾き。+は上昇、−は下降。

%/月：傾き÷平均年計。規模によらず比較可能。

zスコア：全SKUの傾き分布に対する標準化。|z|≥1.5で急勾配の目安。

山/谷：前半と後半の平均変化率の符号が**＋→−（山）／−→＋（谷）かつ振幅が十分**。
"""
    )

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
        st.download_button(
            "PNGエクスポート",
            data=png_bytes,
            file_name=f"band_overlay_{end_m}.png",
            mime="image/png",
        )
    except Exception:
        pass

    with st.expander("分布（オプション）", expanded=False):
        hist_fig = apply_elegant_theme(
            hist_fig, theme=st.session_state.get("ui_theme", "dark")
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    # ---- Small Multiples ----
    df_nodes = df_main.iloc[0:0].copy()
    HALO = "#ffffff" if st.get_option("theme.base") == "dark" else "#222222"
    SZ = 6
    dtick = "M1"
    drag = {"ズーム": "zoom", "パン": "pan", "選択": "select"}[op_mode]

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
    ymax = (
        df_long[df_long["product_code"].isin(main_codes)]["year_sum"].max()
        / UNIT_MAP[unit]
        if share_y
        else None
    )
    for i, code in enumerate(page_codes):
        g = df_long[df_long["product_code"] == code]
        disp = g["display_name"].iloc[0] if not g.empty else code
        palette = fig.layout.colorway or px.colors.qualitative.Safe
        fig_s = px.line(
            g,
            x="month",
            y="year_sum",
            color_discrete_sequence=[palette[i % len(palette)]],
            custom_data=["display_name"],
        )
        fig_s.update_traces(
            mode="lines",
            line=dict(width=1.5),
            opacity=0.8,
            showlegend=False,
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>月：%{{x|%Y-%m}}<br>年計：%{{y:,.0f}} {unit}<extra></extra>",
        )
        fig_s.update_xaxes(tickformat="%Y-%m", dtick=dtick, title_text="月（YYYY-MM）")
        fig_s.update_yaxes(
            tickformat="~,d",
            range=[0, ymax] if ymax else None,
            title_text=f"売上 年計（{unit}）",
        )
        fig_s.update_layout(font=dict(family="Noto Sans JP, Meiryo, Arial", size=12))
        fig_s.update_layout(
            hoverlabel=dict(
                bgcolor="rgba(30,30,30,0.92)", font=dict(color="#fff", size=12)
            )
        )
        fig_s.update_layout(dragmode=drag)
        if hover_mode == "個別":
            fig_s.update_layout(hovermode="closest")
        else:
            fig_s.update_layout(hovermode="x unified", hoverlabel=dict(align="left"))
        last_val = (
            g.sort_values("month")["year_sum"].iloc[-1] / UNIT_MAP[unit]
            if not g.empty
            else np.nan
        )
        with cols[i % col_count]:
            st.metric(
                disp, f"{last_val:,.0f} {unit}" if not np.isnan(last_val) else "—"
            )
            fig_s = apply_elegant_theme(
                fig_s, theme=st.session_state.get("ui_theme", "dark")
            )
            st.plotly_chart(
                fig_s,
                use_container_width=True,
                height=225,
                config=PLOTLY_CONFIG,
            )

    # 5) SKU詳細
elif page == "SKU詳細":
    require_data()
    section_header("SKU 詳細", "個別SKUのトレンドとメモを一元管理。", icon="🗂️")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_detail")
    prods = (
        st.session_state.data_year[["product_code", "product_name"]]
        .drop_duplicates()
        .sort_values("product_code")
    )
    mode = st.radio("表示モード", ["単品", "複数比較"], horizontal=True)
    tb = toolbar_sku_detail(multi_mode=(mode == "複数比較"))
    df_year = st.session_state.data_year.copy()
    df_year["display_name"] = df_year["product_name"].fillna(df_year["product_code"])

    ai_on = st.toggle(
        "AIサマリー",
        value=False,
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    chart_rendered = False
    modal_codes: List[str] | None = None
    modal_is_multi = False

    if mode == "単品":
        prod_label = st.selectbox(
            "SKU選択", options=prods["product_code"] + " | " + prods["product_name"]
        )
        code = prod_label.split(" | ")[0]
        build_chart_card(
            df_year,
            selected_codes=[code],
            multi_mode=False,
            tb=tb,
            height=600,
        )
        chart_rendered = True
        modal_codes = [code]
        modal_is_multi = False

        g_y = df_year[df_year["product_code"] == code].sort_values("month")
        row = g_y[g_y["month"] == end_m]
        if not row.empty:
            rr = row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric(
                "年計", f"{int(rr['year_sum']) if not pd.isna(rr['year_sum']) else '—'}"
            )
            c2.metric(
                "YoY", f"{rr['yoy']*100:.1f} %" if not pd.isna(rr["yoy"]) else "—"
            )
            c3.metric("Δ", f"{int(rr['delta'])}" if not pd.isna(rr["delta"]) else "—")

        if ai_on and not row.empty:
            st.info(
                _ai_explain(
                    {
                        "年計": (
                            float(rr["year_sum"])
                            if not pd.isna(rr["year_sum"])
                            else 0.0
                        ),
                        "YoY": float(rr["yoy"]) if not pd.isna(rr["yoy"]) else 0.0,
                        "Δ": float(rr["delta"]) if not pd.isna(rr["delta"]) else 0.0,
                    }
                )
            )

        st.subheader("メモ / タグ")
        note = st.text_area(
            "メモ（保存で保持）", value=st.session_state.notes.get(code, ""), height=100
        )
        tags_str = st.text_input(
            "タグ（カンマ区切り）", value=",".join(st.session_state.tags.get(code, []))
        )
        c1, c2 = st.columns([1, 1])
        if c1.button("保存"):
            st.session_state.notes[code] = note
            st.session_state.tags[code] = [
                t.strip() for t in tags_str.split(",") if t.strip()
            ]
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
        sel = st.multiselect("SKU選択（最大60件）", options=opts, max_selections=60)
        codes = [s.split(" | ")[0] for s in sel]
        if codes or (tb.get("slope_conf") and tb["slope_conf"].get("quick") != "なし"):
            build_chart_card(
                df_year,
                selected_codes=codes,
                multi_mode=True,
                tb=tb,
                height=600,
            )
            chart_rendered = True
            modal_codes = codes
            modal_is_multi = True
            snap = latest_yearsum_snapshot(df_year, end_m)
            if codes:
                snap = snap[snap["product_code"].isin(codes)]
            if ai_on and not snap.empty:
                st.info(_ai_sum_df(snap[["year_sum", "yoy", "delta"]]))
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

    if tb.get("expand_mode") and chart_rendered:
        with st.modal("グラフ拡大モード", key="sku_expand_modal"):
            st.caption("操作パネルは拡大表示中も利用できます。")
            tb_modal = toolbar_sku_detail(
                multi_mode=modal_is_multi,
                key_prefix="sku_modal",
                include_expand_toggle=False,
            )
            build_chart_card(
                df_year,
                selected_codes=modal_codes,
                multi_mode=modal_is_multi,
                tb=tb_modal,
                height=tb_modal.get("chart_height", 760),
            )
            if st.button("閉じる", key="close_expand_modal"):
                st.session_state.setdefault("ui", {})["expand_mode"] = False
                st.session_state["sku_expand_mode"] = False
                st.experimental_rerun()

# 5) 異常検知
elif page == "異常検知":
    require_data()
    section_header("異常検知", "回帰残差ベースで異常ポイントを抽出します。", icon="🚨")
    year_df = st.session_state.data_year.copy()
    unit = st.session_state.settings.get("currency_unit", "円")
    scale = UNIT_MAP.get(unit, 1)

    col_a, col_b = st.columns([1.1, 1.1])
    with col_a:
        window = st.slider("学習窓幅（月）", 6, 18, st.session_state.get("anomaly_window", 12), key="anomaly_window")
    with col_b:
        score_method = st.radio("スコア基準", ["zスコア", "MADスコア"], horizontal=True, key="anomaly_score_method")

    if score_method == "zスコア":
        thr_key = "anomaly_thr_z"
        threshold = st.slider(
            "異常判定しきい値",
            2.0,
            5.0,
            value=float(st.session_state.get(thr_key, 3.0)),
            step=0.1,
            key=thr_key,
        )
        robust = False
    else:
        thr_key = "anomaly_thr_mad"
        threshold = st.slider(
            "異常判定しきい値",
            2.5,
            6.0,
            value=float(st.session_state.get(thr_key, 3.5)),
            step=0.1,
            key=thr_key,
        )
        robust = True

    prod_opts = (
        year_df[["product_code", "product_name"]]
        .drop_duplicates()
        .sort_values("product_code")
    )
    prod_opts["label"] = (
        prod_opts["product_code"]
        + " | "
        + prod_opts["product_name"].fillna(prod_opts["product_code"])
    )
    selected_labels = st.multiselect(
        "対象SKU（未選択=全件）",
        options=prod_opts["label"].tolist(),
        key="anomaly_filter_codes",
    )
    selected_codes = [lab.split(" | ")[0] for lab in selected_labels]

    records: List[pd.DataFrame] = []
    for code, g in year_df.groupby("product_code"):
        if selected_codes and code not in selected_codes:
            continue
        s = g.sort_values("month").set_index("month")["year_sum"]
        res = detect_linear_anomalies(
            s,
            window=int(window),
            threshold=float(threshold),
            robust=robust,
        )
        if res.empty:
            continue
        res["product_code"] = code
        res["product_name"] = g["product_name"].iloc[0]
        res = res.merge(
            g[["month", "year_sum", "yoy", "delta"]],
            on="month",
            how="left",
        )
        res["score_abs"] = res["score"].abs()
        records.append(res)

    if not records:
        st.success("異常値は検出されませんでした。窓幅やしきい値を調整してください。")
    else:
        anomalies = pd.concat(records, ignore_index=True)
        anomalies = anomalies.sort_values("score_abs", ascending=False)
        anomalies["year_sum_disp"] = anomalies["year_sum"] / scale
        anomalies["delta_disp"] = anomalies["delta"] / scale
        total_count = len(anomalies)
        sku_count = anomalies["product_code"].nunique()
        pos_cnt = int((anomalies["score"] > 0).sum())
        neg_cnt = int((anomalies["score"] < 0).sum())

        m1, m2, m3 = st.columns(3)
        m1.metric("異常件数", f"{total_count:,}")
        m2.metric("対象SKU", f"{sku_count:,}")
        m3.metric("上振れ/下振れ", f"{pos_cnt:,} / {neg_cnt:,}")

        max_top = min(200, total_count)
        top_default = min(50, max_top)
        top_n = int(
            st.slider(
                "表示件数",
                min_value=1,
                max_value=max_top,
                value=top_default,
                key="anomaly_view_top",
            )
        )
        view = anomalies.head(top_n).copy()
        view_table = view[
            [
                "product_code",
                "product_name",
                "month",
                "year_sum_disp",
                "yoy",
                "delta_disp",
                "score",
            ]
        ].rename(
            columns={
                "product_code": "商品コード",
                "product_name": "商品名",
                "month": "月",
                "year_sum_disp": f"年計({unit})",
                "yoy": "YoY",
                "delta_disp": f"Δ({unit})",
                "score": "スコア",
            }
        )
        st.dataframe(view_table, use_container_width=True)
        st.caption("値は指定した単位換算、スコアはローカル回帰残差の標準化値です。")
        st.download_button(
            "CSVダウンロード",
            data=view_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"anomalies_{score_method}_{threshold:.1f}.csv",
            mime="text/csv",
        )

        if st.toggle("AI異常サマリー", value=False, key="anomaly_ai_toggle") and not view.empty:
            ai_df = view[["product_name", "month", "score", "year_sum", "yoy", "delta"]].fillna(0)
            st.info(_ai_anomaly_report(ai_df))

        option_labels = [
            f"{row['product_code']}｜{row['product_name'] or row['product_code']}｜{row['month']}"
            for _, row in view.iterrows()
        ]
        if option_labels:
            sel_label = st.selectbox("詳細チャート", options=option_labels, key="anomaly_detail_select")
            code_sel, name_sel, month_sel = sel_label.split("｜")
            g = year_df[year_df["product_code"] == code_sel].sort_values("month").copy()
            g["year_sum_disp"] = g["year_sum"] / scale
            fig_anom = px.line(
                g,
                x="month",
                y="year_sum_disp",
                markers=True,
                title=f"{name_sel} 年計推移",
            )
            fig_anom.update_yaxes(title_text=f"年計（{unit}）", tickformat="~,d")
            fig_anom.update_traces(hovertemplate="月：%{x|%Y-%m}<br>年計：%{y:,.0f} {unit}<extra></extra>")

            code_anoms = anomalies[anomalies["product_code"] == code_sel]
            if not code_anoms.empty:
                fig_anom.add_scatter(
                    x=code_anoms["month"],
                    y=code_anoms["year_sum"] / scale,
                    mode="markers",
                    name="異常値",
                    marker=dict(color="#d94c53", size=10, symbol="triangle-up"),
                    hovertemplate="異常月：%{x|%Y-%m}<br>年計：%{y:,.0f} {unit}<br>スコア：%{customdata[0]:.2f}<extra></extra>",
                    customdata=np.stack([code_anoms["score"]], axis=-1),
                    showlegend=False,
                )
            target = code_anoms[code_anoms["month"] == month_sel]
            if not target.empty:
                tgt = target.iloc[0]
                fig_anom.add_annotation(
                    x=month_sel,
                    y=tgt["year_sum"] / scale,
                    text=f"スコア {tgt['score']:.2f}",
                    showarrow=True,
                    arrowcolor="#d94c53",
                    arrowhead=2,
                )
                yoy_txt = (
                    f"{tgt['yoy'] * 100:.1f}%" if tgt.get("yoy") is not None and not pd.isna(tgt.get("yoy")) else "—"
                )
                delta_txt = format_amount(tgt.get("delta"), unit)
                st.info(
                    f"{name_sel} {month_sel} の年計は {tgt['year_sum_disp']:.0f} {unit}、YoY {yoy_txt}、Δ {delta_txt}。"
                    f" 異常スコアは {tgt['score']:.2f} です。"
                )
            fig_anom = apply_elegant_theme(fig_anom, theme=st.session_state.get("ui_theme", "dark"))
            st.plotly_chart(fig_anom, use_container_width=True, config=PLOTLY_CONFIG)

# 6) 相関分析
elif page == "相関分析":
    require_data()
    section_header("相関分析", "指標間の関係性からインサイトを発掘。", icon="🧭")
    end_m = end_month_selector(st.session_state.data_year, key="corr_end_month")
    snapshot = latest_yearsum_snapshot(st.session_state.data_year, end_m)

    metric_opts = [
        "year_sum",
        "yoy",
        "delta",
        "slope_beta",
        "slope6m",
        "std6m",
        "hhi_share",
    ]
    metrics = st.multiselect(
        "指標",
        [m for m in metric_opts if m in snapshot.columns],
        default=[
            m
            for m in ["year_sum", "yoy", "delta", "slope_beta"]
            if m in snapshot.columns
        ],
    )
    method = st.radio(
        "相関の種類",
        ["pearson", "spearman"],
        horizontal=True,
        format_func=lambda x: "Pearson" if x == "pearson" else "Spearman",
    )
    winsor_pct = st.slider("外れ値丸め(%)", 0.0, 5.0, 1.0)
    log_enable = st.checkbox("ログ変換", value=False)
    r_thr = st.slider("相関 r 閾値（|r|≥）", 0.0, 1.0, 0.0, 0.05)

    ai_on = st.toggle(
        "AIサマリー",
        value=False,
        help="要約・コメント・自動説明を表示（オンデマンド計算）",
    )

    if metrics:
        df_plot = snapshot.copy()
        df_plot = winsorize_frame(df_plot, metrics, p=winsor_pct / 100)
        df_plot = maybe_log1p(df_plot, metrics, log_enable)
        tbl = corr_table(df_plot, metrics, method=method)
        tbl = tbl[abs(tbl["r"]) >= r_thr]

        st.subheader("相関の要点")
        for line in narrate_top_insights(tbl, NAME_MAP):
            st.write("・", line)
        sig_cnt = int((tbl["sig"] == "有意(95%)").sum())
        weak_cnt = int((tbl["r"].abs() < 0.2).sum())
        st.write(f"統計的に有意な相関: {sig_cnt} 組")
        st.write(f"|r|<0.2 の組み合わせ: {weak_cnt} 組")

        if ai_on and not tbl.empty:
            r_mean = float(tbl["r"].abs().mean())
            st.info(
                _ai_explain(
                    {
                        "有意本数": int((tbl["sig"] == "有意(95%)").sum()),
                        "平均|r|": r_mean,
                    }
                )
            )

        st.subheader("相関ヒートマップ")
        st.caption("右上=強い正、左下=強い負、白=関係薄")
        corr = df_plot[metrics].corr(method=method)
        fig_corr = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=True
        )
        fig_corr = apply_elegant_theme(
            fig_corr, theme=st.session_state.get("ui_theme", "dark")
        )
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
            fig_sc = px.scatter(
                df_xy, x=x_col, y=y_col, hover_data=["product_code", "product_name"]
            )
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
                fig_sc.add_annotation(
                    x=row[x_col], y=row[y_col], text=label, showarrow=True, arrowhead=1
                )
            fig_sc = apply_elegant_theme(
                fig_sc, theme=st.session_state.get("ui_theme", "dark")
            )
            st.plotly_chart(fig_sc, use_container_width=True, config=PLOTLY_CONFIG)
            st.caption("rは -1〜+1。0は関連が薄い。CIに0を含まなければ有意。")
            st.caption("散布図の点が右上・左下に伸びれば正、右下・左上なら負。")
    else:
        st.info("指標を選択してください。")

    with st.expander("相関の読み方"):
        st.write("正の相関：片方が大きいほどもう片方も大きい")
        st.write("負の相関：片方が大きいほどもう片方は小さい")
        st.write(
            "|r|<0.2は弱い、0.2-0.5はややあり、0.5-0.8は中~強、>0.8は非常に強い（目安）"
        )

# 6) アラート
elif page == "アラート":
    require_data()
    section_header("アラート", "閾値に該当したリスクSKUを自動抽出。", icon="⚠️")
    end_m = end_month_selector(st.session_state.data_year, key="end_month_alert")
    s = st.session_state.settings
    alerts = build_alerts(
        st.session_state.data_year,
        end_month=end_m,
        yoy_threshold=s["yoy_threshold"],
        delta_threshold=s["delta_threshold"],
        slope_threshold=s["slope_threshold"],
    )
    if alerts.empty:
        st.success("閾値に該当するアラートはありません。")
    else:
        st.dataframe(alerts, use_container_width=True)
        st.download_button(
            "CSVダウンロード",
            data=alerts.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"alerts_{end_m}.csv",
            mime="text/csv",
        )

# 6) 設定
elif page == "設定":
    section_header("設定", "年計計算条件や閾値を調整します。", icon="⚙️")
    s = st.session_state.settings
    c1, c2, c3 = st.columns(3)
    with c1:
        s["window"] = st.number_input(
            "年計ウィンドウ（月）",
            min_value=3,
            max_value=24,
            value=int(s["window"]),
            step=1,
        )
        s["last_n"] = st.number_input(
            "傾き算出の対象点数",
            min_value=3,
            max_value=36,
            value=int(s["last_n"]),
            step=1,
        )
    with c2:
        s["yoy_threshold"] = st.number_input(
            "YoY 閾値（<=）", value=float(s["yoy_threshold"]), step=0.01, format="%.2f"
        )
        s["delta_threshold"] = int_input("Δ 閾値（<= 円）", int(s["delta_threshold"]))
    with c3:
        s["slope_threshold"] = st.number_input(
            "傾き 閾値（<=）",
            value=float(s["slope_threshold"]),
            step=0.1,
            format="%.2f",
        )
        s["currency_unit"] = st.selectbox(
            "通貨単位表記",
            options=["円", "千円", "百万円"],
            index=["円", "千円", "百万円"].index(s["currency_unit"]),
        )

    st.caption("※ 設定変更後は再計算が必要です。")
    if st.button("年計の再計算を実行", type="primary"):
        if st.session_state.data_monthly is None:
            st.warning("先にデータを取り込んでください。")
        else:
            long_df = st.session_state.data_monthly
            year_df = compute_year_rolling(
                long_df, window=s["window"], policy=s["missing_policy"]
            )
            year_df = compute_slopes(year_df, last_n=s["last_n"])
            st.session_state.data_year = year_df
            st.success("再計算が完了しました。")

# 7) 保存ビュー
elif page == "保存ビュー":
    section_header("保存ビュー", "設定や比較条件をブックマーク。", icon="🔖")
    s = st.session_state.settings
    cparams = st.session_state.compare_params
    st.write("現在の設定・選択（閾値、ウィンドウ、単位など）を名前を付けて保存します。")

    name = st.text_input("ビュー名")
    if st.button("保存"):
        if not name:
            st.warning("ビュー名を入力してください。")
        else:
            st.session_state.saved_views[name] = {
                "settings": dict(s),
                "compare": dict(cparams),
            }
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
