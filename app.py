from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.anomaly import ANOMALY_PRESETS, detect_anomalies
from utils.data_loader import build_sales_dataframe, load_supported_file
from utils.database import upsert_sales_records
from utils.forms import render_mapping_form
from utils.state import bootstrap_state
from utils.summary import generate_summary
from utils.theme import apply_streamlit_theme, sidebar_mode_toggle

APP_TITLE = "12カ月移動累計ダッシュボード"
DB_PATH = "data/sales.duckdb"


def _load_uploaded_data(uploaded_file: io.BytesIO) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv") or name.endswith(".txt"):
        return load_supported_file(uploaded_file, source="csv")
    return load_supported_file(uploaded_file, source="excel")


def _aggregate_monthly(df: pd.DataFrame, window: int) -> pd.DataFrame:
    monthly = df.groupby("month", as_index=False)["amount"].sum().sort_values("month")
    monthly["month_start"] = pd.to_datetime(monthly["month"], format="%Y-%m")
    monthly["rolling"] = monthly["amount"].rolling(window=window, min_periods=1).sum()
    monthly["rolling_delta"] = monthly["rolling"].diff()
    monthly["rolling_yoy"] = monthly["rolling"] / monthly["rolling"].shift(window) - 1
    return monthly


def _compute_slope(series: pd.Series, window: int) -> float:
    valid = series.dropna().tail(window)
    if len(valid) < 2:
        return 0.0
    x = np.arange(len(valid))
    y = valid.values
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    bootstrap_state(DB_PATH)

    st.title(APP_TITLE)
    st.caption("複数ソースの売上データを統合し、意思決定に役立つインサイトを提示します。")

    theme_mode = sidebar_mode_toggle()
    apply_streamlit_theme(theme_mode)

    st.sidebar.markdown("### モード設定")
    detail_toggle = st.sidebar.toggle(
        "詳細モード",
        value=st.session_state.detail_mode,
        help="高度なフィルターやしきい値を操作する場合に使用します。",
    )
    st.session_state.detail_mode = detail_toggle

    with st.sidebar.expander("データアップロード", expanded=True):
        st.markdown("CSV/Excel/Googleスプレッドシートに対応しています。")
        upload = st.file_uploader("ファイルを選択", type=["csv", "xlsx", "xls", "txt"])
        gsheet_url = st.text_input("GoogleスプレッドシートURL", help="共有リンクを貼り付けてください。")
        if upload is not None:
            try:
                st.session_state.raw_df = _load_uploaded_data(upload)
                st.success("ファイルを読み込みました。下部で列マッピングを確認してください。")
            except Exception as exc:  # pragma: no cover - user input dependent
                st.error(f"読み込みに失敗しました: {exc}")
        if gsheet_url:
            try:
                st.session_state.raw_df = load_supported_file(gsheet_url, source="google")
                st.success("Googleスプレッドシートを読み込みました。")
            except Exception as exc:  # pragma: no cover
                st.error(f"スプレッドシートの取得に失敗しました: {exc}")

    advanced_params: Dict[str, float] = {"window": 12, "slope_window": 6}
    anomaly_config = {"method": "zscore", "threshold": 2.5}
    if st.session_state.detail_mode:
        with st.sidebar.expander("高度パラメータ", expanded=True):
            advanced_params["window"] = st.number_input(
                "移動累計の窓幅 (月)",
                min_value=6,
                max_value=24,
                value=12,
                help="12ヶ月ローリングで年計を算出するのが基本ですが、短期トレンド確認にも調整可能です。",
            )
            advanced_params["slope_window"] = st.slider(
                "傾き計算期間",
                min_value=3,
                max_value=12,
                value=6,
                help="直近nポイントの線形回帰で傾きを算出します。",
            )
            preset_name = st.selectbox(
                "アラートプリセット",
                options=list(ANOMALY_PRESETS.keys()) + ["カスタム"],
                help="典型的な異常検知条件を選択できます。",
            )
            if preset_name == "カスタム":
                method = st.selectbox("検知手法", options=["zscore", "isolation_forest", "seasonal_esd"], help="データ特性に応じて検知ロジックを選択")
                if method == "zscore":
                    threshold = st.slider("Zスコア閾値", min_value=-4.0, max_value=4.0, value=-2.5, step=0.1)
                    anomaly_config = {"method": method, "threshold": threshold}
                elif method == "isolation_forest":
                    contamination = st.slider("異常割合推定", min_value=0.01, max_value=0.4, value=0.1, step=0.01)
                    anomaly_config = {"method": method, "contamination": contamination}
                else:
                    alpha = st.slider("有意水準", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
                    anomaly_config = {"method": method, "alpha": alpha}
            else:
                anomaly_config = ANOMALY_PRESETS[preset_name]
    else:
        st.sidebar.caption("基本モードでは主要指標と簡易フィルターのみ表示されます。")

    if st.session_state.raw_df is not None:
        st.subheader("アップロードデータの確認")
        st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
        mapping = render_mapping_form(st.session_state.raw_df)
        if mapping is not None:
            try:
                sales_df = build_sales_dataframe(st.session_state.raw_df, mapping)
                st.session_state.sales_df = sales_df
                upsert_sales_records(sales_df, path=DB_PATH)
                st.success("データベースを更新しました。各ページから分析が可能です。")
            except Exception as exc:
                st.error(f"データ整形に失敗しました: {exc}")

    sales_df = st.session_state.sales_df
    monthly = _aggregate_monthly(sales_df, int(advanced_params["window"]))
    slope = _compute_slope(monthly["rolling"], int(advanced_params["slope_window"]))

    current_row = monthly.iloc[-1]
    current_total = current_row["rolling"]
    if pd.isna(current_total):
        current_total = current_row["amount"]
    current_total = float(current_total)
    delta_raw = current_row.get("rolling_delta")
    delta = float(delta_raw) if delta_raw is not None and not pd.isna(delta_raw) else 0.0
    yoy = current_row.get("rolling_yoy")
    if yoy is not None and (pd.isna(yoy) or not np.isfinite(yoy)):
        yoy = None
    timeframe = current_row["month"]

    fig = px.line(
        monthly,
        x="month_start",
        y="rolling",
        markers=True,
        labels={"rolling": "12カ月累計", "month_start": "月"},
        title="年計トレンド",
    )
    fig.update_traces(hovertemplate="%{x|%Y-%m}: %{y:,.0f}円")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with col2:
        st.metric("年計売上", f"{current_total:,.0f}円", help="移動12ヶ月分の総売上です。")
        st.metric(
            "前年比",
            f"{yoy * 100:.1f}%" if yoy is not None else "―",
            help="同月1年前の年計と比較した成長率。",
        )
        st.metric(
            "前月差",
            f"{delta:,.0f}円",
            help="年計の増減額。プラスなら改善傾向です。",
        )
        st.metric("傾きβ", f"{slope:.2f}", help="直近期間のトレンド勾配。プラスで上向き。")

    anomaly_method = anomaly_config.get("method", "zscore")
    anomaly_args = {k: v for k, v in anomaly_config.items() if k != "method"}
    anomaly_source = monthly.dropna(subset=["rolling"]).copy()
    anomaly_result = detect_anomalies(anomaly_source, value_col="rolling", method=anomaly_method, **anomaly_args)
    anomaly_source["score"] = anomaly_result["score"].values
    anomaly_source["is_anomaly"] = anomaly_result["is_anomaly"].values
    anomalies = anomaly_source[anomaly_source["is_anomaly"]]

    group_cols = [col for col in ["product_code", "product_name"] if col in sales_df.columns]
    if group_cols:
        top_products = (
            sales_df.groupby(group_cols, dropna=False)["amount"].sum().reset_index().sort_values("amount", ascending=False)
        )
        if "product_name" not in top_products.columns:
            top_products["product_name"] = top_products[group_cols[0]]
    else:
        top_products = monthly[["month", "rolling"]].rename(columns={"month": "product_name", "rolling": "amount"})

    summary_text = generate_summary(
        {"total": current_total, "yoy": yoy, "delta": delta, "slope": slope},
        timeframe=timeframe,
        highlights=top_products,
        anomalies=anomalies,
    )
    st.session_state.summary_text = summary_text

    with st.expander("AIサマリー", expanded=False):
        st.write(summary_text)

    st.subheader("異常シグナル")
    if anomalies.empty:
        st.info("しきい値を超える異常はありません。")
    else:
        st.dataframe(
            anomalies[["month", "rolling", "score"]],
            use_container_width=True,
        )

    st.subheader("AIチャット")
    st.caption("例: 今週の売上低下の原因を教えて")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("質問を入力")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        answer = st.session_state.assistant.ask(prompt, st.session_state.summary_text)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)


if __name__ == "__main__":
    main()
