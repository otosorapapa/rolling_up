from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.alerts import load_alert_settings, save_alert_settings, send_alert
from utils.state import bootstrap_state
from utils.theme import apply_streamlit_theme, sidebar_mode_toggle

SETTINGS_PATH = "settings.yaml"


def _compute_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    months = sorted(df["month"].unique())
    if len(months) < 2:
        return pd.DataFrame()
    latest = months[-1]
    prev = months[-2]
    group_cols = [col for col in ["product_code", "product_name"] if col in df.columns]
    if not group_cols:
        group_cols = ["month"]
    current = df[df["month"] == latest].groupby(group_cols, dropna=False)["amount"].sum()
    previous = df[df["month"] == prev].groupby(group_cols, dropna=False)["amount"].sum()
    combined = pd.DataFrame({"current": current, "previous": previous}).fillna(0.0)
    combined["delta"] = combined["current"] - combined["previous"]
    combined["yoy"] = np.where(combined["previous"] == 0, np.nan, combined["delta"] / combined["previous"])

    recent_months = months[-3:]
    pivot = (
        df[df["month"].isin(recent_months)]
        .pivot_table(index=group_cols, columns="month", values="amount", fill_value=0.0)
    )
    slopes = {}
    for idx, row in pivot.iterrows():
        values = row.values
        if len(values) < 2:
            slopes[idx] = 0.0
        else:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            slopes[idx] = float(slope)
    slope_series = pd.Series(slopes)
    combined["slope"] = slope_series
    combined.reset_index(inplace=True)
    combined["month"] = latest
    return combined


def main() -> None:
    st.set_page_config(page_title="アラート管理", layout="wide")
    bootstrap_state()

    theme_mode = sidebar_mode_toggle()
    apply_streamlit_theme(theme_mode)

    st.title("アラートと通知")
    st.caption("閾値に基づくアラートを一覧化し、メールやSlackで通知できます。")

    sales_df = st.session_state.get("sales_df")
    if sales_df is None or sales_df.empty:
        st.warning("データが読み込まれていません。")
        return

    settings = load_alert_settings(SETTINGS_PATH)
    thresholds = settings.get("thresholds", {})

    with st.sidebar.expander("閾値設定", expanded=True):
        yoy_threshold = st.number_input(
            "YoY閾値", value=float(thresholds.get("yoy", -0.1)), step=0.01,
            help="前年同月比がこの値を下回るとアラート。負の値ほど厳しい条件です。",
        )
        delta_threshold = st.number_input(
            "前月差閾値", value=float(thresholds.get("delta", -300000)), step=50000.0,
            help="前月比の減少額が閾値を下回ると通知します。",
        )
        slope_threshold = st.number_input(
            "傾き閾値", value=float(thresholds.get("slope", -1.0)), step=0.5,
            help="直近トレンドの傾きがマイナス方向に大きい場合に検知。",
        )
        if st.button("設定を保存"):
            settings["thresholds"] = {
                "yoy": yoy_threshold,
                "delta": delta_threshold,
                "slope": slope_threshold,
            }
            save_alert_settings(SETTINGS_PATH, settings)
            st.success("設定を保存しました。")

    metrics_df = _compute_product_metrics(sales_df)
    if metrics_df.empty:
        st.info("比較できる月数が足りません。")
        return

    alerts = []
    for _, row in metrics_df.iterrows():
        name = row.get("product_name") or row.get("product_code") or "全体"
        if row["yoy"] <= yoy_threshold:
            alerts.append({"product_name": name, "metric": "yoy", "actual": row["yoy"], "month": row["month"]})
        if row["delta"] <= delta_threshold:
            alerts.append({"product_name": name, "metric": "delta", "actual": row["delta"], "month": row["month"]})
        if row["slope"] <= slope_threshold:
            alerts.append({"product_name": name, "metric": "slope", "actual": row["slope"], "month": row["month"]})

    alert_df = pd.DataFrame(alerts)
    st.subheader("アラート一覧")
    if alert_df.empty:
        st.info("現在の閾値ではアラートは発生していません。")
    else:
        st.table(alert_df)

    st.subheader("通知送信")
    dry_run = not st.checkbox("実際に通知を送信", value=False, help="テスト時はOFFのまま乾式送信を行います。")
    if st.button("メール/Slack通知を送信"):
        result = send_alert(alert_df, settings, dry_run=dry_run)
        st.success(f"通知処理を実行しました。メッセージ:\n{result['message']}")


if __name__ == "__main__":
    main()
