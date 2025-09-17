from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.forecast import forecast_chart, forecast_sales, prepare_time_series
from utils.state import bootstrap_state
from utils.summary import generate_summary
from utils.theme import apply_streamlit_theme, sidebar_mode_toggle


def main() -> None:
    st.set_page_config(page_title="需要予測", layout="wide")
    bootstrap_state()

    theme_mode = sidebar_mode_toggle()
    apply_streamlit_theme(theme_mode)

    st.title("需要予測")
    st.caption("過去実績から将来の売上推移を推計し、在庫や資金繰りの意思決定に活用します。")

    sales_df = st.session_state.get("sales_df")
    if sales_df is None or sales_df.empty:
        st.warning("データが読み込まれていません。トップページでデータをアップロードしてください。")
        return

    horizon = st.sidebar.slider(
        "予測期間 (月)",
        min_value=3,
        max_value=24,
        value=6,
        help="需要計画の期間に合わせて予測対象を選択します。短期は在庫調整、長期は投資判断に活用できます。",
    )

    history = prepare_time_series(sales_df)
    forecast = forecast_sales(sales_df, periods=horizon)
    fig = forecast_chart(history.rename(columns={"ds": "month_start", "y": "amount"}), forecast)
    st.plotly_chart(fig, use_container_width=True)

    future = forecast.tail(horizon)
    latest = future.iloc[-1]
    metrics = {
        "total": float(latest["yhat"]),
        "delta": float(latest["yhat"] - future.iloc[0]["yhat"]),
        "slope": float((future["yhat"].iloc[-1] - future["yhat"].iloc[0]) / max(horizon - 1, 1)),
    }
    timeframe = latest["ds"].strftime("%Y-%m")
    summary = generate_summary(metrics, timeframe=timeframe)

    with st.expander("AIサマリー", expanded=False):
        st.write(summary)

    st.subheader("予測数値一覧")
    display_df = forecast.copy()
    display_df["ds"] = display_df["ds"].dt.strftime("%Y-%m")
    st.dataframe(display_df, use_container_width=True)


if __name__ == "__main__":
    main()
