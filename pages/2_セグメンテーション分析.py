from __future__ import annotations

import streamlit as st

from utils.segmentation import abc_analysis, correlation_heatmap, rfm_analysis, scatter_matrix
from utils.state import bootstrap_state
from utils.summary import generate_summary
from utils.theme import apply_streamlit_theme, sidebar_mode_toggle


def main() -> None:
    st.set_page_config(page_title="セグメンテーション分析", layout="wide")
    bootstrap_state()

    theme_mode = sidebar_mode_toggle()
    apply_streamlit_theme(theme_mode)

    st.title("セグメンテーション分析")
    st.caption("顧客・商品単位での貢献度や相関を把握し、重点施策を設計します。")

    df = st.session_state.get("sales_df")
    if df is None or df.empty:
        st.warning("データが読み込まれていません。トップページでデータをアップロードしてください。")
        return

    tabs = st.tabs(["RFM分析", "ABC分析", "相関・散布図"])

    with tabs[0]:
        st.subheader("RFM分析")
        customer_options = [col for col in df.columns if "id" in col.lower() or "customer" in col.lower()]
        if not customer_options:
            st.info("顧客ID列が見つかりませんでした。列マッピングで顧客IDを指定すると分析できます。")
        else:
            customer_col = st.selectbox("顧客ID列", options=customer_options, help="RFMでは顧客単位の再来率と価値を評価します。")
            date_col = "month_start" if "month_start" in df.columns else "month"
            amount_col = "amount"
            rfm_result = rfm_analysis(df, customer_col=customer_col, date_col=date_col, amount_col=amount_col)
            st.dataframe(rfm_result.dataframe, use_container_width=True)
            summary = rfm_result.dataframe["RFM"].value_counts().to_frame(name="件数")
            st.bar_chart(summary)
            with st.expander("AIサマリー", expanded=False):
                latest_month = df["month"].max()
                summary_text = generate_summary({"total": float(df[amount_col].sum())}, timeframe=str(latest_month))
                st.write(summary_text)

    with tabs[1]:
        st.subheader("カテゴリ別ABC分析")
        group_options = [col for col in ["product_name", "product_code", "category"] if col in df.columns]
        if not group_options:
            st.info("商品名やカテゴリの列が必要です。")
        else:
            group_col = st.selectbox("分類列", options=group_options, help="累積寄与率を基にA/B/Cランクを判定します。")
            abc_df = abc_analysis(df, value_col="amount", group_col=group_col)
            st.dataframe(abc_df, use_container_width=True)
            st.caption("A:売上上位80%、B:次の15%、C:残りのロングテール。")

    with tabs[2]:
        st.subheader("相関ヒートマップと散布図")
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        selection = st.multiselect(
            "指標を選択",
            options=numeric_cols,
            default=[col for col in ["amount", "quantity"] if col in numeric_cols],
            help="関連性を確認したい指標を複数選択してください。",
        )
        if len(selection) < 2:
            st.info("2つ以上の数値列を選択してください。")
        else:
            heatmap = correlation_heatmap(df, selection)
            st.plotly_chart(heatmap, use_container_width=True)
            scatter = scatter_matrix(df, selection)
            st.plotly_chart(scatter, use_container_width=True)


if __name__ == "__main__":
    main()
