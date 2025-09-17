from __future__ import annotations

import datetime as dt

import streamlit as st

from connectors import (
    AirRegiConnector,
    ConnectorConfig,
    FreeeConnector,
    MoneyForwardConnector,
    ShopifyConnector,
    SquareConnector,
)
from utils.database import upsert_sales_records
from utils.scheduler import DataSyncScheduler
from utils.state import bootstrap_state
from utils.theme import apply_streamlit_theme, sidebar_mode_toggle

CONNECTOR_MAP = {
    "Square": SquareConnector,
    "Airレジ": AirRegiConnector,
    "freee": FreeeConnector,
    "Money Forward": MoneyForwardConnector,
    "Shopify": ShopifyConnector,
}


def _persist(df):
    upsert_sales_records(df)
    st.session_state.sales_df = df


def _sync(connector_cls, api_key: str, account_id: str | None = None):
    config = ConnectorConfig(api_key=api_key, account_id=account_id)
    connector = connector_cls(config)
    if not connector.authenticate():
        raise ValueError("APIキーが正しくありません。")
    df = connector.fetch_sales()
    _persist(df)
    return df


def main() -> None:
    st.set_page_config(page_title="データ連携設定", layout="wide")
    bootstrap_state()

    theme_mode = sidebar_mode_toggle()
    apply_streamlit_theme(theme_mode)

    st.title("データ連携とスケジューラ")
    st.caption("POSや会計ソフトと連携し、売上データを自動更新します。")

    if "scheduler" not in st.session_state:
        st.session_state.scheduler = DataSyncScheduler()
        st.session_state.scheduler_jobs = []
    else:
        st.session_state.scheduler.run_pending()

    connector_name = st.selectbox("連携サービス", options=list(CONNECTOR_MAP.keys()), help="利用中のサービスを選択してください。")
    connector_cls = CONNECTOR_MAP[connector_name]

    api_key = st.text_input("APIキー", type="password", help="各サービスで発行したAPIキーを入力します。")
    account_id = st.text_input("アカウントID", help="必要な場合のみ入力。Squareなど複数店舗をお持ちの場合に指定します。")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("今すぐ同期", use_container_width=True):
            if not api_key:
                st.error("APIキーを入力してください。")
            else:
                try:
                    df = _sync(connector_cls, api_key, account_id or None)
                    st.success(f"{connector_name}から{len(df)}件のレコードを取得しました。")
                    st.session_state.last_sync = dt.datetime.now()
                except Exception as exc:
                    st.error(f"同期に失敗しました: {exc}")
    with col2:
        interval = st.selectbox(
            "自動更新間隔",
            options=[15, 30, 60, 120],
            format_func=lambda m: f"{m}分",
            help="スケジューラが定期的にAPIを呼び出します。",
        )
        if st.button("スケジュールを登録", use_container_width=True):
            if not api_key:
                st.error("APIキーを入力してください。")
            else:
                scheduler: DataSyncScheduler = st.session_state.scheduler
                job = scheduler.schedule(
                    connector_cls(ConnectorConfig(api_key=api_key, account_id=account_id or None)),
                    minutes=interval,
                    callback=_persist,
                )
                st.session_state.scheduler_jobs.append({
                    "service": connector_name,
                    "interval": interval,
                    "next_run": job.next_run,
                })
                st.success("スケジュールを登録しました。画面操作のたびにペンディングを処理します。")

    st.subheader("登録済みスケジュール")
    if not st.session_state.scheduler_jobs:
        st.info("登録済みのスケジュールはありません。")
    else:
        rows = []
        for job in st.session_state.scheduler_jobs:
            rows.append({
                "サービス": job["service"],
                "間隔": f"{job['interval']}分",
                "次回実行予定": job["next_run"].strftime("%Y-%m-%d %H:%M"),
            })
        st.table(rows)

    last_sync = st.session_state.get("last_sync")
    if last_sync:
        st.caption(f"最終同期: {last_sync.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
