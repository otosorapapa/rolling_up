"""Session state helpers shared across pages."""

from __future__ import annotations

import streamlit as st

from connectors import ConnectorConfig, SquareConnector
from utils.chat import LLMAssistant
from utils.database import create_db_and_tables, upsert_sales_records


def bootstrap_state(db_path: str = "data/sales.duckdb") -> None:
    """Ensure key session state entries exist."""

    if "detail_mode" not in st.session_state:
        st.session_state.detail_mode = False
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "sales_df" not in st.session_state:
        demo_connector = SquareConnector(ConnectorConfig(api_key="demo"))
        demo_data = demo_connector.fetch_sales()
        st.session_state.sales_df = demo_data
        create_db_and_tables(db_path)
        upsert_sales_records(demo_data, path=db_path)
    if "assistant" not in st.session_state:
        st.session_state.assistant = LLMAssistant()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = ""
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
