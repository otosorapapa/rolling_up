"""Reusable Streamlit forms for the dashboard."""

from __future__ import annotations

from typing import Dict, Optional

import streamlit as st

from .data_loader import ColumnMapping, detect_column_candidates

REQUIRED_FIELDS = {
    "date": "年月（例: 2024-01）",
    "value": "売上金額",
}

OPTIONAL_FIELDS = {
    "product_name": "商品名（任意）",
    "product_code": "商品コード（任意）",
    "category": "カテゴリ（任意）",
    "customer_id": "顧客ID（RFM分析用任意）",
    "quantity": "販売数量（任意）",
}


def render_mapping_form(df) -> Optional[ColumnMapping]:
    """Render a column mapping form returning ``ColumnMapping``."""

    detected = detect_column_candidates(df)
    columns = ["--"] + df.columns.tolist()
    with st.form("column_mapping"):
        st.markdown("### 列マッピング")
        st.caption("必須項目は太字で表示されます。正しい列を選択してください。")
        selections: Dict[str, Optional[str]] = {}
        for key, label in REQUIRED_FIELDS.items():
            selections[key] = st.selectbox(
                f"**{label}**",
                options=columns,
                index=columns.index(detected.get(key)) if detected.get(key) in columns else 0,
                help="売上集計の基準となる必須項目です。",
            )
        for key, label in OPTIONAL_FIELDS.items():
            selections[key] = st.selectbox(
                label,
                options=columns,
                index=columns.index(detected.get(key)) if detected.get(key) in columns else 0,
                help="分析を充実させる任意項目です。",
            )
        submitted = st.form_submit_button("データを確定")
    if not submitted:
        return None
    mapping_dict = {k: v for k, v in selections.items() if v and v != "--"}
    missing = [k for k in REQUIRED_FIELDS if k not in mapping_dict]
    if missing:
        st.warning("必須項目をすべて選択してください。")
        return None
    return ColumnMapping(**mapping_dict)
