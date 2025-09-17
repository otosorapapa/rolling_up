import io

import pandas as pd
import pytest

from utils.data_loader import (
    ColumnMapping,
    build_sales_dataframe,
    detect_column_candidates,
    normalise_month_series,
)


def test_normalise_month_series_handles_various_formats():
    values = ["2024-01", "2024/02/15", "2024年03月", "2024-04-01"]
    series = normalise_month_series(values)
    assert series.tolist() == ["2024-01", "2024-02", "2024-03", "2024-04"]


def test_detect_column_candidates_prefers_known_aliases():
    df = pd.DataFrame({"売上日": ["2024-01-01"], "売上金額": [100], "商品名": ["A"]})
    detected = detect_column_candidates(df)
    assert detected["date"] == "売上日"
    assert detected["value"] == "売上金額"
    assert detected["product_name"] == "商品名"


def test_build_sales_dataframe_with_wide_format():
    df = pd.DataFrame({"商品名": ["A"], "2024-01": [100], "2024-02": [200]})
    mapping = ColumnMapping(date="2024-01", value="2024-01", product_name="商品名")
    result = build_sales_dataframe(df, mapping)
    assert set(result.columns) >= {"month", "amount", "product_name"}
    assert len(result) == 2
    assert result.loc[result["month"] == "2024-02", "amount"].iloc[0] == 200


def test_build_sales_dataframe_requires_date_and_value():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError):
        build_sales_dataframe(df, ColumnMapping(date="A", value=""))
