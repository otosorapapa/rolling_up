"""Utility package for the Streamlit dashboard."""

from .data_loader import (
    load_supported_file,
    detect_column_candidates,
    normalise_month_series,
    build_sales_dataframe,
)
from .summary import generate_summary

__all__ = [
    "load_supported_file",
    "detect_column_candidates",
    "normalise_month_series",
    "build_sales_dataframe",
    "generate_summary",
]
