"""DuckDB storage helpers without requiring SQLModel."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import List, Optional

import duckdb
import pandas as pd


_SCHEMA_INITIALIZED: set[str] = set()
_SCHEMA_LOCK = threading.Lock()


def get_connection(path: str = "data/sales.duckdb") -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection for the given path."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def create_db_and_tables(path: str = "data/sales.duckdb") -> None:
    """Ensure that the sales table exists."""

    with _SCHEMA_LOCK:
        if path in _SCHEMA_INITIALIZED:
            return
        con = get_connection(path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sales_records (
                    month_key TEXT NOT NULL,
                    month DATE NOT NULL,
                    product_code TEXT,
                    product_name TEXT,
                    category TEXT,
                    customer_id TEXT,
                    amount DOUBLE NOT NULL,
                    quantity DOUBLE
                )
                """
            )
        finally:
            con.close()
        _SCHEMA_INITIALIZED.add(path)


def _to_date(value: object) -> date:
    """Convert a value to a Python ``date``."""

    parsed = pd.to_datetime(value)
    if pd.isna(parsed):
        raise ValueError("Invalid month value")
    return parsed.date()


def _safe_get(row: pd.Series, key: str) -> Optional[object]:
    """Return a value from a Series with ``None`` for missing/NaN."""

    if key not in row.index:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    return value


def upsert_sales_records(df: pd.DataFrame, path: str = "data/sales.duckdb") -> int:
    """Insert or update sales rows returning affected count."""

    if df.empty:
        return 0

    create_db_and_tables(path)
    con = get_connection(path)
    try:
        has_month_start = "month_start" in df.columns
        prepared: List[dict[str, object]] = []

        for _, row in df.iterrows():
            month_key = str(row["month"])
            if has_month_start and "month_start" in row.index and pd.notna(row["month_start"]):
                month_value = _to_date(row["month_start"])
            else:
                month_value = _to_date(row["month"])
            product_code = _safe_get(row, "product_code")
            product_name = _safe_get(row, "product_name")
            category = _safe_get(row, "category")
            customer_id = _safe_get(row, "customer_id")
            quantity_value = _safe_get(row, "quantity")
            quantity = float(quantity_value) if quantity_value is not None else 0.0
            amount = float(row["amount"])
            prepared.append(
                {
                    "month_key": month_key,
                    "month": month_value,
                    "product_code": product_code,
                    "product_name": product_name,
                    "category": category,
                    "customer_id": customer_id,
                    "amount": amount,
                    "quantity": quantity,
                }
            )

        con.execute("BEGIN TRANSACTION")
        try:
            for record in prepared:
                product_code = record["product_code"]
                con.execute(
                    """
                    DELETE FROM sales_records
                    WHERE month_key = ?
                      AND (
                            (product_code IS NULL AND ? IS NULL)
                            OR product_code = ?
                      )
                    """,
                    [record["month_key"], product_code, product_code],
                )
                con.execute(
                    """
                    INSERT INTO sales_records (
                        month_key,
                        month,
                        product_code,
                        product_name,
                        category,
                        customer_id,
                        amount,
                        quantity
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        record["month_key"],
                        record["month"],
                        record["product_code"],
                        record["product_name"],
                        record["category"],
                        record["customer_id"],
                        record["amount"],
                        record["quantity"],
                    ],
                )
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        con.close()

    return len(prepared)


def fetch_monthly_totals(path: str = "data/sales.duckdb", limit: int = 24) -> pd.DataFrame:
    """Return aggregated totals for the latest months."""

    create_db_and_tables(path)
    con = get_connection(path)
    try:
        df = con.execute(
            """
            SELECT month_key AS month, SUM(amount) AS amount
            FROM sales_records
            GROUP BY month_key
            ORDER BY month_key
            """
        ).df()
    finally:
        con.close()
    if limit:
        df = df.tail(limit).reset_index(drop=True)
    if not df.empty:
        df["month_start"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    return df


_ALLOWED_SEGMENT_COLUMNS = {"product_code", "product_name", "category", "customer_id"}


def fetch_segment(path: str, column: str) -> pd.DataFrame:
    """Aggregate amount by a specific column (e.g. category)."""

    if column not in _ALLOWED_SEGMENT_COLUMNS:
        raise ValueError(f"Unsupported segment column: {column}")

    create_db_and_tables(path)
    con = get_connection(path)
    try:
        df = con.execute(
            f"""
            SELECT {column} AS value, SUM(amount) AS amount
            FROM sales_records
            WHERE {column} IS NOT NULL
            GROUP BY {column}
            ORDER BY amount DESC
            """
        ).df()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame(columns=[column, "amount"])
    df = df.rename(columns={"value": column})
    return df
