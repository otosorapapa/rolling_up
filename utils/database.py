"""DuckDB storage helpers built on top of SQLModel."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import func


class SalesRecord(SQLModel, table=True):
    """Represents a single monthly sales observation."""

    id: Optional[int] = Field(default=None, primary_key=True)
    month_key: str = Field(index=True)
    month: date
    product_code: Optional[str] = Field(default=None, index=True)
    product_name: Optional[str] = None
    category: Optional[str] = None
    customer_id: Optional[str] = Field(default=None, index=True)
    amount: float
    quantity: Optional[float] = None


_ENGINE_CACHE = {}


def get_engine(path: str = "data/sales.duckdb"):
    """Return (and memoise) a DuckDB SQLModel engine."""

    if path not in _ENGINE_CACHE:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _ENGINE_CACHE[path] = create_engine(f"duckdb:///{path}")
    return _ENGINE_CACHE[path]


def create_db_and_tables(path: str = "data/sales.duckdb") -> None:
    engine = get_engine(path)
    SQLModel.metadata.create_all(engine)


def upsert_sales_records(df: pd.DataFrame, path: str = "data/sales.duckdb") -> int:
    """Insert or update sales rows returning affected count."""

    if df.empty:
        return 0
    engine = get_engine(path)
    create_db_and_tables(path)
    records = []
    has_month_start = "month_start" in df.columns
    for _, row in df.iterrows():
        month_value = (
            pd.to_datetime(row["month_start"]).date()
            if has_month_start and pd.notna(row.get("month_start"))
            else pd.to_datetime(row["month"]).date()
        )
        record = SalesRecord(
            month_key=row["month"],
            month=month_value,
            product_code=row.get("product_code"),
            product_name=row.get("product_name"),
            category=row.get("category"),
            customer_id=row.get("customer_id"),
            amount=float(row["amount"]),
            quantity=float(row.get("quantity", 0) or 0),
        )
        records.append(record)

    with Session(engine) as session:
        affected = 0
        for record in records:
            stmt = select(SalesRecord).where(SalesRecord.month_key == record.month_key)
            if record.product_code:
                stmt = stmt.where(SalesRecord.product_code == record.product_code)
            else:
                stmt = stmt.where(SalesRecord.product_code.is_(None))
            existing = session.exec(stmt).first()
            if existing:
                existing.amount = record.amount
                existing.quantity = record.quantity
                existing.product_name = record.product_name
                existing.category = record.category
                existing.customer_id = record.customer_id
            else:
                session.add(record)
            affected += 1
        session.commit()
    return affected


def fetch_monthly_totals(path: str = "data/sales.duckdb", limit: int = 24) -> pd.DataFrame:
    """Return aggregated totals for the latest months."""

    engine = get_engine(path)
    with Session(engine) as session:
        statement = (
            select(
                SalesRecord.month_key,
                func.sum(SalesRecord.amount).label("amount"),
            )
            .group_by(SalesRecord.month_key)
            .order_by(SalesRecord.month_key)
        )
        results = session.exec(statement).all()
    df = pd.DataFrame(results, columns=["month", "amount"])
    if limit:
        df = df.tail(limit)
    if not df.empty:
        df["month_start"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df


def fetch_segment(path: str, column: str) -> pd.DataFrame:
    """Aggregate amount by a specific column (e.g. category)."""

    engine = get_engine(path)
    with Session(engine) as session:
        statement = (
            select(getattr(SalesRecord, column), func.sum(SalesRecord.amount))
            .group_by(getattr(SalesRecord, column))
            .where(getattr(SalesRecord, column).is_not(None))
        )
        rows = session.exec(statement).all()
    return pd.DataFrame(rows, columns=[column, "amount"])
