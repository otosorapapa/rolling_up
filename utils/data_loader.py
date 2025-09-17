"""Utilities for ingesting heterogeneous sales data sources.

This module centralises the logic for reading user supplied files and
converting them into a standard long-form dataframe that the rest of the
application can operate on.  The functions are intentionally written to
be side effect free so that they can be unit tested without Streamlit.
"""

from __future__ import annotations

import io
from io import BufferedReader
import json
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Union

import pandas as pd
import requests


SUPPORTED_FILE_TYPES = {"csv", "excel", "google"}


@dataclass
class ColumnMapping:
    """Represents the mapping from raw column names to canonical fields.

    Attributes
    ----------
    date : str
        Column containing the transaction or aggregation date.
    value : str
        Column containing the monetary amount to aggregate.
    product_name : Optional[str]
        Optional descriptive name for the SKU or service.
    product_code : Optional[str]
        Optional SKU code identifier.
    category : Optional[str]
        Optional product category column.
    customer_id : Optional[str]
        Optional customer identifier column for RFM analysis.
    quantity : Optional[str]
        Optional sales quantity column.
    """

    date: str
    value: str
    product_name: Optional[str] = None
    product_code: Optional[str] = None
    category: Optional[str] = None
    customer_id: Optional[str] = None
    quantity: Optional[str] = None


COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "date": (
        "date",
        "日付",
        "売上日",
        "年月",
        "年月度",
        "month",
        "月",
        "期間",
        "対象月",
    ),
    "value": (
        "sales",
        "売上",
        "売上金額",
        "金額",
        "売上高",
        "実績",
        "amount",
        "revenue",
    ),
    "product_name": (
        "商品名",
        "品目",
        "サービス",
        "product",
        "product_name",
        "sku",
        "アイテム",
    ),
    "product_code": (
        "商品コード",
        "品番",
        "product_code",
        "sku_code",
        "code",
    ),
    "category": (
        "カテゴリ",
        "カテゴリ名",
        "カテゴリー",
        "category",
        "部門",
        "セグメント",
    ),
    "customer_id": (
        "顧客id",
        "customer_id",
        "会員id",
        "client",
        "得意先",
    ),
    "quantity": (
        "数量",
        "販売数量",
        "量",
        "quantity",
        "販売数",
    ),
}


def _normalise_column_name(name: str) -> str:
    """Return a normalised representation for matching aliases."""

    clean = re.sub(r"[^0-9a-zA-Z一-龥ぁ-んァ-ン]+", "", str(name)).lower()
    return clean


MONTH_HEADER_PATTERN = re.compile(
    r"^(?P<year>20\d{2})[^0-9]?(?P<month>0[1-9]|1[0-2])"
)


def _detect_month_headers(columns: Iterable[str]) -> Tuple[str, ...]:
    """Return columns that look like month headers (wide format)."""

    found = []
    for col in columns:
        if isinstance(col, str) and MONTH_HEADER_PATTERN.search(col.replace("年", "-")):
            found.append(col)
    return tuple(found)


def detect_column_candidates(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Automatically detect likely column mappings.

    Parameters
    ----------
    df:
        Input dataframe uploaded by the user.

    Returns
    -------
    dict
        Dictionary mapping canonical field names to detected column names
        or ``None`` when no suitable candidate was found.
    """

    normalised = {col: _normalise_column_name(col) for col in df.columns}
    detected: MutableMapping[str, Optional[str]] = {k: None for k in COLUMN_ALIASES}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalise_column_name(alias)
            for col, col_norm in normalised.items():
                if canonical != "date" and col == detected.get("date"):
                    continue
                if alias_norm in col_norm:
                    detected[canonical] = col
                    break
            if detected[canonical] is not None:
                break

    month_headers = _detect_month_headers(df.columns)

    # Additional heuristic: if month columns provided as headers (wide
    # table) try to detect the first column as product name.
    if detected["date"] is None and df.columns.size >= 2 and month_headers:
        first_col = df.columns[0]
        if normalised[first_col] not in {"id", "コード", "code"}:
            detected["product_name"] = detected["product_name"] or first_col
        # In wide format the value column is the first month header.
        if month_headers:
            detected["date"] = month_headers[0]
            detected["value"] = month_headers[0]

    # Fallback: choose the first numeric column as value when none was
    # detected.  This helps with generic exports where column names do
    # not include obvious keywords.
    if detected["value"] is None:
        numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_candidates:
            detected["value"] = numeric_candidates[0]

    # Fallback for date column: look for parseable month headers across
    # the dataset.
    if detected["date"] is None:
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(5)
            try:
                normalise_month_series(sample)
                detected["date"] = col
                break
            except Exception:
                continue
    return dict(detected)


def normalise_month_series(values: Iterable) -> pd.Series:
    """Normalise an iterable of date-like values into ``YYYY-MM`` strings.

    The function accepts strings such as ``YYYY-MM``, ``YYYY/MM/DD`` or
    ``YYYY年MM月`` and returns a pandas ``Series`` with uniform
    ``YYYY-MM`` formatting.  Invalid entries raise ``ValueError`` to make
    validation explicit for the caller.
    """

    normalised = []
    for value in values:
        if pd.isna(value):
            raise ValueError("Month value is NaN")
        text = str(value).strip()
        if not text:
            raise ValueError("Empty month string")
        # Replace Japanese era/characters
        text = text.replace("年", "-").replace("月", "").replace("/", "-")
        if re.match(r"^\d{4}-\d{1,2}(-\d{1,2})?$", text):
            parts = text.split("-")
            year = int(parts[0])
            month = int(parts[1])
            if not 1 <= month <= 12:
                raise ValueError(f"Month out of range: {value}")
            normalised.append(f"{year:04d}-{month:02d}")
            continue
        # Try pandas parsing which supports 2024/01/01など
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"Unrecognised month value: {value}")
        normalised.append(parsed.strftime("%Y-%m"))
    return pd.Series(normalised, dtype="string")


def _load_csv(data: io.BytesIO, **kwargs) -> pd.DataFrame:
    """Load a CSV file with encoding detection."""

    raw = data.getvalue()
    if not raw:
        raise ValueError("空のCSVファイルです。")
    sample = raw[:4096]
    try:
        import chardet

        encoding = chardet.detect(sample)["encoding"] or "utf-8"
    except Exception:
        encoding = "utf-8"
    data.seek(0)
    return pd.read_csv(data, encoding=encoding, **kwargs)


def _load_excel(data: io.BytesIO, **kwargs) -> pd.DataFrame:
    """Load an Excel file from bytes."""

    return pd.read_excel(data, **kwargs)


def _load_google_sheet(url: str, worksheet: Optional[str] = None) -> pd.DataFrame:
    """Load a Google Spreadsheet via the public CSV export endpoint."""

    if "spreadsheets" not in url:
        raise ValueError("GoogleスプレッドシートのURLを指定してください。")
    if "export" in url:
        export_url = url
    else:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
        if not match:
            raise ValueError("シートIDを抽出できませんでした。")
        sheet_id = match.group(1)
        params = "&" + worksheet if worksheet and "gid=" not in worksheet else ""
        export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv{params}"
    response = requests.get(export_url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


FileLike = Union[io.BytesIO, BufferedReader]


def load_supported_file(
    file: Union[FileLike, str],
    source: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load CSV/Excel/Google Sheets into a pandas dataframe.

    Parameters
    ----------
    file:
        Either a file-like object (``BytesIO``) or a string path/URL.
    source:
        Optional explicit source type (``"csv"``, ``"excel"`` or
        ``"google"``).  When ``None`` the type is derived from the file
        extension or URL pattern.
    """

    if source is None:
        candidate_name: Optional[str] = None
        if isinstance(file, (str, pathlib.Path)):
            candidate_name = str(file)
        else:
            candidate_name = getattr(file, "name", None)
        if candidate_name is None:
            raise ValueError("sourceを明示するか、ファイルパス文字列を指定してください。")
        lower_name = candidate_name.lower()
        if lower_name.endswith((".csv", ".txt")):
            source = "csv"
        elif lower_name.endswith((".xlsx", ".xlsm", ".xls")):
            source = "excel"
        elif "spreadsheets.google" in lower_name:
            source = "google"
        else:
            raise ValueError("対応していないファイル拡張子です。")

    source = source.lower()
    if source not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"未知のデータソースです: {source}")

    if source == "google":
        if not isinstance(file, str):
            raise ValueError("GoogleスプレッドシートはURL文字列で指定してください。")
        worksheet = kwargs.pop("worksheet", None)
        return _load_google_sheet(file, worksheet=worksheet)

    if isinstance(file, (str, pathlib.Path)):
        with open(file, "rb") as f:
            data = io.BytesIO(f.read())
    else:
        data = file
    if source == "csv":
        return _load_csv(data, **kwargs)
    return _load_excel(data, **kwargs)


def build_sales_dataframe(df: pd.DataFrame, mapping: ColumnMapping | Mapping[str, str]) -> pd.DataFrame:
    """Create a normalised sales dataframe based on column mapping.

    The returned dataframe contains at least the columns ``month`` and
    ``amount`` which downstream modules rely on.  Additional metadata
    such as product name or category is preserved when available.
    """

    if isinstance(mapping, Mapping) and not isinstance(mapping, ColumnMapping):
        mapping = ColumnMapping(**mapping)  # type: ignore[arg-type]
    if not mapping.date or not mapping.value:
        raise ValueError("dateとvalue列は必須です。")

    data = df.copy()

    month_headers = _detect_month_headers(data.columns)
    is_wide = mapping.date not in data.columns or (
        mapping.date in month_headers and mapping.value in month_headers
    )
    if is_wide:
        if month_headers:
            id_vars = [c for c in data.columns if c not in month_headers]
            melted = data.melt(
                id_vars=id_vars,
                value_vars=list(month_headers),
                var_name="__month",
                value_name="__value",
            )
            data = melted
            mapping.date = "__month"
            mapping.value = "__value"
        else:
            raise ValueError("日付列が見つかりません。手動でマッピングを指定してください。")

    date_series = normalise_month_series(data[mapping.date])
    amount_series = pd.to_numeric(data[mapping.value], errors="coerce").fillna(0.0)

    normalised = pd.DataFrame({
        "month": date_series,
        "amount": amount_series,
    })
    if mapping.product_code and mapping.product_code in data.columns:
        normalised["product_code"] = data[mapping.product_code].astype("string")
    if mapping.product_name and mapping.product_name in data.columns:
        normalised["product_name"] = data[mapping.product_name].astype("string")
    if mapping.category and mapping.category in data.columns:
        normalised["category"] = data[mapping.category].astype("string")
    if mapping.customer_id and mapping.customer_id in data.columns:
        normalised["customer_id"] = data[mapping.customer_id].astype("string")
    if mapping.quantity and mapping.quantity in data.columns:
        normalised["quantity"] = pd.to_numeric(data[mapping.quantity], errors="coerce").fillna(0.0)

    # Provide an integer timestamp column useful for forecasting
    normalised["month_start"] = pd.to_datetime(normalised["month"], format="%Y-%m")
    return normalised


def data_frame_from_json(text: str) -> pd.DataFrame:
    """Load dataframe from a JSON payload containing records.

    This helper is primarily used in unit tests where building a
    ``BytesIO`` stream is unnecessary.  It accepts either a list of
    dictionaries or a dictionary with the ``"data"`` key.
    """

    payload = json.loads(text)
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    return pd.DataFrame(payload)
