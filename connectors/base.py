"""Base classes for POS/accounting connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np


@dataclass
class ConnectorConfig:
    api_key: str
    account_id: Optional[str] = None


class BaseConnector:
    """Abstract connector definition."""

    service_name: str = ""

    def __init__(self, config: ConnectorConfig):
        self.config = config

    def authenticate(self) -> bool:  # pragma: no cover - trivial
        return bool(self.config.api_key)

    def fetch_sales(self) -> pd.DataFrame:
        raise NotImplementedError

    def _mock_sales(self, months: int = 12) -> pd.DataFrame:
        """Generate deterministic mock sales data for demos/tests."""

        seed = abs(hash((self.service_name, self.config.api_key))) % (2**32)
        rng = np.random.default_rng(seed)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="MS")
        amounts = rng.integers(300000, 900000, size=len(dates)).astype(float)
        df = pd.DataFrame({
            "month": dates.strftime("%Y-%m"),
            "month_start": dates,
            "amount": amounts,
            "product_code": f"{self.service_name}-総計",
            "product_name": f"{self.service_name} 集計",
        })
        return df
