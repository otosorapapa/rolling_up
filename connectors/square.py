"""Square POS connector."""

from __future__ import annotations

from .base import BaseConnector


class SquareConnector(BaseConnector):
    service_name = "Square"

    def fetch_sales(self):  # pragma: no cover - simple wrapper
        return self._mock_sales()
