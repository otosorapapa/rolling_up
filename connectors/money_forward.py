"""Money Forward connector."""

from __future__ import annotations

from .base import BaseConnector


class MoneyForwardConnector(BaseConnector):
    service_name = "MoneyForward"

    def fetch_sales(self):  # pragma: no cover
        return self._mock_sales()
