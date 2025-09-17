"""freee accounting connector."""

from __future__ import annotations

from .base import BaseConnector


class FreeeConnector(BaseConnector):
    service_name = "freee"

    def fetch_sales(self):  # pragma: no cover
        return self._mock_sales()
