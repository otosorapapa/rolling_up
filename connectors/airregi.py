"""Airレジ connector."""

from __future__ import annotations

from .base import BaseConnector


class AirRegiConnector(BaseConnector):
    service_name = "AirRegi"

    def fetch_sales(self):  # pragma: no cover
        return self._mock_sales()
