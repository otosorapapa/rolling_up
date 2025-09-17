"""Shopify connector."""

from __future__ import annotations

from .base import BaseConnector


class ShopifyConnector(BaseConnector):
    service_name = "Shopify"

    def fetch_sales(self):  # pragma: no cover
        return self._mock_sales()
