from connectors import (
    AirRegiConnector,
    ConnectorConfig,
    FreeeConnector,
    MoneyForwardConnector,
    ShopifyConnector,
    SquareConnector,
)


def test_mock_connectors_return_dataframe():
    connectors = [
        SquareConnector(ConnectorConfig(api_key="demo")),
        AirRegiConnector(ConnectorConfig(api_key="demo")),
        FreeeConnector(ConnectorConfig(api_key="demo")),
        MoneyForwardConnector(ConnectorConfig(api_key="demo")),
        ShopifyConnector(ConnectorConfig(api_key="demo")),
    ]
    for connector in connectors:
        df = connector.fetch_sales()
        assert not df.empty
        assert {"month", "amount"}.issubset(df.columns)
