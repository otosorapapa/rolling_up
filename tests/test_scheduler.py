import pandas as pd

from connectors.base import BaseConnector, ConnectorConfig
from utils.scheduler import DataSyncScheduler


class DummyConnector(BaseConnector):
    service_name = "Dummy"

    def __init__(self):
        super().__init__(ConnectorConfig(api_key="demo"))
        self.calls = 0

    def fetch_sales(self):
        self.calls += 1
        return pd.DataFrame({"month": ["2024-01"], "month_start": pd.to_datetime(["2024-01-01"]), "amount": [100]})


def test_scheduler_runs_pending_jobs():
    connector = DummyConnector()
    scheduler = DataSyncScheduler()
    collected = {}

    def callback(df):
        collected["rows"] = len(df)

    scheduler.schedule(connector, minutes=0, callback=callback)
    scheduler.run_pending()
    assert connector.calls == 1
    assert collected["rows"] == 1
