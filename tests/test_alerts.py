import pandas as pd

from utils.alerts import build_alert_message, send_alert


def test_build_alert_message_handles_empty():
    df = pd.DataFrame()
    message = build_alert_message(df)
    assert "ありません" in message


def test_send_alert_dry_run_returns_message():
    df = pd.DataFrame([
        {"product_name": "A", "metric": "yoy", "actual": -0.2, "month": "2024-01"}
    ])
    result = send_alert(df, settings={}, dry_run=True)
    assert result["email_sent"] is False
    assert "A" in result["message"]
