import pandas as pd

from utils.anomaly import detect_anomalies


def test_zscore_detects_negative_outliers():
    df = pd.DataFrame({"value": [100, 110, 120, 20, 115]})
    result = detect_anomalies(df, value_col="value", method="zscore", threshold=-1.5)
    assert result.loc[3, "is_anomaly"]


def test_isolation_forest_returns_scores():
    df = pd.DataFrame({"value": [100] * 10 + [1000]})
    result = detect_anomalies(df, value_col="value", method="isolation_forest", contamination=0.1)
    assert "score" in result.columns
    assert result["is_anomaly"].any()


def test_seasonal_esd_handles_short_series():
    df = pd.DataFrame({"value": [100, 110, 120, 130, 125]})
    result = detect_anomalies(df, value_col="value", method="seasonal_esd")
    assert len(result) == len(df)
