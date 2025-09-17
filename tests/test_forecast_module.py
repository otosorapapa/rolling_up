import pandas as pd

from utils.forecast import forecast_sales, prepare_time_series


def test_forecast_returns_future_periods():
    data = pd.DataFrame({
        "month": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"],
        "amount": [100, 120, 130, 125, 140, 150],
        "month_start": pd.date_range("2024-01-01", periods=6, freq="MS"),
    })
    forecast = forecast_sales(data, periods=3)
    assert len(forecast) >= 9
    assert "yhat" in forecast.columns


def test_prepare_time_series_groups_months():
    data = pd.DataFrame({"month": ["2024-01", "2024-01", "2024-02"], "amount": [10, 20, 30]})
    series = prepare_time_series(data)
    assert series.loc[0, "y"] == 30
    assert len(series) == 2
