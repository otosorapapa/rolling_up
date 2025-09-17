"""Demand forecasting utilities using Prophet or pmdarima."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:  # pragma: no cover - optional dependency
    import pmdarima as pm
except Exception:  # pragma: no cover - gracefully handle missing package
    pm = None

try:  # pragma: no cover
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None  # type: ignore


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the dataframe to monthly totals."""

    if "month_start" not in df.columns:
        df = df.copy()
        df["month_start"] = pd.to_datetime(df["month"], format="%Y-%m")
    series = (
        df.groupby("month_start", as_index=False)["amount"].sum().sort_values("month_start")
    )
    series.rename(columns={"month_start": "ds", "amount": "y"}, inplace=True)
    return series


def forecast_sales(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """Forecast future sales using Prophet or pmdarima."""

    series = prepare_time_series(df)
    if series.empty:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    if Prophet is not None and len(series) >= 18:  # prefer Prophet when available
        model = Prophet()
        model.fit(series)
        future = model.make_future_dataframe(periods=periods, freq="MS")
        forecast = model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    if pm is not None and len(series) >= 6:
        model = pm.auto_arima(series["y"], seasonal=False, suppress_warnings=True)
        forecast_values, conf_int = model.predict(n_periods=periods, return_conf_int=True)
        last_date = series["ds"].max()
        future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
        forecast_df = pd.DataFrame(
            {
                "ds": future_index,
                "yhat": forecast_values,
                "yhat_lower": conf_int[:, 0],
                "yhat_upper": conf_int[:, 1],
            }
        )
        history = series.rename(columns={"y": "yhat"})
        history["yhat_lower"] = np.nan
        history["yhat_upper"] = np.nan
        return pd.concat([history, forecast_df], ignore_index=True)

    # Fallback: use simple moving average forecast
    history = series.copy()
    history.rename(columns={"y": "yhat"}, inplace=True)
    history["yhat_lower"] = np.nan
    history["yhat_upper"] = np.nan
    if len(history) >= 3:
        avg = history["yhat"].tail(3).mean()
    else:
        avg = history["yhat"].mean()
    last_date = history["ds"].max()
    future_index = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    fallback = pd.DataFrame(
        {
            "ds": future_index,
            "yhat": avg,
            "yhat_lower": avg * 0.8,
            "yhat_upper": avg * 1.2,
        }
    )
    return pd.concat([history, fallback], ignore_index=True)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame) -> go.Figure:
    """Create a responsive Plotly chart visualising the forecast."""

    fig = go.Figure()
    hist = history.sort_values("month_start")
    fig.add_trace(
        go.Scatter(
            x=hist["month_start"],
            y=hist["amount"],
            mode="lines+markers",
            name="実績",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"],
            y=forecast["yhat"],
            mode="lines",
            name="予測",
        )
    )
    if "yhat_lower" in forecast.columns and forecast["yhat_lower"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
                y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(100, 149, 237, 0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="信頼区間",
            )
        )
    fig.update_layout(margin=dict(l=40, r=30, t=20, b=50))
    return fig
