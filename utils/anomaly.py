"""Anomaly detection strategies for sales monitoring."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

Method = Literal["zscore", "isolation_forest", "seasonal_esd"]

ANOMALY_PRESETS = {
    "大幅減少": {"method": "zscore", "threshold": -2.5},
    "急増": {"method": "zscore", "threshold": 2.5},
    "構造変化": {"method": "seasonal_esd", "alpha": 0.05},
}


def zscore_anomaly(series: pd.Series, threshold: float = 3.0) -> pd.DataFrame:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0:
        scores = pd.Series(np.zeros(len(series)), index=series.index)
    else:
        scores = (series - mean) / std
    if threshold >= 0:
        anomalies = scores >= threshold
    else:
        anomalies = scores <= threshold
    return pd.DataFrame({"score": scores, "is_anomaly": anomalies})


def isolation_forest_anomaly(series: pd.Series, contamination: float = 0.1) -> pd.DataFrame:
    model = IsolationForest(contamination=contamination, random_state=42)
    reshaped = series.values.reshape(-1, 1)
    model.fit(reshaped)
    scores = model.decision_function(reshaped)
    preds = model.predict(reshaped)
    anomalies = preds == -1
    return pd.DataFrame({"score": scores, "is_anomaly": anomalies})


def seasonal_esd_anomaly(
    series: pd.Series,
    alpha: float = 0.05,
    max_anomalies: Optional[int] = None,
) -> pd.DataFrame:
    if len(series) < 6:
        return zscore_anomaly(series, threshold=3.0)
    stl = STL(series, period=12, robust=True)
    res = stl.fit()
    residual = res.resid
    std = residual.std(ddof=0)
    if std == 0:
        scores = pd.Series(np.zeros(len(series)), index=series.index)
        anomalies = pd.Series(False, index=series.index)
        return pd.DataFrame({"score": scores, "is_anomaly": anomalies})

    if max_anomalies is None:
        max_anomalies = max(1, int(len(series) * 0.2))

    scores = (residual - residual.mean()) / std
    sorted_idx = scores.abs().sort_values(ascending=False).index
    limit = max_anomalies
    anomalies = pd.Series(False, index=series.index)
    threshold = abs(scipy_esd_threshold(len(series), alpha))
    for idx in sorted_idx[:limit]:
        anomalies.loc[idx] = scores.loc[idx].abs() > threshold
    return pd.DataFrame({"score": scores, "is_anomaly": anomalies})


def scipy_esd_threshold(n: int, alpha: float) -> float:
    from scipy.stats import t

    p = 1 - alpha / (2 * n)
    t_value = t.ppf(p, n - 2)
    lam = ((n - 1) * t_value) / (np.sqrt(n) * np.sqrt(n - 2 + t_value**2))
    return lam


def detect_anomalies(
    df: pd.DataFrame,
    value_col: str,
    method: Method = "zscore",
    **kwargs,
) -> pd.DataFrame:
    """Dispatch to the requested anomaly detection routine."""

    series = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0)
    if method == "zscore":
        threshold = kwargs.get("threshold", 3.0)
        return zscore_anomaly(series, threshold=threshold)
    if method == "isolation_forest":
        contamination = kwargs.get("contamination", 0.1)
        return isolation_forest_anomaly(series, contamination=contamination)
    if method == "seasonal_esd":
        alpha = kwargs.get("alpha", 0.05)
        max_anomalies = kwargs.get("max_anomalies")
        return seasonal_esd_anomaly(series, alpha=alpha, max_anomalies=max_anomalies)
    raise ValueError(f"未知の異常検知手法です: {method}")
