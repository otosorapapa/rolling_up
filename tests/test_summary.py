import pandas as pd

from utils.summary import generate_summary


def test_generate_summary_includes_metrics():
    metrics = {"total": 1000.0, "yoy": 0.1, "delta": 50.0, "slope": 0.2}
    highlights = pd.DataFrame({"product_name": ["A"], "amount": [500]})
    summary = generate_summary(metrics, timeframe="2024-01", highlights=highlights)
    assert "2024-01" in summary
    assert "A" in summary or "分析" in summary
