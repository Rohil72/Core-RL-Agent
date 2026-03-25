import numpy as np
import pandas as pd

from src.data.features import compute_technical_features


def test_compute_technical_features_adds_trend_template_gate():
    index = pd.date_range("2023-01-02", periods=320, freq="B", tz="UTC")
    close = np.linspace(50, 180, len(index))
    df = pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000_000, 2_000_000, len(index)),
        },
        index=index,
    )

    features = compute_technical_features(df)
    last_row = features.iloc[-1]

    assert "tech_minervini_template_score" in features.columns
    assert "tech_minervini_gate" in features.columns
    assert "tech_sma_200" in features.columns
    assert last_row["tech_minervini_template_score"] >= 6.0
    assert last_row["tech_minervini_gate"] == 1.0
    assert last_row["tech_pct_above_52w_low"] > 0.30
    assert last_row["tech_pct_from_52w_high"] > -0.25
