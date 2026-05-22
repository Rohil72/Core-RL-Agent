import numpy as np
import pandas as pd

from src.cycle.oracle import decode_return_spans


def test_decode_return_spans_basic_hysteresis_and_cooldown():
    # simple synthetic sequence to exercise enter, hysteresis exit, and cooldown
    dates = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame(index=dates)
    df["ticker"] = "TICK"
    df["close"] = 100.0 + np.arange(len(df))

    # Predicted future extremes (synthetic)
    df["future_max_return_63"] = [0.0, 0.06, 0.06, 0.06, 0.02, 0.02, 0.06, 0.02, 0.02, 0.0]
    df["future_min_return_63"] = [0.0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0]

    spans = decode_return_spans(
        df,
        pred_max_col="future_max_return_63",
        pred_min_col="future_min_return_63",
        enter_threshold=0.05,
        exit_threshold=0.03,
        risk_threshold=0.05,
        hysteresis_days=2,
        cooldown_days=1,
    )

    # expect a single span that opened at index 1 and closed at index 5 (after hysteresis)
    assert len(spans) == 1
    s = spans[0]
    assert s.start_date == dates[1]
    assert s.end_date == dates[5]

    expected_return = (float(df["close"].iloc[5]) / float(df["close"].iloc[1])) - 1.0
    assert abs(s.return_pct - expected_return) < 1e-9
