"""Acceptance & Regression Test for R08: Feature contract alignment and oracle fixtures."""

import numpy as np
import pandas as pd
import pytest

from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED
from memory_study_v2.features import compute_technical_features


def test_trend_slope_divides_by_current_close_not_window_mean():
    # Construct a series of 25 bars with constant linear upward slope:
    # Closes: 10, 11, 12, ..., 34
    n = 30
    closes = np.arange(10.0, 10.0 + n, dtype=np.float64)
    # At t = 20: window is closes[0:21] = 10..30.
    # Mean of window is 20.0. Current close C[20] is 30.0.
    # Slope of indices 0..20 with y = 10..30 is exactly 1.0!
    # Under contract formula: slope / (C[t] + eps) = 1.0 / 30.0 = 0.0333333333.
    # Under buggy formula: slope / (mean + eps) = 1.0 / 20.0 = 0.05.
    df = pd.DataFrame({
        "session": [f"2020-01-{i+1:02d}" for i in range(n)],
        "tr_open": closes,
        "tr_high": closes + 0.5,
        "tr_low": closes - 0.5,
        "tr_close": closes,
        "normalized_volume": 1000.0,
    })
    res = compute_technical_features(df)
    slope_t20 = res["trend_slope_21"].iloc[20]
    expected_contract = 1.0 / (30.0 + 1e-9)
    buggy_mean_based = 1.0 / (20.0 + 1e-9)

    assert abs(slope_t20 - expected_contract) < 1e-6, f"Expected {expected_contract}, got {slope_t20}"
    assert abs(slope_t20 - buggy_mean_based) > 0.01, "Feature still divides by window mean!"


def test_up_down_volume_flat_up_evaluates_to_zero():
    # 60 bars where price is constantly decreasing: C[t] < C[t-1]
    # No up days! up_vol == 0.0.
    n = 60
    closes = np.linspace(100.0, 50.0, n)
    df = pd.DataFrame({
        "session": [f"2020-01-{i+1:02d}" for i in range(n)],
        "tr_open": closes,
        "tr_high": closes + 0.1,
        "tr_low": closes - 0.1,
        "tr_close": closes,
        "normalized_volume": 500.0,
    })
    res = compute_technical_features(df)
    val = res["up_down_volume_50"].iloc[55]
    # Under contract: sum(up) / (sum(down) + eps) = 0.0 / (down + eps) = 0.0.
    assert val == 0.0, f"Expected 0.0 for zero up-volume, got {val}"


def test_segment_aware_warmup_and_ema_reset():
    # Two disjoint segments of 100 bars each
    n_seg = 100
    closes = np.full(n_seg * 2, 100.0)
    segments = [1] * n_seg + [2] * n_seg
    sessions = [f"2020-01-{i+1:02d}" for i in range(n_seg)] + [f"2020-06-{i+1:02d}" for i in range(n_seg)]
    df = pd.DataFrame({
        "session": sessions,
        "tr_open": closes,
        "tr_high": closes,
        "tr_low": closes,
        "tr_close": closes,
        "normalized_volume": 1000.0,
        "segment_id": segments,
    })
    res = compute_technical_features(df)
    # Since neither segment has >= 252 contiguous bars, all valid_mask must be False!
    assert not np.any(res["valid_mask"]), "Warmup should require 252 contiguous bars per segment"
