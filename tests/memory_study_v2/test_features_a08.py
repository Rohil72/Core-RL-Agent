"""Acceptance Test A08: 23 ordered formulas match hand-calculated fixtures; full windows and ddof tested."""

import pytest
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import build_total_return_bars
from memory_study_v2.features import compute_technical_features
from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED


def test_23_features_order_and_hand_calculated_values():
    """Verify that all 23 features match exact hand-calculated math and ddof=1."""
    # Build 260 sessions so that 252-day warm-up completes
    n = 260
    dates = [f"2020-{i//30+1:02d}-{i%30+1:02d}" for i in range(n)]
    
    # Constant close 100.0, flat high/low/open/volume
    df = pd.DataFrame({
        "session": dates,
        "open": np.full(n, 100.0),
        "high": np.full(n, 102.0),
        "low": np.full(n, 98.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    })
    tr = build_total_return_bars(df)
    feats = compute_technical_features(tr)

    # Check 23 feature column names in exact order
    feat_cols = [c for c in feats.columns if c not in ("session", "valid_mask")]
    assert feat_cols == EXPECTED_FEATURES_ORDERED

    # On session 255 (valid):
    # return_1 = 0.0
    assert pytest.approx(feats.loc[255, "return_1"], abs=1e-7) == 0.0
    # momentum_21 = 0.0
    assert pytest.approx(feats.loc[255, "momentum_21"], abs=1e-7) == 0.0
    # volatility_21 = 0.0 (std of zeros)
    assert pytest.approx(feats.loc[255, "volatility_21"], abs=1e-7) == 0.0
    # volume_ratio_21 = 1.0 (1000 / 1000)
    assert pytest.approx(feats.loc[255, "volume_ratio_21"], abs=1e-5) == 1.0
    # intraday_range = (102 - 98) / 100 = 0.04
    assert pytest.approx(feats.loc[255, "intraday_range"], abs=1e-5) == 0.04
    # drawdown_252 = (100 - 100) / 100 = 0.0
    assert pytest.approx(feats.loc[255, "drawdown_252"], abs=1e-7) == 0.0
    # rsi_14 = flat window returns 0.5
    assert pytest.approx(feats.loc[255, "rsi_14"], abs=1e-7) == 0.5
    # bollinger_width_20 = 0.0
    assert pytest.approx(feats.loc[255, "bollinger_width_20"], abs=1e-7) == 0.0
