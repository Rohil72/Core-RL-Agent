"""Acceptance Test A07: Appending future data/actions never changes earlier features or predictions."""

import pytest
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import build_total_return_bars, CorporateAction
from memory_study_v2.features import compute_technical_features


def test_appending_future_data_does_not_mutate_earlier_outputs():
    """Verify strict causality: appending T+1..T+50 never changes features at sessions 0..T."""
    # Generate 300 sessions of synthetic bars
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=350, freq="B").strftime("%Y-%m-%d").tolist()
    
    prices = [100.0]
    for _ in range(349):
        prices.append(prices[-1] * (1.0 + np.random.normal(0, 0.01)))
    prices = np.array(prices)

    df_full = pd.DataFrame({
        "session": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.uniform(500, 1500, 350),
    })

    # Historical slice: first 300 bars
    df_early = df_full.iloc[:300].copy()
    tr_early = build_total_return_bars(df_early)
    feat_early = compute_technical_features(tr_early)

    # Full slice: 350 bars (appending future data and a future split at bar 320)
    actions = [CorporateAction(session=dates[320], split_ratio=2.0, cash_dividend=1.0)]
    tr_full = build_total_return_bars(df_full, actions=actions)
    feat_full = compute_technical_features(tr_full)

    # Check that earlier 300 bars in tr_bars are byte-for-byte or exact float identical
    np.testing.assert_array_almost_equal(
        tr_early["tr_close"].to_numpy(),
        tr_full.iloc[:300]["tr_close"].to_numpy(),
        decimal=10,
    )

    # Check that features on earlier 300 bars are identical
    for col in feat_early.columns:
        if col in ("session", "valid_mask"):
            continue
        v_early = feat_early[col].to_numpy()
        v_full = feat_full.iloc[:300][col].to_numpy()
        valid_both = np.isfinite(v_early) & np.isfinite(v_full)
        np.testing.assert_array_almost_equal(
            v_early[valid_both],
            v_full[valid_both],
            decimal=10,
            err_msg=f"Feature '{col}' mutated when future data was appended!"
        )
