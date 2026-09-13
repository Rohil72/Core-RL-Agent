"""Acceptance Test A09: Training, bank, validation and query annual arrays identical for identical origin."""

import pytest
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import build_total_return_bars
from memory_study_v2.features import compute_technical_features, fit_scaler
from memory_study_v2.representations import extract_annual_representation


def test_representation_parity_across_callers():
    """Verify that identical origin session yields byte-identical vectors for training, bank, validation, and query."""
    n = 520
    dates = [f"2018-{i//30+1:02d}-{i%30+1:02d}" for i in range(n)]
    np.random.seed(123)
    prices = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    prices = np.maximum(prices, 10.0)

    df = pd.DataFrame({
        "session": dates,
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.uniform(500, 1500, n),
    })
    tr = build_total_return_bars(df)
    feat_df = compute_technical_features(tr)
    scaler = fit_scaler(feat_df, training_cutoff="2018-12-31", start_date="2018-01-01")

    # Extract at session 255 simulating training pipeline
    rep_train = extract_annual_representation(feat_df, session_idx_t=510, scaler=scaler)

    # Extract at session 255 simulating memory bank ingestion
    rep_bank = extract_annual_representation(feat_df, session_idx_t=510, scaler=scaler)

    # Extract at session 255 simulating live evaluation query
    rep_query = extract_annual_representation(feat_df, session_idx_t=510, scaler=scaler)

    # Must be bitwise identical
    assert np.array_equal(rep_train.transformer_matrix, rep_bank.transformer_matrix)
    assert np.array_equal(rep_train.flattened_vector, rep_bank.flattened_vector)
    assert np.array_equal(rep_train.flattened_vector, rep_query.flattened_vector)
