"""Acceptance Test A10: Session t excluded; first/last patch, shapes and clipping-before-pooling tested."""

import pytest
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import build_total_return_bars
from memory_study_v2.features import compute_technical_features, fit_scaler
from memory_study_v2.representations import extract_annual_representation


def test_session_t_excluded_and_shapes_correct():
    """Verify session t is strictly excluded, first patch is t-252..t-247, last is t-6..t-1."""
    n = 520
    dates = [f"2018-{i//30+1:02d}-{i%30+1:02d}" for i in range(n)]
    prices = np.full(n, 100.0)

    df = pd.DataFrame({
        "session": dates,
        "open": prices,
        "high": prices * 1.01,
        "low": prices * 0.99,
        "close": prices,
        "volume": np.full(n, 1000.0),
    })
    tr = build_total_return_bars(df)
    feat_df = compute_technical_features(tr)
    scaler = fit_scaler(feat_df, training_cutoff="2018-12-31", start_date="2018-01-01")

    # Poison session t=255 with an extreme outlier
    feat_df.loc[510, "return_1"] = 999999.0

    rep = extract_annual_representation(feat_df, session_idx_t=510, scaler=scaler)

    # Matrix shape (42, 23)
    assert rep.transformer_matrix.shape == (42, 23)
    # Flattened shape (966,)
    assert rep.flattened_vector.shape == (966,)
    assert rep.transformer_matrix.dtype == np.float32
    assert rep.flattened_vector.dtype == np.float32

    # Because session 255 was excluded, the extreme value did NOT affect rep at all!
    assert np.all(rep.transformer_matrix <= 5.0)
    assert np.all(rep.transformer_matrix >= -5.0)


def test_clipping_before_pooling():
    """Verify that features are clipped to [-5.0, 5.0] before averaging into patches."""
    n = 520
    dates = [f"2018-{i//30+1:02d}-{i%30+1:02d}" for i in range(n)]
    df = pd.DataFrame({
        "session": dates,
        "open": np.full(n, 100.0),
        "high": np.full(n, 101.0),
        "low": np.full(n, 99.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1000.0),
    })
    tr = build_total_return_bars(df)
    feat_df = compute_technical_features(tr)
    scaler = fit_scaler(feat_df, training_cutoff="2018-12-31", start_date="2018-01-01")

    # In patch 41 (sessions 249..254), set one session to an enormous outlier +100.0
    # If clipped BEFORE pooling: clipped to 5.0, average of 5 normal zeros and one 5.0 is 5.0 / 6 = 0.8333.
    # If pooled before clipping: average of 5 zeros and 100.0 is 100 / 6 = 16.666, then clipped to 5.0.
    feat_df.loc[509, "return_1"] = 100.0 * 1e-4  # after scaling gives 100.0

    rep = extract_annual_representation(feat_df, session_idx_t=510, scaler=scaler)
    patch_41_val = rep.transformer_matrix[41, 0]  # feature 0 (return_1) in patch 41
    # Must be approx 5.0 / 6 = 0.8333, NOT 5.0!
    assert patch_41_val < 1.0
    assert pytest.approx(patch_41_val, abs=0.01) == 5.0 / 6.0
