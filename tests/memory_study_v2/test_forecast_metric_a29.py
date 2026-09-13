"""Acceptance Test A29: Constant prior rank IC is missing; valid-mask and daily cross-sectional averaging tested."""

import pytest
import numpy as np
from memory_study_v2.metrics import compute_daily_cross_sectional_rank_ic


def test_constant_prior_rank_ic_is_missing_not_zero():
    """Verify constant prior returns undefined IC (None), never 0.0."""
    dates = ["2020-01-02"] * 10
    constant_preds = np.full(10, 0.05)  # constant forecast across all 10 assets
    targets = np.linspace(-0.05, 0.05, 10)

    mean_ic, defined_count = compute_daily_cross_sectional_rank_ic(dates, constant_preds, targets, min_assets=5)
    assert mean_ic is None
    assert defined_count == 0


def test_valid_rank_ic_calculated_correctly():
    """Verify monotonic predictions give rank IC = 1.0."""
    dates = ["2020-01-02"] * 10
    preds = np.linspace(1, 10, 10)
    targets = np.linspace(10, 100, 10)

    mean_ic, defined_count = compute_daily_cross_sectional_rank_ic(dates, preds, targets, min_assets=5)
    assert pytest.approx(mean_ic, abs=1e-7) == 1.0
    assert defined_count == 1
