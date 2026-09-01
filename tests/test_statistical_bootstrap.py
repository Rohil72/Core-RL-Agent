"""Tests for statistical bootstrap, Holm, and FDR corrections."""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.statistical_bootstrap import (
    compute_pbo_from_matrix,
    fdr_benjamini_hochberg,
    holm_bonferroni_correction,
    moving_block_bootstrap_paired_diff,
)


def test_moving_block_bootstrap():
    np.random.seed(42)
    n = 100
    # A is clearly superior to B
    series_a = np.random.normal(0.002, 0.01, size=n)
    series_b = np.random.normal(0.000, 0.01, size=n)

    def mean_return(x: np.ndarray) -> float:
        return float(np.mean(x))

    res = moving_block_bootstrap_paired_diff(
        series_a,
        series_b,
        metric_func=mean_return,
        block_length=21,
        n_bootstraps=200,
        random_seed=7,
    )
    assert res.point_estimate > 0.0
    assert res.ci_lower < res.point_estimate < res.ci_upper
    assert res.block_length == 21


def test_holm_bonferroni():
    raw_p = [0.01, 0.04, 0.03, 0.005]
    adj_p = holm_bonferroni_correction(raw_p)
    assert len(adj_p) == len(raw_p)
    # Smallest p (0.005) multiplied by 4 -> 0.02
    assert np.isclose(adj_p[3], 0.02)
    # Monotonicity preserved
    assert all(a >= r for a, r in zip(adj_p, raw_p))


def test_fdr_benjamini_hochberg():
    raw_p = [0.01, 0.04, 0.03, 0.005]
    fdr_p = fdr_benjamini_hochberg(raw_p)
    assert len(fdr_p) == len(raw_p)
    assert all(0.0 <= p <= 1.0 for p in fdr_p)


def test_compute_pbo():
    np.random.seed(42)
    # 100 days, 10 random noise strategies
    returns = np.random.normal(0.0, 0.01, size=(100, 10))
    pbo = compute_pbo_from_matrix(returns)
    assert 0.0 <= pbo <= 1.0
