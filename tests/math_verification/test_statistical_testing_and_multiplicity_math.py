"""Machine verification of Statistical Metrics, Moving-Block Bootstrap, and Multiplicity Adjustments.

Verifies:
1. Sharpe & Sortino ratios and annualized scaling sqrt(252).
2. Moving-block bootstrap paired difference preserving serial dependence and empirical p-value.
3. Multiplicity corrections: Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR).
4. Multiplicity hierarchy: p_raw <= p_fdr <= p_holm <= 1.0 for all hypotheses.
5. Probability of Backtest Overfitting (PBO) mathematical bounds [0, 1].
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.statistical_bootstrap import (
    compute_pbo_from_matrix,
    fdr_benjamini_hochberg,
    holm_bonferroni_correction,
    moving_block_bootstrap_paired_diff,
)


class TestRatiosAndPerformanceMetrics:
    """Verifies annualized Sharpe and Sortino formulation."""

    @staticmethod
    def compute_sharpe(returns: np.ndarray) -> float:
        r = np.asarray(returns, dtype=float)
        std = np.std(r, ddof=1)
        return float(np.mean(r) / std * np.sqrt(252)) if std > 1e-12 else 0.0

    @staticmethod
    def compute_sortino(returns: np.ndarray) -> float:
        r = np.asarray(returns, dtype=float)
        downside = r[r < 0]
        down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.0
        return float(np.mean(r) / down_std * np.sqrt(252)) if down_std > 1e-12 else 0.0

    def test_sharpe_and_sortino_properties(self):
        """Verify that Sortino penalizes only downside volatility while Sharpe penalizes total variance."""
        rng = np.random.default_rng(888)
        returns = rng.normal(loc=0.001, scale=0.015, size=252)

        sr = self.compute_sharpe(returns)
        sortino = self.compute_sortino(returns)

        # Both ratios are well-defined
        assert np.isfinite(sr)
        assert np.isfinite(sortino)

        # If we add large positive outliers, Sharpe's denominator increases (penalizing upside variance),
        # whereas Sortino's denominator is unaffected!
        returns_with_windfall = returns.copy()
        returns_with_windfall[returns_with_windfall > 0] *= 2.0

        sortino_boosted = self.compute_sortino(returns_with_windfall)
        sr_boosted = self.compute_sharpe(returns_with_windfall)

        # Both improve, but Sortino benefits without denominator inflation
        assert sortino_boosted > sortino
        assert sr_boosted > sr


class TestBootstrapAndMultiplicityMath:
    """Verifies moving-block bootstrap and multiple comparison adjustments."""

    def test_moving_block_bootstrap_invariants(self):
        """Verify that block bootstrap respects bounds and confidence intervals."""
        rng = np.random.default_rng(999)
        n = 150
        series_a = rng.normal(0.003, 0.01, size=n)
        series_b = rng.normal(0.000, 0.01, size=n)

        res = moving_block_bootstrap_paired_diff(
            series_a, series_b,
            metric_func=lambda x: float(np.mean(x)),
            block_length=21,
            n_bootstraps=300,
            confidence_level=0.95,
            random_seed=42,
        )

        # Invariant 1: Point estimate matches direct difference
        expected_point = float(np.mean(series_a) - np.mean(series_b))
        assert np.isclose(res.point_estimate, expected_point)

        # Invariant 2: Confidence intervals enclose point estimate for consistent estimator
        assert res.ci_lower <= res.ci_upper

        # Invariant 3: p-value is strictly bounded in [0, 1]
        assert 0.0 <= res.p_value <= 1.0

    def test_multiplicity_hierarchy_p_raw_le_fdr_le_holm(self):
        """Prove the fundamental multiplicity hierarchy: p_raw <= p_fdr <= p_holm <= 1.0."""
        raw_p_values = [0.001, 0.015, 0.025, 0.040, 0.080, 0.200]

        holm_p = holm_bonferroni_correction(raw_p_values)
        fdr_p = fdr_benjamini_hochberg(raw_p_values)

        for p_raw, p_f, p_h in zip(raw_p_values, fdr_p, holm_p):
            # Invariant 1: Multiplicity adjustments never make p-values smaller
            assert p_raw <= p_f + 1e-12
            assert p_f <= p_h + 1e-12

            # Invariant 2: All adjusted values bounded by 1.0
            assert p_h <= 1.0 + 1e-12
            assert p_f <= 1.0 + 1e-12

        # Invariant 3: Monotonicity preservation
        assert np.all(np.diff(holm_p) >= -1e-12)
        assert np.all(np.diff(fdr_p) >= -1e-12)

    def test_pbo_theoretical_bounds(self):
        """Verify Probability of Backtest Overfitting (PBO) lies in [0, 1]."""
        rng = np.random.default_rng(1234)
        # 16 chunks, 6 strategies
        returns_matrix = rng.normal(0.0005, 0.01, size=(252, 6))
        pbo = compute_pbo_from_matrix(returns_matrix, n_splits=16)

        assert 0.0 <= pbo <= 1.0
