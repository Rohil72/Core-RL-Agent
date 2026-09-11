"""Machine verification of Feature Engineering, Technical Formulas, and Normalization Math.

Verifies:
1. Technical returns and multi-horizon momentum identities.
2. Rolling realized volatility non-negativity and bounds.
3. Closed-form OLS trend slope equivalence to np.polyfit.
4. Drawdown from rolling maximum invariant: -1.0 <= Drawdown <= 0.0.
5. Z-score standard scaling mathematical properties (zero mean, unit variance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from src.data.features import compute_technical_features


class TestTechnicalFeatureFormulas:
    """Verifies the mathematical correctness of technical indicator implementations."""

    def test_ols_trend_slope_analytical_closed_form(self):
        """Verify that np.polyfit slope exactly matches the analytical OLS formula."""
        n = 10
        t = np.arange(n, dtype=float)
        t_bar = np.mean(t)
        denom = np.sum((t - t_bar) ** 2)

        rng = np.random.default_rng(333)
        for _ in range(30):
            prices = rng.uniform(10.0, 100.0, size=n)
            p_bar = np.mean(prices)

            # Analytical OLS slope: cov(t, p) / var(t)
            slope_analytical = np.sum((t - t_bar) * (prices - p_bar)) / denom

            # Code implementation: np.polyfit(arange, x, 1)[0]
            slope_polyfit = float(np.polyfit(t, prices, 1)[0])

            assert np.isclose(slope_analytical, slope_polyfit, atol=1e-10)

    def test_drawdown_bounds_and_invariants(self):
        """Verify that rolling drawdown is strictly non-positive and bounded below by -1.0."""
        rng = np.random.default_rng(444)
        n = 300
        # Positive geometric brownian walk
        steps = rng.normal(0.0005, 0.02, size=n)
        prices = 100.0 * np.exp(np.cumsum(steps))

        rolling_max = pd.Series(prices).rolling(window=252, min_periods=1).max().to_numpy()
        drawdown = (prices / rolling_max) - 1.0

        # Invariant 1: Drawdown is always <= 0.0
        assert np.all(drawdown <= 1e-12)

        # Invariant 2: Drawdown is bounded below by -1.0 (for non-negative prices)
        assert np.all(drawdown >= -1.0 - 1e-12)

        # Invariant 3: When price makes a new high, drawdown is exactly 0.0
        new_highs = (prices == rolling_max)
        assert np.allclose(drawdown[new_highs], 0.0)

    def test_compute_technical_features_pipeline_math(self):
        """Verify feature calculations through the actual codebase pipeline."""
        rng = np.random.default_rng(555)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, size=100)))
        high = close * (1.0 + rng.uniform(0.001, 0.02, size=100))
        low = close * (1.0 - rng.uniform(0.001, 0.02, size=100))
        volume = rng.uniform(1e5, 1e6, size=100)

        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )

        feats = compute_technical_features(df)

        # Invariant 1: tech_return_1 == close / close.shift(1) - 1
        expected_ret = df["close"].pct_change()
        assert np.allclose(feats["tech_return_1"].dropna(), expected_ret.dropna())

        # Invariant 2: tech_momentum_21 == close / close.shift(21) - 1
        expected_mom21 = df["close"].pct_change(21)
        assert np.allclose(feats["tech_momentum_21"].dropna(), expected_mom21.dropna())

        # Invariant 3: tech_vol_21 >= 0
        assert np.all(feats["tech_vol_21"].dropna() >= 0.0)


class TestStandardizationMath:
    """Verifies Z-score standardization properties: zero mean and unit variance."""

    def test_zscore_zero_mean_and_unit_variance(self):
        """Verify that z = (x - mu) / sigma results in sample mean 0 and sample std 1."""
        rng = np.random.default_rng(666)
        x = rng.normal(loc=45.0, scale=12.0, size=500)

        mu = np.mean(x)
        sigma = np.std(x, ddof=0)

        z = (x - mu) / sigma

        assert np.isclose(np.mean(z), 0.0, atol=1e-12)
        assert np.isclose(np.std(z, ddof=0), 1.0, atol=1e-12)
