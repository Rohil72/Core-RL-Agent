"""Acceptance Test A31: Fresh full-precision summaries and bootstrap outputs replay at declared 1e-10 tolerance."""

import pytest
import numpy as np
from memory_study_v2.metrics import compute_portfolio_metrics
from memory_study_v2.inference import compute_bootstrap_p_and_ci


def test_full_precision_replay_tolerance():
    """Verify 1e-10 replay tolerance on fixed portfolio return and bootstrap inputs."""
    np.random.seed(999)
    navs = 100000.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 252)))
    notionals = np.full(252, 1000.0)
    holdings = navs * 0.8

    m1 = compute_portfolio_metrics(navs, notionals, holdings)
    m2 = compute_portfolio_metrics(navs, notionals, holdings)

    assert abs(m1.annualized_return - m2.annualized_return) < 1e-10
    assert abs(m1.annualized_volatility - m2.annualized_volatility) < 1e-10
    assert abs(m1.annualized_sharpe - m2.annualized_sharpe) < 1e-10

    # Bootstrap p and CI replay
    theta = 0.05
    draws = np.random.normal(0.05, 0.02, 1000)
    ci_l1, ci_u1, p1 = compute_bootstrap_p_and_ci(theta, draws)
    ci_l2, ci_u2, p2 = compute_bootstrap_p_and_ci(theta, draws)

    assert abs(ci_l1 - ci_l2) < 1e-10
    assert abs(ci_u1 - ci_u2) < 1e-10
    assert abs(p1 - p2) < 1e-10
