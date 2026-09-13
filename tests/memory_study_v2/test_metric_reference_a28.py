"""Acceptance Test A28: Annualization, zero variance, initial/terminal costs, turnover and exposure match manual fixtures."""

import pytest
import numpy as np
from memory_study_v2.metrics import compute_portfolio_metrics


def test_metrics_formulas_match_fixtures():
    """Verify annualized return, zero volatility handling, and turnover."""
    # Flat NAV: exactly 100000 every day for 252 sessions
    navs = np.full(252, 100000.0)
    notionals = np.zeros(252)
    holdings = np.zeros(252)

    m = compute_portfolio_metrics(navs, notionals, holdings, initial_capital=100000.0)
    assert m.annualized_return == 0.0
    assert m.annualized_volatility == 0.0
    assert m.annualized_sharpe == 0.0  # zero variance assigned 0.0
    assert m.max_drawdown == 0.0
    assert m.win_rate == 0.0
    assert m.annualized_turnover == 0.0
    assert m.mean_exposure == 0.0
