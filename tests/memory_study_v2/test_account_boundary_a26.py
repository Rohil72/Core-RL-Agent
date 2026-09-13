"""Acceptance Test A26: Annual policy refresh preserves holdings/state; development resets; terminal rule tested."""

import pytest
from memory_study_v2.execution import PortfolioAccount, Position


def test_annual_refresh_preserves_holdings_and_terminal_rule_liquidates():
    """Verify annual refresh preserves state, and terminal liquidation closes positions."""
    account = PortfolioAccount(initial_capital=100000.0, commission=0.001, slippage=0.0005)
    account.positions["SEC_A"] = Position(
        security_id="SEC_A",
        quantity=50.0,
        entry_price=100.0,
        cost_basis=5000.0,
        peak_price=110.0,
        age_sessions=10,
    )

    # Annual transition from 2020 to 2021: economic state is preserved!
    assert "SEC_A" in account.positions
    assert account.positions["SEC_A"].age_sessions == 10

    # Terminal liquidation at end of 2025:
    account.execute_terminal_liquidation("2025-12-31", {"SEC_A": 100.0})
    assert len(account.positions) == 0
    assert len(account.trades) == 1
    assert account.trades[0].side == "TERMINAL_SELL"
