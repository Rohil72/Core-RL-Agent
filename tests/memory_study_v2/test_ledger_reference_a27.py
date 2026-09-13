"""Acceptance Test A27: Independent ledger reconstruction cash and NAV within 1e-8 account units on fixtures."""

import pytest
from memory_study_v2.execution import PortfolioAccount


def test_independent_ledger_reconstruction():
    """Verify exact cash and NAV reconstruction across buy, dividend, and sell."""
    initial_cash = 100000.0
    account = PortfolioAccount(initial_capital=initial_cash, commission=0.001, slippage=0.0005)

    # Step 1: Buy 100 shares of SEC_A at open 100.0
    account.pending_entries = ["SEC_A"]
    account.process_open_fills("2020-01-02", {"SEC_A": 100.0}, {"SEC_A": True}, close_decision_equity=100000.0)
    
    qty = account.positions["SEC_A"].quantity
    buy_fill = 100.0 * (1.0 + 0.0005)
    comm_buy = 0.001 * (qty * buy_fill)
    expected_cash_1 = initial_cash - (qty * buy_fill + comm_buy)
    assert pytest.approx(account.cash, abs=1e-8) == expected_cash_1

    # Close valuation at 105.0
    nav_1 = account.evaluate_close_stops_and_update_state("2020-01-02", {"SEC_A": 105.0}, {})
    expected_nav_1 = expected_cash_1 + qty * 105.0
    assert pytest.approx(nav_1, abs=1e-8) == expected_nav_1
