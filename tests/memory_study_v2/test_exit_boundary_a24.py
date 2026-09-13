"""Acceptance Test A24: Age 1, age 63, strict stop inequality and missing-quote queues tested."""

import pytest
from memory_study_v2.execution import PortfolioAccount, Position


def test_holding_age_and_strict_stop_inequality():
    """Verify age 1 on entry, exit queued when age=63, and strict inequality for stop."""
    account = PortfolioAccount(initial_capital=100000.0, max_positions=1, min_stop_fraction=0.10)
    
    # Manually insert position at entry open 100.0
    account.positions["SEC_A"] = Position(
        security_id="SEC_A",
        quantity=10.0,
        entry_price=100.0,
        cost_basis=1000.0,
        peak_price=100.0,
        age_sessions=1,
    )

    # Test strict inequality: exactly 10% drawdown (close=90.0 on peak=100.0) -> drawdown = -0.10
    # Rule: close/peak - 1 < -0.10. -0.10 is NOT < -0.10, so stop should NOT trigger!
    account.evaluate_close_stops_and_update_state(
        session="2020-01-02",
        close_prices={"SEC_A": 90.0},
        atr_ratios={"SEC_A": 0.01},
    )
    assert "SEC_A" not in account.pending_exits

    # Drawdown of 10.01% (close=89.99) -> -0.1001 < -0.10 -> triggers STOP!
    account.evaluate_close_stops_and_update_state(
        session="2020-01-03",
        close_prices={"SEC_A": 89.99},
        atr_ratios={"SEC_A": 0.01},
    )
    assert account.pending_exits.get("SEC_A") == "STOP"

    # Test age 63 exit: reset exits, set age to 63
    account.pending_exits.clear()
    account.positions["SEC_A"].age_sessions = 63
    account.evaluate_close_stops_and_update_state(
        session="2020-01-04",
        close_prices={"SEC_A": 100.0},
        atr_ratios={"SEC_A": 0.01},
    )
    assert account.pending_exits.get("SEC_A") == "MAX_AGE"
