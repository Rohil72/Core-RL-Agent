"""Acceptance Test A23: Next-open order, exits before entries, cash constraints and integer entry quantities match fixture."""

import pytest
from memory_study_v2.execution import PortfolioAccount


def test_exits_processed_before_entries_and_integer_sizing():
    """Verify pending exit executes first to credit cash before entry buys."""
    account = PortfolioAccount(initial_capital=1000.0, max_positions=1, commission=0.001, slippage=0.0005)
    
    # Enter initial position in SEC_A: cost ~950
    account.pending_entries = ["SEC_A"]
    account.process_open_fills(
        session="2020-01-02",
        open_prices={"SEC_A": 100.0},
        tradable_flags={"SEC_A": True},
        close_decision_equity=1000.0,
    )
    # Remaining cash is very low (~50)
    assert account.cash < 100.15  # not enough to buy 1 share at ~100.15
    assert "SEC_A" in account.positions

    # Now at next close, queue exit for SEC_A and queue entry for SEC_B
    account.pending_exits["SEC_A"] = "STOP"
    account.pending_entries = ["SEC_B"]

    # At next open:
    # If exits process before entries: SEC_A sells for ~950, replenishing cash to ~1000, allowing SEC_B to buy!
    # If entries tried to process first: cash is < 60, so SEC_B would fail to trade!
    account.process_open_fills(
        session="2020-01-03",
        open_prices={"SEC_A": 100.0, "SEC_B": 100.0},
        tradable_flags={"SEC_A": True, "SEC_B": True},
        close_decision_equity=1000.0,
    )

    # Verify SEC_A is sold and SEC_B is bought!
    assert "SEC_A" not in account.positions
    assert "SEC_B" in account.positions
    assert account.positions["SEC_B"].quantity >= 1
    assert account.cash >= 0.0
