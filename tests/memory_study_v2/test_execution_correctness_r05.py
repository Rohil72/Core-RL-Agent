"""Acceptance & Regression Test for R05: Action-adjusted mark for missing prices and zero-slot planning."""

import pytest

from memory_study_v2.execution import PortfolioAccount, Position


def test_missing_held_price_uses_last_valid_price_not_peak():
    account = PortfolioAccount(initial_capital=100000.0)
    # Manually establish position: cost 100, peak was 100, last_valid fell to 80
    account.positions["SEC_0"] = Position(
        security_id="SEC_0",
        quantity=100,
        cost_basis=100.0,
        peak_price=100.0,
        last_valid_price=80.0,
        stale_sessions=0,
        age_sessions=5,
    )
    account.cash = 90000.0

    # Valuation on session with missing close for SEC_0:
    # Close prices dict is empty
    nav = account.evaluate_close_stops_and_update_state("2020-01-10", {}, {})
    # Expected holdings = 100 * 80.0 = 8000.0 (NOT 100 * 100.0 = 10000.0!)
    expected_nav = 90000.0 + 8000.0
    assert abs(nav - expected_nav) < 1e-6, f"Expected {expected_nav}, got {nav}"
    assert account.positions["SEC_0"].stale_sessions == 1


def test_zero_empty_slots_immediate_return_no_entries_queued():
    account = PortfolioAccount(initial_capital=100000.0, max_positions=3)
    # 3 positions occupied
    account.positions["SEC_1"] = Position("SEC_1", 10, 100.0, 100.0, 100.0)
    account.positions["SEC_2"] = Position("SEC_2", 10, 100.0, 100.0, 100.0)
    account.positions["SEC_3"] = Position("SEC_3", 10, 100.0, 100.0, 100.0)

    # Queue an exit for tomorrow
    account.pending_exits["SEC_1"] = "STOP"

    # Plan entries at close with unheld candidates
    account.plan_entries_at_close(["SEC_NEW1", "SEC_NEW2"])

    # Under Section 8.1 rule: empty slots at close t = 3 - 3 = 0.
    # An exit queued for tomorrow still occupies a slot today!
    # No new entries can be queued!
    assert len(account.pending_entries) == 0, f"Expected 0 pending entries, got {account.pending_entries}"


def test_missing_terminal_quote_raises_error():
    account = PortfolioAccount(initial_capital=10000.0)
    account.positions["SEC_A"] = Position("SEC_A", 10.0, 100.0, 100.0, 100.0)

    with pytest.raises(ValueError, match="Missing or non-positive terminal close price"):
        account.execute_terminal_liquidation("2025-12-31", {})


def test_terminal_liquidation_deducts_costs_from_nav():
    account = PortfolioAccount(initial_capital=0.0, commission=0.001, slippage=0.0005)
    account.cash = 0.0
    account.positions["SEC_A"] = Position("SEC_A", 10.0, 100.0, 100.0, 100.0)

    # Establish daily history
    account.evaluate_close_stops_and_update_state("2025-12-30", {"SEC_A": 100.0}, {})

    final_nav = account.execute_terminal_liquidation("2025-12-30", {"SEC_A": 100.0})
    # sell_fill = 100 * (1 - 0.0005) = 99.95
    # gross = 10 * 99.95 = 999.5
    # comm = 0.001 * 999.5 = 0.9995
    # net_cash = 999.5 - 0.9995 = 998.5005
    expected_nav = 998.5005
    assert pytest.approx(final_nav, rel=1e-6) == expected_nav
    assert pytest.approx(account.daily_history[-1].total_nav, rel=1e-6) == expected_nav
    assert account.daily_history[-1].num_positions == 0


def test_conflicting_dividend_alias_raises_error():
    from memory_study_v2.execution import CorporateAction
    account = PortfolioAccount(initial_capital=10000.0)
    account.positions["SEC_A"] = Position("SEC_A", 10.0, 100.0, 100.0, 100.0)

    class DeprecatedAction:
        split_ratio = 1.0
        dividend_cash = 2.5

    with pytest.raises(ValueError, match="Conflicting/deprecated field 'dividend_cash'"):
        account.handle_corporate_actions_before_open("2020-01-05", {"SEC_A": DeprecatedAction()})


def test_terminal_liquidation_appends_row_when_called_directly():
    """Verify Case B: terminal liquidation called directly on new session appends reconciled ledger row (C4)."""
    import math
    account = PortfolioAccount(initial_capital=10000.0, commission=0.001, slippage=0.0005)
    # Day 1: buy
    account.plan_entries_at_close(["SEC_A"])
    account.process_open_fills("2025-12-29", {"SEC_A": 100.0}, {"SEC_A": True}, 10000.0)
    account.evaluate_close_stops_and_update_state("2025-12-29", {"SEC_A": 102.0}, {})
    
    # Day 2: hold
    account.evaluate_close_stops_and_update_state("2025-12-30", {"SEC_A": 105.0}, {})
    
    # Day 3: Terminal liquidation directly called without prior evaluate_close_stops on 2025-12-31
    final_nav = account.execute_terminal_liquidation("2025-12-31", {"SEC_A": 104.0})
    
    # Verify no duplicate sessions
    sessions = [r.session for r in account.daily_history]
    assert len(sessions) == len(set(sessions)) == 3
    assert sessions[-1] == "2025-12-31"
    
    # Verify final ledger row matches cash and final NAV
    last_row = account.daily_history[-1]
    assert pytest.approx(last_row.total_nav, rel=1e-6) == final_nav
    assert pytest.approx(last_row.cash, rel=1e-6) == account.cash
    assert last_row.holdings_value == 0.0
    assert last_row.num_positions == 0
    
    # Verify compounded return reconciliation
    compounded_equity = account.initial_capital * math.prod(1.0 + r.daily_return for r in account.daily_history)
    assert pytest.approx(compounded_equity, rel=1e-6) == final_nav


def test_terminal_liquidation_replaces_row_when_close_called_first():
    """Verify Case A: close stops evaluated then liquidated on same day replaces row without duplicate (C4)."""
    import math
    account = PortfolioAccount(initial_capital=10000.0, commission=0.001, slippage=0.0005)
    account.plan_entries_at_close(["SEC_A"])
    account.process_open_fills("2025-12-30", {"SEC_A": 100.0}, {"SEC_A": True}, 10000.0)
    account.evaluate_close_stops_and_update_state("2025-12-30", {"SEC_A": 102.0}, {})
    
    # Day 2: evaluate close stops first
    account.evaluate_close_stops_and_update_state("2025-12-31", {"SEC_A": 106.0}, {})
    assert len(account.daily_history) == 2
    pre_liquidation_nav = account.daily_history[-1].total_nav
    
    # Then execute terminal liquidation on same session
    final_nav = account.execute_terminal_liquidation("2025-12-31", {"SEC_A": 106.0})
    
    # Verify no duplicate row
    assert len(account.daily_history) == 2
    assert account.daily_history[-1].session == "2025-12-31"
    
    # Friction must reduce NAV from pre-liquidation valuation
    assert final_nav < pre_liquidation_nav
    assert pytest.approx(account.daily_history[-1].total_nav, rel=1e-6) == final_nav
    
    # Reconcile compounded returns
    compounded_equity = account.initial_capital * math.prod(1.0 + r.daily_return for r in account.daily_history)
    assert pytest.approx(compounded_equity, rel=1e-6) == final_nav
