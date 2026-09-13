"""Acceptance Test A25: Pence/account units, dividend/split basis and action-adjusted peak verified."""

import pytest
from memory_study_v2.execution import PortfolioAccount, Position
from memory_study_v2.canonical_data import CorporateAction


def test_action_dividend_credit_and_peak_adjustment():
    """Verify dividend cash credit, split ratio quantity multiplication, and peak adjustment."""
    account = PortfolioAccount(initial_capital=10000.0, max_positions=1)
    # Position: 100 shares at peak 100.0
    account.positions["SEC_A"] = Position(
        security_id="SEC_A",
        quantity=100.0,
        entry_price=100.0,
        cost_basis=10000.0,
        peak_price=100.0,
        age_sessions=5,
    )
    account.cash = 500.0

    # Simultaneous split 2-for-1 (S=2) and dividend D=10.0
    action = CorporateAction(session="2020-01-10", split_ratio=2.0, cash_dividend=10.0)
    account.handle_corporate_actions_before_open("2020-01-10", {"SEC_A": action})

    # Cash credited: 100 shares * 10 = 1000 -> cash = 1500.0
    assert pytest.approx(account.cash, abs=1e-8) == 1500.0
    # Shares: 100 * 2 = 200 shares
    assert pytest.approx(account.positions["SEC_A"].quantity, abs=1e-8) == 200.0
    # Adjusted peak: max(1e-9, (100 - 10) / 2) = 45.0
    assert pytest.approx(account.positions["SEC_A"].peak_price, abs=1e-8) == 45.0
