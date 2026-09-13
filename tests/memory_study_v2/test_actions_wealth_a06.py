"""Acceptance Test A06: Split-only, dividend-only and simultaneous-action fixtures conserve expected wealth."""

import pytest
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import build_total_return_bars, CorporateAction


def test_split_only_conserves_wealth():
    """Verify a 2-for-1 split (S=2) with price halving preserves 100% total-return value."""
    # Pre-split close 100.0, post-split close 50.0
    df = pd.DataFrame({
        "session": ["2023-01-02", "2023-01-03"],
        "open": [98.0, 49.0],
        "high": [102.0, 51.0],
        "low": [97.0, 48.0],
        "close": [100.0, 50.0],
        "volume": [1000.0, 2000.0],
    })
    actions = [CorporateAction(session="2023-01-03", split_ratio=2.0, cash_dividend=0.0)]
    tr_df = build_total_return_bars(df, actions=actions)

    # TR close at t=0 is 100.0
    assert tr_df.loc[0, "tr_close"] == 100.0
    # Growth = (2.0 * 50.0 + 0) / 100.0 = 1.0 -> TR close at t=1 must be exactly 100.0
    assert pytest.approx(tr_df.loc[1, "tr_close"], rel=1e-9) == 100.0
    # Normalized volume at t=1 must be 2000 / 2 = 1000.0
    assert pytest.approx(tr_df.loc[1, "normalized_volume"], rel=1e-9) == 1000.0


def test_dividend_only_conserves_wealth():
    """Verify a cash dividend D=5.0 with price dropping by 5.0 preserves 100% total-return value."""
    # Pre-div close 100.0, post-div close 95.0, D = 5.0
    df = pd.DataFrame({
        "session": ["2023-01-02", "2023-01-03"],
        "open": [98.0, 93.0],
        "high": [102.0, 97.0],
        "low": [97.0, 92.0],
        "close": [100.0, 95.0],
        "volume": [1000.0, 1000.0],
    })
    actions = [CorporateAction(session="2023-01-03", split_ratio=1.0, cash_dividend=5.0)]
    tr_df = build_total_return_bars(df, actions=actions)

    assert tr_df.loc[0, "tr_close"] == 100.0
    # Growth = (1.0 * 95.0 + 5.0) / 100.0 = 1.0 -> TR close at t=1 must be exactly 100.0
    assert pytest.approx(tr_df.loc[1, "tr_close"], rel=1e-9) == 100.0


def test_simultaneous_split_and_dividend_conserves_wealth():
    """Verify simultaneous 2-for-1 split (S=2) and dividend D=10.0 (pre-split basis)."""
    # 1 old share at 100 gets 10 cash, remaining 90 splits 2-for-1 -> new price 45.0
    # Growth = (2.0 * 45.0 + 10.0) / 100.0 = 1.0
    df = pd.DataFrame({
        "session": ["2023-01-02", "2023-01-03"],
        "open": [98.0, 44.0],
        "high": [102.0, 46.0],
        "low": [97.0, 43.0],
        "close": [100.0, 45.0],
        "volume": [1000.0, 2000.0],
    })
    actions = [CorporateAction(session="2023-01-03", split_ratio=2.0, cash_dividend=10.0)]
    tr_df = build_total_return_bars(df, actions=actions)

    assert tr_df.loc[0, "tr_close"] == 100.0
    assert pytest.approx(tr_df.loc[1, "tr_close"], rel=1e-9) == 100.0
    assert pytest.approx(tr_df.loc[1, "normalized_volume"], rel=1e-9) == 1000.0


def test_fractional_split_quantity_retained_in_portfolio():
    """Verify that a 3-for-2 split (S=1.5) transforms 3 shares into 4.5 shares without flooring."""
    from memory_study_v2.execution import PortfolioAccount, Position, CorporateAction as ExecCorporateAction

    account = PortfolioAccount(initial_capital=1000.0)
    account.positions["SEC_X"] = Position(
        security_id="SEC_X",
        quantity=3.0,
        cost_basis=100.0,
        peak_price=100.0,
        last_valid_price=100.0,
    )

    # Apply 3-for-2 split: S=1.5
    action = ExecCorporateAction(split_ratio=1.5, cash_dividend=0.0)
    account.handle_corporate_actions_before_open("2023-01-03", {"SEC_X": action})

    pos = account.positions["SEC_X"]
    # Must retain exact fractional quantity 4.5 (not floored to 4)
    assert pos.quantity == 4.5
    assert pytest.approx(pos.cost_basis, rel=1e-9) == 100.0 / 1.5

    # Total wealth conserved at post-split price
    post_split_price = 100.0 / 1.5
    equity = account.current_equity({"SEC_X": post_split_price})
    assert pytest.approx(equity, rel=1e-9) == 1000.0 + 3.0 * 100.0
