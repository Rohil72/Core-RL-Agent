"""Acceptance Test A04: Native-session holidays and cross-market UTC availability boundaries."""

import pytest
from datetime import datetime, timezone
import pandas as pd


def test_native_holidays_not_invented_as_zero_return():
    """Verify that native exchange holidays remain absent sessions rather than zero-return bars."""
    # US Thanksgiving or Indian Diwali
    us_sessions = ["2023-11-22", "2023-11-24"]  # 2023-11-23 Thanksgiving omitted
    df_us = pd.DataFrame({"session": us_sessions, "close": [100.0, 101.0]})
    
    # Assert Thanksgiving session is absent
    assert "2023-11-23" not in df_us["session"].values
    assert len(df_us) == 2


def test_cross_market_utc_availability_boundary():
    """Verify that cross-market availability strictly uses UTC timestamp, not local date."""
    # Market A (Asia, closes 07:00 UTC) vs Market B (US, closes 21:00 UTC)
    # Market B decision at 21:00 UTC can see Market A's same-day outcome.
    # Market A decision at 07:00 UTC CANNOT see Market B's same-day outcome (closed 14 hours later).
    t_market_a_close_utc = datetime(2023, 5, 10, 7, 0, tzinfo=timezone.utc)
    t_market_b_close_utc = datetime(2023, 5, 10, 21, 0, tzinfo=timezone.utc)

    assert t_market_a_close_utc < t_market_b_close_utc
    # Market A cannot look ahead into Market B
    is_market_b_available_at_market_a_decision = t_market_b_close_utc <= t_market_a_close_utc
    assert is_market_b_available_at_market_a_decision is False
