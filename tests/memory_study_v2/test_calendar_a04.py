"""Acceptance Test A04: Native-session holidays and cross-market UTC availability boundaries (R11)."""

from datetime import datetime, timezone
import pandas as pd
import pytest

from memory_study_v2.canonical_data import (
    align_to_venue_calendar,
    check_cross_market_utc_availability,
)


def test_native_holidays_not_invented_as_zero_return():
    """Verify that native exchange holidays remain absent sessions rather than zero-return bars (A04/R11)."""
    # Official venue calendar for US Thanksgiving week
    official_venue_calendar = ["2023-11-20", "2023-11-21", "2023-11-22", "2023-11-24"]  # 2023-11-23 Thanksgiving absent

    # Raw feed contains trading sessions
    raw_df = pd.DataFrame({
        "session": ["2023-11-20", "2023-11-21", "2023-11-22", "2023-11-24"],
        "close": [100.0, 101.0, 102.0, 103.0],
    })

    aligned = align_to_venue_calendar(raw_df, official_venue_calendar)

    # Assert Thanksgiving session is absent, not filled with zero return
    assert "2023-11-23" not in aligned["session"].values
    assert len(aligned) == 4


def test_cross_market_utc_availability_boundary():
    """Verify that cross-market availability strictly uses UTC timestamp (A04/R11)."""
    # Market A (Asia, closes 07:00 UTC) vs Market B (US, closes 21:00 UTC)
    t_market_a_close_utc = datetime(2023, 5, 10, 7, 0, tzinfo=timezone.utc)
    t_market_b_close_utc = datetime(2023, 5, 10, 21, 0, tzinfo=timezone.utc)

    # Market B close cannot be accessed at Market A close
    assert check_cross_market_utc_availability(t_market_a_close_utc, t_market_b_close_utc) is False
    # Market A close CAN be accessed at Market B close
    assert check_cross_market_utc_availability(t_market_b_close_utc, t_market_a_close_utc) is True
