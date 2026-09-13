"""Acceptance & Regression Test for R04: Continuous valid calendar path enforcement."""

import numpy as np
import pandas as pd
import pytest

from memory_study_v2.labels import compute_target_labels


def test_interior_missing_bar_invalidates_target():
    # 70 bars total, but bar at t = 30 has is_valid_bar = False
    n = 70
    closes = np.full(n, 100.0)
    valid_flags = np.ones(n, dtype=bool)
    valid_flags[30] = False  # Interior invalid bar!

    df = pd.DataFrame({
        "session": [f"2020-01-{i+1:02d}" for i in range(n)],
        "tr_close": closes,
        "is_valid_bar": valid_flags,
    })
    res = compute_target_labels(df)

    # For origin t = 0: window is 0..63, which contains t = 30 -> valid_63 must be False!
    assert res["valid_63"].iloc[0] is False or res["valid_63"].iloc[0] == False
    # For origin t = 31: window is 31..94 (wait, 31 + 63 = 94 > 70, so false anyway)


def test_venue_calendar_gap_invalidates_target():
    # Scheduled sessions have 64 sessions: S0, S1, ..., S63
    scheduled = [f"2020-01-{i+1:02d}" for i in range(100)]
    # tr_bars is missing session S15
    actual_sessions = [s for i, s in enumerate(scheduled) if i != 15]
    n = len(actual_sessions)
    df = pd.DataFrame({
        "session": actual_sessions[:70],
        "tr_close": np.full(70, 100.0),
        "is_valid_bar": True,
    })
    res = compute_target_labels(df, venue_sessions=scheduled)
    # Origin 0 spans across the missing session S15 -> valid_63 must be False!
    assert res["valid_63"].iloc[0] == False


def test_missing_venue_calendar_entry_fails_closed():
    """Verify that if a bar's session is missing from venue_sessions, it fails closed (valid=False)."""
    # venue_sessions only lists sessions 0..60, missing target session at t+63
    scheduled = [f"2020-01-{i+1:02d}" for i in range(60)]
    df = pd.DataFrame({
        "session": [f"2020-01-{i+1:02d}" for i in range(70)],
        "tr_close": np.full(70, 100.0),
        "is_valid_bar": True,
    })
    res = compute_target_labels(df, venue_sessions=scheduled)
    # Origin 0 has target session 2020-01-64 which is NOT in scheduled calendar -> must be False!
    assert res["valid_63"].iloc[0] == False

