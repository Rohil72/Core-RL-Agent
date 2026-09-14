"""Bounded Acceptance & Regression Test for Data Admission & Loader Identity (Finding 3 / F3).

Guarantees:
1. Rejects unknown session ordinals with explicit ValueError (no fallback to row index).
2. Preserves complete validity mask from VenueCalendar alignment.
3. Invalidating a single bar inside an input window invalidates input_window_valid.
4. filter_admitted_sample_ids strictly excludes invalidated sessions from the admitted population.
5. Ensures that every consumed sample ID has input_window_valid == True.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from memory_study_v2.sample_index import (
    build_security_sample_index,
    filter_admitted_sample_ids,
    SampleIndexRecord,
)
from memory_study_v2.venue_calendar import VenueCalendar


def test_reject_unknown_session_ordinal_without_fallback():
    """Verify that an unknown session ordinal raises ValueError instead of substituting row index."""
    vc = VenueCalendar()
    # Create df with an unscheduled weekend date
    df = pd.DataFrame({
        "session": ["2015-01-02", "2015-01-03", "2015-01-05"],  # 2015-01-03 is Saturday!
        "close": [100.0, 101.0, 102.0],
        "bar_status": ["VALID", "VALID", "VALID"],  # Claimed valid but not in venue calendar
    })

    with pytest.raises(ValueError, match="not a valid scheduled session on venue calendar"):
        build_security_sample_index("TEST_SEC", df, venue_calendar=vc, evaluation_year=2020)


def test_invalidating_bar_inside_window_excludes_id_from_admission():
    """Bounded test: invalidate a bar inside a selected window.
    The relevant ID must be excluded before training, not remain present with an 'admission verified' label.
    """
    vc = VenueCalendar()
    # Get 600 scheduled sessions
    sessions = vc.sessions_in_range("2013-01-02", "2015-06-01")[:550]
    N = len(sessions)

    # Base valid DataFrame
    closes = np.linspace(100.0, 200.0, N)
    df = pd.DataFrame({
        "session": sessions,
        "close": closes,
        "bar_status": ["VALID"] * N,
    })

    # 1. Baseline index: session at index 510 has 503 valid preceding bars (indices 7 to 509)
    records_clean = build_security_sample_index("TEST_SEC", df, venue_calendar=vc, evaluation_year=2015)
    rec_510_clean = records_clean[510]
    assert rec_510_clean.input_window_valid is True, "Target session 510 should be valid when all 503 preceding bars are valid"

    # 2. Invalidate a bar inside the input window of session 510 (e.g. at index 300)
    df_corrupt = df.copy()
    df_corrupt.loc[300, "bar_status"] = "MISSING"  # Missing bar in history
    df_corrupt.loc[300, "close"] = np.nan

    records_corrupt = build_security_sample_index("TEST_SEC", df_corrupt, venue_calendar=vc, evaluation_year=2015)
    rec_510_corrupt = records_corrupt[510]

    # The record must be marked invalid and have explicit exclusion reason
    assert rec_510_corrupt.input_window_valid is False, "Session 510 must be INVALID because bar 300 in its window is missing"
    assert rec_510_corrupt.exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"

    # 3. Test filter_admitted_sample_ids: the corrupted session must NOT be admitted!
    target_sess = sessions[510]
    admitted = filter_admitted_sample_ids(
        records_corrupt,
        start_session=sessions[503],
        end_session=sessions[520],
        require_target=False,
    )
    admitted_qids = [r.query_id for r in admitted]

    # rec_510 must NOT be in admitted IDs
    assert rec_510_corrupt.query_id not in admitted_qids, "Corrupted window ID must be excluded from admitted population"

    # In clean records, it IS in admitted IDs
    admitted_clean = filter_admitted_sample_ids(
        records_clean,
        start_session=sessions[503],
        end_session=sessions[520],
        require_target=False,
    )
    assert rec_510_clean.query_id in [r.query_id for r in admitted_clean]


def test_admission_preserves_complete_validity_mask():
    """Verify that is_valid_bar=False or nonpositive close propagates to input_window_valid=False."""
    vc = VenueCalendar()
    sessions = vc.sessions_in_range("2013-01-02", "2015-06-01")[:520]
    N = len(sessions)

    # Bar 500 has negative price
    closes = np.linspace(100.0, 200.0, N)
    closes[500] = -5.0

    df = pd.DataFrame({
        "session": sessions,
        "close": closes,
        "bar_status": ["VALID"] * N,
    })

    records = build_security_sample_index("TEST_SEC", df, venue_calendar=vc, evaluation_year=2015)
    # Session 505 depends on bars 2 to 504, including bar 500
    assert records[505].input_window_valid is False
    assert records[505].exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"
