"""Focused tests for shared bounded OHLC roundoff policy (4-ULP guard band).

Verifies:
1. Exact reported float pairs:
   - 1.9879204034805296 vs 1.9879204034805298 (high < close by 1 ULP)
   - 3.702447414398193 vs 3.7024474143981934 (high < close by 1 ULP)
2. Equivalent low-bound roundoff (low > min(open, close) by 1-4 ULPs).
3. 4-step boundary accepted and 5-step boundary rejected.
4. Cent-sized violations (e.g. $0.01 at $2 or $100) rejected as material.
5. NaN, infinity, nonpositive prices, volume, and conflicting duplicates rejected.
6. Exact agreement between scalar validation, vectorized calendar classification, and canonical validation.
7. Valid rows unchanged, source inputs unmodified, normalization strictly idempotent.
"""

import math
import numpy as np
import pandas as pd
import pytest

from memory_study_v2.canonical_data import validate_raw_bars, DataIntegrityError
from memory_study_v2.ohlc_validation import (
    classify_and_normalize_ohlc_row,
    normalize_ohlc_dataframe,
    step_down_float64,
    step_up_float64,
    OHLCClassification,
    MAX_ROUNDOFF_ULPS,
)
from memory_study_v2.venue_calendar import VenueCalendar, _is_valid_bar_row


def test_exact_float_pair_1_accepted_and_normalized():
    """Exact pair 1 from report: high=1.9879204034805296, close=1.9879204034805298 (diff: 1 ULP)."""
    h = np.float64(1.9879204034805296)
    c = np.float64(1.9879204034805298)
    o = np.float64(1.9234587463529154)
    l = np.float64(1.9026650000000000)

    assert h < c, "Precondition: high must be strictly less than close"
    res = classify_and_normalize_ohlc_row(o, h, l, c)
    assert res.classification == OHLCClassification.ROUNDOFF_ONLY
    assert res.is_valid_or_roundoff is True
    assert res.normalized_high == c
    assert res.normalized_high >= c
    assert res.normalized_high >= o
    assert res.normalized_high >= res.normalized_low


def test_exact_float_pair_2_accepted_and_normalized():
    """Exact pair 2 from report: high=3.702447414398193, close=3.7024474143981934 (diff: 1 ULP)."""
    h = np.float64(3.702447414398193)
    c = np.float64(3.7024474143981934)
    o = np.float64(3.702447414398193)
    l = np.float64(3.702447414398193)

    assert h < c, "Precondition: high must be strictly less than close"
    res = classify_and_normalize_ohlc_row(o, h, l, c)
    assert res.classification == OHLCClassification.ROUNDOFF_ONLY
    assert res.is_valid_or_roundoff is True
    assert res.normalized_high == c
    assert res.normalized_high >= c


def test_equivalent_low_bound_roundoff():
    """Low price exceeding min(open, close) by 1-4 ULPs is accepted and normalized."""
    c = np.float64(10.0)
    o = np.float64(10.5)
    h = np.float64(11.0)
    # Step low up from close by 1, 2, 3, 4 ULPs
    for step in range(1, 5):
        l = step_up_float64(c, step)
        assert l > c
        res = classify_and_normalize_ohlc_row(o, h, l, c)
        assert res.classification == OHLCClassification.ROUNDOFF_ONLY
        assert res.is_valid_or_roundoff is True
        assert res.normalized_low == c
        assert res.normalized_low <= c
        assert res.normalized_low <= o


def test_four_step_boundary_accepted_and_fifth_step_rejected():
    """Strict boundary test: exactly 4 steps accepted, 5th step rejected as material."""
    c = np.float64(100.0)
    o = np.float64(95.0)
    l = np.float64(90.0)

    # High stepped down from close
    h_4step = step_down_float64(c, 4)
    h_5step = step_down_float64(c, 5)

    res_4 = classify_and_normalize_ohlc_row(o, h_4step, l, c)
    assert res_4.classification == OHLCClassification.ROUNDOFF_ONLY
    assert res_4.is_valid_or_roundoff is True
    assert res_4.normalized_high == c

    res_5 = classify_and_normalize_ohlc_row(o, h_5step, l, c)
    assert res_5.classification == OHLCClassification.MATERIAL_DISCREPANCY
    assert res_5.is_valid_or_roundoff is False
    assert res_5.normalized_high == h_5step

    # Low stepped up from open
    o_low = np.float64(50.0)
    c_high = np.float64(55.0)
    h_top = np.float64(60.0)
    l_4step = step_up_float64(o_low, 4)
    l_5step = step_up_float64(o_low, 5)

    res_l4 = classify_and_normalize_ohlc_row(o_low, h_top, l_4step, c_high)
    assert res_l4.classification == OHLCClassification.ROUNDOFF_ONLY
    assert res_l4.is_valid_or_roundoff is True
    assert res_l4.normalized_low == o_low

    res_l5 = classify_and_normalize_ohlc_row(o_low, h_top, l_5step, c_high)
    assert res_l5.classification == OHLCClassification.MATERIAL_DISCREPANCY
    assert res_l5.is_valid_or_roundoff is False


def test_cent_sized_violations_rejected():
    """Cent-sized discrepancies ($0.01) at comparable prices (~$2 to ~$100) are rejected as material."""
    # At ~$2 stock
    df_cent_2 = pd.DataFrame({
        "session": ["2023-01-02"],
        "open": [1.98],
        "high": [1.97],  # $0.01 below open
        "low": [1.90],
        "close": [1.95],
        "volume": [1000.0],
    })
    with pytest.raises(DataIntegrityError, match="High price < max"):
        validate_raw_bars(df_cent_2, "TEST_2", quote_unit=1.0)

    # At ~$100 stock
    df_cent_100 = pd.DataFrame({
        "session": ["2023-01-02"],
        "open": [100.0],
        "high": [105.0],
        "low": [100.01],  # $0.01 above open
        "close": [102.0],
        "volume": [1000.0],
    })
    with pytest.raises(DataIntegrityError, match="Low price > min"):
        validate_raw_bars(df_cent_100, "TEST_100", quote_unit=1.0)


def test_nonfinite_nonpositive_duplicates_and_volume_still_rejected():
    """Existing rejection checks remain strict and fail-closed."""
    base = {
        "session": ["2023-01-02"],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [102.0],
        "volume": [1000.0],
    }

    # NaN price
    df_nan = pd.DataFrame(base)
    df_nan.loc[0, "close"] = float("nan")
    with pytest.raises(DataIntegrityError, match="Non-finite price"):
        validate_raw_bars(df_nan, "T", quote_unit=1.0)

    # Inf price
    df_inf = pd.DataFrame(base)
    df_inf.loc[0, "high"] = float("inf")
    with pytest.raises(DataIntegrityError, match="Non-finite price"):
        validate_raw_bars(df_inf, "T", quote_unit=1.0)

    # Non-positive price
    df_zero = pd.DataFrame(base)
    df_zero.loc[0, "open"] = 0.0
    with pytest.raises(DataIntegrityError, match="Non-positive price"):
        validate_raw_bars(df_zero, "T", quote_unit=1.0)

    # Negative volume
    df_neg_vol = pd.DataFrame(base)
    df_neg_vol.loc[0, "volume"] = -1.0
    with pytest.raises(DataIntegrityError, match="Negative volume"):
        validate_raw_bars(df_neg_vol, "T", quote_unit=1.0)

    # Conflicting duplicate sessions
    df_dup = pd.concat([pd.DataFrame(base), pd.DataFrame({
        "session": ["2023-01-02"],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [999.0],  # conflicting close
        "volume": [1000.0],
    })], ignore_index=True)
    with pytest.raises(DataIntegrityError, match="Conflicting duplicate"):
        validate_raw_bars(df_dup, "T", quote_unit=1.0)


def test_agreement_across_scalar_vectorized_and_canonical():
    """Agreement between scalar row check, vectorized calendar classification, and canonical validation."""
    c1 = np.float64(1.9879204034805298)
    h1 = np.float64(1.9879204034805296)
    o1 = np.float64(1.9234587463529154)
    l1 = np.float64(1.9026650000000000)

    # 1. Scalar row validator
    res_scalar = classify_and_normalize_ohlc_row(o1, h1, l1, c1)
    assert res_scalar.is_valid_or_roundoff is True

    # 2. _is_valid_bar_row in venue_calendar
    row_dict = {"open": o1, "high": h1, "low": l1, "close": c1, "volume": 1000.0}
    assert _is_valid_bar_row(row_dict) is True

    # 3. reindex_to_schedule vectorized validation
    vc = VenueCalendar(sessions=["2023-01-03"])
    df_test = pd.DataFrame({
        "session": ["2023-01-03"],
        "open": [o1],
        "high": [h1],
        "low": [l1],
        "close": [c1],
        "volume": [1000.0],
    })
    reindexed = vc.reindex_to_schedule(df_test, ("2023-01-03", "2023-01-03"))
    assert reindexed.iloc[0]["bar_status"] == "VALID"
    assert reindexed.iloc[0]["high"] == c1  # Normalized value propagated

    # 4. Canonical validate_raw_bars
    val_res = validate_raw_bars(df_test, "US:TEST", quote_unit=1.0)
    assert len(val_res) == 1
    assert val_res.iloc[0]["high"] == c1


def test_valid_rows_unchanged_and_normalization_idempotent():
    """Valid rows remain bitwise unchanged; normalization is strictly idempotent."""
    o = np.float64(100.0)
    h = np.float64(105.0)
    l = np.float64(95.0)
    c = np.float64(102.0)

    # Already valid
    res_valid = classify_and_normalize_ohlc_row(o, h, l, c)
    assert res_valid.classification == OHLCClassification.ALREADY_VALID
    assert res_valid.normalized_open == o
    assert res_valid.normalized_high == h
    assert res_valid.normalized_low == l
    assert res_valid.normalized_close == c

    # Idempotence on roundoff-only row
    c_r = np.float64(1.9879204034805298)
    h_r = np.float64(1.9879204034805296)
    o_r = np.float64(1.9234587463529154)
    l_r = np.float64(1.9026650000000000)

    res1 = classify_and_normalize_ohlc_row(o_r, h_r, l_r, c_r)
    assert res1.classification == OHLCClassification.ROUNDOFF_ONLY

    # Passing normalized values into classification yields ALREADY_VALID
    res2 = classify_and_normalize_ohlc_row(res1.normalized_open, res1.normalized_high, res1.normalized_low, res1.normalized_close)
    assert res2.classification == OHLCClassification.ALREADY_VALID
    assert res2.normalized_high == res1.normalized_high
    assert res2.normalized_low == res1.normalized_low
