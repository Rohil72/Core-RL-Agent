"""Acceptance Test A05: Zero/nonfinite prices, conflicting duplicates and missing quote units rejected."""

import pytest
import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import validate_raw_bars, DataIntegrityError


def make_valid_bars():
    return pd.DataFrame({
        "session": ["2023-01-02", "2023-01-03", "2023-01-04"],
        "open": [100.0, 102.0, 101.0],
        "high": [105.0, 104.0, 103.0],
        "low": [99.0, 101.0, 100.0],
        "close": [103.0, 101.5, 102.5],
        "volume": [1000.0, 1500.0, 1200.0],
    })


def test_valid_bars_pass():
    df = make_valid_bars()
    res = validate_raw_bars(df, "US:AAPL", quote_unit=1.0)
    assert len(res) == 3


def test_missing_quote_unit_rejected():
    df = make_valid_bars()
    with pytest.raises(DataIntegrityError, match="Missing or invalid quote_unit"):
        validate_raw_bars(df, "UK:VOD", quote_unit=None)


def test_zero_or_negative_price_rejected():
    df = make_valid_bars()
    df.loc[1, "close"] = 0.0
    with pytest.raises(DataIntegrityError, match="Non-positive price"):
        validate_raw_bars(df, "US:AAPL", quote_unit=1.0)

    df2 = make_valid_bars()
    df2.loc[1, "open"] = -5.0
    with pytest.raises(DataIntegrityError, match="Non-positive price"):
        validate_raw_bars(df2, "US:AAPL", quote_unit=1.0)


def test_nonfinite_price_rejected():
    df = make_valid_bars()
    df.loc[1, "high"] = float("nan")
    with pytest.raises(DataIntegrityError, match="Non-finite price"):
        validate_raw_bars(df, "US:AAPL", quote_unit=1.0)


def test_inconsistent_high_low_rejected():
    df = make_valid_bars()
    df.loc[1, "open"] = 102.0
    df.loc[1, "low"] = 95.0
    df.loc[1, "close"] = 100.0
    df.loc[1, "high"] = 99.0  # high (99) < open (102) but high > low (95)
    with pytest.raises(DataIntegrityError, match="High price < max"):
        validate_raw_bars(df, "US:AAPL", quote_unit=1.0)

    df2 = make_valid_bars()
    df2.loc[1, "low"] = 106.0  # low > high
    with pytest.raises(DataIntegrityError, match="High price < Low price"):
        validate_raw_bars(df2, "US:AAPL", quote_unit=1.0)


def test_negative_volume_rejected():
    df = make_valid_bars()
    df.loc[1, "volume"] = -100.0
    with pytest.raises(DataIntegrityError, match="Negative volume"):
        validate_raw_bars(df, "US:AAPL", quote_unit=1.0)


def test_conflicting_duplicates_rejected():
    df = make_valid_bars()
    # Add conflicting row for same session with different close
    conflicting_row = pd.DataFrame([{
        "session": "2023-01-03",
        "open": 102.0,
        "high": 104.0,
        "low": 101.0,
        "close": 999.0,
        "volume": 1500.0,
    }])
    df = pd.concat([df, conflicting_row], ignore_index=True)
    with pytest.raises(DataIntegrityError, match="Conflicting duplicate session records"):
        validate_raw_bars(df, "US:AAPL", quote_unit=1.0)
