"""Canonical data adapter, corporate actions, and total-return series (v2).

Acceptance criteria addressed:
- A04: Native-session holidays and cross-market UTC availability boundaries.
- A05: Zero/nonfinite prices, conflicting duplicates, and missing quote units rejected.
- A06: Split-only, dividend-only, and simultaneous-action fixtures conserve expected wealth.
- A07: Appending future data/actions never changes earlier features or predictions (strict causality).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from memory_study_v2.contracts import ContractValidationError


class DataIntegrityError(Exception):
    """Raised when market data violates pricing, action, or calendar integrity invariants."""
    pass


@dataclass
class CorporateAction:
    session: str  # YYYY-MM-DD
    split_ratio: float = 1.0  # S: new shares per old share effective before today's open
    cash_dividend: float = 0.0  # D: cash distribution normalized to one pre-action share in account currency
    action_type: str = "none"  # "split", "dividend", "both"
    currency: str = ""
    metadata_verified: bool = True


def validate_raw_bars(df: pd.DataFrame, security_id: str, quote_unit: Optional[float] = None) -> pd.DataFrame:
    """Validate raw provider bars against data contract (Test A05).

    Rejects:
    - Zero, negative, or nonfinite OHLC.
    - Negative volume.
    - Inconsistent high/low bounds (high < max(open, close, low) or low > min(open, close, high)).
    - Missing or unknown quote units (must not default to 1.0).
    - Conflicting duplicate session records.
    """
    if quote_unit is None or math.isnan(quote_unit) or quote_unit <= 0:
        raise DataIntegrityError(
            f"Missing or invalid quote_unit for security '{security_id}': {quote_unit}. "
            "Quote units must be verified and cannot be defaulted."
        )

    required_cols = ["session", "open", "high", "low", "close", "volume"]
    for c in required_cols:
        if c not in df.columns:
            raise DataIntegrityError(f"Missing required column '{c}' in bars for '{security_id}'")

    if df.empty:
        return df

    # Check duplicates on session
    duplicates = df[df.duplicated(subset=["session"], keep=False)]
    if not duplicates.empty:
        # Check if duplicates are exact or conflicting
        exact_dups = df[df.duplicated(subset=required_cols, keep=False)]
        if len(exact_dups) != len(duplicates):
            raise DataIntegrityError(
                f"Conflicting duplicate session records found for security '{security_id}': "
                f"{duplicates['session'].unique().tolist()}"
            )
        # Collapse exact duplicates
        df = df.drop_duplicates(subset=["session"], keep="first").reset_index(drop=True)

    # Sort by session
    df = df.sort_values(by="session").reset_index(drop=True)

    # Check finite and positive prices
    price_cols = ["open", "high", "low", "close"]
    for p_col in price_cols:
        vals = df[p_col].to_numpy()
        if not np.all(np.isfinite(vals)):
            raise DataIntegrityError(f"Non-finite price found in '{p_col}' for '{security_id}'")
        if not np.all(vals > 0):
            raise DataIntegrityError(f"Non-positive price found in '{p_col}' for '{security_id}'")

    # Check bounds
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()

    if np.any(h < l):
        raise DataIntegrityError(f"High price < Low price for '{security_id}'")
    if np.any(h < o) or np.any(h < c):
        raise DataIntegrityError(f"High price < max(Open, Close) for '{security_id}'")
    if np.any(l > o) or np.any(l > c):
        raise DataIntegrityError(f"Low price > min(Open, Close) for '{security_id}'")

    # Check volume
    vol = df["volume"].to_numpy()
    if not np.all(np.isfinite(vol)):
        raise DataIntegrityError(f"Non-finite volume found for '{security_id}'")
    if np.any(vol < 0):
        raise DataIntegrityError(f"Negative volume found for '{security_id}'")

    return df


def build_total_return_bars(
    df: pd.DataFrame,
    actions: Optional[List[CorporateAction]] = None,
    quote_unit: float = 1.0,
) -> pd.DataFrame:
    """Build causal total-return OHLC and normalized volume series (Tests A06, A07).

    Specification (Section 3.4):
    - Mode: as_traded_with_actions.
    - S: new shares per old share effective before today's open (1.0 without split).
    - D: today's cash distribution normalized to one pre-action share in account currency.
    - Total-return gross growth for session t:
        g_t = (S * close_today + D) / close_previous
    - Starting at 100 for each uninterrupted valid segment:
        TR_close_t = TR_close_{t-1} * g_t
    - Map open/high/low with the same positive affine transformation:
        TR_price_t = TR_close_{t-1} * (S * raw_price_t + D) / raw_close_{t-1}
    - Volume: divide as-traded volume by cumulative split factor within the segment:
        norm_volume_t = raw_volume_t / cum_split_factor_t
    - Causality: appending future sessions/actions never alters earlier rows.
    """
    if df.empty:
        cols = [
            "session", "raw_open", "raw_high", "raw_low", "raw_close", "raw_volume",
            "tr_open", "tr_high", "tr_low", "tr_close", "normalized_volume",
            "split_ratio", "cash_dividend", "cum_split_factor"
        ]
        return pd.DataFrame(columns=cols)

    df_sorted = df.sort_values(by="session").reset_index(drop=True)
    n = len(df_sorted)

    # Normalize raw quotes to account currency
    raw_open = df_sorted["open"].to_numpy(dtype=np.float64) * quote_unit
    raw_high = df_sorted["high"].to_numpy(dtype=np.float64) * quote_unit
    raw_low = df_sorted["low"].to_numpy(dtype=np.float64) * quote_unit
    raw_close = df_sorted["close"].to_numpy(dtype=np.float64) * quote_unit
    raw_volume = df_sorted["volume"].to_numpy(dtype=np.float64)
    sessions = df_sorted["session"].tolist()

    # Action map by session
    action_map: Dict[str, CorporateAction] = {}
    if actions:
        for act in actions:
            action_map[act.session] = act

    tr_open = np.zeros(n, dtype=np.float64)
    tr_high = np.zeros(n, dtype=np.float64)
    tr_low = np.zeros(n, dtype=np.float64)
    tr_close = np.zeros(n, dtype=np.float64)
    norm_volume = np.zeros(n, dtype=np.float64)
    splits = np.ones(n, dtype=np.float64)
    dividends = np.zeros(n, dtype=np.float64)
    cum_split = np.ones(n, dtype=np.float64)

    # Base index = 100.0 at session 0
    cum_s = 1.0
    tr_close[0] = 100.0
    # On session 0, map open, high, low proportionately
    scale_0 = 100.0 / raw_close[0]
    tr_open[0] = raw_open[0] * scale_0
    tr_high[0] = raw_high[0] * scale_0
    tr_low[0] = raw_low[0] * scale_0
    norm_volume[0] = raw_volume[0]
    cum_split[0] = cum_s

    for t in range(1, n):
        sess = sessions[t]
        act = action_map.get(sess)
        S = act.split_ratio if act else 1.0
        D = act.cash_dividend if act else 0.0

        splits[t] = S
        dividends[t] = D
        cum_s *= S
        cum_split[t] = cum_s

        p_prev = raw_close[t - 1]
        tr_prev = tr_close[t - 1]

        # Causal gross growth
        growth = (S * raw_close[t] + D) / p_prev
        tr_close[t] = tr_prev * growth

        # Positive affine transformation for open, high, low
        tr_open[t] = tr_prev * (S * raw_open[t] + D) / p_prev
        tr_high[t] = tr_prev * (S * raw_high[t] + D) / p_prev
        tr_low[t] = tr_prev * (S * raw_low[t] + D) / p_prev

        # Volume normalized by cumulative split factor
        norm_volume[t] = raw_volume[t] / cum_s

    result = pd.DataFrame({
        "session": sessions,
        "raw_open": raw_open,
        "raw_high": raw_high,
        "raw_low": raw_low,
        "raw_close": raw_close,
        "raw_volume": raw_volume,
        "tr_open": tr_open,
        "tr_high": tr_high,
        "tr_low": tr_low,
        "tr_close": tr_close,
        "normalized_volume": norm_volume,
        "split_ratio": splits,
        "cash_dividend": dividends,
        "cum_split_factor": cum_split,
    })
    return result


def align_to_venue_calendar(
    df: pd.DataFrame,
    venue_calendar_sessions: List[str],
) -> pd.DataFrame:
    """Align raw session observations against official venue calendar (A04/R11).

    Guarantees:
    - Retains native exchange holidays as absent rows (never invents synthetic zero-return bars).
    - Preserves verified venue calendar boundaries.
    """
    valid_sessions_set = set(venue_calendar_sessions)
    filtered = df[df["session"].isin(valid_sessions_set)].copy()
    sess_order = {s: i for i, s in enumerate(venue_calendar_sessions)}
    filtered["_order"] = filtered["session"].map(sess_order)
    filtered = filtered.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return filtered


def check_cross_market_utc_availability(
    query_utc: datetime,
    source_market_close_utc: datetime,
) -> bool:
    """Check whether a market session's close is strictly available at query UTC timestamp (A04/R11).

    Cross-market availability is governed strictly by UTC wall clock:
    Available if and only if source_market_close_utc <= query_utc.
    """
    return source_market_close_utc <= query_utc
