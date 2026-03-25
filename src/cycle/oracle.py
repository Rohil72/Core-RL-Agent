from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.cycle.cycle_detector import Cycle

ACTION_NEUTRAL = 0
ACTION_ENTER = 1
ACTION_HOLD = 2
ACTION_EXIT = 3

ACTION_NAMES = {
    ACTION_NEUTRAL: "neutral",
    ACTION_ENTER: "enter",
    ACTION_HOLD: "hold_or_renew",
    ACTION_EXIT: "exit",
}


@dataclass
class CycleSpan:
    ticker: str
    start_idx: int
    end_idx: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_price: float
    end_price: float
    peak_idx: int | None = None
    peak_date: pd.Timestamp | None = None
    cycle_id: int | None = None

    @property
    def duration(self) -> int:
        return max(self.end_idx - self.start_idx + 1, 1)

    @property
    def return_pct(self) -> float:
        if self.start_price <= 0:
            return 0.0
        return float((self.end_price / self.start_price) - 1.0)


def _future_extreme_return(
    prices: np.ndarray,
    horizon: int,
    reducer: str,
) -> np.ndarray:
    out = np.full(len(prices), np.nan, dtype=np.float32)
    for i in range(len(prices)):
        end = min(len(prices), i + horizon + 1)
        window = prices[i + 1 : end]
        if len(window) == 0 or prices[i] <= 0:
            continue
        if reducer == "max":
            extreme = np.max(window)
        elif reducer == "min":
            extreme = np.min(window)
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
        out[i] = float((extreme / prices[i]) - 1.0)
    return out


def annotate_cycle_targets(
    price_df: pd.DataFrame,
    cycles: Sequence[Cycle],
    future_horizons: Iterable[int] = (21, 63, 126, 252),
    catastrophic_return: float = -0.10,
    positive_return_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Add action labels, cycle identifiers, and future trend targets.
    """
    out = price_df.copy()
    n = len(out)
    cycle_id = np.full(n, -1, dtype=np.int32)
    action = np.full(n, ACTION_NEUTRAL, dtype=np.int8)
    cycle_start_idx = np.full(n, -1, dtype=np.int32)
    cycle_end_idx = np.full(n, -1, dtype=np.int32)
    cycle_peak_idx = np.full(n, -1, dtype=np.int32)
    cycle_duration = np.zeros(n, dtype=np.int32)
    cycle_progress = np.zeros(n, dtype=np.float32)
    lead_to_peak = np.full(n, np.nan, dtype=np.float32)

    for cid, cycle in enumerate(cycles):
        start = int(cycle.start_idx)
        end = int(cycle.end_idx)
        if start < 0 or end >= n or end < start:
            continue

        cycle_slice = slice(start, end + 1)
        cycle_id[cycle_slice] = cid
        cycle_start_idx[cycle_slice] = start
        cycle_end_idx[cycle_slice] = end
        cycle_peak_idx[cycle_slice] = int(cycle.peak_idx)
        cycle_duration[cycle_slice] = int(cycle.duration_days)
        cycle_progress[cycle_slice] = np.linspace(
            0.0, 1.0, end - start + 1, dtype=np.float32
        )
        lead_to_peak[cycle_slice] = cycle.peak_idx - np.arange(start, end + 1)

        action[start] = ACTION_ENTER
        if end > start:
            action[start + 1 : end] = ACTION_HOLD
            action[end] = ACTION_EXIT

    close = out["close"].astype(float)
    for horizon in future_horizons:
        out[f"future_return_{horizon}"] = close.shift(-horizon) / close - 1.0

    prices = close.to_numpy(dtype=np.float32)
    out["future_max_return_63"] = _future_extreme_return(prices, 63, reducer="max")
    out["future_max_return_252"] = _future_extreme_return(prices, 252, reducer="max")
    out["future_min_return_63"] = _future_extreme_return(prices, 63, reducer="min")
    out["oracle_base_goodness"] = (
        out[[c for c in out.columns if c.startswith("future_return_")]]
        .max(axis=1)
        .clip(lower=0.0)
        .fillna(0.0)
    )
    out["oracle_hard_negative"] = (
        (out["future_min_return_63"] <= catastrophic_return)
        & (out["future_max_return_63"] < positive_return_threshold)
    ).astype(np.int8)

    out["oracle_cycle_id"] = cycle_id
    out["oracle_action"] = action
    out["oracle_cycle_start_idx"] = cycle_start_idx
    out["oracle_cycle_end_idx"] = cycle_end_idx
    out["oracle_cycle_peak_idx"] = cycle_peak_idx
    out["oracle_cycle_duration"] = cycle_duration
    out["oracle_cycle_progress"] = cycle_progress
    out["oracle_lead_to_peak_days"] = lead_to_peak
    out["oracle_cycle_entry"] = (action == ACTION_ENTER).astype(np.int8)
    out["oracle_cycle_exit"] = (action == ACTION_EXIT).astype(np.int8)
    return out


def extract_oracle_spans(df: pd.DataFrame) -> list[CycleSpan]:
    if "oracle_cycle_id" not in df.columns:
        return []

    ticker = (
        str(df["ticker"].iloc[0])
        if "ticker" in df.columns and not df.empty
        else "unknown"
    )
    spans: list[CycleSpan] = []
    for cycle_id, group in df[df["oracle_cycle_id"] >= 0].groupby("oracle_cycle_id"):
        entry_rows = (
            group[group["oracle_cycle_entry"] == 1]
            if "oracle_cycle_entry" in group.columns
            else group.iloc[:1]
        )
        exit_rows = (
            group[group["oracle_cycle_exit"] == 1]
            if "oracle_cycle_exit" in group.columns
            else group.iloc[-1:]
        )
        if entry_rows.empty:
            continue

        start_label = entry_rows.index[0]
        end_label = exit_rows.index[-1] if not exit_rows.empty else group.index[-1]
        peak_label = group["close"].idxmax()
        start_idx = int(df.index.get_indexer([start_label])[0])
        end_idx = int(df.index.get_indexer([end_label])[0])
        peak_idx = int(df.index.get_indexer([peak_label])[0])
        spans.append(
            CycleSpan(
                ticker=ticker,
                start_idx=start_idx,
                end_idx=end_idx,
                start_date=start_label,
                end_date=end_label,
                start_price=float(df.iloc[start_idx]["close"]),
                end_price=float(df.iloc[end_idx]["close"]),
                peak_idx=peak_idx,
                peak_date=peak_label if peak_idx >= 0 else None,
                cycle_id=int(cycle_id),
            )
        )
    return spans


def decode_action_spans(
    df: pd.DataFrame,
    actions: Sequence[int],
    cooldown_days: int = 42,
) -> list[CycleSpan]:
    if len(df) != len(actions):
        raise ValueError("actions length must match dataframe length")

    ticker = (
        str(df["ticker"].iloc[0])
        if "ticker" in df.columns and not df.empty
        else "unknown"
    )
    gate = (
        df["tech_minervini_gate"].to_numpy(dtype=np.float32)
        if "tech_minervini_gate" in df.columns
        else np.ones(len(df), dtype=np.float32)
    )
    dates = pd.to_datetime(df.index)
    closes = df["close"].to_numpy(dtype=np.float32)

    open_start: int | None = None
    cooldown_until: pd.Timestamp | None = None
    spans: list[CycleSpan] = []

    for idx, raw_action in enumerate(actions):
        action = int(raw_action)
        current_date = dates[idx]

        if open_start is None:
            if cooldown_until is not None and current_date < cooldown_until:
                continue
            if action == ACTION_ENTER and gate[idx] >= 0.5:
                open_start = idx
            continue

        if action == ACTION_EXIT:
            spans.append(
                CycleSpan(
                    ticker=ticker,
                    start_idx=open_start,
                    end_idx=idx,
                    start_date=dates[open_start],
                    end_date=current_date,
                    start_price=float(closes[open_start]),
                    end_price=float(closes[idx]),
                )
            )
            cooldown_until = current_date + pd.Timedelta(days=cooldown_days)
            open_start = None

    if open_start is not None:
        last_idx = len(df) - 1
        spans.append(
            CycleSpan(
                ticker=ticker,
                start_idx=open_start,
                end_idx=last_idx,
                start_date=dates[open_start],
                end_date=dates[last_idx],
                start_price=float(closes[open_start]),
                end_price=float(closes[last_idx]),
            )
        )

    return spans
