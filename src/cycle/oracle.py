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

ORACLE_TIME_IDX_COL = "oracle_time_idx"
ORACLE_START_DATE_COL = "oracle_cycle_start_date"
ORACLE_END_DATE_COL = "oracle_cycle_end_date"
ORACLE_PEAK_DATE_COL = "oracle_cycle_peak_date"

EVENT_TARGET_COLUMNS = [
    "event_peak_offset_63",
    "event_drawdown_offset_63",
    "event_upside_before_drawdown_126",
    "event_upside_hit_126",
    "event_drawdown_hit_126",
]


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


def _first_hit_offset(returns: np.ndarray, threshold: float, direction: str) -> int | None:
    if direction == "up":
        hits = np.flatnonzero(returns >= threshold)
    elif direction == "down":
        hits = np.flatnonzero(returns <= threshold)
    else:
        raise ValueError(f"Unsupported hit direction: {direction}")
    if len(hits) == 0:
        return None
    return int(hits[0] + 1)


def _build_event_outcome_targets(
    prices: np.ndarray,
    peak_horizon: int = 63,
    drawdown_horizon: int = 63,
    ordering_horizon: int = 126,
    upside_threshold: float = 0.20,
    drawdown_threshold: float = -0.10,
) -> dict[str, np.ndarray]:
    n = len(prices)
    peak_offset = np.full(n, np.nan, dtype=np.float32)
    drawdown_offset = np.full(n, np.nan, dtype=np.float32)
    upside_before_drawdown = np.full(n, np.nan, dtype=np.float32)
    upside_hit = np.full(n, np.nan, dtype=np.float32)
    drawdown_hit = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        base_price = float(prices[i])
        if base_price <= 0:
            continue

        peak_end = min(n, i + peak_horizon + 1)
        peak_window = prices[i + 1 : peak_end]
        if len(peak_window) > 0:
            peak_idx = int(np.argmax(peak_window)) + 1
            peak_offset[i] = float(peak_idx / peak_horizon)

        drawdown_end = min(n, i + drawdown_horizon + 1)
        drawdown_window = prices[i + 1 : drawdown_end]
        if len(drawdown_window) > 0:
            drawdown_idx = int(np.argmin(drawdown_window)) + 1
            drawdown_offset[i] = float(drawdown_idx / drawdown_horizon)

        ordering_end = min(n, i + ordering_horizon + 1)
        ordering_window = prices[i + 1 : ordering_end]
        if len(ordering_window) == 0:
            continue
        future_returns = (ordering_window / base_price) - 1.0
        first_upside = _first_hit_offset(
            future_returns,
            threshold=upside_threshold,
            direction="up",
        )
        first_drawdown = _first_hit_offset(
            future_returns,
            threshold=drawdown_threshold,
            direction="down",
        )
        upside_hit[i] = 1.0 if first_upside is not None else 0.0
        drawdown_hit[i] = 1.0 if first_drawdown is not None else 0.0
        upside_before_drawdown[i] = (
            1.0
            if first_upside is not None
            and (first_drawdown is None or first_upside < first_drawdown)
            else 0.0
        )

    return {
        "event_peak_offset_63": peak_offset,
        "event_drawdown_offset_63": drawdown_offset,
        "event_upside_before_drawdown_126": upside_before_drawdown,
        "event_upside_hit_126": upside_hit,
        "event_drawdown_hit_126": drawdown_hit,
    }


def ensure_event_outcome_targets(
    price_df: pd.DataFrame,
    peak_horizon: int = 63,
    drawdown_horizon: int = 63,
    ordering_horizon: int = 126,
    upside_threshold: float = 0.20,
    drawdown_threshold: float = -0.10,
) -> pd.DataFrame:
    """
    Add self-supervised event targets derived only from future prices.

    These targets are independent of the cycle detector. They teach the model
    event timing and reward/risk ordering directly from realized market paths.
    """
    out = price_df.copy()
    if "close" not in out.columns:
        return out

    missing = [col for col in EVENT_TARGET_COLUMNS if col not in out.columns]
    if not missing:
        return out

    prices = out["close"].astype(float).to_numpy(dtype=np.float32)
    targets = _build_event_outcome_targets(
        prices,
        peak_horizon=peak_horizon,
        drawdown_horizon=drawdown_horizon,
        ordering_horizon=ordering_horizon,
        upside_threshold=upside_threshold,
        drawdown_threshold=drawdown_threshold,
    )
    for col in missing:
        out[col] = targets[col]
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
    cycle_start_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    cycle_end_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    cycle_peak_date = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")

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
        cycle_start_date.iloc[start : end + 1] = cycle.start_date
        cycle_end_date.iloc[start : end + 1] = cycle.end_date
        cycle_peak_date.iloc[start : end + 1] = cycle.peak_date

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
    out = ensure_event_outcome_targets(
        out,
        drawdown_threshold=catastrophic_return,
        upside_threshold=positive_return_threshold,
    )
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
    out[ORACLE_TIME_IDX_COL] = np.arange(n, dtype=np.int32)
    out[ORACLE_START_DATE_COL] = cycle_start_date
    out[ORACLE_END_DATE_COL] = cycle_end_date
    out[ORACLE_PEAK_DATE_COL] = cycle_peak_date
    out["oracle_cycle_duration"] = cycle_duration
    out["oracle_cycle_progress"] = cycle_progress
    out["oracle_lead_to_peak_days"] = lead_to_peak
    out["oracle_cycle_entry"] = (action == ACTION_ENTER).astype(np.int8)
    out["oracle_cycle_exit"] = (action == ACTION_EXIT).astype(np.int8)
    return out


def ensure_oracle_cycle_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill full-history oracle metadata without re-running cycle detection.

    These columns are evaluation metadata only and are not part of the model
    feature set, so they do not introduce episode-time leakage.
    """
    out = df.copy()
    n = len(out)

    if ORACLE_TIME_IDX_COL not in out.columns:
        out[ORACLE_TIME_IDX_COL] = np.arange(n, dtype=np.int32)

    if "oracle_action" not in out.columns or "oracle_cycle_id" not in out.columns:
        return out

    if "oracle_cycle_entry" not in out.columns:
        out["oracle_cycle_entry"] = (out["oracle_action"] == ACTION_ENTER).astype(
            np.int8
        )
    if "oracle_cycle_exit" not in out.columns:
        out["oracle_cycle_exit"] = (out["oracle_action"] == ACTION_EXIT).astype(np.int8)

    if "oracle_cycle_start_idx" not in out.columns:
        out["oracle_cycle_start_idx"] = np.full(n, -1, dtype=np.int32)
    if "oracle_cycle_end_idx" not in out.columns:
        out["oracle_cycle_end_idx"] = np.full(n, -1, dtype=np.int32)
    if "oracle_cycle_peak_idx" not in out.columns:
        out["oracle_cycle_peak_idx"] = np.full(n, -1, dtype=np.int32)
    if "oracle_cycle_duration" not in out.columns:
        out["oracle_cycle_duration"] = np.zeros(n, dtype=np.int32)
    if "oracle_cycle_progress" not in out.columns:
        out["oracle_cycle_progress"] = np.zeros(n, dtype=np.float32)
    if "oracle_lead_to_peak_days" not in out.columns:
        out["oracle_lead_to_peak_days"] = np.full(n, np.nan, dtype=np.float32)
    if ORACLE_START_DATE_COL not in out.columns:
        out[ORACLE_START_DATE_COL] = pd.Series(
            pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
        )
    if ORACLE_END_DATE_COL not in out.columns:
        out[ORACLE_END_DATE_COL] = pd.Series(
            pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
        )
    if ORACLE_PEAK_DATE_COL not in out.columns:
        out[ORACLE_PEAK_DATE_COL] = pd.Series(
            pd.NaT, index=out.index, dtype="datetime64[ns, UTC]"
        )

    cycle_mask = out["oracle_cycle_id"] >= 0
    for _, group in out[cycle_mask].groupby("oracle_cycle_id"):
        group_index = group.index
        if group_index.empty:
            continue

        positions = out.index.get_indexer(group_index)
        start_idx = (
            int(
                group["oracle_cycle_start_idx"]
                .where(group["oracle_cycle_start_idx"] >= 0)
                .dropna()
                .iloc[0]
            )
            if (group["oracle_cycle_start_idx"] >= 0).any()
            else int(positions.min())
        )
        end_idx = (
            int(
                group["oracle_cycle_end_idx"]
                .where(group["oracle_cycle_end_idx"] >= 0)
                .dropna()
                .iloc[0]
            )
            if (group["oracle_cycle_end_idx"] >= 0).any()
            else int(positions.max())
        )

        peak_candidates = (
            group["oracle_cycle_peak_idx"]
            .where(group["oracle_cycle_peak_idx"] >= 0)
            .dropna()
        )
        if not peak_candidates.empty:
            peak_idx = int(peak_candidates.iloc[0])
        else:
            peak_label = group["close"].idxmax()
            peak_idx = int(out.index.get_indexer([peak_label])[0])

        start_label = out.index[start_idx]
        end_label = out.index[end_idx]
        peak_label = out.index[peak_idx]
        duration = max(end_idx - start_idx + 1, 1)

        out.loc[group_index, "oracle_cycle_start_idx"] = start_idx
        out.loc[group_index, "oracle_cycle_end_idx"] = end_idx
        out.loc[group_index, "oracle_cycle_peak_idx"] = peak_idx
        out.loc[group_index, "oracle_cycle_duration"] = duration
        out.loc[group_index, ORACLE_START_DATE_COL] = start_label
        out.loc[group_index, ORACLE_END_DATE_COL] = end_label
        out.loc[group_index, ORACLE_PEAK_DATE_COL] = peak_label

        progress = np.linspace(0.0, 1.0, len(group_index), dtype=np.float32)
        out.loc[group_index, "oracle_cycle_progress"] = progress
        out.loc[group_index, "oracle_lead_to_peak_days"] = peak_idx - positions

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
        start_candidates = group.get("oracle_cycle_start_idx")
        end_candidates = group.get("oracle_cycle_end_idx")
        peak_candidates = group.get("oracle_cycle_peak_idx")

        start_idx = (
            int(start_candidates[start_candidates >= 0].iloc[0])
            if start_candidates is not None and (start_candidates >= 0).any()
            else int(df.index.get_indexer([group.index[0]])[0])
        )
        end_idx = (
            int(end_candidates[end_candidates >= 0].iloc[0])
            if end_candidates is not None and (end_candidates >= 0).any()
            else int(df.index.get_indexer([group.index[-1]])[0])
        )
        peak_idx = (
            int(peak_candidates[peak_candidates >= 0].iloc[0])
            if peak_candidates is not None and (peak_candidates >= 0).any()
            else int(df.index.get_indexer([group["close"].idxmax()])[0])
        )

        start_dates = group.get(ORACLE_START_DATE_COL)
        end_dates = group.get(ORACLE_END_DATE_COL)
        peak_dates = group.get(ORACLE_PEAK_DATE_COL)
        start_label = (
            pd.Timestamp(start_dates.dropna().iloc[0])
            if start_dates is not None and start_dates.notna().any()
            else group.index[0]
        )
        end_label = (
            pd.Timestamp(end_dates.dropna().iloc[0])
            if end_dates is not None and end_dates.notna().any()
            else group.index[-1]
        )
        peak_label = (
            pd.Timestamp(peak_dates.dropna().iloc[0])
            if peak_dates is not None and peak_dates.notna().any()
            else group["close"].idxmax()
        )

        spans.append(
            CycleSpan(
                ticker=ticker,
                start_idx=start_idx,
                end_idx=end_idx,
                start_date=start_label,
                end_date=end_label,
                start_price=float(group["close"].iloc[0]),
                end_price=float(group["close"].iloc[-1]),
                peak_idx=peak_idx,
                peak_date=peak_label if peak_idx >= 0 else None,
                cycle_id=int(cycle_id),
            )
        )
    return spans


def decode_action_spans(
    df: pd.DataFrame,
    actions: Sequence[int],
    cooldown_days: int = 0,
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
    time_idx = (
        df[ORACLE_TIME_IDX_COL].to_numpy(dtype=np.int64)
        if ORACLE_TIME_IDX_COL in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    dates = pd.to_datetime(df.index)
    closes = df["close"].to_numpy(dtype=np.float32)

    open_start: int | None = None
    spans: list[CycleSpan] = []

    for idx, raw_action in enumerate(actions):
        action = int(raw_action)

        if open_start is None:
            if action == ACTION_ENTER and gate[idx] >= 0.5:
                open_start = idx
            continue

        if action == ACTION_EXIT:
            current_date = dates[idx]
            spans.append(
                CycleSpan(
                    ticker=ticker,
                    start_idx=int(time_idx[open_start]),
                    end_idx=int(time_idx[idx]),
                    start_date=dates[open_start],
                    end_date=current_date,
                    start_price=float(closes[open_start]),
                    end_price=float(closes[idx]),
                )
            )
            open_start = None

    if open_start is not None:
        last_idx = len(df) - 1
        spans.append(
            CycleSpan(
                ticker=ticker,
                start_idx=int(time_idx[open_start]),
                end_idx=int(time_idx[last_idx]),
                start_date=dates[open_start],
                end_date=dates[last_idx],
                start_price=float(closes[open_start]),
                end_price=float(closes[last_idx]),
            )
        )

    return spans


def decode_return_spans(
    df: pd.DataFrame,
    pred_max_col: str = "future_max_return_63",
    pred_min_col: str = "future_min_return_63",
    enter_threshold: float = 0.20,
    risk_threshold: float = 0.10,
    exit_threshold: float = 0.10,
    hysteresis_days: int = 3,
    cooldown_days: int = 0,
) -> list[CycleSpan]:
    """
    Decode ENTER/EXIT spans from continuous return/risk predictions.

    - ENTER when predicted max return >= enter_threshold AND predicted min return >= -risk_threshold
    - EXIT when predicted max return < exit_threshold OR predicted min return < -risk_threshold
    Hysteresis requires the exit condition to hold for `hysteresis_days` consecutive bars.
    cooldown_days prevents immediate re-entry after a close.
    """
    if df is None or df.empty:
        return []

    ticker = (
        str(df["ticker"].iloc[0])
        if "ticker" in df.columns and not df.empty
        else "unknown"
    )

    if pred_max_col not in df.columns or pred_min_col not in df.columns:
        return []

    pred_max = df[pred_max_col].to_numpy(dtype=float)
    pred_min = df[pred_min_col].to_numpy(dtype=float)
    time_idx = (
        df[ORACLE_TIME_IDX_COL].to_numpy(dtype=np.int64)
        if ORACLE_TIME_IDX_COL in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    dates = pd.to_datetime(df.index)
    closes = df["close"].to_numpy(dtype=float) if "close" in df.columns else np.full(len(df), np.nan, dtype=float)

    n = len(pred_max)
    open_start = None
    exit_counter = 0
    last_closed_pos = -10_000
    spans: list[CycleSpan] = []

    for i in range(n):
        pm = pred_max[i]
        pn = pred_min[i]
        if not np.isfinite(pm):
            pm = float("-inf")
        if not np.isfinite(pn):
            pn = float("inf")

        if open_start is None:
            # respect cooldown period
            if i - last_closed_pos <= cooldown_days:
                continue
            if pm >= enter_threshold and pn >= -risk_threshold:
                open_start = i
                exit_counter = 0
            continue

        # if open, check exit condition
        exit_condition = (pm < exit_threshold) or (pn < -risk_threshold)
        if exit_condition:
            exit_counter += 1
        else:
            exit_counter = 0

        if exit_counter >= max(1, int(hysteresis_days)):
            start_idx = int(time_idx[open_start])
            end_idx = int(time_idx[i])
            start_date = dates[open_start]
            end_date = dates[i]
            start_price = float(closes[open_start]) if np.isfinite(closes[open_start]) else 0.0
            end_price = float(closes[i]) if np.isfinite(closes[i]) else 0.0
            spans.append(
                CycleSpan(
                    ticker=ticker,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_date=start_date,
                    end_date=end_date,
                    start_price=start_price,
                    end_price=end_price,
                )
            )
            last_closed_pos = i
            open_start = None
            exit_counter = 0

    # finalize open span
    if open_start is not None:
        last_i = n - 1
        start_idx = int(time_idx[open_start])
        end_idx = int(time_idx[last_i])
        start_date = dates[open_start]
        end_date = dates[last_i]
        start_price = float(closes[open_start]) if np.isfinite(closes[open_start]) else 0.0
        end_price = float(closes[last_i]) if np.isfinite(closes[last_i]) else 0.0
        spans.append(
            CycleSpan(
                ticker=ticker,
                start_idx=start_idx,
                end_idx=end_idx,
                start_date=start_date,
                end_date=end_date,
                start_price=start_price,
                end_price=end_price,
            )
        )

    return spans
