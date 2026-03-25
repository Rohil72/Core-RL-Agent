from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

_DEFAULT_TEMPLATE_MAX_SCORE = 7.0


@dataclass
class Cycle:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_idx: int
    end_idx: int
    duration_days: int
    net_return: float
    peak_date: pd.Timestamp
    peak_idx: int
    max_drawdown_from_start: float = 0.0
    quality_score: float = 0.5
    cycle_score: float = 0.0


def _safe_clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))


def _build_volatility_series(prices: pd.Series, window: int) -> np.ndarray:
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window=window, min_periods=max(5, window // 3)).std()
    fallback_vol = float(returns.std(skipna=True))
    if not np.isfinite(fallback_vol) or fallback_vol <= 0:
        fallback_vol = 0.02
    rolling_vol = rolling_vol.fillna(fallback_vol)
    return rolling_vol.to_numpy(dtype=np.float64)


def _normalize_template_score(values: pd.Series) -> float:
    if values.empty:
        return 0.5
    score = float(values.mean(skipna=True))
    if not np.isfinite(score):
        return 0.5
    return _safe_clip(score / _DEFAULT_TEMPLATE_MAX_SCORE)


def _candidate_quality_score(
    feature_frame: pd.DataFrame | None,
    t_start: int,
    t_end: int,
) -> float:
    if feature_frame is None or feature_frame.empty:
        return 0.5

    early_end = min(len(feature_frame), t_start + 21)
    confirm_end = min(len(feature_frame), max(t_end + 1, t_start + 63))

    early_window = feature_frame.iloc[t_start:early_end]
    confirm_window = feature_frame.iloc[t_start:confirm_end]

    components: list[tuple[float, float]] = []

    if "tech_minervini_gate" in early_window.columns and not early_window.empty:
        gate_value = float(early_window["tech_minervini_gate"].max())
        components.append((_safe_clip(gate_value), 0.25))

    if (
        "tech_minervini_template_score" in early_window.columns
        and not early_window.empty
    ):
        template_value = _normalize_template_score(
            early_window["tech_minervini_template_score"]
        )
        components.append((template_value, 0.35))

    if (
        "fund_minervini_score" in confirm_window.columns
        and "fund_report_available" in confirm_window.columns
        and not confirm_window.empty
        and float(confirm_window["fund_report_available"].max()) > 0.0
    ):
        fund_value = float(confirm_window["fund_minervini_score"].max(skipna=True))
        if np.isfinite(fund_value):
            components.append((_safe_clip(fund_value), 0.40))

    if not components:
        return 0.5

    weighted_sum = sum(value * weight for value, weight in components)
    total_weight = sum(weight for _, weight in components)
    return weighted_sum / max(total_weight, 1e-9)


def _drawdown_scores(
    worst_drawdown: float,
    realized_vol: float,
    soft_pullback_limit: float,
    hard_pullback_limit: float,
    volatility_multiplier: float,
) -> tuple[bool, float]:
    vol_buffer = max(realized_vol, 0.0) * volatility_multiplier
    soft_limit = max(soft_pullback_limit, vol_buffer)
    hard_limit = max(hard_pullback_limit, soft_limit * 1.75)

    if worst_drawdown < -hard_limit:
        return False, 0.0
    if worst_drawdown >= -soft_limit:
        return True, 1.0
    if hard_limit <= soft_limit:
        return True, 0.5

    excess = abs(worst_drawdown) - soft_limit
    span = hard_limit - soft_limit
    return True, _safe_clip(1.0 - (excess / span))


def detect_cycles(
    prices: pd.Series,
    min_duration_days: int = 21,
    max_duration_days: int = 252,
    min_return: float = 0.30,
    feature_frame: pd.DataFrame | None = None,
    soft_pullback_limit: float = 0.05,
    hard_pullback_limit: float = 0.12,
    volatility_window: int = 21,
    volatility_multiplier: float = 2.0,
    min_cycle_score: float = 0.58,
    min_quality_score: float = 0.20,
) -> List[Cycle]:
    """
    Detect price cycles with soft structural margins and optional quality filtering.

    The detector still requires significant profit, but it no longer invalidates
    a cycle on minor pullbacks. When engineered features are supplied, it can
    softly reward Minervini-style technical/fundamental quality and filter out
    obviously low-quality runs.
    """
    if prices.empty:
        return []
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if min_duration_days <= 0 or max_duration_days <= 0:
        raise ValueError("Duration parameters must be positive")
    if min_duration_days > max_duration_days:
        raise ValueError("min_duration_days must be <= max_duration_days")
    if min_return < 0:
        raise ValueError("min_return must be non-negative")
    if soft_pullback_limit < 0 or hard_pullback_limit < 0:
        raise ValueError("Pullback limits must be non-negative")

    price_values = prices.to_numpy(dtype=np.float64)
    dates = prices.index
    n = len(prices)

    if n <= min_duration_days:
        return []

    aligned_features = None
    if feature_frame is not None and not feature_frame.empty:
        aligned_features = feature_frame.copy()
        aligned_features.index = pd.to_datetime(aligned_features.index, utc=True)
        aligned_features = aligned_features.reindex(dates)

    rolling_vol = _build_volatility_series(prices, window=volatility_window)
    candidates: list[Cycle] = []

    for t_start in range(n - min_duration_days):
        p_start = price_values[t_start]
        if p_start <= 0:
            continue

        best_candidate: Cycle | None = None

        for duration in range(min_duration_days, max_duration_days + 1):
            t_end = t_start + duration
            if t_end >= n:
                break

            p_end = price_values[t_end]
            ret = (p_end / p_start) - 1.0
            if ret < min_return:
                continue

            window_prices = price_values[t_start : t_end + 1]
            trough = np.min(window_prices)
            worst_drawdown = float((trough / p_start) - 1.0)
            valid_structure, drawdown_score = _drawdown_scores(
                worst_drawdown=worst_drawdown,
                realized_vol=float(rolling_vol[t_start]),
                soft_pullback_limit=soft_pullback_limit,
                hard_pullback_limit=hard_pullback_limit,
                volatility_multiplier=volatility_multiplier,
            )
            if not valid_structure:
                continue

            actual_peak_rel = int(np.argmax(window_prices))
            actual_peak_idx = t_start + actual_peak_rel
            peak_price = float(window_prices[actual_peak_rel])
            peak_capture_ratio = p_end / peak_price if peak_price > 0 else 0.0
            peak_capture_score = _safe_clip((peak_capture_ratio - 0.75) / 0.25)
            return_score = _safe_clip(
                ret / (min_return * 2.0 if min_return > 0 else 1.0)
            )
            duration_score = _safe_clip(duration / max_duration_days)
            quality_score = _candidate_quality_score(aligned_features, t_start, t_end)

            if aligned_features is not None and quality_score < min_quality_score:
                continue

            cycle_score = (
                0.38 * return_score
                + 0.24 * drawdown_score
                + 0.14 * peak_capture_score
                + 0.08 * duration_score
                + 0.16 * quality_score
            )
            if cycle_score < min_cycle_score:
                continue

            candidate = Cycle(
                start_date=dates[t_start],
                end_date=dates[t_end],
                start_idx=t_start,
                end_idx=t_end,
                duration_days=duration,
                net_return=float(ret),
                peak_date=dates[actual_peak_idx],
                peak_idx=actual_peak_idx,
                max_drawdown_from_start=worst_drawdown,
                quality_score=quality_score,
                cycle_score=cycle_score,
            )

            if best_candidate is None:
                best_candidate = candidate
                continue

            candidate_key = (
                candidate.cycle_score,
                candidate.net_return,
                candidate.peak_idx - candidate.end_idx,
                -candidate.max_drawdown_from_start,
            )
            current_key = (
                best_candidate.cycle_score,
                best_candidate.net_return,
                best_candidate.peak_idx - best_candidate.end_idx,
                -best_candidate.max_drawdown_from_start,
            )
            if candidate_key > current_key:
                best_candidate = candidate

        if best_candidate is not None:
            candidates.append(best_candidate)

    candidates.sort(
        key=lambda c: (
            c.cycle_score,
            c.net_return,
            c.quality_score,
            -c.max_drawdown_from_start,
        ),
        reverse=True,
    )

    final_cycles = []
    occupied_indices = np.zeros(n, dtype=bool)
    for candidate in candidates:
        if not np.any(occupied_indices[candidate.start_idx : candidate.end_idx + 1]):
            final_cycles.append(candidate)
            occupied_indices[candidate.start_idx : candidate.end_idx + 1] = True

    final_cycles.sort(key=lambda c: c.start_idx)
    return final_cycles
