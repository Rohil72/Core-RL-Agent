from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

_DEFAULT_TEMPLATE_MAX_SCORE = 7.0
MAX_CYCLE_LENGTH_TRADING_DAYS = 42


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
    required_return: float = 0.0


@dataclass
class DetectionContext:
    dates: pd.DatetimeIndex
    price_values: np.ndarray
    aligned_features: pd.DataFrame | None
    rolling_vol: np.ndarray
    rolling_abs_return: np.ndarray


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


def _build_abs_return_series(prices: pd.Series, window: int) -> np.ndarray:
    returns = prices.pct_change().abs()
    rolling_abs = returns.rolling(window=window, min_periods=max(5, window // 3)).mean()
    fallback_abs = float(returns.mean(skipna=True))
    if not np.isfinite(fallback_abs) or fallback_abs <= 0:
        fallback_abs = 0.01
    rolling_abs = rolling_abs.fillna(fallback_abs)
    return rolling_abs.to_numpy(dtype=np.float64)


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


def _find_cycle_end_after_peak(
    price_values: np.ndarray,
    t_peak: int,
    max_end_idx: int,
    realized_vol: float,
    hard_pullback_limit: float,
    volatility_multiplier: float,
) -> int:
    """
    Cut a cycle at the first meaningful post-peak breakdown.

    If no breakdown is observed inside the allowed horizon, the cycle ends at
    the peak instead of dragging the label deep into the stage-4 decline.
    """
    peak_price = float(price_values[t_peak])
    if peak_price <= 0:
        return t_peak

    decline_limit = max(
        hard_pullback_limit, max(realized_vol, 0.0) * volatility_multiplier
    )
    for idx in range(t_peak + 1, max_end_idx + 1):
        drawdown_from_peak = (float(price_values[idx]) / peak_price) - 1.0
        if drawdown_from_peak <= -decline_limit:
            return idx
    return t_peak


def _prepare_detection_context(
    prices: pd.Series,
    feature_frame: pd.DataFrame | None,
    volatility_window: int,
) -> DetectionContext:
    ordered_prices = prices.copy()
    ordered_prices.index = pd.to_datetime(ordered_prices.index, utc=True)
    ordered_prices = ordered_prices[~ordered_prices.index.duplicated(keep="last")]
    ordered_prices = ordered_prices.sort_index()

    aligned_features = None
    if feature_frame is not None and not feature_frame.empty:
        aligned_features = feature_frame.copy()
        aligned_features.index = pd.to_datetime(aligned_features.index, utc=True)
        aligned_features = aligned_features[
            ~aligned_features.index.duplicated(keep="last")
        ]
        aligned_features = aligned_features.sort_index().reindex(ordered_prices.index)

    return DetectionContext(
        dates=ordered_prices.index,
        price_values=ordered_prices.to_numpy(dtype=np.float64),
        aligned_features=aligned_features,
        rolling_vol=_build_volatility_series(ordered_prices, window=volatility_window),
        rolling_abs_return=_build_abs_return_series(
            ordered_prices, window=volatility_window
        ),
    )


def _required_peak_return(
    context: DetectionContext,
    t_start: int,
    t_peak: int,
    min_return: float,
) -> float:
    """
    Adaptive hurdle based on local move intensity.

    Stable names stay close to the hard floor, while volatile names need a
    meaningfully larger move before the run-up is treated as a cycle.
    """
    duration = max(t_peak - t_start, 1)
    duration_scale = np.sqrt(duration)
    base_floor = max(float(min_return), 0.10)
    rate_component = float(context.rolling_abs_return[t_start]) * 1.5 * duration_scale
    vol_component = float(context.rolling_vol[t_start]) * 0.75 * duration_scale
    return max(base_floor, rate_component + vol_component)


def _build_cycle_candidate(
    context: DetectionContext,
    t_start: int,
    t_peak: int,
    max_end_idx: int,
    max_duration_days: int,
    min_return: float,
    soft_pullback_limit: float,
    hard_pullback_limit: float,
    volatility_multiplier: float,
    min_cycle_score: float,
    min_quality_score: float,
) -> Cycle | None:
    p_start = float(context.price_values[t_start])
    peak_price = float(context.price_values[t_peak])
    if p_start <= 0 or peak_price <= 0:
        return None

    peak_return = (peak_price / p_start) - 1.0
    required_peak_return = _required_peak_return(
        context=context,
        t_start=t_start,
        t_peak=t_peak,
        min_return=min_return,
    )
    if peak_return < required_peak_return:
        return None

    window_prices = context.price_values[t_start : t_peak + 1]
    trough = np.min(window_prices)
    worst_drawdown = float((trough / p_start) - 1.0)
    valid_structure, drawdown_score = _drawdown_scores(
        worst_drawdown=worst_drawdown,
        realized_vol=float(context.rolling_vol[t_start]),
        soft_pullback_limit=soft_pullback_limit,
        hard_pullback_limit=hard_pullback_limit,
        volatility_multiplier=volatility_multiplier,
    )
    if not valid_structure:
        return None

    t_end = _find_cycle_end_after_peak(
        price_values=context.price_values,
        t_peak=t_peak,
        max_end_idx=max_end_idx,
        realized_vol=float(context.rolling_vol[t_start]),
        hard_pullback_limit=hard_pullback_limit,
        volatility_multiplier=volatility_multiplier,
    )
    p_end = float(context.price_values[t_end])
    end_return = (p_end / p_start) - 1.0
    peak_capture_ratio = p_end / peak_price if peak_price > 0 else 0.0
    peak_capture_score = _safe_clip((peak_capture_ratio - 0.75) / 0.25)
    return_score = _safe_clip(
        peak_return / (required_peak_return * 2.0 if required_peak_return > 0 else 1.0)
    )
    duration_score = (
        _safe_clip((t_peak - t_start) / max_duration_days)
        if max_duration_days > 0
        else 0.0
    )
    quality_score = _candidate_quality_score(context.aligned_features, t_start, t_peak)

    if context.aligned_features is not None and quality_score < min_quality_score:
        return None

    cycle_score = (
        0.40 * return_score
        + 0.24 * drawdown_score
        + 0.12 * peak_capture_score
        + 0.08 * duration_score
        + 0.16 * quality_score
    )
    if cycle_score < min_cycle_score:
        return None

    return Cycle(
        start_date=context.dates[t_start],
        end_date=context.dates[t_end],
        start_idx=t_start,
        end_idx=t_end,
        duration_days=t_end - t_start + 1,
        net_return=end_return,
        peak_date=context.dates[t_peak],
        peak_idx=t_peak,
        max_drawdown_from_start=worst_drawdown,
        quality_score=quality_score,
        cycle_score=cycle_score,
        required_return=required_peak_return,
    )


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
    Detect cycles by walking forward through time and selecting the strongest
    valid cycle starting at the current cursor.

    This keeps labeling sequential and interpretable for the oracle targets:
    once a cycle is accepted, the detector resumes scanning from the next bar
    after that cycle ends. There is no cooldown gap between accepted cycles.
    """
    if prices.empty:
        return []
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if min_duration_days <= 0 or max_duration_days <= 0:
        raise ValueError("Duration parameters must be positive")
    effective_max_duration_days = min(
        int(max_duration_days), MAX_CYCLE_LENGTH_TRADING_DAYS
    )
    if min_duration_days > effective_max_duration_days:
        raise ValueError(
            "min_duration_days must be <= the enforced max cycle length "
            f"({MAX_CYCLE_LENGTH_TRADING_DAYS} trading days)"
        )
    if min_return < 0:
        raise ValueError("min_return must be non-negative")
    if soft_pullback_limit < 0 or hard_pullback_limit < 0:
        raise ValueError("Pullback limits must be non-negative")

    context = _prepare_detection_context(
        prices=prices,
        feature_frame=feature_frame,
        volatility_window=volatility_window,
    )
    n = len(context.price_values)

    if n <= min_duration_days:
        return []

    final_cycles: list[Cycle] = []
    t_start = 0
    while t_start < n - min_duration_days:
        p_start = float(context.price_values[t_start])
        if p_start <= 0:
            t_start += 1
            continue

        best_candidate: Cycle | None = None
        max_peak_idx = min(n - 1, t_start + effective_max_duration_days)
        running_peak_price = p_start
        for t_peak in range(t_start + min_duration_days, max_peak_idx + 1):
            peak_price = float(context.price_values[t_peak])
            if peak_price <= 0:
                continue
            if peak_price < running_peak_price:
                continue
            running_peak_price = peak_price

            candidate = _build_cycle_candidate(
                context=context,
                t_start=t_start,
                t_peak=t_peak,
                max_end_idx=max_peak_idx,
                max_duration_days=effective_max_duration_days,
                min_return=min_return,
                soft_pullback_limit=soft_pullback_limit,
                hard_pullback_limit=hard_pullback_limit,
                volatility_multiplier=volatility_multiplier,
                min_cycle_score=min_cycle_score,
                min_quality_score=min_quality_score,
            )
            if candidate is None:
                continue

            if best_candidate is None:
                best_candidate = candidate
                continue

            candidate_key = (
                candidate.peak_idx,
                candidate.net_return / max(candidate.required_return, 1e-9),
                candidate.net_return,
                (peak_price / p_start) - 1.0,
                candidate.cycle_score,
                candidate.net_return,
                -candidate.max_drawdown_from_start,
            )
            current_key = (
                best_candidate.peak_idx,
                best_candidate.net_return / max(best_candidate.required_return, 1e-9),
                best_candidate.net_return,
                (context.price_values[best_candidate.peak_idx] / p_start) - 1.0,
                best_candidate.cycle_score,
                best_candidate.net_return,
                -best_candidate.max_drawdown_from_start,
            )
            if candidate_key > current_key:
                best_candidate = candidate

        if best_candidate is not None:
            final_cycles.append(best_candidate)
            t_start = best_candidate.end_idx + 1
            continue
        t_start += 1

    return final_cycles
