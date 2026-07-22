from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.cycle import cycle_detector
from src.cycle.cycle_detector import Cycle, DetectionContext


def generate_candidates(
    prices: pd.Series,
    feature_frame: pd.DataFrame | None = None,
    min_duration_days: int = 21,
    max_duration_days: int = 252,
    min_return: float = 0.0,
    soft_pullback_limit: float = 0.05,
    hard_pullback_limit: float = 0.12,
    volatility_window: int = 21,
    volatility_multiplier: float = 2.0,
) -> tuple[DetectionContext, List[Cycle]]:
    """Generate cycle candidates without applying hard heuristic thresholds.

    Returns the DetectionContext and list of Cycle objects (may be empty).
    """
    context = cycle_detector._prepare_detection_context(
        prices=prices, feature_frame=feature_frame, volatility_window=volatility_window
    )
    n = len(context.price_values)
    effective_max_duration_days = min(int(max_duration_days), cycle_detector.MAX_CYCLE_LENGTH_TRADING_DAYS)

    candidates: list[Cycle] = []
    t_start = 0
    while t_start < n - min_duration_days:
        p_start = float(context.price_values[t_start])
        if p_start <= 0:
            t_start += 1
            continue

        max_peak_idx = min(n - 1, t_start + effective_max_duration_days)
        running_peak_price = p_start
        for t_peak in range(t_start + min_duration_days, max_peak_idx + 1):
            peak_price = float(context.price_values[t_peak])
            if peak_price <= 0:
                continue
            if peak_price < running_peak_price:
                continue
            running_peak_price = peak_price

            candidate = cycle_detector._build_cycle_candidate(
                context=context,
                t_start=t_start,
                t_peak=t_peak,
                max_end_idx=max_peak_idx,
                max_duration_days=effective_max_duration_days,
                min_return=min_return,
                soft_pullback_limit=soft_pullback_limit,
                hard_pullback_limit=hard_pullback_limit,
                volatility_multiplier=volatility_multiplier,
                min_cycle_score=0.0,
                min_quality_score=0.0,
            )
            if candidate is None:
                continue
            candidates.append(candidate)

        t_start += 1

    return context, candidates


def candidate_to_feature_vector(candidate: Cycle, context: DetectionContext, feature_frame: pd.DataFrame | None = None) -> np.ndarray:
    """Convert a Cycle candidate into a numeric feature vector for scoring.

    Features include returns, duration, drawdown, required return, quality, and local volatility metrics.
    """
    t_start = candidate.start_idx
    t_peak = candidate.peak_idx
    t_end = candidate.end_idx

    p_start = float(context.price_values[t_start])
    p_peak = float(context.price_values[t_peak])
    p_end = float(context.price_values[t_end])

    peak_return = (p_peak / p_start) - 1.0
    end_return = (p_end / p_start) - 1.0
    duration = float(candidate.duration_days)
    required_return = float(candidate.required_return)
    drawdown = float(candidate.max_drawdown_from_start)
    cycle_score = float(candidate.cycle_score)
    quality = float(candidate.quality_score)
    peak_capture_ratio = p_end / p_peak if p_peak > 0 else 0.0
    peak_capture_score = float(max(0.0, min(1.0, (peak_capture_ratio - 0.75) / 0.25)))

    vol_at_start = float(context.rolling_vol[t_start]) if len(context.rolling_vol) > t_start else 0.0
    abs_return_at_start = float(context.rolling_abs_return[t_start]) if len(context.rolling_abs_return) > t_start else 0.0

    # optional feature values from aligned feature_frame at t_start
    template_score = 0.0
    gate = 0.0
    fund_score = 0.0
    if feature_frame is not None and not feature_frame.empty:
        try:
            row = feature_frame.iloc[t_start]
            template_score = float(row.get("tech_minervini_template_score", 0.0) or 0.0)
            gate = float(row.get("tech_minervini_gate", 0.0) or 0.0)
            fund_score = float(row.get("fund_minervini_score", 0.0) or 0.0)
        except Exception:
            pass

    vec = np.array(
        [
            peak_return,
            end_return,
            duration,
            required_return,
            drawdown,
            cycle_score,
            quality,
            peak_capture_score,
            vol_at_start,
            abs_return_at_start,
            template_score,
            gate,
            fund_score,
        ],
        dtype=np.float32,
    )
    return vec
