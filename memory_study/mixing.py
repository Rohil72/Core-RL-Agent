"""
Forecast Combination and Trading Score Modules.

Implements:
- Mean-return memory forecasting: r_mem = sum_i (w_i * r_i^63)
- Fixed hybrid mixture: r_hybrid = (1 - lambda) * r_base + lambda * r_mem (lambda=0.25 for M1-M4, 0 for M0)
- Trading score: S_t = r_hybrid / (vol_t + 1e-4)
- Deterministic base-predictor fallback on invalid memory
"""

from typing import Tuple, Optional
import numpy as np


def compute_memory_prediction(
    weights: np.ndarray,
    neighbour_returns: np.ndarray,
) -> float:
    """Computes weighted mean return from retrieved precedents."""
    if len(weights) == 0 or len(neighbour_returns) == 0:
        return 0.0
    return float(np.sum(weights * neighbour_returns))


def compute_hybrid_prediction(
    base_prediction: float,
    memory_prediction: float,
    lam: float = 0.25,
    memory_valid: bool = True,
) -> Tuple[float, float]:
    """
    Combines base neural prediction and memory prediction.
    If memory_valid is False, falls back to base_prediction (lam=0).
    Returns (hybrid_prediction, effective_lambda).
    """
    if not memory_valid or lam <= 0.0:
        return base_prediction, 0.0
    hybrid = (1.0 - lam) * base_prediction + lam * memory_prediction
    return float(hybrid), float(lam)


def compute_trading_score(
    hybrid_prediction: float,
    daily_volatility: float,
    eps: float = 1e-4,
) -> float:
    """
    Computes cross-sectional ranking score S_t:
    S_t = r_hybrid / (vol_t + eps).
    """
    return float(hybrid_prediction / (daily_volatility + eps))
