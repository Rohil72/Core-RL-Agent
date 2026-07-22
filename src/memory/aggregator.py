from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AggregationConfig:
    """Controls robust evidence aggregation from retrieved experiences."""

    method: str = "gaussian"
    gaussian_bandwidth: float | None = None
    trim_fraction: float = 0.10
    tail_fraction: float = 0.25
    confidence_interval_z: float = 1.96


@dataclass(frozen=True)
class DistributionEstimate:
    """Distribution summary for one historical outcome variable."""

    expected: float | None
    median: float | None
    variance: float | None
    std: float | None
    p10: float | None
    p25: float | None
    p75: float | None
    p90: float | None
    ci_low: float | None
    ci_high: float | None
    tail_risk: float | None
    trimmed_mean: float | None
    sample_count: int


def distance_weights(distances: np.ndarray, config: AggregationConfig) -> np.ndarray:
    """Convert neighbor distances into normalized evidence weights."""
    d = np.asarray(distances, dtype=float)
    finite = np.isfinite(d)
    weights = np.zeros_like(d, dtype=float)
    if not finite.any():
        return weights
    safe = np.maximum(d[finite], 1e-9)
    if config.method == "inverse_distance":
        weights[finite] = 1.0 / safe
    elif config.method == "uniform":
        weights[finite] = 1.0
    else:
        bandwidth = config.gaussian_bandwidth
        if bandwidth is None:
            bandwidth = float(np.nanmedian(safe)) or 1.0
        bandwidth = max(float(bandwidth), 1e-9)
        weights[finite] = np.exp(-0.5 * (safe / bandwidth) ** 2)
    total = float(weights.sum())
    return weights / total if total > 0 else weights


def estimate_distribution(values: np.ndarray, weights: np.ndarray, config: AggregationConfig) -> DistributionEstimate:
    """Estimate a robust weighted distribution summary."""
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w >= 0)
    if not mask.any():
        return DistributionEstimate(None, None, None, None, None, None, None, None, None, None, None, None, 0)
    vals = vals[mask]
    w = w[mask]
    w = w / w.sum() if w.sum() > 0 else np.full(len(vals), 1.0 / len(vals))
    expected = float(np.average(vals, weights=w))
    variance = float(np.average((vals - expected) ** 2, weights=w))
    std = float(np.sqrt(max(variance, 0.0)))
    ess = effective_sample_size(w)
    se = std / np.sqrt(max(ess, 1.0))
    sorted_vals = np.sort(vals)
    trim_n = int(np.floor(len(sorted_vals) * max(config.trim_fraction, 0.0)))
    trimmed = sorted_vals[trim_n : len(sorted_vals) - trim_n] if len(sorted_vals) - 2 * trim_n > 0 else sorted_vals
    tail_n = max(1, int(np.ceil(len(sorted_vals) * max(config.tail_fraction, 0.01))))
    return DistributionEstimate(
        expected=expected,
        median=float(np.median(vals)),
        variance=variance,
        std=std,
        p10=float(np.percentile(vals, 10)),
        p25=float(np.percentile(vals, 25)),
        p75=float(np.percentile(vals, 75)),
        p90=float(np.percentile(vals, 90)),
        ci_low=float(expected - config.confidence_interval_z * se),
        ci_high=float(expected + config.confidence_interval_z * se),
        tail_risk=float(np.mean(sorted_vals[:tail_n])),
        trimmed_mean=float(np.mean(trimmed)),
        sample_count=int(len(vals)),
    )


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the effective independent sample size implied by weights."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w >= 0)]
    if w.size == 0 or w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))
