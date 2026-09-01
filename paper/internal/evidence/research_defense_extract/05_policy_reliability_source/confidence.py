from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.memory.aggregator import effective_sample_size


@dataclass(frozen=True)
class ConfidenceConfig:
    """Weights used to turn historical agreement into confidence."""

    min_effective_sample_size: float = 5.0
    agreement_weight: float = 0.35
    density_weight: float = 0.25
    diversity_weight: float = 0.20
    sample_weight: float = 0.20
    reference_distance: float | None = None


@dataclass(frozen=True)
class ConfidenceEstimate:
    """Agreement and confidence diagnostics for retrieved evidence."""

    confidence: float
    agreement_score: float
    disagreement_score: float
    effective_sample_size: float
    retrieval_entropy: float
    distance_weighted_confidence: float
    historical_diversity: float


def estimate_confidence(
    outcomes: np.ndarray,
    distances: np.ndarray,
    weights: np.ndarray,
    tickers: np.ndarray,
    config: ConfidenceConfig | None = None,
) -> ConfidenceEstimate:
    """Estimate confidence from neighbor agreement, density, diversity, and sample size."""
    cfg = config or ConfidenceConfig()
    vals = np.asarray(outcomes, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w >= 0)
    if not mask.any():
        return ConfidenceEstimate(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    vals = vals[mask]
    w = w[mask]
    d = np.asarray(distances, dtype=float)[mask]
    t = np.asarray(tickers, dtype=str)[mask]
    w = w / w.sum() if w.sum() > 0 else np.full(len(vals), 1.0 / len(vals))

    positive_mass = float(w[vals > 0].sum())
    negative_mass = float(w[vals <= 0].sum())
    agreement = abs(positive_mass - negative_mass)
    disagreement = 1.0 - agreement
    ess = effective_sample_size(w)
    entropy = _normalised_entropy(w)
    median_distance = float(np.nanmedian(np.maximum(d, 0.0))) if len(d) else float("inf")
    if cfg.reference_distance is not None:
        reference = max(float(cfg.reference_distance), 1e-9)
        density = float(np.exp(-median_distance / reference)) if np.isfinite(median_distance) else 0.0
    else:
        density = float(1.0 / (1.0 + median_distance)) if np.isfinite(median_distance) else 0.0
    diversity = _normalised_entropy_from_labels(t)
    sample_score = min(1.0, ess / max(cfg.min_effective_sample_size, 1e-9))
    confidence = (
        cfg.agreement_weight * agreement
        + cfg.density_weight * density
        + cfg.diversity_weight * diversity
        + cfg.sample_weight * sample_score
    )
    return ConfidenceEstimate(
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        agreement_score=float(agreement),
        disagreement_score=float(disagreement),
        effective_sample_size=float(ess),
        retrieval_entropy=float(entropy),
        distance_weighted_confidence=float(density),
        historical_diversity=float(diversity),
    )


def _normalised_entropy(weights: np.ndarray) -> float:
    w = weights[weights > 0]
    if w.size <= 1:
        return 0.0
    entropy = float(-np.sum(w * np.log2(w)))
    return entropy / float(np.log2(w.size))


def _normalised_entropy_from_labels(labels: np.ndarray) -> float:
    if labels.size <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    entropy = float(-np.sum(p * np.log2(p)))
    return entropy / float(np.log2(len(counts))) if len(counts) > 1 else 0.0
