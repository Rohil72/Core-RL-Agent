from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.memory.aggregator import AggregationConfig, DistributionEstimate, distance_weights, estimate_distribution
from src.memory.confidence import ConfidenceConfig, ConfidenceEstimate, estimate_confidence


@dataclass(frozen=True)
class EvidenceSummary:
    """Historical reasoning summary produced for one query state."""

    upside: DistributionEstimate
    alpha: DistributionEstimate
    downside: DistributionEstimate
    holding_period: DistributionEstimate
    path_quality: DistributionEstimate
    confidence: ConfidenceEstimate
    opportunity_quality: float | None
    score: float | None
    neighbor_count: int
    same_ticker_rate: float | None
    cross_ticker_rate: float | None
    median_distance: float | None
    ood_pass: bool

    def to_signal_payload(self) -> dict:
        """Flatten evidence into policy-compatible signal columns."""
        return {
            "retrieval_expected_upside": self.upside.expected,
            "retrieval_expected_alpha": self.alpha.expected,
            "retrieval_alpha_p10": self.alpha.p10,
            "retrieval_alpha_p90": self.alpha.p90,
            "retrieval_alpha_ci_low": self.alpha.ci_low,
            "retrieval_alpha_ci_high": self.alpha.ci_high,
            "retrieval_alpha_std": self.alpha.std,
            "retrieval_expected_downside": self.downside.expected,
            "retrieval_upside_p10": self.upside.p10,
            "retrieval_upside_p90": self.upside.p90,
            "retrieval_upside_ci_low": self.upside.ci_low,
            "retrieval_upside_ci_high": self.upside.ci_high,
            "retrieval_downside_cvar": self.downside.tail_risk,
            "retrieval_outcome_std": self.upside.std,
            "retrieval_expected_holding_period": self.holding_period.expected,
            "retrieval_upside_before_drawdown_prob": self.path_quality.expected,
            "retrieval_confidence": self.confidence.confidence,
            "retrieval_agreement_score": self.confidence.agreement_score,
            "retrieval_disagreement_score": self.confidence.disagreement_score,
            "retrieval_effective_sample_size": self.confidence.effective_sample_size,
            "retrieval_entropy": self.confidence.retrieval_entropy,
            "retrieval_distance_weighted_confidence": self.confidence.distance_weighted_confidence,
            "retrieval_historical_diversity": self.confidence.historical_diversity,
            "retrieval_neighbor_count": self.neighbor_count,
            "retrieval_same_ticker_rate": self.same_ticker_rate,
            "retrieval_cross_ticker_rate": self.cross_ticker_rate,
            "retrieval_median_distance": self.median_distance,
            "retrieval_ood_pass": self.ood_pass,
            "opportunity_quality": self.opportunity_quality,
            "opportunity_score": self.score,
        }

    def to_dict(self) -> dict:
        """Return a JSON-serialisable evidence summary."""
        return asdict(self)


def build_evidence_summary(
    neighbors: pd.DataFrame,
    distances: np.ndarray,
    query_ticker: str,
    *,
    upside_col: str,
    alpha_col: str | None,
    downside_col: str,
    path_quality_col: str | None,
    holding_period_col: str | None,
    aggregation: AggregationConfig,
    confidence: ConfidenceConfig,
    score_weights: dict[str, Any],
) -> EvidenceSummary:
    """Aggregate retrieved historical experiences into reasoning evidence."""
    weights = distance_weights(distances, aggregation)
    upside = neighbors[upside_col].to_numpy(dtype=float)
    alpha = neighbors[alpha_col].to_numpy(dtype=float) if alpha_col else upside
    downside = neighbors[downside_col].to_numpy(dtype=float)
    path_quality = (
        neighbors[path_quality_col].to_numpy(dtype=float)
        if path_quality_col and path_quality_col in neighbors
        else np.where(upside > np.abs(downside), 1.0, 0.0)
    )
    holding_period = (
        neighbors[holding_period_col].to_numpy(dtype=float)
        if holding_period_col and holding_period_col in neighbors
        else np.full(len(neighbors), np.nan)
    )

    up_dist = estimate_distribution(upside, weights, aggregation)
    alpha_dist = estimate_distribution(alpha, weights, aggregation)
    down_dist = estimate_distribution(downside, weights, aggregation)
    hold_dist = estimate_distribution(holding_period, weights, aggregation)
    path_dist = estimate_distribution(path_quality, weights, aggregation)
    conf = estimate_confidence(
        outcomes=alpha,
        distances=distances,
        weights=weights,
        tickers=neighbors["ticker"].astype(str).to_numpy(),
        config=confidence,
    )
    same = float(np.mean(neighbors["ticker"].astype(str).to_numpy() == str(query_ticker))) if len(neighbors) else None
    quality = _quality(up_dist.expected, down_dist.expected, path_dist.expected, conf.confidence)
    score = _score(alpha_dist, up_dist, down_dist, path_dist, conf, score_weights)
    return EvidenceSummary(
        upside=up_dist,
        alpha=alpha_dist,
        downside=down_dist,
        holding_period=hold_dist,
        path_quality=path_dist,
        confidence=conf,
        opportunity_quality=quality,
        score=score,
        neighbor_count=int(len(neighbors)),
        same_ticker_rate=same,
        cross_ticker_rate=1.0 - same if same is not None else None,
        median_distance=float(np.nanmedian(distances)) if len(distances) else None,
        ood_pass=True,
    )


def _quality(upside: float | None, downside: float | None, path: float | None, confidence: float) -> float | None:
    if upside is None or downside is None:
        return None
    path_val = path if path is not None else 0.0
    return float((upside - abs(downside)) * (0.5 + 0.5 * confidence) + 0.05 * path_val)


def _score(
    alpha: DistributionEstimate,
    upside: DistributionEstimate,
    downside: DistributionEstimate,
    path: DistributionEstimate,
    confidence: ConfidenceEstimate,
    score_weights: dict[str, Any],
) -> float | None:
    if upside.expected is None or downside.expected is None:
        return None
    if score_weights.get("score_mode") == "alpha_lcb":
        if alpha.expected is None or alpha.ci_low is None:
            return None
        return float(
            alpha.expected
            + 0.50 * alpha.ci_low
            - 0.50 * abs(downside.tail_risk or downside.expected)
            - 0.25 * (alpha.std or 0.0)
            + 0.01 * (path.expected or 0.0)
        )
    return float(
        score_weights.get("expected_upside_weight", 1.0) * upside.expected
        + score_weights.get("path_quality_weight", 0.10) * (path.expected or 0.0)
        - score_weights.get("downside_weight", 0.80) * abs(downside.expected)
        - score_weights.get("uncertainty_weight", 0.20) * (upside.std or 0.0)
        + score_weights.get("confidence_weight", 0.10) * confidence.confidence
        - score_weights.get("disagreement_weight", 0.10) * confidence.disagreement_score
    )
