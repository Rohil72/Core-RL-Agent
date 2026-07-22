from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Literal

import numpy as np
import pandas as pd


CombinationMode = Literal["minimum", "product"]


@dataclass(frozen=True)
class ExposureControllerConfig:
    """Configuration for causal portfolio-level exposure scaling."""

    enabled: bool = True
    target_annualized_volatility: float | None = 0.15
    volatility_lookback: int = 20
    volatility_min_observations: int = 10
    volatility_floor: float = 0.05
    maximum_volatility_scalar: float = 1.0
    drawdown_soft_limit: float | None = 0.10
    drawdown_hard_limit: float | None = 0.25
    drawdown_floor_scalar: float = 0.25
    evidence_enabled: bool = True
    confidence_reference: float = 0.60
    agreement_reference: float = 0.50
    effective_sample_size_reference: float = 20.0
    positive_breadth_reference: float = 0.40
    confidence_weight: float = 0.35
    agreement_weight: float = 0.25
    effective_sample_size_weight: float = 0.20
    positive_breadth_weight: float = 0.20
    evidence_top_fraction: float = 1.0
    evidence_score_column: str = "opportunity_score"
    seed_consensus_enabled: bool = False
    seed_vote_reference: float = 0.67
    seed_rank_dispersion_limit: float = 0.20
    seed_positive_reference: float = 0.67
    seed_vote_weight: float = 0.40
    seed_rank_stability_weight: float = 0.30
    seed_positive_weight: float = 0.30
    evidence_floor_scalar: float = 0.35
    combination: CombinationMode = "minimum"
    minimum_exposure: float = 0.0
    maximum_exposure: float = 1.0
    rebalance_threshold: float = 0.10

    def __post_init__(self) -> None:
        """Reject configurations that could produce ambiguous or leveraged exposure."""
        if self.volatility_lookback <= 0 or self.volatility_min_observations <= 0:
            raise ValueError("Volatility windows must be positive.")
        if self.volatility_min_observations > self.volatility_lookback:
            raise ValueError("volatility_min_observations cannot exceed volatility_lookback.")
        if self.target_annualized_volatility is not None and self.target_annualized_volatility <= 0:
            raise ValueError("target_annualized_volatility must be positive when enabled.")
        if not 0.0 < self.maximum_volatility_scalar <= 1.0:
            raise ValueError("maximum_volatility_scalar must lie in (0, 1].")
        if self.volatility_floor <= 0:
            raise ValueError("volatility_floor must be positive.")
        if not 0.0 <= self.minimum_exposure <= self.maximum_exposure <= 1.0:
            raise ValueError("Exposure bounds must satisfy 0 <= minimum <= maximum <= 1.")
        if not 0.0 <= self.rebalance_threshold <= 1.0:
            raise ValueError("rebalance_threshold must lie in [0, 1].")
        if not 0.0 <= self.drawdown_floor_scalar <= 1.0:
            raise ValueError("drawdown_floor_scalar must lie in [0, 1].")
        if not 0.0 <= self.evidence_floor_scalar <= 1.0:
            raise ValueError("evidence_floor_scalar must lie in [0, 1].")
        if not 0.0 < self.evidence_top_fraction <= 1.0:
            raise ValueError("evidence_top_fraction must lie in (0, 1].")
        if not self.evidence_score_column:
            raise ValueError("evidence_score_column cannot be empty.")
        if self.drawdown_soft_limit is not None or self.drawdown_hard_limit is not None:
            if self.drawdown_soft_limit is None or self.drawdown_hard_limit is None:
                raise ValueError("Both drawdown limits must be supplied together.")
            if not 0.0 <= self.drawdown_soft_limit < self.drawdown_hard_limit <= 1.0:
                raise ValueError("Drawdown limits must satisfy 0 <= soft < hard <= 1.")
        if self.combination not in {"minimum", "product"}:
            raise ValueError("combination must be 'minimum' or 'product'.")
        weights = (
            self.confidence_weight,
            self.agreement_weight,
            self.effective_sample_size_weight,
            self.positive_breadth_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("Evidence weights must be non-negative and have positive sum.")
        references = (
            self.confidence_reference,
            self.agreement_reference,
            self.effective_sample_size_reference,
            self.positive_breadth_reference,
        )
        if any(reference <= 0 for reference in references):
            raise ValueError("Evidence references must be positive.")
        seed_weights = (
            self.seed_vote_weight,
            self.seed_rank_stability_weight,
            self.seed_positive_weight,
        )
        if any(weight < 0 for weight in seed_weights) or sum(seed_weights) <= 0:
            raise ValueError("Seed-consensus weights must be non-negative and have positive sum.")
        if (
            self.seed_vote_reference <= 0
            or self.seed_rank_dispersion_limit <= 0
            or self.seed_positive_reference <= 0
        ):
            raise ValueError("Seed-consensus references must be positive.")


@dataclass(frozen=True)
class ExposureDecision:
    """Auditable portfolio exposure decision using information known at decision time."""

    target_exposure: float
    volatility_scalar: float
    drawdown_scalar: float
    evidence_scalar: float
    realized_annualized_volatility: float | None
    current_drawdown: float
    median_confidence: float | None
    median_agreement: float | None
    median_effective_sample_size: float | None
    positive_alpha_breadth: float | None
    reason: str
    evidence_candidate_count: int = 0
    mean_seed_vote_fraction: float | None = None
    median_seed_rank_stability: float | None = None
    mean_seed_score_positive_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for decision logs."""
        return asdict(self)


class CausalExposureController:
    """Scale gross exposure from lagged portfolio risk and current memory quality."""

    def __init__(self, config: ExposureControllerConfig | None = None) -> None:
        self.config = config or ExposureControllerConfig()

    def decide(
        self,
        signal_rows: pd.DataFrame,
        prior_equity: pd.DataFrame,
    ) -> ExposureDecision:
        """Return target exposure without observing current-bar or future outcomes."""
        cfg = self.config
        if not cfg.enabled:
            return ExposureDecision(1.0, 1.0, 1.0, 1.0, None, 0.0, None, None, None, None, "disabled")

        realized_volatility, volatility_scalar = self._volatility_scalar(prior_equity)
        current_drawdown, drawdown_scalar = self._drawdown_scalar(prior_equity)
        evidence = self._evidence_snapshot(signal_rows)
        evidence_scalar = evidence.scalar
        active = [volatility_scalar, drawdown_scalar, evidence_scalar]
        combined = float(np.prod(active)) if cfg.combination == "product" else float(min(active))
        target = float(np.clip(combined, cfg.minimum_exposure, cfg.maximum_exposure))

        components = {
            "volatility": volatility_scalar,
            "drawdown": drawdown_scalar,
            "evidence": evidence_scalar,
        }
        reason = min(components, key=components.get) if target < cfg.maximum_exposure else "full_exposure"
        return ExposureDecision(
            target_exposure=target,
            volatility_scalar=volatility_scalar,
            drawdown_scalar=drawdown_scalar,
            evidence_scalar=evidence_scalar,
            realized_annualized_volatility=realized_volatility,
            current_drawdown=current_drawdown,
            median_confidence=evidence.median_confidence,
            median_agreement=evidence.median_agreement,
            median_effective_sample_size=evidence.median_effective_sample_size,
            positive_alpha_breadth=evidence.positive_alpha_breadth,
            reason=reason,
            evidence_candidate_count=evidence.candidate_count,
            mean_seed_vote_fraction=evidence.mean_seed_vote_fraction,
            median_seed_rank_stability=evidence.median_seed_rank_stability,
            mean_seed_score_positive_fraction=evidence.mean_seed_score_positive_fraction,
        )

    def _volatility_scalar(self, prior_equity: pd.DataFrame) -> tuple[float | None, float]:
        cfg = self.config
        if cfg.target_annualized_volatility is None:
            return None, 1.0
        returns = _prior_returns(prior_equity).tail(cfg.volatility_lookback)
        if len(returns) < cfg.volatility_min_observations:
            return None, 1.0
        realized = float(returns.std(ddof=1) * np.sqrt(252.0))
        if not np.isfinite(realized):
            return None, 1.0
        denominator = max(realized, cfg.volatility_floor)
        scalar = float(
            np.clip(
                cfg.target_annualized_volatility / denominator,
                0.0,
                cfg.maximum_volatility_scalar,
            )
        )
        return realized, scalar

    def _drawdown_scalar(self, prior_equity: pd.DataFrame) -> tuple[float, float]:
        cfg = self.config
        if cfg.drawdown_soft_limit is None or cfg.drawdown_hard_limit is None:
            return _current_drawdown(prior_equity), 1.0
        drawdown = _current_drawdown(prior_equity)
        depth = abs(min(drawdown, 0.0))
        if depth <= cfg.drawdown_soft_limit:
            return drawdown, 1.0
        if depth >= cfg.drawdown_hard_limit:
            return drawdown, cfg.drawdown_floor_scalar
        progress = (depth - cfg.drawdown_soft_limit) / (
            cfg.drawdown_hard_limit - cfg.drawdown_soft_limit
        )
        scalar = 1.0 - progress * (1.0 - cfg.drawdown_floor_scalar)
        return drawdown, float(np.clip(scalar, cfg.drawdown_floor_scalar, 1.0))

    def _evidence_snapshot(self, signal_rows: pd.DataFrame) -> "_EvidenceSnapshot":
        """Summarize current cross-sectional memory quality without future data."""
        cfg = self.config
        candidates = _top_evidence_candidates(
            signal_rows,
            cfg.evidence_score_column,
            cfg.evidence_top_fraction,
        )
        confidence = _finite_median(candidates, "retrieval_confidence")
        agreement = _finite_median(candidates, "retrieval_agreement_score")
        sample_size = _finite_median(candidates, "retrieval_effective_sample_size")
        breadth = _positive_alpha_breadth(candidates)
        vote_fraction = _finite_mean(candidates, "seed_vote_fraction")
        rank_dispersion = _finite_median(candidates, "seed_rank_std")
        rank_stability = (
            None
            if rank_dispersion is None
            else float(np.clip(1.0 - rank_dispersion / cfg.seed_rank_dispersion_limit, 0.0, 1.0))
        )
        positive_fraction = _finite_mean(candidates, "seed_score_positive_fraction")
        if not cfg.evidence_enabled:
            return _EvidenceSnapshot(
                1.0,
                len(candidates),
                confidence,
                agreement,
                sample_size,
                breadth,
                vote_fraction,
                rank_stability,
                positive_fraction,
            )

        components = [
            (_normalise(confidence, cfg.confidence_reference), cfg.confidence_weight),
            (_normalise(agreement, cfg.agreement_reference), cfg.agreement_weight),
            (_normalise(sample_size, cfg.effective_sample_size_reference), cfg.effective_sample_size_weight),
            (_normalise(breadth, cfg.positive_breadth_reference), cfg.positive_breadth_weight),
        ]
        if cfg.seed_consensus_enabled:
            components.extend(
                [
                    (_normalise(vote_fraction, cfg.seed_vote_reference), cfg.seed_vote_weight),
                    (_normalise(rank_stability, 1.0), cfg.seed_rank_stability_weight),
                    (_normalise(positive_fraction, cfg.seed_positive_reference), cfg.seed_positive_weight),
                ]
            )
        total_weight = sum(weight for _, weight in components)
        quality = sum(score * weight for score, weight in components) / total_weight
        scalar = cfg.evidence_floor_scalar + (1.0 - cfg.evidence_floor_scalar) * quality
        return _EvidenceSnapshot(
            float(np.clip(scalar, cfg.evidence_floor_scalar, 1.0)),
            len(candidates),
            confidence,
            agreement,
            sample_size,
            breadth,
            vote_fraction,
            rank_stability,
            positive_fraction,
        )


@dataclass(frozen=True)
class _EvidenceSnapshot:
    scalar: float
    candidate_count: int
    median_confidence: float | None
    median_agreement: float | None
    median_effective_sample_size: float | None
    positive_alpha_breadth: float | None
    mean_seed_vote_fraction: float | None
    median_seed_rank_stability: float | None
    mean_seed_score_positive_fraction: float | None


def _prior_returns(equity: pd.DataFrame) -> pd.Series:
    if equity.empty or "equity" not in equity:
        return pd.Series(dtype=float)
    values = pd.to_numeric(equity["equity"], errors="coerce")
    return values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _current_drawdown(equity: pd.DataFrame) -> float:
    if equity.empty or "equity" not in equity:
        return 0.0
    values = pd.to_numeric(equity["equity"], errors="coerce").dropna()
    if values.empty:
        return 0.0
    peak = float(values.max())
    return float(values.iloc[-1] / peak - 1.0) if peak > 0 else 0.0


def _finite_median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else None


def _finite_mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if not values.empty else None


def _top_evidence_candidates(
    frame: pd.DataFrame,
    score_column: str,
    fraction: float,
) -> pd.DataFrame:
    if frame.empty or fraction >= 1.0 or score_column not in frame:
        return frame
    scores = pd.to_numeric(frame[score_column], errors="coerce")
    valid = frame.loc[scores.notna()].copy()
    if valid.empty:
        return frame
    valid["_evidence_score"] = scores.loc[valid.index]
    count = max(1, int(ceil(len(valid) * fraction)))
    return valid.nlargest(count, "_evidence_score").drop(columns="_evidence_score")


def _positive_alpha_breadth(frame: pd.DataFrame) -> float | None:
    if "retrieval_alpha_ci_low" not in frame:
        return None
    alpha = pd.to_numeric(frame["retrieval_alpha_ci_low"], errors="coerce")
    valid = alpha.notna()
    if "retrieval_ood_pass" in frame:
        valid &= frame["retrieval_ood_pass"].fillna(False).astype(bool)
    return float((alpha[valid] > 0.0).mean()) if valid.any() else None


def _normalise(value: float | None, reference: float) -> float:
    if value is None or not np.isfinite(value) or reference <= 0:
        return 0.0
    return float(np.clip(value / reference, 0.0, 1.0))
