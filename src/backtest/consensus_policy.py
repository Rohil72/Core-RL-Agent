from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.market_memory_backtester import PolicyConfig, tradability_reason


@dataclass(frozen=True)
class ConsensusSignalConfig:
    """Causal seed-consensus entry and persistent-exit signal construction."""

    minimum_votes: int = 2
    exit_smoothing_span: int = 1
    exit_score_quantile: float = 0.50

    def __post_init__(self) -> None:
        if self.minimum_votes < 0:
            raise ValueError("minimum_votes cannot be negative.")
        if self.exit_smoothing_span <= 0:
            raise ValueError("exit_smoothing_span must be positive.")
        if not 0.0 <= self.exit_score_quantile <= 1.0:
            raise ValueError("exit_score_quantile must lie in [0, 1].")


MEDIAN_EVIDENCE_COLUMNS = (
    "retrieval_expected_upside",
    "retrieval_expected_alpha",
    "retrieval_alpha_p10",
    "retrieval_alpha_p90",
    "retrieval_alpha_ci_low",
    "retrieval_alpha_ci_high",
    "retrieval_alpha_std",
    "retrieval_expected_downside",
    "retrieval_upside_p10",
    "retrieval_upside_p90",
    "retrieval_upside_ci_low",
    "retrieval_upside_ci_high",
    "retrieval_downside_cvar",
    "retrieval_outcome_std",
    "retrieval_expected_holding_period",
    "retrieval_upside_before_drawdown_prob",
    "retrieval_confidence",
    "retrieval_agreement_score",
    "retrieval_disagreement_score",
    "retrieval_effective_sample_size",
    "retrieval_entropy",
    "retrieval_distance_weighted_confidence",
    "retrieval_historical_diversity",
    "retrieval_neighbor_count",
    "retrieval_same_ticker_rate",
    "retrieval_cross_ticker_rate",
    "retrieval_median_distance",
    "opportunity_quality",
    "pred_utility_q50",
)


def build_consensus_signals(
    seed_frames: dict[int, pd.DataFrame],
    policy: PolicyConfig,
    config: ConsensusSignalConfig | None = None,
) -> pd.DataFrame:
    """Aggregate incompatible seed spaces at the evidence layer, never in latent coordinates."""
    cfg = config or ConsensusSignalConfig()
    if len(seed_frames) < max(cfg.minimum_votes, 1):
        raise ValueError("Not enough seed frames for the configured vote threshold.")
    parts: list[pd.DataFrame] = []
    for seed, source in sorted(seed_frames.items()):
        frame = source.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["ticker"] = frame["ticker"].astype(str)
        frame["seed"] = int(seed)
        frame["seed_score_rank"] = frame.groupby("timestamp")["opportunity_score"].rank(
            method="average", pct=True
        )
        frame["seed_tradable"] = frame.apply(
            lambda row: tradability_reason(row, policy, "opportunity_score") is None,
            axis=1,
        )
        parts.append(frame)
    stacked = pd.concat(parts, ignore_index=True, sort=False)
    rows: list[dict[str, object]] = []
    for (timestamp, ticker), group in stacked.groupby(["timestamp", "ticker"], sort=True):
        base = group.iloc[0].to_dict()
        scores = pd.to_numeric(group["opportunity_score"], errors="coerce")
        ranks = pd.to_numeric(group["seed_score_rank"], errors="coerce")
        votes = group["seed_tradable"].astype(bool)
        base.update(
            {
                "timestamp": timestamp,
                "ticker": str(ticker),
                "seed_model_count": int(len(group)),
                "seed_vote_count": int(votes.sum()),
                "seed_vote_fraction": float(votes.mean()),
                "seed_votes": ",".join(
                    f"{int(seed)}:{int(vote)}" for seed, vote in zip(group["seed"], votes)
                ),
                "consensus_economic_score": float(scores.median()),
                "consensus_entry_rank": float(ranks.mean()),
                "consensus_exit_raw": float(scores.quantile(cfg.exit_score_quantile)),
                "seed_score_std": float(scores.std(ddof=0)),
                "seed_score_range": float(scores.max() - scores.min()),
                "seed_rank_std": float(ranks.std(ddof=0)),
                "seed_score_positive_fraction": float((scores > 0.0).mean()),
                "retrieval_ood_pass": bool(group["retrieval_ood_pass"].fillna(False).mean() >= 0.5),
            }
        )
        for column in MEDIAN_EVIDENCE_COLUMNS:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                base[column] = float(values.median()) if len(values) else np.nan
        rows.append(base)
    consensus = pd.DataFrame(rows).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    consensus["consensus_exit_score"] = consensus.groupby("ticker", sort=False)[
        "consensus_exit_raw"
    ].transform(
        lambda values: values.ewm(span=cfg.exit_smoothing_span, adjust=False, min_periods=1).mean()
    )
    consensus["score_persistence_ratio"] = consensus["consensus_exit_score"] / (
        consensus["consensus_economic_score"].abs() + 1e-6
    )
    consensus["consensus_pass"] = (
        True if cfg.minimum_votes == 0 else consensus["seed_vote_count"] >= cfg.minimum_votes
    )
    return consensus.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def consensus_row_filter(
    row: pd.Series,
    policy: PolicyConfig,
    minimum_votes: int,
) -> bool:
    """Apply vote and ordinary A2 evidence gates to an ensemble entry."""
    if minimum_votes > 0 and int(row.get("seed_vote_count", 0)) < minimum_votes:
        return False
    return tradability_reason(row, policy, "consensus_economic_score") is None
