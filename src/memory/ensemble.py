from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnsembleConfig:
    """Signal-level consensus rules for independently trained latent spaces."""

    minimum_votes: int = 2
    min_score: float = 0.0
    min_alpha_lcb: float = 0.0
    min_confidence: float = 0.60
    max_downside_cvar: float = 0.10


def combine_seed_signals(
    frames: dict[int, pd.DataFrame],
    config: EnsembleConfig | None = None,
) -> pd.DataFrame:
    """Combine seed evidence without averaging incompatible latent coordinates."""
    cfg = config or EnsembleConfig()
    if len(frames) < cfg.minimum_votes:
        raise ValueError("Not enough seed signal frames for the requested consensus.")
    parts = []
    for seed, frame in frames.items():
        part = frame.copy()
        part["seed"] = int(seed)
        part["timestamp"] = pd.to_datetime(part["timestamp"], utc=True)
        parts.append(part)
    stacked = pd.concat(parts, ignore_index=True)
    rows = []
    for (ticker, timestamp), group in stacked.groupby(["ticker", "timestamp"], sort=True):
        passes = group.apply(lambda row: _seed_pass(row, cfg), axis=1)
        base = group.iloc[0].to_dict()
        base.update(
            {
                "ticker": str(ticker),
                "timestamp": timestamp,
                "seed_vote_count": int(passes.sum()),
                "seed_model_count": int(len(group)),
                "seed_agreement": float(passes.mean()),
                "seed_votes": ",".join(
                    f"{int(seed)}:{int(passed)}" for seed, passed in zip(group["seed"], passes)
                ),
                "opportunity_score": _median(group, "opportunity_score"),
                "retrieval_expected_alpha": _median(group, "retrieval_expected_alpha"),
                "retrieval_alpha_ci_low": _median(group, "retrieval_alpha_ci_low"),
                "retrieval_alpha_ci_high": _median(group, "retrieval_alpha_ci_high"),
                "retrieval_expected_upside": _median(group, "retrieval_expected_upside"),
                "retrieval_expected_downside": _median(group, "retrieval_expected_downside"),
                "retrieval_downside_cvar": _minimum(group, "retrieval_downside_cvar"),
                "retrieval_confidence": _median(group, "retrieval_confidence"),
                "retrieval_neighbor_count": _median(group, "retrieval_neighbor_count"),
                "retrieval_ood_pass": bool(passes.sum() >= cfg.minimum_votes),
                "ensemble_pass": bool(passes.sum() >= cfg.minimum_votes),
            }
        )
        if passes.sum() < cfg.minimum_votes:
            base["opportunity_score"] = None
            base["retrieval_rejection_reason"] = "seed_disagreement"
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def _seed_pass(row: pd.Series, config: EnsembleConfig) -> bool:
    values = {
        "score": row.get("opportunity_score"),
        "alpha_lcb": row.get("retrieval_alpha_ci_low"),
        "confidence": row.get("retrieval_confidence"),
        "downside_cvar": row.get("retrieval_downside_cvar"),
    }
    try:
        if not all(value is not None and pd.notna(value) and np.isfinite(float(value)) for value in values.values()):
            return False
        return bool(
            float(values["score"]) >= config.min_score
            and float(values["alpha_lcb"]) >= config.min_alpha_lcb
            and float(values["confidence"]) >= config.min_confidence
            and abs(float(values["downside_cvar"])) <= config.max_downside_cvar
            and bool(row.get("retrieval_ood_pass", True))
        )
    except (TypeError, ValueError):
        return False


def _median(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if len(values) else None


def _minimum(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if len(values) else None
