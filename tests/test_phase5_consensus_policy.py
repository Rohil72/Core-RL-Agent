from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.consensus_policy import (
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import PolicyConfig


def _policy() -> PolicyConfig:
    return PolicyConfig(
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=0.20,
        min_confidence=0.0,
        min_alpha_lcb=0.0,
        max_downside_cvar=0.20,
        min_neighbor_count=1,
        require_ood_pass=True,
    )


def _seed_frame(seed: int, a_scores: list[float], b_scores: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=len(a_scores), tz="UTC", freq="D")
    rows = []
    for date, a_score, b_score in zip(dates, a_scores, b_scores):
        for ticker, score in (("A", a_score), ("B", b_score)):
            rows.append(
                {
                    "timestamp": date,
                    "ticker": ticker,
                    "opportunity_score": score,
                    "retrieval_expected_upside": max(score, 0.01),
                    "retrieval_expected_downside": -0.05,
                    "retrieval_expected_alpha": score,
                    "retrieval_alpha_ci_low": score,
                    "retrieval_alpha_ci_high": score + 0.02,
                    "retrieval_downside_cvar": -0.08,
                    "retrieval_confidence": 0.70,
                    "retrieval_agreement_score": 0.75,
                    "retrieval_neighbor_count": 25,
                    "retrieval_ood_pass": True,
                    "open": 100.0 + seed,
                    "close": 100.0 + seed,
                }
            )
    return pd.DataFrame(rows)


def test_consensus_records_votes_and_cross_seed_rank():
    frames = {
        7: _seed_frame(7, [0.20, 0.10], [-0.10, -0.10]),
        17: _seed_frame(17, [0.30, 0.20], [-0.10, 0.05]),
        37: _seed_frame(37, [0.40, 0.30], [-0.10, 0.04]),
    }
    result = build_consensus_signals(frames, _policy(), ConsensusSignalConfig(minimum_votes=2))
    first_a = result.loc[(result["ticker"] == "A") & (result["timestamp"] == result["timestamp"].min())].iloc[0]
    first_b = result.loc[(result["ticker"] == "B") & (result["timestamp"] == result["timestamp"].min())].iloc[0]
    assert first_a["seed_vote_count"] == 3
    assert first_a["consensus_pass"]
    assert first_a["consensus_entry_rank"] > first_b["consensus_entry_rank"]
    assert not consensus_row_filter(first_b, _policy(), minimum_votes=2)


def test_exit_smoothing_is_causal():
    frames = {
        7: _seed_frame(7, [0.30, 0.20, 0.10], [0.01, 0.01, 0.01]),
        17: _seed_frame(17, [0.40, 0.30, 0.20], [0.01, 0.01, 0.01]),
        37: _seed_frame(37, [0.50, 0.40, 0.30], [0.01, 0.01, 0.01]),
    }
    cfg = ConsensusSignalConfig(minimum_votes=2, exit_smoothing_span=3, exit_score_quantile=0.75)
    original = build_consensus_signals(frames, _policy(), cfg)
    changed = {seed: frame.copy() for seed, frame in frames.items()}
    for frame in changed.values():
        last_date = frame["timestamp"].max()
        frame.loc[(frame["ticker"] == "A") & (frame["timestamp"] == last_date), "opportunity_score"] = 100.0
    modified = build_consensus_signals(changed, _policy(), cfg)
    cutoff = original["timestamp"].sort_values().unique()[1]
    left = original.loc[(original["ticker"] == "A") & (original["timestamp"] <= cutoff), "consensus_exit_score"]
    right = modified.loc[(modified["ticker"] == "A") & (modified["timestamp"] <= cutoff), "consensus_exit_score"]
    assert np.allclose(left, right)


def test_upper_consensus_exit_exceeds_median_for_disagreeing_seeds():
    frames = {
        7: _seed_frame(7, [0.10], [0.01]),
        17: _seed_frame(17, [0.30], [0.01]),
        37: _seed_frame(37, [0.90], [0.01]),
    }
    median = build_consensus_signals(
        frames, _policy(), ConsensusSignalConfig(exit_score_quantile=0.50)
    )
    upper = build_consensus_signals(
        frames, _policy(), ConsensusSignalConfig(exit_score_quantile=0.75)
    )
    median_a = median.loc[median["ticker"] == "A", "consensus_exit_score"].iloc[0]
    upper_a = upper.loc[upper["ticker"] == "A", "consensus_exit_score"].iloc[0]
    assert upper_a > median_a


def test_consensus_uses_median_direct_adapter_score_across_seeds():
    frames = {
        7: _seed_frame(7, [0.20], [0.10]),
        17: _seed_frame(17, [0.20], [0.10]),
        37: _seed_frame(37, [0.20], [0.10]),
    }
    for frame, value in zip(frames.values(), (0.10, 0.80, 0.30)):
        frame["pred_utility_q50"] = value

    consensus = build_consensus_signals(frames, _policy())

    assert np.allclose(consensus["pred_utility_q50"], 0.30)
