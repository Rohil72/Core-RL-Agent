import numpy as np
import pandas as pd

from src.backtest.market_memory_backtester import PolicyConfig, run_long_only_backtest
from src.policy.opportunity_allocator import (
    OpportunityAllocatorConfig,
    build_opportunity_features,
    fit_calibrated_obvious_allocator,
    fit_nonlinear_allocator,
    predict_calibrated_obvious_allocator,
    predict_nonlinear_score,
    scores_to_allocations,
)


def _evidence_frame(rows: int = 20) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime("2022-01-01", utc=True) + pd.to_timedelta(index // 5, unit="D"),
            "ticker": [f"T{i % 5}" for i in index],
            "consensus_economic_score": 0.01 + 0.002 * index,
            "consensus_entry_rank": (index % 5 + 1) / 5,
            "retrieval_expected_alpha": -0.02 + 0.003 * index,
            "retrieval_alpha_ci_low": -0.03 + 0.003 * index,
            "retrieval_alpha_p10": -0.04 + 0.003 * index,
            "retrieval_expected_upside": 0.04 + 0.004 * index,
            "retrieval_downside_cvar": -0.12 + 0.002 * index,
            "retrieval_outcome_std": 0.08 + 0.001 * index,
            "retrieval_median_distance": 2.0 + 0.01 * index,
            "retrieval_confidence": 0.4 + 0.01 * (index % 5),
            "retrieval_agreement_score": 0.3 + 0.02 * (index % 5),
            "retrieval_distance_weighted_confidence": 0.2,
            "retrieval_historical_diversity": 0.8,
            "retrieval_entropy": 0.9,
            "retrieval_cross_ticker_rate": 1.0,
            "retrieval_upside_before_drawdown_prob": 0.6,
            "retrieval_effective_sample_size": 24.0,
            "seed_vote_fraction": (index % 5) / 4,
            "seed_score_positive_fraction": (index % 5) / 4,
            "seed_rank_std": 0.02 + 0.01 * (index % 5),
            "retrieval_alpha_std": 0.03 + 0.001 * index,
            "retrieval_ood_pass": True,
            "decision_net_alpha": -0.02 + 0.004 * index,
            "decision_mae": -0.08 + 0.001 * index,
            "decision_is_mature": True,
        }
    )


def test_opportunity_features_are_invariant_to_return_scale():
    frame = _evidence_frame()
    scaled = frame.copy()
    return_columns = [
        "consensus_economic_score",
        "retrieval_expected_alpha",
        "retrieval_alpha_ci_low",
        "retrieval_alpha_p10",
        "retrieval_expected_upside",
        "retrieval_downside_cvar",
        "retrieval_outcome_std",
        "retrieval_alpha_std",
    ]
    scaled[return_columns] *= 10.0

    original, names = build_opportunity_features(frame)
    transformed, transformed_names = build_opportunity_features(scaled)

    assert names == transformed_names
    np.testing.assert_allclose(original[names], transformed[names], atol=1e-6)


def test_nonlinear_allocator_fits_and_emits_finite_scores():
    frame = pd.concat([_evidence_frame(100), _evidence_frame(100)], ignore_index=True)
    frame["timestamp"] = pd.date_range("2022-01-01", periods=len(frame), freq="D", tz="UTC")
    config = OpportunityAllocatorConfig(max_iter=20, max_leaf_nodes=7, min_samples_leaf=5)

    model, names = fit_nonlinear_allocator(frame, config)
    scores = predict_nonlinear_score(model, frame.iloc[:20], names)

    assert len(names) >= 20
    assert len(scores) == 20
    assert np.isfinite(scores).all()


def test_score_rank_maps_to_fixed_allocation_grid():
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2023-01-01", tz="UTC")] * 5,
            "score": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    actions = scores_to_allocations(frame, frame["score"])

    assert actions.tolist() == [0.0, 0.0, 0.25, 0.75, 1.0]


def test_entry_allocator_can_abstain_and_fall_through_to_next_candidate():
    rows = []
    for day in range(3):
        timestamp = pd.Timestamp("2023-01-01", tz="UTC") + pd.Timedelta(days=day)
        for ticker, score, allocation in (("AAA", 0.9, 0.0), ("BBB", 0.8, 0.25)):
            rows.append(
                {
                    "timestamp": timestamp,
                    "ticker": ticker,
                    "open": 100.0,
                    "close": 100.0,
                    "opportunity_score": score,
                    "retrieval_expected_upside": 0.10,
                    "retrieval_expected_downside": -0.02,
                    "retrieval_confidence": 1.0,
                    "entry_allocation_fraction": allocation,
                }
            )

    trades, _ = run_long_only_backtest(
        pd.DataFrame(rows),
        PolicyConfig(top_k=1, min_hold_days=10, max_hold_days=20, slippage_bps=0),
        entry_allocation_col="entry_allocation_fraction",
    )

    assert len(trades) == 1
    assert trades.iloc[0]["ticker"] == "BBB"
    assert trades.iloc[0]["entry_allocation_fraction"] == 0.25
    assert np.isclose(trades.iloc[0]["cost_basis"], 25_000.0)


def test_calibrated_obvious_allocator_uses_absolute_utility_and_support_veto():
    frame = pd.concat([_evidence_frame(100), _evidence_frame(100)], ignore_index=True)
    frame["timestamp"] = pd.date_range("2022-01-01", periods=len(frame), freq="D", tz="UTC")
    config = OpportunityAllocatorConfig(
        absolute_utility_thresholds=(0.0, 0.02, 0.04, 0.08),
        support_floor_quantile=0.20,
        support_reference_quantile=0.50,
    )

    model = fit_calibrated_obvious_allocator(frame, config)
    evaluation = frame.iloc[-20:].copy()
    evaluation["timestamp"] = evaluation["timestamp"] + pd.Timedelta(days=365)
    evaluation["retrieval_confidence"] = 1.0
    evaluation["retrieval_agreement_score"] = 1.0
    evaluation["seed_vote_fraction"] = 1.0
    evaluation["seed_score_positive_fraction"] = 1.0
    supported = predict_calibrated_obvious_allocator(model, evaluation, config)

    weak = evaluation.copy()
    weak["retrieval_confidence"] = 0.0
    weak["retrieval_agreement_score"] = 0.0
    weak["seed_vote_fraction"] = 0.0
    weak["seed_score_positive_fraction"] = 0.0
    vetoed = predict_calibrated_obvious_allocator(model, weak, config)

    assert model.fit_end < evaluation["timestamp"].max()
    assert supported["allocator_score"].notna().all()
    assert (supported["entry_allocation_fraction"] > 0.0).any()
    assert vetoed["support_veto"].all()
    assert (vetoed["entry_allocation_fraction"] == 0.0).all()
