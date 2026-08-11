import numpy as np
import pandas as pd

from src.backtest.risk_ledger import RiskLedgerConfig, TickerRiskLedger
from src.eval.representation_metrics import compute_memory_metrics
from src.memory.ensemble import EnsembleConfig, combine_seed_signals
from src.memory.market_memory import MarketMemoryConfig, score_market_memory
from src.memory.metric import FittedRetrievalMetric, MetricLearningConfig, fit_retrieval_metric
from src.memory.outcomes import RelativeOutcomeConfig, attach_relative_outcomes


def test_relative_outcomes_are_cross_sectional_and_sector_fallback_is_causal():
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "timestamp": pd.to_datetime(["2023-01-01"] * 3, utc=True),
            "sector": ["Tech", "Tech", "Health"],
            "future_return_63": [0.30, 0.10, -0.10],
        }
    )
    result = attach_relative_outcomes(
        frame,
        RelativeOutcomeConfig(minimum_sector_observations=2, sector_weight=0.5),
    )

    assert np.allclose(result["future_universe_alpha_63"], [0.20, 0.0, -0.20])
    assert np.isclose(result.loc[0, "future_blended_alpha_63"], 0.15)
    assert np.isclose(result.loc[2, "future_blended_alpha_63"], -0.20)


def test_relative_outcomes_can_be_scoped_to_market():
    frame = pd.DataFrame(
        {
            "ticker": ["US_A", "US_B", "IN_A", "IN_B"],
            "market": ["US", "US", "India", "India"],
            "timestamp": pd.to_datetime(["2023-01-01"] * 4, utc=True),
            "future_return_63": [0.30, 0.10, 0.08, 0.02],
        }
    )
    result = attach_relative_outcomes(
        frame,
        RelativeOutcomeConfig(group_column="market", sector_weight=0.0),
    )

    assert np.allclose(result["future_universe_alpha_63"], [0.10, -0.10, 0.03, -0.03])


def test_metric_archive_is_non_pickle_and_round_trips(tmp_path):
    metric = FittedRetrievalMetric(
        kind="diagonal",
        mean=np.array([1.0, 2.0], dtype=np.float32),
        scale=np.array([2.0, 4.0], dtype=np.float32),
        parameter=np.array([1.0, 4.0], dtype=np.float32),
        config={"seed": 7},
    )
    path = metric.save(tmp_path / "metric.npz")
    restored = FittedRetrievalMetric.load(path)
    values = np.array([[3.0, 6.0]], dtype=np.float32)

    assert restored.config == {"seed": 7}
    assert np.allclose(restored.transform(values), metric.transform(values))
    with np.load(path, allow_pickle=False) as archive:
        assert "config_json" in archive.files


def test_diagonal_metric_fits_frozen_latents_with_a_bounded_budget():
    dates = pd.date_range("2020-01-01", periods=120, freq="14D", tz="UTC")
    frame = pd.DataFrame(
        {
            "ticker": [f"T{i % 5}" for i in range(len(dates))],
            "timestamp": dates,
            "outcome_available_timestamp": dates,
            "latent_0": np.sin(np.arange(len(dates)) / 7.0),
            "latent_1": np.cos(np.arange(len(dates)) / 9.0),
            "future_blended_alpha_63": np.sin(np.arange(len(dates)) / 7.0) * 0.1,
            "future_min_return_63": -0.02 - np.abs(np.cos(np.arange(len(dates)) / 9.0)) * 0.05,
            "event_upside_before_drawdown_126": (np.arange(len(dates)) % 3 == 0).astype(float),
        }
    )
    metric, summary = fit_retrieval_metric(
        frame,
        MetricLearningConfig(
            kind="diagonal",
            fit_end_year=2021,
            validation_year=2022,
            steps=2,
            validation_interval=1,
            patience=2,
            batch_size=32,
            device="cpu",
            alpha_target="future_blended_alpha_63",
        ),
    )

    assert metric.kind == "diagonal"
    assert summary["steps"] == 2
    assert np.isfinite(summary["validation_loss"])
    assert np.isfinite(metric.transform(frame[["latent_0", "latent_1"]].to_numpy())).all()


def _semantic_memory_frames():
    memory = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "timestamp": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"], utc=True),
            "outcome_available_timestamp": pd.to_datetime(["2022-04-01", "2022-05-01", "2022-06-01"], utc=True),
            "latent_0": [0.0, 0.1, 0.2],
            "latent_1": [0.0, 0.0, 0.0],
            "future_max_return_63": [0.20, 0.12, 0.08],
            "future_universe_alpha_63": [0.10, 0.02, -0.02],
            "future_blended_alpha_63": [0.08, 0.01, -0.03],
            "future_min_return_63": [-0.03, -0.04, -0.08],
            "event_upside_before_drawdown_126": [1.0, 1.0, 0.0],
        }
    )
    query = pd.DataFrame(
        {
            "ticker": ["ZZZ"],
            "timestamp": pd.to_datetime(["2023-01-01"], utc=True),
            "latent_0": [0.05],
            "latent_1": [0.0],
            "future_blended_alpha_63": [0.05],
            "future_min_return_63": [-0.04],
            "event_upside_before_drawdown_126": [1.0],
        }
    )
    return memory, query


def test_alpha_evidence_ood_gate_and_real_memory_metrics():
    memory, query = _semantic_memory_frames()
    config = MarketMemoryConfig(
        k=3,
        same_ticker_mode="exclude",
        target_alpha="future_blended_alpha_63",
        score_mode="alpha_lcb",
        min_neighbors=3,
    )
    signals, neighbors = score_market_memory(
        memory,
        query,
        config,
        keep_query_columns=[
            "future_blended_alpha_63",
            "future_min_return_63",
            "event_upside_before_drawdown_126",
        ],
    )
    metrics = compute_memory_metrics(signals, neighbors, target_names=["future_blended_alpha_63"])

    assert signals.loc[0, "retrieval_neighbor_count"] == 3
    assert pd.notna(signals.loc[0, "retrieval_expected_alpha"])
    assert {"future_universe_alpha_63", "future_blended_alpha_63"}.issubset(neighbors.columns)
    assert np.isfinite(metrics["outcome_ndcg_at_25"])
    assert not np.isfinite(metrics["alpha_spearman_correlation"])

    rejected, _ = score_market_memory(
        memory,
        query.assign(latent_0=10.0),
        MarketMemoryConfig(k=3, min_neighbors=3, max_median_distance=0.01),
    )
    assert rejected.loc[0, "retrieval_rejection_reason"] == "ood"
    assert not bool(rejected.loc[0, "retrieval_ood_pass"])


def test_seed_ensemble_requires_consensus_and_uses_worst_downside():
    base = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "open": [10.0],
            "close": [10.0],
            "opportunity_score": [0.10],
            "retrieval_expected_alpha": [0.05],
            "retrieval_alpha_ci_low": [0.02],
            "retrieval_alpha_ci_high": [0.08],
            "retrieval_expected_upside": [0.12],
            "retrieval_expected_downside": [-0.04],
            "retrieval_downside_cvar": [-0.08],
            "retrieval_confidence": [0.80],
            "retrieval_neighbor_count": [25],
            "retrieval_ood_pass": [True],
        }
    )
    weak = base.assign(retrieval_alpha_ci_low=-0.01, retrieval_downside_cvar=-0.11)
    result = combine_seed_signals(
        {7: base, 17: base.assign(retrieval_downside_cvar=-0.09), 37: weak},
        EnsembleConfig(minimum_votes=2, min_confidence=0.2, max_downside_cvar=0.10),
    )

    assert result.loc[0, "seed_vote_count"] == 2
    assert result.loc[0, "ensemble_pass"]
    assert np.isclose(result.loc[0, "retrieval_downside_cvar"], -0.11)


def test_risk_ledger_expiries_only_move_forward():
    ledger = TickerRiskLedger(
        RiskLedgerConfig(cooldown_sessions=5, rolling_window_sessions=20, quarantine_sessions=10)
    )
    ledger.register_exit("AAA", 10, -0.11, "stop_loss")
    first = ledger.snapshot()["AAA"]
    ledger.register_exit("AAA", 12, -0.08, "stop_loss")
    second = ledger.snapshot()["AAA"]

    assert first["cooldown_until"] == 15
    assert second["cooldown_until"] >= first["cooldown_until"]
    assert second["quarantine_until"] == 22
    assert ledger.blocked_reason("AAA", 18) == "quarantine"
