import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import scripts.run_local_rank_ensemble as experiment
from src.backtest.consensus_policy import ConsensusSignalConfig, build_consensus_signals
from src.backtest.market_memory_backtester import PolicyConfig
from src.eval.local_rank_ensemble import evaluate_local_rank_market
from src.eval.policy_baselines import BaselineSuiteConfig


def _signals(seed: int) -> pd.DataFrame:
    rows = []
    for day, timestamp in enumerate(
        pd.date_range("2024-01-02", periods=25, freq="B", tz="UTC")
    ):
        for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
            score = 0.02 * np.sin((day + ticker_index + seed) / 5.0)
            price = 100.0 + day * (0.1 + ticker_index * 0.03)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": price,
                    "close": price * 1.001,
                    "opportunity_score": score,
                    "pred_utility_q50": score,
                    "decision_net_alpha": score * 0.8,
                    "retrieval_expected_alpha": score,
                    "retrieval_alpha_ci_low": score - 0.01,
                    "retrieval_expected_upside": max(score + 0.04, 0.001),
                    "retrieval_expected_downside": -0.02,
                    "retrieval_downside_cvar": -0.04,
                    "retrieval_confidence": 0.75,
                    "retrieval_agreement_score": 0.70,
                    "retrieval_neighbor_count": 25,
                    "retrieval_ood_pass": True,
                }
            )
    return pd.DataFrame(rows)


def test_consensus_can_use_median_rank_without_changing_default():
    seed_frames = {seed: _signals(seed) for seed in (7, 17, 37)}
    policy = PolicyConfig(min_score=0.0, min_expected_upside=0.0)

    mean = build_consensus_signals(
        seed_frames,
        policy,
        ConsensusSignalConfig(minimum_votes=0),
    )
    median = build_consensus_signals(
        seed_frames,
        policy,
        ConsensusSignalConfig(minimum_votes=0, rank_aggregation="median"),
    )

    assert len(mean) == len(median)
    assert mean["consensus_entry_rank"].notna().all()
    assert median["consensus_entry_rank"].notna().all()


def test_local_rank_evaluation_writes_strategy_and_baselines(tmp_path):
    policy = PolicyConfig(
        top_k=1,
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=0.12,
        min_confidence=0.0,
        min_alpha_lcb=None,
        max_downside_cvar=None,
        min_neighbor_count=None,
        require_ood_pass=False,
        min_hold_days=2,
        max_hold_days=8,
    )
    result = evaluate_local_rank_market(
        seed_signals={seed: _signals(seed) for seed in (7, 17, 37)},
        policy=policy,
        consensus_config=ConsensusSignalConfig(
            minimum_votes=0,
            rank_aggregation="median",
        ),
        baseline_config=BaselineSuiteConfig(random_trials=3),
        output=tmp_path,
    )

    assert result["promotion_allowed"] is False
    assert set(result["seed_memory"]) == {
        "seed_7_memory",
        "seed_17_memory",
        "seed_37_memory",
    }
    assert (tmp_path / "baselines" / "local_rank_memory" / "metrics.json").exists()
    assert (tmp_path / "baselines" / "equal_weight_buy_hold" / "metrics.json").exists()
    assert result["diagnostics"]["realized_target"] == "decision_net_alpha"


def test_growing_bank_preserves_outcome_availability(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(Path("configs/local_rank_ensemble.yaml").read_text())
    config["experiment"]["output_root"] = "reports/local"
    config["experiment"]["markets"] = ["US"]
    config["experiment"]["seeds"] = [7]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "reports" / "local" / "run"
    key = "regional_US_seed_7"
    dates = pd.date_range("2021-01-04", periods=4, freq="B", tz="UTC")

    def frame(start: int, count: int) -> pd.DataFrame:
        chosen = dates[start : start + count]
        return pd.DataFrame(
            {
                "ticker": ["AAA"] * len(chosen),
                "timestamp": chosen,
                "decision_outcome_available_timestamp": chosen
                + pd.offsets.BDay(2),
                "decision_0": np.arange(len(chosen), dtype=float),
                "decision_mfe": 0.05,
                "decision_net_alpha": 0.02,
                "decision_mae": -0.03,
                "decision_path_quality": 0.7,
                "decision_holding_sessions": 10,
            }
        )

    adapter = output / "adapters" / key
    decisions = output / "decisions" / key
    adapter.mkdir(parents=True)
    decisions.mkdir(parents=True)
    frame(0, 2).to_parquet(adapter / "train_decisions.parquet", index=False)
    frame(1, 2).to_parquet(decisions / "development.parquet", index=False)

    result = experiment.build_memory_bank(
        str(config_path), "run", "US", 7, "development"
    )
    bank = pd.read_parquet(
        output / "growing_memory" / key / "development.parquet"
    )

    assert result["rows"] == 3
    assert bank["outcome_available_timestamp"].notna().all()
    assert (
        bank["outcome_available_timestamp"]
        >= pd.to_datetime(bank["timestamp"], utc=True)
    ).all()


def test_manifest_reuses_encoders_and_contains_no_rl(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(Path("configs/local_rank_ensemble.yaml").read_text())
    config["experiment"]["output_root"] = "reports/local"
    testbed = yaml.safe_load(Path("configs/final_research_testbed.yaml").read_text())
    testbed_path = tmp_path / "testbed.yaml"
    testbed_path.write_text(yaml.safe_dump(testbed, sort_keys=False), encoding="utf-8")
    config["experiment"]["source_testbed_config"] = str(testbed_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = experiment.build(str(config_path), "run", "python")
    manifest = yaml.safe_load(
        (tmp_path / "reports" / "local" / "run" / "experiment_manifest.yaml").read_text()
    )
    commands = [" ".join(job["command"]) for job in manifest["jobs"]]

    assert result["job_count"] == 202
    assert result["gpu_job_count"] == 72
    assert result["transformer_training_jobs"] == 0
    assert result["offline_rl_jobs"] == 0
    assert not any("train_cycle_model" in command for command in commands)
    assert not any("offline_policy" in command or "phase5-rl" in command for command in commands)
    assert sum("train_phase5_adapter_from_latents.py" in command for command in commands) == 18
    assert json.loads(
        (tmp_path / "reports" / "local" / "run" / "build_summary.json").read_text()
    )["promotion_allowed"] is False


def test_aggregate_reports_seed_lift_but_never_promotes(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    config = yaml.safe_load(Path("configs/local_rank_ensemble.yaml").read_text())
    config["experiment"]["output_root"] = "reports/local"
    config["experiment"]["markets"] = ["US"]
    config["gates"]["minimum_positive_markets"] = 1
    config["gates"]["maximum_profit_concentration"] = 1.0
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    root = tmp_path / "reports" / "local" / "run" / "evaluation" / "selection" / "US"
    payload = {
        "ensemble": {
            "total_return": 0.10,
            "sharpe": 2.1,
            "max_drawdown": -0.10,
        },
        "ensemble_sharpe_lift_vs_mean_seed": 0.25,
        "baselines": {
            "equal_weight_buy_hold": {
                "total_return": 0.04,
                "sharpe": 0.8,
                "max_drawdown": -0.08,
            }
        },
    }
    root.mkdir(parents=True)
    (root / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    for strategy, values in (
        ("local_rank_memory", [100_000.0, 101_000.0, 110_000.0]),
        ("equal_weight_buy_hold", [100_000.0, 101_000.0, 104_000.0]),
    ):
        destination = root / "baselines" / strategy
        destination.mkdir(parents=True)
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=3, freq="B", tz="UTC"),
                "equity": values,
            }
        ).to_csv(destination / "equity_curve.csv", index=False)

    result = experiment.aggregate_period(str(config_path), "run", "selection")

    assert result["gates"]["minimum_ensemble_seed_lift"] is True
    assert result["promotion_allowed"] is False
    assert result["failed_confirmation_immutable"] is True
