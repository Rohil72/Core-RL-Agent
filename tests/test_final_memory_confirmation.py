import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.run_final_memory_confirmation as confirmation
from scripts.run_final_memory_confirmation import _aggregate, _build, _load
from src.backtest.consensus_policy import ConsensusSignalConfig
from src.backtest.market_memory_backtester import PolicyConfig
from src.eval.locked_memory_evaluation import (
    LockedBaselineConfig,
    _equal_weight_buy_hold,
    evaluate_locked_market,
)
from src.eval.memory_reliability import ReliabilityConfig


def test_confirmation_contract_matches_frozen_candidate():
    config, policy, _ = _load(Path("configs/final_memory_confirmation.yaml"))

    assert config["protocol"]["candidate"] == "global_global__coverage_025"
    assert config["protocol"]["rl_used"] is False
    assert config["protocol"]["development_candidate_passed"] is False
    assert config["baseline_suite"]["nominal_coverage"] == 0.25
    assert policy["protocol"]["further_development_tuning_allowed"] is False
    baseline = LockedBaselineConfig(
        **{
            key: value
            for key, value in config["baseline_suite"].items()
            if key != "names"
        }
    )
    assert baseline.random_trials == 20


def test_confirmation_manifest_has_no_training_or_rl(monkeypatch, tmp_path):
    config_path = Path("configs/final_memory_confirmation.yaml")
    config, policy, testbed = _load(config_path)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"locked")
    monkeypatch.setattr(confirmation, "_lock_artifacts", lambda *_args: [artifact])
    monkeypatch.setattr(
        confirmation,
        "create_confirmation_lock",
        lambda path, *_args, **_kwargs: path.write_text("{}", encoding="utf-8"),
    )

    summary = _build(config_path, config, policy, testbed, tmp_path, "python")
    manifest = yaml.safe_load(
        (tmp_path / "experiment_manifest.yaml").read_text(encoding="utf-8")
    )
    commands = [" ".join(job["command"]) for job in manifest["jobs"]]

    assert summary["job_count"] == 44
    assert summary["gpu_job_count"] == 18
    assert summary["declared_gpu_hours"] == pytest.approx(5.4)
    assert not any(
        "train_" in command or "offline_policy" in command or "phase5-rl" in command
        for command in commands
    )
    assert len(manifest["jobs"][-1]["expected_outputs"]) == 1
    assert manifest["jobs"][-1]["expected_outputs"][0].replace("\\", "/").endswith(
        "/confirmation_closed.json"
    )


def test_equal_weight_buy_hold_pays_one_round_trip():
    dates = pd.date_range("2025-01-02", periods=3, freq="B", tz="UTC")
    signals = pd.DataFrame(
        {
            "timestamp": list(dates) * 2,
            "ticker": ["AAA"] * 3 + ["BBB"] * 3,
            "close": [100.0, 110.0, 121.0, 100.0, 100.0, 100.0],
        }
    )

    result = _equal_weight_buy_hold(signals, 100_000.0, 10.0)

    expected_relative = (1.21 + 1.0) / 2.0
    expected_return = (0.999 * expected_relative * 0.999) - 1.0
    assert abs(result["metrics"]["total_return"] - expected_return) < 1e-12
    assert result["metrics"]["trade_count"] == 0


def _evidence(seed: int, start: str, dates: int) -> pd.DataFrame:
    rows = []
    for day_index, timestamp in enumerate(
        pd.date_range(start, periods=dates, freq="B", tz="UTC")
    ):
        for ticker_index, ticker in enumerate(("AAA", "BBB", "CCC")):
            wave = np.sin((day_index + ticker_index) / 7.0)
            realised = 0.025 * wave
            expected = realised + 0.002 * np.cos(day_index + seed)
            price = 100.0 + day_index * (0.10 + ticker_index * 0.02)
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": price,
                    "close": price * 1.001,
                    "future_blended_alpha_63": realised,
                    "future_universe_alpha_63": realised,
                    "future_return_63": realised + 0.01,
                    "future_min_return_63": -0.03,
                    "future_max_return_63": 0.06,
                    "retrieval_expected_alpha": expected,
                    "retrieval_alpha_ci_low": expected - 0.02,
                    "retrieval_alpha_ci_high": expected + 0.02,
                    "retrieval_alpha_std": 0.03,
                    "retrieval_expected_upside": max(expected + 0.04, 0.001),
                    "retrieval_expected_downside": -0.03,
                    "retrieval_downside_cvar": -0.05,
                    "retrieval_outcome_std": 0.08,
                    "retrieval_confidence": 0.70,
                    "retrieval_agreement_score": 0.65,
                    "retrieval_effective_sample_size": 20.0,
                    "retrieval_entropy": 0.80,
                    "retrieval_distance_weighted_confidence": 0.60,
                    "retrieval_historical_diversity": 0.75,
                    "retrieval_median_distance": 4.0,
                    "retrieval_neighbor_count": 25,
                    "retrieval_ood_pass": True,
                    "opportunity_score": expected,
                    "pred_utility_q50": expected,
                }
            )
    return pd.DataFrame(rows)


def test_locked_market_evaluation_writes_predeclared_baselines(tmp_path):
    development = {seed: _evidence(seed, "2022-01-03", 90) for seed in (7, 17, 37)}
    query = {seed: _evidence(seed, "2025-01-02", 40) for seed in (7, 17, 37)}
    policy = PolicyConfig(
        top_k=2,
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=0.12,
        min_confidence=0.0,
        min_alpha_lcb=None,
        max_downside_cvar=0.12,
        min_neighbor_count=15,
        require_ood_pass=False,
        min_hold_days=2,
        max_hold_days=10,
        slippage_bps=10.0,
    )

    result = evaluate_locked_market(
        development_signals=development,
        development_neighbors={},
        query_signals=query,
        query_neighbors={},
        policy=policy,
        consensus_config=ConsensusSignalConfig(minimum_votes=2),
        reliability_config=ReliabilityConfig(
            fit_fraction=0.60,
            calibration_embargo_sessions=5,
            minimum_fit_rows=100,
            minimum_calibration_rows=50,
        ),
        useful_alpha_after_costs=0.002,
        baseline_config=LockedBaselineConfig(random_trials=3),
        output=tmp_path,
    )

    assert set(result["baselines"]) == {
        "locked_memory",
        "raw_memory",
        "momentum_21",
        "direct_adapter_q50",
        "equal_weight_buy_hold",
        "random_median",
    }
    assert result["random"]["trials"] == 3
    assert (tmp_path / "baselines" / "random_trials.csv").exists()
    assert (tmp_path / "reliability_model_audit.json").exists()


def test_confirmation_aggregate_reports_baselines_and_concentration(tmp_path):
    config, _, _ = _load(Path("configs/final_memory_confirmation.yaml"))
    strategies = ["locked_memory", *config["baseline_suite"]["names"]]
    for market_index, market in enumerate(config["experiment"]["markets"]):
        root = tmp_path / "evaluation" / market
        payload = {"reliability": {"brier_skill": 0.1}, "baselines": {}}
        for strategy_index, strategy in enumerate(strategies):
            total_return = 0.10 + market_index * 0.01 if strategy == "locked_memory" else 0.02
            payload["baselines"][strategy] = {
                "total_return": total_return,
                "sharpe": 1.0,
                "max_drawdown": -0.05,
            }
            destination = root / "baselines" / strategy
            destination.mkdir(parents=True, exist_ok=True)
            equity = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-02", periods=4, freq="B", tz="UTC"),
                    "equity": [100_000.0, 101_000.0, 102_000.0, 103_000.0],
                }
            )
            equity.to_csv(destination / "equity_curve.csv", index=False)
        root.mkdir(parents=True, exist_ok=True)
        (root / "metrics.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    result = _aggregate(config, tmp_path)

    assert result["equal_weight_wins"] == 6
    assert result["positive_brier_skill_markets"] == 6
    assert result["promotion_allowed"] is False
    assert result["profitable_strategy_claim_allowed"] is False
    assert (tmp_path / "strategy_summary.csv").exists()
    assert (tmp_path / "calibration_results.csv").exists()
