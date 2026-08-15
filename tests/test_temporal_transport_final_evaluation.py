import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import scripts.run_temporal_transport_final_evaluation as final_module
from src.eval.closing_evaluation import (
    aggregate_equal_weight_market_returns,
    stationary_bootstrap_comparison,
    validate_equity_curve,
)
from src.eval.policy_baselines import equal_weight_buy_hold


def _curve(dates, values):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True),
            "equity": values,
            "exposure": 1.0,
        }
    )
    frame["return"] = frame["equity"].pct_change().fillna(0.0)
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame


def test_equal_weight_market_aggregate_treats_closed_market_as_zero_return():
    us = _curve(["2024-01-01", "2024-01-02", "2024-01-03"], [100.0, 110.0, 121.0])
    uk = _curve(["2024-01-01", "2024-01-03"], [100.0, 100.0])

    pooled = aggregate_equal_weight_market_returns({"US": us, "UK": uk}, 100.0)

    assert np.isclose(pooled.iloc[1]["return"], 0.05)
    assert np.isclose(pooled.iloc[2]["return"], 0.05)
    assert validate_equity_curve(pooled, 100.0)["passed"] is True


def test_equal_weight_buy_hold_carries_missing_security_mark():
    signals = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-03"],
                utc=True,
            ),
            "ticker": ["AAA", "AAA", "AAA", "BBB", "BBB"],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )

    result = equal_weight_buy_hold(signals, 1000.0, 0.0)

    assert np.allclose(result["equity"]["equity"], 1000.0)


def test_stationary_bootstrap_is_paired_and_deterministic():
    baseline = np.array([0.0, -0.01, 0.01, 0.0, 0.005, -0.002])
    candidate = baseline + 0.001

    first = stationary_bootstrap_comparison(
        baseline, candidate, samples=50, mean_block_length=3, seed=9
    )
    second = stationary_bootstrap_comparison(
        baseline, candidate, samples=50, mean_block_length=3, seed=9
    )

    assert first == second
    assert first["probability_annualized_return_delta_positive"] == 1.0


def _config(source: str = "reports/source"):
    return {
        "study": {
            "source_run": source,
            "output_root": "reports/final",
            "diagnostic_only": True,
            "promotion_allowed": False,
        },
        "design": {
            "variants": ["baseline", "candidate"],
            "baseline": "baseline",
            "candidate": "candidate",
            "seeds": [7],
            "periods": ["test"],
            "markets": ["US", "UK"],
            "development_period": "test",
            "observed_period": "test",
            "observed_period_is_contaminated": True,
        },
        "evaluation": {
            "primary_cost_bps": 0,
            "cost_sensitivity_bps": [0],
            "momentum_window": 2,
            "random_trials": 2,
            "random_seed": 11,
            "bootstrap_samples": 20,
            "bootstrap_mean_block_sessions": 2,
            "bootstrap_seed": 11,
            "minimum_daily_return": -0.95,
            "write_trade_ledgers": True,
        },
        "policy": {
            "top_k": 1,
            "min_score": 0.0,
            "min_expected_upside": 0.0,
            "max_expected_downside": 0.5,
            "min_confidence": 0.0,
            "min_alpha_lcb": None,
            "max_downside_cvar": None,
            "min_neighbor_count": None,
            "require_ood_pass": False,
            "min_hold_days": 1,
            "max_hold_days": 3,
            "holding_mode": "current",
            "exit_score_fraction": 0.5,
            "exit_min_score": 0.0,
            "stop_loss": 0.5,
            "slippage_bps": 0,
            "initial_capital": 1000,
            "risk_guard_enabled": False,
        },
        "development_gates": {
            "minimum_seed_wins": 0,
            "minimum_mean_pooled_sharpe_delta": -100.0,
            "minimum_market_seed_win_fraction": 0.0,
            "maximum_candidate_drawdown": 1.0,
        },
    }


def _signals():
    rows = []
    for market, ticker in (("US", "AAA"), ("UK", "BBB")):
        for day in range(8):
            price = 100.0 + day
            rows.append(
                {
                    "market": market,
                    "ticker": ticker,
                    "timestamp": pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day),
                    "open": price,
                    "close": price,
                    "opportunity_score": 0.1,
                    "retrieval_expected_upside": 0.1,
                    "retrieval_expected_downside": -0.02,
                    "retrieval_confidence": 1.0,
                    "pred_future_max_return_63": 0.1,
                    "pred_future_min_return_63": -0.02,
                }
            )
    return pd.DataFrame(rows)


def test_build_is_cpu_only_and_evaluation_writes_reviewer_artifacts(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(final_module, "PROJECT_ROOT", tmp_path)
    config = _config()
    config_path = tmp_path / "study.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    signals = _signals()
    for variant in config["design"]["variants"]:
        root = tmp_path / "reports/source/memory" / f"{variant}_seed_7/test/eval"
        root.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(root / "signals.parquet", index=False)
        (root / "metrics.json").write_text(json.dumps({"sharpe": 1.0}), encoding="utf-8")

    built = final_module.build(config_path, "run", "python")
    manifest = yaml.safe_load(
        (tmp_path / built["manifest"]).read_text(encoding="utf-8")
    )
    result = final_module.evaluate(config_path, "run", "baseline", 7, "test")
    final_module.evaluate(config_path, "run", "candidate", 7, "test")
    verdict = final_module.compare(config_path, "run")

    assert built["gpu_jobs"] == 0
    assert len(manifest["jobs"]) == 4
    assert all(job["uses_gpu"] is False for job in manifest["jobs"])
    assert result["accounting_audit_passed"] is True
    assert verdict["status"] == "closing_evaluation_completed"
    assert verdict["promotion_allowed"] is False
    destination = tmp_path / "reports/final/run/evaluations/baseline_seed_7/test"
    assert (destination / "market_metrics.csv").is_file()
    assert (destination / "pooled_metrics.csv").is_file()
    assert (destination / "random_trials.csv").is_file()
    assert json.loads((destination / "accounting_audit.json").read_text())["passed"] is True
    assert (tmp_path / "reports/final/run/comparison/reviewer_report.md").is_file()
