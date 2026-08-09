from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.run_temporal_decision_ablation import _paired_robustness


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_action_memory_study_is_regional_and_bounded():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "action_memory_robustness.yaml").read_text(encoding="utf-8")
    )

    assert set(config["variants"]) == {"baseline", "action_only"}
    assert config["study"]["expected_source_count"] == 18
    assert config["common"]["experiment"]["source_representations"] == ["regional"]
    assert len(config["common"]["experiment"]["active_markets"]) == 6
    assert config["comparison"]["gates"]["minimum_market_wins"] == 4
    assert config["common"]["temporal_evaluation"]["periods"]["q1_to_q1"] == {
        "start": pd.Timestamp("2025-01-01").date(),
        "end": pd.Timestamp("2026-03-31").date(),
    }
    assert config["common"]["temporal_evaluation"]["require_existing_adapter"] is True
    assert config["comparison"]["metric_namespace"] == "temporal_q1_to_q1"


def test_paired_robustness_requires_market_seed_floor_and_drawdown_support(tmp_path):
    markets = ["US", "India", "China", "Brazil", "France", "UK"]
    baseline_rows = []
    candidate_rows = []
    for market in markets:
        for seed in (7, 17, 37):
            common = {"fold": f"regional_{market}", "seed": seed}
            baseline_rows.append(
                {
                    **common,
                    "backtest_sharpe": 1.0,
                    "backtest_total_return": 0.10,
                    "backtest_max_drawdown": -0.12,
                }
            )
            candidate_rows.append(
                {
                    **common,
                    "backtest_sharpe": 1.2,
                    "backtest_total_return": 0.12,
                    "backtest_max_drawdown": -0.10,
                }
            )
    for name, rows in (("baseline", baseline_rows), ("action_only", candidate_rows)):
        destination = tmp_path / name
        destination.mkdir()
        pd.DataFrame(rows).to_csv(destination / "adapter_summary.csv", index=False)
    comparison = {
        "baseline": "baseline",
        "candidate": "action_only",
        "gates": {
            "minimum_market_wins": 4,
            "minimum_paired_run_win_fraction": 0.55,
            "minimum_mean_sharpe_delta": 0.0,
            "minimum_worst_sharpe_delta": 0.0,
            "minimum_drawdown_win_fraction": 0.50,
            "minimum_worst_drawdown_improvement": 0.0,
        },
    }

    paired, market, verdict = _paired_robustness(tmp_path, comparison)

    assert len(paired) == 18
    assert len(market) == 6
    assert verdict["status"] == "robustness_supported"
    assert verdict["market_wins"] == 6
    assert verdict["worst_sharpe_delta"] > 0
    assert verdict["worst_drawdown_improvement"] > 0


def test_paired_robustness_can_select_temporal_metrics(tmp_path):
    rows = [
        {
            "fold": "regional_US",
            "seed": 7,
            "temporal_q1_to_q1_sharpe": sharpe,
            "temporal_q1_to_q1_total_return": 0.10,
            "temporal_q1_to_q1_max_drawdown": -0.10,
        }
        for sharpe in (0.4,)
    ]
    for name, lift in (("baseline", 0.0), ("action_only", 0.2)):
        destination = tmp_path / name
        destination.mkdir()
        frame = pd.DataFrame(rows)
        frame["temporal_q1_to_q1_sharpe"] += lift
        frame.to_csv(destination / "adapter_summary.csv", index=False)

    paired, _, verdict = _paired_robustness(
        tmp_path,
        {
            "baseline": "baseline",
            "candidate": "action_only",
            "metric_namespace": "temporal_q1_to_q1",
            "gates": {},
        },
    )

    assert paired.loc[0, "sharpe_delta"] == pytest.approx(0.2)
    assert verdict["metric_namespace"] == "temporal_q1_to_q1"
