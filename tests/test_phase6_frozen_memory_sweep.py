from pathlib import Path

import pandas as pd
import yaml

from scripts.run_phase6_frozen_memory_sweep import (
    _hybrid_memory_config,
    _prepare_reliability_source,
    _reliability_config,
)


def _configs() -> tuple[dict, dict]:
    config = yaml.safe_load(Path("configs/phase6_frozen_memory_sweep.yaml").read_text(encoding="utf-8"))
    testbed = yaml.safe_load(Path("configs/final_research_testbed.yaml").read_text(encoding="utf-8"))
    return config, testbed


def test_frozen_sweep_is_small_and_has_no_rl_or_encoder_stage():
    config, _ = _configs()

    assert [item["id"] for item in config["topologies"]] == [
        "regional_regional",
        "global_global",
        "global_regional",
    ]
    assert [item["nominal_coverage"] for item in config["coverage_candidates"]] == [
        1.0,
        0.75,
        0.5,
        0.25,
    ]
    assert config["consensus"]["minimum_votes"] == 2
    assert "policy_algorithms" not in config


def test_reliability_target_uses_realized_net_alpha_without_mutating_input():
    source = pd.DataFrame({"decision_net_alpha": [0.03, -0.01]})

    prepared = _prepare_reliability_source(source)

    assert "future_blended_alpha_63" not in source
    assert prepared["future_blended_alpha_63"].tolist() == [0.03, -0.01]
    assert prepared["future_universe_alpha_63"].equals(prepared["future_blended_alpha_63"])


def test_market_cost_sets_reliability_target_and_backtest_slippage():
    config, testbed = _configs()
    output = Path("reports/test_phase6_frozen_memory")

    reliability, useful_alpha = _reliability_config(config, testbed, "UK")
    memory = _hybrid_memory_config(
        config,
        testbed,
        output,
        "UK",
        7,
        "policy_selection",
    )

    assert useful_alpha == 0.005
    assert reliability.useful_alpha_after_costs == 0.005
    assert memory["policy"]["slippage_bps"] == 25.0
    assert memory["data"]["train_latents"].replace("\\", "/").endswith("global_seed_7/UK.parquet")
    assert "global_UK_seed_7/policy_selection.parquet" in memory["data"]["test_latents"]
    assert memory["memory"]["require_outcome_availability"] is False
