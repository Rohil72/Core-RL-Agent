from pathlib import Path

import pandas as pd
import yaml

import scripts.run_phase6_frozen_memory_sweep as frozen_sweep
from scripts.run_phase6_frozen_memory_sweep import (
    _build_manifest,
    _hybrid_memory_config,
    _load,
    _prepare_reliability_source,
    _reliability_config,
    _required_source_paths,
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


def test_final_policy_is_one_locked_global_candidate_without_rl():
    config = yaml.safe_load(Path("configs/final_memory_policy.yaml").read_text(encoding="utf-8"))

    assert config["protocol"]["candidate"] == "global_global__coverage_025"
    assert config["protocol"]["rl_used"] is False
    assert [item["id"] for item in config["topologies"]] == ["global_global"]
    assert [item["id"] for item in config["coverage_candidates"]] == ["coverage_025"]
    assert config["selection"]["require_pbo"] is False

    required = [path.as_posix() for path in _required_source_paths(config)]
    assert required
    assert all("/memory/global_" in path for path in required)
    assert not any("/adapters/" in path or "/regional_" in path for path in required)


def test_final_policy_manifest_contains_only_evaluation(monkeypatch, tmp_path):
    config_path = Path("configs/final_memory_policy.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    testbed = yaml.safe_load(Path("configs/final_research_testbed.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        frozen_sweep,
        "_preflight",
        lambda _config, _testbed: {"required_artifacts": 72, "markets": 6},
    )

    summary = _build_manifest(config_path, config, testbed, tmp_path, "python")
    manifest = yaml.safe_load((tmp_path / "experiment_manifest.yaml").read_text(encoding="utf-8"))

    assert summary["job_count"] == 1
    assert summary["jobs_by_stage"] == {"prepare": 0, "retrieve": 0, "evaluate": 1}
    assert manifest["jobs"][0]["id"] == "evaluate_frozen_memory_sweep"
    assert manifest["jobs"][0]["depends_on"] == []
    assert not any("offline_policy" in token or "phase5-rl" in token for token in manifest["jobs"][0]["command"])


def test_final_policy_lock_rejects_candidate_expansion(tmp_path):
    config = yaml.safe_load(Path("configs/final_memory_policy.yaml").read_text(encoding="utf-8"))
    config["coverage_candidates"].append({"id": "coverage_050", "nominal_coverage": 0.50})
    path = tmp_path / "expanded.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    try:
        _load(path)
    except ValueError as exc:
        assert "PBO may be disabled only for one preselected" in str(exc)
    else:
        raise AssertionError("The frozen final policy must reject additional candidates.")
