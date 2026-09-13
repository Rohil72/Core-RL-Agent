"""Acceptance & Regression Test for R12: Connected pre-2020 restricted pilot runner."""

import json
import pytest

from memory_study_v2.connected_pilot import run_connected_restricted_pilot


def test_connected_restricted_pilot_runs_end_to_end(tmp_path):
    report = run_connected_restricted_pilot(tmp_path / "pilot_out")
    assert report["status"] == "CONNECTED_PILOT_SUCCESS"
    expected_phases = [
        "data_canonicalization",
        "feature_engineering",
        "backbone_training",
        "memory_bank_construction",
        "prediction_sealing",
        "portfolio_execution",
        "inference_contrasts",
        "release_verification",
    ]
    assert report["phases_executed"] == expected_phases
    assert report["release_verification"]["status"] == "RELEASE_VERIFIED"
    assert report["replay_verification"]["status"] == "REPLAY_VERIFIED"

    # Verify input manifest (C1)
    input_manifest_path = tmp_path / "pilot_out" / "input_manifest.json"
    assert input_manifest_path.exists()
    with open(input_manifest_path, "r", encoding="utf-8") as f:
        input_manifest = json.load(f)
    assert input_manifest["imputation_policy"]["imputed_targets_count"] == 0
    assert input_manifest["imputation_policy"]["fallback_targets_count"] == 0
    assert input_manifest["imputation_policy"]["substitute_policy_returns_count"] == 0
    assert len(input_manifest["input_files"]) == 2
    for file_info in input_manifest["input_files"]:
        assert len(file_info["sha256"]) == 64

    # Verify output manifest (C1)
    output_manifest_path = tmp_path / "pilot_out" / "output_manifest.json"
    assert output_manifest_path.exists()
    with open(output_manifest_path, "r", encoding="utf-8") as f:
        output_manifest = json.load(f)
    assert output_manifest["status"] == "CONNECTED_PILOT_SUCCESS"
    assert output_manifest["policies_evaluated"] == ["MEM_SIM", "MLP_BASE", "TRANS_BASE"]
    assert output_manifest["policies_unrun_status"] == "NOT_RUN"
    assert output_manifest["verification"]["replay_verification_status"] == "REPLAY_VERIFIED"


def test_connected_pilot_contrasts_distinguish_evaluated_vs_unrun(tmp_path):
    """Verify primary contrasts distinguish evaluated arms from un-run comparison arms without fake noise (C1)."""
    run_connected_restricted_pilot(tmp_path / "pilot_out")
    contrasts_path = tmp_path / "pilot_out" / "release_analysis" / "primary_contrasts.json"
    assert contrasts_path.exists()
    with open(contrasts_path, "r", encoding="utf-8") as f:
        contrasts = json.load(f)

    # 8 primary contrasts
    assert len(contrasts) == 8
    contrasts_by_id = {c["contrast_id"]: c for c in contrasts}

    # P5 (MEM_SIM vs MLP_BASE) and P6 (MEM_SIM vs TRANS_BASE) were both evaluated
    assert contrasts_by_id["P5"]["status"] == "COMPLETED"
    assert contrasts_by_id["P6"]["status"] == "COMPLETED"

    # Arms not run in this restricted pilot (e.g. MEM_RANDOM, KNN_PLAIN) are marked NOT_RUN
    assert contrasts_by_id["P1"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P2"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P3"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P4"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P7"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P8"]["status"] == "NOT_RUN"

