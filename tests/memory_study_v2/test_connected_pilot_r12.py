"""Acceptance & Regression Test for R12: Connected pre-2020 restricted pilot runner."""

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
