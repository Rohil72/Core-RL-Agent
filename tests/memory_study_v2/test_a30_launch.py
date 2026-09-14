"""Tests for A30 production deployment launcher (Finding 5 / C6).

Verifies:
1. Authorized launch calls driver_fn exactly once and records execution.
2. In-memory configuration authorization preserves repository config guard (production_authorized: false).
3. Missing required storage/manifests immediately fails closed without calling driver.
4. Driver failure records PRODUCTION_FAILED receipt and propagates error.
5. Unauthorized invocation without --authorize-production does not execute driver.
6. Check-only mode completes preflight checks without calling driver.
7. Corrupted repository config with production_authorized: true immediately raises error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
import pytest

from scripts.launch_a30_production import launch_a30_deployment


@pytest.fixture
def mock_repo_env(tmp_path: Path):
    """Set up valid mock environment with repo config, manifests, and fold files."""
    manifest_dir = tmp_path / "rebuild_plan"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Mock config with production_authorized: false
    config_path = manifest_dir / "config.proposed.json"
    config_path.write_text(
        json.dumps({
            "production_authorized": False,
            "experiment_name": "mock_production",
            "neural": {
                "seeds": [7],
                "architectures": ["MLP_MOCK"],
            },
        }),
        encoding="utf-8",
    )

    # Mock manifests
    (manifest_dir / "universe_request.csv").write_text("project_security_id\ntest\n", encoding="utf-8")
    (manifest_dir / "fold_dimensions_manifest.json").write_text(json.dumps({"folds": {"2020": {}}}), encoding="utf-8")

    # Mock sample ID manifests for all 6 folds
    sample_ids_dir = manifest_dir / "sample_ids"
    sample_ids_dir.mkdir(parents=True, exist_ok=True)
    for year in range(2020, 2026):
        (sample_ids_dir / f"fold_{year}_sample_ids.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config_path": config_path,
        "sample_ids_dir": sample_ids_dir,
        "output_dir": output_dir,
        "manifest_dir": manifest_dir,
    }


def test_authorized_launch_calls_driver_and_preserves_repo_guard(mock_repo_env):
    """Authorized launch invokes driver_fn, provides runtime authorization, and keeps file guard."""
    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        assert context["config"]["production_authorized"] is True
        return {
            "status": "PRODUCTION_SUCCESS",
            "jobs_started": ["fold_2020_job1"],
            "jobs_completed": ["fold_2020_job1"],
            "jobs_failed": [],
        }

    receipt = launch_a30_deployment(
        config_path=mock_repo_env["config_path"],
        output_dir=mock_repo_env["output_dir"],
        sample_ids_dir=mock_repo_env["sample_ids_dir"],
        authorize_production=True,
        check_only=False,
        selected_folds=[2020],
        driver_fn=mock_driver,
    )

    # Driver must be called exactly once
    assert len(calls) == 1
    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["execution"]["executed"] is True
    assert receipt["execution"]["driver_result"]["status"] == "PRODUCTION_SUCCESS"

    # Repository config file MUST remain production_authorized: false
    with open(mock_repo_env["config_path"], "r", encoding="utf-8") as f:
        disk_cfg = json.load(f)
    assert disk_cfg["production_authorized"] is False


def test_missing_storage_manifest_fails_closed_without_calling_driver(mock_repo_env):
    """Missing required manifest must raise RuntimeError and never execute the driver."""
    # Delete one of the required manifests
    missing_manifest = mock_repo_env["sample_ids_dir"] / "fold_2024_sample_ids.json"
    missing_manifest.unlink()

    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"status": "PRODUCTION_SUCCESS"}

    with pytest.raises(RuntimeError, match="Storage preflight check failed"):
        launch_a30_deployment(
            config_path=mock_repo_env["config_path"],
            output_dir=mock_repo_env["output_dir"],
            sample_ids_dir=mock_repo_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            driver_fn=mock_driver,
        )

    # Driver must never have been called
    assert len(calls) == 0

    # Receipt written on disk must record PREFLIGHT_STORAGE_ERROR
    receipt_file = mock_repo_env["output_dir"] / "a30_launch_receipt.json"
    assert receipt_file.exists()
    with open(receipt_file, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)
    assert receipt_data["status"] == "PREFLIGHT_STORAGE_ERROR"
    assert any("sample ID manifests" in err for err in receipt_data["storage_errors"])


def test_driver_failure_propagates_and_records_production_failed(mock_repo_env):
    """If the production driver raises an exception, receipt must record PRODUCTION_FAILED."""
    def failing_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("CUDA Out of Memory during training")

    with pytest.raises(RuntimeError, match="CUDA Out of Memory"):
        launch_a30_deployment(
            config_path=mock_repo_env["config_path"],
            output_dir=mock_repo_env["output_dir"],
            sample_ids_dir=mock_repo_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
            driver_fn=failing_driver,
        )

    receipt_file = mock_repo_env["output_dir"] / "a30_launch_receipt.json"
    assert receipt_file.exists()
    with open(receipt_file, "r", encoding="utf-8") as f:
        receipt_data = json.load(f)
    assert receipt_data["status"] == "PRODUCTION_FAILED"
    assert "CUDA Out of Memory" in receipt_data["execution"]["error"]


def test_unauthorized_launch_does_not_execute_driver(mock_repo_env):
    """Calling without --authorize-production must lock execution and never call driver."""
    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"status": "PRODUCTION_SUCCESS"}

    receipt = launch_a30_deployment(
        config_path=mock_repo_env["config_path"],
        output_dir=mock_repo_env["output_dir"],
        sample_ids_dir=mock_repo_env["sample_ids_dir"],
        authorize_production=False,
        check_only=False,
        driver_fn=mock_driver,
    )

    assert len(calls) == 0
    assert receipt["status"] == "LOCKED_AWAITING_AUTHORIZATION"
    assert receipt["execution"]["executed"] is False
    assert receipt["authorization"]["can_proceed_to_production"] is False


def test_check_only_mode(mock_repo_env):
    """Check-only mode completes preflight checks without calling driver."""
    calls = []

    def mock_driver(context: Dict[str, Any]) -> Dict[str, Any]:
        calls.append(context)
        return {"status": "PRODUCTION_SUCCESS"}

    receipt = launch_a30_deployment(
        config_path=mock_repo_env["config_path"],
        output_dir=mock_repo_env["output_dir"],
        sample_ids_dir=mock_repo_env["sample_ids_dir"],
        authorize_production=False,
        check_only=True,
        driver_fn=mock_driver,
    )

    assert len(calls) == 0
    assert receipt["status"] == "PREFLIGHT_PASS"
    assert receipt["execution"]["mode"] == "CHECK_ONLY"
    assert receipt["execution"]["executed"] is False


def test_corrupted_repo_config_with_authorized_true_fails(mock_repo_env):
    """If repository config has production_authorized: true, launcher must fail immediately."""
    # Corrupt repository configuration
    mock_repo_env["config_path"].write_text(
        json.dumps({"production_authorized": True}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Invariant violated"):
        launch_a30_deployment(
            config_path=mock_repo_env["config_path"],
            output_dir=mock_repo_env["output_dir"],
            sample_ids_dir=mock_repo_env["sample_ids_dir"],
            authorize_production=True,
            check_only=False,
        )
