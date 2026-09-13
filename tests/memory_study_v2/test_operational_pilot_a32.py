"""Acceptance Test A32: Operational pilot report and runtime projection (R01, R11, B6)."""

import json
from pathlib import Path
import pytest


def test_pilot_report_exists_and_valid():
    report_path = Path("rebuild_plan/pilot_report.json")
    assert report_path.exists(), "pilot_report.json must exist in rebuild_plan/"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep.get("status") in ["LOCAL_PILOT_COMPLETE", "LOCAL_ONLY", "PENDING_VM"]
    assert "timestamp_utc" in rep
    assert rep.get("workload_profile_mode") == "REALISTIC_WORKLOAD_PROFILING"

    # Environment check
    env = rep.get("environment", {})
    assert "platform" in env
    assert env.get("cpu_cores", 0) > 0
    assert env.get("host_ram_gb", 0) > 0
    assert "python_version" in env
    assert "torch_version" in env

    # Benchmarks check (R01, B6 realistic benchmarks)
    bm = rep.get("benchmarks", {})
    assert bm.get("mlp_effective_batch_512_macro_step_ms", 0) > 0
    assert bm.get("transformer_effective_batch_512_macro_step_ms", 0) > 0
    assert bm.get("validation_pass_seconds", 0) > 0
    assert bm.get("retrieval_20k_bank_qps", 0) > 0
    assert bm.get("populated_engine_sim_year_seconds", -1) >= 0
    assert bm.get("bootstrap_contrasts_1000_draws_seconds", -1) >= 0
    assert bm.get("peak_rss_mb", 0) > 0

    # Fold dimensions check (B6: measured fold counts)
    folds = rep.get("fold_dimensions", [])
    assert len(folds) == 6, f"Expected 6 walk-forward folds, got {len(folds)}"
    for f_info in folds:
        assert f_info["evaluation_year"] in range(2020, 2026)
        assert f_info["train_samples"] > 100000
        assert f_info["evaluation_queries"] > 20000
        assert f_info["bank_samples"] > 100000
        assert f_info["macro_steps_per_epoch"] > 0

    # Cap-based Projection check
    proj = rep.get("projection", {})
    assert proj.get("epochs_cap") == 50
    assert proj.get("fits_count") == 36
    assert 180000 <= proj.get("average_training_samples_per_fold", 0) <= 200000
    assert proj.get("effective_batch_size") == 512
    assert proj.get("micro_batch_size") == 64
    assert 350 <= proj.get("macro_steps_per_epoch", 0) <= 400
    assert proj.get("total_macro_steps_all_fits", 0) > 500000
    assert proj.get("total_evaluation_queries", 0) > 100000
    assert proj.get("compute_safety_multiplier") == 1.5
    assert proj.get("export_and_verify_reserve_hours", 0) >= 2.0
    assert proj.get("contingency_reserve_hours", 0) >= 2.0

    # Condition consistency check (R11, B6)
    # The condition flag must truthfully reflect whether total_budget <= vm_allocation
    total_budget = proj.get("total_budget_needed_hours", 999.0)
    vm_allocation = proj.get("vm_allocation_hours", 24.0)
    expected_condition = (total_budget <= vm_allocation)
    assert proj.get("acceptance_condition_met") == expected_condition
    assert proj.get("hardware_preflight_status") == "PENDING_A30_VM_EXECUTION"

    # Unmeasured phases verification (C6)
    unmeas = rep.get("unmeasured_phases", {})
    assert unmeas.get("hardware_preflight_status") == "PENDING_A30_VM_EXECUTION"
    assert "a30_gpu_hardware_acceleration" in unmeas
    assert "cloud_persistent_storage_io" in unmeas
    assert "multi_worker_parallel_retrieval" in unmeas


def test_fold_dimensions_manifest_traceability_c6():
    """Verify C6 requirement: traceable hashed admissible-row manifest across 103 securities."""
    import hashlib

    manifest_path = Path("rebuild_plan/fold_dimensions_manifest.json")
    assert manifest_path.exists(), "fold_dimensions_manifest.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("manifest_version") == "2.0.0"
    assert manifest.get("security_count") == 103

    files = manifest.get("file_manifest", [])
    assert len(files) == 103, f"Expected 103 files in manifest, got {len(files)}"

    for entry in files:
        assert "filename" in entry
        assert "sha256" in entry and len(entry["sha256"]) == 64
        assert entry.get("file_size_bytes", 0) > 0
        assert entry.get("total_rows", 0) > 0
        assert "annual_admissible_rows" in entry

    fold_dims = manifest.get("fold_dimensions", [])
    assert len(fold_dims) == 6

    # Verify report references the manifest with matching hash
    report_path = Path("rebuild_plan/pilot_report.json")
    assert report_path.exists()
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    if "fold_manifest" in report:
        fm = report["fold_manifest"]
        assert fm["security_count"] == 103
        actual_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert fm["manifest_sha256"] == actual_sha

