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
        assert f_info["train_samples"] > 50000
        assert f_info["evaluation_queries"] > 20000
        assert f_info["bank_samples"] > 50000
        assert f_info["macro_steps_per_epoch"] > 0

    # Cap-based Projection check
    proj = rep.get("projection", {})
    assert proj.get("epochs_cap") == 50
    assert proj.get("fits_count") == 36
    assert 120000 <= proj.get("average_training_samples_per_fold", 0) <= 150000
    assert proj.get("effective_batch_size") == 512
    assert proj.get("micro_batch_size") == 64
    assert 250 <= proj.get("macro_steps_per_epoch", 0) <= 400
    assert proj.get("total_macro_steps_all_fits", 0) > 400000
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

    assert manifest.get("manifest_version") in ("2.0.0", "2.1.0")
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

    # Verify 6 per-population counts per fold (Finding 3 / C6)
    for fd in fold_dims:
        assert "train_samples" in fd and fd["train_samples"] > 0
        assert "validation_samples" in fd and fd["validation_samples"] > 0
        assert "development_samples" in fd and fd["development_samples"] > 0
        assert "evaluation_queries" in fd and fd["evaluation_queries"] > 0
        assert "scored_eval_query_samples" in fd and fd["scored_eval_query_samples"] > 0
        assert "bank_samples" in fd and fd["bank_samples"] > 0
        # Bank samples strictly <= train samples due to 126 vs 63-session maturity
        assert fd["bank_samples"] <= fd["train_samples"], (
            f"Bank samples ({fd['bank_samples']}) must be <= train samples ({fd['train_samples']})"
        )

    # In fold 2025, future target availability does not gate eval eligibility
    fd_2025 = next(fd for fd in fold_dims if fd["evaluation_year"] == 2025)
    assert fd_2025["scored_eval_query_samples"] < fd_2025["evaluation_queries"], (
        "In 2025, scored_eval_query_samples must be strictly less than evaluation_queries"
    )

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


def test_manifest_id_consistency_c6():
    """Finding 3: Persisted sample IDs match fold partition counts and have stable structure."""
    sample_ids_dir = Path("rebuild_plan/sample_ids")
    if not sample_ids_dir.exists():
        pytest.skip("sample_ids directory not populated in this environment.")

    manifest_path = Path("rebuild_plan/fold_dimensions_manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for fd in manifest.get("fold_dimensions", []):
        y = fd["evaluation_year"]
        id_file = sample_ids_dir / f"fold_{y}_sample_ids.json"
        assert id_file.exists(), f"Sample IDs file missing for year {y}"
        with open(id_file, "r", encoding="utf-8") as f:
            ids = json.load(f)
        assert len(ids["train_query_ids"]) == fd["train_samples"]
        assert len(ids["val_query_ids"]) == fd["validation_samples"]
        assert len(ids["dev_query_ids"]) == fd["development_samples"]
        assert len(ids["eval_query_ids"]) == fd["evaluation_queries"]
        assert len(ids["bank_query_ids"]) == fd["bank_samples"]


def test_manifest_rejection_on_invalid(tmp_path):
    """Finding 3: load_measured_fold_dimensions rejects missing/corrupted manifest in acceptance mode."""
    from memory_study_v2.pilot import load_measured_fold_dimensions

    # 1. Missing manifest in acceptance mode raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_measured_fold_dimensions(tmp_path / "nonexistent.json", execution_mode="acceptance")

    # 2. Corrupted manifest without fold_dimensions raises ValueError
    bad_manifest = tmp_path / "bad_manifest.json"
    with open(bad_manifest, "w", encoding="utf-8") as f:
        json.dump({"manifest_version": "2.1.0", "empty": True}, f)

    with pytest.raises(ValueError, match="does not contain valid 'fold_dimensions'"):
        load_measured_fold_dimensions(bad_manifest, execution_mode="acceptance")


def test_f3_early_origin_insufficient_history_excluded_from_manifest_and_loader(tmp_path):
    """Finding 3 acceptance test:
    Deliberately include an early origin with a completed forward return but insufficient input history.
    It must be excluded from both the training manifest and the loader query IDs.
    Compare complete ID sets, not merely their lengths.
    """
    import numpy as np
    import pandas as pd
    from memory_study_v2.folds import FoldBoundaries
    from memory_study_v2.sample_index import (
        build_security_sample_index,
        partition_sample_index,
        save_fold_sample_ids,
        load_fold_sample_ids,
    )
    from memory_study_v2.venue_calendar import VenueCalendar

    vc = VenueCalendar()

    dates = vc.sessions_in_range("2013-01-02", "2017-12-31")[:650]
    df = pd.DataFrame({
        "session": dates,
        "close": np.linspace(100.0, 200.0, len(dates)),
        "open": np.linspace(100.0, 200.0, len(dates)),
        "high": np.linspace(102.0, 202.0, len(dates)),
        "low": np.linspace(98.0, 198.0, len(dates)),
        "volume": np.ones(len(dates)) * 1000.0,
    })

    records = build_security_sample_index("TEST_SEC", df, venue_calendar=vc, evaluation_year=2020)

    # Check origin 50 (early origin with completed forward return)
    early_rec = records[50]
    assert early_rec.target_63_valid is True, "Forward 63-day return must be completed"
    assert early_rec.input_window_valid is False, "Early origin (< 503 preceding bars) must have input_window_valid == False"
    assert early_rec.exclusion_reason == "INSUFFICIENT_INPUT_HISTORY"

    # Check origin 550 (mature origin with sufficient input history)
    mature_rec = records[550]
    assert mature_rec.target_63_valid is True
    assert mature_rec.input_window_valid is True
    assert mature_rec.exclusion_reason is None

    # Partition fold 2020
    bound = FoldBoundaries(
        evaluation_year=2020,
        train_start="2013-01-01",
        train_end="2017-12-31",
        val_start="2018-01-01",
        val_end="2018-12-31",
        dev_start="2019-01-01",
        dev_end="2019-12-31",
        eval_start="2020-01-01",
        eval_end="2020-12-31",
        bank_cutoff="2017-12-31",
    )
    part = partition_sample_index(records, bound)

    # 1. Early origin must NOT be in train query IDs; mature origin MUST be in train query IDs
    assert early_rec.query_id not in part.train_query_ids
    assert mature_rec.query_id in part.train_query_ids

    # 2. Persist sample IDs and reload via loader function
    save_fold_sample_ids(tmp_path, 2020, part, records)
    loaded_ids = load_fold_sample_ids(2020, sample_ids_dir=tmp_path)

    # 3. Compare complete ID sets, not merely lengths!
    assert set(loaded_ids["train_query_ids"]) == set(part.train_query_ids)
    assert early_rec.query_id not in set(loaded_ids["train_query_ids"])
    assert mature_rec.query_id in set(loaded_ids["train_query_ids"])

