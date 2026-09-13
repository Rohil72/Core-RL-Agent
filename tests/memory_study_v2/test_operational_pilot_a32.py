"""Acceptance Test A32: Operational pilot report and runtime projection (R01, R11)."""

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

    # Benchmarks check (R01 realistic benchmarks)
    bm = rep.get("benchmarks", {})
    assert bm.get("mlp_effective_batch_512_macro_step_ms", 0) > 0
    assert bm.get("transformer_effective_batch_512_macro_step_ms", 0) > 0
    assert bm.get("validation_pass_seconds", 0) > 0
    assert bm.get("retrieval_20k_bank_qps", 0) > 0
    assert bm.get("populated_engine_sim_year_seconds", -1) >= 0
    assert bm.get("bootstrap_contrasts_1000_draws_seconds", -1) >= 0
    assert bm.get("peak_rss_mb", 0) > 0

    # Cap-based Projection check
    proj = rep.get("projection", {})
    assert proj.get("epochs_cap") == 50
    assert proj.get("fits_count") == 36
    assert proj.get("average_training_samples_per_fold") == 195000
    assert proj.get("effective_batch_size") == 512
    assert proj.get("micro_batch_size") == 64
    assert proj.get("macro_steps_per_epoch") == 380
    assert proj.get("total_macro_steps_per_fit") == 19000
    assert proj.get("compute_safety_multiplier") == 1.5
    assert proj.get("export_and_verify_reserve_hours", 0) >= 2.0
    assert proj.get("contingency_reserve_hours", 0) >= 2.0

    # Condition consistency check (R11)
    # The condition flag must truthfully reflect whether total_budget <= vm_allocation
    total_budget = proj.get("total_budget_needed_hours", 999.0)
    vm_allocation = proj.get("vm_allocation_hours", 24.0)
    expected_condition = (total_budget <= vm_allocation)
    assert proj.get("acceptance_condition_met") == expected_condition
