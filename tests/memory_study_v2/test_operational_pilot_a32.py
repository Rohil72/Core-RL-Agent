"""Acceptance Test A32: Operational pilot report and runtime projection.

Acceptance criteria:
- Measured cap-based runtime projection fits actual remaining VM time plus reserves (pilot_report.json).
- 50-epoch cap, 36 fits, 1.5x compute safety, >=2h export reserve, >=2h contingency reserve.
- Total budget needed <= 24.0 hours VM allocation.
"""

import json
from pathlib import Path
import pytest


def test_pilot_report_exists_and_valid():
    report_path = Path("rebuild_plan/pilot_report.json")
    assert report_path.exists(), "pilot_report.json must exist in rebuild_plan/"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Status check
    assert rep.get("status") in ["LOCAL_PILOT_COMPLETE", "LOCAL_ONLY", "PENDING_VM"]
    assert "timestamp_utc" in rep

    # Environment check
    env = rep.get("environment", {})
    assert "platform" in env
    assert env.get("cpu_cores", 0) > 0
    assert env.get("host_ram_gb", 0) > 0
    assert "python_version" in env
    assert "torch_version" in env

    # Benchmarks check
    bm = rep.get("benchmarks", {})
    assert bm.get("mlp_step_time_ms", 0) > 0
    assert bm.get("transformer_step_time_ms", 0) > 0
    assert bm.get("retrieval_qps", 0) > 0
    assert bm.get("engine_sim_year_seconds", -1) >= 0
    assert bm.get("bootstrap_1000_draws_seconds", -1) >= 0

    # Cap-based Projection check
    proj = rep.get("projection", {})
    assert proj.get("epochs_cap") == 50
    assert proj.get("fits_count") == 36
    assert proj.get("compute_safety_multiplier") == 1.5
    assert proj.get("export_and_verify_reserve_hours", 0) >= 2.0
    assert proj.get("contingency_reserve_hours", 0) >= 2.0

    total_budget = proj.get("total_budget_needed_hours", 999.0)
    vm_allocation = proj.get("vm_allocation_hours", 24.0)
    assert total_budget <= vm_allocation, (
        f"Projected total budget ({total_budget}h) exceeds VM allocation ({vm_allocation}h)"
    )
    assert proj.get("acceptance_condition_met") is True
