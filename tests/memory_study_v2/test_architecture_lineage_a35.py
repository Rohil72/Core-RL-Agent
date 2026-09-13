"""Acceptance Test A35: Architecture lineage and component mapping.

Acceptance criteria:
- All contribution components mapped to modules and ablations or explicitly excluded with approval (architecture_lineage.csv).
"""

import csv
import importlib
from pathlib import Path
import pytest


def test_architecture_lineage_completeness():
    lineage_path = Path("rebuild_plan/architecture_lineage.csv")
    assert lineage_path.exists(), "architecture_lineage.csv must exist in rebuild_plan/"

    with open(lineage_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    required_fields = [
        "claimed_component_name",
        "implemented_module",
        "mathematical_operation",
        "input_availability",
        "ablation",
        "in_study_status",
        "notes",
    ]
    assert reader, "Lineage file is empty"
    for field in required_fields:
        assert field in reader[0], f"Missing required column: {field}"

    in_study_count = 0
    outside_study_count = 0

    for row in reader:
        status = row["in_study_status"]
        assert status in ["IN_STUDY", "OUTSIDE_STUDY"], f"Invalid status: {status}"
        assert row["claimed_component_name"], "Missing component name"
        assert row["notes"], "Missing notes/justification"

        if status == "IN_STUDY":
            in_study_count += 1
            assert row["implemented_module"] != "None", f"Missing module for in-study component: {row['claimed_component_name']}"
            assert row["mathematical_operation"] != "None"
            assert row["ablation"] != "None"

            # Check importability if it specifies a dotted python symbol
            module_ref = row["implemented_module"]
            if "." in module_ref and "memory_study_v2" in module_ref:
                parts = module_ref.split(".")
                mod_name = ".".join(parts[:-1])
                sym_name = parts[-1]
                mod = importlib.import_module(mod_name)
                assert hasattr(mod, sym_name), f"Module {mod_name} missing symbol {sym_name}"

        else:
            outside_study_count += 1
            assert row["implemented_module"] == "None"
            assert row["mathematical_operation"] == "None"

    assert in_study_count == 14, f"Expected 14 in-study components, found {in_study_count}"
    assert outside_study_count == 3, f"Expected 3 outside-study components, found {outside_study_count}"
    assert len(reader) == 17, f"Expected 17 total components, found {len(reader)}"
