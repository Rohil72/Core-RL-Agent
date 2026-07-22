from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.run_phase5_opportunity_allocator import _native
from src.policy.opportunity_allocator import OpportunityAllocatorConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase5_opportunity_allocator_contract_is_predeclared():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase5_opportunity_allocator.yaml").read_text(encoding="utf-8")
    )
    allocator = dict(config["allocator"])
    allocator["action_rank_thresholds"] = tuple(allocator["action_rank_thresholds"])
    allocator["action_levels"] = tuple(allocator["action_levels"])
    resolved = OpportunityAllocatorConfig(**allocator)

    assert config["experiment"]["allocator_query_start"].year == 2022
    assert resolved.action_levels == (0.0, 0.25, 0.50, 0.75, 1.0)
    assert config["learnability"]["minimum_obvious_positive_folds"] == 4
    assert config["learnability"]["minimum_nonlinear_fold_wins"] == 4
    assert config["selection"]["maximum_worst_drawdown"] <= 0.20


def test_audit_json_values_convert_pandas_timestamps():
    converted = _native(
        {
            "timestamp": pd.Timestamp("2023-01-03", tz="UTC"),
            "missing": pd.NaT,
            "passed": np.bool_(True),
        }
    )

    assert converted == {
        "timestamp": "2023-01-03T00:00:00+00:00",
        "missing": None,
        "passed": True,
    }


def test_final_allocator_is_bounded_and_reuses_causal_evidence():
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase5_final_allocator.yaml").read_text(encoding="utf-8")
    )

    assert config["experiment"]["source_evidence_run"].endswith("phase5_opportunity_allocator_v1")
    assert config["experiment"]["candidates"] == [
        "B0_c0_static",
        "B1_obvious_signal",
        "B3_calibrated_obvious",
    ]
    assert config["selection"]["minimum_out_of_year_positive_folds"] == 4
    assert config["selection"]["minimum_improvement_folds"] == 4
