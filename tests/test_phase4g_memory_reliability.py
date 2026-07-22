import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.run_phase4g_memory_reliability import _merge_csv, run_confirmation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config():
    return yaml.safe_load((PROJECT_ROOT / "configs" / "phase4g_memory_reliability.yaml").read_text())


def test_phase4g_coverage_candidates_are_fixed_and_unique():
    candidates = _config()["coverage_candidates"]

    assert [item["nominal_coverage"] for item in candidates] == [1.0, 0.75, 0.5, 0.25]
    assert len({item["id"] for item in candidates}) == len(candidates)


def test_phase4g_confirmation_remains_sealed_after_failed_validation(tmp_path):
    selection = {
        "confirmation_unlocked": False,
        "smoke_only": False,
        "failure_reason": "reliability gate failed",
    }

    run_confirmation(_config(), tmp_path, [], selection, resume=False)

    status = json.loads((tmp_path / "confirmation_status.json").read_text())
    assert status == {"executed": False, "reason": "reliability gate failed"}


def test_phase4g_csv_merge_preserves_other_splits_and_replaces_matching_rows(tmp_path):
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        [
            {"split": "val", "fold": "fold_01", "score": 1.0},
            {"split": "test", "fold": "fold_01", "score": 2.0},
        ]
    ).to_csv(path, index=False)

    _merge_csv(
        path,
        pd.DataFrame([{"split": "test", "fold": "fold_01", "score": 3.0}]),
        ["split", "fold"],
    )

    result = pd.read_csv(path)
    assert result.to_dict("records") == [
        {"split": "test", "fold": "fold_01", "score": 3.0},
        {"split": "val", "fold": "fold_01", "score": 1.0},
    ]
