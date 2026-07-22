"""Export identical offline tuples for contextual bandit, CQL, IQL, and TD3+BC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.offline_policy import OfflinePolicyDatasetConfig, build_offline_policy_dataset, save_offline_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase5-run", default="reports/phase5/phase5_v1")
    parser.add_argument("--output", default="reports/phase5/policy_datasets")
    args = parser.parse_args()
    source, output = Path(args.phase5_run), Path(args.output)
    exported = []
    for path in sorted(source.glob("fold_*/seed_*/train_decisions.parquet")):
        fold, seed = path.parents[1].name, path.parent.name
        destination = output / fold / seed / "offline_policy_dataset.npz"
        save_offline_dataset(destination, build_offline_policy_dataset(pd.read_parquet(path), OfflinePolicyDatasetConfig()))
        exported.append(str(destination))
    if not exported:
        raise FileNotFoundError(f"No Phase 5 train decision tables found under {source}.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"datasets": exported}, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_count": len(exported), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
