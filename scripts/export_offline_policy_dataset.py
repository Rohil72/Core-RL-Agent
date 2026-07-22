"""Export one chronological offline-policy dataset from market-memory signals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.offline_policy import (  # noqa: E402
    OfflinePolicyDatasetConfig,
    build_offline_policy_dataset,
    save_offline_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()

    values = {}
    if args.config:
        raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        allowed = {item.name for item in fields(OfflinePolicyDatasetConfig)}
        values = {key: value for key, value in raw.items() if key in allowed}
        for key in ("exposures", "behavior_policies"):
            if key in values:
                values[key] = tuple(values[key])
    config = OfflinePolicyDatasetConfig(**values)
    dataset = build_offline_policy_dataset(pd.read_parquet(args.signals), config)
    save_offline_dataset(args.output, dataset)
    summary = {
        "output": str(Path(args.output)),
        "rows": int(len(dataset["observations"])),
        "observation_dim": int(dataset["observations"].shape[1]),
        "episode_count": int(len(set(dataset["episode_ids"].tolist()))),
        "action_values": sorted(set(float(value) for value in dataset["actions"].ravel())),
    }
    summary_path = Path(args.output).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
