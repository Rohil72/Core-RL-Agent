"""Train the fixed Phase 5 adapter from arbitrary source-market latent exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.adapter import DecisionAdapterConfig  # noqa: E402
from src.decision.dataset import DecisionDatasetConfig, build_decision_frame  # noqa: E402
from src.decision.losses import DecisionLossConfig  # noqa: E402
from src.decision.trainer import DecisionTrainingConfig, train_decision_adapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", required=True)
    parser.add_argument("--val-latents", required=True)
    parser.add_argument("--precomputed-glob", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    train = build_decision_frame(pd.read_parquet(args.train_latents), args.precomputed_glob, DecisionDatasetConfig())
    val = build_decision_frame(pd.read_parquet(args.val_latents), args.precomputed_glob, DecisionDatasetConfig())
    result = train_decision_adapter(
        train, val, args.output, DecisionAdapterConfig(),
        DecisionTrainingConfig(seed=args.seed), DecisionLossConfig(),
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
