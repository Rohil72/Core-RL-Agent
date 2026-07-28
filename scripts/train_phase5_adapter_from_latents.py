"""Train the fixed Phase 5 adapter from arbitrary source-market latent exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.decision.adapter import DecisionAdapterConfig  # noqa: E402
from src.decision.dataset import DecisionDatasetConfig, build_decision_frame  # noqa: E402
from src.decision.losses import DecisionLossConfig  # noqa: E402
from src.decision.trainer import DecisionTrainingConfig, train_decision_adapter  # noqa: E402


def run(
    *,
    train_latents: str,
    val_latents: str,
    precomputed_glob: list[str],
    output: str,
    seed: int,
    config_path: str | None = None,
) -> dict[str, object]:
    """Train one decision adapter from frozen latent exports and declared settings."""

    values = (
        yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if config_path
        else {}
    ) or {}
    dataset_cfg = DecisionDatasetConfig(**values.get("decision_dataset", {}))
    adapter_cfg = DecisionAdapterConfig(**values.get("adapter", {}))
    training_values = dict(values.get("training", {}))
    training_values["seed"] = int(seed)
    training_cfg = DecisionTrainingConfig(**training_values)
    loss_cfg = DecisionLossConfig(**values.get("loss", {}))
    train = build_decision_frame(
        pd.read_parquet(train_latents),
        precomputed_glob,
        dataset_cfg,
    )
    val = build_decision_frame(
        pd.read_parquet(val_latents),
        precomputed_glob,
        dataset_cfg,
    )
    return train_decision_adapter(
        train,
        val,
        output,
        adapter_cfg,
        training_cfg,
        loss_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", required=True)
    parser.add_argument("--val-latents", required=True)
    parser.add_argument("--precomputed-glob", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = run(
        train_latents=args.train_latents,
        val_latents=args.val_latents,
        precomputed_glob=args.precomputed_glob,
        output=args.output,
        seed=args.seed,
        config_path=args.config,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
