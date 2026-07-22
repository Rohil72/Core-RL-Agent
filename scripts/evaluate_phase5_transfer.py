"""Backtest source-market decision memory on a source-normalized target market."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.market_memory_evaluator import run_market_memory_evaluation  # noqa: E402
from src.memory.experience import latent_columns  # noqa: E402


def _retrieval_copy(source: str, destination: Path) -> None:
    frame = pd.read_parquet(source)
    frame = frame.drop(columns=latent_columns(frame))
    decision = sorted(
        [column for column in frame if column.startswith("decision_") and column.removeprefix("decision_").isdigit()],
        key=lambda column: int(column.removeprefix("decision_")),
    )
    frame = frame.rename(columns={column: f"latent_{index}" for index, column in enumerate(decision)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-memory", required=True)
    parser.add_argument("--target-decisions", required=True)
    parser.add_argument("--target-precomputed-glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase4e-config", default="configs/phase4e_cross_market_sharpe.yaml")
    args = parser.parse_args()
    output = Path(args.output)
    source = output / "retrieval_source.parquet"
    target = output / "retrieval_target.parquet"
    _retrieval_copy(args.source_memory, source)
    _retrieval_copy(args.target_decisions, target)
    base = yaml.safe_load(Path(args.phase4e_config).read_text(encoding="utf-8"))
    evaluation = {
        "data": {
            "train_latents": str(source.resolve()), "test_latents": str(target.resolve()),
            "precomputed_glob": str(Path(args.target_precomputed_glob).resolve()), "output_dir": str(output.resolve()),
        },
        "memory": {**base["memory_defaults"], "target_alpha": "decision_net_alpha", "score_mode": "alpha_lcb"},
        "policy": base["policy"],
        "evaluation": {"baselines": "", "memory_metric_target": "decision_utility"},
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }
    result = run_market_memory_evaluation(evaluation, PROJECT_ROOT, run_id="eval")
    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
