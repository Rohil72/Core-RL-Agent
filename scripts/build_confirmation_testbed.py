"""Compile the one-shot confirmation DAG after development selection passes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_final_testbed import _job, _memory_config, _rel, _write_json, _write_yaml  # noqa: E402
from src.eval.confirmation_lock import create_confirmation_lock  # noqa: E402


def build(config_path: Path, run_id: str, core_python: str, rl_python: str, development_state: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["experiment"]["output_root"] / run_id
    selection_path = output / "selection" / "final_policy_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "selected" or not selection.get("selected_candidate"):
        raise RuntimeError("Final development selection has not passed; confirmation remains sealed.")
    algorithm, representation = str(selection["selected_candidate"]).split("__", 1)
    development_contract_path = development_state / "contract.json"
    development_contract = json.loads(development_contract_path.read_text(encoding="utf-8"))
    phase4e = yaml.safe_load((PROJECT_ROOT / "configs/phase4e_cross_market_sharpe.yaml").read_text(encoding="utf-8"))
    markets = list(config["markets"])
    seeds = [int(value) for value in config["research_matrix"]["seeds"]]
    data_root = Path(config["experiment"]["data_root"])
    start, end = config["periods"]["locked_confirmation"]
    jobs: list[dict[str, Any]] = []
    records = []
    locked_artifacts: list[Path] = [config_path, selection_path, development_contract_path]
    locked_artifacts.extend(sorted(data_root.glob("*/*.parquet")))

    for market in markets:
        for seed in seeds:
            key = f"{representation}_{market}_seed_{seed}"
            encoder_name = f"regional_{market}_seed_{seed}" if representation == "regional" else f"global_seed_{seed}"
            checkpoint = output / "models" / encoder_name / "final_model.pt"
            adapter_key = f"global_seed_{seed}" if representation == "global" else key
            adapter = output / "adapters" / adapter_key / "decision_adapter.pt"
            extension = ".pt" if algorithm == "bandit" else ".d3"
            policy_model = output / "full_policies" / key / f"{algorithm}{extension}"
            dataset = output / "policy_datasets" / key / "development.npz"
            locked_artifacts.extend([checkpoint, adapter, policy_model, dataset])
            target_glob = f"{data_root.as_posix()}/{market}/*.parquet"
            decisions = output / "confirmation" / "decisions" / key / "confirmation.parquet"
            export_job = _job(
                jobs,
                f"confirmation_export_{key}",
                [core_python, "scripts/export_phase5_transfer_latents.py", "--encoder-checkpoint", _rel(checkpoint), "--adapter-checkpoint", _rel(adapter), "--target-glob", target_glob, "--start", str(start), "--end", str(end), "--output", _rel(decisions)],
                [decisions, decisions.with_suffix(".json")],
                stage="confirmation",
                inputs=[checkpoint, adapter, target_glob],
                gpu=True,
                hours=0.3,
            )
            memory_dir = output / "confirmation" / "memory" / key
            memory_cfg = output / "confirmation" / "generated_configs" / "memory" / f"{key}.yaml"
            train_decisions = output / "adapters" / key / "train_decisions.parquet"
            _write_yaml(memory_cfg, _memory_config(phase4e, config["markets"][market], train_decisions, decisions, target_glob, memory_dir))
            memory_job = _job(
                jobs,
                f"confirmation_memory_{key}",
                [core_python, "scripts/run_market_memory_backtest.py", "--config", _rel(memory_cfg), "--run-id", "eval"],
                [memory_dir / "eval" / "metrics.json", memory_dir / "eval" / "signals.parquet"],
                stage="confirmation",
                dependencies=[export_job],
                inputs=[memory_cfg, train_decisions, decisions, target_glob],
                hours=0.3,
            )
            evaluation = output / "confirmation" / "evaluations" / key / algorithm
            eval_job = _job(
                jobs,
                f"confirmation_eval_{key}_{algorithm}",
                [rl_python, "scripts/evaluate_phase5_offline_policy.py", "--algorithm", algorithm, "--model", _rel(policy_model), "--dataset", _rel(dataset), "--decisions", _rel(decisions), "--signals", _rel(memory_dir / "eval" / "signals.parquet"), "--output", _rel(evaluation)],
                [evaluation / "metrics.json", evaluation / "equity_curve.csv"],
                stage="confirmation",
                dependencies=[memory_job],
                inputs=[policy_model, dataset, decisions, memory_dir / "eval" / "signals.parquet"],
                gpu=True,
                hours=0.2,
            )
            records.append(
                {
                    "market": market,
                    "seed": seed,
                    "metrics": _rel(evaluation / "metrics.json"),
                    "equity": _rel(evaluation / "equity_curve.csv"),
                    "job": eval_job,
                }
            )

    lock_path = output / "confirmation_lock.json"
    create_confirmation_lock(lock_path, locked_artifacts, markets, confirmation_year=2025)
    matrix_path = output / "confirmation" / "confirmation_matrix.json"
    _write_json(matrix_path, {"selected_candidate": selection["selected_candidate"], "records": [{key: value for key, value in row.items() if key != "job"} for row in records]})
    summary_path = output / "confirmation" / "confirmation_summary.json"
    aggregate_job = _job(
        jobs,
        "aggregate_confirmation",
        [core_python, "scripts/aggregate_confirmation_results.py", "--matrix", _rel(matrix_path), "--config", _rel(config_path), "--output", _rel(summary_path)],
        [summary_path, summary_path.with_name("confirmation_market_metrics.csv")],
        stage="confirmation",
        dependencies=[row["job"] for row in records],
        inputs=[matrix_path, *[row["metrics"] for row in records], *[row["equity"] for row in records]],
    )
    _job(
        jobs,
        "close_confirmation_lock",
        [core_python, "scripts/run_phase5_international.py", "--config", _rel(config_path), "--run-id", run_id, "--stage", "close", "--idempotent-close", "--results", _rel(summary_path), *[row["metrics"] for row in records]],
        [lock_path],
        stage="confirmation",
        dependencies=[aggregate_job],
        inputs=[summary_path, *[row["metrics"] for row in records]],
    )
    manifest = {
        "run": {
            "id": f"{run_id}_confirmation",
            "strict_environment": True,
            "stale_lock_seconds": 900,
            "inputs": [_rel(config_path), _rel(selection_path), _rel(development_contract_path)],
            "source_patterns": ["src/**/*.py", "scripts/**/*.py", "configs/**/*.yaml", "requirements*.txt"],
            "hardware": {
                **config["hardware"],
                "minimum_free_storage_gb": config["storage"]["minimum_free_gb"],
                "required_runtime_fingerprint": development_contract["runtime"],
                "required_source_fingerprint": development_contract["source"],
            },
            "budgets": {"confirmation": config["budgets"]["confirmation"]},
        },
        "jobs": jobs,
    }
    manifest_path = output / "confirmation" / "confirmation_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    result = {
        "run_id": run_id,
        "selected_candidate": selection["selected_candidate"],
        "manifest": _rel(manifest_path),
        "lock": _rel(lock_path),
        "job_count": len(jobs),
        "status": "locked_not_executed",
    }
    _write_json(output / "confirmation" / "build_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_research_testbed.yaml")
    parser.add_argument("--run-id", default="phase6_international_v1")
    parser.add_argument("--development-state", required=True)
    parser.add_argument("--core-python", default=sys.executable)
    parser.add_argument("--rl-python", default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(build(Path(args.config), args.run_id, args.core_python, args.rl_python, Path(args.development_state)), indent=2))


if __name__ == "__main__":
    main()
