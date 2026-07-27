"""Build and run global-transformer, dual-regional-memory Phase 6 jobs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_phase5_transfer_latents import run as export_transfer  # noqa: E402
from scripts.run_phase6_frozen_memory_sweep import (  # noqa: E402
    _evaluate_all,
    _job,
    _native,
    _rel,
    _resolve,
    _write_json,
    _write_yaml,
)
from src.backtest.market_memory_evaluator import run_market_memory_evaluation  # noqa: E402


TOPOLOGY = "global_dual_regional"


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    testbed = yaml.safe_load(
        _resolve(config["experiment"]["source_testbed_config"]).read_text(encoding="utf-8")
    )
    missing = sorted(set(config["experiment"]["markets"]) - set(testbed["markets"]))
    if missing:
        raise ValueError(f"Markets absent from source testbed: {missing}")
    return config, testbed


def _key(market: str, seed: int) -> str:
    return f"{TOPOLOGY}_{market}_seed_{seed}"


def _artifacts(output: Path, market: str, seed: int) -> dict[str, Path]:
    root = output / "aligned" / _key(market, seed)
    return {
        "alignment": root / "alignment.pt",
        "alignment_audit": root / "alignment.json",
        "encoder": root / "encoder.pt",
        "adapter": root / "adapter.pt",
    }


def _regional_encoder_config(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> Path:
    source = _resolve(config["experiment"]["source_run"])
    original = source / "generated_configs" / "encoders" / f"global_seed_{seed}.yaml"
    resolved = yaml.safe_load(original.read_text(encoding="utf-8"))
    data_root = Path(testbed["experiment"]["data_root"])
    resolved["data"].pop("precomputed_sources", None)
    resolved["data"]["precomputed_dir"] = f"{data_root.as_posix()}/{market}/*.parquet"
    resolved["data"]["market"] = market
    resolved["data"]["market_balanced_sampling"] = False
    resolved["training"]["seed"] = seed
    destination = output / "generated_configs" / "alignment" / f"{_key(market, seed)}.yaml"
    _write_yaml(destination, resolved)
    return destination


def _preflight(config: dict[str, Any], testbed: dict[str, Any]) -> dict[str, Any]:
    source = _resolve(config["experiment"]["source_run"])
    data_root = _resolve(testbed["experiment"]["data_root"])
    required: list[Path] = []
    for seed in map(int, config["experiment"]["seeds"]):
        required.extend(
            (
                source / "models" / f"global_seed_{seed}" / "final_model.pt",
                source / "adapters" / f"global_seed_{seed}" / "decision_adapter.pt",
                source / "generated_configs" / "encoders" / f"global_seed_{seed}.yaml",
            )
        )
    missing = [str(path) for path in required if not path.exists()]
    for market in config["experiment"]["markets"]:
        if not list((data_root / market).glob("*.parquet")):
            missing.append(str(data_root / market / "*.parquet"))
    if missing:
        raise FileNotFoundError("Dual-memory preflight failed:\n" + "\n".join(f"- {p}" for p in missing))
    return {"required_artifacts": len(required), "markets": len(config["experiment"]["markets"])}


def _export_all(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> dict[str, Any]:
    artifacts = _artifacts(output, market, seed)
    periods = {
        "train": testbed["periods"]["encoder_train"],
        str(config["experiment"]["development_period"]): testbed["periods"]["policy_development"],
        str(config["experiment"]["selection_period"]): testbed["periods"]["policy_selection"],
    }
    target_glob = f"{testbed['experiment']['data_root']}/{market}/*.parquet"
    destination = output / "decisions" / _key(market, seed)
    rows: dict[str, int] = {}
    for period, (start, end) in periods.items():
        path = destination / f"{period}.parquet"
        if path.exists() and path.stat().st_size > 0:
            summary = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        else:
            summary = export_transfer(
                str(artifacts["encoder"]),
                str(artifacts["adapter"]),
                target_glob,
                str(start),
                str(end),
                str(path),
            )
        rows[period] = int(summary["rows"])
    payload = {"market": market, "seed": seed, "rows": rows, "status": "completed"}
    _write_json(destination / "export_complete.json", payload)
    return payload


def _memory_config(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
    period: str,
) -> dict[str, Any]:
    decision_root = output / "decisions" / _key(market, seed)
    memory = {
        **config["memory"],
        "target_upside": "decision_mfe",
        "target_alpha": "decision_net_alpha",
        "target_downside": "decision_mae",
        "target_path_quality": "decision_path_quality",
        "target_holding_period": "decision_holding_sessions",
        "score_mode": "alpha_lcb",
        "require_outcome_availability": False,
        "max_median_distance": None,
        "confidence_reference_distance": None,
    }
    policy = copy.deepcopy(config["policy"])
    policy["slippage_bps"] = float(testbed["markets"][market]["execution_cost_bps"])
    return {
        "data": {
            "train_latents": _rel(decision_root / "train.parquet"),
            "test_latents": _rel(decision_root / f"{period}.parquet"),
            "precomputed_glob": f"{testbed['experiment']['data_root']}/{market}/*.parquet",
            "output_dir": _rel(output / "memory" / _key(market, seed) / period),
        },
        "memory": memory,
        "policy": policy,
        "evaluation": {
            "baselines": "model_head,momentum,random",
            "memory_metric_target": "decision_net_alpha",
        },
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }


def _retrieve_all(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> dict[str, Any]:
    periods = (
        str(config["experiment"]["development_period"]),
        str(config["experiment"]["selection_period"]),
    )
    completed: list[str] = []
    for period in periods:
        result = output / "memory" / _key(market, seed) / period / "eval"
        expected = (result / "signals.parquet", result / "neighbors.parquet", result / "metrics.json")
        if not all(path.exists() and path.stat().st_size > 0 for path in expected):
            resolved = _memory_config(config, testbed, output, market, seed, period)
            config_path = output / "generated_configs" / "memory" / _key(market, seed) / f"{period}.yaml"
            _write_yaml(config_path, resolved)
            run_market_memory_evaluation(resolved, PROJECT_ROOT, run_id="eval")
        if not all(path.exists() and path.stat().st_size > 0 for path in expected):
            raise RuntimeError(f"Incomplete regional retrieval for {market} seed {seed} {period}.")
        completed.append(period)
    marker = output / "retrieval_status" / f"{_key(market, seed)}.json"
    payload = {"market": market, "seed": seed, "periods": completed, "status": "completed"}
    _write_json(marker, payload)
    return payload


def _build(
    config_path: Path,
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    python: str,
) -> dict[str, Any]:
    preflight = _preflight(config, testbed)
    source = _resolve(config["experiment"]["source_run"])
    jobs: list[dict[str, Any]] = []
    script = _rel(Path(__file__))
    common = [python, script, "--config", _rel(config_path), "--run-id", output.name]
    retrieval_jobs: list[str] = []
    for seed in map(int, config["experiment"]["seeds"]):
        for market in config["experiment"]["markets"]:
            key = _key(market, seed)
            regional_config = _regional_encoder_config(config, testbed, output, market, seed)
            artifacts = _artifacts(output, market, seed)
            align_job = f"align_{key}"
            jobs.append(
                {
                    **_job(
                        align_job,
                        [
                            python,
                            "scripts/run_phase5_transformer_alignment.py",
                            "--base-checkpoint",
                            _rel(source / "models" / f"global_seed_{seed}" / "final_model.pt"),
                            "--base-config",
                            _rel(regional_config),
                            "--adapter-checkpoint",
                            _rel(source / "adapters" / f"global_seed_{seed}" / "decision_adapter.pt"),
                            "--output",
                            _rel(artifacts["alignment"]),
                            "--config",
                            _rel(config_path),
                            "--transfer-encoder-output",
                            _rel(artifacts["encoder"]),
                            "--transfer-adapter-output",
                            _rel(artifacts["adapter"]),
                            "--require-eligible",
                        ],
                        list(artifacts.values()),
                        stage="align",
                        inputs=[
                            source / "models" / f"global_seed_{seed}" / "final_model.pt",
                            source / "adapters" / f"global_seed_{seed}" / "decision_adapter.pt",
                            regional_config,
                            f"{testbed['experiment']['data_root']}/{market}/*.parquet",
                        ],
                    ),
                    "uses_gpu": True,
                    "estimated_hours": 0.75,
                }
            )
            export_job = f"export_{key}"
            decision_root = output / "decisions" / key
            jobs.append(
                {
                    **_job(
                        export_job,
                        [*common, "--stage", "export", "--market", market, "--seed", str(seed)],
                        [
                            decision_root / "train.parquet",
                            decision_root / f"{config['experiment']['development_period']}.parquet",
                            decision_root / f"{config['experiment']['selection_period']}.parquet",
                            decision_root / "export_complete.json",
                        ],
                        stage="export",
                        dependencies=[align_job],
                        inputs=[artifacts["encoder"], artifacts["adapter"], f"{testbed['experiment']['data_root']}/{market}/*.parquet"],
                    ),
                    "uses_gpu": True,
                    "estimated_hours": 0.40,
                }
            )
            retrieve_job = f"retrieve_{key}"
            retrieval_jobs.append(retrieve_job)
            outputs = [output / "retrieval_status" / f"{key}.json"]
            for period in (
                str(config["experiment"]["development_period"]),
                str(config["experiment"]["selection_period"]),
            ):
                root = output / "memory" / key / period / "eval"
                outputs.extend((root / "signals.parquet", root / "neighbors.parquet", root / "metrics.json"))
            jobs.append(
                _job(
                    retrieve_job,
                    [*common, "--stage", "retrieve", "--market", market, "--seed", str(seed)],
                    outputs,
                    stage="retrieve",
                    dependencies=[export_job],
                    inputs=[
                        decision_root / "train.parquet",
                        decision_root / f"{config['experiment']['development_period']}.parquet",
                        decision_root / f"{config['experiment']['selection_period']}.parquet",
                        f"{testbed['experiment']['data_root']}/{market}/*.parquet",
                    ],
                )
            )
    jobs.append(
        _job(
            "evaluate_dual_regional_memory",
            [*common, "--stage", "evaluate"],
            [output / "market_results.csv", output / "leaderboard.csv", output / "selection.json", output / "evaluation_complete.json"],
            stage="evaluate",
            dependencies=retrieval_jobs,
            inputs=[
                output / "memory" / _key(market, seed) / period / "eval" / artifact
                for market in config["experiment"]["markets"]
                for seed in map(int, config["experiment"]["seeds"])
                for period in (
                    str(config["experiment"]["development_period"]),
                    str(config["experiment"]["selection_period"]),
                )
                for artifact in ("signals.parquet", "neighbors.parquet")
            ],
        )
    )
    manifest = {
        "run": {
            "id": output.name,
            "strict_environment": True,
            "stale_lock_seconds": 900,
            "inputs": [_rel(config_path), str(config["experiment"]["source_testbed_config"])],
            "source_patterns": [
                "src/decision/**/*.py",
                "src/models/**/*.py",
                "src/backtest/**/*.py",
                "src/eval/**/*.py",
                "src/memory/**/*.py",
                "src/trainers/train_cycle_model.py",
                "scripts/run_phase5_transformer_alignment.py",
                "scripts/export_phase5_transfer_latents.py",
                "scripts/run_phase6_frozen_memory_sweep.py",
                "scripts/run_phase6_dual_regional_memory.py",
                "configs/phase6_dual_regional_memory.yaml",
            ],
            "hardware": {
                **config["hardware"],
                "minimum_free_storage_gb": 10,
            },
            "budgets": {},
        },
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    summary = {
        "run_id": output.name,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "job_count": len(jobs),
        "jobs_by_stage": {
            stage: sum(job["stage"] == stage for job in jobs)
            for stage in ("align", "export", "retrieve", "evaluate")
        },
        "declared_gpu_hours": sum(job["estimated_hours"] for job in jobs if job["uses_gpu"]),
        "preflight": preflight,
    }
    _write_json(output / "build_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config, testbed = _load(config_path)
    output = _resolve(config["experiment"]["output_root"]) / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        return _build(config_path, config, testbed, output, args.python)
    if args.stage == "export":
        if args.seed is None or not args.market:
            raise ValueError("--seed and --market are required for export.")
        return _export_all(config, testbed, output, args.market, args.seed)
    if args.stage == "retrieve":
        if args.seed is None or not args.market:
            raise ValueError("--seed and --market are required for retrieve.")
        return _retrieve_all(config, testbed, output, args.market, args.seed)
    if args.stage == "evaluate":
        return _evaluate_all(config, testbed, output)
    raise ValueError(f"Unknown stage: {args.stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_dual_regional_memory.yaml")
    parser.add_argument("--run-id", default="phase6_dual_regional_memory_v1")
    parser.add_argument("--stage", choices=("build", "export", "retrieve", "evaluate"), default="build")
    parser.add_argument("--market")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(_native(run(args)), indent=2))


if __name__ == "__main__":
    main()

