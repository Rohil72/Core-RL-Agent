"""Build and compare the final temporal-transport encoder experiment."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _job(
    job_id: str,
    command: list[str],
    outputs: list[Path],
    *,
    stage: str,
    dependencies: list[str] | None = None,
    inputs: list[str | Path] | None = None,
    gpu: bool = False,
    hours: float = 0.0,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "stage": stage,
        "command": command,
        "depends_on": dependencies or [],
        "inputs": [str(value) if isinstance(value, str) else _rel(value) for value in (inputs or [])],
        "expected_outputs": [_rel(path) for path in outputs],
        "uses_gpu": gpu,
        "gpu_count": 1 if gpu else 0,
        "estimated_hours": hours,
    }


def _encoder_config(
    base: dict[str, Any],
    study: dict[str, Any],
    variant: str,
    seed: int,
    output: Path,
) -> tuple[Path, Path, Path]:
    resolved = copy.deepcopy(base)
    data_cfg = study["data"]
    root = Path(str(data_cfg["root"]))
    markets = [str(value) for value in data_cfg["markets"]]
    resolved["data"].update(
        {
            "precomputed_sources": [
                {"market": market, "glob": f"{root.as_posix()}/{market}/*.parquet"}
                for market in markets
            ],
            "market_balanced_sampling": bool(data_cfg.get("market_balanced_sampling", True)),
            "relative_outcomes": copy.deepcopy(data_cfg["relative_outcomes"]),
            "cross_sectional_relative_features": copy.deepcopy(
                data_cfg["cross_sectional_relative_features"]
            ),
            "exclude_years": [],
            "use_detector_targets": False,
        }
    )
    resolved["data"].pop("precomputed_dir", None)
    relative_feature_names = list(data_cfg["cross_sectional_relative_features"].values())
    for feature in relative_feature_names:
        if feature not in resolved["features"]["sequence"]:
            resolved["features"]["sequence"].append(feature)
    for target in ("future_universe_alpha_63", "future_blended_alpha_63"):
        if target not in resolved["features"]["future_targets"]:
            resolved["features"]["future_targets"].append(target)

    training = study["training"]
    run_name = f"{variant}_seed_{seed}"
    model_dir = output / "models" / run_name
    latent_dir = output / "latents" / run_name
    resolved["training"].update(
        {
            "seed": seed,
            "epochs": int(training["epochs"]),
            "batch_size": int(training["batch_size"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "grad_clip": float(training["grad_clip"]),
            "validation_selection_metric": str(training["validation_selection_metric"]),
            "validation_selection_mode": str(training["validation_selection_mode"]),
            "use_amp": bool(training["use_amp"]),
            "checkpoint_interval_steps": int(training["checkpoint_interval_steps"]),
            "auto_resume": True,
            "device": "cuda",
            "model_dir": _rel(model_dir),
            "loss": copy.deepcopy(study["variants"][variant]["loss"]),
        }
    )
    periods = study["periods"]
    resolved["split"].update(
        {
            "train_years": int(periods["train_years"]),
            "val_years": int(periods["validation_years"]),
            "test_years": int(periods["test_years"]),
            "step_years": 1,
            "selected_fold": int(periods["selected_fold"]),
            "ticker_holdout_fraction": 0.0,
            "seed": seed,
            "evaluation_periods": {"observed": copy.deepcopy(periods["observed"])},
        }
    )
    resolved["hardware"] = {"vram_fraction": float(study["hardware"]["vram_fraction"])}
    resolved["latent_export"] = {
        "splits": ["train", "val", "test", "observed"],
        "output_dir": _rel(latent_dir),
    }
    resolved["evaluation"]["reports_dir"] = _rel(output / "encoder_reports" / run_name)
    path = output / "generated_configs" / "encoders" / f"{run_name}.yaml"
    _write_yaml(path, resolved)
    return path, model_dir / "final_model.pt", latent_dir


def build(config_path: Path, run_id: str, python: str) -> dict[str, Any]:
    study = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = PROJECT_ROOT / study["study"]["base_encoder_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / study["study"]["output_root"] / run_id
    jobs: list[dict[str, Any]] = []
    evaluation_jobs: list[str] = []
    price_glob = f"{Path(study['data']['root']).as_posix()}/**/*.parquet"

    for variant in study["variants"]:
        for seed_value in study["training"]["seeds"]:
            seed = int(seed_value)
            run_name = f"{variant}_seed_{seed}"
            encoder_cfg, checkpoint, latent_dir = _encoder_config(
                base, study, variant, seed, output
            )
            train_id = f"train_{run_name}"
            jobs.append(
                _job(
                    train_id,
                    [python, "-m", "src.trainers.train_cycle_model", "--config-path", _rel(encoder_cfg), "--resume"],
                    [
                        checkpoint,
                        checkpoint.parent / "training_complete.json",
                        latent_dir / "train_latents.parquet",
                        latent_dir / "test_latents.parquet",
                        latent_dir / "observed_latents.parquet",
                    ],
                    stage="train",
                    inputs=[encoder_cfg, price_glob],
                    gpu=True,
                    hours=float(study["hardware"]["estimated_training_hours_per_job"]),
                )
            )
            for period in ("test", "observed"):
                eval_root = output / "memory" / run_name / period
                query_latents = latent_dir / f"{period}_latents.parquet"
                memory_cfg = {
                    "data": {
                        "train_latents": _rel(latent_dir / "train_latents.parquet"),
                        "test_latents": _rel(query_latents),
                        "precomputed_glob": price_glob,
                        "output_dir": _rel(eval_root),
                    },
                    "memory": copy.deepcopy(study["memory"]),
                    "policy": copy.deepcopy(study["policy"]),
                    "evaluation": {
                        "baselines": "momentum,random",
                        "memory_metric_target": "future_blended_alpha_63",
                    },
                    "run": {"write_neighbors": True, "write_memory_reports": False},
                }
                memory_path = (
                    output / "generated_configs" / "memory" / f"{run_name}_{period}.yaml"
                )
                _write_yaml(memory_path, memory_cfg)
                eval_id = f"evaluate_{run_name}_{period}"
                evaluation_jobs.append(eval_id)
                jobs.append(
                    _job(
                        eval_id,
                        [python, "scripts/run_market_memory_backtest.py", "--config", _rel(memory_path), "--run-id", "eval"],
                        [eval_root / "eval" / "metrics.json", eval_root / "eval" / "trades.csv"],
                        stage="evaluate",
                        dependencies=[train_id],
                        inputs=[memory_path, checkpoint, latent_dir / "train_latents.parquet", query_latents],
                    )
                )

    comparison = output / "comparison"
    jobs.append(
        _job(
            "compare",
            [python, "scripts/run_temporal_transport_study.py", "--config", _rel(config_path), "--run-id", run_id, "--stage", "compare"],
            [comparison / "variant_summary.csv", comparison / "paired_results.csv", comparison / "verdict.json"],
            stage="compare",
            dependencies=evaluation_jobs,
            inputs=[output / "memory"],
        )
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "promotion_allowed": False,
        "run": {
            "id": run_id,
            "strict_environment": True,
            "hardware": {
                "vram_fraction": float(study["hardware"]["vram_fraction"]),
            },
        },
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    result = {
        "run_id": run_id,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "training_jobs": sum(job["stage"] == "train" for job in jobs),
        "evaluation_jobs": len(evaluation_jobs),
        "diagnostic_only": True,
    }
    _write_json(output / "build_summary.json", result)
    return result


def compare(config_path: Path, run_id: str) -> dict[str, Any]:
    study = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = PROJECT_ROOT / study["study"]["output_root"] / run_id
    rows: list[dict[str, Any]] = []
    for period in ("test", "observed"):
        for variant in study["variants"]:
            for seed_value in study["training"]["seeds"]:
                seed = int(seed_value)
                metrics_path = (
                    output / "memory" / f"{variant}_seed_{seed}" / period / "eval" / "metrics.json"
                )
                if not metrics_path.is_file():
                    raise FileNotFoundError(f"Missing evaluation metrics: {metrics_path}")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "period": period,
                        "variant": variant,
                        "seed": seed,
                        "total_return": float(metrics["total_return"]),
                        "annualized_return": float(metrics["annualized_return"]),
                        "sharpe": float(metrics["sharpe"]),
                        "max_drawdown": float(metrics["max_drawdown"]),
                        "trade_count": int(metrics["trade_count"]),
                        "score_realized_return_spearman": float(
                            metrics.get("score_realized_return_spearman", 0.0)
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    summary = frame.groupby(["period", "variant"], as_index=False).agg(
        mean_sharpe=("sharpe", "mean"),
        median_sharpe=("sharpe", "median"),
        worst_sharpe=("sharpe", "min"),
        mean_return=("total_return", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        mean_trades=("trade_count", "mean"),
        mean_score_spearman=("score_realized_return_spearman", "mean"),
    )
    baseline = str(study["comparison"]["baseline"])
    candidate = str(study["comparison"]["candidate"])
    paired = frame[frame["variant"] == baseline].merge(
        frame[frame["variant"] == candidate], on=["period", "seed"], suffixes=("_baseline", "_candidate")
    )
    paired["sharpe_delta"] = paired["sharpe_candidate"] - paired["sharpe_baseline"]
    paired["return_delta"] = paired["total_return_candidate"] - paired["total_return_baseline"]
    paired["drawdown_delta"] = paired["max_drawdown_candidate"] - paired["max_drawdown_baseline"]
    gates = study["comparison"]
    development_paired = paired[paired["period"] == "test"]
    observed_paired = paired[paired["period"] == "observed"]
    candidate_summary = summary.set_index(["period", "variant"]).loc[("test", candidate)]
    seed_wins = int((development_paired["sharpe_delta"] > 0).sum())
    passed = bool(
        seed_wins >= int(gates["minimum_seed_wins"])
        and float(development_paired["sharpe_delta"].mean()) > float(gates["minimum_mean_sharpe_delta"])
        and float(development_paired["sharpe_delta"].min()) >= float(gates["minimum_worst_seed_sharpe_delta"])
        and abs(float(candidate_summary["worst_drawdown"])) <= float(gates["maximum_candidate_drawdown"])
    )
    verdict = {
        "status": "diagnostic_completed",
        "candidate": candidate,
        "candidate_passed_development_gate": passed,
        "development_seed_wins": seed_wins,
        "development_mean_sharpe_delta": float(development_paired["sharpe_delta"].mean()),
        "development_worst_seed_sharpe_delta": float(development_paired["sharpe_delta"].min()),
        "observed_mean_sharpe_delta": float(observed_paired["sharpe_delta"].mean()),
        "observed_seed_wins": int((observed_paired["sharpe_delta"] > 0).sum()),
        "promotion_allowed": False,
        "observed_period_is_contaminated": True,
    }
    frame.to_csv(comparison / "run_results.csv", index=False)
    summary.to_csv(comparison / "variant_summary.csv", index=False)
    paired.to_csv(comparison / "paired_results.csv", index=False)
    _write_json(comparison / "verdict.json", verdict)
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/temporal_transport_study.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("build", "compare"), default="build")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config
    result = (
        build(config_path, args.run_id, args.python)
        if args.stage == "build"
        else compare(config_path, args.run_id)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
