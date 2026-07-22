"""Compile the frozen international encoder/memory/offline-RL research DAG."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _job(
    jobs: list[dict[str, Any]],
    job_id: str,
    command: list[str],
    outputs: list[Path],
    *,
    stage: str,
    dependencies: list[str] | None = None,
    inputs: list[str | Path] | None = None,
    gpu: bool = False,
    hours: float = 0.0,
) -> str:
    jobs.append(
        {
            "id": job_id,
            "stage": stage,
            "command": command,
            "depends_on": dependencies or [],
            "inputs": [str(value) if isinstance(value, str) else _rel(value) for value in (inputs or [])],
            "expected_outputs": [_rel(path) for path in outputs],
            "uses_gpu": gpu,
            "gpu_count": 1,
            "estimated_hours": hours,
        }
    )
    return job_id


def _encoder_config(
    base: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    seed: int,
    markets: list[str],
    name: str,
) -> tuple[Path, Path, Path]:
    resolved = copy.deepcopy(base)
    data_root = Path(config["experiment"]["data_root"])
    if len(markets) == 1:
        market = markets[0]
        resolved["data"]["precomputed_dir"] = f"{data_root.as_posix()}/{market}/*.parquet"
        resolved["data"]["market"] = market
        resolved["data"].pop("precomputed_sources", None)
        resolved["data"]["market_balanced_sampling"] = False
    else:
        resolved["data"].pop("precomputed_dir", None)
        resolved["data"]["precomputed_sources"] = [
            {"market": market, "glob": f"{data_root.as_posix()}/{market}/*.parquet"}
            for market in markets
        ]
        resolved["data"]["market_balanced_sampling"] = True
    resolved["data"]["exclude_years"] = []
    resolved["data"]["use_detector_targets"] = False
    resolved["model"]["memory_mode"] = "static_parameter"
    resolved["model"]["memory_update_rate"] = 0.0
    resolved["training"].update(
        {
            "seed": seed,
            "device": "cuda",
            "model_dir": _rel(output / "models" / name),
            "auto_resume": True,
            "checkpoint_interval_steps": int(config["hardware"]["checkpoint_interval_steps"]),
            "allow_hardware_mismatch_resume": False,
            "use_amp": bool(config["hardware"]["amp"]),
        }
    )
    resolved["hardware"] = {"vram_fraction": float(config["hardware"]["vram_fraction"])}
    resolved["split"].update(
        {
            "train_years": 8,
            "val_years": 1,
            "test_years": 1,
            "step_years": 1,
            "selected_fold": 0,
            "ticker_holdout_fraction": 0.0,
            "seed": seed,
        }
    )
    resolved["evaluation"]["reports_dir"] = _rel(output / "encoder_reports" / name)
    latent_dir = output / "latents" / name
    resolved.setdefault("latent_export", {})["output_dir"] = _rel(latent_dir)
    config_path = output / "generated_configs" / "encoders" / f"{name}.yaml"
    _write_yaml(config_path, resolved)
    return config_path, output / "models" / name / "final_model.pt", latent_dir


def _memory_config(
    base: dict[str, Any],
    market: dict[str, Any],
    train_decisions: Path,
    query_decisions: Path,
    query_glob: str,
    output_dir: Path,
) -> dict[str, Any]:
    memory = dict(base["memory_defaults"])
    memory.update(
        {
            "target_upside": "decision_mfe",
            "target_alpha": "decision_net_alpha",
            "target_downside": "decision_mae",
            "target_path_quality": "decision_path_quality",
            "target_holding_period": "decision_holding_sessions",
            "score_mode": "alpha_lcb",
            "require_outcome_availability": False,
            "same_ticker_mode": "exclude",
        }
    )
    policy = dict(base["policy"])
    policy["slippage_bps"] = float(market["execution_cost_bps"])
    return {
        "data": {
            "train_latents": _rel(train_decisions),
            "test_latents": _rel(query_decisions),
            "precomputed_glob": query_glob,
            "output_dir": _rel(output_dir),
        },
        "memory": memory,
        "policy": policy,
        "evaluation": {
            "baselines": "model_head,momentum,random",
            "memory_metric_target": "decision_net_alpha",
        },
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }


def build(
    config_path: Path,
    base_encoder_path: Path,
    run_id: str | None,
    core_python: str,
    rl_python: str,
    hourly_cost_inr: float | None = None,
    pilot_max_cost_inr: float | None = None,
    full_max_cost_inr: float | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if hourly_cost_inr is not None:
        for stage in ("pilot", "full", "confirmation"):
            config["budgets"][stage]["hourly_cost_inr"] = float(hourly_cost_inr)
    if pilot_max_cost_inr is not None:
        config["budgets"]["pilot"]["max_cost_inr"] = float(pilot_max_cost_inr)
    if full_max_cost_inr is not None:
        config["budgets"]["full"]["max_cost_inr"] = float(full_max_cost_inr)
    base_encoder = yaml.safe_load(base_encoder_path.read_text(encoding="utf-8"))
    phase4e = yaml.safe_load((PROJECT_ROOT / "configs/phase4e_cross_market_sharpe.yaml").read_text(encoding="utf-8"))
    run_name = run_id or str(config["experiment"]["run_id"])
    output = PROJECT_ROOT / config["experiment"]["output_root"] / run_name
    output.mkdir(parents=True, exist_ok=True)
    markets = list(config["markets"])
    seeds = [int(value) for value in config["research_matrix"]["seeds"]]
    pilot_seed = int(config["research_matrix"]["pilot_seed"])
    pilot_markets = list(config["research_matrix"]["pilot_markets"])
    algorithms = list(config["research_matrix"]["policy_algorithms"])
    data_root = Path(config["experiment"]["data_root"])
    jobs: list[dict[str, Any]] = []

    offline_cfg = output / "generated_configs" / "offline_policy_dataset.yaml"
    _write_yaml(offline_cfg, config["policy_training"]["offline_dataset"])
    universe_paths: dict[str, Path] = {}
    data_jobs = []
    for market, metadata in config["markets"].items():
        universe = output / "universes" / f"{market}.yaml"
        _write_yaml(universe, {"tickers": metadata["tickers"]})
        universe_paths[market] = universe
        destination = PROJECT_ROOT / data_root / market
        data_jobs.append(
            _job(
                jobs,
                f"data_{market}",
                [core_python, "scripts/precompute_ground_truth.py", "--config-path", _rel(universe), "--output-dir", _rel(destination), "--start", str(config["periods"]["download_start"]), "--end", str(config["periods"]["download_end"])],
                [destination / "manifest.json"],
                stage="data",
                inputs=[universe],
            )
        )
    audit_job = _job(
        jobs,
        "audit_all_markets",
        [core_python, "scripts/run_phase5_international.py", "--config", _rel(config_path), "--run-id", run_name, "--stage", "audit"],
        [output / "market_audit_summary.csv"],
        stage="data",
        dependencies=data_jobs,
        inputs=[f"{data_root.as_posix()}/**/*.parquet"],
    )

    encoder_info: dict[tuple[str, str, int], dict[str, Any]] = {}
    pilot_encoder_jobs = []
    final_gate_id = "select_pilot_policies"
    for market in markets:
        for seed in seeds:
            name = f"regional_{market}_seed_{seed}"
            cfg, checkpoint, latent_dir = _encoder_config(base_encoder, config, output, seed, [market], name)
            stage = "pilot" if market in pilot_markets and seed == pilot_seed else "full"
            dependencies = [audit_job] if stage == "pilot" else [audit_job, final_gate_id]
            job_id = _job(
                jobs,
                f"encoder_{name}",
                [core_python, "-m", "src.trainers.train_cycle_model", "--config-path", _rel(cfg), "--resume"],
                [checkpoint, checkpoint.parent / "training_complete.json", latent_dir / "train_latents.parquet", latent_dir / "val_latents.parquet"],
                stage=stage,
                dependencies=dependencies,
                inputs=[cfg, f"{data_root.as_posix()}/{market}/*.parquet"],
                gpu=True,
                hours=8.0,
            )
            if stage == "pilot":
                pilot_encoder_jobs.append(job_id)
            encoder_info[("regional", market, seed)] = {"job": job_id, "checkpoint": checkpoint, "latents": latent_dir, "stage": stage, "sources": [market]}
    for seed in seeds:
        name = f"global_seed_{seed}"
        cfg, checkpoint, latent_dir = _encoder_config(base_encoder, config, output, seed, markets, name)
        job_id = _job(
            jobs,
            f"encoder_{name}",
            [core_python, "-m", "src.trainers.train_cycle_model", "--config-path", _rel(cfg), "--resume"],
            [checkpoint, checkpoint.parent / "training_complete.json", latent_dir / "train_latents.parquet", latent_dir / "val_latents.parquet"],
            stage="full",
            dependencies=[audit_job, final_gate_id],
            inputs=[cfg, f"{data_root.as_posix()}/**/*.parquet"],
            gpu=True,
            hours=20.0,
        )
        for market in markets:
            encoder_info[("global", market, seed)] = {"job": job_id, "checkpoint": checkpoint, "latents": latent_dir, "stage": "full", "sources": markets}

    pilot_matrix: dict[str, dict[str, dict[str, str]]] = {algorithm: {} for algorithm in algorithms}
    final_matrix: dict[str, list[dict[str, Any]]] = {
        f"{algorithm}__{representation}": []
        for algorithm in algorithms
        for representation in ("regional", "global")
    }
    adapter_cache: dict[str, tuple[Path, str]] = {}

    def add_pipeline(representation: str, market: str, seed: int, info: dict[str, Any]) -> None:
        key = f"{representation}_{market}_seed_{seed}"
        stage = info["stage"]
        adapter_key = f"global_seed_{seed}" if representation == "global" else key
        cached_adapter = adapter_cache.get(adapter_key)
        if cached_adapter is None:
            adapter_dir = output / "adapters" / adapter_key
            source_globs = [f"{data_root.as_posix()}/{source}/*.parquet" for source in info["sources"]]
            adapter_job = _job(
                jobs,
                f"adapter_{adapter_key}",
                [core_python, "scripts/train_phase5_adapter_from_latents.py", "--train-latents", _rel(info["latents"] / "train_latents.parquet"), "--val-latents", _rel(info["latents"] / "val_latents.parquet"), "--precomputed-glob", *source_globs, "--output", _rel(adapter_dir), "--seed", str(seed)],
                [adapter_dir / "decision_adapter.pt", adapter_dir / "train_decisions.parquet", adapter_dir / "summary.json"],
                stage=stage,
                dependencies=[info["job"]],
                inputs=[info["latents"] / "train_latents.parquet", info["latents"] / "val_latents.parquet", *source_globs],
                gpu=True,
                hours=0.5,
            )
            adapter_cache[adapter_key] = (adapter_dir, adapter_job)
        else:
            adapter_dir, adapter_job = cached_adapter
        period_outputs: dict[str, dict[str, Path | str]] = {}
        target_glob = f"{data_root.as_posix()}/{market}/*.parquet"
        for period_name in ("policy_development", "policy_selection"):
            start, end = config["periods"][period_name]
            decisions = output / "decisions" / key / f"{period_name}.parquet"
            export_job = _job(
                jobs,
                f"export_{key}_{period_name}",
                [core_python, "scripts/export_phase5_transfer_latents.py", "--encoder-checkpoint", _rel(info["checkpoint"]), "--adapter-checkpoint", _rel(adapter_dir / "decision_adapter.pt"), "--target-glob", target_glob, "--start", str(start), "--end", str(end), "--output", _rel(decisions)],
                [decisions, decisions.with_suffix(".json")],
                stage=stage,
                dependencies=[adapter_job],
                inputs=[info["checkpoint"], adapter_dir / "decision_adapter.pt", target_glob],
                gpu=True,
                hours=0.3,
            )
            memory_dir = output / "memory" / key / period_name
            memory_cfg = output / "generated_configs" / "memory" / key / f"{period_name}.yaml"
            _write_yaml(memory_cfg, _memory_config(phase4e, config["markets"][market], adapter_dir / "train_decisions.parquet", decisions, target_glob, memory_dir))
            memory_job = _job(
                jobs,
                f"memory_{key}_{period_name}",
                [core_python, "scripts/run_market_memory_backtest.py", "--config", _rel(memory_cfg), "--run-id", "eval"],
                [memory_dir / "eval" / "metrics.json", memory_dir / "eval" / "signals.parquet", memory_dir / "eval" / "equity_curve.csv"],
                stage=stage,
                dependencies=[export_job],
                inputs=[memory_cfg, adapter_dir / "train_decisions.parquet", decisions, target_glob],
                hours=0.3,
            )
            period_outputs[period_name] = {"decisions": decisions, "memory_dir": memory_dir, "memory_job": memory_job}
        dataset = output / "policy_datasets" / key / "development.npz"
        dataset_job = _job(
            jobs,
            f"dataset_{key}",
            [core_python, "scripts/export_offline_policy_dataset.py", "--signals", _rel(period_outputs["policy_development"]["memory_dir"] / "eval" / "signals.parquet"), "--output", _rel(dataset), "--config", _rel(offline_cfg)],
            [dataset, dataset.with_suffix(".json")],
            stage=stage,
            dependencies=[str(period_outputs["policy_development"]["memory_job"])],
            inputs=[period_outputs["policy_development"]["memory_dir"] / "eval" / "signals.parquet", offline_cfg],
        )
        is_pilot = representation == "regional" and market in pilot_markets and seed == pilot_seed
        for algorithm in algorithms:
            extension = ".pt" if algorithm == "bandit" else ".d3"
            if is_pilot:
                model = output / "pilot_policies" / market / f"{algorithm}{extension}"
                train_job = _job(
                    jobs,
                    f"pilot_train_{market}_{algorithm}",
                    [rl_python, "scripts/train_phase5_offline_policy.py", "--dataset", _rel(dataset), "--algorithm", algorithm, "--steps", str(config["policy_training"]["pilot_steps"]), "--seed", str(seed), "--device", "cuda", "--output", _rel(model)],
                    [model],
                    stage="pilot",
                    dependencies=[dataset_job],
                    inputs=[dataset, "requirements-phase5-rl.txt"],
                    gpu=True,
                    hours=0.5,
                )
                evaluation = output / "pilot_evaluations" / market / algorithm
                eval_job = _job(
                    jobs,
                    f"pilot_eval_{market}_{algorithm}",
                    [rl_python, "scripts/evaluate_phase5_offline_policy.py", "--algorithm", algorithm, "--model", _rel(model), "--dataset", _rel(dataset), "--decisions", _rel(period_outputs["policy_selection"]["decisions"]), "--signals", _rel(period_outputs["policy_selection"]["memory_dir"] / "eval" / "signals.parquet"), "--output", _rel(evaluation)],
                    [evaluation / "metrics.json", evaluation / "equity_curve.csv"],
                    stage="pilot",
                    dependencies=[train_job, str(period_outputs["policy_selection"]["memory_job"])],
                    inputs=[model, dataset, period_outputs["policy_selection"]["decisions"], period_outputs["policy_selection"]["memory_dir"] / "eval" / "signals.parquet"],
                    gpu=True,
                    hours=0.2,
                )
                pilot_matrix[algorithm][market] = {
                    "candidate": _rel(evaluation / "metrics.json"),
                    "baseline": _rel(period_outputs["policy_selection"]["memory_dir"] / "eval" / "metrics.json"),
                    "job": eval_job,
                }

            status = output / "full_policy_status" / key / f"{algorithm}.json"
            model = output / "full_policies" / key / f"{algorithm}{extension}"
            train_job = _job(
                jobs,
                f"full_train_{key}_{algorithm}",
                [rl_python, "scripts/run_if_selected.py", "--selection", _rel(output / "selection" / "pilot_policy_selection.json"), "--algorithm", algorithm, "--status-output", _rel(status), "--artifact", _rel(model), "--", rl_python, "scripts/train_phase5_offline_policy.py", "--dataset", _rel(dataset), "--algorithm", algorithm, "--steps", str(config["policy_training"]["final_steps"]), "--seed", str(seed), "--device", "cuda", "--output", _rel(model)],
                [status],
                stage="full",
                dependencies=[dataset_job, final_gate_id],
                inputs=[dataset, output / "selection" / "pilot_policy_selection.json", "requirements-phase5-rl.txt"],
                gpu=True,
                hours=1.0,
            )
            eval_status = output / "full_evaluation_status" / key / f"{algorithm}.json"
            evaluation = output / "full_evaluations" / key / algorithm
            eval_job = _job(
                jobs,
                f"full_eval_{key}_{algorithm}",
                [rl_python, "scripts/run_if_selected.py", "--selection", _rel(output / "selection" / "pilot_policy_selection.json"), "--algorithm", algorithm, "--status-output", _rel(eval_status), "--artifact", _rel(evaluation / "metrics.json"), "--", rl_python, "scripts/evaluate_phase5_offline_policy.py", "--algorithm", algorithm, "--model", _rel(model), "--dataset", _rel(dataset), "--decisions", _rel(period_outputs["policy_selection"]["decisions"]), "--signals", _rel(period_outputs["policy_selection"]["memory_dir"] / "eval" / "signals.parquet"), "--output", _rel(evaluation)],
                [eval_status],
                stage="full",
                dependencies=[train_job, str(period_outputs["policy_selection"]["memory_job"])],
                inputs=[status, dataset, period_outputs["policy_selection"]["decisions"], period_outputs["policy_selection"]["memory_dir"] / "eval" / "signals.parquet"],
                gpu=True,
                hours=0.2,
            )
            final_matrix[f"{algorithm}__{representation}"].append(
                {
                    "market": market,
                    "seed": seed,
                    "metrics": _rel(evaluation / "metrics.json"),
                    "equity": _rel(evaluation / "equity_curve.csv"),
                    "baseline": _rel(period_outputs["policy_selection"]["memory_dir"] / "eval" / "metrics.json"),
                    "job": eval_job,
                }
            )

    for market in pilot_markets:
        add_pipeline("regional", market, pilot_seed, encoder_info[("regional", market, pilot_seed)])

    pilot_matrix_path = output / "generated_configs" / "pilot_policy_matrix.json"
    clean_pilot_matrix = {
        "candidates": {
            algorithm: {
                market: {key: value for key, value in paths.items() if key != "job"}
                for market, paths in values.items()
            }
            for algorithm, values in pilot_matrix.items()
        }
    }
    _write_json(pilot_matrix_path, clean_pilot_matrix)
    pilot_eval_jobs = [paths["job"] for values in pilot_matrix.values() for paths in values.values()]
    selection_path = output / "selection" / "pilot_policy_selection.json"
    _job(
        jobs,
        final_gate_id,
        [core_python, "scripts/select_offline_policies.py", "--matrix", _rel(pilot_matrix_path), "--output", _rel(selection_path), "--top-n", str(config["research_matrix"]["pilot_policy_winners"]), "--minimum-market-wins", str(config["research_matrix"]["minimum_pilot_market_wins"])],
        [selection_path, selection_path.with_suffix(".csv")],
        stage="pilot",
        dependencies=pilot_eval_jobs,
        inputs=[pilot_matrix_path, *[paths["candidate"] for values in clean_pilot_matrix["candidates"].values() for paths in values.values()], *[paths["baseline"] for values in clean_pilot_matrix["candidates"].values() for paths in values.values()]],
    )

    for market in markets:
        for seed in seeds:
            if not (market in pilot_markets and seed == pilot_seed):
                add_pipeline("regional", market, seed, encoder_info[("regional", market, seed)])
            add_pipeline("global", market, seed, encoder_info[("global", market, seed)])

    final_matrix_path = output / "generated_configs" / "final_policy_matrix.json"
    clean_final = {
        "candidates": {
            candidate: [{key: value for key, value in record.items() if key != "job"} for record in records]
            for candidate, records in final_matrix.items()
        }
    }
    _write_json(final_matrix_path, clean_final)
    final_eval_jobs = [record["job"] for records in final_matrix.values() for record in records]
    final_selection = output / "selection" / "final_policy_selection.json"
    _job(
        jobs,
        "select_final_policy",
        [core_python, "scripts/select_final_policy.py", "--matrix", _rel(final_matrix_path), "--pilot-selection", _rel(selection_path), "--output", _rel(final_selection), "--minimum-market-wins", str(config["research_matrix"]["minimum_final_market_wins"]), "--maximum-pbo", str(config["promotion_gates"]["maximum_pbo"]), "--maximum-drawdown", str(config["promotion_gates"]["maximum_drawdown"])],
        [final_selection, final_selection.with_name("final_policy_leaderboard.csv")],
        stage="full",
        dependencies=final_eval_jobs,
        inputs=[final_matrix_path, selection_path, *[output / "full_evaluation_status" / f"{representation}_{market}_seed_{seed}" / f"{algorithm}.json" for representation in ("regional", "global") for market in markets for seed in seeds for algorithm in algorithms]],
    )

    manifest = {
        "run": {
            "id": run_name,
            "strict_environment": True,
            "stale_lock_seconds": 900,
            "inputs": [_rel(config_path), _rel(base_encoder_path), "configs/phase4e_cross_market_sharpe.yaml", "requirements.txt", "requirements-phase5-rl.txt"],
            "source_patterns": ["src/**/*.py", "scripts/**/*.py", "configs/**/*.yaml", "requirements*.txt"],
            "hardware": {
                **config["hardware"],
                "minimum_free_storage_gb": config["storage"]["minimum_free_gb"],
            },
            "budgets": config["budgets"],
        },
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    summary = {
        "run_id": run_name,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "job_count": len(jobs),
        "jobs_by_stage": {stage: sum(job["stage"] == stage for job in jobs) for stage in ("data", "pilot", "full", "confirmation")},
        "confirmation_enabled": bool(config["research_matrix"]["confirmation_enabled"]),
        "confirmation_note": "Confirmation jobs are intentionally absent until select_final_policy passes and a one-shot lock is created.",
    }
    _write_json(output / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_research_testbed.yaml")
    parser.add_argument("--base-encoder", default="configs/final_encoder_training.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--core-python", default=sys.executable)
    parser.add_argument("--rl-python", default=sys.executable)
    parser.add_argument("--hourly-cost-inr", type=float)
    parser.add_argument("--pilot-max-cost-inr", type=float)
    parser.add_argument("--full-max-cost-inr", type=float)
    args = parser.parse_args()
    result = build(
        Path(args.config),
        Path(args.base_encoder),
        args.run_id,
        args.core_python,
        args.rl_python,
        args.hourly_cost_inr,
        args.pilot_max_cost_inr,
        args.full_max_cost_inr,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
