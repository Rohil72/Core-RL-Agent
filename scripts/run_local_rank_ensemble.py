"""Build and run the market-local continuous-rank reconstruction."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.consensus_policy import ConsensusSignalConfig  # noqa: E402
from src.backtest.market_memory_backtester import (  # noqa: E402
    PolicyConfig,
    compute_backtest_metrics,
)
from src.eval.local_rank_ensemble import evaluate_local_rank_market  # noqa: E402
from src.eval.policy_baselines import BaselineSuiteConfig, write_json  # noqa: E402


def _load(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _validate_protocol(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    if protocol.get("reliability_filter_used") is not False:
        raise ValueError("This reconstruction must not use the rejected reliability gate.")
    if int(protocol.get("minimum_seed_votes", -1)) != 0:
        raise ValueError("The reconstruction requires minimum_seed_votes=0.")
    if protocol.get("rank_aggregation") != "median":
        raise ValueError("The reconstruction requires median rank aggregation.")
    if protocol.get("promotion_claim_allowed") is not False:
        raise ValueError("Observed periods cannot be used for a new promotion claim.")
    if config["consensus"] != {
        "minimum_votes": 0,
        "exit_smoothing_span": 1,
        "exit_score_quantile": 0.5,
        "rank_aggregation": "median",
    }:
        raise ValueError("Consensus settings drifted from the declared reconstruction.")
    if not bool(config["memory"].get("require_outcome_availability")):
        raise ValueError("Growing memory must fail closed on outcome availability.")


def _source_paths(
    config: dict[str, Any], market: str, seed: int
) -> tuple[Path, Path, Path]:
    source = PROJECT_ROOT / config["experiment"]["source_run"]
    key = f"regional_{market}_seed_{seed}"
    return (
        source / "models" / key / "final_model.pt",
        source / "latents" / key / "train_latents.parquet",
        source / "latents" / key / "val_latents.parquet",
    )


def _output_root(config: dict[str, Any], run_id: str) -> Path:
    return PROJECT_ROOT / config["experiment"]["output_root"] / run_id


def _period_names(config: dict[str, Any]) -> list[str]:
    return list(config["periods"])


def preflight(config_path: str, run_id: str) -> dict[str, Any]:
    """Fail before compute if any frozen regional source artifact is absent."""

    config = _load(config_path)
    _validate_protocol(config)
    missing: list[str] = []
    data_root = Path(_load(config["experiment"]["source_testbed_config"])["experiment"]["data_root"])
    for market in config["experiment"]["markets"]:
        if not glob.glob(str(PROJECT_ROOT / data_root / market / "*.parquet")):
            missing.append(f"{data_root.as_posix()}/{market}/*.parquet")
        for seed in config["experiment"]["seeds"]:
            for path in _source_paths(config, market, int(seed)):
                if not path.exists():
                    missing.append(_rel(path))
    if missing:
        raise FileNotFoundError(
            "Local-rank reconstruction source artifacts are incomplete:\n"
            + "\n".join(f"- {value}" for value in missing)
        )
    payload = {
        "status": "ready",
        "source_run": config["experiment"]["source_run"],
        "markets": config["experiment"]["markets"],
        "seeds": config["experiment"]["seeds"],
        "transformer_training_required": False,
        "offline_rl_used": False,
    }
    write_json(_output_root(config, run_id) / "preflight.json", payload)
    return payload


def build_memory_bank(
    config_path: str,
    run_id: str,
    market: str,
    seed: int,
    period: str,
) -> dict[str, Any]:
    """Create an expanding memory whose labels mature before each query."""

    config = _load(config_path)
    _validate_protocol(config)
    output = _output_root(config, run_id)
    key = f"regional_{market}_seed_{seed}"
    period_order = _period_names(config)
    if period not in period_order:
        raise ValueError(f"Unknown period: {period}")
    components = [output / "adapters" / key / "train_decisions.parquet"]
    components.extend(
        output / "decisions" / key / f"{name}.parquet"
        for name in period_order[: period_order.index(period) + 1]
    )
    frames: list[pd.DataFrame] = []
    for path in components:
        frame = pd.read_parquet(path)
        if "decision_outcome_available_timestamp" not in frame:
            raise ValueError(f"{path} lacks decision_outcome_available_timestamp.")
        frame["memory_component"] = _rel(path)
        frames.append(frame)
    bank = pd.concat(frames, ignore_index=True, sort=False)
    bank["timestamp"] = pd.to_datetime(bank["timestamp"], utc=True)
    bank["ticker"] = bank["ticker"].astype(str)
    bank["outcome_available_timestamp"] = pd.to_datetime(
        bank["decision_outcome_available_timestamp"], utc=True, errors="coerce"
    )
    bank = (
        bank.sort_values(["ticker", "timestamp", "memory_component"])
        .drop_duplicates(["ticker", "timestamp"], keep="last")
        .sort_values(["timestamp", "ticker"])
        .reset_index(drop=True)
    )
    destination = output / "growing_memory" / key / f"{period}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(destination, index=False, compression="zstd")
    payload = {
        "status": "completed",
        "market": market,
        "seed": seed,
        "period": period,
        "rows": int(len(bank)),
        "mature_rows": int(bank["outcome_available_timestamp"].notna().sum()),
        "components": [_rel(path) for path in components],
        "causal_contract": "neighbor outcome_available_timestamp <= query timestamp",
    }
    write_json(destination.with_suffix(".json"), payload)
    return payload


def evaluate_market(
    config_path: str,
    run_id: str,
    market: str,
    period: str,
) -> dict[str, Any]:
    """Evaluate the three-seed local rank ensemble for one market-period."""

    config = _load(config_path)
    _validate_protocol(config)
    output = _output_root(config, run_id)
    seed_signals = {
        int(seed): pd.read_parquet(
            output
            / "memory"
            / f"regional_{market}_seed_{seed}"
            / period
            / "eval"
            / "signals.parquet"
        )
        for seed in config["experiment"]["seeds"]
    }
    testbed = _load(config["experiment"]["source_testbed_config"])
    policy_values = dict(config["policy"])
    policy_values["slippage_bps"] = float(
        testbed["markets"][market]["execution_cost_bps"]
    )
    return evaluate_local_rank_market(
        seed_signals=seed_signals,
        policy=PolicyConfig(**policy_values),
        consensus_config=ConsensusSignalConfig(**config["consensus"]),
        baseline_config=BaselineSuiteConfig(**config["baselines"]),
        output=output / "evaluation" / period / market,
    )


def _pooled_metrics(curves: list[pd.DataFrame], initial_capital: float) -> dict[str, Any]:
    returns = []
    for index, curve in enumerate(curves):
        frame = curve.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp")
        frame[f"market_{index}"] = pd.to_numeric(frame["equity"], errors="coerce").pct_change()
        returns.append(frame.set_index("timestamp")[[f"market_{index}"]])
    panel = pd.concat(returns, axis=1).sort_index()
    daily = panel.mean(axis=1, skipna=True).fillna(0.0)
    equity = initial_capital * (1.0 + daily).cumprod()
    pooled = pd.DataFrame({"timestamp": equity.index, "equity": equity.to_numpy()})
    pooled["return"] = daily.to_numpy()
    pooled["drawdown"] = pooled["equity"] / pooled["equity"].cummax() - 1.0
    pooled["exposure"] = np.nan
    return {
        "metrics": compute_backtest_metrics(pd.DataFrame(), pooled, initial_capital),
        "equity": pooled,
    }


def aggregate_period(config_path: str, run_id: str, period: str) -> dict[str, Any]:
    """Pool market results without changing the no-promotion evidence boundary."""

    config = _load(config_path)
    _validate_protocol(config)
    output = _output_root(config, run_id)
    markets = config["experiment"]["markets"]
    records: list[dict[str, Any]] = []
    for market in markets:
        payload = json.loads(
            (output / "evaluation" / period / market / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        records.append(
            {
                "market": market,
                "strategy": "local_rank_memory",
                "ensemble_sharpe_lift_vs_mean_seed": payload[
                    "ensemble_sharpe_lift_vs_mean_seed"
                ],
                **payload["ensemble"],
            }
        )
        records.extend(
            {"market": market, "strategy": name, **metrics}
            for name, metrics in payload["baselines"].items()
        )
    market_table = pd.DataFrame(records)
    destination = output / "aggregate" / period
    destination.mkdir(parents=True, exist_ok=True)
    market_table.to_csv(destination / "market_results.csv", index=False)

    summaries: list[dict[str, Any]] = []
    initial_capital = float(config["policy"]["initial_capital"])
    for strategy, group in market_table.groupby("strategy", sort=True):
        curves = [
            pd.read_csv(
                output
                / "evaluation"
                / period
                / market
                / "baselines"
                / strategy
                / "equity_curve.csv"
            )
            for market in markets
        ]
        pooled = _pooled_metrics(curves, initial_capital)
        pooled["equity"].to_csv(
            destination / f"{strategy}_pooled_equity.csv", index=False
        )
        positive = group["total_return"].clip(lower=0.0)
        concentration = (
            float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0
        )
        summaries.append(
            {
                "strategy": strategy,
                "pooled_total_return": pooled["metrics"]["total_return"],
                "pooled_sharpe": pooled["metrics"]["sharpe"],
                "pooled_max_drawdown": pooled["metrics"]["max_drawdown"],
                "median_market_sharpe": float(group["sharpe"].median()),
                "positive_markets": int((group["total_return"] > 0).sum()),
                "profit_concentration": concentration,
            }
        )
    summary_table = pd.DataFrame(summaries).sort_values(
        "pooled_sharpe", ascending=False
    )
    summary_table.to_csv(destination / "strategy_summary.csv", index=False)
    primary = summary_table.loc[
        summary_table["strategy"] == "local_rank_memory"
    ].iloc[0]
    gates = config["gates"]
    gate_results = {
        "target_pooled_sharpe": bool(
            primary["pooled_sharpe"] >= gates["target_pooled_sharpe"]
        ),
        "maximum_drawdown": bool(
            abs(primary["pooled_max_drawdown"]) <= gates["maximum_drawdown"]
        ),
        "minimum_positive_markets": bool(
            primary["positive_markets"] >= gates["minimum_positive_markets"]
        ),
        "maximum_profit_concentration": bool(
            primary["profit_concentration"] <= gates["maximum_profit_concentration"]
        ),
        "minimum_ensemble_seed_lift": bool(
            market_table.loc[
                market_table["strategy"] == "local_rank_memory",
                "ensemble_sharpe_lift_vs_mean_seed",
            ].mean()
            >= gates["minimum_ensemble_seed_lift"]
        ),
    }
    payload = {
        "status": "exploratory_pass" if all(gate_results.values()) else "exploratory_rejected",
        "period": period,
        "evidence_status": config["periods"][period]["evidence_status"],
        "promotion_allowed": False,
        "failed_confirmation_immutable": True,
        "primary": primary.to_dict(),
        "gates": gate_results,
    }
    write_json(destination / "summary.json", payload)
    return payload


def _job(
    jobs: list[dict[str, Any]],
    job_id: str,
    command: list[str],
    outputs: list[Path],
    *,
    dependencies: list[str] | None = None,
    inputs: list[str | Path] | None = None,
    gpu: bool = False,
    hours: float = 0.0,
) -> str:
    jobs.append(
        {
            "id": job_id,
            "stage": "full",
            "command": command,
            "depends_on": dependencies or [],
            "inputs": [
                value if isinstance(value, str) else _rel(value)
                for value in (inputs or [])
            ],
            "expected_outputs": [_rel(path) for path in outputs],
            "uses_gpu": gpu,
            "gpu_count": 1,
            "estimated_hours": hours,
        }
    )
    return job_id


def build(config_path: str, run_id: str, python: str) -> dict[str, Any]:
    """Compile a resumable adapter-memory-policy DAG without transformer or RL jobs."""

    config = _load(config_path)
    _validate_protocol(config)
    output = _output_root(config, run_id)
    testbed = _load(config["experiment"]["source_testbed_config"])
    data_root = Path(testbed["experiment"]["data_root"])
    jobs: list[dict[str, Any]] = []
    preflight_job = _job(
        jobs,
        "preflight",
        [
            python,
            "scripts/run_local_rank_ensemble.py",
            "--config",
            config_path,
            "--run-id",
            run_id,
            "--stage",
            "preflight",
        ],
        [output / "preflight.json"],
        inputs=[config_path, config["experiment"]["source_testbed_config"]],
    )

    retrieval_jobs: dict[tuple[str, int, str], str] = {}
    evaluation_jobs: dict[tuple[str, str], str] = {}
    for market in config["experiment"]["markets"]:
        target_glob = f"{data_root.as_posix()}/{market}/*.parquet"
        for seed_value in config["experiment"]["seeds"]:
            seed = int(seed_value)
            key = f"regional_{market}_seed_{seed}"
            checkpoint, train_latents, val_latents = _source_paths(config, market, seed)
            adapter = output / "adapters" / key
            adapter_job = _job(
                jobs,
                f"adapter_{key}",
                [
                    python,
                    "scripts/train_phase5_adapter_from_latents.py",
                    "--train-latents",
                    _rel(train_latents),
                    "--val-latents",
                    _rel(val_latents),
                    "--precomputed-glob",
                    target_glob,
                    "--output",
                    _rel(adapter),
                    "--seed",
                    str(seed),
                    "--config",
                    config_path,
                ],
                [adapter / "decision_adapter.pt", adapter / "train_decisions.parquet"],
                dependencies=[preflight_job],
                inputs=[train_latents, val_latents, target_glob, config_path],
                gpu=True,
                hours=float(config["hardware"]["adapter_gpu_hours"]),
            )
            exports: dict[str, str] = {}
            for period, period_config in config["periods"].items():
                decisions = output / "decisions" / key / f"{period}.parquet"
                export_job = _job(
                    jobs,
                    f"export_{key}_{period}",
                    [
                        python,
                        "scripts/export_phase5_transfer_latents.py",
                        "--encoder-checkpoint",
                        _rel(checkpoint),
                        "--adapter-checkpoint",
                        _rel(adapter / "decision_adapter.pt"),
                        "--target-glob",
                        target_glob,
                        "--start",
                        str(period_config["start"]),
                        "--end",
                        str(period_config["end"]),
                        "--output",
                        _rel(decisions),
                        "--config",
                        config_path,
                    ],
                    [decisions, decisions.with_suffix(".json")],
                    dependencies=[adapter_job],
                    inputs=[checkpoint, adapter / "decision_adapter.pt", target_glob],
                    gpu=True,
                    hours=float(config["hardware"]["export_gpu_hours"]),
                )
                exports[period] = export_job
                bank = output / "growing_memory" / key / f"{period}.parquet"
                prior_exports = [
                    exports[name]
                    for name in _period_names(config)
                    if name in exports
                ]
                bank_job = _job(
                    jobs,
                    f"bank_{key}_{period}",
                    [
                        python,
                        "scripts/run_local_rank_ensemble.py",
                        "--config",
                        config_path,
                        "--run-id",
                        run_id,
                        "--stage",
                        "bank",
                        "--market",
                        market,
                        "--seed",
                        str(seed),
                        "--period",
                        period,
                    ],
                    [bank, bank.with_suffix(".json")],
                    dependencies=[adapter_job, *prior_exports],
                    inputs=[adapter / "train_decisions.parquet"],
                )
                memory_dir = output / "memory" / key / period
                memory_config = (
                    output / "generated_configs" / "memory" / key / f"{period}.yaml"
                )
                policy_values = dict(config["policy"])
                policy_values["slippage_bps"] = float(
                    testbed["markets"][market]["execution_cost_bps"]
                )
                _write_yaml(
                    memory_config,
                    {
                        "data": {
                            "train_latents": _rel(bank),
                            "test_latents": _rel(decisions),
                            "precomputed_glob": target_glob,
                            "output_dir": _rel(memory_dir),
                        },
                        "memory": config["memory"],
                        "policy": policy_values,
                        "evaluation": {
                            "baselines": "",
                            "memory_metric_target": "decision_net_alpha",
                        },
                        "run": {
                            "write_neighbors": True,
                            "write_memory_reports": False,
                        },
                    },
                )
                retrieval_jobs[(market, seed, period)] = _job(
                    jobs,
                    f"memory_{key}_{period}",
                    [
                        python,
                        "scripts/run_market_memory_backtest.py",
                        "--config",
                        _rel(memory_config),
                        "--run-id",
                        "eval",
                    ],
                    [
                        memory_dir / "eval" / "metrics.json",
                        memory_dir / "eval" / "signals.parquet",
                        memory_dir / "eval" / "neighbors.parquet",
                    ],
                    dependencies=[bank_job, export_job],
                    inputs=[memory_config, bank, decisions, target_glob],
                )

        for period in config["periods"]:
            dependencies = [
                retrieval_jobs[(market, int(seed), period)]
                for seed in config["experiment"]["seeds"]
            ]
            evaluation = output / "evaluation" / period / market
            evaluation_jobs[(market, period)] = _job(
                jobs,
                f"evaluate_{market}_{period}",
                [
                    python,
                    "scripts/run_local_rank_ensemble.py",
                    "--config",
                    config_path,
                    "--run-id",
                    run_id,
                    "--stage",
                    "evaluate",
                    "--market",
                    market,
                    "--period",
                    period,
                ],
                [evaluation / "metrics.json", evaluation / "consensus_signals.parquet"],
                dependencies=dependencies,
            )

    for period in config["periods"]:
        destination = output / "aggregate" / period
        _job(
            jobs,
            f"aggregate_{period}",
            [
                python,
                "scripts/run_local_rank_ensemble.py",
                "--config",
                config_path,
                "--run-id",
                run_id,
                "--stage",
                "aggregate",
                "--period",
                period,
            ],
            [destination / "summary.json", destination / "strategy_summary.csv"],
            dependencies=[
                evaluation_jobs[(market, period)]
                for market in config["experiment"]["markets"]
            ],
        )

    manifest = {
        "run": {
            "id": run_id,
            "strict_environment": True,
            "stale_lock_seconds": 900,
            "inputs": [config_path, config["experiment"]["source_testbed_config"]],
            "source_patterns": [
                "src/**/*.py",
                "scripts/**/*.py",
                "configs/**/*.yaml",
                "requirements*.txt",
            ],
            "hardware": {
                **testbed["hardware"],
                "minimum_free_storage_gb": testbed["storage"]["minimum_free_gb"],
            },
            "budgets": {
                "full": {"max_gpu_hours": float(config["hardware"]["max_gpu_hours"])}
            },
        },
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    summary = {
        "run_id": run_id,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "job_count": len(jobs),
        "gpu_job_count": sum(bool(job["uses_gpu"]) for job in jobs),
        "transformer_training_jobs": 0,
        "offline_rl_jobs": 0,
        "promotion_allowed": False,
    }
    write_json(output / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_rank_ensemble.yaml")
    parser.add_argument("--run-id", default="local_rank_ensemble_v1")
    parser.add_argument(
        "--stage",
        choices=("build", "preflight", "bank", "evaluate", "aggregate"),
        default="build",
    )
    parser.add_argument("--market")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--period")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.stage == "build":
        result = build(args.config, args.run_id, args.python)
    elif args.stage == "preflight":
        result = preflight(args.config, args.run_id)
    elif args.stage == "bank":
        if args.market is None or args.seed is None or args.period is None:
            parser.error("--stage bank requires --market, --seed, and --period.")
        result = build_memory_bank(
            args.config, args.run_id, args.market, args.seed, args.period
        )
    elif args.stage == "evaluate":
        if args.market is None or args.period is None:
            parser.error("--stage evaluate requires --market and --period.")
        result = evaluate_market(args.config, args.run_id, args.market, args.period)
    else:
        if args.period is None:
            parser.error("--stage aggregate requires --period.")
        result = aggregate_period(args.config, args.run_id, args.period)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
