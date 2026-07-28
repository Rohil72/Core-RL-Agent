"""Build and execute the sealed, no-RL final memory confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_phase5_transfer_latents import run as export_transfer  # noqa: E402
from scripts.run_phase6_frozen_memory_sweep import (  # noqa: E402
    _job,
    _native,
    _rel,
    _resolve,
    _write_json,
    _write_yaml,
)
from src.backtest.consensus_policy import ConsensusSignalConfig  # noqa: E402
from src.backtest.market_memory_backtester import PolicyConfig  # noqa: E402
from src.backtest.market_memory_evaluator import run_market_memory_evaluation  # noqa: E402
from src.eval.confirmation_lock import (  # noqa: E402
    create_confirmation_lock,
    mark_confirmation_executed,
)
from src.eval.locked_memory_evaluation import (  # noqa: E402
    LockedBaselineConfig,
    evaluate_locked_market,
)
from src.eval.memory_reliability import ReliabilityConfig  # noqa: E402
from src.eval.statistical_promotion import deflated_sharpe_probability  # noqa: E402


def _load(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    policy = yaml.safe_load(
        _resolve(config["experiment"]["source_policy_config"]).read_text(encoding="utf-8")
    )
    testbed = yaml.safe_load(
        _resolve(config["experiment"]["source_testbed_config"]).read_text(encoding="utf-8")
    )
    expected = f"{policy['topologies'][0]['id']}__{policy['coverage_candidates'][0]['id']}"
    if config["protocol"]["candidate"] != expected:
        raise ValueError("Confirmation candidate differs from the frozen policy candidate.")
    if config["protocol"]["architecture_changes_allowed"] or config["protocol"]["threshold_changes_allowed"]:
        raise ValueError("The sealed confirmation cannot permit architecture or threshold changes.")
    if config["protocol"]["rl_used"]:
        raise ValueError("The final memory confirmation cannot enable RL.")
    expected_baselines = {
        "equal_weight_buy_hold",
        "raw_memory",
        "momentum_21",
        "direct_adapter_q50",
        "random_median",
    }
    if set(map(str, config["baseline_suite"]["names"])) != expected_baselines:
        raise ValueError("Confirmation baseline suite differs from the sealed protocol.")
    if float(config["baseline_suite"]["nominal_coverage"]) != float(
        policy["coverage_candidates"][0]["nominal_coverage"]
    ):
        raise ValueError("Confirmation coverage differs from the frozen policy.")
    if list(config["experiment"]["markets"]) != list(policy["experiment"]["markets"]):
        raise ValueError("Confirmation markets differ from the frozen policy.")
    if list(map(int, config["experiment"]["seeds"])) != list(map(int, policy["experiment"]["seeds"])):
        raise ValueError("Confirmation seeds differ from the frozen policy.")
    return config, policy, testbed


def _key(market: str, seed: int) -> str:
    return f"global_{market}_seed_{seed}"


def _paths(config: dict[str, Any], output: Path, market: str, seed: int) -> dict[str, Path]:
    source = _resolve(config["experiment"]["source_run"])
    key = _key(market, seed)
    return {
        "checkpoint": source / "models" / f"global_seed_{seed}" / "final_model.pt",
        "adapter": source / "adapters" / f"global_seed_{seed}" / "decision_adapter.pt",
        "memory": source / "adapters" / f"global_seed_{seed}" / "train_decisions.parquet",
        "development_signals": source
        / "memory"
        / key
        / config["experiment"]["development_period"]
        / "eval"
        / "signals.parquet",
        "development_neighbors": source
        / "memory"
        / key
        / config["experiment"]["development_period"]
        / "eval"
        / "neighbors.parquet",
        "decisions": output / "decisions" / key / "confirmation.parquet",
        "memory_output": output / "memory" / key / "confirmation",
        "evaluation": output / "evaluation" / market,
    }


def _query_memory_config(
    config: dict[str, Any],
    policy_config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> dict[str, Any]:
    paths = _paths(config, output, market, seed)
    memory = {
        **policy_config["memory"],
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
    policy = copy.deepcopy(policy_config["policy"])
    policy["slippage_bps"] = float(testbed["markets"][market]["execution_cost_bps"])
    return {
        "data": {
            "train_latents": _rel(paths["memory"]),
            "test_latents": _rel(paths["decisions"]),
            "precomputed_glob": f"{testbed['experiment']['data_root']}/{market}/*.parquet",
            "output_dir": _rel(paths["memory_output"]),
        },
        "memory": memory,
        "policy": policy,
        "evaluation": {
            "baselines": "",
            "memory_metric_target": "decision_net_alpha",
        },
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }


def _lock_artifacts(
    config_path: Path,
    config: dict[str, Any],
    output: Path,
) -> list[Path]:
    artifacts = [
        config_path,
        _resolve(config["experiment"]["source_policy_config"]),
        _resolve(config["experiment"]["source_testbed_config"]),
        PROJECT_ROOT / "requirements.txt",
    ]
    locked_source_patterns = (
        "src/backtest/consensus_policy.py",
        "src/backtest/market_memory_backtester.py",
        "src/eval/confirmation_lock.py",
        "src/eval/locked_memory_evaluation.py",
        "src/eval/memory_reliability.py",
        "src/eval/statistical_promotion.py",
        "src/memory/*.py",
        "scripts/export_phase5_transfer_latents.py",
        "scripts/run_final_memory_confirmation.py",
        "scripts/run_market_memory_backtest.py",
    )
    for pattern in locked_source_patterns:
        artifacts.extend(path for path in PROJECT_ROOT.glob(pattern) if path.is_file())
    testbed = yaml.safe_load(artifacts[2].read_text(encoding="utf-8"))
    for market in config["experiment"]["markets"]:
        artifacts.extend(sorted(_resolve(testbed["experiment"]["data_root"]).joinpath(market).glob("*.parquet")))
        for seed in map(int, config["experiment"]["seeds"]):
            paths = _paths(config, output, market, seed)
            artifacts.extend(
                (
                    paths["checkpoint"],
                    paths["adapter"],
                    paths["memory"],
                    paths["development_signals"],
                    paths["development_neighbors"],
                )
            )
    return sorted(set(artifacts), key=lambda item: str(item).lower())


def _export(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> dict[str, Any]:
    paths = _paths(config, output, market, seed)
    target_glob = f"{testbed['experiment']['data_root']}/{market}/*.parquet"
    return export_transfer(
        str(paths["checkpoint"]),
        str(paths["adapter"]),
        target_glob,
        str(config["experiment"]["confirmation_start"]),
        str(config["experiment"]["confirmation_end"]),
        str(paths["decisions"]),
    )


def _retrieve(
    config: dict[str, Any],
    policy_config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> dict[str, Any]:
    resolved = _query_memory_config(config, policy_config, testbed, output, market, seed)
    config_path = output / "generated_configs" / "memory" / f"{_key(market, seed)}.yaml"
    _write_yaml(config_path, resolved)
    result = run_market_memory_evaluation(resolved, PROJECT_ROOT, run_id="eval")
    return {"market": market, "seed": seed, "output": _rel(result), "status": "completed"}


def _load_seed_evidence(
    config: dict[str, Any],
    output: Path,
    market: str,
) -> tuple[
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
    dict[int, pd.DataFrame],
]:
    development_signals: dict[int, pd.DataFrame] = {}
    development_neighbors: dict[int, pd.DataFrame] = {}
    query_signals: dict[int, pd.DataFrame] = {}
    query_neighbors: dict[int, pd.DataFrame] = {}

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if "future_blended_alpha_63" not in out and "decision_net_alpha" in out:
            out["future_blended_alpha_63"] = pd.to_numeric(
                out["decision_net_alpha"],
                errors="coerce",
            )
        if "future_universe_alpha_63" not in out and "future_blended_alpha_63" in out:
            out["future_universe_alpha_63"] = out["future_blended_alpha_63"]
        return out

    for seed in map(int, config["experiment"]["seeds"]):
        paths = _paths(config, output, market, seed)
        development_signals[seed] = prepare(pd.read_parquet(paths["development_signals"]))
        development_neighbors[seed] = pd.read_parquet(paths["development_neighbors"])
        query_root = paths["memory_output"] / "eval"
        query_signals[seed] = prepare(pd.read_parquet(query_root / "signals.parquet"))
        query_neighbors[seed] = pd.read_parquet(query_root / "neighbors.parquet")
    return development_signals, development_neighbors, query_signals, query_neighbors


def _evaluate_market(
    config: dict[str, Any],
    policy_config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
) -> dict[str, Any]:
    development_signals, development_neighbors, query_signals, query_neighbors = (
        _load_seed_evidence(config, output, market)
    )
    policy_values = dict(policy_config["policy"])
    policy_values["slippage_bps"] = float(testbed["markets"][market]["execution_cost_bps"])
    reliability_values = dict(policy_config["reliability"])
    minimum_alpha = float(reliability_values.pop("minimum_useful_alpha"))
    cost_multiplier = float(reliability_values.pop("round_trip_cost_multiplier"))
    useful_alpha = max(
        minimum_alpha,
        cost_multiplier * float(testbed["markets"][market]["execution_cost_bps"]) / 10000.0,
    )
    reliability_values["useful_alpha_after_costs"] = useful_alpha
    return evaluate_locked_market(
        development_signals=development_signals,
        development_neighbors=development_neighbors,
        query_signals=query_signals,
        query_neighbors=query_neighbors,
        policy=PolicyConfig(**policy_values),
        consensus_config=ConsensusSignalConfig(**policy_config["consensus"]),
        reliability_config=ReliabilityConfig(**reliability_values),
        useful_alpha_after_costs=useful_alpha,
        baseline_config=LockedBaselineConfig(
            **{
                key: value
                for key, value in config["baseline_suite"].items()
                if key != "names"
            }
        ),
        output=_paths(config, output, market, int(config["experiment"]["seeds"][0]))[
            "evaluation"
        ],
    )


def _equity_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return (
        frame.set_index("timestamp")["equity"]
        .astype(float)
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _aggregate(
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    strategies = ["locked_memory", *config["baseline_suite"]["names"]]
    market_rows: list[dict[str, Any]] = []
    pooled: dict[str, list[pd.Series]] = {name: [] for name in strategies}
    reliability_rows: list[dict[str, Any]] = []
    for market in config["experiment"]["markets"]:
        root = output / "evaluation" / market
        payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        reliability_rows.append({"market": market, **payload["reliability"]})
        for strategy in strategies:
            metrics = payload["baselines"][strategy]
            market_rows.append({"market": market, "strategy": strategy, **metrics})
            pooled[strategy].append(
                _equity_returns(root / "baselines" / strategy / "equity_curve.csv").rename(market)
            )
    markets = pd.DataFrame(market_rows)
    reliability = pd.DataFrame(reliability_rows)
    locked_market = markets.loc[markets["strategy"] == "locked_memory"].copy()
    summary_rows: list[dict[str, Any]] = []
    pooled_locked: pd.Series | None = None
    for strategy in strategies:
        returns = pd.concat(pooled[strategy], axis=1).mean(axis=1, skipna=True).dropna()
        if strategy == "locked_memory":
            pooled_locked = returns
        sharpe = (
            float(returns.mean() / returns.std(ddof=1) * np.sqrt(252.0))
            if len(returns) > 2 and returns.std(ddof=1) > 1e-12
            else 0.0
        )
        curve = (1.0 + returns).cumprod()
        summary_rows.append(
            {
                "strategy": strategy,
                "pooled_total_return": float(curve.iloc[-1] - 1.0) if len(curve) else 0.0,
                "pooled_sharpe": sharpe,
                "pooled_max_drawdown": float((curve / curve.cummax() - 1.0).min())
                if len(curve)
                else 0.0,
                "positive_markets": int(
                    (
                        markets.loc[markets["strategy"] == strategy, "total_return"]
                        > 0.0
                    ).sum()
                ),
            }
        )
    strategy_summary = pd.DataFrame(summary_rows)
    locked = strategy_summary.loc[strategy_summary["strategy"] == "locked_memory"].iloc[0]
    equal_market = markets.loc[
        markets["strategy"] == "equal_weight_buy_hold",
        ["market", "total_return"],
    ].rename(columns={"total_return": "equal_weight_return"})
    comparison = locked_market.merge(equal_market, on="market", validate="one_to_one")
    equal_weight_wins = int(
        (comparison["total_return"] > comparison["equal_weight_return"]).sum()
    )
    positive_profit = locked_market.set_index("market")["total_return"].clip(lower=0.0)
    concentration = (
        float(positive_profit.max() / positive_profit.sum())
        if positive_profit.sum() > 0.0
        else 1.0
    )
    dominant_market = (
        str(positive_profit.idxmax()) if positive_profit.sum() > 0.0 else None
    )
    brier_positive = int((reliability["brier_skill"] > 0.0).sum())
    assert pooled_locked is not None
    dsr = deflated_sharpe_probability(pooled_locked.to_numpy(dtype=float), 1)
    gates = config["confirmation_gates"]
    passes = bool(
        float(locked["pooled_sharpe"]) >= float(gates["target_pooled_sharpe"])
        and abs(float(locked["pooled_max_drawdown"]))
        <= float(gates["maximum_drawdown"])
        and int(locked["positive_markets"]) >= int(gates["minimum_positive_markets"])
        and equal_weight_wins >= int(gates["minimum_equal_weight_wins"])
        and brier_positive >= int(gates["minimum_positive_brier_skill_markets"])
        and concentration <= float(gates["maximum_profit_concentration"])
        and dsr >= float(gates["minimum_deflated_sharpe_probability"])
    )
    payload = {
        "status": "external_evaluation_pass" if passes else "external_evaluation_rejected",
        "candidate": config["protocol"]["candidate"],
        "development_candidate_passed": False,
        "architecture_changed": False,
        "thresholds_changed": False,
        "rl_used": False,
        "pooled_total_return": float(locked["pooled_total_return"]),
        "pooled_sharpe": float(locked["pooled_sharpe"]),
        "pooled_max_drawdown": float(locked["pooled_max_drawdown"]),
        "positive_markets": int(locked["positive_markets"]),
        "equal_weight_wins": equal_weight_wins,
        "positive_brier_skill_markets": brier_positive,
        "profit_concentration": concentration,
        "dominant_profit_market": dominant_market,
        "deflated_sharpe_probability": dsr,
        "external_gate_pass": passes,
        "promotion_allowed": False,
        "profitable_strategy_claim_allowed": False,
    }
    markets.to_csv(output / "market_baseline_results.csv", index=False)
    strategy_summary.to_csv(output / "strategy_summary.csv", index=False)
    reliability.to_csv(output / "calibration_results.csv", index=False)
    _write_json(output / "confirmation_summary.json", payload)
    lines = [
        "# Frozen Market-Memory External Evaluation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Candidate: `{payload['candidate']}`",
        f"- Pooled Sharpe: `{payload['pooled_sharpe']:.6f}`",
        f"- Pooled return: `{payload['pooled_total_return']:.6f}`",
        f"- Maximum drawdown: `{payload['pooled_max_drawdown']:.6f}`",
        f"- Positive markets: `{payload['positive_markets']}`",
        f"- Equal-weight wins: `{payload['equal_weight_wins']}`",
        f"- Positive Brier-skill markets: `{payload['positive_brier_skill_markets']}`",
        f"- Profit concentration: `{payload['profit_concentration']:.6f}`",
        f"- Dominant profit market: `{payload['dominant_profit_market']}`",
        "",
        "The candidate failed its development promotion gate before this sealed evaluation. "
        "This report is external robustness evidence, not a retroactive promotion.",
    ]
    (output / "confirmation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _close(output: Path) -> dict[str, Any]:
    results = [
        output / "confirmation_summary.json",
        output / "strategy_summary.csv",
        output / "market_baseline_results.csv",
        output / "calibration_results.csv",
        output / "confirmation_report.md",
    ]
    lock = mark_confirmation_executed(
        output / "confirmation_lock.json",
        results,
        idempotent=True,
    )
    marker = {
        "status": lock["status"],
        "executed_at": lock["executed_at"],
        "result_count": len(results),
    }
    _write_json(output / "confirmation_closed.json", marker)
    return marker


def _build(
    config_path: Path,
    config: dict[str, Any],
    policy_config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    python: str,
) -> dict[str, Any]:
    artifacts = _lock_artifacts(config_path, config, output)
    create_confirmation_lock(
        output / "confirmation_lock.json",
        artifacts,
        config["experiment"]["markets"],
        confirmation_year=2025,
    )
    script = _rel(Path(__file__))
    common = [python, script, "--config", _rel(config_path), "--run-id", output.name]
    jobs: list[dict[str, Any]] = []
    market_eval_jobs: list[str] = []
    for market in config["experiment"]["markets"]:
        retrieval_dependencies: list[str] = []
        for seed in map(int, config["experiment"]["seeds"]):
            paths = _paths(config, output, market, seed)
            export_job = f"export_{_key(market, seed)}"
            jobs.append(
                {
                    **_job(
                        export_job,
                        [*common, "--stage", "export", "--market", market, "--seed", str(seed)],
                        [paths["decisions"], paths["decisions"].with_suffix(".json")],
                        stage="confirmation",
                        inputs=[
                            paths["checkpoint"],
                            paths["adapter"],
                            f"{testbed['experiment']['data_root']}/{market}/*.parquet",
                        ],
                    ),
                    "uses_gpu": True,
                    "estimated_hours": 0.30,
                }
            )
            retrieval_job = f"retrieve_{_key(market, seed)}"
            retrieval_dependencies.append(retrieval_job)
            query = paths["memory_output"] / "eval"
            jobs.append(
                _job(
                    retrieval_job,
                    [*common, "--stage", "retrieve", "--market", market, "--seed", str(seed)],
                    [
                        query / "signals.parquet",
                        query / "neighbors.parquet",
                        query / "metrics.json",
                    ],
                    stage="confirmation",
                    dependencies=[export_job],
                    inputs=[paths["memory"], paths["decisions"]],
                )
            )
        evaluation_job = f"evaluate_{market}"
        market_eval_jobs.append(evaluation_job)
        evaluation = output / "evaluation" / market
        jobs.append(
            _job(
                evaluation_job,
                [*common, "--stage", "evaluate", "--market", market],
                [
                    evaluation / "metrics.json",
                    evaluation / "reliability_metrics.json",
                    evaluation / "baselines" / "locked_memory" / "metrics.json",
                    evaluation / "baselines" / "equal_weight_buy_hold" / "metrics.json",
                    evaluation / "baselines" / "random_summary.json",
                ],
                stage="confirmation",
                dependencies=retrieval_dependencies,
                inputs=[
                    artifact
                    for seed in map(int, config["experiment"]["seeds"])
                    for artifact in (
                        _paths(config, output, market, seed)["development_signals"],
                        _paths(config, output, market, seed)["development_neighbors"],
                        _paths(config, output, market, seed)["memory_output"]
                        / "eval"
                        / "signals.parquet",
                        _paths(config, output, market, seed)["memory_output"]
                        / "eval"
                        / "neighbors.parquet",
                    )
                ],
            )
        )
    jobs.append(
        _job(
            "aggregate_confirmation",
            [*common, "--stage", "aggregate"],
            [
                output / "confirmation_summary.json",
                output / "strategy_summary.csv",
                output / "market_baseline_results.csv",
                output / "calibration_results.csv",
                output / "confirmation_report.md",
            ],
            stage="confirmation",
            dependencies=market_eval_jobs,
        )
    )
    jobs.append(
        _job(
            "close_confirmation_lock",
            [*common, "--stage", "close"],
            [output / "confirmation_closed.json"],
            stage="confirmation",
            dependencies=["aggregate_confirmation"],
            inputs=[output / "confirmation_lock.json", output / "confirmation_summary.json"],
        )
    )
    manifest = {
        "run": {
            "id": output.name,
            "strict_environment": True,
            "stale_lock_seconds": 900,
            "inputs": [
                _rel(config_path),
                str(config["experiment"]["source_policy_config"]),
                str(config["experiment"]["source_testbed_config"]),
            ],
            "source_patterns": [
                "src/backtest/**/*.py",
                "src/eval/**/*.py",
                "src/memory/**/*.py",
                "scripts/export_phase5_transfer_latents.py",
                "scripts/run_market_memory_backtest.py",
                "scripts/run_final_memory_confirmation.py",
                _rel(config_path),
                str(config["experiment"]["source_policy_config"]),
            ],
            "hardware": {
                **config["hardware"],
                "minimum_free_storage_gb": int(config["storage"]["minimum_free_gb"]),
            },
            "budgets": config["budgets"],
        },
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    summary = {
        "run_id": output.name,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "lock": _rel(output / "confirmation_lock.json"),
        "job_count": len(jobs),
        "gpu_job_count": sum(bool(job["uses_gpu"]) for job in jobs),
        "declared_gpu_hours": sum(
            float(job["estimated_hours"]) for job in jobs if job["uses_gpu"]
        ),
        "status": "locked_not_executed",
    }
    _write_json(output / "build_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config, policy_config, testbed = _load(config_path)
    output = _resolve(config["experiment"]["output_root"]) / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        return _build(config_path, config, policy_config, testbed, output, args.python)
    if args.stage == "export":
        if not args.market or args.seed is None:
            raise ValueError("--market and --seed are required for export.")
        return _export(config, testbed, output, args.market, args.seed)
    if args.stage == "retrieve":
        if not args.market or args.seed is None:
            raise ValueError("--market and --seed are required for retrieve.")
        return _retrieve(config, policy_config, testbed, output, args.market, args.seed)
    if args.stage == "evaluate":
        if not args.market:
            raise ValueError("--market is required for evaluate.")
        return _evaluate_market(config, policy_config, testbed, output, args.market)
    if args.stage == "aggregate":
        return _aggregate(config, output)
    if args.stage == "close":
        return _close(output)
    raise ValueError(f"Unsupported stage: {args.stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_memory_confirmation.yaml")
    parser.add_argument("--run-id", default="final_memory_confirmation_v1")
    parser.add_argument(
        "--stage",
        choices=("build", "export", "retrieve", "evaluate", "aggregate", "close"),
        default="build",
    )
    parser.add_argument("--market")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(_native(run(args)), indent=2))


if __name__ == "__main__":
    main()
