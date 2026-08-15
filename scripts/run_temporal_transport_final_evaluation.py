"""Run the frozen, accounting-corrected temporal-transport closing evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.market_memory_backtester import PolicyConfig, compute_backtest_metrics
from src.backtest.market_memory_evaluator import (
    model_head_score_frame,
    momentum_score_frame,
    random_score_frame,
)
from src.eval.closing_evaluation import (
    EquityAuditLimits,
    aggregate_equal_weight_market_returns,
    stationary_bootstrap_comparison,
    validate_equity_curve,
)
from src.eval.policy_baselines import (
    equal_weight_buy_hold,
    neutral_baseline_policy,
    run_scored_policy,
)

def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2, allow_nan=False), encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _source_signal(config: dict[str, Any], variant: str, seed: int, period: str) -> Path:
    source = _resolve(config["study"]["source_run"])
    return source / "memory" / f"{variant}_seed_{seed}" / period / "eval" / "signals.parquet"


def _source_metrics(config: dict[str, Any], variant: str, seed: int, period: str) -> Path:
    return _source_signal(config, variant, seed, period).with_name("metrics.json")


def _output(config: dict[str, Any], run_id: str) -> Path:
    return _resolve(config["study"]["output_root"]) / run_id


def _evaluation_root(
    output: Path, variant: str, seed: int, period: str
) -> Path:
    return output / "evaluations" / f"{variant}_seed_{seed}" / period


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(config_path: Path, run_id: str) -> dict[str, Any]:
    """Validate the frozen source cohort and write its immutable inventory."""

    config = _load_config(config_path)
    output = _output(config, run_id)
    design = config["design"]
    required = {
        "market",
        "ticker",
        "timestamp",
        "open",
        "close",
        "opportunity_score",
        "retrieval_expected_upside",
        "retrieval_expected_downside",
        "retrieval_confidence",
        "pred_future_max_return_63",
        "pred_future_min_return_63",
    }
    inventory: list[dict[str, Any]] = []
    for variant in design["variants"]:
        for seed_value in design["seeds"]:
            seed = int(seed_value)
            for period in design["periods"]:
                signals_path = _source_signal(config, variant, seed, period)
                metrics_path = _source_metrics(config, variant, seed, period)
                if not signals_path.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"Missing frozen source artifacts for {variant} seed {seed} {period}.")
                frame = pd.read_parquet(signals_path, columns=list(required))
                missing = required.difference(frame.columns)
                if missing:
                    raise ValueError(f"{signals_path} is missing columns: {sorted(missing)}")
                present_markets = sorted(frame["market"].dropna().astype(str).unique())
                expected_markets = sorted(str(value) for value in design["markets"])
                if present_markets != expected_markets:
                    raise ValueError(
                        f"Market cohort mismatch for {variant} seed {seed} {period}: {present_markets}"
                    )
                inventory.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "period": period,
                        "signals_path": _rel(signals_path),
                        "signals_rows": int(len(frame)),
                        "signals_sha256": _sha256(signals_path),
                        "metrics_path": _rel(metrics_path),
                        "metrics_sha256": _sha256(metrics_path),
                    }
                )
    table = pd.DataFrame(inventory)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "source_inventory.csv", index=False)
    result = {
        "status": "ready",
        "source_run": str(config["study"]["source_run"]),
        "source_artifact_sets": int(len(table)),
        "models_frozen": True,
        "gpu_jobs": 0,
        "mixed_currency_capital_pooling": False,
    }
    _write_json(output / "preflight.json", result)
    return result


def _save_result(root: Path, result: dict[str, Any], write_trades: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    result["equity"].to_csv(root / "equity_curve.csv", index=False)
    if write_trades:
        result["trades"].to_csv(root / "trades.csv", index=False)
    _write_json(root / "metrics.json", result["metrics"])


def _run_strategies(
    signals: pd.DataFrame,
    policy: PolicyConfig,
    momentum_window: int,
) -> dict[str, dict[str, Any]]:
    results = {
        "retrieval": run_scored_policy(signals, policy, "opportunity_score"),
        "equal_weight_buy_hold": equal_weight_buy_hold(
            signals, policy.initial_capital, policy.slippage_bps
        ),
    }
    neutral = neutral_baseline_policy(policy)
    model_frame = model_head_score_frame(signals)
    if model_frame is None:
        raise ValueError("Frozen signals do not contain the direct model-head outputs.")
    results["model_head"] = run_scored_policy(model_frame, neutral, "model_head_score")
    momentum_frame = momentum_score_frame(signals, window=momentum_window)
    if momentum_frame is None:
        raise ValueError("Frozen signals do not contain prices for the momentum baseline.")
    results[f"momentum_{momentum_window}"] = run_scored_policy(
        momentum_frame, neutral, "momentum_score"
    )
    return results


def _pooled_result(
    market_results: dict[str, dict[str, Any]], initial_capital: float
) -> dict[str, Any]:
    curves = {market: result["equity"] for market, result in market_results.items()}
    equity = aggregate_equal_weight_market_returns(curves, initial_capital)
    metrics = compute_backtest_metrics(pd.DataFrame(), equity, initial_capital)
    metrics["trade_count"] = int(
        sum(int(result["metrics"].get("trade_count", 0)) for result in market_results.values())
    )
    metrics["market_count"] = int(len(market_results))
    return {"equity": equity, "metrics": metrics}


def evaluate(
    config_path: Path,
    run_id: str,
    variant: str,
    seed: int,
    period: str,
) -> dict[str, Any]:
    """Evaluate one frozen variant/seed/period with independent market books."""

    config = _load_config(config_path)
    design = config["design"]
    if variant not in design["variants"] or seed not in [int(v) for v in design["seeds"]]:
        raise ValueError("Requested variant or seed is outside the predeclared cohort.")
    if period not in design["periods"]:
        raise ValueError("Requested period is outside the predeclared cohort.")

    settings = config["evaluation"]
    output = _output(config, run_id)
    destination = _evaluation_root(output, variant, seed, period)
    signals = pd.read_parquet(_source_signal(config, variant, seed, period))
    signals["market"] = signals["market"].astype(str)
    policy = PolicyConfig(**config["policy"])
    limits = EquityAuditLimits(minimum_daily_return=float(settings["minimum_daily_return"]))
    primary_cost = float(settings["primary_cost_bps"])
    write_trades = bool(settings.get("write_trade_ledgers", True))
    market_rows: list[dict[str, Any]] = []
    pooled_rows: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}

    for cost_value in settings["cost_sensitivity_bps"]:
        cost = float(cost_value)
        cost_policy = replace(policy, slippage_bps=cost)
        strategy_markets: dict[str, dict[str, dict[str, Any]]] = {}
        for market in design["markets"]:
            market_frame = signals.loc[signals["market"] == str(market)].copy()
            results = _run_strategies(
                market_frame, cost_policy, int(settings["momentum_window"])
            )
            for strategy, result in results.items():
                audit_key = f"cost_{cost:g}/{market}/{strategy}"
                audits[audit_key] = validate_equity_curve(
                    result["equity"], cost_policy.initial_capital, limits
                )
                strategy_markets.setdefault(strategy, {})[str(market)] = result
                market_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "period": period,
                        "cost_bps": cost,
                        "market": market,
                        "strategy": strategy,
                        **result["metrics"],
                    }
                )
                if cost == primary_cost:
                    _save_result(
                        destination / "markets" / str(market) / strategy,
                        result,
                        write_trades,
                    )
        for strategy, market_results in strategy_markets.items():
            pooled = _pooled_result(market_results, cost_policy.initial_capital)
            audits[f"cost_{cost:g}/pooled/{strategy}"] = validate_equity_curve(
                pooled["equity"], cost_policy.initial_capital, limits
            )
            pooled_rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "period": period,
                    "cost_bps": cost,
                    "strategy": strategy,
                    **pooled["metrics"],
                }
            )
            pooled_dir = destination / "pooled" / f"cost_{cost:g}" / strategy
            pooled_dir.mkdir(parents=True, exist_ok=True)
            pooled["equity"].to_csv(pooled_dir / "equity_curve.csv", index=False)
            _write_json(pooled_dir / "metrics.json", pooled["metrics"])

    random_rows: list[dict[str, Any]] = []
    neutral = neutral_baseline_policy(replace(policy, slippage_bps=primary_cost))
    for trial in range(int(settings["random_trials"])):
        trial_markets: dict[str, dict[str, Any]] = {}
        trial_seed = int(settings["random_seed"]) + trial
        for market_index, market in enumerate(design["markets"]):
            market_frame = signals.loc[signals["market"] == str(market)].copy()
            random_frame = random_score_frame(
                market_frame, seed=trial_seed + market_index * 10000
            )
            trial_markets[str(market)] = run_scored_policy(
                random_frame, neutral, "random_score"
            )
        pooled = _pooled_result(trial_markets, neutral.initial_capital)
        validate_equity_curve(pooled["equity"], neutral.initial_capital, limits)
        random_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "period": period,
                "trial": trial,
                "random_seed": trial_seed,
                **pooled["metrics"],
            }
        )

    market_table = pd.DataFrame(market_rows)
    pooled_table = pd.DataFrame(pooled_rows)
    random_table = pd.DataFrame(random_rows)
    destination.mkdir(parents=True, exist_ok=True)
    market_table.to_csv(destination / "market_metrics.csv", index=False)
    pooled_table.to_csv(destination / "pooled_metrics.csv", index=False)
    random_table.to_csv(destination / "random_trials.csv", index=False)
    _write_json(
        destination / "accounting_audit.json",
        {
            "passed": True,
            "protocol": "independent_market_books_equal_weight_normalized_return_index",
            "closed_market_return": 0.0,
            "mixed_currency_capital_pooling": False,
            "audit_count": len(audits),
            "audits": audits,
        },
    )
    result = {
        "variant": variant,
        "seed": seed,
        "period": period,
        "market_count": len(design["markets"]),
        "cost_scenarios": len(settings["cost_sensitivity_bps"]),
        "random_trials": len(random_table),
        "accounting_audit_passed": True,
    }
    _write_json(destination / "evaluation_summary.json", result)
    return result


def _paired_returns(left: Path, right: Path) -> tuple[np.ndarray, np.ndarray]:
    base = pd.read_csv(left)
    candidate = pd.read_csv(right)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    candidate["timestamp"] = pd.to_datetime(candidate["timestamp"], utc=True)
    paired = base[["timestamp", "return"]].merge(
        candidate[["timestamp", "return"]], on="timestamp", suffixes=("_baseline", "_candidate")
    )
    return (
        paired["return_baseline"].to_numpy(dtype=float),
        paired["return_candidate"].to_numpy(dtype=float),
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def compare(config_path: Path, run_id: str) -> dict[str, Any]:
    """Create the final paired tables, uncertainty intervals, and frozen verdict."""

    config = _load_config(config_path)
    design = config["design"]
    settings = config["evaluation"]
    output = _output(config, run_id)
    comparison = output / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    market_parts: list[pd.DataFrame] = []
    pooled_parts: list[pd.DataFrame] = []
    random_parts: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for variant in design["variants"]:
        for seed_value in design["seeds"]:
            seed = int(seed_value)
            for period in design["periods"]:
                root = _evaluation_root(output, variant, seed, period)
                market_parts.append(pd.read_csv(root / "market_metrics.csv"))
                pooled_parts.append(pd.read_csv(root / "pooled_metrics.csv"))
                random_parts.append(pd.read_csv(root / "random_trials.csv"))
                source_metrics = json.loads(
                    _source_metrics(config, variant, seed, period).read_text(encoding="utf-8")
                )
                source_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "period": period,
                        "alpha_mae": source_metrics.get("alpha_retrieval_mae"),
                        "downside_mae": source_metrics.get("downside_retrieval_mae"),
                        "alpha_spearman": source_metrics.get("alpha_spearman_correlation"),
                        "weekly_information_coefficient": source_metrics.get("weekly_alpha_rank_ic"),
                        "top_decile_precision": source_metrics.get("top_decile_opportunity_precision"),
                        "ndcg_at_25": source_metrics.get("outcome_ndcg_at_25"),
                        "interval_coverage": source_metrics.get("prediction_interval_coverage"),
                        "neighbor_ticker_hhi": source_metrics.get("neighbor_ticker_hhi"),
                    }
                )
    market = pd.concat(market_parts, ignore_index=True)
    pooled = pd.concat(pooled_parts, ignore_index=True)
    random_trials = pd.concat(random_parts, ignore_index=True)
    source_quality = pd.DataFrame(source_rows)
    primary_cost = float(settings["primary_cost_bps"])
    retrieval = pooled.loc[
        (pooled["strategy"] == "retrieval") & (pooled["cost_bps"] == primary_cost)
    ]
    baseline_name = str(design["baseline"])
    candidate_name = str(design["candidate"])
    paired = retrieval.loc[retrieval["variant"] == baseline_name].merge(
        retrieval.loc[retrieval["variant"] == candidate_name],
        on=["seed", "period", "cost_bps", "strategy"],
        suffixes=("_baseline", "_candidate"),
    )
    for metric in ("total_return", "annualized_return", "sharpe", "max_drawdown"):
        paired[f"{metric}_delta"] = paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"]

    market_primary = market.loc[
        (market["strategy"] == "retrieval") & (market["cost_bps"] == primary_cost)
    ]
    paired_market = market_primary.loc[market_primary["variant"] == baseline_name].merge(
        market_primary.loc[market_primary["variant"] == candidate_name],
        on=["seed", "period", "cost_bps", "market", "strategy"],
        suffixes=("_baseline", "_candidate"),
    )
    paired_market["sharpe_delta"] = (
        paired_market["sharpe_candidate"] - paired_market["sharpe_baseline"]
    )

    bootstrap_rows: list[dict[str, Any]] = []
    for row in paired.itertuples(index=False):
        base_path = (
            _evaluation_root(output, baseline_name, int(row.seed), str(row.period))
            / "pooled"
            / f"cost_{primary_cost:g}"
            / "retrieval"
            / "equity_curve.csv"
        )
        candidate_path = (
            _evaluation_root(output, candidate_name, int(row.seed), str(row.period))
            / "pooled"
            / f"cost_{primary_cost:g}"
            / "retrieval"
            / "equity_curve.csv"
        )
        base_returns, candidate_returns = _paired_returns(base_path, candidate_path)
        bootstrap_rows.append(
            {
                "seed": int(row.seed),
                "period": str(row.period),
                **stationary_bootstrap_comparison(
                    base_returns,
                    candidate_returns,
                    samples=int(settings["bootstrap_samples"]),
                    mean_block_length=int(settings["bootstrap_mean_block_sessions"]),
                    seed=int(settings["bootstrap_seed"]) + int(row.seed),
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    development = paired.loc[paired["period"] == design["development_period"]]
    development_market = paired_market.loc[
        paired_market["period"] == design["development_period"]
    ]
    candidate_development = retrieval.loc[
        (retrieval["variant"] == candidate_name)
        & (retrieval["period"] == design["development_period"])
    ]
    gates = config["development_gates"]
    seed_wins = int((development["sharpe_delta"] > 0.0).sum())
    market_win_fraction = float((development_market["sharpe_delta"] > 0.0).mean())
    maximum_drawdown = abs(float(candidate_development["max_drawdown"].min()))
    development_pass = bool(
        seed_wins >= int(gates["minimum_seed_wins"])
        and float(development["sharpe_delta"].mean())
        > float(gates["minimum_mean_pooled_sharpe_delta"])
        and market_win_fraction >= float(gates["minimum_market_seed_win_fraction"])
        and maximum_drawdown <= float(gates["maximum_candidate_drawdown"])
    )
    observed = paired.loc[paired["period"] == design["observed_period"]]
    random_summary = random_trials.groupby(["variant", "seed", "period"], as_index=False).agg(
        random_median_sharpe=("sharpe", "median"),
        random_sharpe_p05=("sharpe", lambda values: values.quantile(0.05)),
        random_sharpe_p95=("sharpe", lambda values: values.quantile(0.95)),
    )
    strategy_summary = pooled.loc[pooled["cost_bps"] == primary_cost].groupby(
        ["period", "variant", "strategy"], as_index=False
    ).agg(
        mean_total_return=("total_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        worst_drawdown=("max_drawdown", "min"),
        mean_trade_count=("trade_count", "mean"),
    )

    verdict = {
        "status": "closing_evaluation_completed",
        "candidate": candidate_name,
        "development_gate_passed": development_pass,
        "development_seed_wins": seed_wins,
        "development_mean_pooled_sharpe_delta": float(development["sharpe_delta"].mean()),
        "development_market_seed_win_fraction": market_win_fraction,
        "development_candidate_worst_drawdown": maximum_drawdown,
        "observed_mean_pooled_sharpe_delta": float(observed["sharpe_delta"].mean()),
        "observed_seed_wins": int((observed["sharpe_delta"] > 0.0).sum()),
        "observed_period_is_contaminated": bool(design["observed_period_is_contaminated"]),
        "promotion_allowed": False,
        "models_retrained": False,
        "mixed_currency_capital_pooling": False,
        "primary_accounting_protocol": "independent_market_books_equal_weight_normalized_return_index",
    }
    market.to_csv(comparison / "market_results.csv", index=False)
    pooled.to_csv(comparison / "pooled_results.csv", index=False)
    paired.to_csv(comparison / "paired_variant_results.csv", index=False)
    paired_market.to_csv(comparison / "paired_market_results.csv", index=False)
    bootstrap.to_csv(comparison / "bootstrap_intervals.csv", index=False)
    random_trials.to_csv(comparison / "random_trials.csv", index=False)
    random_summary.to_csv(comparison / "random_summary.csv", index=False)
    strategy_summary.to_csv(comparison / "strategy_summary.csv", index=False)
    source_quality.to_csv(comparison / "source_retrieval_quality.csv", index=False)
    _write_json(comparison / "final_verdict.json", verdict)

    report_table = strategy_summary.loc[
        strategy_summary["period"] == design["development_period"],
        ["variant", "strategy", "mean_total_return", "mean_sharpe", "worst_drawdown"],
    ]
    report = "\n".join(
        [
            "# Temporal Transport Closing Evaluation",
            "",
            "## Protocol",
            "",
            "Frozen saved signals were re-evaluated without retraining. Each market uses an independent local-currency book. The pooled result is an equal-weight index of normalized market daily returns; it is not a globally funded multi-currency portfolio.",
            "",
            "## Development Results",
            "",
            _markdown_table(report_table),
            "",
            "## Decision",
            "",
            f"Development gate passed: **{development_pass}**. Promotion remains forbidden because the later period has already influenced research decisions.",
            "",
            "## Required Interpretation",
            "",
            "The observed-period table is a post-selection robustness diagnostic, not untouched confirmation. Bootstrap intervals preserve paired temporal dependence but do not erase the history of model and policy selection.",
        ]
    )
    (comparison / "reviewer_report.md").write_text(report + "\n", encoding="utf-8")
    return verdict


def _job(
    job_id: str,
    command: list[str],
    outputs: list[Path],
    *,
    stage: str,
    dependencies: list[str] | None = None,
    inputs: list[Path] | None = None,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "stage": stage,
        "command": command,
        "depends_on": dependencies or [],
        "inputs": [_rel(path) for path in (inputs or [])],
        "expected_outputs": [_rel(path) for path in outputs],
        "uses_gpu": False,
        "gpu_count": 0,
        "estimated_hours": 0.0,
    }


def build(config_path: Path, run_id: str, python: str) -> dict[str, Any]:
    """Build a resumable CPU-only manifest over the frozen source artifacts."""

    config = _load_config(config_path)
    output = _output(config, run_id)
    script = PROJECT_ROOT / "scripts" / "run_temporal_transport_final_evaluation.py"
    jobs: list[dict[str, Any]] = [
        _job(
            "preflight",
            [python, _rel(script), "--config", _rel(config_path), "--run-id", run_id, "--stage", "preflight"],
            [output / "preflight.json", output / "source_inventory.csv"],
            stage="preflight",
            inputs=[config_path],
        )
    ]
    eval_ids: list[str] = []
    for variant in config["design"]["variants"]:
        for seed_value in config["design"]["seeds"]:
            seed = int(seed_value)
            for period in config["design"]["periods"]:
                job_id = f"evaluate_{variant}_seed_{seed}_{period}"
                eval_ids.append(job_id)
                destination = _evaluation_root(output, variant, seed, period)
                jobs.append(
                    _job(
                        job_id,
                        [
                            python,
                            _rel(script),
                            "--config",
                            _rel(config_path),
                            "--run-id",
                            run_id,
                            "--stage",
                            "evaluate",
                            "--variant",
                            variant,
                            "--seed",
                            str(seed),
                            "--period",
                            period,
                        ],
                        [
                            destination / "market_metrics.csv",
                            destination / "pooled_metrics.csv",
                            destination / "random_trials.csv",
                            destination / "accounting_audit.json",
                            destination / "evaluation_summary.json",
                        ],
                        stage="evaluate",
                        dependencies=["preflight"],
                        inputs=[
                            config_path,
                            _source_signal(config, variant, seed, period),
                            _source_metrics(config, variant, seed, period),
                        ],
                    )
                )
    comparison = output / "comparison"
    jobs.append(
        _job(
            "compare",
            [python, _rel(script), "--config", _rel(config_path), "--run-id", run_id, "--stage", "compare"],
            [
                comparison / "final_verdict.json",
                comparison / "strategy_summary.csv",
                comparison / "bootstrap_intervals.csv",
                comparison / "reviewer_report.md",
            ],
            stage="compare",
            dependencies=eval_ids,
            inputs=[config_path],
        )
    )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "promotion_allowed": False,
        "frozen_models": True,
        "run": {"id": run_id, "strict_environment": True},
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    result = {
        "run_id": run_id,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "job_count": len(jobs),
        "evaluation_jobs": len(eval_ids),
        "gpu_jobs": 0,
        "transformer_training_jobs": 0,
        "promotion_allowed": False,
    }
    _write_json(output / "build_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/temporal_transport_final_evaluation.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("build", "preflight", "evaluate", "compare"), default="build")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--period")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    if args.stage == "build":
        result = build(config_path, args.run_id, args.python)
    elif args.stage == "preflight":
        result = preflight(config_path, args.run_id)
    elif args.stage == "evaluate":
        if args.variant is None or args.seed is None or args.period is None:
            parser.error("evaluate requires --variant, --seed, and --period")
        result = evaluate(config_path, args.run_id, args.variant, args.seed, args.period)
    else:
        result = compare(config_path, args.run_id)
    print(json.dumps(_native(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
