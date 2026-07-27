"""Run the final frozen Phase 6 memory topology and reliability sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.consensus_policy import (  # noqa: E402
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import (  # noqa: E402
    PolicyConfig,
    compute_backtest_metrics,
    equal_weight_baseline,
    run_long_only_backtest,
)
from src.backtest.market_memory_evaluator import run_market_memory_evaluation  # noqa: E402
from src.eval.memory_reliability import (  # noqa: E402
    ReliabilityConfig,
    build_consensus_frame,
    fit_reliability_model,
    reliability_metrics,
    risk_coverage_table,
)
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2), encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    testbed_path = _resolve(config["experiment"]["source_testbed_config"])
    testbed = yaml.safe_load(testbed_path.read_text(encoding="utf-8"))
    configured_markets = list(config["experiment"]["markets"])
    missing = sorted(set(configured_markets) - set(testbed["markets"]))
    if missing:
        raise ValueError(f"Markets absent from source testbed configuration: {missing}")
    if len(config["experiment"]["seeds"]) < int(config["consensus"]["minimum_votes"]):
        raise ValueError("The sweep has fewer seeds than the consensus vote requirement.")
    return config, testbed


def _source_signal_root(
    config: dict[str, Any],
    sweep_root: Path,
    topology: str,
    market: str,
    seed: int,
    period: str,
) -> Path:
    if topology not in {"regional_regional", "global_global"}:
        return sweep_root / "memory" / f"{topology}_{market}_seed_{seed}" / period / "eval"
    source = _resolve(config["experiment"]["source_run"])
    representation = "regional" if topology == "regional_regional" else "global"
    return source / "memory" / f"{representation}_{market}_seed_{seed}" / period / "eval"


def _required_source_paths(config: dict[str, Any]) -> list[Path]:
    source = _resolve(config["experiment"]["source_run"])
    development = str(config["experiment"]["development_period"])
    selection = str(config["experiment"]["selection_period"])
    required: list[Path] = []
    for seed in map(int, config["experiment"]["seeds"]):
        required.append(source / "adapters" / f"global_seed_{seed}" / "train_decisions.parquet")
        for market in config["experiment"]["markets"]:
            for representation in ("regional", "global"):
                for period in (development, selection):
                    root = source / "memory" / f"{representation}_{market}_seed_{seed}" / period / "eval"
                    required.extend((root / "signals.parquet", root / "neighbors.parquet"))
            for period in (development, selection):
                required.append(
                    source / "decisions" / f"global_{market}_seed_{seed}" / f"{period}.parquet"
                )
    return required


def _preflight(config: dict[str, Any], testbed: dict[str, Any]) -> dict[str, Any]:
    missing = [str(path) for path in _required_source_paths(config) if not path.exists()]
    data_root = _resolve(testbed["experiment"]["data_root"])
    for market in config["experiment"]["markets"]:
        if not list((data_root / market).glob("*.parquet")):
            missing.append(str(data_root / market / "*.parquet"))
    if missing:
        preview = "\n".join(f"- {path}" for path in missing[:20])
        suffix = f"\n... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(f"Frozen sweep preflight found missing artifacts:\n{preview}{suffix}")
    return {"required_artifacts": len(_required_source_paths(config)), "markets": len(config["experiment"]["markets"])}


def _filter_seed(config: dict[str, Any], output: Path, seed: int) -> dict[str, Any]:
    source = _resolve(config["experiment"]["source_run"])
    input_path = source / "adapters" / f"global_seed_{seed}" / "train_decisions.parquet"
    table = pq.read_table(input_path)
    if "market" not in table.schema.names:
        raise ValueError(f"{input_path} has no market column; regional filtering is impossible.")
    destination = output / "filtered_memory" / f"global_seed_{seed}"
    counts: dict[str, int] = {}
    for market in config["experiment"]["markets"]:
        path = destination / f"{market}.parquet"
        if path.exists() and path.stat().st_size > 0:
            counts[market] = int(pq.ParquetFile(path).metadata.num_rows)
            continue
        market_table = table.filter(pc.equal(table["market"], market))
        if market_table.num_rows == 0:
            raise ValueError(f"Global seed {seed} contains no historical rows for {market}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(market_table, path, compression="zstd")
        counts[market] = int(market_table.num_rows)
    summary = {"seed": seed, "source": _rel(input_path), "market_rows": counts}
    _write_json(destination / "summary.json", summary)
    return summary


def _hybrid_memory_config(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
    period: str,
) -> dict[str, Any]:
    source = _resolve(config["experiment"]["source_run"])
    data_root = Path(testbed["experiment"]["data_root"])
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
    market_policy = dict(config["policy"])
    market_policy["slippage_bps"] = float(testbed["markets"][market]["execution_cost_bps"])
    key = f"global_regional_{market}_seed_{seed}"
    return {
        "data": {
            "train_latents": _rel(output / "filtered_memory" / f"global_seed_{seed}" / f"{market}.parquet"),
            "test_latents": _rel(source / "decisions" / f"global_{market}_seed_{seed}" / f"{period}.parquet"),
            "precomputed_glob": f"{data_root.as_posix()}/{market}/*.parquet",
            "output_dir": _rel(output / "memory" / key / period),
        },
        "memory": memory,
        "policy": market_policy,
        "evaluation": {"baselines": "model_head,momentum,random", "memory_metric_target": "decision_net_alpha"},
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }


def _retrieve(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
) -> dict[str, Any]:
    completed: list[str] = []
    for period in (
        str(config["experiment"]["development_period"]),
        str(config["experiment"]["selection_period"]),
    ):
        result_dir = output / "memory" / f"global_regional_{market}_seed_{seed}" / period / "eval"
        expected = (result_dir / "signals.parquet", result_dir / "neighbors.parquet", result_dir / "metrics.json")
        if all(path.exists() and path.stat().st_size > 0 for path in expected):
            completed.append(period)
            continue
        resolved = _hybrid_memory_config(config, testbed, output, market, seed, period)
        config_path = output / "generated_configs" / "memory" / f"global_regional_{market}_seed_{seed}" / f"{period}.yaml"
        _write_yaml(config_path, resolved)
        run_market_memory_evaluation(resolved, PROJECT_ROOT, run_id="eval")
        if not all(path.exists() and path.stat().st_size > 0 for path in expected):
            raise RuntimeError(f"Hybrid retrieval did not produce complete artifacts for {market} seed {seed} {period}.")
        completed.append(period)
    marker = output / "retrieval_status" / f"global_regional_{market}_seed_{seed}.json"
    payload = {"market": market, "seed": seed, "periods": completed, "status": "completed"}
    _write_json(marker, payload)
    return payload


def _prepare_reliability_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Map the Phase 6 realized decision outcome into the frozen reliability contract."""
    if "decision_net_alpha" not in frame:
        raise ValueError("Phase 6 signals are missing decision_net_alpha.")
    out = frame.copy()
    out["future_blended_alpha_63"] = pd.to_numeric(out["decision_net_alpha"], errors="coerce")
    out["future_universe_alpha_63"] = out["future_blended_alpha_63"]
    return out


def _load_evidence(
    config: dict[str, Any],
    output: Path,
    topology: str,
    market: str,
    period: str,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    signals: dict[int, pd.DataFrame] = {}
    neighbors: dict[int, pd.DataFrame] = {}
    for seed in map(int, config["experiment"]["seeds"]):
        root = _source_signal_root(config, output, topology, market, seed, period)
        signals[seed] = _prepare_reliability_source(pd.read_parquet(root / "signals.parquet"))
        neighbors[seed] = pd.read_parquet(
            root / "neighbors.parquet",
            columns=["query_ticker", "query_timestamp", "experience_id"],
        )
    return signals, neighbors


def _policy(config: dict[str, Any], testbed: dict[str, Any], market: str) -> PolicyConfig:
    values = dict(config["policy"])
    values["slippage_bps"] = float(testbed["markets"][market]["execution_cost_bps"])
    return PolicyConfig(**values)


def _reliability_config(config: dict[str, Any], testbed: dict[str, Any], market: str) -> tuple[ReliabilityConfig, float]:
    values = dict(config["reliability"])
    minimum = float(values.pop("minimum_useful_alpha"))
    multiplier = float(values.pop("round_trip_cost_multiplier"))
    cost = float(testbed["markets"][market]["execution_cost_bps"]) / 10000.0
    useful_alpha = max(minimum, multiplier * cost)
    values["useful_alpha_after_costs"] = useful_alpha
    return ReliabilityConfig(**values), useful_alpha


def _evaluate_candidate(
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    topology: str,
    market: str,
) -> dict[str, Any]:
    candidate_root = output / "evaluation" / topology / market
    marker = candidate_root / "candidate_complete.json"
    if marker.exists() and marker.stat().st_size > 0:
        return json.loads(marker.read_text(encoding="utf-8"))
    development = str(config["experiment"]["development_period"])
    selection = str(config["experiment"]["selection_period"])
    train_signals, train_neighbors = _load_evidence(config, output, topology, market, development)
    test_signals, test_neighbors = _load_evidence(config, output, topology, market, selection)
    reliability_cfg, useful_alpha = _reliability_config(config, testbed, market)
    train = build_consensus_frame(
        train_signals,
        train_neighbors,
        useful_alpha_after_costs=useful_alpha,
    )
    test = build_consensus_frame(
        test_signals,
        test_neighbors,
        useful_alpha_after_costs=useful_alpha,
    )
    model = fit_reliability_model(train, reliability_cfg)
    predictions = model.predict(test)
    candidate_root.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(candidate_root / "selection_reliability.parquet", index=False)
    _write_json(candidate_root / "reliability_model_audit.json", model.to_dict())
    reliability_audit = reliability_metrics(predictions, model)
    _write_json(candidate_root / "reliability_metrics.json", reliability_audit)
    risk_coverage_table(predictions, model).to_csv(candidate_root / "risk_coverage.csv", index=False)

    policy = _policy(config, testbed, market)
    consensus_cfg = ConsensusSignalConfig(**config["consensus"])
    consensus = build_consensus_signals(test_signals, policy, consensus_cfg)
    reliability_columns = predictions[
        ["ticker", "timestamp", "reliability_probability", "predicted_absolute_error", "state_shift_score"]
    ]
    consensus = consensus.merge(reliability_columns, on=["ticker", "timestamp"], how="left", validate="one_to_one")
    consensus.to_parquet(candidate_root / "consensus_signals.parquet", index=False)

    coverage_results: dict[str, Any] = {}
    for coverage in config["coverage_candidates"]:
        identifier = str(coverage["id"])
        nominal = float(coverage["nominal_coverage"])
        threshold = float(model.calibration_thresholds[nominal])
        signals = consensus.copy()
        signals["reliability_pass"] = signals["reliability_probability"] >= threshold
        decisions: list[dict[str, Any]] = []
        trades, equity = run_long_only_backtest(
            signals,
            policy,
            score_col="consensus_entry_rank",
            exit_score_col="consensus_exit_score",
            row_filter=lambda row, cfg, _score, votes=consensus_cfg.minimum_votes: (
                bool(row.get("reliability_pass", False)) and consensus_row_filter(row, cfg, votes)
            ),
            decision_log=decisions,
        )
        metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
        metrics.update(equal_weight_baseline(signals, policy.initial_capital))
        metrics.update(
            {
                "topology": topology,
                "market": market,
                "coverage": identifier,
                "nominal_coverage": nominal,
                "realized_coverage": float(signals["reliability_pass"].mean()),
                "reliability_threshold": threshold,
                "useful_alpha_threshold": useful_alpha,
                "mean_seed_vote_fraction": float(signals["seed_vote_fraction"].mean()),
                "mean_seed_rank_std": float(signals["seed_rank_std"].mean()),
                "mean_neighbor_overlap": float(test["cross_seed_neighbor_overlap"].mean()),
            }
        )
        result = candidate_root / identifier
        result.mkdir(parents=True, exist_ok=True)
        trades.to_csv(result / "trades.csv", index=False)
        equity.to_csv(result / "equity_curve.csv", index=False)
        pd.DataFrame(decisions).to_csv(result / "decisions.csv", index=False)
        _write_json(result / "metrics.json", metrics)
        coverage_results[identifier] = metrics
    payload = {"topology": topology, "market": market, "status": "completed", "coverages": coverage_results}
    _write_json(marker, payload)
    return payload


def _equity_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().dropna()


def _sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 1e-12:
        return 0.0
    return float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0))


def _profit_concentration(returns: pd.Series) -> float:
    positive = returns.clip(lower=0.0)
    return float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0


def _summarize(config: dict[str, Any], output: Path) -> dict[str, Any]:
    market_rows: list[dict[str, Any]] = []
    pooled_returns: dict[str, pd.Series] = {}
    for topology in [str(item["id"]) for item in config["topologies"]]:
        for coverage in config["coverage_candidates"]:
            coverage_id = str(coverage["id"])
            candidate = f"{topology}__{coverage_id}"
            series: list[pd.Series] = []
            for market in config["experiment"]["markets"]:
                root = output / "evaluation" / topology / market / coverage_id
                metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
                reliability = json.loads(
                    (output / "evaluation" / topology / market / "reliability_metrics.json").read_text(
                        encoding="utf-8"
                    )
                )
                market_rows.append(
                    {
                        "candidate": candidate,
                        **metrics,
                        **{f"reliability_{key}": value for key, value in reliability.items()},
                    }
                )
                series.append(_equity_returns(root / "equity_curve.csv").rename(market))
            pooled_returns[candidate] = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
    markets = pd.DataFrame(market_rows)
    markets.to_csv(output / "market_results.csv", index=False)
    trial_count = len(pooled_returns)
    return_matrix = pd.concat(pooled_returns, axis=1).dropna(how="all").fillna(0.0)
    pbo = probability_of_backtest_overfitting(return_matrix.to_numpy())
    rows: list[dict[str, Any]] = []
    for candidate, group in markets.groupby("candidate", sort=False):
        returns = pooled_returns[candidate]
        rows.append(
            {
                "candidate": candidate,
                "pooled_sharpe": _sharpe(returns),
                "median_market_sharpe": float(group["sharpe"].median()),
                "mean_market_sharpe": float(group["sharpe"].mean()),
                "mean_total_return": float(group["total_return"].mean()),
                "positive_markets": int((group["total_return"] > 0.0).sum()),
                "baseline_wins": int((group["total_return"] > group["equal_weight_baseline_return"]).sum()),
                "positive_brier_skill_markets": int((group["reliability_brier_skill"] > 0.0).sum()),
                "positive_reliability_spread_markets": int(
                    (group["reliability_top_bottom_alpha_spread"] > 0.0).sum()
                ),
                "worst_drawdown": float(group["max_drawdown"].min()),
                "profit_concentration": _profit_concentration(group.set_index("market")["total_return"]),
                "median_trade_count": float(group["trade_count"].median()),
                "deflated_sharpe_probability": deflated_sharpe_probability(returns.to_numpy(), trial_count),
                "pbo": pbo,
            }
        )
    leaderboard = pd.DataFrame(rows)
    gates = config["selection"]
    leaderboard["reaches_sharpe_target"] = leaderboard["pooled_sharpe"] >= float(gates["target_pooled_sharpe"])
    leaderboard["robustness_pass"] = (
        (leaderboard["median_market_sharpe"] >= float(gates["minimum_median_market_sharpe"]))
        & (leaderboard["positive_markets"] >= int(gates["minimum_positive_markets"]))
        & (leaderboard["baseline_wins"] >= int(gates["minimum_baseline_wins"]))
        & (
            leaderboard["positive_brier_skill_markets"]
            >= int(gates["minimum_positive_brier_skill_markets"])
        )
        & (
            leaderboard["positive_reliability_spread_markets"]
            >= int(gates["minimum_positive_reliability_spread_markets"])
        )
        & (leaderboard["worst_drawdown"].abs() <= float(gates["maximum_worst_drawdown"]))
        & (leaderboard["profit_concentration"] <= float(gates["maximum_profit_concentration"]))
        & (
            leaderboard["deflated_sharpe_probability"]
            >= float(gates["minimum_deflated_sharpe_probability"])
        )
        & (leaderboard["pbo"] <= float(gates["maximum_pbo"]))
    )
    leaderboard["passes"] = leaderboard["reaches_sharpe_target"] & leaderboard["robustness_pass"]
    leaderboard = leaderboard.sort_values(
        ["passes", "robustness_pass", "pooled_sharpe", "median_market_sharpe"],
        ascending=False,
    ).reset_index(drop=True)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    eligible = leaderboard.loc[leaderboard["passes"]]
    selection = {
        "status": "selected" if not eligible.empty else "rejected",
        "selected_candidate": str(eligible.iloc[0]["candidate"]) if not eligible.empty else None,
        "best_observed_candidate": str(leaderboard.iloc[0]["candidate"]),
        "best_observed_pooled_sharpe": float(leaderboard.iloc[0]["pooled_sharpe"]),
        "pbo": pbo,
        "candidate_count": len(leaderboard),
        "development_split_only": True,
        "transformer_retrained": False,
        "rl_used": False,
    }
    _write_json(output / "selection.json", selection)
    _write_json(output / "evaluation_complete.json", {"status": "completed", **selection})
    return selection


def _evaluate_all(config: dict[str, Any], testbed: dict[str, Any], output: Path) -> dict[str, Any]:
    for topology in [str(item["id"]) for item in config["topologies"]]:
        for market in config["experiment"]["markets"]:
            print(f"EVALUATE topology={topology} market={market}", flush=True)
            _evaluate_candidate(config, testbed, output, topology, market)
    return _summarize(config, output)


def _job(
    identifier: str,
    command: list[str],
    outputs: list[Path],
    *,
    stage: str,
    dependencies: list[str] | None = None,
    inputs: list[str | Path] | None = None,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "stage": stage,
        "command": command,
        "depends_on": dependencies or [],
        "inputs": [str(item) if isinstance(item, str) else _rel(item) for item in (inputs or [])],
        "expected_outputs": [_rel(path) for path in outputs],
        "uses_gpu": False,
        "estimated_hours": 0.0,
    }


def _build_manifest(
    config_path: Path,
    config: dict[str, Any],
    testbed: dict[str, Any],
    output: Path,
    python: str,
) -> dict[str, Any]:
    preflight = _preflight(config, testbed)
    jobs: list[dict[str, Any]] = []
    source = _resolve(config["experiment"]["source_run"])
    script = _rel(Path(__file__))
    common = [python, script, "--config", _rel(config_path), "--run-id", output.name]
    filter_jobs: dict[int, str] = {}
    for seed in map(int, config["experiment"]["seeds"]):
        identifier = f"filter_global_seed_{seed}"
        filter_jobs[seed] = identifier
        jobs.append(
            _job(
                identifier,
                [*common, "--stage", "filter", "--seed", str(seed)],
                [
                    output / "filtered_memory" / f"global_seed_{seed}" / f"{market}.parquet"
                    for market in config["experiment"]["markets"]
                ]
                + [output / "filtered_memory" / f"global_seed_{seed}" / "summary.json"],
                stage="prepare",
                inputs=[source / "adapters" / f"global_seed_{seed}" / "train_decisions.parquet"],
            )
        )
    retrieval_jobs: list[str] = []
    development = str(config["experiment"]["development_period"])
    selection = str(config["experiment"]["selection_period"])
    for seed in map(int, config["experiment"]["seeds"]):
        for market in config["experiment"]["markets"]:
            identifier = f"retrieve_global_regional_{market}_seed_{seed}"
            retrieval_jobs.append(identifier)
            outputs: list[Path] = [output / "retrieval_status" / f"global_regional_{market}_seed_{seed}.json"]
            for period in (development, selection):
                root = output / "memory" / f"global_regional_{market}_seed_{seed}" / period / "eval"
                outputs.extend((root / "signals.parquet", root / "neighbors.parquet", root / "metrics.json"))
            jobs.append(
                _job(
                    identifier,
                    [*common, "--stage", "retrieve", "--market", market, "--seed", str(seed)],
                    outputs,
                    stage="retrieve",
                    dependencies=[filter_jobs[seed]],
                    inputs=[
                        output / "filtered_memory" / f"global_seed_{seed}" / f"{market}.parquet",
                        source / "decisions" / f"global_{market}_seed_{seed}" / f"{development}.parquet",
                        source / "decisions" / f"global_{market}_seed_{seed}" / f"{selection}.parquet",
                        f"{testbed['experiment']['data_root']}/{market}/*.parquet",
                    ],
                )
            )
    evidence_inputs: list[str | Path] = []
    for representation in ("regional", "global"):
        for market in config["experiment"]["markets"]:
            for seed in map(int, config["experiment"]["seeds"]):
                for period in (development, selection):
                    root = source / "memory" / f"{representation}_{market}_seed_{seed}" / period / "eval"
                    evidence_inputs.extend((root / "signals.parquet", root / "neighbors.parquet"))
    evidence_inputs.extend(
        (
            output / "memory" / f"global_regional_{market}_seed_{seed}" / period / "eval" / artifact
        )
        for market in config["experiment"]["markets"]
        for seed in map(int, config["experiment"]["seeds"])
        for period in (development, selection)
        for artifact in ("signals.parquet", "neighbors.parquet")
    )
    jobs.append(
        _job(
            "evaluate_frozen_memory_sweep",
            [*common, "--stage", "evaluate"],
            [output / "market_results.csv", output / "leaderboard.csv", output / "selection.json", output / "evaluation_complete.json"],
            stage="evaluate",
            dependencies=retrieval_jobs,
            inputs=evidence_inputs,
        )
    )
    manifest = {
        "run": {
            "id": output.name,
            "strict_environment": True,
            "stale_lock_seconds": 900,
            "inputs": [_rel(config_path), str(config["experiment"]["source_testbed_config"])],
            "source_patterns": [
                "src/backtest/**/*.py",
                "src/eval/memory_reliability.py",
                "src/eval/statistical_promotion.py",
                "src/memory/**/*.py",
                "scripts/run_phase6_frozen_memory_sweep.py",
                "scripts/run_market_memory_backtest.py",
                "configs/phase6_frozen_memory_sweep.yaml",
            ],
            "hardware": {"minimum_free_storage_gb": 5},
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
            for stage in ("prepare", "retrieve", "evaluate")
        },
        "gpu_jobs": 0,
        "preflight": preflight,
    }
    _write_json(output / "build_summary.json", summary)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config, testbed = _load(config_path)
    run_id = args.run_id
    output = _resolve(config["experiment"]["output_root"]) / run_id
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "build":
        return _build_manifest(config_path, config, testbed, output, args.python)
    if args.stage == "filter":
        if args.seed is None:
            raise ValueError("--seed is required for filter.")
        return _filter_seed(config, output, args.seed)
    if args.stage == "retrieve":
        if args.seed is None or not args.market:
            raise ValueError("--seed and --market are required for retrieve.")
        return _retrieve(config, testbed, output, args.market, args.seed)
    if args.stage == "evaluate":
        return _evaluate_all(config, testbed, output)
    raise ValueError(f"Unsupported stage: {args.stage}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_frozen_memory_sweep.yaml")
    parser.add_argument("--run-id", default="phase6_frozen_memory_v1")
    parser.add_argument("--stage", choices=("build", "filter", "retrieve", "evaluate"), default="build")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--market")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    print(json.dumps(_native(run(args)), indent=2))


if __name__ == "__main__":
    main()
