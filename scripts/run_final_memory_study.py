"""Run the final no-retraining exact-C0 and enriched-memory comparison."""

from __future__ import annotations

import argparse
import json
import sys
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
from src.backtest.market_memory_evaluator import (  # noqa: E402
    run_market_memory_evaluation,
)
from src.eval.local_rank_ensemble import evaluate_local_rank_market  # noqa: E402
from src.eval.policy_baselines import BaselineSuiteConfig, write_json  # noqa: E402
from src.decision.dataset import bounded_path_quality  # noqa: E402


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


def _output(config: dict[str, Any], run_id: str) -> Path:
    return PROJECT_ROOT / config["experiment"]["output_root"] / run_id


def _source(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["experiment"]["source_run"]


def _decision_source(config: dict[str, Any]) -> Path:
    """Resolve reusable query decisions independently of model artifacts."""

    return PROJECT_ROOT / config["experiment"].get(
        "source_decision_run",
        config["experiment"]["source_run"],
    )


def _source_key(config: dict[str, Any], market: str, seed: int) -> str:
    layout = config["experiment"].get("source_layout", "regional_market")
    if layout == "regional_market":
        return f"regional_{market}_seed_{seed}"
    if layout == "global_shared":
        return f"global_seed_{seed}"
    raise ValueError(f"Unsupported source layout: {layout}")


def _query_key(config: dict[str, Any], market: str, seed: int) -> str:
    layout = config["experiment"].get("source_layout", "regional_market")
    prefix = "global" if layout == "global_shared" else "regional"
    return f"{prefix}_{market}_seed_{seed}"


def _decision_path(
    config: dict[str, Any],
    output: Path,
    market: str,
    seed: int,
    period: str,
) -> Path:
    if bool(config.get("inference_refresh", {}).get("enabled", False)):
        if (
            config.get("inference_refresh", {}).get("period_mode", "per_period")
            == "combined"
        ):
            return (
                output
                / "decisions"
                / _query_key(config, market, seed)
                / "all_periods.parquet"
            )
        return output / "decisions" / _query_key(config, market, seed) / f"{period}.parquet"
    source_name = (
        "all_periods.parquet"
        if config["experiment"].get("source_decision_period_mode") == "combined"
        else f"{period}.parquet"
    )
    return (
        _decision_source(config)
        / "decisions"
        / _query_key(config, market, seed)
        / source_name
    )


def _validate(config: dict[str, Any]) -> None:
    protocol = config["protocol"]
    forbidden = (
        "transformer_training_allowed",
        "adapter_training_allowed",
        "offline_rl_allowed",
        "promotion_claim_allowed",
    )
    if any(protocol.get(key) is not False for key in forbidden):
        raise ValueError("The final memory study must remain no-training and no-promotion.")
    if config["consensus"].get("rank_aggregation") != "mean":
        raise ValueError("Exact C0 requires mean cross-seed percentile rank.")
    identifiers = [variant["id"] for variant in config["variants"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Variant IDs must be unique.")
    if protocol.get("causal_timestamp_unit", "ns") != "ns":
        raise ValueError("The final memory study requires nanosecond causal timestamps.")
    path_mode = protocol.get("path_quality_transform", "stored")
    if path_mode not in {"stored", "bounded_excursion_share"}:
        raise ValueError(f"Unsupported path-quality transform: {path_mode}")
    if bool(config.get("inference_refresh", {}).get("enabled", False)):
        if "source_encoder_run" not in config["experiment"]:
            raise ValueError("Inference refresh requires experiment.source_encoder_run.")


def _embedding_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    columns = [
        column
        for column in frame
        if column.startswith(prefix)
        and column.removeprefix(prefix).isdigit()
    ]
    return sorted(columns, key=lambda column: int(column.removeprefix(prefix)))


def materialize_retrieval_view(
    frame: pd.DataFrame,
    embedding_space: str,
    path_quality_transform: str = "stored",
) -> pd.DataFrame:
    """Make the retrieval geometry explicit and attach causal alpha metadata."""

    out = frame.copy()
    raw_columns = _embedding_columns(out, "latent_")
    decision_columns = _embedding_columns(out, "decision_")
    if embedding_space == "adapter":
        if not decision_columns:
            raise ValueError("Adapter retrieval requested but no decision embedding exists.")
        out = out.drop(columns=raw_columns)
        out = out.rename(
            columns={
                column: f"latent_{index}"
                for index, column in enumerate(decision_columns)
            }
        )
        expected_dimension = len(decision_columns)
    elif embedding_space == "raw":
        if not raw_columns:
            raise ValueError("Raw retrieval requested but no encoder latent exists.")
        out = out.drop(columns=decision_columns)
        expected_dimension = len(raw_columns)
    else:
        raise ValueError("embedding_space must be 'raw' or 'adapter'.")

    if "decision_outcome_available_timestamp" not in out:
        raise ValueError("Decision frame lacks exact outcome maturity timestamps.")
    out["outcome_available_timestamp"] = pd.to_datetime(
        out["decision_outcome_available_timestamp"],
        utc=True,
        errors="coerce",
    )
    if "future_blended_alpha_63" not in out:
        required = {"decision_return_63", "decision_benchmark_return"}
        if missing := required.difference(out.columns):
            raise ValueError(f"Cannot derive local alpha; missing {sorted(missing)}.")
        out["future_blended_alpha_63"] = (
            pd.to_numeric(out["decision_return_63"], errors="coerce")
            - pd.to_numeric(out["decision_benchmark_return"], errors="coerce")
        )
        out["alpha_target_source"] = "derived_local_cross_sectional_alpha"
    else:
        out["alpha_target_source"] = "stored_future_blended_alpha_63"
    if path_quality_transform == "bounded_excursion_share":
        required = {"decision_mfe", "decision_mae"}
        if missing := required.difference(out.columns):
            raise ValueError(
                f"Cannot derive bounded path quality; missing {sorted(missing)}."
            )
        out["decision_path_quality_bounded"] = bounded_path_quality(
            out["decision_mfe"],
            out["decision_mae"],
        )
        out["path_quality_target_source"] = "bounded_excursion_share"
    elif path_quality_transform == "stored":
        out["path_quality_target_source"] = "stored"
    else:
        raise ValueError(
            f"Unsupported path-quality transform: {path_quality_transform}"
        )
    out["retrieval_embedding_space"] = embedding_space
    actual_dimension = len(_embedding_columns(out, "latent_"))
    if actual_dimension != expected_dimension:
        raise RuntimeError(
            f"Retrieval view dimension mismatch: expected {expected_dimension}, "
            f"found {actual_dimension}."
        )
    return out


def _growing_bank(
    train: pd.DataFrame,
    period_frames: list[pd.DataFrame],
    embedding_space: str,
    path_quality_transform: str = "stored",
) -> pd.DataFrame:
    frames = [
        materialize_retrieval_view(
            train,
            embedding_space,
            path_quality_transform,
        )
    ]
    frames.extend(
        materialize_retrieval_view(
            frame,
            embedding_space,
            path_quality_transform,
        )
        for frame in period_frames
    )
    bank = pd.concat(frames, ignore_index=True, sort=False)
    bank["timestamp"] = pd.to_datetime(bank["timestamp"], utc=True)
    bank["ticker"] = bank["ticker"].astype(str)
    return (
        bank.sort_values(["ticker", "timestamp"])
        .drop_duplicates(["ticker", "timestamp"], keep="last")
        .sort_values(["timestamp", "ticker"])
        .reset_index(drop=True)
    )


def preflight(config_path: str, run_id: str) -> dict[str, Any]:
    """Validate that every reusable adapter and decision artifact is present."""

    config = _load(config_path)
    _validate(config)
    source = _source(config)
    output = _output(config, run_id)
    refresh = bool(config.get("inference_refresh", {}).get("enabled", False))
    testbed = _load(config["experiment"]["source_testbed_config"])
    data_root = PROJECT_ROOT / testbed["experiment"]["data_root"]
    encoder_source = PROJECT_ROOT / config["experiment"].get(
        "source_encoder_run",
        config["experiment"]["source_run"],
    )
    missing: list[str] = []
    for market in config["experiment"]["markets"]:
        for seed in config["experiment"]["seeds"]:
            source_key = _source_key(config, market, int(seed))
            required = [
                source / "adapters" / source_key / "train_decisions.parquet",
            ]
            if refresh:
                required.extend(
                    [
                        source / "adapters" / source_key / "decision_adapter.pt",
                        encoder_source / "models" / source_key / "final_model.pt",
                    ]
                )
            else:
                required.extend(
                    _decision_path(
                        config,
                        output,
                        market,
                        int(seed),
                        period,
                    )
                    for period in config["periods"]
                )
            missing.extend(_rel(path) for path in required if not path.exists())
        if refresh and not list((data_root / market).glob("*.parquet")):
            missing.append(_rel(data_root / market / "*.parquet"))
    if missing:
        raise FileNotFoundError(
            "Final memory study source artifacts are incomplete:\n"
            + "\n".join(f"- {path}" for path in missing)
        )
    payload = {
        "status": "ready",
        "source_run": config["experiment"]["source_run"],
        "markets": config["experiment"]["markets"],
        "seeds": config["experiment"]["seeds"],
        "transformer_training_jobs": 0,
        "adapter_training_jobs": 0,
        "offline_rl_jobs": 0,
        "inference_refresh": refresh,
    }
    write_json(_output(config, run_id) / "preflight.json", payload)
    return payload


def prepare_views(
    config_path: str,
    run_id: str,
    market: str,
    seed: int,
) -> dict[str, Any]:
    """Materialize raw and adapter retrieval views plus causal growing banks."""

    config = _load(config_path)
    _validate(config)
    source = _source(config)
    output = _output(config, run_id)
    source_key = _source_key(config, market, seed)
    key = _query_key(config, market, seed)
    train = pd.read_parquet(
        source / "adapters" / source_key / "train_decisions.parquet"
    )
    balance_columns = {
        str(variant["memory_overrides"]["balance_group_column"])
        for variant in config["variants"]
        if variant.get("memory_overrides", {}).get("balance_group_column")
    }
    if missing := balance_columns.difference(train.columns):
        raise ValueError(
            f"Training memory lacks configured balance columns: {sorted(missing)}"
        )
    period_names = list(config["periods"])
    period_frames: dict[str, pd.DataFrame] = {}
    cached_decisions: dict[Path, pd.DataFrame] = {}
    for period, period_config in config["periods"].items():
        path = _decision_path(config, output, market, seed, period)
        if path not in cached_decisions:
            cached_decisions[path] = pd.read_parquet(path)
            cached_decisions[path]["timestamp"] = pd.to_datetime(
                cached_decisions[path]["timestamp"],
                utc=True,
            )
        start = pd.Timestamp(period_config["start"], tz="UTC")
        end = pd.Timestamp(period_config["end"], tz="UTC")
        frame = cached_decisions[path]
        period_frames[period] = frame.loc[
            (frame["timestamp"] >= start) & (frame["timestamp"] <= end)
        ].copy()
        period_frames[period]["market"] = market
    root = output / "views" / key
    root.mkdir(parents=True, exist_ok=True)
    dimensions: dict[str, int] = {}
    path_quality_transform = config["protocol"].get(
        "path_quality_transform",
        "stored",
    )
    for embedding_space in ("raw", "adapter"):
        view = materialize_retrieval_view(
            train,
            embedding_space,
            path_quality_transform,
        )
        dimensions[embedding_space] = len(_embedding_columns(view, "latent_"))
        view.to_parquet(
            root / f"static_{embedding_space}.parquet",
            index=False,
            compression="zstd",
        )
    prior: list[pd.DataFrame] = []
    for period in period_names:
        period_root = root / period
        period_root.mkdir(parents=True, exist_ok=True)
        current = period_frames[period]
        for embedding_space in ("raw", "adapter"):
            materialize_retrieval_view(
                current,
                embedding_space,
                path_quality_transform,
            ).to_parquet(
                period_root / f"query_{embedding_space}.parquet",
                index=False,
                compression="zstd",
            )
        prior.append(current)
        for embedding_space in ("raw", "adapter"):
            _growing_bank(
                train,
                prior,
                embedding_space,
                path_quality_transform,
            ).to_parquet(
                period_root / f"growing_{embedding_space}.parquet",
                index=False,
                compression="zstd",
            )
    payload = {
        "status": "completed",
        "market": market,
        "seed": seed,
        "dimensions": dimensions,
        "alpha_target_source": (
            "stored_future_blended_alpha_63"
            if "future_blended_alpha_63" in train
            else "derived_local_cross_sectional_alpha"
        ),
        "path_quality_transform": path_quality_transform,
        "coverage": {},
    }
    for period in period_names:
        summary = json.loads(
            _decision_path(
                config,
                output,
                market,
                seed,
                period,
            ).with_suffix(".json").read_text(encoding="utf-8")
        )
        payload["coverage"][period] = summary.get("coverage", {})
    write_json(root / "summary.json", payload)
    return payload


def _variant(config: dict[str, Any], identifier: str) -> dict[str, Any]:
    matches = [value for value in config["variants"] if value["id"] == identifier]
    if len(matches) != 1:
        raise ValueError(f"Unknown variant: {identifier}")
    return matches[0]


def _memory_values(
    config: dict[str, Any],
    variant: dict[str, Any],
) -> dict[str, Any]:
    values = dict(config["memory_defaults"])
    target_key = (
        "exact_c0_targets" if variant["targets"] == "exact_c0" else "rally_targets"
    )
    values.update(config[target_key])
    values.update(variant.get("memory_overrides", {}))
    return values


def run_retrieval(
    config_path: str,
    run_id: str,
    variant_id: str,
    market: str,
    seed: int,
    period: str,
) -> dict[str, Any]:
    """Run one frozen retrieval variant from an explicit materialized view."""

    config = _load(config_path)
    _validate(config)
    variant = _variant(config, variant_id)
    output = _output(config, run_id)
    key = _query_key(config, market, seed)
    views = output / "views" / key
    embedding = variant["embedding_space"]
    if variant["memory_mode"] == "static":
        memory_path = views / f"static_{embedding}.parquet"
    else:
        memory_path = views / period / f"growing_{embedding}.parquet"
    query_path = views / period / f"query_{embedding}.parquet"
    destination = output / "retrieval" / variant_id / key / period
    testbed = _load(config["experiment"]["source_testbed_config"])
    data_root = Path(testbed["experiment"]["data_root"])
    policy = dict(config["policy"])
    policy.update(variant.get("policy_overrides", {}))
    policy["slippage_bps"] = float(
        testbed["markets"][market]["execution_cost_bps"]
    )
    evaluation = {
        "data": {
            "train_latents": _rel(memory_path),
            "test_latents": _rel(query_path),
            "precomputed_glob": f"{data_root.as_posix()}/{market}/*.parquet",
            "output_dir": _rel(destination),
        },
        "memory": _memory_values(config, variant),
        "policy": policy,
        "evaluation": {
            "baselines": "",
            "memory_metric_target": "future_blended_alpha_63",
        },
        "run": {
            "write_neighbors": bool(
                config.get("diagnostics", {}).get("write_neighbors", False)
            ),
            "write_memory_reports": False,
        },
    }
    config_snapshot = (
        output
        / "generated_configs"
        / "memory"
        / variant_id
        / key
        / f"{period}.yaml"
    )
    _write_yaml(config_snapshot, evaluation)
    result_dir = run_market_memory_evaluation(
        evaluation,
        PROJECT_ROOT,
        run_id="eval",
    )
    payload = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "status": "completed",
        "variant": variant_id,
        "market": market,
        "seed": seed,
        "period": period,
        "metrics": payload,
    }


def evaluate_variant(
    config_path: str,
    run_id: str,
    variant_id: str,
    market: str,
    period: str,
) -> dict[str, Any]:
    """Evaluate one variant's three-seed mean-rank consensus."""

    config = _load(config_path)
    _validate(config)
    variant = _variant(config, variant_id)
    output = _output(config, run_id)
    seed_signals = {
        int(seed): pd.read_parquet(
            output
            / "retrieval"
            / variant_id
            / _query_key(config, market, int(seed))
            / period
            / "eval"
            / "signals.parquet"
        )
        for seed in config["experiment"]["seeds"]
    }
    testbed = _load(config["experiment"]["source_testbed_config"])
    policy = dict(config["policy"])
    policy.update(variant.get("policy_overrides", {}))
    policy["slippage_bps"] = float(
        testbed["markets"][market]["execution_cost_bps"]
    )
    return evaluate_local_rank_market(
        seed_signals=seed_signals,
        policy=PolicyConfig(**policy),
        consensus_config=ConsensusSignalConfig(**config["consensus"]),
        baseline_config=BaselineSuiteConfig(**config["baselines"]),
        output=output / "evaluation" / variant_id / period / market,
    )


def _pooled(curves: list[pd.DataFrame], initial_capital: float) -> dict[str, Any]:
    returns = []
    for index, curve in enumerate(curves):
        frame = curve.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values("timestamp")
        returns.append(
            frame.set_index("timestamp")["equity"]
            .astype(float)
            .pct_change()
            .rename(str(index))
        )
    daily = pd.concat(returns, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    equity = initial_capital * (1.0 + daily).cumprod()
    frame = pd.DataFrame({"timestamp": equity.index, "equity": equity.to_numpy()})
    frame["return"] = daily.to_numpy()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    frame["exposure"] = np.nan
    return {
        "metrics": compute_backtest_metrics(pd.DataFrame(), frame, initial_capital),
        "equity": frame,
    }


def aggregate_variant(
    config_path: str,
    run_id: str,
    variant_id: str,
    period: str,
) -> dict[str, Any]:
    """Pool one predeclared variant across all markets."""

    config = _load(config_path)
    _validate(config)
    output = _output(config, run_id)
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    lifts: list[float] = []
    for market in config["experiment"]["markets"]:
        root = output / "evaluation" / variant_id / period / market
        payload = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
        equal_weight = payload["baselines"]["equal_weight_buy_hold"]
        momentum_key = f"momentum_{config['baselines']['momentum_window']}"
        momentum = payload["baselines"][momentum_key]
        diagnostics = payload["diagnostics"]
        seed_coverages = []
        for seed in config["experiment"]["seeds"]:
            summary_path = (
                output
                / "views"
                / _query_key(config, market, int(seed))
                / "summary.json"
            )
            summary = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                if summary_path.exists()
                else {}
            )
            coverage = summary.get("coverage", {}).get(period, {})
            seed_coverages.append(
                float(
                    coverage.get(
                        "minimum_ticker_fraction",
                        coverage.get("total_fraction", 1.0),
                    )
                )
            )
        rows.append(
            {
                "market": market,
                **payload["ensemble"],
                "equal_weight_sharpe": equal_weight["sharpe"],
                "momentum_sharpe": momentum["sharpe"],
                "excess_sharpe_vs_equal_weight": (
                    payload["ensemble"]["sharpe"] - equal_weight["sharpe"]
                ),
                "excess_sharpe_vs_momentum": (
                    payload["ensemble"]["sharpe"] - momentum["sharpe"]
                ),
                "mean_retrieval_confidence": diagnostics.get(
                    "mean_retrieval_confidence"
                ),
                "mean_retrieval_agreement": diagnostics.get(
                    "mean_retrieval_agreement_score"
                ),
                "mean_historical_diversity": diagnostics.get(
                    "mean_retrieval_historical_diversity"
                ),
                "mean_effective_sample_size": diagnostics.get(
                    "mean_retrieval_effective_sample_size"
                ),
                "mean_cross_ticker_rate": diagnostics.get(
                    "mean_retrieval_cross_ticker_rate"
                ),
                "mean_scale_score_std": diagnostics.get(
                    "mean_retrieval_scale_score_std"
                ),
                "mean_scale_sign_agreement": diagnostics.get(
                    "mean_retrieval_scale_sign_agreement"
                ),
                "minimum_query_coverage": min(seed_coverages, default=1.0),
            }
        )
        lifts.append(float(payload["ensemble_sharpe_lift_vs_mean_seed"]))
        curves.append(
            pd.read_csv(
                root
                / "baselines"
                / "local_rank_memory"
                / "equity_curve.csv"
            )
        )
    table = pd.DataFrame(rows)
    pooled = _pooled(curves, float(config["policy"]["initial_capital"]))
    positive = table["total_return"].clip(lower=0.0)
    concentration = (
        float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0
    )
    destination = output / "aggregate" / variant_id / period
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "market_results.csv", index=False)
    pooled["equity"].to_csv(destination / "pooled_equity.csv", index=False)
    gates = config["gates"]
    metrics = pooled["metrics"]
    gate_results = {
        "target_pooled_sharpe": bool(
            metrics["sharpe"] >= gates["target_pooled_sharpe"]
        ),
        "maximum_drawdown": bool(
            abs(metrics["max_drawdown"]) <= gates["maximum_drawdown"]
        ),
        "minimum_positive_markets": bool(
            (table["total_return"] > 0).sum() >= gates["minimum_positive_markets"]
        ),
        "maximum_profit_concentration": bool(
            concentration <= gates["maximum_profit_concentration"]
        ),
        "minimum_ensemble_seed_lift": bool(
            np.mean(lifts) >= gates["minimum_ensemble_seed_lift"]
        ),
    }
    if "maximum_market_drawdown" in gates:
        gate_results["maximum_market_drawdown"] = bool(
            table["max_drawdown"].abs().max()
            <= float(gates["maximum_market_drawdown"])
        )
    if "minimum_market_query_coverage" in gates:
        gate_results["minimum_market_query_coverage"] = bool(
            table["minimum_query_coverage"].min()
            >= float(gates["minimum_market_query_coverage"])
        )
    if "minimum_median_market_sharpe" in gates:
        gate_results["minimum_median_market_sharpe"] = bool(
            table["sharpe"].median()
            >= float(gates["minimum_median_market_sharpe"])
        )
    if "minimum_baseline_wins" in gates:
        gate_results["minimum_baseline_wins"] = bool(
            (table["excess_sharpe_vs_equal_weight"] > 0).sum()
            >= int(gates["minimum_baseline_wins"])
        )
    payload = {
        "status": (
            "exploratory_pass" if all(gate_results.values()) else "exploratory_rejected"
        ),
        "variant": variant_id,
        "period": period,
        "evidence_status": config["periods"][period]["evidence_status"],
        "promotion_allowed": False,
        "pooled": metrics,
        "median_market_sharpe": float(table["sharpe"].median()),
        "median_excess_sharpe_vs_equal_weight": float(
            table["excess_sharpe_vs_equal_weight"].median()
        ),
        "median_excess_sharpe_vs_momentum": float(
            table["excess_sharpe_vs_momentum"].median()
        ),
        "positive_markets": int((table["total_return"] > 0).sum()),
        "profit_concentration": concentration,
        "mean_ensemble_seed_lift": float(np.mean(lifts)),
        "minimum_market_query_coverage": float(
            table["minimum_query_coverage"].min()
        ),
        "worst_market_drawdown": float(table["max_drawdown"].min()),
        "baseline_wins": int(
            (table["excess_sharpe_vs_equal_weight"] > 0).sum()
        ),
        "gates": gate_results,
    }
    write_json(destination / "summary.json", payload)
    return payload


def compare_period(config_path: str, run_id: str, period: str) -> dict[str, Any]:
    """Write one no-selection leaderboard across the fixed memory variants."""

    config = _load(config_path)
    _validate(config)
    output = _output(config, run_id)
    rows = [
        json.loads(
            (
                output
                / "aggregate"
                / variant["id"]
                / period
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for variant in config["variants"]
    ]
    table = pd.DataFrame(
        [
            {
                "variant": row["variant"],
                "pooled_return": row["pooled"]["total_return"],
                "pooled_sharpe": row["pooled"]["sharpe"],
                "max_drawdown": row["pooled"]["max_drawdown"],
                "median_market_sharpe": row["median_market_sharpe"],
                "median_excess_sharpe_vs_equal_weight": row[
                    "median_excess_sharpe_vs_equal_weight"
                ],
                "median_excess_sharpe_vs_momentum": row[
                    "median_excess_sharpe_vs_momentum"
                ],
                "positive_markets": row["positive_markets"],
                "profit_concentration": row["profit_concentration"],
                "mean_ensemble_seed_lift": row["mean_ensemble_seed_lift"],
                "minimum_market_query_coverage": row[
                    "minimum_market_query_coverage"
                ],
                "worst_market_drawdown": row["worst_market_drawdown"],
                "baseline_wins": row["baseline_wins"],
                "status": row["status"],
            }
            for row in rows
        ]
    )
    raw = table.loc[table["variant"] == "raw_c0_static"]
    if len(raw) != 1:
        raise RuntimeError("Comparison requires exactly one raw_c0_static control.")
    table["delta_sharpe_vs_raw_c0"] = (
        table["pooled_sharpe"] - float(raw.iloc[0]["pooled_sharpe"])
    )
    table["delta_return_vs_raw_c0"] = (
        table["pooled_return"] - float(raw.iloc[0]["pooled_return"])
    )
    table = table.sort_values("pooled_sharpe", ascending=False)
    destination = output / "comparison" / period
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination / "leaderboard.csv", index=False)
    payload = {
        "period": period,
        "evidence_status": config["periods"][period]["evidence_status"],
        "promotion_allowed": False,
        "variant_count": len(table),
        "highest_sharpe_variant": str(table.iloc[0]["variant"]),
        "highest_sharpe": float(table.iloc[0]["pooled_sharpe"]),
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
            "estimated_hours": float(hours),
        }
    )
    return job_id


def build(config_path: str, run_id: str, python: str) -> dict[str, Any]:
    """Compile the final frozen-model memory study DAG."""

    config = _load(config_path)
    _validate(config)
    output = _output(config, run_id)
    source = _source(config)
    testbed = _load(config["experiment"]["source_testbed_config"])
    data_root = Path(testbed["experiment"]["data_root"])
    encoder_source = PROJECT_ROOT / config["experiment"].get(
        "source_encoder_run",
        config["experiment"]["source_run"],
    )
    refresh = bool(config.get("inference_refresh", {}).get("enabled", False))
    jobs: list[dict[str, Any]] = []
    preflight_job = _job(
        jobs,
        "preflight",
        [
            python,
            "scripts/run_final_memory_study.py",
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
    prepare_jobs: dict[tuple[str, int], str] = {}
    export_jobs: dict[tuple[str, int, str], str] = {}
    retrieval_jobs: dict[tuple[str, str, int, str], str] = {}
    evaluation_jobs: dict[tuple[str, str, str], str] = {}
    aggregate_jobs: dict[tuple[str, str], str] = {}
    tabular_jobs: dict[tuple[str, str], str] = {}
    compare_jobs: dict[str, str] = {}

    for market in config["experiment"]["markets"]:
        for seed_value in config["experiment"]["seeds"]:
            seed = int(seed_value)
            source_key = _source_key(config, market, seed)
            key = _query_key(config, market, seed)
            views = output / "views" / key
            source_inputs = [
                source / "adapters" / source_key / "train_decisions.parquet"
            ]
            prepare_dependencies = [preflight_job]
            if refresh:
                target_glob = f"{data_root.as_posix()}/{market}/*.parquet"
                checkpoint = (
                    encoder_source / "models" / source_key / "final_model.pt"
                )
                adapter = (
                    source
                    / "adapters"
                    / source_key
                    / "decision_adapter.pt"
                )
                refresh_mode = config.get("inference_refresh", {}).get(
                    "period_mode",
                    "per_period",
                )
                if refresh_mode == "combined":
                    period_exports = {
                        "all_periods": {
                            "start": min(
                                value["start"]
                                for value in config["periods"].values()
                            ),
                            "end": max(
                                value["end"]
                                for value in config["periods"].values()
                            ),
                        }
                    }
                elif refresh_mode == "per_period":
                    period_exports = config["periods"]
                else:
                    raise ValueError(
                        f"Unsupported inference refresh period mode: {refresh_mode}"
                    )
                for export_period, period_config in period_exports.items():
                    decisions = _decision_path(
                        config,
                        output,
                        market,
                        seed,
                        (
                            next(iter(config["periods"]))
                            if export_period == "all_periods"
                            else export_period
                        ),
                    )
                    export_id = _job(
                        jobs,
                        f"export_{key}_{export_period}",
                        [
                            python,
                            "scripts/export_phase5_transfer_latents.py",
                            "--encoder-checkpoint",
                            _rel(checkpoint),
                            "--adapter-checkpoint",
                            _rel(adapter),
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
                        dependencies=[preflight_job],
                        inputs=[checkpoint, adapter, target_glob, config_path],
                        gpu=True,
                        hours=float(
                            config.get("hardware", {}).get(
                                "export_gpu_hours",
                                0.3,
                            )
                        ),
                    )
                    export_jobs[(market, seed, export_period)] = export_id
                    prepare_dependencies.append(export_id)
                    source_inputs.append(decisions)
            else:
                source_inputs.extend(
                    _decision_path(
                        config,
                        output,
                        market,
                        seed,
                        period,
                    )
                    for period in config["periods"]
                )
            expected = [
                views / "summary.json",
                views / "static_raw.parquet",
                views / "static_adapter.parquet",
            ]
            for period in config["periods"]:
                expected.extend(
                    [
                        views / period / "query_raw.parquet",
                        views / period / "query_adapter.parquet",
                        views / period / "growing_adapter.parquet",
                        views / period / "growing_raw.parquet",
                    ]
                )
            prepare_jobs[(market, seed)] = _job(
                jobs,
                f"prepare_{key}",
                [
                    python,
                    "scripts/run_final_memory_study.py",
                    "--config",
                    config_path,
                    "--run-id",
                    run_id,
                    "--stage",
                    "prepare",
                    "--market",
                    market,
                    "--seed",
                    str(seed),
                ],
                expected,
                dependencies=prepare_dependencies,
                inputs=source_inputs,
            )

    for variant in config["variants"]:
        variant_id = variant["id"]
        for market in config["experiment"]["markets"]:
            for seed_value in config["experiment"]["seeds"]:
                seed = int(seed_value)
                key = _query_key(config, market, seed)
                for period in config["periods"]:
                    destination = (
                        output / "retrieval" / variant_id / key / period / "eval"
                    )
                    retrieval_outputs = [
                        destination / "metrics.json",
                        destination / "signals.parquet",
                    ]
                    if bool(
                        config.get("diagnostics", {}).get(
                            "write_neighbors", False
                        )
                    ):
                        retrieval_outputs.append(destination / "neighbors.parquet")
                    retrieval_jobs[(variant_id, market, seed, period)] = _job(
                        jobs,
                        f"retrieve_{variant_id}_{key}_{period}",
                        [
                            python,
                            "scripts/run_final_memory_study.py",
                            "--config",
                            config_path,
                            "--run-id",
                            run_id,
                            "--stage",
                            "retrieve",
                            "--variant",
                            variant_id,
                            "--market",
                            market,
                            "--seed",
                            str(seed),
                            "--period",
                            period,
                        ],
                        retrieval_outputs,
                        dependencies=[prepare_jobs[(market, seed)]],
                    )
            for period in config["periods"]:
                destination = output / "evaluation" / variant_id / period / market
                evaluation_jobs[(variant_id, market, period)] = _job(
                    jobs,
                    f"evaluate_{variant_id}_{market}_{period}",
                    [
                        python,
                        "scripts/run_final_memory_study.py",
                        "--config",
                        config_path,
                        "--run-id",
                        run_id,
                        "--stage",
                        "evaluate",
                        "--variant",
                        variant_id,
                        "--market",
                        market,
                        "--period",
                        period,
                    ],
                    [
                        destination / "metrics.json",
                        destination / "consensus_signals.parquet",
                    ],
                    dependencies=[
                        retrieval_jobs[(variant_id, market, int(seed), period)]
                        for seed in config["experiment"]["seeds"]
                    ],
                )
        for period in config["periods"]:
            destination = output / "aggregate" / variant_id / period
            aggregate_jobs[(variant_id, period)] = _job(
                jobs,
                f"aggregate_{variant_id}_{period}",
                [
                    python,
                    "scripts/run_final_memory_study.py",
                    "--config",
                    config_path,
                    "--run-id",
                    run_id,
                    "--stage",
                    "aggregate",
                    "--variant",
                    variant_id,
                    "--period",
                    period,
                ],
                [destination / "summary.json", destination / "market_results.csv"],
                dependencies=[
                    evaluation_jobs[(variant_id, market, period)]
                    for market in config["experiment"]["markets"]
                ],
            )

    if bool(config.get("tabular_baselines", {}).get("enabled", False)):
        reference_variant = config.get("tabular_baselines", {}).get(
            "reference_variant",
            config["variants"][0]["id"],
        )
        for market in config["experiment"]["markets"]:
            for period in config["periods"]:
                destination = output / "tabular_baselines" / period / market
                tabular_jobs[(market, period)] = _job(
                    jobs,
                    f"tabular_decoders_{market}_{period}",
                    [
                        python,
                        "scripts/evaluate_frozen_tabular_decoders.py",
                        "--config",
                        config_path,
                        "--run-id",
                        run_id,
                        "--market",
                        market,
                        "--period",
                        period,
                    ],
                    [destination / "summary.json"],
                    dependencies=[
                        retrieval_jobs[(reference_variant, market, int(seed), period)]
                        for seed in config["experiment"]["seeds"]
                    ],
                )
    for period in config["periods"]:
        destination = output / "comparison" / period
        compare_jobs[period] = _job(
            jobs,
            f"compare_{period}",
            [
                python,
                "scripts/run_final_memory_study.py",
                "--config",
                config_path,
                "--run-id",
                run_id,
                "--stage",
                "compare",
                "--period",
                period,
            ],
            [destination / "summary.json", destination / "leaderboard.csv"],
            dependencies=[
                aggregate_jobs[(variant["id"], period)]
                for variant in config["variants"]
            ],
        )

    if bool(config.get("credibility", {}).get("enabled", False)):
        for period in config["periods"]:
            destination = output / "credibility" / period
            dependencies = [compare_jobs[period]]
            dependencies.extend(
                evaluation_jobs[(variant["id"], market, period)]
                for variant in config["variants"]
                for market in config["experiment"]["markets"]
            )
            dependencies.extend(
                tabular_jobs[(market, period)]
                for market in config["experiment"]["markets"]
                if (market, period) in tabular_jobs
            )
            _job(
                jobs,
                f"credibility_{period}",
                [
                    python,
                    "scripts/run_transfer_credibility_audit.py",
                    "--config",
                    config_path,
                    "--run-id",
                    run_id,
                    "--period",
                    period,
                ],
                [
                    destination / "summary.json",
                    destination / "strategy_summary.csv",
                    destination / "block_bootstrap.csv",
                    destination / "country_jackknife.csv",
                    destination / "neighbor_age_profile.csv",
                    destination / "neighbor_market_profile.csv",
                    destination / "reliability_deciles.csv",
                    destination / "audit_report.md",
                ],
                dependencies=dependencies,
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
            "hardware": {"minimum_free_storage_gb": 20},
            "budgets": {
                "full": {
                    "max_gpu_hours": float(
                        config.get("hardware", {}).get("max_gpu_hours", 0.0)
                    )
                }
            },
        },
        "jobs": jobs,
    }
    manifest_path = output / "experiment_manifest.yaml"
    _write_yaml(manifest_path, manifest)
    payload = {
        "run_id": run_id,
        "output": _rel(output),
        "manifest": _rel(manifest_path),
        "job_count": len(jobs),
        "variant_count": len(config["variants"]),
        "gpu_job_count": sum(bool(job["uses_gpu"]) for job in jobs),
        "transformer_training_jobs": 0,
        "adapter_training_jobs": 0,
        "offline_rl_jobs": 0,
        "inference_refresh_jobs": len(export_jobs),
        "promotion_allowed": False,
    }
    write_json(output / "build_summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_memory_study.yaml")
    parser.add_argument("--run-id", default="final_memory_study_v1")
    parser.add_argument(
        "--stage",
        choices=(
            "build",
            "preflight",
            "prepare",
            "retrieve",
            "evaluate",
            "aggregate",
            "compare",
        ),
        default="build",
    )
    parser.add_argument("--variant")
    parser.add_argument("--market")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--period")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.stage == "build":
        result = build(args.config, args.run_id, args.python)
    elif args.stage == "preflight":
        result = preflight(args.config, args.run_id)
    elif args.stage == "prepare":
        if args.market is None or args.seed is None:
            parser.error("--stage prepare requires --market and --seed.")
        result = prepare_views(
            args.config, args.run_id, args.market, args.seed
        )
    elif args.stage == "retrieve":
        if None in (args.variant, args.market, args.seed, args.period):
            parser.error(
                "--stage retrieve requires --variant, --market, --seed, and --period."
            )
        result = run_retrieval(
            args.config,
            args.run_id,
            args.variant,
            args.market,
            args.seed,
            args.period,
        )
    elif args.stage == "evaluate":
        if None in (args.variant, args.market, args.period):
            parser.error("--stage evaluate requires --variant, --market, and --period.")
        result = evaluate_variant(
            args.config, args.run_id, args.variant, args.market, args.period
        )
    elif args.stage == "aggregate":
        if args.variant is None or args.period is None:
            parser.error("--stage aggregate requires --variant and --period.")
        result = aggregate_variant(
            args.config, args.run_id, args.variant, args.period
        )
    else:
        if args.period is None:
            parser.error("--stage compare requires --period.")
        result = compare_period(args.config, args.run_id, args.period)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
