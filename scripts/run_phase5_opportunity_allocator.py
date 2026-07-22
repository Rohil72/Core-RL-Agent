"""Train and evaluate per-opportunity allocation over frozen C0 market-memory signals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
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
from src.backtest.market_memory_evaluator import join_prices  # noqa: E402
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
    stationary_bootstrap_delta,
)
from src.memory import MarketMemoryConfig, score_market_memory  # noqa: E402
from src.memory.experience import latent_columns  # noqa: E402
from src.policy.opportunity_allocator import (  # noqa: E402
    OpportunityAllocatorConfig,
    allocation_utility,
    build_opportunity_features,
    decile_diagnostics,
    fit_calibrated_obvious_allocator,
    fit_nonlinear_allocator,
    obvious_signal_score,
    predict_calibrated_obvious_allocator,
    predict_nonlinear_score,
    scores_to_allocations,
)


STAGES = ("evidence", "learnability", "backtest", "all")
LABEL_COLUMNS = (
    "decision_net_alpha",
    "decision_utility",
    "decision_mae",
    "decision_mfe",
    "decision_return_63",
    "decision_is_mature",
    "decision_outcome_available_timestamp",
)
EVALUATION_COLUMNS = (
    *LABEL_COLUMNS,
    "open",
    "close",
    "future_return_63",
    "future_max_return_63",
    "future_min_return_63",
    "event_upside_before_drawdown_126",
    "pred_future_max_return_63",
    "pred_future_min_return_63",
)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64, datetime, date)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2), encoding="utf-8")


def _decision_space(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=latent_columns(frame)).copy()
    decision_columns = sorted(
        [column for column in out if column.startswith("decision_") and column.removeprefix("decision_").isdigit()],
        key=lambda column: int(column.removeprefix("decision_")),
    )
    if not decision_columns:
        raise ValueError("Decision-space retrieval requires decision_0...decision_n columns.")
    return out.rename(columns={column: f"latent_{index}" for index, column in enumerate(decision_columns)})


def _fold_seed_paths(source: Path, filename: str) -> dict[str, dict[int, Path]]:
    found: dict[str, dict[int, Path]] = {}
    for path in source.glob(f"fold_*/seed_*/{filename}"):
        fold = path.parents[1].name
        seed = int(path.parent.name.removeprefix("seed_"))
        found.setdefault(fold, {})[seed] = path
    return found


def _market_memory_config(phase4e: dict[str, Any]) -> MarketMemoryConfig:
    return MarketMemoryConfig(
        **phase4e["memory_defaults"],
        target_alpha="future_blended_alpha_63",
        score_mode="alpha_lcb",
    )


def _evidence_root(config: dict[str, Any], output: Path) -> Path:
    """Resolve immutable causal evidence, optionally reusing a completed prior run."""
    source = config["experiment"].get("source_evidence_run")
    return (PROJECT_ROOT / source).resolve() if source else output


def _existing_evidence_summary(root: Path) -> dict[str, Any]:
    """Validate and summarize a referenced causal evidence run without recomputation."""
    audit_path = root / "causal_retrieval_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError(f"Referenced evidence audit is missing: {audit_path}")
    audit = pd.read_csv(audit_path)
    violations = int(pd.to_numeric(audit["causal_violations"], errors="coerce").fillna(1).sum())
    if violations:
        raise RuntimeError(f"Referenced evidence contains {violations} causal violations.")
    training_folds = {path.parent.name for path in (root / "training_evidence").glob("fold_*/consensus.parquet")}
    evaluation_folds = {path.parent.name for path in (root / "evaluation_evidence").glob("fold_*/consensus.parquet")}
    if not training_folds or training_folds != evaluation_folds:
        raise RuntimeError("Referenced training and evaluation evidence folds are incomplete or mismatched.")
    return {
        "reused_from": str(root),
        "fold_count": len(training_folds),
        "seed_run_count": len(audit),
        "causal_violations": violations,
        "query_start": str(audit["query_start"].min()),
    }


def build_causal_training_evidence(
    config: dict[str, Any],
    output: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Create corrected 2022 training and 2023 evaluation retrieval evidence."""
    experiment = config["experiment"]
    source = PROJECT_ROOT / experiment["source_phase5_run"]
    paths = _fold_seed_paths(source, "train_decisions.parquet")
    if len(paths) < int(experiment["minimum_folds"]):
        raise RuntimeError(f"Expected {experiment['minimum_folds']} folds, found {len(paths)}.")
    phase4e = yaml.safe_load((PROJECT_ROOT / experiment["phase4e_config"]).read_text(encoding="utf-8"))
    memory_cfg = _market_memory_config(phase4e)
    policy = PolicyConfig(**phase4e["policy"])
    consensus_cfg = ConsensusSignalConfig(minimum_votes=0, exit_smoothing_span=1, exit_score_quantile=0.50)
    cutoff = pd.Timestamp(experiment["allocator_query_start"], tz="UTC")
    audit_rows: list[dict[str, Any]] = []

    for fold, seed_paths in sorted(paths.items()):
        seed_signals: dict[int, pd.DataFrame] = {}
        evaluation_signals: dict[int, pd.DataFrame] = {}
        for seed, source_path in sorted(seed_paths.items()):
            cache = output / "training_evidence" / fold / f"seed_{seed}.parquet"
            audit_path = cache.with_suffix(".audit.json")
            if resume and cache.exists():
                seed_signals[seed] = pd.read_parquet(cache)
                if audit_path.exists():
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                else:
                    audit = _recovered_cache_audit(fold, seed, "training", seed_signals[seed])
                    _write_json(audit_path, audit)
                audit_rows.append(audit)
            else:
                decisions = _decision_space(pd.read_parquet(source_path))
                decisions["timestamp"] = pd.to_datetime(decisions["timestamp"], utc=True)
                memory = decisions.loc[decisions["timestamp"] < cutoff].copy()
                query = decisions.loc[decisions["timestamp"] >= cutoff].copy()
                if memory.empty or query.empty:
                    raise ValueError(f"{fold} seed {seed} does not straddle {cutoff.date()}.")
                signals, neighbors = score_market_memory(
                    memory,
                    query,
                    memory_cfg,
                    keep_query_columns=LABEL_COLUMNS,
                )
                audit = _retrieval_audit(fold, seed, "training", memory, query, signals, neighbors)
                if audit["causal_violations"]:
                    raise RuntimeError(f"Causal training retrieval audit failed for {fold} seed {seed}.")
                _write_json(audit_path, audit)
                signals.to_parquet(cache, index=False)
                seed_signals[seed] = signals
                audit_rows.append(audit)

            evaluation_cache = output / "evaluation_evidence" / fold / f"seed_{seed}.parquet"
            evaluation_audit_path = evaluation_cache.with_suffix(".audit.json")
            if resume and evaluation_cache.exists():
                evaluation_signals[seed] = pd.read_parquet(evaluation_cache)
                if evaluation_audit_path.exists():
                    audit = json.loads(evaluation_audit_path.read_text(encoding="utf-8"))
                else:
                    audit = _recovered_cache_audit(
                        fold, seed, "evaluation", evaluation_signals[seed]
                    )
                    _write_json(evaluation_audit_path, audit)
                audit_rows.append(audit)
            else:
                full_memory = _decision_space(pd.read_parquet(source_path))
                full_memory["timestamp"] = pd.to_datetime(full_memory["timestamp"], utc=True)
                validation_path = source / fold / f"seed_{seed}" / "val_decisions.parquet"
                if not validation_path.exists():
                    raise FileNotFoundError(f"Missing validation decisions: {validation_path}")
                validation = _decision_space(pd.read_parquet(validation_path))
                validation = join_prices(
                    validation,
                    phase4e["data"]["precomputed_glob"],
                    PROJECT_ROOT,
                )
                scored, neighbors = score_market_memory(
                    full_memory,
                    validation,
                    memory_cfg,
                    keep_query_columns=EVALUATION_COLUMNS,
                )
                audit = _retrieval_audit(
                    fold, seed, "evaluation", full_memory, validation, scored, neighbors
                )
                if audit["causal_violations"]:
                    raise RuntimeError(f"Causal evaluation retrieval audit failed for {fold} seed {seed}.")
                _write_json(evaluation_audit_path, audit)
                scored.to_parquet(evaluation_cache, index=False)
                evaluation_signals[seed] = scored
                audit_rows.append(audit)
        consensus = build_consensus_signals(seed_signals, policy, consensus_cfg)
        consensus["allocator_eligible"] = consensus.apply(
            lambda row: consensus_row_filter(row, policy, 0), axis=1
        )
        consensus.to_parquet(output / "training_evidence" / fold / "consensus.parquet", index=False)
        evaluation_consensus = build_consensus_signals(evaluation_signals, policy, consensus_cfg)
        evaluation_consensus["allocator_eligible"] = evaluation_consensus.apply(
            lambda row: consensus_row_filter(row, policy, 0), axis=1
        )
        evaluation_consensus.to_parquet(
            output / "evaluation_evidence" / fold / "consensus.parquet", index=False
        )

    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(output / "causal_retrieval_audit.csv", index=False)
    summary = {
        "fold_count": len(paths),
        "seed_run_count": len(audit_frame),
        "causal_violations": int(audit_frame["causal_violations"].sum()),
        "query_start": str(cutoff),
    }
    _write_json(output / "evidence_summary.json", summary)
    return summary


def _retrieval_audit(
    fold: str,
    seed: int,
    split: str,
    memory: pd.DataFrame,
    query: pd.DataFrame,
    signals: pd.DataFrame,
    neighbors: pd.DataFrame,
) -> dict[str, Any]:
    if neighbors.empty:
        causal_violations = 0
    else:
        neighbor_available = pd.to_datetime(
            neighbors["neighbor_outcome_available_timestamp"], utc=True, errors="coerce"
        )
        query_timestamp = pd.to_datetime(neighbors["query_timestamp"], utc=True, errors="coerce")
        causal_violations = int((neighbor_available > query_timestamp).fillna(False).sum())
    return {
        "fold": fold,
        "seed": seed,
        "split": split,
        "memory_rows": len(memory),
        "query_rows": len(query),
        "signal_rows": len(signals),
        "neighbor_rows": len(neighbors),
        "causal_violations": causal_violations,
        "memory_end": memory["timestamp"].max(),
        "query_start": query["timestamp"].min(),
        "query_end": query["timestamp"].max(),
    }


def _recovered_cache_audit(
    fold: str,
    seed: int,
    split: str,
    signals: pd.DataFrame,
) -> dict[str, Any]:
    """Recover a cache written only after its in-memory causal audit passed."""
    timestamps = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")
    return {
        "fold": fold,
        "seed": seed,
        "split": split,
        "memory_rows": None,
        "query_rows": len(signals),
        "signal_rows": len(signals),
        "neighbor_rows": None,
        "causal_violations": 0,
        "memory_end": None,
        "query_start": timestamps.min(),
        "query_end": timestamps.max(),
        "recovered_after_audit_serialization_failure": True,
    }


def run_learnability(
    config: dict[str, Any],
    output: Path,
) -> pd.DataFrame:
    """Compare obvious and nonlinear inference on a strictly later slice of 2022."""
    allocator_cfg = OpportunityAllocatorConfig(**_allocator_kwargs(config["allocator"]))
    fraction = float(config["learnability"]["fit_date_fraction"])
    rows: list[dict[str, Any]] = []
    (output / "learnability").mkdir(parents=True, exist_ok=True)
    evidence_root = _evidence_root(config, output)
    paths = sorted((evidence_root / "training_evidence").glob("fold_*/consensus.parquet"))
    if not paths:
        raise FileNotFoundError("No causal training evidence found; run --stage evidence first.")
    for path in paths:
        fold = path.parent.name
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        dates = np.asarray(sorted(frame["timestamp"].unique()))
        split_index = min(max(int(len(dates) * fraction), 1), len(dates) - 1)
        fit = frame.loc[frame["timestamp"] <= dates[split_index - 1]].copy()
        test = frame.loc[frame["timestamp"] >= dates[split_index]].copy()
        test_features, _ = build_opportunity_features(test)
        diagnostic_scores = {"obvious": obvious_signal_score(test_features)}
        candidates = tuple(config["experiment"].get(
            "candidates", ("B0_c0_static", "B1_obvious_signal", "B2_nonlinear_inference")
        ))
        if "B2_nonlinear_inference" in candidates:
            model, feature_names = fit_nonlinear_allocator(fit, allocator_cfg)
            diagnostic_scores["nonlinear"] = predict_nonlinear_score(model, test, feature_names)
        if "B3_calibrated_obvious" in candidates:
            calibrated = fit_calibrated_obvious_allocator(fit, allocator_cfg)
            prediction = predict_calibrated_obvious_allocator(calibrated, test, allocator_cfg)
            diagnostic_scores["calibrated_obvious"] = prediction["allocator_score"]
        eligible = test.get("allocator_eligible", True)
        eligible = pd.Series(eligible, index=test.index).fillna(False).astype(bool)
        for name, scores in diagnostic_scores.items():
            table, metrics = decile_diagnostics(
                test.loc[eligible], scores.loc[eligible], allocator_cfg.downside_weight
            )
            table.insert(0, "model", name)
            table.insert(0, "fold", fold)
            table.to_csv(output / "learnability" / f"{fold}_{name}_deciles.csv", index=False)
            rows.append({"fold": fold, "model": name, **metrics, "fit_rows": len(fit), "test_rows": len(test)})
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(output / "learnability" / "summary.csv", index=False)
    report = [
        "# Opportunity Allocator Learnability",
        "",
        "The fit slice precedes the test slice within 2022. No 2023 outcomes are used here.",
        "",
        diagnostics.to_markdown(index=False),
    ]
    (output / "learnability" / "report.md").write_text("\n".join(report), encoding="utf-8")
    return diagnostics


def _allocator_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    out = dict(values)
    for key in ("action_rank_thresholds", "absolute_utility_thresholds", "action_levels"):
        if key in out:
            out[key] = tuple(float(value) for value in out[key])
    return out


def _entry_explanations(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    dates = sorted(pd.to_datetime(signals["timestamp"], utc=True).unique())
    prior = {dates[index]: dates[index - 1] for index in range(1, len(dates))}
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"], utc=True)
    out["signal_date"] = out["entry_date"].map(prior)
    columns = [
        "ticker", "timestamp", "allocator_score", "entry_allocation_fraction",
        "consensus_entry_rank", "consensus_economic_score", "seed_vote_fraction",
        "seed_rank_std", "retrieval_expected_alpha", "retrieval_alpha_ci_low",
        "retrieval_downside_cvar", "retrieval_confidence", "retrieval_agreement_score",
    ]
    available = [column for column in columns if column in signals]
    return out.merge(
        signals[available],
        left_on=["ticker", "signal_date"],
        right_on=["ticker", "timestamp"],
        how="left",
        suffixes=("", "_signal"),
    )


def _equity_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().dropna()


def _sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 1e-12:
        return 0.0
    return float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0))


def _pooled(paths: list[Path]) -> pd.Series:
    series = [_equity_returns(path).rename(str(index)) for index, path in enumerate(paths) if path.exists()]
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna() if series else pd.Series(dtype=float)


def run_backtests(config: dict[str, Any], output: Path) -> pd.DataFrame:
    """Fit on all causal 2022 evidence and evaluate entry sizing on frozen 2023 C0 signals."""
    experiment = config["experiment"]
    phase4e = yaml.safe_load((PROJECT_ROOT / experiment["phase4e_config"]).read_text(encoding="utf-8"))
    policy = PolicyConfig(**phase4e["policy"])
    allocator_cfg = OpportunityAllocatorConfig(**_allocator_kwargs(config["allocator"]))
    variants = tuple(experiment.get(
        "candidates", ("B0_c0_static", "B1_obvious_signal", "B2_nonlinear_inference")
    ))
    supported = {"B0_c0_static", "B1_obvious_signal", "B2_nonlinear_inference", "B3_calibrated_obvious"}
    unknown = set(variants) - supported
    if unknown:
        raise ValueError(f"Unknown opportunity allocator candidates: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    (output / "learnability").mkdir(parents=True, exist_ok=True)
    evidence_root = _evidence_root(config, output)

    for train_path in sorted((evidence_root / "training_evidence").glob("fold_*/consensus.parquet")):
        fold = train_path.parent.name
        train = pd.read_parquet(train_path)
        evaluation_path = evidence_root / "evaluation_evidence" / fold / "consensus.parquet"
        if not evaluation_path.exists():
            raise FileNotFoundError(f"Missing corrected evaluation evidence: {evaluation_path}")
        evaluation = pd.read_parquet(evaluation_path)

        features, _ = build_opportunity_features(evaluation)
        scores: dict[str, pd.Series] = {"B1_obvious_signal": obvious_signal_score(features)}
        calibrated_predictions: dict[str, pd.DataFrame] = {}
        model_path = output / "models"
        model_path.mkdir(parents=True, exist_ok=True)
        if "B2_nonlinear_inference" in variants:
            model, feature_names = fit_nonlinear_allocator(train, allocator_cfg)
            scores["B2_nonlinear_inference"] = predict_nonlinear_score(
                model, evaluation, feature_names
            )
            joblib.dump(
                {"model": model, "feature_names": feature_names, "config": asdict(allocator_cfg)},
                model_path / f"{fold}_B2_nonlinear_inference.joblib",
            )
        if "B3_calibrated_obvious" in variants:
            calibrated = fit_calibrated_obvious_allocator(train, allocator_cfg)
            prediction = predict_calibrated_obvious_allocator(calibrated, evaluation, allocator_cfg)
            calibrated_predictions["B3_calibrated_obvious"] = prediction
            scores["B3_calibrated_obvious"] = prediction["allocator_score"]
            joblib.dump(
                {"model": calibrated, "config": asdict(allocator_cfg)},
                model_path / f"{fold}_B3_calibrated_obvious.joblib",
            )
        oracle_table, oracle_metrics = decile_diagnostics(
            evaluation.loc[evaluation["allocator_eligible"]],
            allocation_utility(evaluation, allocator_cfg.downside_weight).loc[evaluation["allocator_eligible"]],
            allocator_cfg.downside_weight,
        )
        oracle_table.to_csv(output / "learnability" / f"{fold}_oracle_deciles.csv", index=False)
        _write_json(output / "learnability" / f"{fold}_oracle_metrics.json", oracle_metrics)

        for variant in variants:
            signals = evaluation.copy()
            allocation_column = None
            if variant != "B0_c0_static":
                if variant in calibrated_predictions:
                    for column in calibrated_predictions[variant]:
                        signals[column] = calibrated_predictions[variant][column]
                else:
                    signals["allocator_score"] = scores[variant]
                    signals["entry_allocation_fraction"] = scores_to_allocations(
                        signals, signals["allocator_score"], allocator_cfg
                    )
                allocation_column = "entry_allocation_fraction"
                deciles, diagnostics = decile_diagnostics(
                    signals.loc[signals["allocator_eligible"]],
                    signals.loc[signals["allocator_eligible"], "allocator_score"],
                    allocator_cfg.downside_weight,
                )
                deciles.to_csv(output / "learnability" / f"{fold}_{variant}_2023_deciles.csv", index=False)
                _write_json(output / "learnability" / f"{fold}_{variant}_2023_metrics.json", diagnostics)
            trades, equity = run_long_only_backtest(
                signals,
                policy,
                score_col="consensus_entry_rank",
                exit_score_col="consensus_exit_score",
                row_filter=lambda row, cfg, _score: consensus_row_filter(row, cfg, 0),
                entry_allocation_col=allocation_column,
            )
            metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
            metrics.update(equal_weight_baseline(signals, policy.initial_capital))
            if allocation_column:
                actions = signals[allocation_column]
                metrics["mean_entry_allocation"] = float(actions.mean())
                metrics["allocator_abstain_fraction"] = float((actions <= 0.0).mean())
            else:
                metrics["mean_entry_allocation"] = 1.0
                metrics["allocator_abstain_fraction"] = 0.0
            result = output / "policies" / variant / fold
            result.mkdir(parents=True, exist_ok=True)
            trades.to_csv(result / "trades.csv", index=False)
            equity.to_csv(result / "equity_curve.csv", index=False)
            _entry_explanations(trades, signals).to_csv(result / "trade_explanations.csv", index=False)
            if allocation_column:
                allocation_columns = [
                    column for column in (
                        "ticker", "timestamp", "allocator_score", allocation_column,
                        "calibrated_utility", "base_allocation_fraction", "support_multiplier", "market_support",
                        "evidence_support", "support_veto",
                    ) if column in signals
                ]
                signals[allocation_columns].to_parquet(
                    result / "allocation_signals.parquet", index=False
                )
            _write_json(result / "metrics.json", metrics)
            rows.append({"candidate": variant, "fold": fold, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "fold_summary.csv", index=False)
    return summary


def select_candidate(config: dict[str, Any], output: Path, summary: pd.DataFrame) -> dict[str, Any]:
    """Apply the fixed robustness and statistical gates without opening confirmation data."""
    candidates = sorted(summary["candidate"].unique())
    records: list[dict[str, Any]] = []
    return_series: dict[str, pd.Series] = {}
    for candidate in candidates:
        selected = summary.loc[summary["candidate"] == candidate]
        paths = [output / "policies" / candidate / fold / "equity_curve.csv" for fold in selected["fold"]]
        pooled = _pooled(paths)
        return_series[candidate] = pooled
        excess = pd.to_numeric(selected["total_return"], errors="coerce") - pd.to_numeric(
            selected["equal_weight_baseline_return"], errors="coerce"
        )
        bootstrap = stationary_bootstrap_delta(
            pooled.to_numpy(), np.zeros(len(pooled), dtype=float),
            expected_block_length=int(config["selection"]["bootstrap_block_length"]),
            samples=int(config["selection"]["bootstrap_samples"]), seed=7,
        )
        records.append(
            {
                "candidate": candidate,
                "pooled_sharpe": _sharpe(pooled),
                "mean_return": float(selected["total_return"].mean()),
                "minimum_fold_sharpe": float(selected["sharpe"].min()),
                "fold_sharpe_dispersion": float(selected["sharpe"].std(ddof=0)),
                "worst_drawdown": float(selected["max_drawdown"].min()),
                "mean_excess_return": float(excess.mean()),
                "positive_fold_fraction": float((selected["total_return"] > 0.0).mean()),
                "mean_trade_count": float(selected["trade_count"].mean()),
                "mean_entry_allocation": float(selected["mean_entry_allocation"].mean()),
                "deflated_sharpe_probability": deflated_sharpe_probability(
                    pooled.to_numpy(), len(candidates)
                ),
                "bootstrap_probability_positive": bootstrap.probability_positive,
                "bootstrap_ci_low": bootstrap.ci_low,
                "bootstrap_ci_high": bootstrap.ci_high,
            }
        )
    leaderboard = pd.DataFrame(records)
    matrix = pd.concat(return_series, axis=1).fillna(0.0)
    pbo = probability_of_backtest_overfitting(matrix.to_numpy())
    gates = config["selection"]
    leaderboard["pbo"] = pbo
    leaderboard["passes"] = (
        (leaderboard["pooled_sharpe"] >= float(gates["minimum_pooled_sharpe"]))
        & (leaderboard["minimum_fold_sharpe"] >= float(gates["minimum_fold_sharpe"]))
        & (leaderboard["worst_drawdown"].abs() <= float(gates["maximum_worst_drawdown"]))
        & (leaderboard["mean_excess_return"] > float(gates["minimum_mean_excess_return"]))
        & (leaderboard["positive_fold_fraction"] >= float(gates["minimum_positive_fold_fraction"]))
        & (leaderboard["mean_trade_count"] >= float(gates["minimum_mean_trade_count"]))
        & (leaderboard["deflated_sharpe_probability"] >= float(gates["minimum_deflated_sharpe_probability"]))
        & (leaderboard["bootstrap_probability_positive"] >= float(gates["minimum_bootstrap_probability"]))
        & (pbo <= float(gates["maximum_pbo"]))
    )
    final_gate = "minimum_out_of_year_positive_folds" in gates
    gate_details: dict[str, Any] = {}
    if final_gate:
        baseline = str(gates["baseline_candidate"])
        baseline_folds = summary.loc[summary["candidate"] == baseline, ["fold", "sharpe", "max_drawdown"]]
        required_ooy = int(gates["minimum_out_of_year_positive_folds"])
        required_improvement = int(gates["minimum_improvement_folds"])
        ooy_positive: dict[str, int] = {}
        improvement_folds: dict[str, int] = {}
        for candidate in candidates:
            if candidate == baseline:
                ooy_positive[candidate] = 0
                improvement_folds[candidate] = 0
                continue
            positive = 0
            for path in sorted((output / "learnability").glob(f"fold_*_{candidate}_2023_metrics.json")):
                metrics = json.loads(path.read_text(encoding="utf-8"))
                if float(metrics.get("spearman") or 0.0) > 0.0 and float(
                    metrics.get("top_decile_lift") or 0.0
                ) > 0.0:
                    positive += 1
            ooy_positive[candidate] = positive
            candidate_folds = summary.loc[
                summary["candidate"] == candidate, ["fold", "sharpe", "max_drawdown"]
            ]
            compared = candidate_folds.merge(
                baseline_folds, on="fold", suffixes=("_candidate", "_baseline"), validate="one_to_one"
            )
            improved = (compared["sharpe_candidate"] > compared["sharpe_baseline"]) | (
                compared["max_drawdown_candidate"].abs() < compared["max_drawdown_baseline"].abs()
            )
            improvement_folds[candidate] = int(improved.sum())
        leaderboard["out_of_year_positive_folds"] = leaderboard["candidate"].map(ooy_positive).fillna(0).astype(int)
        leaderboard["improvement_folds"] = leaderboard["candidate"].map(improvement_folds).fillna(0).astype(int)
        leaderboard["learnability_pass"] = (
            leaderboard["out_of_year_positive_folds"] >= required_ooy
        )
        leaderboard["improvement_pass"] = leaderboard["improvement_folds"] >= required_improvement
        leaderboard["passes"] &= leaderboard["learnability_pass"] & leaderboard["improvement_pass"]
        gate_details = {
            "out_of_year_positive_fold_count": ooy_positive,
            "improvement_fold_count": improvement_folds,
            "required_out_of_year_positive_folds": required_ooy,
            "required_improvement_folds": required_improvement,
        }
    else:
        learnability = pd.read_csv(output / "learnability" / "summary.csv")
        pivot = learnability.pivot(index="fold", columns="model", values="spearman")
        nonlinear_wins = int((pivot["nonlinear"] > pivot["obvious"]).sum())
        obvious_rows = learnability.loc[learnability["model"] == "obvious"]
        nonlinear_rows = learnability.loc[learnability["model"] == "nonlinear"]
        obvious_positive = int(
            ((obvious_rows["spearman"] > 0.0) & (obvious_rows["top_decile_lift"] > 0.0)).sum()
        )
        nonlinear_positive = int(
            ((nonlinear_rows["spearman"] > 0.0) & (nonlinear_rows["top_decile_lift"] > 0.0)).sum()
        )
        required_obvious = int(config["learnability"]["minimum_obvious_positive_folds"])
        required_nonlinear = int(config["learnability"]["minimum_nonlinear_fold_wins"])
        learnability_gate = {
            "B0_c0_static": False,
            "B1_obvious_signal": obvious_positive >= required_obvious,
            "B2_nonlinear_inference": nonlinear_wins >= required_nonlinear and nonlinear_positive >= required_nonlinear,
        }
        leaderboard["learnability_pass"] = leaderboard["candidate"].map(learnability_gate).fillna(False)
        leaderboard["improvement_pass"] = True
        leaderboard["passes"] &= leaderboard["learnability_pass"]
        gate_details = {
            "nonlinear_learnability_fold_wins": nonlinear_wins,
            "nonlinear_positive_fold_count": nonlinear_positive,
            "obvious_positive_fold_count": obvious_positive,
            "required_nonlinear_fold_wins": required_nonlinear,
            "required_obvious_positive_folds": required_obvious,
        }
    leaderboard = leaderboard.sort_values(["passes", "pooled_sharpe"], ascending=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    eligible = leaderboard.loc[
        leaderboard["passes"] & (leaderboard["candidate"] != config["selection"]["baseline_candidate"])
    ]
    selected = str(eligible.iloc[0]["candidate"]) if not eligible.empty else None
    selection = {
        "status": "cross_market_pending" if selected else "rejected",
        "selected_candidate": selected,
        "promotion_pass": False,
        "development_split_only": True,
        "cross_market_confirmation_required": True,
        **gate_details,
        "pbo": pbo,
        "output": str(output),
    }
    _write_json(output / "selection.json", selection)
    report_columns = [
        "candidate", "pooled_sharpe", "minimum_fold_sharpe", "fold_sharpe_dispersion",
        "mean_return", "worst_drawdown", "mean_entry_allocation",
        "deflated_sharpe_probability", "bootstrap_probability_positive", "pbo",
        "learnability_pass", "improvement_pass", "passes",
    ]
    if final_gate:
        report_columns[1:1] = ["out_of_year_positive_folds", "improvement_folds"]
    report = [
        "# Phase 5 Opportunity Allocation Selection",
        "",
        f"- Status: `{selection['status']}`",
        f"- Selected development candidate: `{selection['selected_candidate']}`",
        *(
            [
                f"- Required out-of-year positive folds: `{required_ooy}`",
                f"- Required folds improving Sharpe or drawdown over baseline: `{required_improvement}`",
            ]
            if final_gate
            else [
                f"- Obvious positive folds: `{obvious_positive}/{required_obvious}` required",
                f"- Nonlinear wins over obvious: `{nonlinear_wins}/{required_nonlinear}` required",
            ]
        ),
        "- Cross-market confirmation remains locked.",
        "",
        leaderboard[report_columns].to_markdown(index=False),
    ]
    (output / "selection_report.md").write_text("\n".join(report), encoding="utf-8")
    return selection


def run(config_path: str, run_id: str, stage: str, resume: bool) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["experiment"]["output_root"] / run_id
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to reuse non-empty opportunity-allocation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stages = ("evidence", "learnability", "backtest") if stage == "all" else (stage,)
    result: dict[str, Any] = {"run_id": run_id, "output": str(output), "stages": list(stages)}
    if "evidence" in stages:
        evidence_root = _evidence_root(config, output)
        result["evidence"] = (
            _existing_evidence_summary(evidence_root)
            if evidence_root != output.resolve()
            else build_causal_training_evidence(config, output, resume=resume)
        )
    if "learnability" in stages:
        diagnostics = run_learnability(config, output)
        result["learnability_rows"] = len(diagnostics)
    if "backtest" in stages:
        if not (output / "learnability" / "summary.csv").exists():
            raise FileNotFoundError("Backtesting requires learnability/summary.csv; run --stage learnability first.")
        summary = run_backtests(config, output)
        result.update(select_candidate(config, output, summary))
    _write_json(output / "run_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_opportunity_allocator.yaml")
    parser.add_argument("--run-id", default="phase5_opportunity_allocator_v1")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.stage, args.resume), indent=2))


if __name__ == "__main__":
    main()
