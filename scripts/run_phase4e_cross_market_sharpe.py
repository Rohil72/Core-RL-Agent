"""Run bounded cross-market policy tuning on frozen Phase 4C latent exports."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase4d_retrieval_semantics import (  # noqa: E402
    _calibrate_distances,
    _complete_run_dirs,
    _ensure_output,
    _evaluation_config,
    _flat_metrics,
    _native,
    _prepared_paths,
    _read_yaml,
    _resolve,
    _run_key,
    _select_runs,
    _write_json,
    _write_yaml,
    audit_sources,
    run_evaluations,
)
from src.backtest.market_memory_backtester import (  # noqa: E402
    PolicyConfig,
    compute_backtest_metrics,
    equal_weight_baseline,
    run_long_only_backtest,
)
from src.backtest.market_memory_evaluator import (  # noqa: E402
    run_policy_suite,
    signal_quality_metrics,
    write_run_artifacts,
)
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
)
from src.eval.representation_metrics import compute_memory_metrics  # noqa: E402
from src.memory import (  # noqa: E402
    AggregationConfig,
    ConfidenceConfig,
    EnsembleConfig,
    RelativeOutcomeConfig,
    combine_seed_signals,
)
from src.memory.evidence import build_evidence_summary  # noqa: E402

LOGGER = logging.getLogger(__name__)
STAGES = ("audit", "score", "sweep", "select", "confirm", "all")


def _policy_for_candidate(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    policy = copy.deepcopy(config["policy"])
    policy.update(copy.deepcopy(candidate.get("overrides", {})))
    if candidate["semantic"] == "A0_legacy_identity":
        policy["min_alpha_lcb"] = None
    return policy


def _semantic(config: dict[str, Any], semantic_id: str) -> dict[str, Any]:
    return next(item for item in config["variants"] if item["id"] == semantic_id)


def prepare_original_geometry(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    *,
    resume: bool,
) -> None:
    """Prepare relative outcomes and calibrate original-geometry OOD thresholds."""
    relative_cfg = RelativeOutcomeConfig(
        **{key: value for key, value in config["relative_outcomes"].items() if key != "enabled"}
    )
    for run_dir in run_dirs:
        fold, seed = _run_key(run_dir)
        LOGGER.info("Preparing %s seed %s", fold, seed)
        prepared = _prepared_paths(run_dir, output, relative_cfg, resume=resume)
        calibration_path = output / "metrics" / fold / f"seed_{seed}" / "distance_calibration.json"
        if resume and calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if "original" in calibration:
                continue
        else:
            calibration = {}
        train = pd.read_parquet(prepared["train"])
        calibration["original"] = _calibrate_distances(train, None, config["ood_calibration"])
        _write_json(calibration_path, calibration)


def score_semantics(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    splits: Iterable[str],
    *,
    resume: bool,
) -> None:
    """Score each outcome semantic once so policy candidates can reuse signals."""
    prepare_original_geometry(config, output, run_dirs, resume=resume)
    base_config = copy.deepcopy(config)
    base_config["variants"] = [_semantic(config, "A0_legacy_identity")]
    run_evaluations(base_config, output, run_dirs, splits, resume=resume)
    derived = [item for item in config["variants"] if item["id"] != "A0_legacy_identity"]
    for split in splits:
        for run_dir in run_dirs:
            for variant in derived:
                _derive_semantic_cache(config, output, run_dir, str(split), variant, resume=resume)


def _derive_semantic_cache(
    config: dict[str, Any],
    output: Path,
    run_dir: Path,
    split: str,
    variant: dict[str, Any],
    *,
    resume: bool,
) -> None:
    """Re-aggregate cached neighbors under a new outcome semantic."""
    fold, seed = _run_key(run_dir)
    semantic_id = str(variant["id"])
    base_dir = output / "evaluations" / split / "A0_legacy_identity" / fold / f"seed_{seed}" / "eval"
    target_dir = output / "evaluations" / split / semantic_id / fold / f"seed_{seed}" / "eval"
    metrics_path = target_dir / "metrics.json"
    if resume and metrics_path.exists():
        return
    signals = pd.read_parquet(base_dir / "signals.parquet")
    neighbors = pd.read_parquet(base_dir / "neighbors.parquet")
    neighbor_groups = {
        int(query_id): group.sort_values("rank")
        for query_id, group in neighbors.groupby("query_id", sort=False)
    }
    memory_cfg = copy.deepcopy(config["memory_defaults"])
    memory_cfg["target_alpha"] = variant["target_alpha"]
    memory_cfg["score_mode"] = variant["score_mode"]
    calibration_path = output / "metrics" / fold / f"seed_{seed}" / "distance_calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["original"]
    aggregation = AggregationConfig(
        method=memory_cfg.get("aggregation_method", "gaussian"),
        gaussian_bandwidth=memory_cfg.get("gaussian_bandwidth"),
        trim_fraction=float(memory_cfg.get("trim_fraction", 0.10)),
        tail_fraction=float(memory_cfg.get("tail_fraction", 0.25)),
    )
    confidence = ConfidenceConfig(reference_distance=float(calibration["confidence_reference_distance"]))
    score_weights = {
        "expected_upside_weight": memory_cfg.get("expected_upside_weight", 1.0),
        "path_quality_weight": memory_cfg.get("path_quality_weight", 0.10),
        "downside_weight": memory_cfg.get("downside_weight", 0.80),
        "uncertainty_weight": memory_cfg.get("uncertainty_weight", 0.20),
        "confidence_weight": memory_cfg.get("confidence_weight", 0.10),
        "disagreement_weight": memory_cfg.get("disagreement_weight", 0.10),
        "score_mode": variant["score_mode"],
    }
    rows: list[dict[str, Any]] = []
    for signal in signals.itertuples(index=False):
        row = signal._asdict()
        query_id = int(row["query_id"])
        group = neighbor_groups.get(query_id)
        if group is not None and not group.empty:
            historical = group.rename(columns={"neighbor_ticker": "ticker"})
            evidence = build_evidence_summary(
                historical,
                historical["distance"].to_numpy(dtype=float),
                str(row["ticker"]),
                upside_col=str(memory_cfg["target_upside"]),
                alpha_col=str(variant["target_alpha"]),
                downside_col=str(memory_cfg["target_downside"]),
                path_quality_col=str(memory_cfg["target_path_quality"]),
                holding_period_col=(
                    str(memory_cfg["target_holding_period"])
                    if memory_cfg.get("target_holding_period") in historical
                    else None
                ),
                aggregation=aggregation,
                confidence=confidence,
                score_weights=score_weights,
            )
            row.update(evidence.to_signal_payload())
        rows.append(row)
    derived_signals = pd.DataFrame(rows)
    policy_dict = copy.deepcopy(config["policy"])
    policy = PolicyConfig(**policy_dict)
    primary = run_policy_suite(derived_signals, policy, {"baselines": ""})["retrieval"]
    metrics = dict(primary["metrics"])
    metrics.update(equal_weight_baseline(derived_signals, policy.initial_capital))
    metrics.update(signal_quality_metrics(derived_signals))
    metrics.update(
        compute_memory_metrics(
            derived_signals,
            neighbors,
            target_names=["future_blended_alpha_63"],
        )
    )
    relative_cfg = RelativeOutcomeConfig(
        **{key: value for key, value in config["relative_outcomes"].items() if key != "enabled"}
    )
    prepared = _prepared_paths(run_dir, output, relative_cfg, resume=True)
    artifact_config = _evaluation_config(config, output, fold, seed, variant, split, prepared)
    write_run_artifacts(
        target_dir,
        artifact_config,
        derived_signals,
        neighbors,
        primary["trades"],
        primary["equity"],
        metrics,
        primary.get("decisions"),
    )
    summary_path = output / "evaluation_summary.csv"
    record = pd.DataFrame(
        [{"split": split, "variant": semantic_id, "fold": fold, "seed": seed, **_flat_metrics(metrics)}]
    )
    existing = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    summary = pd.concat([existing, record], ignore_index=True)
    summary = summary.drop_duplicates(["split", "variant", "fold", "seed"], keep="last")
    summary.to_csv(summary_path, index=False)


def run_policy_sweep(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    splits: Iterable[str],
    *,
    resume: bool,
    candidates: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Backtest predeclared policies while reusing cached semantic signals."""
    selected_candidates = candidates or config["policy_candidates"]
    rows: list[dict[str, Any]] = []
    for split in splits:
        for candidate in selected_candidates:
            candidate_id = str(candidate["id"])
            semantic_id = str(candidate["semantic"])
            policy_dict = _policy_for_candidate(config, candidate)
            policy = PolicyConfig(**policy_dict)
            for run_dir in run_dirs:
                fold, seed = _run_key(run_dir)
                signal_path = (
                    output / "evaluations" / str(split) / semantic_id / fold / f"seed_{seed}" / "eval" / "signals.parquet"
                )
                if not signal_path.exists():
                    raise FileNotFoundError(f"Missing semantic signal cache: {signal_path}")
                result_dir = output / "policies" / str(split) / candidate_id / fold / f"seed_{seed}" / "eval"
                metrics_path = result_dir / "metrics.json"
                if resume and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    signals = pd.read_parquet(signal_path)
                    decisions: list[dict[str, Any]] = []
                    trades, equity = run_long_only_backtest(signals, policy, decision_log=decisions)
                    metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
                    metrics.update(equal_weight_baseline(signals, policy.initial_capital))
                    result_dir.mkdir(parents=True, exist_ok=True)
                    trades.to_csv(result_dir / "trades.csv", index=False)
                    equity.to_csv(result_dir / "equity_curve.csv", index=False)
                    pd.DataFrame(decisions).to_csv(result_dir / "decisions.csv", index=False)
                    _write_json(metrics_path, metrics)
                    _write_yaml(
                        result_dir / "policy_snapshot.yaml",
                        {"candidate": candidate, "resolved_policy": policy_dict, "signal_source": str(signal_path)},
                    )
                rows.append(
                    {
                        "split": str(split),
                        "candidate": candidate_id,
                        "semantic": semantic_id,
                        "fold": fold,
                        "seed": seed,
                        **{
                            key: metrics.get(key)
                            for key in (
                                "total_return", "annualized_return", "sharpe", "max_drawdown",
                                "profit_factor", "trade_count", "win_rate", "average_trade_return",
                                "turnover", "exposure", "capital_efficiency", "equal_weight_baseline_return",
                            )
                        },
                    }
                )
    summary_path = output / "policy_sweep_summary.csv"
    current = pd.DataFrame(rows)
    if summary_path.exists():
        current = pd.concat([pd.read_csv(summary_path), current], ignore_index=True)
        current = current.drop_duplicates(["split", "candidate", "fold", "seed"], keep="last")
    current.to_csv(summary_path, index=False)
    return current


def _equity_returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().dropna()


def _pooled_returns(
    output: Path,
    split: str,
    candidate: str,
    rows: pd.DataFrame,
) -> pd.Series:
    series: list[pd.Series] = []
    for row in rows.itertuples(index=False):
        path = output / "policies" / split / candidate / row.fold / f"seed_{int(row.seed)}" / "eval" / "equity_curve.csv"
        if path.exists():
            series.append(_equity_returns(path).rename(f"{row.fold}_{int(row.seed)}"))
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna() if series else pd.Series(dtype=float)


def _sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size < 3 or np.std(values, ddof=1) <= 1e-12:
        return 0.0
    return float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0))


def _candidate_stats(
    config: dict[str, Any],
    output: Path,
    rows: pd.DataFrame,
    candidate: str,
    split: str,
) -> dict[str, Any]:
    selected = rows.loc[rows["candidate"] == candidate].copy()
    pooled = _pooled_returns(output, split, candidate, selected)
    fold_sharpes: list[float] = []
    fold_excess: list[float] = []
    for _, fold_rows in selected.groupby("fold"):
        fold_sharpes.append(_sharpe(_pooled_returns(output, split, candidate, fold_rows)))
        fold_excess.append(
            float((fold_rows["total_return"] - fold_rows["equal_weight_baseline_return"]).mean())
        )
    objective = config["selection"]["objective"]
    mean_excess = float((selected["total_return"] - selected["equal_weight_baseline_return"]).mean())
    worst_drawdown = float(selected["max_drawdown"].min())
    pooled_sharpe = _sharpe(pooled)
    minimum_fold_sharpe = float(min(fold_sharpes)) if fold_sharpes else 0.0
    robust_score = (
        float(objective["pooled_sharpe_weight"]) * pooled_sharpe
        + float(objective["minimum_fold_sharpe_weight"]) * minimum_fold_sharpe
        + float(objective["mean_excess_return_weight"]) * mean_excess
        - float(objective["drawdown_weight"]) * abs(worst_drawdown)
    )
    return {
        "candidate": candidate,
        "semantic": str(selected["semantic"].iloc[0]),
        "run_count": int(len(selected)),
        "pooled_sharpe": pooled_sharpe,
        "mean_run_sharpe": float(selected["sharpe"].mean()),
        "median_run_sharpe": float(selected["sharpe"].median()),
        "minimum_fold_sharpe": minimum_fold_sharpe,
        "mean_return": float(selected["total_return"].mean()),
        "mean_excess_return": mean_excess,
        "positive_run_fraction": float((selected["total_return"] > 0.0).mean()),
        "positive_fold_excess_fraction": float(np.mean(np.asarray(fold_excess) > 0.0)) if fold_excess else 0.0,
        "worst_drawdown": worst_drawdown,
        "mean_trade_count": float(selected["trade_count"].mean()),
        "mean_turnover": float(selected["turnover"].mean()),
        "robust_score": float(robust_score),
        "daily_returns": pooled,
    }


def _leave_one_fold_out(
    config: dict[str, Any],
    output: Path,
    rows: pd.DataFrame,
    candidates: list[str],
    split: str,
) -> tuple[pd.DataFrame, float]:
    records: list[dict[str, Any]] = []
    heldout_series: list[pd.Series] = []
    folds = sorted(rows["fold"].unique())
    if len(folds) < 2:
        ranking = sorted(
            (_candidate_stats(config, output, rows, candidate, split) for candidate in candidates),
            key=lambda item: item["robust_score"],
            reverse=True,
        )
        winner = ranking[0]["candidate"]
        returns = _pooled_returns(output, split, winner, rows.loc[rows["candidate"] == winner])
        return pd.DataFrame(
            [{
                "heldout_fold": folds[0] if folds else "none",
                "selected_candidate": winner,
                "training_robust_score": ranking[0]["robust_score"],
                "heldout_sharpe": _sharpe(returns),
                "heldout_mean_return": float(rows.loc[rows["candidate"] == winner, "total_return"].mean()),
                "heldout_mean_excess_return": float(
                    (
                        rows.loc[rows["candidate"] == winner, "total_return"]
                        - rows.loc[rows["candidate"] == winner, "equal_weight_baseline_return"]
                    ).mean()
                ),
                "diagnostic_only": True,
            }]
        ), _sharpe(returns)
    for fold in folds:
        training = rows.loc[rows["fold"] != fold]
        ranking = sorted(
            (_candidate_stats(config, output, training, candidate, split) for candidate in candidates),
            key=lambda item: item["robust_score"],
            reverse=True,
        )
        winner = ranking[0]["candidate"]
        heldout = rows.loc[(rows["fold"] == fold) & (rows["candidate"] == winner)]
        heldout_returns = _pooled_returns(output, split, winner, heldout)
        heldout_series.append(heldout_returns.rename(fold))
        records.append(
            {
                "heldout_fold": fold,
                "selected_candidate": winner,
                "training_robust_score": ranking[0]["robust_score"],
                "heldout_sharpe": _sharpe(heldout_returns),
                "heldout_mean_return": float(heldout["total_return"].mean()),
                "heldout_mean_excess_return": float(
                    (heldout["total_return"] - heldout["equal_weight_baseline_return"]).mean()
                ),
            }
        )
    pooled = pd.concat(heldout_series, axis=1).mean(axis=1, skipna=True).dropna()
    return pd.DataFrame(records), _sharpe(pooled)


def select_policy(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Select a policy using all-fold statistics and leave-one-fold-out evidence."""
    summary = pd.read_csv(output / "policy_sweep_summary.csv")
    split = str(config["experiment"].get("validation_split", "val"))
    rows = summary.loc[summary["split"] == split].copy()
    candidates = [str(item["id"]) for item in config["policy_candidates"]]
    stats = [_candidate_stats(config, output, rows, candidate, split) for candidate in candidates]
    returns_matrix = pd.concat(
        {item["candidate"]: item.pop("daily_returns") for item in stats}, axis=1
    ).fillna(0.0)
    pbo = probability_of_backtest_overfitting(returns_matrix.to_numpy()) if len(candidates) > 1 else 0.0
    trials = max(len(candidates), 1)
    gates = config["selection"]
    for item in stats:
        values = returns_matrix[item["candidate"]].to_numpy(dtype=float)
        item["deflated_sharpe_probability"] = deflated_sharpe_probability(values, trials)
        item["pbo"] = pbo
        item["individual_gate_pass"] = bool(
            item["pooled_sharpe"] >= float(gates["target_pooled_sharpe"])
            and item["minimum_fold_sharpe"] >= float(gates["minimum_fold_sharpe"])
            and item["mean_excess_return"] >= float(gates["minimum_mean_excess_return"])
            and item["positive_run_fraction"] >= float(gates["minimum_positive_run_fraction"])
            and item["positive_fold_excess_fraction"] >= float(gates["minimum_positive_fold_excess_fraction"])
            and abs(item["worst_drawdown"]) <= float(gates["maximum_worst_drawdown"])
            and item["mean_trade_count"] >= float(gates["minimum_mean_trade_count"])
            and item["deflated_sharpe_probability"] >= float(gates["minimum_deflated_sharpe_probability"])
            and pbo <= float(gates["maximum_pbo"])
        )
    leaderboard = pd.DataFrame(stats).sort_values("robust_score", ascending=False)
    leaderboard.to_csv(output / "policy_leaderboard.csv", index=False)

    oof, oof_sharpe = _leave_one_fold_out(config, output, rows, candidates, split)
    oof.to_csv(output / "cross_market_oof.csv", index=False)
    eligible = leaderboard.loc[leaderboard["individual_gate_pass"]]
    promotion_pass = bool(not eligible.empty and oof_sharpe >= float(gates["target_cross_market_oof_sharpe"]))
    winner_id = str(eligible.iloc[0]["candidate"]) if promotion_pass else str(leaderboard.iloc[0]["candidate"])
    winner = next(item for item in config["policy_candidates"] if item["id"] == winner_id)
    selection = {
        "selected_candidate": winner_id,
        "semantic": winner["semantic"],
        "resolved_policy": _policy_for_candidate(config, winner),
        "candidate": winner,
        "promotion_pass": promotion_pass,
        "confirmation_unlocked": promotion_pass,
        "target_sharpe": float(gates["target_pooled_sharpe"]),
        "cross_market_oof_sharpe": oof_sharpe,
        "oof_selection_counts": dict(Counter(oof["selected_candidate"])),
        "best_validation": _native(leaderboard.iloc[0].to_dict()),
        "failure_reason": None if promotion_pass else "Sharpe-2 cross-market promotion gates were not all satisfied.",
        "smoke_only": bool(config.get("_smoke", False)),
    }
    _write_yaml(output / "selected_policy.yaml", selection)
    _write_selection_report(output / "selection_report.md", selection, leaderboard, oof)
    return selection


def _write_selection_report(
    path: Path,
    selection: dict[str, Any],
    leaderboard: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    columns = [
        "candidate", "pooled_sharpe", "mean_run_sharpe", "minimum_fold_sharpe",
        "mean_return", "mean_excess_return", "worst_drawdown", "individual_gate_pass",
    ]
    lines = [
        "# Phase 4E Cross-Market Selection",
        "",
        f"- Selected challenger: `{selection['selected_candidate']}`",
        f"- Promotion pass: `{selection['promotion_pass']}`",
        f"- Cross-market OOF Sharpe: `{selection['cross_market_oof_sharpe']:.4f}`",
        f"- Sharpe target: `{selection['target_sharpe']:.2f}`",
        "",
        "## Leaderboard",
        "",
        leaderboard[columns].to_markdown(index=False),
        "",
        "## Leave-One-Fold-Out",
        "",
        oof.to_markdown(index=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_confirmation(
    config: dict[str, Any],
    output: Path,
    run_dirs: list[Path],
    selection: dict[str, Any],
    *,
    resume: bool,
) -> None:
    """Open 2024 exactly once, and only after validation promotion succeeds."""
    if not selection.get("confirmation_unlocked", False) or selection.get("smoke_only", False):
        _write_json(
            output / "confirmation_status.json",
            {"executed": False, "reason": selection.get("failure_reason") or "Smoke runs cannot unlock confirmation."},
        )
        return
    semantic = _semantic(config, str(selection["semantic"]))
    confirmation_config = copy.deepcopy(config)
    confirmation_config["variants"] = [semantic]
    splits = [str(value) for value in config["experiment"]["confirmation_splits"]]
    score_semantics(confirmation_config, output, run_dirs, splits, resume=resume)
    candidate = next(item for item in config["policy_candidates"] if item["id"] == selection["selected_candidate"])
    summary = run_policy_sweep(
        config, output, run_dirs, splits, resume=resume, candidates=[candidate]
    )
    report: dict[str, Any] = {}
    for split in splits:
        split_rows = summary.loc[
            (summary["split"] == split) & (summary["candidate"] == selection["selected_candidate"])
        ]
        stats = _candidate_stats(config, output, split_rows, selection["selected_candidate"], split)
        stats.pop("daily_returns", None)
        stats["sharpe_2_reached"] = bool(
            stats["pooled_sharpe"] >= float(config["selection"]["target_pooled_sharpe"])
        )
        report[split] = stats
    _write_json(output / "confirmation_report.json", report)
    _write_json(output / "confirmation_status.json", {"executed": True, "splits": splits})


def run(args: argparse.Namespace) -> Path:
    config_path = _resolve(args.config)
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)
    if args.smoke:
        config["_smoke"] = True
    output = _ensure_output(config, args.run_id, resume=args.resume)
    manifest = audit_sources(config, output)
    source = _resolve(config["experiment"]["source_run"])
    run_dirs = _select_runs(_complete_run_dirs(source), config["experiment"], args.max_runs)
    if args.stage == "audit":
        return output
    validation = [str(config["experiment"].get("validation_split", "val"))]
    if args.stage in {"score", "all"}:
        score_semantics(config, output, run_dirs, validation, resume=args.resume)
    if args.stage in {"sweep", "all"}:
        run_policy_sweep(config, output, run_dirs, validation, resume=args.resume)
    if args.stage in {"select", "all"}:
        selection = select_policy(config, output)
    else:
        selected_path = output / "selected_policy.yaml"
        selection = _read_yaml(selected_path) if selected_path.exists() else None
    if args.stage in {"confirm", "all"}:
        if selection is None:
            raise FileNotFoundError("Run the select stage before confirmation.")
        run_confirmation(config, output, run_dirs, selection, resume=args.resume)
    _write_json(
        output / "run_summary.json",
        {"stage": args.stage, "run_count": len(run_dirs), "manifest": manifest},
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4e_cross_market_sharpe.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    print(run(args))


if __name__ == "__main__":
    main()
