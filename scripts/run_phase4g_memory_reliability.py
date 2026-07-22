"""Test when frozen A2 market-memory evidence transfers reliably."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase4d_retrieval_semantics import (  # noqa: E402
    _complete_run_dirs,
    _ensure_output,
    _read_yaml,
    _resolve,
    _run_key,
    _select_runs,
    _write_json,
    _write_yaml,
)
from run_phase4e_cross_market_sharpe import _sharpe, score_semantics  # noqa: E402
from src.backtest.market_memory_backtester import (  # noqa: E402
    PolicyConfig,
    compute_backtest_metrics,
    equal_weight_baseline,
    run_long_only_backtest,
)
from src.eval.memory_reliability import (  # noqa: E402
    ReliabilityConfig,
    build_consensus_frame,
    domain_shift_auc,
    fit_reliability_model,
    reliability_metrics,
    risk_coverage_table,
    univariate_reliability_table,
)
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
)


LOGGER = logging.getLogger(__name__)
STAGES = ("audit", "materialize", "build", "evaluate", "trade", "select", "confirm", "all")


def _phase4e_config(config: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(config["experiment"]["source_config"])
    source = _read_yaml(path)
    source["_config_path"] = str(path)
    keep = {"A0_legacy_identity", str(config["experiment"]["source_semantic"])}
    source["variants"] = [item for item in source["variants"] if str(item["id"]) in keep]
    if config.get("_smoke", False):
        source["_smoke"] = True
    return source


def _source_runs(config: dict[str, Any], max_runs: int | None) -> list[Path]:
    experiment = config["experiment"]
    runs = _complete_run_dirs(_resolve(experiment["source_phase4c_run"]))
    return _select_runs(runs, experiment, max_runs)


def audit_sources(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    *,
    limited: bool,
) -> dict[str, Any]:
    """Verify frozen validation signals and training latents before fitting reliability."""
    source = _resolve(config["experiment"]["source_run"])
    semantic = str(config["experiment"]["source_semantic"])
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for run_dir in run_dirs:
        fold, seed = _run_key(run_dir)
        train = run_dir / "latents" / "train_latents.parquet"
        validation = source / "evaluations" / "val" / semantic / fold / f"seed_{seed}" / "eval" / "signals.parquet"
        neighbors = source / "evaluations" / "val" / "A0_legacy_identity" / fold / f"seed_{seed}" / "eval" / "neighbors.parquet"
        for path in (train, validation, neighbors):
            if not path.exists():
                missing.append(str(path))
        records.append({"fold": fold, "seed": seed, "train_latents": str(train), "validation_signals": str(validation)})
    minimum = int(config["experiment"]["minimum_complete_runs"])
    if missing:
        raise FileNotFoundError("Missing Phase 4G sources:\n" + "\n".join(missing))
    if len(records) < minimum and not limited:
        raise ValueError(f"Expected at least {minimum} complete runs, found {len(records)}.")
    manifest = {
        "phase": "4G",
        "source_run": str(source),
        "training_period": "2021-2022 frozen train latents",
        "validation_period": "2023 frozen A2 signals",
        "validation_only": True,
        "run_count": len(records),
        "runs": records,
    }
    _write_json(output / "source_manifest.json", manifest)
    return manifest


def materialize_training_signals(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    *,
    resume: bool,
) -> None:
    """Causally score 2021-2022 query states against outcome-matured earlier memory."""
    score_semantics(
        _phase4e_config(config),
        output,
        run_dirs,
        [str(config["experiment"]["training_split"])],
        resume=resume,
    )


def build_consensus_datasets(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    splits: Iterable[str],
) -> None:
    """Build one state row per fold, ticker, and timestamp across random seeds."""
    experiment = config["experiment"]
    source = _resolve(experiment["source_run"])
    semantic = str(experiment["source_semantic"])
    by_fold: dict[str, list[tuple[int, Path]]] = {}
    for run_dir in run_dirs:
        fold, seed = _run_key(run_dir)
        by_fold.setdefault(fold, []).append((seed, run_dir))
    for split in splits:
        root = output if split != experiment["validation_split"] else source
        for fold, members in sorted(by_fold.items()):
            seed_frames: dict[int, pd.DataFrame] = {}
            neighbor_frames: dict[int, pd.DataFrame] = {}
            for seed, _ in members:
                eval_root = root / "evaluations" / str(split)
                signal_path = eval_root / semantic / fold / f"seed_{seed}" / "eval" / "signals.parquet"
                neighbor_path = eval_root / "A0_legacy_identity" / fold / f"seed_{seed}" / "eval" / "neighbors.parquet"
                if not signal_path.exists() or not neighbor_path.exists():
                    raise FileNotFoundError(f"Missing consensus source for {split} {fold} seed {seed}.")
                seed_frames[seed] = pd.read_parquet(signal_path)
                neighbor_frames[seed] = pd.read_parquet(neighbor_path)
            consensus = build_consensus_frame(
                seed_frames,
                neighbor_frames,
                useful_alpha_after_costs=float(config["reliability"]["useful_alpha_after_costs"]),
            )
            consensus["fold"] = fold
            path = output / "datasets" / str(split) / f"{fold}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            consensus.to_parquet(path, index=False)


def _reliability_config(config: dict[str, Any]) -> ReliabilityConfig:
    values = copy.deepcopy(config["reliability"])
    if config.get("_smoke", False):
        values.update(
            {
                "fit_fraction": 0.50,
                "calibration_embargo_sessions": 2,
                "minimum_fit_rows": 50,
                "minimum_calibration_rows": 20,
            }
        )
    return ReliabilityConfig(**values)


def _merge_csv(path: Path, incoming: pd.DataFrame, keys: list[str]) -> None:
    """Persist keyed results without discarding rows produced by earlier stages."""
    if path.exists():
        existing = pd.read_csv(path)
        incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    incoming = incoming.drop_duplicates(subset=keys, keep="last")
    incoming = incoming.sort_values(keys, kind="stable").reset_index(drop=True)
    incoming.to_csv(path, index=False)


def evaluate_reliability(
    config: dict[str, Any],
    output: Path,
    splits: Iterable[str],
) -> None:
    """Fit on training history and evaluate transfer on later frozen splits."""
    reliability_cfg = _reliability_config(config)
    metric_rows: list[dict[str, Any]] = []
    risk_rows: list[pd.DataFrame] = []
    univariate_rows: list[pd.DataFrame] = []
    train_paths = sorted((output / "datasets" / "train").glob("fold_*.parquet"))
    for train_path in train_paths:
        fold = train_path.stem
        train = pd.read_parquet(train_path)
        model = fit_reliability_model(train, reliability_cfg)
        model_dir = output / "models" / fold
        _write_json(model_dir / "model_audit.json", model.to_dict())
        for split in splits:
            query_path = output / "datasets" / str(split) / f"{fold}.parquet"
            if not query_path.exists():
                continue
            query = pd.read_parquet(query_path)
            predictions = model.predict(query)
            prediction_path = output / "predictions" / str(split) / f"{fold}.parquet"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(prediction_path, index=False)
            metrics = reliability_metrics(predictions, model)
            metrics["domain_shift_auc"] = domain_shift_auc(
                train,
                query,
                random_seed=reliability_cfg.random_seed,
            )
            metric_rows.append({"split": str(split), "fold": fold, **metrics})
            risk = risk_coverage_table(predictions, model)
            risk.insert(0, "fold", fold)
            risk.insert(0, "split", str(split))
            risk_rows.append(risk)
            if split == config["experiment"]["validation_split"]:
                table = univariate_reliability_table(train, predictions)
                table.insert(0, "fold", fold)
                univariate_rows.append(table)
    if metric_rows:
        _merge_csv(
            output / "reliability_metrics.csv",
            pd.DataFrame(metric_rows),
            ["split", "fold"],
        )
    if risk_rows:
        _merge_csv(
            output / "risk_coverage.csv",
            pd.concat(risk_rows, ignore_index=True),
            ["split", "fold", "nominal_coverage"],
        )
    if univariate_rows:
        _merge_csv(
            output / "univariate_reliability.csv",
            pd.concat(univariate_rows, ignore_index=True),
            ["fold", "feature", "bin"],
        )


def _source_policy(config: dict[str, Any]) -> PolicyConfig:
    snapshot = _read_yaml(_resolve(config["experiment"]["source_run"]) / "selected_policy.yaml")
    expected = str(config["experiment"]["source_candidate"])
    if str(snapshot.get("selected_candidate")) != expected:
        raise ValueError(f"Expected Phase 4E source policy {expected}.")
    return PolicyConfig(**snapshot["resolved_policy"])


def run_coverage_backtests(
    config: dict[str, Any],
    output: Path,
    splits: Iterable[str],
    *,
    resume: bool,
    candidates: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Trade only calibration-approved states at fixed nominal coverage levels."""
    policy = _source_policy(config)
    selected = candidates or config["coverage_candidates"]
    rows: list[dict[str, Any]] = []
    for split in splits:
        for prediction_path in sorted((output / "predictions" / str(split)).glob("fold_*.parquet")):
            fold = prediction_path.stem
            predictions = pd.read_parquet(prediction_path)
            audit = json.loads((output / "models" / fold / "model_audit.json").read_text(encoding="utf-8"))
            thresholds = {float(key): float(value) for key, value in audit["calibration_thresholds"].items()}
            for candidate in selected:
                identifier = str(candidate["id"])
                nominal = float(candidate["nominal_coverage"])
                result_dir = output / "policies" / str(split) / identifier / fold / "eval"
                metrics_path = result_dir / "metrics.json"
                if resume and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    threshold = thresholds[nominal]
                    accepted = predictions["reliability_probability"] >= threshold
                    signals = predictions.copy()
                    signals.loc[~accepted, "opportunity_score"] = -np.inf
                    decisions: list[dict[str, Any]] = []
                    trades, equity = run_long_only_backtest(signals, policy, decision_log=decisions)
                    metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
                    metrics.update(equal_weight_baseline(signals, policy.initial_capital))
                    metrics.update(
                        {
                            "nominal_coverage": nominal,
                            "realized_coverage": float(accepted.mean()),
                            "reliability_threshold": threshold,
                            "mean_accepted_reliability": float(
                                predictions.loc[accepted, "reliability_probability"].mean()
                            )
                            if accepted.any()
                            else None,
                        }
                    )
                    result_dir.mkdir(parents=True, exist_ok=True)
                    trades.to_csv(result_dir / "trades.csv", index=False)
                    equity.to_csv(result_dir / "equity_curve.csv", index=False)
                    pd.DataFrame(decisions).to_csv(result_dir / "decisions.csv", index=False)
                    _write_json(metrics_path, metrics)
                    _write_yaml(
                        result_dir / "policy_snapshot.yaml",
                        {"candidate": candidate, "source_policy": str(config["experiment"]["source_candidate"])},
                    )
                rows.append(
                    {
                        "split": str(split),
                        "candidate": identifier,
                        "fold": fold,
                        **{
                            key: metrics.get(key)
                            for key in (
                                "total_return",
                                "annualized_return",
                                "sharpe",
                                "max_drawdown",
                                "profit_factor",
                                "trade_count",
                                "win_rate",
                                "turnover",
                                "exposure",
                                "capital_efficiency",
                                "equal_weight_baseline_return",
                                "nominal_coverage",
                                "realized_coverage",
                                "mean_accepted_reliability",
                            )
                        },
                    }
                )
    summary_path = output / "coverage_backtest_summary.csv"
    current = pd.DataFrame(rows)
    if summary_path.exists():
        current = pd.concat([pd.read_csv(summary_path), current], ignore_index=True)
        current = current.drop_duplicates(["split", "candidate", "fold"], keep="last")
    current.to_csv(summary_path, index=False)
    return current


def _equity_returns(output: Path, split: str, candidate: str, fold: str) -> pd.Series:
    path = output / "policies" / split / candidate / fold / "eval" / "equity_curve.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().dropna()


def _candidate_stats(
    config: dict[str, Any],
    output: Path,
    rows: pd.DataFrame,
    candidate: str,
    split: str,
) -> dict[str, Any]:
    selected = rows.loc[rows["candidate"] == candidate]
    series = [_equity_returns(output, split, candidate, fold).rename(fold) for fold in selected["fold"]]
    pooled = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
    fold_sharpes = [_sharpe(item) for item in series]
    fold_excess = selected["total_return"] - selected["equal_weight_baseline_return"]
    pooled_sharpe = _sharpe(pooled)
    minimum_fold = float(min(fold_sharpes)) if fold_sharpes else 0.0
    mean_excess = float(fold_excess.mean())
    worst_drawdown = float(selected["max_drawdown"].min())
    objective = config["selection"]["objective"]
    robust_score = (
        float(objective["pooled_sharpe_weight"]) * pooled_sharpe
        + float(objective["minimum_fold_sharpe_weight"]) * minimum_fold
        + float(objective["mean_excess_return_weight"]) * mean_excess
        - float(objective["drawdown_weight"]) * abs(worst_drawdown)
    )
    return {
        "candidate": candidate,
        "fold_count": int(len(selected)),
        "pooled_sharpe": pooled_sharpe,
        "minimum_fold_sharpe": minimum_fold,
        "mean_return": float(selected["total_return"].mean()),
        "mean_excess_return": mean_excess,
        "positive_fold_excess_fraction": float((fold_excess > 0.0).mean()),
        "worst_drawdown": worst_drawdown,
        "mean_trade_count": float(selected["trade_count"].mean()),
        "mean_realized_coverage": float(selected["realized_coverage"].mean()),
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
    heldout_returns: list[pd.Series] = []
    folds = sorted(rows["fold"].unique())
    if len(folds) < 2:
        stats = sorted(
            (_candidate_stats(config, output, rows, candidate, split) for candidate in candidates),
            key=lambda item: item["robust_score"],
            reverse=True,
        )[0]
        return pd.DataFrame([{"heldout_fold": folds[0], "selected_candidate": stats["candidate"], "diagnostic_only": True}]), stats["pooled_sharpe"]
    for fold in folds:
        training = rows.loc[rows["fold"] != fold]
        ranking = sorted(
            (_candidate_stats(config, output, training, candidate, split) for candidate in candidates),
            key=lambda item: item["robust_score"],
            reverse=True,
        )
        winner = ranking[0]["candidate"]
        heldout = rows.loc[(rows["fold"] == fold) & (rows["candidate"] == winner)].iloc[0]
        returns = _equity_returns(output, split, winner, fold)
        heldout_returns.append(returns.rename(fold))
        records.append(
            {
                "heldout_fold": fold,
                "selected_candidate": winner,
                "training_robust_score": ranking[0]["robust_score"],
                "heldout_sharpe": _sharpe(returns),
                "heldout_return": float(heldout["total_return"]),
                "heldout_excess_return": float(
                    heldout["total_return"] - heldout["equal_weight_baseline_return"]
                ),
            }
        )
    pooled = pd.concat(heldout_returns, axis=1).mean(axis=1, skipna=True).dropna()
    return pd.DataFrame(records), _sharpe(pooled)


def select_reliability_policy(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Apply diagnostic, trading, cross-market, DSR, and PBO promotion gates."""
    split = str(config["experiment"]["validation_split"])
    summary = pd.read_csv(output / "coverage_backtest_summary.csv")
    rows = summary.loc[summary["split"] == split].copy()
    candidates = [str(item["id"]) for item in config["coverage_candidates"]]
    stats = [_candidate_stats(config, output, rows, candidate, split) for candidate in candidates]
    returns_matrix = pd.concat(
        {item["candidate"]: item.pop("daily_returns") for item in stats}, axis=1
    ).fillna(0.0)
    pbo = probability_of_backtest_overfitting(returns_matrix.to_numpy())
    reliability = pd.read_csv(output / "reliability_metrics.csv")
    reliability = reliability.loc[reliability["split"] == split]
    positive_brier_fraction = float((reliability["brier_skill"] > 0.0).mean())
    positive_spread_fraction = float((reliability["top_bottom_alpha_spread"] > 0.0).mean())
    gates = config["selection"]
    baseline = str(gates["baseline_candidate"])
    for item in stats:
        values = returns_matrix[item["candidate"]].to_numpy(dtype=float)
        item["deflated_sharpe_probability"] = deflated_sharpe_probability(values, len(candidates))
        item["pbo"] = pbo
        item["positive_brier_skill_fraction"] = positive_brier_fraction
        item["positive_top_bottom_spread_fraction"] = positive_spread_fraction
        item["individual_gate_pass"] = bool(
            item["candidate"] != baseline
            and item["pooled_sharpe"] >= float(gates["target_pooled_sharpe"])
            and item["mean_excess_return"] >= float(gates["minimum_mean_excess_return"])
            and item["positive_fold_excess_fraction"]
            >= float(gates["minimum_positive_fold_excess_fraction"])
            and abs(item["worst_drawdown"]) <= float(gates["maximum_worst_drawdown"])
            and item["mean_trade_count"] >= float(gates["minimum_mean_trade_count"])
            and item["deflated_sharpe_probability"]
            >= float(gates["minimum_deflated_sharpe_probability"])
            and pbo <= float(gates["maximum_pbo"])
            and positive_brier_fraction >= float(gates["minimum_positive_brier_skill_fraction"])
            and positive_spread_fraction
            >= float(gates["minimum_positive_top_bottom_spread_fraction"])
        )
    leaderboard = pd.DataFrame(stats).sort_values("robust_score", ascending=False)
    leaderboard.to_csv(output / "reliability_policy_leaderboard.csv", index=False)
    oof, oof_sharpe = _leave_one_fold_out(config, output, rows, candidates, split)
    oof.to_csv(output / "cross_market_oof.csv", index=False)
    eligible = leaderboard.loc[leaderboard["individual_gate_pass"]]
    promotion = bool(
        not eligible.empty
        and oof_sharpe >= float(gates["target_cross_market_oof_sharpe"])
        and not config.get("_smoke", False)
    )
    winner_id = str(eligible.iloc[0]["candidate"]) if promotion else str(leaderboard.iloc[0]["candidate"])
    winner = next(item for item in config["coverage_candidates"] if item["id"] == winner_id)
    selection = {
        "selected_candidate": winner_id,
        "candidate": winner,
        "promotion_pass": promotion,
        "confirmation_unlocked": promotion,
        "cross_market_oof_sharpe": oof_sharpe,
        "oof_selection_counts": dict(Counter(oof["selected_candidate"])),
        "positive_brier_skill_fraction": positive_brier_fraction,
        "positive_top_bottom_spread_fraction": positive_spread_fraction,
        "best_validation": _native(leaderboard.iloc[0].to_dict()),
        "failure_reason": None if promotion else "Phase 4G transfer-reliability gates were not all satisfied.",
        "smoke_only": bool(config.get("_smoke", False)),
    }
    _write_yaml(output / "selected_reliability_policy.yaml", selection)
    _write_selection_report(output / "selection_report.md", selection, leaderboard, oof, reliability)
    return selection


def _write_selection_report(
    path: Path,
    selection: dict[str, Any],
    leaderboard: pd.DataFrame,
    oof: pd.DataFrame,
    reliability: pd.DataFrame,
) -> None:
    columns = [
        "candidate",
        "pooled_sharpe",
        "minimum_fold_sharpe",
        "mean_return",
        "mean_excess_return",
        "positive_fold_excess_fraction",
        "worst_drawdown",
        "mean_realized_coverage",
        "individual_gate_pass",
    ]
    metric_columns = [
        "fold",
        "brier_skill",
        "roc_auc",
        "reliability_alpha_spearman",
        "top_bottom_alpha_spread",
        "interval_coverage",
        "domain_shift_auc",
    ]
    lines = [
        "# Phase 4G Memory Transfer Reliability",
        "",
        f"- Selected policy: `{selection['selected_candidate']}`",
        f"- Promotion pass: `{selection['promotion_pass']}`",
        f"- Cross-market OOF Sharpe: `{selection['cross_market_oof_sharpe']:.4f}`",
        f"- Positive Brier-skill folds: `{selection['positive_brier_skill_fraction']:.1%}`",
        f"- Positive top-bottom alpha-spread folds: `{selection['positive_top_bottom_spread_fraction']:.1%}`",
        "",
        "## Reliability Diagnostics",
        "",
        reliability[metric_columns].to_markdown(index=False),
        "",
        "## Trading Leaderboard",
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
    """Keep sealed splits inaccessible unless validation promotion succeeds."""
    if not selection.get("confirmation_unlocked", False) or selection.get("smoke_only", False):
        _write_json(
            output / "confirmation_status.json",
            {"executed": False, "reason": selection.get("failure_reason") or "Smoke cannot confirm."},
        )
        return
    splits = [str(value) for value in config["experiment"]["confirmation_splits"]]
    score_semantics(_phase4e_config(config), output, run_dirs, splits, resume=resume)
    build_consensus_datasets(config, output, run_dirs, splits)
    evaluate_reliability(config, output, splits)
    winner = next(
        item for item in config["coverage_candidates"] if item["id"] == selection["selected_candidate"]
    )
    summary = run_coverage_backtests(config, output, splits, resume=resume, candidates=[winner])
    report = {
        split: _native(
            summary.loc[
                (summary["split"] == split) & (summary["candidate"] == selection["selected_candidate"])
            ].to_dict(orient="records")
        )
        for split in splits
    }
    _write_json(output / "confirmation_report.json", report)
    _write_json(output / "confirmation_status.json", {"executed": True, "splits": splits})


def run(args: argparse.Namespace) -> Path:
    config_path = _resolve(args.config)
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)
    if args.smoke:
        config["_smoke"] = True
    output = _ensure_output(config, args.run_id, resume=args.resume)
    run_dirs = _source_runs(config, args.max_runs)
    manifest = audit_sources(
        config,
        output,
        run_dirs,
        limited=bool(args.smoke or args.max_runs is not None),
    )
    if args.stage == "audit":
        return output
    if args.stage in {"materialize", "all"}:
        materialize_training_signals(config, output, run_dirs, resume=args.resume)
    validation = [str(config["experiment"]["validation_split"])]
    if args.stage in {"build", "all"}:
        build_consensus_datasets(config, output, run_dirs, ["train", *validation])
    if args.stage in {"evaluate", "all"}:
        evaluate_reliability(config, output, validation)
    if args.stage in {"trade", "all"}:
        run_coverage_backtests(config, output, validation, resume=args.resume)
    if args.stage in {"select", "all"}:
        selection = select_reliability_policy(config, output)
    else:
        selected_path = output / "selected_reliability_policy.yaml"
        selection = _read_yaml(selected_path) if selected_path.exists() else None
    if args.stage in {"confirm", "all"}:
        if selection is None:
            raise FileNotFoundError("Run select before confirmation.")
        run_confirmation(config, output, run_dirs, selection, resume=args.resume)
    _write_json(
        output / "run_summary.json",
        {"stage": args.stage, "run_count": len(run_dirs), "manifest": manifest},
    )
    return output


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4g_memory_reliability.yaml")
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
