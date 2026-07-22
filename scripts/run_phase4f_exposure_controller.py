"""Evaluate causal exposure control over frozen Phase 4E A2 memory signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase4d_retrieval_semantics import (  # noqa: E402
    _complete_run_dirs,
    _ensure_output,
    _read_yaml,
    _resolve,
    _select_runs,
    _write_json,
    _write_yaml,
)
from run_phase4e_cross_market_sharpe import (  # noqa: E402
    _candidate_stats,
    _leave_one_fold_out,
    _pooled_returns,
    score_semantics,
)
from src.backtest.exposure_controller import (  # noqa: E402
    CausalExposureController,
    ExposureControllerConfig,
)
from src.backtest.market_memory_backtester import (  # noqa: E402
    PolicyConfig,
    compute_backtest_metrics,
    equal_weight_baseline,
    run_long_only_backtest,
)
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
)


LOGGER = logging.getLogger(__name__)
STAGES = ("audit", "sweep", "select", "confirm", "all")


def _controller(config: dict[str, Any], identifier: str) -> dict[str, Any]:
    """Return one predeclared exposure-controller hypothesis."""
    return next(item for item in config["controllers"] if str(item["id"]) == str(identifier))


def _source_policy(config: dict[str, Any]) -> tuple[PolicyConfig, dict[str, Any]]:
    """Load and validate the immutable Phase 4E policy snapshot."""
    source = _resolve(config["experiment"]["source_run"])
    snapshot_path = source / "selected_policy.yaml"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Missing Phase 4E policy snapshot: {snapshot_path}")
    snapshot = _read_yaml(snapshot_path)
    expected = str(config["experiment"]["source_candidate"])
    if str(snapshot.get("selected_candidate")) != expected:
        raise ValueError(
            f"Phase 4F expects source candidate {expected}, found {snapshot.get('selected_candidate')}."
        )
    return PolicyConfig(**snapshot["resolved_policy"]), snapshot


def _signal_paths(
    config: dict[str, Any],
    splits: Iterable[str],
    *,
    max_runs: int | None = None,
) -> list[Path]:
    """Discover frozen semantic signal caches without reading confirmation early."""
    experiment = config["experiment"]
    source = _resolve(experiment["source_run"])
    semantic = str(experiment["source_semantic"])
    active_folds = {str(value) for value in experiment.get("active_folds", [])}
    active_seeds = {int(value) for value in experiment.get("active_seeds", [])}
    found: list[Path] = []
    for split in splits:
        paths = sorted((source / "evaluations" / str(split) / semantic).glob("fold_*/seed_*/eval/signals.parquet"))
        selected: list[Path] = []
        for path in paths:
            fold = path.parents[2].name
            seed = int(path.parents[1].name.removeprefix("seed_"))
            if active_folds and fold not in active_folds:
                continue
            if active_seeds and seed not in active_seeds:
                continue
            selected.append(path)
        if max_runs is not None:
            selected = selected[: int(max_runs)]
        found.extend(selected)
    return found


def audit_sources(config: dict[str, Any], output: Path, *, max_runs: int | None = None) -> dict[str, Any]:
    """Inventory frozen A2 validation caches and preserve their hashes."""
    split = str(config["experiment"].get("validation_split", "val"))
    paths = _signal_paths(config, [split], max_runs=max_runs)
    minimum = int(config["experiment"]["minimum_complete_runs"])
    smoke = bool(config.get("_smoke", False) or max_runs is not None)
    if len(paths) < minimum and not smoke:
        raise ValueError(f"Expected at least {minimum} complete A2 runs, found {len(paths)}.")
    folds = sorted({path.parents[2].name for path in paths})
    if len(folds) < int(config["experiment"]["expected_folds"]) and not smoke:
        raise ValueError(f"Expected {config['experiment']['expected_folds']} folds, found {folds}.")
    manifest = {
        "phase": "4F",
        "source_run": str(_resolve(config["experiment"]["source_run"])),
        "source_candidate": str(config["experiment"]["source_candidate"]),
        "source_semantic": str(config["experiment"]["source_semantic"]),
        "validation_only": True,
        "run_count": len(paths),
        "folds": folds,
        "files": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in paths
        ],
    }
    _write_json(output / "source_manifest.json", manifest)
    return manifest


def run_controller_sweep(
    config: dict[str, Any],
    output: Path,
    signal_paths: Iterable[Path],
    *,
    resume: bool,
    controllers: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Backtest predeclared controllers over cached A2 signals."""
    policy, policy_snapshot = _source_policy(config)
    selected = controllers or config["controllers"]
    rows: list[dict[str, Any]] = []
    for controller_spec in selected:
        identifier = str(controller_spec["id"])
        controller_cfg = ExposureControllerConfig(**controller_spec["config"])
        for signal_path in signal_paths:
            split = signal_path.parents[4].name
            fold = signal_path.parents[2].name
            seed_name = signal_path.parents[1].name
            seed = int(seed_name.removeprefix("seed_"))
            result_dir = output / "policies" / split / identifier / fold / seed_name / "eval"
            metrics_path = result_dir / "metrics.json"
            if resume and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                signals = pd.read_parquet(signal_path)
                decisions: list[dict[str, Any]] = []
                exposure_decisions: list[dict[str, Any]] = []
                controller = CausalExposureController(controller_cfg) if controller_cfg.enabled else None
                trades, equity = run_long_only_backtest(
                    signals,
                    policy,
                    decision_log=decisions,
                    exposure_controller=controller,
                    exposure_log=exposure_decisions,
                )
                metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
                metrics.update(equal_weight_baseline(signals, policy.initial_capital))
                metrics.update(_controller_metrics(pd.DataFrame(exposure_decisions), equity))
                result_dir.mkdir(parents=True, exist_ok=True)
                trades.to_csv(result_dir / "trades.csv", index=False)
                equity.to_csv(result_dir / "equity_curve.csv", index=False)
                pd.DataFrame(decisions).to_csv(result_dir / "decisions.csv", index=False)
                pd.DataFrame(exposure_decisions).to_csv(
                    result_dir / "exposure_decisions.csv", index=False
                )
                _write_json(metrics_path, metrics)
                _write_yaml(
                    result_dir / "controller_snapshot.yaml",
                    {
                        "controller": controller_spec,
                        "resolved_controller": controller_spec["config"],
                        "source_policy": policy_snapshot,
                        "signal_source": str(signal_path),
                    },
                )
            rows.append(
                {
                    "split": split,
                    "candidate": identifier,
                    "semantic": str(config["experiment"]["source_semantic"]),
                    "fold": fold,
                    "seed": seed,
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
                            "average_trade_return",
                            "turnover",
                            "exposure",
                            "capital_efficiency",
                            "equal_weight_baseline_return",
                            "mean_target_exposure",
                            "derisked_session_fraction",
                            "mean_realized_annualized_volatility",
                            "mean_evidence_scalar",
                            "mean_drawdown_scalar",
                            "mean_volatility_scalar",
                            "rebalance_count",
                        )
                    },
                }
            )
    summary_path = output / "controller_sweep_summary.csv"
    current = pd.DataFrame(rows)
    if summary_path.exists():
        current = pd.concat([pd.read_csv(summary_path), current], ignore_index=True)
        current = current.drop_duplicates(["split", "candidate", "fold", "seed"], keep="last")
    current.to_csv(summary_path, index=False)
    return current


def _controller_metrics(decisions: pd.DataFrame, equity: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "mean_target_exposure": 1.0,
            "derisked_session_fraction": 0.0,
            "mean_realized_annualized_volatility": None,
            "mean_evidence_scalar": 1.0,
            "mean_drawdown_scalar": 1.0,
            "mean_volatility_scalar": 1.0,
        }
    target = pd.to_numeric(decisions["target_exposure"], errors="coerce")
    realized = pd.to_numeric(decisions["realized_annualized_volatility"], errors="coerce")
    return {
        "mean_target_exposure": float(target.mean()),
        "derisked_session_fraction": float((target < 0.999).mean()),
        "mean_realized_annualized_volatility": float(realized.mean()) if realized.notna().any() else None,
        "mean_evidence_scalar": float(pd.to_numeric(decisions["evidence_scalar"], errors="coerce").mean()),
        "mean_drawdown_scalar": float(pd.to_numeric(decisions["drawdown_scalar"], errors="coerce").mean()),
        "mean_volatility_scalar": float(pd.to_numeric(decisions["volatility_scalar"], errors="coerce").mean()),
        "mean_realized_exposure": float(equity["exposure"].mean()) if not equity.empty else 0.0,
    }


def select_controller(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Select a controller with cross-market and multiple-testing controls."""
    summary = pd.read_csv(output / "controller_sweep_summary.csv")
    split = str(config["experiment"].get("validation_split", "val"))
    rows = summary.loc[summary["split"] == split].copy()
    candidates = [str(item["id"]) for item in config["controllers"]]
    stats = [_candidate_stats(config, output, rows, candidate, split) for candidate in candidates]
    returns_matrix = pd.concat(
        {item["candidate"]: item.pop("daily_returns") for item in stats}, axis=1
    ).fillna(0.0)
    pbo = probability_of_backtest_overfitting(returns_matrix.to_numpy()) if len(candidates) > 1 else 0.0
    gates = config["selection"]
    baseline = next(item for item in stats if item["candidate"] == gates["baseline_candidate"])
    for item in stats:
        values = returns_matrix[item["candidate"]].to_numpy(dtype=float)
        item["sharpe_improvement_over_baseline"] = item["pooled_sharpe"] - baseline["pooled_sharpe"]
        item["deflated_sharpe_probability"] = deflated_sharpe_probability(values, len(candidates))
        item["pbo"] = pbo
        item["individual_gate_pass"] = bool(
            item["candidate"] != gates["baseline_candidate"]
            and item["pooled_sharpe"] >= float(gates["target_pooled_sharpe"])
            and item["sharpe_improvement_over_baseline"]
            >= float(gates["minimum_sharpe_improvement_over_baseline"])
            and item["minimum_fold_sharpe"] >= float(gates["minimum_fold_sharpe"])
            and item["mean_excess_return"] >= float(gates["minimum_mean_excess_return"])
            and item["positive_run_fraction"] >= float(gates["minimum_positive_run_fraction"])
            and item["positive_fold_excess_fraction"]
            >= float(gates["minimum_positive_fold_excess_fraction"])
            and abs(item["worst_drawdown"]) <= float(gates["maximum_worst_drawdown"])
            and item["mean_trade_count"] >= float(gates["minimum_mean_trade_count"])
            and item["deflated_sharpe_probability"]
            >= float(gates["minimum_deflated_sharpe_probability"])
            and pbo <= float(gates["maximum_pbo"])
        )
    leaderboard = pd.DataFrame(stats).sort_values("robust_score", ascending=False)
    leaderboard.to_csv(output / "controller_leaderboard.csv", index=False)
    oof, oof_sharpe = _leave_one_fold_out(config, output, rows, candidates, split)
    oof.to_csv(output / "cross_market_oof.csv", index=False)
    eligible = leaderboard.loc[leaderboard["individual_gate_pass"]]
    promotion = bool(
        not eligible.empty
        and oof_sharpe >= float(gates["target_cross_market_oof_sharpe"])
        and not config.get("_smoke", False)
    )
    winner_id = str(eligible.iloc[0]["candidate"]) if promotion else str(leaderboard.iloc[0]["candidate"])
    winner = _controller(config, winner_id)
    selection = {
        "selected_candidate": winner_id,
        "controller": winner,
        "promotion_pass": promotion,
        "confirmation_unlocked": promotion,
        "target_sharpe": float(gates["target_pooled_sharpe"]),
        "cross_market_oof_sharpe": oof_sharpe,
        "oof_selection_counts": dict(Counter(oof["selected_candidate"])),
        "best_validation": _native(leaderboard.iloc[0].to_dict()),
        "failure_reason": None if promotion else "Phase 4F cross-market promotion gates were not all satisfied.",
        "smoke_only": bool(config.get("_smoke", False)),
    }
    _write_yaml(output / "selected_controller.yaml", selection)
    _write_selection_report(output / "selection_report.md", selection, leaderboard, oof)
    return selection


def _write_selection_report(
    path: Path,
    selection: dict[str, Any],
    leaderboard: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    columns = [
        "candidate",
        "pooled_sharpe",
        "sharpe_improvement_over_baseline",
        "minimum_fold_sharpe",
        "mean_return",
        "mean_excess_return",
        "worst_drawdown",
        "deflated_sharpe_probability",
        "pbo",
        "individual_gate_pass",
    ]
    lines = [
        "# Phase 4F Causal Exposure Selection",
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
    selection: dict[str, Any],
    *,
    resume: bool,
    max_runs: int | None = None,
) -> None:
    """Open confirmation only after every validation promotion gate passes."""
    if not selection.get("confirmation_unlocked", False) or selection.get("smoke_only", False):
        _write_json(
            output / "confirmation_status.json",
            {"executed": False, "reason": selection.get("failure_reason") or "Smoke runs cannot confirm."},
        )
        return
    splits = [str(value) for value in config["experiment"]["confirmation_splits"]]
    paths = _signal_paths(config, splits, max_runs=max_runs)
    if not paths:
        _materialize_confirmation_signals(config, splits)
        paths = _signal_paths(config, splits, max_runs=max_runs)
    if not paths:
        raise FileNotFoundError("Phase 4E confirmation signals could not be materialized.")
    winner = _controller(config, str(selection["selected_candidate"]))
    summary = run_controller_sweep(config, output, paths, resume=resume, controllers=[winner])
    report: dict[str, Any] = {}
    for split in splits:
        split_rows = summary.loc[
            (summary["split"] == split) & (summary["candidate"] == selection["selected_candidate"])
        ]
        if split_rows.empty:
            continue
        stats = _candidate_stats(config, output, split_rows, selection["selected_candidate"], split)
        stats.pop("daily_returns", None)
        stats["sharpe_2_reached"] = bool(
            stats["pooled_sharpe"] >= float(config["selection"]["target_pooled_sharpe"])
        )
        report[split] = stats
    _write_json(output / "confirmation_report.json", report)
    _write_json(output / "confirmation_status.json", {"executed": True, "splits": list(report)})


def _materialize_confirmation_signals(config: dict[str, Any], splits: list[str]) -> None:
    phase4e_config_path = _resolve(config["experiment"]["source_config"])
    phase4e_config = _read_yaml(phase4e_config_path)
    phase4e_config["_config_path"] = str(phase4e_config_path)
    source_output = _resolve(config["experiment"]["source_run"])
    source_runs = _resolve(phase4e_config["experiment"]["source_run"])
    run_dirs = _select_runs(
        _complete_run_dirs(source_runs),
        phase4e_config["experiment"],
        None,
    )
    keep = {"A0_legacy_identity", str(config["experiment"]["source_semantic"])}
    phase4e_config["variants"] = [
        item for item in phase4e_config["variants"] if str(item["id"]) in keep
    ]
    LOGGER.info("Materializing frozen confirmation signals for %s", ", ".join(splits))
    score_semantics(phase4e_config, source_output, run_dirs, splits, resume=True)


def run(args: argparse.Namespace) -> Path:
    config_path = _resolve(args.config)
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)
    if args.smoke:
        config["_smoke"] = True
    output = _ensure_output(config, args.run_id, resume=args.resume)
    manifest = audit_sources(config, output, max_runs=args.max_runs)
    validation = [str(config["experiment"].get("validation_split", "val"))]
    paths = _signal_paths(config, validation, max_runs=args.max_runs)
    if args.stage == "audit":
        return output
    if args.stage in {"sweep", "all"}:
        run_controller_sweep(config, output, paths, resume=args.resume)
    if args.stage in {"select", "all"}:
        selection = select_controller(config, output)
    else:
        selected_path = output / "selected_controller.yaml"
        selection = _read_yaml(selected_path) if selected_path.exists() else None
    if args.stage in {"confirm", "all"}:
        if selection is None:
            raise FileNotFoundError("Run the select stage before confirmation.")
        run_confirmation(
            config,
            output,
            selection,
            resume=args.resume,
            max_runs=args.max_runs,
        )
    _write_json(
        output / "run_summary.json",
        {"stage": args.stage, "run_count": len(paths), "manifest": manifest},
    )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4f_exposure_controller.yaml")
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
