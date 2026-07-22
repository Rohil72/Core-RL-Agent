"""Evaluate causal prototype-memory entry policies on frozen Phase 4C latents."""

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
    _calibrate_distances,
    _complete_run_dirs,
    _ensure_output,
    _prepared_paths,
    _read_yaml,
    _resolve,
    _run_key,
    _select_runs,
    _write_json,
    _write_yaml,
    audit_sources as audit_phase4d_sources,
)
from run_phase4e_cross_market_sharpe import _sharpe  # noqa: E402
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
from src.memory import (  # noqa: E402
    MarketMemoryConfig,
    RallyPrototypeConfig,
    RallyPrototypeSet,
    RelativeOutcomeConfig,
    build_rally_prototypes,
    score_market_memory,
    score_rally_prototype_membership,
)


LOGGER = logging.getLogger(__name__)
STAGES = ("audit", "materialize", "score", "trade", "select", "confirm", "all")


def _source_policy(config: dict[str, Any]) -> PolicyConfig:
    snapshot = _read_yaml(_resolve(config["experiment"]["source_phase4e_run"]) / "selected_policy.yaml")
    expected = str(config["experiment"]["source_candidate"])
    if str(snapshot.get("selected_candidate")) != expected:
        raise ValueError(f"Expected frozen Phase 4E source candidate {expected!r}.")
    return PolicyConfig(**snapshot["resolved_policy"])


def _relative_config(config: dict[str, Any]) -> RelativeOutcomeConfig:
    values = {key: value for key, value in config["relative_outcomes"].items() if key != "enabled"}
    return RelativeOutcomeConfig(**values)


def _prototype_config(config: dict[str, Any]) -> RallyPrototypeConfig:
    values = copy.deepcopy(config["rally_prototypes"])
    if config.get("_smoke", False):
        values.update({"k": 8, "minimum_neighbors": 2, "retrieval_batch_size": 64})
    return RallyPrototypeConfig(**values)


def _memory_config(config: dict[str, Any], calibration: dict[str, Any]) -> MarketMemoryConfig:
    values = copy.deepcopy(config["memory_defaults"])
    values.update(
        {
            "target_alpha": "future_blended_alpha_63",
            "score_mode": "alpha_lcb",
            "confidence_reference_distance": float(calibration["confidence_reference_distance"]),
            "max_median_distance": float(calibration["max_median_distance"]),
        }
    )
    return MarketMemoryConfig(**values)


def audit_sources(config: dict[str, Any], output: Path, run_dirs: Iterable[Path]) -> dict[str, Any]:
    """Record frozen-source provenance and validate the rally memory contract."""
    audit_config = copy.deepcopy(config)
    audit_config["experiment"]["source_run"] = config["experiment"]["source_phase4c_run"]
    manifest = audit_phase4d_sources(audit_config, output)
    prototype_cfg = _prototype_config(config)
    if prototype_cfg.maturity_sessions != int(config["memory_defaults"]["causal_horizon_sessions"]):
        raise ValueError("Rally maturity_sessions must match the external memory causal horizon.")
    manifest["phase"] = str(config["experiment"]["phase"])
    manifest["run_count"] = len(list(run_dirs))
    manifest["rally_prototype_config"] = vars(prototype_cfg)
    _write_json(output / "source_manifest.json", manifest)
    return manifest


def materialize_prototypes(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    *,
    resume: bool,
) -> None:
    """Build causal prototype banks from frozen train latents."""
    relative_cfg = _relative_config(config)
    prototype_cfg = _prototype_config(config)
    for run_dir in run_dirs:
        fold, seed = _run_key(run_dir)
        paths = _prepared_paths(run_dir, output, relative_cfg, resume=resume)
        directory = output / "prototypes" / fold / f"seed_{seed}"
        success_path = directory / "success.parquet"
        failure_path = directory / "failure.parquet"
        neutral_path = directory / "neutral.parquet"
        audit_path = directory / "audit.json"
        complete = success_path.exists() and failure_path.exists() and audit_path.exists()
        if prototype_cfg.include_neutral:
            complete = complete and neutral_path.exists()
        if resume and complete:
            continue
        train = pd.read_parquet(paths["train"])
        prototypes = build_rally_prototypes(train, prototype_cfg)
        if prototypes.success.empty or prototypes.failure.empty or (
            prototype_cfg.include_neutral and (prototypes.neutral is None or prototypes.neutral.empty)
        ):
            raise RuntimeError(f"{fold} seed {seed} produced an empty prototype class.")
        directory.mkdir(parents=True, exist_ok=True)
        prototypes.success.to_parquet(success_path, index=False)
        prototypes.failure.to_parquet(failure_path, index=False)
        if prototypes.neutral is not None:
            prototypes.neutral.to_parquet(neutral_path, index=False)
        _write_json(audit_path, prototypes.audit)


def _load_prototypes(
    output: Path,
    fold: str,
    seed: int,
    *,
    include_neutral: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, dict[str, Any]]:
    directory = output / "prototypes" / fold / f"seed_{seed}"
    return (
        pd.read_parquet(directory / "success.parquet"),
        pd.read_parquet(directory / "failure.parquet"),
        pd.read_parquet(directory / "neutral.parquet") if include_neutral else None,
        json.loads((directory / "audit.json").read_text(encoding="utf-8")),
    )


def _query_columns(query: pd.DataFrame) -> list[str]:
    names = {
        "open", "close", "Open", "Close", "adj_open", "adj_close",
        "future_max_return_63", "future_return_63", "future_universe_alpha_63",
        "future_blended_alpha_63", "future_min_return_63",
        "event_upside_before_drawdown_126", "event_upside_hit_126", "event_drawdown_hit_126",
    }
    return [column for column in query.columns if column in names]


def _source_signal_path(config: dict[str, Any], split: str, fold: str, seed: int) -> Path:
    return (
        _resolve(config["experiment"]["source_phase4e_run"])
        / "evaluations"
        / split
        / str(config["experiment"]["source_semantic"])
        / fold
        / f"seed_{seed}"
        / "eval"
        / "signals.parquet"
    )


def _baseline_signals(
    config: dict[str, Any],
    train: pd.DataFrame,
    query: pd.DataFrame,
    memory_cfg: MarketMemoryConfig,
    split: str,
    fold: str,
    seed: int,
) -> pd.DataFrame:
    source_path = _source_signal_path(config, split, fold, seed)
    if source_path.exists():
        source = pd.read_parquet(source_path)
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
        keys = query[["ticker", "timestamp"]].copy()
        keys["timestamp"] = pd.to_datetime(keys["timestamp"], utc=True)
        required = [
            "future_max_return_63", "future_min_return_63", "future_return_63",
            "future_blended_alpha_63", "event_upside_before_drawdown_126",
            "event_upside_hit_126", "event_drawdown_hit_126",
        ]
        extras = [column for column in required if column in query and column not in source]
        return source.merge(
            query[["ticker", "timestamp", *extras]].drop_duplicates(),
            on=["ticker", "timestamp"],
            how="inner",
            validate="one_to_one",
        )
    signals, _ = score_market_memory(
        train,
        query,
        memory_cfg,
        keep_query_columns=_query_columns(query),
    )
    return signals


def score_rally_memory(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    splits: Iterable[str],
    *,
    resume: bool,
) -> None:
    """Attach rally success-versus-failure evidence to frozen A2 signals."""
    relative_cfg = _relative_config(config)
    prototype_cfg = _prototype_config(config)
    for split in splits:
        for run_dir in run_dirs:
            fold, seed = _run_key(run_dir)
            result_dir = output / "evaluations" / str(split) / fold / f"seed_{seed}" / "eval"
            signal_path = result_dir / "signals.parquet"
            if resume and signal_path.exists():
                continue
            paths = _prepared_paths(run_dir, output, relative_cfg, resume=resume)
            train = pd.read_parquet(paths["train"])
            query = pd.read_parquet(paths[str(split)])
            if config.get("_smoke", False):
                query = query.head(256).copy()
            calibration = _calibrate_distances(train, None, config["ood_calibration"])
            memory_cfg = _memory_config(config, calibration)
            success, failure, neutral, audit = _load_prototypes(
                output,
                fold,
                seed,
                include_neutral=prototype_cfg.include_neutral,
            )
            prototype_set = RallyPrototypeSet(success=success, failure=failure, audit=audit, neutral=neutral)
            membership, neighbors = score_rally_prototype_membership(
                train,
                prototype_set,
                query,
                prototype_cfg,
                bandwidth=float(calibration["confidence_reference_distance"]),
            )
            baseline = _baseline_signals(config, train, query, memory_cfg, str(split), fold, seed)
            baseline["timestamp"] = pd.to_datetime(baseline["timestamp"], utc=True)
            signals = baseline.merge(
                membership.drop(columns=["query_id"]), on=["ticker", "timestamp"], how="left", validate="one_to_one"
            )
            signals = _attach_rally_outcomes(signals, prototype_cfg)
            result_dir.mkdir(parents=True, exist_ok=True)
            signals.to_parquet(signal_path, index=False)
            neighbors.to_parquet(result_dir / "prototype_neighbors.parquet", index=False)
            _write_json(
                result_dir / "prototype_scoring_audit.json",
                {
                    "fold": fold,
                    "seed": seed,
                    "split": str(split),
                    "query_rows": len(signals),
                    "bandwidth": calibration["confidence_reference_distance"],
                    "ood_threshold": calibration["max_median_distance"],
                    "prototype_audit": audit,
                    "eligible_fraction": float(membership["rally_start_eligible"].fillna(False).mean()),
                    "support_pass_fraction": float(membership["rally_support_pass"].fillna(False).mean()),
                },
            )


def _attach_rally_outcomes(signals: pd.DataFrame, config: RallyPrototypeConfig) -> pd.DataFrame:
    out = signals.copy()
    out["rally_target_success"] = (
        pd.to_numeric(out["future_max_return_63"], errors="coerce") >= config.upside_threshold
    ) & (
        pd.to_numeric(out["event_upside_before_drawdown_126"], errors="coerce") == 1.0
    )
    out["rally_target_failure"] = (
        pd.to_numeric(out["event_drawdown_hit_126"], errors="coerce") == 1.0
    ) & (
        pd.to_numeric(out["event_upside_before_drawdown_126"], errors="coerce") == 0.0
    ) & (
        pd.to_numeric(out["future_min_return_63"], errors="coerce") <= config.drawdown_threshold
    )
    return out


def _candidate_frame(signals: pd.DataFrame, candidate: dict[str, Any]) -> tuple[pd.DataFrame, str, str]:
    frame = signals.copy()
    mode = str(candidate["mode"])
    if mode == "a2_base":
        return frame, "opportunity_score", "opportunity_score"
    accepted = _rally_acceptance(frame, candidate, mode)
    if mode == "rally_rank":
        frame["rally_entry_score"] = frame["rally_start_probability"].where(
            accepted, -np.inf
        )
        return frame, "rally_entry_score", "opportunity_score"
    if mode == "a2_rally_gate":
        frame["rally_gate_score"] = frame["opportunity_score"].where(accepted, -np.inf)
        return frame, "rally_gate_score", "opportunity_score"
    raise ValueError(f"Unsupported prototype-memory candidate mode {mode!r}.")


def _rally_acceptance(frame: pd.DataFrame, candidate: dict[str, Any], mode: str) -> pd.Series:
    """Apply explicit entry evidence thresholds without changing the A2 exit score."""
    minimum_probability = candidate.get("minimum_probability")
    minimum_log_odds = candidate.get("minimum_log_odds")
    if minimum_probability is None:
        minimum_probability = -np.inf
    if minimum_log_odds is None:
        minimum_log_odds = 0.0 if mode == "a2_rally_gate" else -np.inf
    return (
        frame["rally_start_eligible"].fillna(False)
        & (pd.to_numeric(frame["rally_start_probability"], errors="coerce") >= float(minimum_probability))
        & (pd.to_numeric(frame["rally_start_log_odds"], errors="coerce") >= float(minimum_log_odds))
    )


def _entry_metrics(signals: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int | None]:
    if trades.empty:
        return {"rally_start_precision": None, "mean_remaining_upside": None, "mean_entry_probability": None, "rally_entry_count": 0}
    rows: list[pd.Series] = []
    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    for trade in trades.itertuples(index=False):
        subset = frame.loc[(frame["ticker"].astype(str) == str(trade.ticker)) & (frame["timestamp"] < pd.Timestamp(trade.entry_date))]
        if not subset.empty:
            rows.append(subset.sort_values("timestamp").iloc[-1])
    if not rows:
        return {"rally_start_precision": None, "mean_remaining_upside": None, "mean_entry_probability": None, "rally_entry_count": 0}
    entry = pd.DataFrame(rows)
    return {
        "rally_start_precision": float(entry["rally_target_success"].mean()),
        "mean_remaining_upside": float(pd.to_numeric(entry["future_max_return_63"], errors="coerce").mean()),
        "mean_entry_probability": float(pd.to_numeric(entry["rally_start_probability"], errors="coerce").mean()),
        "rally_entry_count": int(len(entry)),
    }


def trade_candidates(
    config: dict[str, Any],
    output: Path,
    splits: Iterable[str],
    *,
    candidates: list[dict[str, Any]],
    resume: bool,
) -> pd.DataFrame:
    """Backtest fixed entry variants while keeping the selected A2 exit score unchanged."""
    policy = _source_policy(config)
    rows: list[dict[str, Any]] = []
    for split in splits:
        for signal_path in sorted((output / "evaluations" / str(split)).glob("fold_*/seed_*/eval/signals.parquet")):
            parts = signal_path.parts
            fold, seed = parts[-4], int(parts[-3].split("_", 1)[1])
            signals = pd.read_parquet(signal_path)
            for candidate in candidates:
                candidate_id = str(candidate["id"])
                result_dir = output / "policies" / str(split) / candidate_id / fold / f"seed_{seed}" / "eval"
                metrics_path = result_dir / "metrics.json"
                if resume and metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    frame, entry_score, exit_score = _candidate_frame(signals, candidate)
                    decisions: list[dict[str, Any]] = []
                    trades, equity = run_long_only_backtest(
                        frame,
                        policy,
                        score_col=entry_score,
                        exit_score_col=exit_score,
                        decision_log=decisions,
                    )
                    metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
                    metrics.update(equal_weight_baseline(frame, policy.initial_capital))
                    metrics.update(_entry_metrics(frame, trades))
                    metrics.update(
                        {
                            "candidate": candidate_id,
                            "entry_score_column": entry_score,
                            "exit_score_column": exit_score,
                            "eligible_rally_fraction": float(frame["rally_start_eligible"].fillna(False).mean()),
                            "accepted_rally_fraction": float(_rally_acceptance(frame, candidate, str(candidate["mode"])).mean())
                            if str(candidate["mode"]) != "a2_base"
                            else 1.0,
                        }
                    )
                    result_dir.mkdir(parents=True, exist_ok=True)
                    trades.to_csv(result_dir / "trades.csv", index=False)
                    equity.to_csv(result_dir / "equity_curve.csv", index=False)
                    pd.DataFrame(decisions).to_csv(result_dir / "decisions.csv", index=False)
                    _write_json(metrics_path, metrics)
                    _write_yaml(result_dir / "policy_snapshot.yaml", {"candidate": candidate, "source_policy": config["experiment"]["source_candidate"]})
                rows.append({"split": str(split), "fold": fold, "seed": seed, "candidate": candidate_id, **_summary_metrics(metrics)})
    summary = pd.DataFrame(rows)
    path = output / "policy_summary.csv"
    if path.exists():
        summary = pd.concat([pd.read_csv(path), summary], ignore_index=True).drop_duplicates(["split", "fold", "seed", "candidate"], keep="last")
    summary.to_csv(path, index=False)
    return summary


def _summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_return", "annualized_return", "sharpe", "max_drawdown", "profit_factor", "trade_count",
        "win_rate", "turnover", "exposure", "capital_efficiency", "equal_weight_baseline_return",
        "rally_start_precision", "mean_remaining_upside", "mean_entry_probability", "rally_entry_count",
        "eligible_rally_fraction",
        "accepted_rally_fraction",
    )
    return {key: metrics.get(key) for key in keys}


def _equity_returns(output: Path, split: str, candidate: str, fold: str, seed: int) -> pd.Series:
    path = output / "policies" / split / candidate / fold / f"seed_{seed}" / "eval" / "equity_curve.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().dropna()


def _candidate_stats(output: Path, rows: pd.DataFrame, candidate: str, objective: dict[str, Any]) -> dict[str, Any]:
    selected = rows.loc[rows["candidate"] == candidate]
    returns = [
        _equity_returns(output, str(row.split), candidate, str(row.fold), int(row.seed)).rename(f"{row.fold}_{row.seed}")
        for row in selected.itertuples(index=False)
    ]
    pooled = pd.concat(returns, axis=1).mean(axis=1, skipna=True).dropna()
    sharpes = [_sharpe(value) for value in returns]
    excess = selected["total_return"] - selected["equal_weight_baseline_return"]
    worst_drawdown = float(selected["max_drawdown"].min())
    robust = (
        float(objective["pooled_sharpe_weight"]) * _sharpe(pooled)
        + float(objective["minimum_fold_sharpe_weight"]) * min(sharpes)
        + float(objective["mean_excess_return_weight"]) * float(excess.mean())
        - float(objective["drawdown_weight"]) * abs(worst_drawdown)
    )
    return {
        "candidate": candidate,
        "run_count": int(len(selected)),
        "pooled_sharpe": _sharpe(pooled),
        "minimum_run_sharpe": float(min(sharpes)),
        "mean_return": float(selected["total_return"].mean()),
        "mean_excess_return": float(excess.mean()),
        "positive_run_fraction": float((selected["total_return"] > 0.0).mean()),
        "positive_fold_excess_fraction": float(_fold_excess_fraction(selected)),
        "worst_drawdown": worst_drawdown,
        "mean_trade_count": float(selected["trade_count"].mean()),
        "mean_rally_start_precision": float(selected["rally_start_precision"].mean()),
        "mean_remaining_upside": float(selected["mean_remaining_upside"].mean()),
        "robust_score": float(robust),
        "daily_returns": pooled,
    }


def _fold_excess_fraction(rows: pd.DataFrame) -> float:
    grouped = rows.assign(
        _excess=rows["total_return"] - rows["equal_weight_baseline_return"]
    ).groupby("fold", sort=True)["_excess"].mean()
    return float((grouped > 0.0).mean()) if len(grouped) else 0.0


def _leave_one_fold_out(output: Path, rows: pd.DataFrame, candidates: list[str], objective: dict[str, Any]) -> tuple[pd.DataFrame, float]:
    records: list[dict[str, Any]] = []
    heldout_series: list[pd.Series] = []
    folds = sorted(rows["fold"].unique())
    if len(folds) < 2:
        ranked = sorted((_candidate_stats(output, rows, candidate, objective) for candidate in candidates), key=lambda item: item["robust_score"], reverse=True)
        winner = ranked[0]["candidate"]
        return (
            pd.DataFrame([{"heldout_fold": folds[0] if folds else "none", "selected_candidate": winner, "diagnostic_only": True}]),
            ranked[0]["pooled_sharpe"],
        )
    for fold in folds:
        train = rows.loc[rows["fold"] != fold]
        ranked = sorted((_candidate_stats(output, train, candidate, objective) for candidate in candidates), key=lambda item: item["robust_score"], reverse=True)
        winner = ranked[0]["candidate"]
        heldout = rows.loc[(rows["fold"] == fold) & (rows["candidate"] == winner)]
        series = [
            _equity_returns(output, str(row.split), winner, str(row.fold), int(row.seed)).rename(f"{row.fold}_{row.seed}")
            for row in heldout.itertuples(index=False)
        ]
        daily = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna()
        heldout_series.append(daily.rename(fold))
        excess = heldout["total_return"] - heldout["equal_weight_baseline_return"]
        records.append({"heldout_fold": fold, "selected_candidate": winner, "training_robust_score": ranked[0]["robust_score"], "heldout_sharpe": _sharpe(daily), "heldout_return": float(heldout["total_return"].mean()), "heldout_excess_return": float(excess.mean())})
    pooled = pd.concat(heldout_series, axis=1).mean(axis=1, skipna=True).dropna()
    return pd.DataFrame(records), _sharpe(pooled)


def select_policy(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Select only a cross-market improvement and keep confirmation sealed otherwise."""
    split = str(config["experiment"]["validation_split"])
    rows = pd.read_csv(output / "policy_summary.csv")
    rows = rows.loc[rows["split"] == split].copy()
    candidates = [str(item["id"]) for item in config["policy_candidates"]]
    objective = config["selection"]["objective"]
    stats = [_candidate_stats(output, rows, candidate, objective) for candidate in candidates]
    returns_matrix = pd.concat({item["candidate"]: item.pop("daily_returns") for item in stats}, axis=1).fillna(0.0)
    pbo = probability_of_backtest_overfitting(returns_matrix.to_numpy())
    baseline = next(item for item in stats if item["candidate"] == config["selection"]["baseline_candidate"])
    gates = config["selection"]
    for item in stats:
        item["deflated_sharpe_probability"] = deflated_sharpe_probability(returns_matrix[item["candidate"]].to_numpy(dtype=float), len(candidates))
        item["pbo"] = pbo
        item["rally_precision_lift"] = item["mean_rally_start_precision"] - baseline["mean_rally_start_precision"]
        item["individual_gate_pass"] = bool(
            item["candidate"] != baseline["candidate"]
            and item["pooled_sharpe"] >= float(gates["target_pooled_sharpe"])
            and item["mean_excess_return"] >= float(gates["minimum_mean_excess_return"])
            and item["positive_run_fraction"] >= float(gates["minimum_positive_run_fraction"])
            and item["positive_fold_excess_fraction"] >= float(gates["minimum_positive_fold_excess_fraction"])
            and abs(item["worst_drawdown"]) <= float(gates["maximum_worst_drawdown"])
            and item["mean_trade_count"] >= float(gates["minimum_mean_trade_count"])
            and item["deflated_sharpe_probability"] >= float(gates["minimum_deflated_sharpe_probability"])
            and pbo <= float(gates["maximum_pbo"])
            and item["rally_precision_lift"] >= float(gates["minimum_rally_precision_lift"])
        )
    leaderboard = pd.DataFrame(stats).sort_values("robust_score", ascending=False).reset_index(drop=True)
    leaderboard.to_csv(output / "policy_leaderboard.csv", index=False)
    oof, oof_sharpe = _leave_one_fold_out(output, rows, candidates, objective)
    oof.to_csv(output / "cross_market_oof.csv", index=False)
    eligible = leaderboard.loc[leaderboard["individual_gate_pass"]]
    promotion = bool(not eligible.empty and oof_sharpe >= float(gates["target_cross_market_oof_sharpe"]) and not config.get("_smoke", False))
    winner_id = str(eligible.iloc[0]["candidate"]) if promotion else str(leaderboard.iloc[0]["candidate"])
    winner = next(item for item in config["policy_candidates"] if item["id"] == winner_id)
    selection = {
        "phase": str(config["experiment"]["phase"]),
        "selected_candidate": winner_id,
        "candidate": winner,
        "promotion_pass": promotion,
        "confirmation_unlocked": promotion,
        "cross_market_oof_sharpe": oof_sharpe,
        "oof_selection_counts": dict(Counter(oof["selected_candidate"])),
        "best_validation": _native(leaderboard.iloc[0].to_dict()),
        "failure_reason": None if promotion else f"Phase {config['experiment']['phase']} prototype-memory gates were not all satisfied.",
        "smoke_only": bool(config.get("_smoke", False)),
    }
    _write_yaml(output / "selected_policy.yaml", selection)
    _write_selection_report(output / "selection_report.md", selection, leaderboard, oof)
    return selection


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _write_selection_report(path: Path, selection: dict[str, Any], leaderboard: pd.DataFrame, oof: pd.DataFrame) -> None:
    columns = ["candidate", "pooled_sharpe", "mean_excess_return", "mean_rally_start_precision", "rally_precision_lift", "worst_drawdown", "pbo", "individual_gate_pass"]
    lines = [
        f"# Phase {selection.get('phase', '4H')} Prototype Memory",
        "",
        f"- Selected policy: `{selection['selected_candidate']}`",
        f"- Promotion pass: `{selection['promotion_pass']}`",
        f"- Cross-market OOF Sharpe: `{selection['cross_market_oof_sharpe']:.4f}`",
        "",
        "## Validation Leaderboard",
        "",
        leaderboard[columns].to_markdown(index=False),
        "",
        "## Leave-One-Fold-Out Selection",
        "",
        oof.to_markdown(index=False),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_confirmation(config: dict[str, Any], output: Path, run_dirs: Iterable[Path], selection: dict[str, Any], *, resume: bool) -> None:
    """Run test and holdout only after validation promotion succeeds."""
    if not selection["confirmation_unlocked"]:
        _write_json(output / "confirmation_status.json", {"executed": False, "reason": selection["failure_reason"]})
        return
    winner = selection["candidate"]
    splits = config["experiment"]["confirmation_splits"]
    score_rally_memory(config, output, run_dirs, splits, resume=resume)
    trade_candidates(config, output, splits, candidates=[winner], resume=resume)
    _write_json(output / "confirmation_status.json", {"executed": True, "candidate": winner["id"], "splits": splits})


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve(args.config)
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)
    config["_smoke"] = bool(args.smoke)
    output = _ensure_output(config, args.run_id, resume=args.resume)
    runs = _select_runs(_complete_run_dirs(_resolve(config["experiment"]["source_phase4c_run"])), config["experiment"], args.max_runs)
    if args.smoke:
        runs = runs[:1]
    stage = args.stage
    if stage in {"audit", "all"}:
        audit_sources(config, output, runs)
    if stage in {"materialize", "all"}:
        materialize_prototypes(config, output, runs, resume=args.resume)
    if stage in {"score", "all"}:
        score_rally_memory(config, output, runs, [config["experiment"]["validation_split"]], resume=args.resume)
    if stage in {"trade", "all"}:
        trade_candidates(config, output, [config["experiment"]["validation_split"]], candidates=config["policy_candidates"], resume=args.resume)
    selection: dict[str, Any] | None = None
    if stage in {"select", "all"}:
        selection = select_policy(config, output)
    if stage in {"confirm", "all"}:
        if selection is None:
            selection = _read_yaml(output / "selected_policy.yaml")
        run_confirmation(config, output, runs, selection, resume=args.resume)
    summary = {"run_id": args.run_id, "stage": stage, "source_run_count": len(runs), "output": str(output)}
    _write_json(output / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4h_rally_start_memory.yaml")
    parser.add_argument("--run-id", default="phase4h_v1")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
