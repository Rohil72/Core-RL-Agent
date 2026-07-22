"""Run causal, rotating unseen-ticker verification for the market memory policy."""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.backtest.market_memory_evaluator import (  # noqa: E402
    load_evaluation_config,
    run_market_memory_evaluation,
)
from src.eval.causal_memory import attach_outcome_availability  # noqa: E402
from src.trainers.train_cycle_model import load_config, train  # noqa: E402
from run_phase4b_sweep import load_market_universe  # noqa: E402

LOGGER = logging.getLogger(__name__)
SEEDS = (7, 17, 37)
MODES = {
    "cross_ticker_only": {"same_ticker_mode": "exclude", "exclude_query_sector": False},
    "sector_excluded": {"same_ticker_mode": "exclude", "exclude_query_sector": True},
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _is_completed_row(row: dict[str, Any]) -> bool:
    """Return whether a saved row contains every required evaluation result."""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for split in ("test", "holdout"):
        split_metrics = metrics.get(split)
        if not isinstance(split_metrics, dict):
            return False
        for mode in MODES:
            mode_metrics = split_metrics.get(mode)
            if not isinstance(mode_metrics, dict) or "error" in mode_metrics:
                return False
    return True


def _load_completed_rows(reports_base: Path) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    """Load only fully evaluated seed rows from a prior interrupted Phase 4C run."""
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, int]] = set()
    for metrics_path in reports_base.glob("fold_*/seed_*/run_metrics.json"):
        try:
            row = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring unreadable prior result: %s", metrics_path)
            continue
        if _is_completed_row(row):
            rows.append(row)
            completed.add((str(row["fold"]), int(row["seed"])))
    return rows, completed


def _normalise_latent_exports(latent_dir: Path) -> None:
    for prefix in ("train_latents", "val_latents", "test_latents", "holdout_latents"):
        files = sorted(latent_dir.glob(f"{prefix}_*.parquet"))
        if not files:
            continue
        target = latent_dir / f"{prefix}.parquet"
        if target.exists():
            target.unlink()
        files[-1].replace(target)
        for stale in files[:-1]:
            stale.unlink(missing_ok=True)


def _training_artifacts_complete(run_dir: Path) -> bool:
    latent_dir = run_dir / "latents"
    required = [latent_dir / f"{split}_latents.parquet" for split in ("train", "val", "test", "holdout")]
    return (run_dir / "model" / "final_model.pt").exists() and all(path.exists() for path in required)


def _configure_cuda(config: dict[str, Any]) -> None:
    requested = str(config.get("training", {}).get("device", "auto"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Phase 4C requested CUDA, but this PyTorch installation cannot access a CUDA GPU.")
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is unavailable; Phase 4C will run on CPU.")
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    LOGGER.info("CUDA enabled: %s", torch.cuda.get_device_name(0))


def _validate_fold_manifest(config: dict[str, Any], folds: list[dict[str, Any]]) -> None:
    pattern = str(config["data"]["precomputed_dir"])
    resolved = pattern if Path(pattern).is_absolute() else str(PROJECT_ROOT / pattern)
    available = {Path(path).stem for path in glob.glob(resolved)}
    requested = [str(ticker) for fold in folds for ticker in fold["holdout_tickers"]]
    missing = sorted(set(requested) - available)
    duplicates = sorted({ticker for ticker in requested if requested.count(ticker) > 1})
    uncovered = sorted(available - set(requested))
    if missing or duplicates or uncovered:
        parts = []
        if missing:
            parts.append("missing datasets: " + ", ".join(missing))
        if duplicates:
            parts.append("duplicated across folds: " + ", ".join(duplicates))
        if uncovered:
            parts.append("not assigned to a fold: " + ", ".join(uncovered))
        raise ValueError("Invalid Phase 4C holdout manifest; " + "; ".join(parts))
    LOGGER.info("Fold manifest preflight passed for %d available tickers.", len(available))


def _enrich_memory(path: Path, mapping: dict[str, dict[str, str]], precomputed_glob: str) -> None:
    frame = pd.read_parquet(path)
    frame["sector"] = frame["ticker"].map(lambda value: mapping.get(str(value), {}).get("sector", "Unknown"))
    frame["industry"] = frame["ticker"].map(lambda value: mapping.get(str(value), {}).get("industry", "Unknown"))
    frame = attach_outcome_availability(frame, precomputed_glob, horizon_sessions=126)
    frame = frame.dropna(subset=["outcome_available_timestamp"])
    frame.to_parquet(path, index=False)


def _enrich_query(path: Path, mapping: dict[str, dict[str, str]]) -> None:
    frame = pd.read_parquet(path)
    frame["sector"] = frame["ticker"].map(lambda value: mapping.get(str(value), {}).get("sector", "Unknown"))
    frame["industry"] = frame["ticker"].map(lambda value: mapping.get(str(value), {}).get("industry", "Unknown"))
    frame.to_parquet(path, index=False)


def _rank_ic(signals_path: Path) -> float | None:
    if not signals_path.exists():
        return None
    frame = pd.read_parquet(signals_path)
    required = [
        "timestamp", "opportunity_score", "future_max_return_63",
        "future_min_return_63", "event_upside_before_drawdown_126",
        "retrieval_confidence",
    ]
    if any(column not in frame.columns for column in required):
        return None
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["utility"] = (
        (frame["future_max_return_63"] - frame["future_min_return_63"].abs())
        * (0.5 + 0.5 * frame["retrieval_confidence"])
        + 0.05 * frame["event_upside_before_drawdown_126"]
    )
    frame["week"] = frame["timestamp"].dt.to_period("W")
    values: list[float] = []
    for _, group in frame.dropna(subset=["opportunity_score", "utility", "week"]).groupby("week"):
        if len(group) < 3:
            continue
        value = float(stats.spearmanr(group["opportunity_score"], group["utility"])[0])
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else None


def _causal_audit(
    neighbors_path: Path,
    signals_path: Path,
    memory_path: Path,
    mode: str,
    require_query_memory_disjoint: bool,
) -> dict[str, Any]:
    neighbors = pd.read_parquet(neighbors_path) if neighbors_path.exists() else pd.DataFrame()
    query = pd.read_parquet(signals_path) if signals_path.exists() else pd.DataFrame()
    memory = pd.read_parquet(memory_path)
    if query.empty or "query_id" not in query.columns:
        return {
            "mode": mode,
            "query_rows": int(len(query)),
            "memory_rows": int(len(memory)),
            "neighbor_rows": int(len(neighbors)),
            "pass": False,
            "reason": "Signals artifact did not contain query_id.",
        }
    query["timestamp"] = pd.to_datetime(query["timestamp"], utc=True, errors="coerce")
    memory["outcome_available_timestamp"] = pd.to_datetime(
        memory["outcome_available_timestamp"], utc=True, errors="coerce"
    )
    audit: dict[str, Any] = {
        "mode": mode,
        "query_rows": int(len(query)),
        "memory_rows": int(len(memory)),
        "neighbor_rows": int(len(neighbors)),
        "same_ticker_rate": None,
        "causality_pass_rate": None,
        "heldout_memory_ticker_rate": None,
        "query_memory_ticker_overlap": float(
            query["ticker"].astype(str).isin(memory["ticker"].astype(str)).mean()
        ),
        "pass": True,
    }
    if neighbors.empty:
        audit["pass"] = mode != "cross_ticker_only"
        audit["reason"] = "No neighbor rows were written."
        return audit
    query_tickers = query.set_index("query_id")["ticker"].astype(str) if "query_id" in query else pd.Series(dtype=str)
    query_times = query.set_index("query_id")["timestamp"] if "query_id" in query else pd.Series(dtype="datetime64[ns, UTC]")
    if {"query_id", "neighbor_ticker"}.issubset(neighbors.columns):
        same = []
        causal = []
        for row in neighbors.itertuples(index=False):
            q = query_tickers.get(getattr(row, "query_id"), None)
            if q is None:
                continue
            same.append(q == str(getattr(row, "neighbor_ticker")))
            available_value = getattr(row, "neighbor_outcome_available_timestamp", None)
            query_time = query_times.get(getattr(row, "query_id"), None)
            if available_value is not None and query_time is not None:
                available_time = pd.to_datetime(available_value, utc=True, errors="coerce")
                causal.append(bool(pd.notna(available_time) and available_time <= query_time))
        if same:
            audit["same_ticker_rate"] = float(np.mean(same))
        if causal:
            audit["causality_pass_rate"] = float(np.mean(causal))
        if mode == "cross_ticker_only" and any(same):
            audit["pass"] = False
        if causal and not all(causal):
            audit["pass"] = False
        if require_query_memory_disjoint and audit["query_memory_ticker_overlap"] > 0.0:
            audit["pass"] = False
    return audit


def _evaluate_modes(
    run_dir: Path,
    eval_config: dict[str, Any],
    split: str,
    memory_path: Path,
    *,
    reuse_existing: bool,
) -> dict[str, Any]:
    mapping = load_market_universe(str(PROJECT_ROOT / "config" / "market_universe.yaml"))
    latent_dir = run_dir / "latents"
    precomputed_glob = str(PROJECT_ROOT / "data" / "precomputed_v2" / "*.parquet")
    _enrich_memory(memory_path, mapping, precomputed_glob)
    query_path = latent_dir / f"{split}_latents.parquet"
    _enrich_query(query_path, mapping)
    results: dict[str, Any] = {}
    for mode, overrides in MODES.items():
        mode_dir = run_dir / split / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(eval_config)
        config["data"] = {
            "train_latents": str(memory_path),
            "test_latents": str(query_path),
            "precomputed_glob": precomputed_glob,
            "output_dir": str(mode_dir),
        }
        config.setdefault("memory", {}).update(overrides)
        config["memory"]["minimum_neighbor_separation_sessions"] = 21
        config.setdefault("run", {})["write_memory_reports"] = False
        try:
            output = mode_dir / "eval"
            reusable = all(
                (output / filename).exists()
                for filename in ("metrics.json", "signals.parquet", "neighbors.parquet")
            )
            if not (reuse_existing and reusable):
                output = run_market_memory_evaluation(config, PROJECT_ROOT, run_id="eval")
            else:
                LOGGER.info("Reusing completed evaluation %s/%s", split, mode)
            metrics_path = output / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["weekly_rank_ic"] = _rank_ic(output / "signals.parquet")
            metrics["causal_audit"] = _causal_audit(
                output / "neighbors.parquet",
                output / "signals.parquet",
                memory_path,
                mode,
                require_query_memory_disjoint=(split == "holdout"),
            )
            _write_json(mode_dir / "retrieval_audit.json", metrics["causal_audit"])
            results[mode] = metrics
        except Exception as exc:  # keep one broken diagnostic mode from hiding other modes
            results[mode] = {"error": str(exc)}
            LOGGER.exception("Evaluation failed for %s/%s/%s", run_dir, split, mode)
    return results


def run(args: argparse.Namespace) -> Path:
    train_config = load_config(args.train_config)
    eval_config = load_evaluation_config(args.memory_config)
    fold_config = _read_yaml(Path(args.fold_config))
    folds = fold_config["folds"][: args.max_folds or None]
    seeds = tuple(fold_config.get("experiment", {}).get("seeds", SEEDS))[: args.max_seeds or None]
    _configure_cuda(train_config)
    _validate_fold_manifest(train_config, fold_config["folds"])
    reports_base = PROJECT_ROOT / "reports" / "phase4c" / args.run_id
    manifest = {"run_id": args.run_id, "folds": folds, "seeds": seeds}
    rows: list[dict[str, Any]] = []
    completed: set[tuple[str, int]] = set()
    if reports_base.exists() and any(reports_base.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to reuse non-empty run directory: {reports_base}")
    if args.resume:
        manifest_path = reports_base / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without a manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("folds") != folds or existing_manifest.get("seeds") != list(seeds):
            raise ValueError("Resume configuration does not match the existing Phase 4C manifest.")
        rows, completed = _load_completed_rows(reports_base)
        LOGGER.info("Resuming %s with %d completed seed runs.", reports_base, len(completed))
    else:
        reports_base.mkdir(parents=True, exist_ok=False)
        _write_json(reports_base / "manifest.json", manifest)
    for fold in folds:
        fold_id = str(fold["id"])
        for seed in seeds:
            if (fold_id, int(seed)) in completed:
                LOGGER.info("Skipping completed run %s seed %s", fold_id, seed)
                continue
            run_dir = reports_base / fold_id / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = copy.deepcopy(train_config)
            config["training"]["seed"] = int(seed)
            config["split"]["holdout_tickers"] = [str(ticker) for ticker in fold["holdout_tickers"]]
            config["split"]["ticker_holdout_fraction"] = 0.0
            config["split"]["seed"] = int(seed)
            config["training"]["model_dir"] = str(run_dir / "model")
            config["evaluation"]["reports_dir"] = str(run_dir / "reports")
            config["latent_export"] = {"output_dir": str(run_dir / "latents"), "splits": ["train", "val", "test", "holdout"]}
            if args.smoke:
                config["training"]["epochs"] = 1
                config["training"]["batch_size"] = 16
            config_path = run_dir / "config_snapshot.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            try:
                if args.resume and _training_artifacts_complete(run_dir):
                    LOGGER.info("Reusing trained model and latent exports for %s seed %s", fold_id, seed)
                else:
                    train(config_path=str(config_path))
                    _normalise_latent_exports(run_dir / "latents")
                memory_path = run_dir / "latents" / "train_latents.parquet"
                metrics = {
                    "test": _evaluate_modes(
                        run_dir, eval_config, "test", memory_path, reuse_existing=args.resume
                    ),
                    "holdout": _evaluate_modes(
                        run_dir, eval_config, "holdout", memory_path, reuse_existing=args.resume
                    ),
                }
                row = {"fold": fold_id, "seed": int(seed), "holdout_tickers": fold["holdout_tickers"], "metrics": metrics}
            except Exception as exc:
                LOGGER.exception("Training failed for %s seed %s", fold_id, seed)
                row = {"fold": fold_id, "seed": int(seed), "holdout_tickers": fold["holdout_tickers"], "error": str(exc)}
            _write_json(run_dir / "run_metrics.json", row)
            rows.append(row)
    aggregate = _aggregate_rows(rows, len(folds), len(seeds))
    _write_json(reports_base / "aggregate_summary.json", aggregate)
    _write_summary(reports_base / "aggregate_summary.md", aggregate)
    return reports_base


def _aggregate_rows(rows: list[dict[str, Any]], fold_count: int, seed_count: int) -> dict[str, Any]:
    fold_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metric = row.get("metrics", {}).get("holdout", {}).get("cross_ticker_only")
        if not metric or "error" in metric:
            continue
        fold_rows.setdefault(str(row["fold"]), []).append(metric)

    fold_aggregates: list[dict[str, Any]] = []
    for fold, metrics in sorted(fold_rows.items()):
        def mean_metric(name: str) -> float | None:
            values = [float(item[name]) for item in metrics if item.get(name) is not None and np.isfinite(item[name])]
            return float(np.mean(values)) if values else None

        returns = mean_metric("total_return")
        equal_weight = mean_metric("equal_weight_baseline_return")
        fold_aggregates.append(
            {
                "fold": fold,
                "completed_seeds": len(metrics),
                "mean_return": returns,
                "mean_sharpe": mean_metric("sharpe"),
                "mean_max_drawdown": mean_metric("max_drawdown"),
                "mean_weekly_rank_ic": mean_metric("weekly_rank_ic"),
                "mean_equal_weight_return": equal_weight,
                "mean_excess_over_equal_weight": (
                    returns - equal_weight
                    if returns is not None and equal_weight is not None
                    else None
                ),
                "all_audits_pass": all(
                    item.get("causal_audit", {}).get("pass", False) for item in metrics
                ),
            }
        )

    def finite_fold_values(name: str) -> list[float]:
        return [
            float(item[name])
            for item in fold_aggregates
            if item.get(name) is not None and np.isfinite(item[name])
        ]

    fold_returns = finite_fold_values("mean_return")
    fold_sharpes = finite_fold_values("mean_sharpe")
    fold_excess = finite_fold_values("mean_excess_over_equal_weight")
    fold_ics = finite_fold_values("mean_weekly_rank_ic")
    complete_runs = sum(item["completed_seeds"] for item in fold_aggregates)
    gates = {
        "all_requested_runs_completed": complete_runs == fold_count * seed_count,
        "at_least_four_of_five_positive_folds": sum(value > 0 for value in fold_returns) >= min(4, fold_count),
        "median_fold_sharpe_positive": bool(fold_sharpes) and float(np.median(fold_sharpes)) > 0,
        "median_excess_over_equal_weight_positive": bool(fold_excess) and float(np.median(fold_excess)) > 0,
        "mean_weekly_rank_ic_positive": bool(fold_ics) and float(np.mean(fold_ics)) > 0,
        "all_retrieval_audits_pass": all(item["all_audits_pass"] for item in fold_aggregates),
    }
    return {
        "fold_aggregates": fold_aggregates,
        "complete_runs": complete_runs,
        "requested_runs": fold_count * seed_count,
        "overall": {
            "median_fold_return": float(np.median(fold_returns)) if fold_returns else None,
            "median_fold_sharpe": float(np.median(fold_sharpes)) if fold_sharpes else None,
            "median_fold_excess_over_equal_weight": float(np.median(fold_excess)) if fold_excess else None,
            "mean_fold_weekly_rank_ic": float(np.mean(fold_ics)) if fold_ics else None,
        },
        "promotion_gates": gates,
        "promotable": all(gates.values()),
    }


def _write_summary(path: Path, aggregate: dict[str, Any]) -> None:
    lines = ["# Phase 4C Rotating Holdout Summary", "", "| fold | completed seeds | return | Sharpe | max drawdown | Rank IC | audit |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in aggregate["fold_aggregates"]:
        lines.append(
            f"| {row['fold']} | {row['completed_seeds']} | {row['mean_return']} | {row['mean_sharpe']} | "
            f"{row['mean_max_drawdown']} | {row['mean_weekly_rank_ic']} | {row['all_audits_pass']} |"
        )
    lines.extend(["", "## Overall", "", f"- Complete runs: {aggregate['complete_runs']} / {aggregate['requested_runs']}"])
    for key, value in aggregate["overall"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Promotion Gates", ""])
    for key, value in aggregate["promotion_gates"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", f"## Promotable: {aggregate['promotable']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/final_encoder_training.yaml")
    parser.add_argument("--memory-config", default="configs/market_memory_backtest.yaml")
    parser.add_argument("--fold-config", default="configs/phase4c_rotating_holdouts.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--max-seeds", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run with the same manifest.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    output = run(args)
    print(f"Phase 4C output: {output}")


if __name__ == "__main__":
    main()
