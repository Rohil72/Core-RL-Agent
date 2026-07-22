"""Run frozen-latent retrieval-semantics experiments and promotion checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.market_memory_backtester import PolicyConfig, compute_backtest_metrics, run_long_only_backtest  # noqa: E402
from src.backtest.market_memory_evaluator import run_market_memory_evaluation  # noqa: E402
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
    stationary_bootstrap_delta,
)
from src.memory import (  # noqa: E402
    EnsembleConfig,
    FittedRetrievalMetric,
    MetricLearningConfig,
    RelativeOutcomeConfig,
    attach_relative_outcomes,
    combine_seed_signals,
    fit_retrieval_metric,
    load_latent_frame,
)
from src.memory.experience import latent_columns  # noqa: E402
from src.memory.retrieval import normalise_latents  # noqa: E402

LOGGER = logging.getLogger(__name__)
REQUIRED_SPLITS = ("train", "val", "test", "holdout")
STAGES = ("audit", "fit-metrics", "ablate", "select", "diagnostic-2024", "all")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2), encoding="utf-8")


def _write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(_native(value), sort_keys=False), encoding="utf-8")


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_run_dirs(source: Path) -> list[Path]:
    complete: list[Path] = []
    for run_dir in sorted(source.glob("fold_*/seed_*")):
        required = [run_dir / "latents" / f"{split}_latents.parquet" for split in REQUIRED_SPLITS]
        if (run_dir / "model" / "final_model.pt").exists() and all(path.exists() for path in required):
            complete.append(run_dir)
    return sorted(complete, key=_run_key)


def _run_key(run_dir: Path) -> tuple[str, int]:
    return run_dir.parent.name, int(run_dir.name.removeprefix("seed_"))


def _select_runs(
    run_dirs: list[Path],
    experiment_config: dict[str, Any],
    max_runs: int | None,
) -> list[Path]:
    active_folds = {str(value) for value in experiment_config.get("active_folds", [])}
    active_seeds = {int(value) for value in experiment_config.get("active_seeds", [])}
    selected = [
        path
        for path in run_dirs
        if (not active_folds or _run_key(path)[0] in active_folds)
        and (not active_seeds or _run_key(path)[1] in active_seeds)
    ]
    if not selected:
        raise RuntimeError("No complete Phase 4C runs matched the configured active fold/seed filter.")
    return selected if max_runs is None else selected[:max_runs]


def _ensure_output(config: dict[str, Any], run_id: str, *, resume: bool) -> Path:
    output = _resolve(config["experiment"]["output_root"]) / run_id
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to reuse non-empty Phase 4D directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def audit_sources(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Validate frozen Phase 4C inputs and record immutable source hashes."""
    source = _resolve(config["experiment"]["source_run"])
    complete = _complete_run_dirs(source)
    minimum = int(config["experiment"].get("minimum_complete_runs", 1))
    if len(complete) < minimum:
        raise RuntimeError(f"Only {len(complete)} complete frozen runs were found; {minimum} are required.")
    expected_seeds = {int(seed) for seed in config["experiment"].get("expected_seeds", [])}
    expected_folds = int(config["experiment"].get("expected_folds", 0))
    expected = {(f"fold_{fold:02d}", seed) for fold in range(1, expected_folds + 1) for seed in expected_seeds}
    observed = {_run_key(path) for path in complete}
    files: list[dict[str, Any]] = []
    for run_dir in complete:
        for split in REQUIRED_SPLITS:
            path = run_dir / "latents" / f"{split}_latents.parquet"
            files.append({"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path), "bytes": path.stat().st_size})
        checkpoint = run_dir / "model" / "final_model.pt"
        files.append({"path": str(checkpoint.relative_to(PROJECT_ROOT)), "sha256": _sha256(checkpoint), "bytes": checkpoint.stat().st_size})
    manifest = {
        "phase": str(config["experiment"].get("phase", "4D")),
        "transformer_training_allowed": False,
        "source": str(source),
        "complete_runs": [{"fold": fold, "seed": seed} for fold, seed in sorted(observed)],
        "missing_runs": [{"fold": fold, "seed": seed} for fold, seed in sorted(expected - observed)],
        "files": files,
        "config_sha256": _sha256(_resolve(config["_config_path"])),
    }
    _write_json(output / "source_manifest.json", manifest)
    return manifest


def _load_enriched_splits(run_dir: Path, relative_cfg: RelativeOutcomeConfig) -> dict[str, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for split in REQUIRED_SPLITS:
        frame = load_latent_frame(run_dir / "latents" / f"{split}_latents.parquet")
        frame["_phase4d_split"] = split
        frame["_phase4d_row"] = np.arange(len(frame), dtype=np.int64)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    enriched = attach_relative_outcomes(combined, relative_cfg)
    result: dict[str, pd.DataFrame] = {}
    for split in REQUIRED_SPLITS:
        part = enriched.loc[enriched["_phase4d_split"] == split].sort_values("_phase4d_row")
        result[split] = part.drop(columns=["_phase4d_split", "_phase4d_row"]).reset_index(drop=True)
    return result


def _prepared_paths(run_dir: Path, output: Path, relative_cfg: RelativeOutcomeConfig, *, resume: bool) -> dict[str, Path]:
    fold, seed = _run_key(run_dir)
    base = output / "prepared" / fold / f"seed_{seed}"
    paths = {split: base / f"{split}.parquet" for split in REQUIRED_SPLITS}
    if resume and all(path.exists() for path in paths.values()):
        return paths
    base.mkdir(parents=True, exist_ok=True)
    for split, frame in _load_enriched_splits(run_dir, relative_cfg).items():
        frame.to_parquet(paths[split], index=False)
    return paths


def fit_metrics(config: dict[str, Any], output: Path, run_dirs: Iterable[Path], *, resume: bool) -> None:
    """Fit bounded retrieval adapters on frozen latent tables only."""
    relative_cfg = RelativeOutcomeConfig(**{k: v for k, v in config["relative_outcomes"].items() if k != "enabled"})
    base_cfg = dict(config["metric_learning"])
    kinds = sorted({str(item["metric"]) for item in config["variants"] if str(item["metric"]) != "original"})
    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        fold, seed = _run_key(run_dir)
        LOGGER.info("Preparing frozen latents for %s seed %s", fold, seed)
        paths = _prepared_paths(run_dir, output, relative_cfg, resume=resume)
        train = pd.read_parquet(paths["train"])
        for kind in kinds:
            LOGGER.info("Fitting %s retrieval metric for %s seed %s", kind, fold, seed)
            metric_path = output / "metrics" / fold / f"seed_{seed}" / f"{kind}.npz"
            summary_path = metric_path.with_suffix(".json")
            if resume and metric_path.exists() and summary_path.exists():
                summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
                continue
            metric_cfg = MetricLearningConfig(kind=kind, seed=seed, **base_cfg)
            metric, summary = fit_retrieval_metric(train, metric_cfg)
            metric.save(metric_path)
            record = {"fold": fold, "seed": seed, "metric_path": str(metric_path), **summary}
            _write_json(summary_path, record)
            summaries.append(record)
        calibration_path = output / "metrics" / fold / f"seed_{seed}" / "distance_calibration.json"
        calibration: dict[str, Any] = (
            json.loads(calibration_path.read_text(encoding="utf-8"))
            if resume and calibration_path.exists()
            else {}
        )
        missing_calibrations = [kind for kind in ["original", *kinds] if kind not in calibration]
        if missing_calibrations:
            for kind in missing_calibrations:
                metric = None if kind == "original" else FittedRetrievalMetric.load(
                    output / "metrics" / fold / f"seed_{seed}" / f"{kind}.npz"
                )
                calibration[kind] = _calibrate_distances(train, metric, config["ood_calibration"])
            _write_json(calibration_path, calibration)
    pd.DataFrame(summaries).to_csv(output / "metric_fit_summary.csv", index=False)


def _calibrate_distances(
    frame: pd.DataFrame,
    metric: FittedRetrievalMetric | None,
    calibration_config: dict[str, Any],
    k: int = 25,
) -> dict[str, float | int]:
    """Calibrate density and OOD thresholds from causal cross-ticker distances."""
    cols = latent_columns(frame)
    raw = frame[cols].to_numpy(dtype=np.float32)
    transformed = metric.transform(raw) if metric is not None else normalise_latents(raw, raw)[0]
    sample_size = min(int(calibration_config.get("sample_size", 2048)), len(frame))
    rng = np.random.default_rng(int(calibration_config.get("seed", 7)))
    sample_idx = np.sort(rng.choice(len(frame), size=sample_size, replace=False))
    values = np.asarray(transformed[sample_idx], dtype=np.float32)
    squared = (
        np.sum(values * values, axis=1, keepdims=True)
        + np.sum(values * values, axis=1)[None, :]
        - 2.0 * (values @ values.T)
    )
    distances = np.sqrt(np.maximum(squared, 0.0)).astype(np.float32, copy=False)
    sample = frame.iloc[sample_idx]
    timestamps = pd.to_datetime(sample["timestamp"], utc=True).astype("int64").to_numpy()
    available = pd.to_datetime(
        sample.get("outcome_available_timestamp", sample["timestamp"]), utc=True, errors="coerce"
    )
    available_ns = available.astype("int64").to_numpy()
    available_ns[available.isna().to_numpy()] = np.iinfo(np.int64).max
    tickers = sample["ticker"].astype(str).to_numpy()
    legal = (available_ns[None, :] <= timestamps[:, None]) & (tickers[None, :] != tickers[:, None])
    distances[~legal] = np.inf
    medians: list[float] = []
    for row in distances:
        finite = row[np.isfinite(row)]
        if finite.size:
            nearest = np.partition(finite, min(k, finite.size) - 1)[:k]
            medians.append(float(np.median(nearest)))
    if not medians:
        raise RuntimeError("Distance calibration found no causal cross-ticker neighbors.")
    quantile = float(calibration_config.get("distance_quantile", 0.95))
    return {
        "sample_rows": sample_size,
        "calibrated_queries": len(medians),
        "confidence_reference_distance": float(np.median(medians)),
        "max_median_distance": float(np.quantile(medians, quantile)),
        "distance_quantile": quantile,
    }


def _evaluation_config(
    config: dict[str, Any],
    output: Path,
    fold: str,
    seed: int,
    variant: dict[str, Any],
    split: str,
    prepared: dict[str, Path],
) -> dict[str, Any]:
    memory = copy.deepcopy(config["memory_defaults"])
    memory["target_alpha"] = variant.get("target_alpha")
    memory["score_mode"] = variant.get("score_mode", "legacy")
    memory.update(copy.deepcopy(variant.get("memory_overrides", {})))
    evaluation_dir = output / "evaluations" / split / variant["id"] / fold / f"seed_{seed}"
    query_path = prepared[split]
    if config.get("_smoke", False):
        query_path = query_path.with_name(f"{split}_smoke_256.parquet")
        if not query_path.exists():
            pd.read_parquet(prepared[split]).head(256).to_parquet(query_path, index=False)
    result: dict[str, Any] = {
        "data": {
            "train_latents": str(prepared["train"]),
            "test_latents": str(query_path),
            "precomputed_glob": str(_resolve(config["data"]["precomputed_glob"])),
            "output_dir": str(evaluation_dir),
        },
        "memory": memory,
        "policy": copy.deepcopy(config["policy"]),
        "evaluation": {"baselines": "", "memory_metric_target": "future_blended_alpha_63"},
        "run": {"write_neighbors": True, "write_memory_reports": False},
    }
    result["policy"].update(copy.deepcopy(variant.get("policy_overrides", {})))
    if variant.get("target_alpha") is None:
        result["policy"]["min_alpha_lcb"] = None
    metric_kind = str(variant["metric"])
    calibration_path = output / "metrics" / fold / f"seed_{seed}" / "distance_calibration.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))[metric_kind]
        result["memory"]["confidence_reference_distance"] = calibration["confidence_reference_distance"]
        result["memory"]["max_median_distance"] = calibration["max_median_distance"]
    if metric_kind != "original":
        result["retrieval_metric"] = {"path": str(output / "metrics" / fold / f"seed_{seed}" / f"{metric_kind}.npz")}
    return result


def run_evaluations(
    config: dict[str, Any],
    output: Path,
    run_dirs: Iterable[Path],
    splits: Iterable[str],
    *,
    resume: bool,
) -> None:
    """Evaluate configured semantic variants without encoder inference or training."""
    relative_cfg = RelativeOutcomeConfig(**{k: v for k, v in config["relative_outcomes"].items() if k != "enabled"})
    rows: list[dict[str, Any]] = []
    for split in splits:
        for variant in config["variants"]:
            for run_dir in run_dirs:
                fold, seed = _run_key(run_dir)
                prepared = _prepared_paths(run_dir, output, relative_cfg, resume=resume)
                eval_cfg = _evaluation_config(config, output, fold, seed, variant, split, prepared)
                eval_root = Path(eval_cfg["data"]["output_dir"]) / "eval"
                metrics_path = eval_root / "metrics.json"
                if not (resume and metrics_path.exists()):
                    run_market_memory_evaluation(eval_cfg, PROJECT_ROOT, run_id="eval")
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                rows.append({"split": split, "variant": variant["id"], "fold": fold, "seed": seed, **_flat_metrics(metrics)})
    summary_path = output / "evaluation_summary.csv"
    current = pd.DataFrame(rows)
    if summary_path.exists():
        current = pd.concat([pd.read_csv(summary_path), current], ignore_index=True)
        current = current.drop_duplicates(["split", "variant", "fold", "seed"], keep="last")
    current.to_csv(summary_path, index=False)


def _flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "total_return", "sharpe", "max_drawdown", "profit_factor", "trade_count", "win_rate",
        "average_trade_return", "capital_efficiency", "outcome_ndcg_at_25",
        "alpha_spearman_correlation", "weekly_alpha_rank_ic", "alpha_retrieval_mae",
        "downside_cvar_calibration_error", "causal_eligibility_coverage", "effective_sample_size",
    )
    return {key: metrics.get(key) for key in keys}


def _daily_returns(equity_path: Path) -> pd.Series:
    frame = pd.read_csv(equity_path)
    if frame.empty:
        return pd.Series(dtype=float)
    date_col = "timestamp" if "timestamp" in frame else "date"
    value_col = "equity" if "equity" in frame else "portfolio_value"
    values = frame[[date_col, value_col]].dropna().copy()
    values[date_col] = pd.to_datetime(values[date_col], utc=True)
    return values.set_index(date_col)[value_col].astype(float).pct_change().dropna()


def select_variant(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Apply paired, fold-aware promotion gates on the validation split only."""
    summary = pd.read_csv(output / "evaluation_summary.csv")
    split = str(config["experiment"].get("validation_split", "val"))
    summary = summary.loc[summary["split"] == split].copy()
    promotion = config["promotion"]
    baseline = str(promotion["baseline_variant"])
    baseline_rows = summary.loc[summary["variant"] == baseline].set_index(["fold", "seed"])
    candidates: list[dict[str, Any]] = []
    return_series: dict[str, pd.Series] = {}
    for variant in [str(item["id"]) for item in config["variants"]]:
        rows = summary.loc[summary["variant"] == variant].set_index(["fold", "seed"])
        paired_index = rows.index.intersection(baseline_rows.index)
        delta = rows.loc[paired_index, "total_return"] - baseline_rows.loc[paired_index, "total_return"]
        fold_delta = delta.groupby(level=0).mean()
        drawdown_regression = (
            rows.loc[paired_index, "max_drawdown"].abs() - baseline_rows.loc[paired_index, "max_drawdown"].abs()
        )
        memory_delta = rows.loc[paired_index, "outcome_ndcg_at_25"] - baseline_rows.loc[paired_index, "outcome_ndcg_at_25"]
        paired_days: list[pd.DataFrame] = []
        candidate_days: list[pd.Series] = []
        for fold, seed in paired_index:
            candidate_path = output / "evaluations" / split / variant / fold / f"seed_{seed}" / "eval" / "equity_curve.csv"
            baseline_path = output / "evaluations" / split / baseline / fold / f"seed_{seed}" / "eval" / "equity_curve.csv"
            left = _daily_returns(candidate_path).rename("candidate")
            right = _daily_returns(baseline_path).rename("baseline")
            paired_days.append(pd.concat([left, right], axis=1).dropna())
            candidate_days.append(left.rename(f"{fold}_{seed}"))
        paired = pd.concat(paired_days, ignore_index=True) if paired_days else pd.DataFrame()
        bootstrap = stationary_bootstrap_delta(
            paired.get("candidate", pd.Series(dtype=float)).to_numpy(),
            paired.get("baseline", pd.Series(dtype=float)).to_numpy(),
            samples=int(promotion["bootstrap_samples"]),
            expected_block_length=int(promotion["bootstrap_block_length"]),
            seed=7,
        ) if len(paired) else None
        joined = pd.concat(candidate_days, axis=1).fillna(0.0) if candidate_days else pd.DataFrame()
        return_series[variant] = joined.mean(axis=1) if not joined.empty else pd.Series(dtype=float)
        record = {
            "variant": variant,
            "paired_runs": int(len(paired_index)),
            "mean_return": float(rows["total_return"].mean()),
            "mean_sharpe": float(rows["sharpe"].mean()),
            "mean_return_delta": float(delta.mean()),
            "fold_win_rate": float((fold_delta > 0).mean()),
            "seed_win_rate": float((delta > 0).mean()),
            "worst_drawdown_regression": float(drawdown_regression.max()),
            "mean_memory_ndcg_gain": float(memory_delta.mean()),
            "bootstrap_delta_low": bootstrap.ci_low if bootstrap else None,
            "bootstrap_probability_positive": bootstrap.probability_positive if bootstrap else None,
        }
        record["passes_gates"] = bool(
            variant != baseline
            and record["fold_win_rate"] >= float(promotion["minimum_fold_win_rate"])
            and record["seed_win_rate"] >= float(promotion["minimum_seed_win_rate"])
            and record["worst_drawdown_regression"] <= float(promotion["maximum_drawdown_regression"])
            and record["mean_memory_ndcg_gain"] >= float(promotion["minimum_memory_metric_gain"])
            and (record["bootstrap_delta_low"] is not None and record["bootstrap_delta_low"] > 0.0)
        )
        candidates.append(record)
    matrix = pd.concat(return_series, axis=1).fillna(0.0) if return_series else pd.DataFrame()
    pbo = probability_of_backtest_overfitting(matrix.to_numpy()) if matrix.shape[1] > 1 else 0.0
    trial_count = max(len(return_series), 1)
    for record in candidates:
        values = return_series[record["variant"]].to_numpy()
        record["deflated_sharpe_probability"] = deflated_sharpe_probability(values, trial_count) if values.size else 0.0
        record["pbo"] = pbo
        record["passes_gates"] = bool(
            record["passes_gates"]
            and pbo <= float(promotion["maximum_pbo"])
            and record["deflated_sharpe_probability"] >= float(promotion["minimum_deflated_sharpe_probability"])
        )
    ladder = pd.DataFrame(candidates).sort_values(
        ["passes_gates", "mean_return_delta", "mean_memory_ndcg_gain"], ascending=[False, False, False]
    )
    ladder.to_csv(output / "ablation_ladder.csv", index=False)
    promoted = ladder.loc[ladder["passes_gates"]]
    selected_id = str(promoted.iloc[0]["variant"]) if not promoted.empty else baseline
    selected_variant = next(item for item in config["variants"] if item["id"] == selected_id)
    selected_policy = copy.deepcopy(config["policy"])
    if selected_variant.get("target_alpha") is None:
        selected_policy["min_alpha_lcb"] = None
    selected = {
        "selected_variant": selected_id,
        "fallback_to_baseline": bool(promoted.empty),
        "selection_split": split,
        "variant": selected_variant,
        "memory": {**config["memory_defaults"], "target_alpha": selected_variant.get("target_alpha"), "score_mode": selected_variant.get("score_mode")},
        "policy": selected_policy,
        "promotion": promotion,
        "smoke_only": bool(config.get("_smoke", False)),
    }
    _write_yaml(output / "selected_stack.yaml", selected)
    return selected


def run_ensemble(config: dict[str, Any], output: Path, selected: dict[str, Any], split: str) -> None:
    ensemble_cfg = EnsembleConfig(**{key: value for key, value in config["ensemble"].items() if key not in {"fold", "seeds"}})
    fold = str(config["ensemble"]["fold"])
    variant = str(selected["selected_variant"])
    frames: list[pd.DataFrame] = []
    for seed in config["ensemble"]["seeds"]:
        path = output / "evaluations" / split / variant / fold / f"seed_{seed}" / "eval" / "signals.parquet"
        if not path.exists():
            LOGGER.warning("Skipping unavailable ensemble signal file: %s", path)
            continue
        frame = pd.read_parquet(path)
        frame["seed"] = int(seed)
        frames.append(frame)
    if len(frames) < ensemble_cfg.minimum_votes:
        LOGGER.warning("Not enough completed seeds for %s %s ensemble.", fold, split)
        return
    signals = combine_seed_signals({int(frame["seed"].iloc[0]): frame for frame in frames}, ensemble_cfg)
    decisions: list[dict[str, Any]] = []
    trades, equity = run_long_only_backtest(signals, PolicyConfig(**selected["policy"]), decision_log=decisions)
    metrics = compute_backtest_metrics(trades, equity, float(selected["policy"]["initial_capital"]))
    base = output / "ensemble" / split / fold
    base.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(base / "signals.parquet", index=False)
    trades.to_csv(base / "trades.csv", index=False)
    equity.to_csv(base / "equity_curve.csv", index=False)
    pd.DataFrame(decisions).to_csv(base / "decisions.csv", index=False)
    _write_json(base / "metrics.json", metrics)


def run(args: argparse.Namespace) -> Path:
    config_path = _resolve(args.config)
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)
    if args.smoke:
        config["_smoke"] = True
        config["metric_learning"].update(
            {"steps": 2, "validation_interval": 1, "patience": 2, "batch_size": 64, "device": "cpu"}
        )
        config["variants"] = [
            item
            for item in config["variants"]
            if item["id"] in {"A0_legacy_identity", "A3_blended_alpha_diagonal", "A4_blended_alpha_low_rank"}
        ]
    output = _ensure_output(config, args.run_id, resume=args.resume)
    manifest = audit_sources(config, output)
    source = _resolve(config["experiment"]["source_run"])
    run_dirs = _select_runs(_complete_run_dirs(source), config["experiment"], args.max_runs)
    if args.stage == "audit":
        return output
    if args.stage in {"fit-metrics", "all"}:
        fit_metrics(config, output, run_dirs, resume=args.resume)
    if args.stage in {"ablate", "all"}:
        run_evaluations(config, output, run_dirs, [config["experiment"]["validation_split"]], resume=args.resume)
    if args.stage in {"select", "all"}:
        selected = select_variant(config, output)
    else:
        selected_path = output / "selected_stack.yaml"
        selected = _read_yaml(selected_path) if selected_path.exists() else None
    if args.stage in {"diagnostic-2024", "all"}:
        if selected is None:
            raise FileNotFoundError("Run the select stage before diagnostic-2024.")
        selected_id = selected["selected_variant"]
        diagnostic_config = copy.deepcopy(config)
        diagnostic_config["variants"] = [item for item in config["variants"] if item["id"] == selected_id]
        splits = config["experiment"].get("diagnostic_splits", ["test", "holdout"])
        run_evaluations(diagnostic_config, output, run_dirs, splits, resume=args.resume)
        for split in splits:
            run_ensemble(config, output, selected, str(split))
    _write_json(output / "run_summary.json", {"stage": args.stage, "run_count": len(run_dirs), "manifest": manifest})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4d_retrieval_semantics.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit runs for smoke verification.")
    parser.add_argument("--smoke", action="store_true", help="Use a two-step CPU adapter fit and two variants.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    output = run(args)
    print(output)


if __name__ == "__main__":
    main()
