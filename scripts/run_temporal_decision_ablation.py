"""Run a configurable frozen-latent decision-loss ablation."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase5_decision_alignment import (  # noqa: E402
    discover_adapter_sources,
    run as run_variant,
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _aggregate(root: Path, variants: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        path = root / variant / "adapter_summary.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number")
        row: dict[str, Any] = {"variant": variant, "run_count": int(len(frame))}
        for column in numeric:
            values = pd.to_numeric(numeric[column], errors="coerce")
            row[f"mean_{column}"] = float(values.mean()) if values.notna().any() else None
        if "backtest_max_drawdown" in frame:
            drawdown = pd.to_numeric(frame["backtest_max_drawdown"], errors="coerce")
            row["worst_backtest_drawdown"] = float(drawdown.min()) if drawdown.notna().any() else None
        for column in ("backtest_sharpe", "learned_action_sharpe", "memory_gated_sharpe"):
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                row[f"median_{column}"] = float(values.median())
                row[f"std_{column}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                row[f"worst_{column}"] = float(values.min())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty and (summary["variant"] == "baseline").any():
        baseline = summary.loc[summary["variant"] == "baseline"].iloc[0]
        for metric in (
            "mean_backtest_sharpe",
            "mean_top3_mean_utility",
            "mean_neighbor_utility_spearman",
            "mean_temporal_event_brier",
            "mean_opportunity_opportunity_regret",
        ):
            if metric in summary and pd.notna(baseline.get(metric)):
                summary[f"delta_{metric}"] = summary[metric] - float(baseline[metric])
    return summary


def _paired_robustness(
    root: Path,
    comparison: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
    """Compare a candidate against its matching baseline market and seed runs."""

    baseline_name = str(comparison["baseline"])
    candidate_name = str(comparison["candidate"])
    baseline_path = root / baseline_name / "adapter_summary.csv"
    candidate_path = root / candidate_name / "adapter_summary.csv"
    if not baseline_path.exists() or not candidate_path.exists():
        return None
    keys = ["fold", "seed"]
    metric_namespace = str(comparison.get("metric_namespace", "backtest"))
    source_metrics = {
        f"{metric_namespace}_sharpe": "backtest_sharpe",
        f"{metric_namespace}_total_return": "backtest_total_return",
        f"{metric_namespace}_max_drawdown": "backtest_max_drawdown",
    }
    baseline_frame = pd.read_csv(baseline_path)
    candidate_frame = pd.read_csv(candidate_path)
    missing = set(source_metrics).difference(baseline_frame.columns).union(
        set(source_metrics).difference(candidate_frame.columns)
    )
    if missing:
        raise ValueError(
            f"Robustness comparison lacks {metric_namespace} metrics: {sorted(missing)}"
        )
    baseline = baseline_frame[keys + list(source_metrics)].rename(columns=source_metrics)
    candidate = candidate_frame[keys + list(source_metrics)].rename(columns=source_metrics)
    paired = baseline.merge(candidate, on=keys, suffixes=("_baseline", "_candidate"), validate="one_to_one")
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise RuntimeError("Robustness comparison requires complete one-to-one baseline/candidate pairs.")
    paired["market"] = paired["fold"].astype(str).str.removeprefix("regional_")
    paired["sharpe_delta"] = paired["backtest_sharpe_candidate"] - paired["backtest_sharpe_baseline"]
    paired["return_delta"] = paired["backtest_total_return_candidate"] - paired["backtest_total_return_baseline"]
    paired["drawdown_improvement"] = (
        paired["backtest_max_drawdown_candidate"] - paired["backtest_max_drawdown_baseline"]
    )
    paired["sharpe_win"] = paired["sharpe_delta"] > 0.0
    paired["drawdown_win"] = paired["drawdown_improvement"] >= 0.0
    market = (
        paired.groupby("market", as_index=False)
        .agg(
            run_count=("seed", "count"),
            baseline_mean_sharpe=("backtest_sharpe_baseline", "mean"),
            candidate_mean_sharpe=("backtest_sharpe_candidate", "mean"),
            mean_sharpe_delta=("sharpe_delta", "mean"),
            paired_seed_wins=("sharpe_win", "sum"),
            baseline_worst_drawdown=("backtest_max_drawdown_baseline", "min"),
            candidate_worst_drawdown=("backtest_max_drawdown_candidate", "min"),
        )
    )
    market["market_win"] = market["mean_sharpe_delta"] > 0.0
    gates = comparison.get("gates", {})
    baseline_sharpe = paired["backtest_sharpe_baseline"].astype(float)
    candidate_sharpe = paired["backtest_sharpe_candidate"].astype(float)
    paired_win_fraction = float(paired["sharpe_win"].mean())
    drawdown_win_fraction = float(paired["drawdown_win"].mean())
    baseline_worst_drawdown = float(paired["backtest_max_drawdown_baseline"].min())
    candidate_worst_drawdown = float(paired["backtest_max_drawdown_candidate"].min())
    verdict = {
        "status": "robustness_supported",
        "baseline": baseline_name,
        "candidate": candidate_name,
        "metric_namespace": metric_namespace,
        "paired_run_count": int(len(paired)),
        "paired_run_wins": int(paired["sharpe_win"].sum()),
        "paired_run_win_fraction": paired_win_fraction,
        "drawdown_win_fraction": drawdown_win_fraction,
        "market_count": int(len(market)),
        "market_wins": int(market["market_win"].sum()),
        "baseline_mean_sharpe": float(baseline_sharpe.mean()),
        "candidate_mean_sharpe": float(candidate_sharpe.mean()),
        "mean_sharpe_delta": float(paired["sharpe_delta"].mean()),
        "baseline_worst_sharpe": float(baseline_sharpe.min()),
        "candidate_worst_sharpe": float(candidate_sharpe.min()),
        "worst_sharpe_delta": float(candidate_sharpe.min() - baseline_sharpe.min()),
        "baseline_sharpe_std": float(baseline_sharpe.std(ddof=1)),
        "candidate_sharpe_std": float(candidate_sharpe.std(ddof=1)),
        "baseline_worst_drawdown": baseline_worst_drawdown,
        "candidate_worst_drawdown": candidate_worst_drawdown,
        "worst_drawdown_improvement": candidate_worst_drawdown - baseline_worst_drawdown,
        "diagnostic_only": True,
        "promotion_allowed": False,
    }
    checks = {
        "market_wins": verdict["market_wins"] >= int(gates.get("minimum_market_wins", 0)),
        "paired_run_win_fraction": paired_win_fraction
        >= float(gates.get("minimum_paired_run_win_fraction", 0.0)),
        "mean_sharpe_delta": verdict["mean_sharpe_delta"]
        >= float(gates.get("minimum_mean_sharpe_delta", 0.0)),
        "worst_sharpe_delta": verdict["worst_sharpe_delta"]
        >= float(gates.get("minimum_worst_sharpe_delta", 0.0)),
        "drawdown_win_fraction": drawdown_win_fraction
        >= float(gates.get("minimum_drawdown_win_fraction", 0.0)),
        "worst_drawdown_improvement": verdict["worst_drawdown_improvement"]
        >= float(gates.get("minimum_worst_drawdown_improvement", 0.0)),
    }
    verdict["checks"] = checks
    if not all(checks.values()):
        verdict["status"] = "robustness_not_supported"
    return paired, market, verdict


def run(
    config_path: str,
    run_id: str,
    *,
    selected_variants: set[str] | None = None,
    max_runs: int | None = None,
    epochs_override: int | None = None,
    resume: bool = False,
    skip_backtest: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    study_path = Path(config_path)
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    base_path = PROJECT_ROOT / study["study"]["base_config"]
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    variants = study["variants"]
    unknown = (selected_variants or set()).difference(variants)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    names = [name for name in variants if not selected_variants or name in selected_variants]
    root = PROJECT_ROOT / study["study"]["output_root"] / run_id
    generated = root / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)
    preview_override = {key: value for key, value in variants[names[0]].items() if key != "description"}
    preview = _deep_merge(_deep_merge(base, study.get("common", {})), preview_override)
    sources = discover_adapter_sources(preview, max_runs)
    expected_source_count = study["study"].get("expected_source_count")
    if max_runs is None and expected_source_count is not None and len(sources) != int(expected_source_count):
        raise RuntimeError(
            f"Expected {expected_source_count} complete sources, but resolved {len(sources)}."
        )
    preflight = {
        "source_mode": preview["experiment"].get("source_mode", "phase4c"),
        "source_count": len(sources),
        "sources": [
            {
                "name": source.name,
                "group": source.group,
                "seed": source.seed,
                "train_latents": str(source.train_path),
                "val_latents": str(source.val_path),
                "precomputed_globs": list(source.precomputed_globs),
            }
            for source in sources
        ],
    }
    (root / "source_preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if preflight_only:
        return {
            "run_id": run_id,
            "output": str(root),
            "status": "preflight_ready",
            **preflight,
        }
    completed: list[str] = []
    for name in names:
        variant_override = {key: value for key, value in variants[name].items() if key != "description"}
        resolved = _deep_merge(_deep_merge(base, study.get("common", {})), variant_override)
        if epochs_override is not None:
            resolved["training"]["epochs"] = int(epochs_override)
            resolved["training"]["minimum_epochs"] = min(
                int(resolved["training"].get("minimum_epochs", epochs_override)),
                int(epochs_override),
            )
        resolved["experiment"]["output_root"] = str(root.relative_to(PROJECT_ROOT))
        resolved.setdefault("study_metadata", {})
        resolved["study_metadata"].update(
            {
                "study": str(study["study"].get("name", "temporal_decision_ablation")),
                "variant": name,
                "description": variants[name].get("description", ""),
                "diagnostic_only": bool(study["study"].get("diagnostic_only", True)),
            }
        )
        generated_path = generated / f"{name}.yaml"
        generated_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        run_variant(
            str(generated_path),
            name,
            max_runs=max_runs,
            resume=resume,
            skip_backtest=skip_backtest,
        )
        completed.append(name)
        partial = _aggregate(root, completed)
        partial.to_csv(root / "variant_summary.csv", index=False)
    summary = _aggregate(root, names)
    summary.to_csv(root / "variant_summary.csv", index=False)
    robustness = _paired_robustness(root, study.get("comparison", {})) if study.get("comparison") else None
    if robustness is not None:
        paired, market, verdict = robustness
        paired.to_csv(root / "paired_results.csv", index=False)
        market.to_csv(root / "market_robustness.csv", index=False)
        (root / "robustness_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    else:
        verdict = None
    result = {
        "run_id": run_id,
        "output": str(root),
        "variants": names,
        "completed_variants": completed,
        "max_source_runs_per_variant": max_runs,
        "resolved_source_count": len(sources),
        "resolved_sources": [source.name for source in sources],
        "robustness_status": verdict["status"] if verdict else None,
        "epochs_override": epochs_override,
        "diagnostic_only": True,
        "promotion_allowed": False,
    }
    (root / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/temporal_decision_ablation.yaml")
    parser.add_argument("--run-id", default="temporal_decision_ablation_v1")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.config,
                args.run_id,
                selected_variants=set(args.variants) if args.variants else None,
                max_runs=args.max_runs,
                epochs_override=args.epochs,
                resume=args.resume,
                skip_backtest=args.skip_backtest,
                preflight_only=args.preflight,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
