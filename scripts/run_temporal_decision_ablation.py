"""Run the frozen-latent temporal and decision-loss factorial ablation."""

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
                "study": "temporal_decision_ablation",
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
    result = {
        "run_id": run_id,
        "output": str(root),
        "variants": names,
        "completed_variants": completed,
        "max_source_runs_per_variant": max_runs,
        "resolved_source_count": len(sources),
        "resolved_sources": [source.name for source in sources],
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
