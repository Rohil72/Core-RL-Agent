"""Build and train the Phase 5 causal decision adapter over frozen Phase 4 states."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase4d_retrieval_semantics import _complete_run_dirs, _run_key, _select_runs  # noqa: E402
from src.decision.adapter import DecisionAdapterConfig  # noqa: E402
from src.decision.dataset import DecisionDatasetConfig, build_decision_frame  # noqa: E402
from src.decision.losses import DecisionLossConfig  # noqa: E402
from src.decision.trainer import DecisionTrainingConfig, train_decision_adapter  # noqa: E402
from src.backtest.market_memory_evaluator import run_market_memory_evaluation  # noqa: E402
from src.memory.experience import latent_columns  # noqa: E402


@dataclass(frozen=True)
class AdapterSource:
    """One frozen train/validation latent source and its matching price data."""

    name: str
    group: str
    seed: int
    train_path: Path
    val_path: Path
    precomputed_globs: tuple[str, ...]
    backtest_glob: str
    signal_path: Path | None = None


def _phase6_sources(
    experiment: dict[str, Any],
    max_runs: int | None,
    project_root: Path,
) -> list[AdapterSource]:
    source_root = project_root / experiment["source_testbed_run"]
    latent_root = source_root / "latents"
    representations = tuple(experiment.get("source_representations", ("regional", "global")))
    market_priority = tuple(experiment.get("active_markets", ()))
    seed_priority = tuple(int(seed) for seed in experiment.get("active_seeds", ()))
    active_markets = set(market_priority)
    active_seeds = set(seed_priority)
    data_root = str(experiment.get("data_root", "data/international")).rstrip("/\\")
    sources: list[AdapterSource] = []
    if not latent_root.exists():
        raise RuntimeError(
            f"Phase 6 latent root does not exist: {latent_root}. "
            "Expected completed train_latents.parquet and val_latents.parquet artifacts."
        )
    directories = sorted((path for path in latent_root.iterdir() if path.is_dir()), key=lambda path: path.name)
    for representation in representations:
        for directory in directories:
            regional = re.fullmatch(r"regional_(.+)_seed_(\d+)", directory.name)
            global_match = re.fullmatch(r"global_seed_(\d+)", directory.name)
            if representation == "regional" and regional:
                market, seed = regional.group(1), int(regional.group(2))
                group = f"regional_{market}"
                globs = (f"{data_root}/{market}/*.parquet",)
                backtest_glob = globs[0]
            elif representation == "global" and global_match:
                market, seed = "global", int(global_match.group(1))
                group = "global"
                globs = (f"{data_root}/*/*.parquet",)
                backtest_glob = globs[0]
            else:
                continue
            if active_markets and market != "global" and market not in active_markets:
                continue
            if active_seeds and seed not in active_seeds:
                continue
            train_path = directory / "train_latents.parquet"
            val_path = directory / "val_latents.parquet"
            if train_path.exists() and val_path.exists():
                sources.append(
                    AdapterSource(
                        name=directory.name,
                        group=group,
                        seed=seed,
                        train_path=train_path,
                        val_path=val_path,
                        precomputed_globs=globs,
                        backtest_glob=backtest_glob,
                    )
                )
    if not sources:
        found = [path.name for path in directories]
        raise RuntimeError(
            f"No complete Phase 6 latent sources matched under {latent_root}. "
            f"Found directories: {found or 'none'}; representations={representations}; "
            f"active_markets={sorted(active_markets) or 'all'}; active_seeds={sorted(active_seeds) or 'all'}."
        )
    representation_order = {name: index for index, name in enumerate(representations)}
    market_order = {name: index for index, name in enumerate(market_priority)}
    seed_order = {seed: index for index, seed in enumerate(seed_priority)}
    sources.sort(
        key=lambda source: (
            representation_order.get("global" if source.group == "global" else "regional", len(representations)),
            market_order.get(source.group.removeprefix("regional_"), len(market_order)),
            seed_order.get(source.seed, len(seed_order)),
            source.name,
        )
    )
    return sources[:max_runs] if max_runs is not None else sources


def discover_adapter_sources(
    config: dict[str, Any],
    max_runs: int | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[AdapterSource]:
    """Resolve frozen latent inputs for legacy Phase 4C or the Phase 6 testbed."""

    experiment = config["experiment"]
    source_mode = str(experiment.get("source_mode", "phase4c"))
    if source_mode == "phase6_testbed":
        return _phase6_sources(experiment, max_runs, project_root)
    if source_mode != "phase4c":
        raise ValueError(f"Unsupported adapter source_mode: {source_mode}")
    source = project_root / experiment["source_phase4c_run"]
    filter_cfg = {
        "active_folds": experiment.get("active_folds", []),
        "active_seeds": experiment.get("active_seeds", []),
    }
    runs = _select_runs(_complete_run_dirs(source), filter_cfg, max_runs)
    resolved: list[AdapterSource] = []
    for run_dir in runs:
        fold, seed = _run_key(run_dir)
        prepared = project_root / experiment["source_phase4e_run"] / "prepared" / fold / f"seed_{seed}"
        train_path = prepared / "train.parquet"
        val_path = prepared / "val.parquet"
        if not train_path.exists() or not val_path.exists():
            train_path = run_dir / "latents" / "train_latents.parquet"
            val_path = run_dir / "latents" / "val_latents.parquet"
        signal_path = (
            project_root / experiment["source_phase4e_run"] / "evaluations" / "val"
            / "A2_blended_alpha_identity" / fold / f"seed_{seed}" / "eval" / "signals.parquet"
        )
        glob_value = str(experiment["precomputed_glob"])
        resolved.append(
            AdapterSource(
                name=f"{fold}_seed_{seed}",
                group=fold,
                seed=seed,
                train_path=train_path,
                val_path=val_path,
                precomputed_globs=(glob_value,),
                backtest_glob=glob_value,
                signal_path=signal_path if signal_path.exists() else None,
            )
        )
    return resolved


def _retrieval_frame(path: Path, destination: Path) -> None:
    """Make the existing evaluator consume decision space without changing it."""
    frame = pd.read_parquet(path)
    frame = frame.drop(columns=latent_columns(frame))
    decision_cols = sorted(
        [c for c in frame if c.startswith("decision_") and c.removeprefix("decision_").isdigit()],
        key=lambda c: int(c.removeprefix("decision_")),
    )
    frame = frame.rename(columns={column: f"latent_{index}" for index, column in enumerate(decision_cols)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def _adapter_backtest(
    config: dict[str, Any],
    run_output: Path,
    source: AdapterSource,
) -> dict[str, Any]:
    phase4e_path = PROJECT_ROOT / config["experiment"]["source_phase4e_run"] / "resolved_config.yaml"
    if phase4e_path.exists():
        phase4e = yaml.safe_load(phase4e_path.read_text(encoding="utf-8"))
    else:
        phase4e = yaml.safe_load((PROJECT_ROOT / "configs/phase4e_cross_market_sharpe.yaml").read_text(encoding="utf-8"))
    retrieval_train = run_output / "retrieval" / "train.parquet"
    retrieval_val = run_output / "retrieval" / "val.parquet"
    _retrieval_frame(run_output / "train_decisions.parquet", retrieval_train)
    _retrieval_frame(run_output / "val_decisions.parquet", retrieval_val)
    evaluation = {
        "data": {
            "train_latents": str(retrieval_train),
            "test_latents": str(retrieval_val),
            "precomputed_glob": source.backtest_glob,
            "output_dir": str(run_output / "adapter_backtest"),
        },
        "memory": {
            **phase4e["memory_defaults"],
            "target_upside": "decision_mfe",
            "target_alpha": "decision_net_alpha",
            "target_downside": "decision_mae",
            "target_path_quality": "decision_path_quality",
            "target_holding_period": "decision_holding_sessions",
            "score_mode": "alpha_lcb",
            "require_outcome_availability": False,
            "same_ticker_mode": "exclude",
        },
        "policy": phase4e["policy"],
        "evaluation": {"baselines": "", "memory_metric_target": "decision_net_alpha"},
        "run": {"write_neighbors": False, "write_memory_reports": False},
    }
    result_dir = run_market_memory_evaluation(evaluation, PROJECT_ROOT, run_id="eval")
    return json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))


def _pooled_adapter_sharpe(output: Path) -> float | None:
    series = []
    for path in output.glob("fold_*/seed_*/adapter_backtest/eval/equity_curve.csv"):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        series.append(frame.set_index("timestamp")["equity"].astype(float).pct_change().rename(path.parent.parent.parent.parent.name))
    if not series:
        return None
    returns = pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna().to_numpy(dtype=float)
    if len(returns) < 3 or np.std(returns, ddof=1) <= 1e-12:
        return 0.0
    return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252.0))


def _dataclass_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in fields(cls)}
    result = {key: value for key, value in values.items() if key in names}
    for key, value in list(result.items()):
        if key in {"horizons", "quantiles", "temporal_horizons"}:
            result[key] = tuple(value)
    return result


def run(
    config_path: str,
    run_id: str,
    max_runs: int | None = None,
    resume: bool = False,
    skip_backtest: bool = False,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output = PROJECT_ROOT / experiment["output_root"] / run_id
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to reuse non-empty Phase 5 directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    sources = discover_adapter_sources(config, max_runs)
    dataset_cfg = DecisionDatasetConfig(**_dataclass_kwargs(DecisionDatasetConfig, config["decision_dataset"]))
    adapter_cfg = DecisionAdapterConfig(**_dataclass_kwargs(DecisionAdapterConfig, config["adapter"]))
    if dataset_cfg.temporal_horizons != adapter_cfg.temporal_horizons:
        raise ValueError("decision_dataset and adapter temporal_horizons must match exactly.")
    loss_cfg = DecisionLossConfig(**_dataclass_kwargs(DecisionLossConfig, config["loss"]))
    summaries: list[dict[str, Any]] = []
    for source in sources:
        fold, seed = source.group, source.seed
        run_output = output / fold / f"seed_{seed}"
        summary_path = run_output / "summary.json"
        if resume and summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if not skip_backtest and "backtest" not in existing:
                existing["backtest"] = _adapter_backtest(config, run_output, source)
                summary_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
            summaries.append({"fold": fold, "seed": seed, **existing})
            continue
        signals = pd.read_parquet(source.signal_path) if source.signal_path else None
        train_frame = build_decision_frame(
            pd.read_parquet(source.train_path), source.precomputed_globs, dataset_cfg
        )
        val_frame = build_decision_frame(
            pd.read_parquet(source.val_path), source.precomputed_globs, dataset_cfg, signals
        )
        train_cfg_values = dict(config["training"])
        train_cfg_values["seed"] = seed
        train_cfg = DecisionTrainingConfig(**_dataclass_kwargs(DecisionTrainingConfig, train_cfg_values))
        summary = train_decision_adapter(train_frame, val_frame, run_output, adapter_cfg, train_cfg, loss_cfg)
        if not skip_backtest:
            backtest = _adapter_backtest(config, run_output, source)
            summary["backtest"] = backtest
            summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summaries.append({"fold": fold, "seed": seed, **summary})
    flat = [
        {
            "fold": item["fold"], "seed": item["seed"],
            "best_validation_loss": item["best_validation_loss"], **item["diagnostics"],
            **{f"temporal_{key}": value for key, value in item.get("temporal_diagnostics", {}).items()},
            **{f"opportunity_{key}": value for key, value in item.get("opportunity_diagnostics", {}).items()},
            **{f"raw_{key}": value for key, value in item["raw_latent_diagnostics"].items()},
            **{f"backtest_{key}": item.get("backtest", {}).get(key) for key in ("total_return", "sharpe", "max_drawdown", "trade_count")},
        }
        for item in summaries
    ]
    summary_frame = pd.DataFrame(flat)
    summary_frame.to_csv(output / "adapter_summary.csv", index=False)
    fold_means = summary_frame.groupby("fold", as_index=False).mean(numeric_only=True)
    gates = config["adapter_gate"]
    utility_wins = int((fold_means["top3_mean_utility"] > fold_means["raw_top3_mean_utility"]).sum())
    correlation_wins = int(
        (fold_means["neighbor_utility_spearman"] > fold_means["raw_neighbor_utility_spearman"]).sum()
    )
    fold_wins = min(utility_wins, correlation_wins)
    pooled_sharpe = _pooled_adapter_sharpe(output)
    backtest_sharpe = pd.to_numeric(summary_frame.get("backtest_sharpe"), errors="coerce")
    backtest_drawdown = pd.to_numeric(summary_frame.get("backtest_max_drawdown"), errors="coerce")
    backtest_trades = pd.to_numeric(summary_frame.get("backtest_trade_count"), errors="coerce")
    trading_gate = bool(
        pooled_sharpe is not None
        and pooled_sharpe >= float(gates["minimum_oof_sharpe"])
        and backtest_drawdown.notna().any()
        and abs(float(backtest_drawdown.min())) <= float(gates["maximum_drawdown"])
        and backtest_trades.notna().any()
        and float(backtest_trades.mean()) >= float(gates["minimum_trades"])
    )
    retrieval_gate = fold_wins >= int(gates["minimum_fold_wins"])
    promotion = {
        "status": "eligible_for_alignment" if retrieval_gate and trading_gate else "rejected",
        "folds_passing_both_retrieval_gates": fold_wins,
        "folds_with_top3_utility_improvement": utility_wins,
        "folds_with_neighbor_correlation_improvement": correlation_wins,
        "required_fold_wins": int(gates["minimum_fold_wins"]),
        "pooled_oof_sharpe": pooled_sharpe,
        "mean_backtest_sharpe": float(backtest_sharpe.mean()) if backtest_sharpe.notna().any() else None,
        "worst_backtest_drawdown": float(backtest_drawdown.min()) if backtest_drawdown.notna().any() else None,
        "mean_trade_count": float(backtest_trades.mean()) if backtest_trades.notna().any() else None,
        "retrieval_gate_passed": retrieval_gate,
        "trading_gate_passed": trading_gate,
    }
    (output / "adapter_promotion.json").write_text(json.dumps(promotion, indent=2), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "source_run_count": len(sources),
        "sources": [source.name for source in sources],
        "source_mode": str(experiment.get("source_mode", "phase4c")),
        "output": str(output),
        "stages": ["decision_dataset", "adapter"],
    }
    (output / "run_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_decision_alignment.yaml")
    parser.add_argument("--run-id", default="phase5_v1")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.max_runs, args.resume, args.skip_backtest), indent=2))


if __name__ == "__main__":
    main()
