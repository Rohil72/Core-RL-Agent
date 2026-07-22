"""Build and train the Phase 5 causal decision adapter over frozen Phase 4 states."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
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


def _adapter_backtest(config: dict[str, Any], run_output: Path) -> dict[str, Any]:
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
            "precomputed_glob": str(PROJECT_ROOT / config["experiment"]["precomputed_glob"]),
            "output_dir": str(run_output / "adapter_backtest"),
        },
        "memory": {
            **phase4e["memory_defaults"],
            "target_alpha": "future_blended_alpha_63",
            "score_mode": "alpha_lcb",
        },
        "policy": phase4e["policy"],
        "evaluation": {"baselines": "", "memory_metric_target": "future_blended_alpha_63"},
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
        if key in {"horizons", "quantiles"}:
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
    source = PROJECT_ROOT / experiment["source_phase4c_run"]
    filter_cfg = {
        "active_folds": experiment.get("active_folds", []),
        "active_seeds": experiment.get("active_seeds", []),
    }
    runs = _select_runs(_complete_run_dirs(source), filter_cfg, max_runs)
    dataset_cfg = DecisionDatasetConfig(**_dataclass_kwargs(DecisionDatasetConfig, config["decision_dataset"]))
    adapter_cfg = DecisionAdapterConfig(**_dataclass_kwargs(DecisionAdapterConfig, config["adapter"]))
    loss_cfg = DecisionLossConfig(**_dataclass_kwargs(DecisionLossConfig, config["loss"]))
    summaries: list[dict[str, Any]] = []
    for run_dir in runs:
        fold, seed = _run_key(run_dir)
        run_output = output / fold / f"seed_{seed}"
        summary_path = run_output / "summary.json"
        if resume and summary_path.exists():
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if not skip_backtest and "backtest" not in existing:
                existing["backtest"] = _adapter_backtest(config, run_output)
                summary_path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
            summaries.append({"fold": fold, "seed": seed, **existing})
            continue
        prepared = PROJECT_ROOT / experiment["source_phase4e_run"] / "prepared" / fold / f"seed_{seed}"
        train_path = prepared / "train.parquet"
        val_path = prepared / "val.parquet"
        if not train_path.exists() or not val_path.exists():
            train_path = run_dir / "latents" / "train_latents.parquet"
            val_path = run_dir / "latents" / "val_latents.parquet"
        signal_path = (
            PROJECT_ROOT / experiment["source_phase4e_run"] / "evaluations" / "val"
            / "A2_blended_alpha_identity" / fold / f"seed_{seed}" / "eval" / "signals.parquet"
        )
        signals = pd.read_parquet(signal_path) if signal_path.exists() else None
        train_frame = build_decision_frame(pd.read_parquet(train_path), experiment["precomputed_glob"], dataset_cfg)
        val_frame = build_decision_frame(pd.read_parquet(val_path), experiment["precomputed_glob"], dataset_cfg, signals)
        train_cfg_values = dict(config["training"])
        train_cfg_values["seed"] = seed
        train_cfg = DecisionTrainingConfig(**_dataclass_kwargs(DecisionTrainingConfig, train_cfg_values))
        summary = train_decision_adapter(train_frame, val_frame, run_output, adapter_cfg, train_cfg, loss_cfg)
        if not skip_backtest:
            backtest = _adapter_backtest(config, run_output)
            summary["backtest"] = backtest
            summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        summaries.append({"fold": fold, "seed": seed, **summary})
    flat = [
        {
            "fold": item["fold"], "seed": item["seed"],
            "best_validation_loss": item["best_validation_loss"], **item["diagnostics"],
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
    manifest = {"run_id": run_id, "source_run_count": len(runs), "output": str(output), "stages": ["decision_dataset", "adapter"]}
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
