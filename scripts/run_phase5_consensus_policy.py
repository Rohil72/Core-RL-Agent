"""Test seed-consensus entry and persistent conviction over frozen Phase 5 signals."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.consensus_policy import (  # noqa: E402
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import (  # noqa: E402
    PolicyConfig,
    compute_backtest_metrics,
    run_long_only_backtest,
)
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
    stationary_bootstrap_delta,
)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2), encoding="utf-8")


def _returns(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.Series(dtype=float)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().dropna()


def _sharpe(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 3 or np.std(values, ddof=1) <= 1e-12:
        return 0.0
    return float(np.mean(values) / np.std(values, ddof=1) * np.sqrt(252.0))


def _pooled(paths: list[Path]) -> pd.Series:
    series = [_returns(path).rename(str(index)) for index, path in enumerate(paths) if path.exists()]
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna() if series else pd.Series(dtype=float)


def _seed_signal_paths(source: Path) -> dict[str, dict[int, Path]]:
    result: dict[str, dict[int, Path]] = {}
    for path in source.glob("fold_*/seed_*/adapter_backtest/eval/signals.parquet"):
        fold = path.parents[3].name
        seed = int(path.parents[2].name.removeprefix("seed_"))
        result.setdefault(fold, {})[seed] = path
    return result


def _entry_explanations(trades: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    dates = sorted(pd.to_datetime(signals["timestamp"], utc=True).unique())
    prior = {dates[index]: dates[index - 1] for index in range(1, len(dates))}
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"], utc=True)
    out["signal_date"] = out["entry_date"].map(prior)
    columns = [
        "ticker", "timestamp", "seed_vote_count", "seed_model_count", "seed_vote_fraction",
        "seed_score_std", "seed_rank_std", "consensus_economic_score",
        "consensus_entry_rank", "consensus_exit_score", "retrieval_expected_alpha",
        "retrieval_downside_cvar", "retrieval_confidence", "retrieval_agreement_score",
    ]
    return out.merge(signals[columns], left_on=["ticker", "signal_date"], right_on=["ticker", "timestamp"], how="left")


def run(config_path: str, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output = PROJECT_ROOT / experiment["output_root"] / run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty consensus directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / experiment["source_phase5_run"]
    fold_paths = _seed_signal_paths(source)
    if len(fold_paths) < int(experiment["minimum_folds"]):
        raise RuntimeError(f"Expected {experiment['minimum_folds']} folds, found {len(fold_paths)}.")
    phase4e = yaml.safe_load((PROJECT_ROOT / experiment["phase4e_config"]).read_text(encoding="utf-8"))
    policy = PolicyConfig(**phase4e["policy"])
    rows: list[dict[str, Any]] = []
    variant_return_paths: dict[str, list[Path]] = {}
    for variant in config["variants"]:
        identifier = str(variant["id"])
        variant_return_paths[identifier] = []
        signal_cfg = ConsensusSignalConfig(
            minimum_votes=int(variant["minimum_votes"]),
            exit_smoothing_span=int(variant["exit_smoothing_span"]),
            exit_score_quantile=float(variant["exit_score_quantile"]),
        )
        for fold, seed_paths in sorted(fold_paths.items()):
            if len(seed_paths) < int(experiment["minimum_seed_models"]):
                raise RuntimeError(f"{fold} has only {len(seed_paths)} seed signals.")
            seed_frames = {seed: pd.read_parquet(path) for seed, path in seed_paths.items()}
            signals = build_consensus_signals(seed_frames, policy, signal_cfg)
            trades, equity = run_long_only_backtest(
                signals,
                policy,
                score_col="consensus_entry_rank",
                exit_score_col="consensus_exit_score",
                row_filter=lambda row, cfg, _score, votes=signal_cfg.minimum_votes: consensus_row_filter(row, cfg, votes),
            )
            metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
            result_dir = output / "policies" / identifier / fold
            result_dir.mkdir(parents=True, exist_ok=True)
            signals.to_parquet(result_dir / "signals.parquet", index=False)
            trades.to_csv(result_dir / "trades.csv", index=False)
            equity.to_csv(result_dir / "equity_curve.csv", index=False)
            _entry_explanations(trades, signals).to_csv(result_dir / "trade_explanations.csv", index=False)
            _write_json(result_dir / "metrics.json", metrics)
            _write_json(result_dir / "config.json", asdict(signal_cfg))
            variant_return_paths[identifier].append(result_dir / "equity_curve.csv")
            rows.append({"variant": identifier, "fold": fold, **metrics})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "fold_summary.csv", index=False)
    baseline_root = PROJECT_ROOT / experiment["source_phase4e_run"] / "policies" / "val" / "A2_base"
    baseline_paths = list(baseline_root.glob("fold_*/seed_*/eval/equity_curve.csv"))
    baseline_returns = _pooled(baseline_paths)
    baseline_metrics = pd.read_csv(PROJECT_ROOT / experiment["source_phase4e_run"] / "policy_sweep_summary.csv")
    baseline_metrics = baseline_metrics.loc[baseline_metrics["candidate"] == "A2_base"]
    leaderboard = []
    aligned_returns: list[pd.Series] = []
    for variant in config["variants"]:
        identifier = str(variant["id"])
        selected = summary.loc[summary["variant"] == identifier]
        returns = _pooled(variant_return_paths[identifier])
        aligned_returns.append(returns.rename(identifier))
        candidate, baseline = returns.align(baseline_returns, join="inner")
        bootstrap = stationary_bootstrap_delta(
            candidate.to_numpy(), baseline.to_numpy(),
            expected_block_length=int(config["selection"]["bootstrap_block_length"]),
            samples=int(config["selection"]["bootstrap_samples"]), seed=7,
        )
        fold_excess = []
        for fold, fold_rows in selected.groupby("fold"):
            baseline_fold = baseline_metrics.loc[baseline_metrics["fold"] == fold, "total_return"]
            fold_excess.append(float(fold_rows["total_return"].mean() - baseline_fold.mean()))
        leaderboard.append(
            {
                "variant": identifier,
                "pooled_sharpe": _sharpe(returns),
                "mean_return": float(selected["total_return"].mean()),
                "mean_sharpe": float(selected["sharpe"].mean()),
                "worst_drawdown": float(selected["max_drawdown"].min()),
                "mean_trade_count": float(selected["trade_count"].mean()),
                "mean_excess_return": float(np.mean(fold_excess)),
                "positive_fold_fraction": float(np.mean(np.asarray(fold_excess) > 0.0)),
                "deflated_sharpe_probability": deflated_sharpe_probability(
                    returns.to_numpy(), len(config["variants"])
                ),
                "bootstrap_probability_positive": bootstrap.probability_positive,
                "bootstrap_ci_low": bootstrap.ci_low,
                "bootstrap_ci_high": bootstrap.ci_high,
            }
        )
    leaderboard_frame = pd.DataFrame(leaderboard).sort_values("pooled_sharpe", ascending=False)
    return_matrix = pd.concat(aligned_returns, axis=1).dropna(how="all")
    pbo = probability_of_backtest_overfitting(return_matrix.to_numpy())
    leaderboard_frame["pbo"] = pbo
    gates = config["selection"]
    leaderboard_frame["passes"] = (
        (leaderboard_frame["pooled_sharpe"] >= float(gates["minimum_pooled_sharpe"]))
        & (leaderboard_frame["worst_drawdown"].abs() <= float(gates["maximum_worst_drawdown"]))
        & (leaderboard_frame["mean_excess_return"] > float(gates["minimum_mean_excess_return"]))
        & (leaderboard_frame["positive_fold_fraction"] >= float(gates["minimum_positive_fold_fraction"]))
        & (leaderboard_frame["deflated_sharpe_probability"] >= float(gates["minimum_deflated_sharpe_probability"]))
        & (leaderboard_frame["bootstrap_probability_positive"] >= float(gates["minimum_bootstrap_probability"]))
        & (pbo <= float(gates["maximum_pbo"]))
    )
    leaderboard_frame.to_csv(output / "leaderboard.csv", index=False)
    eligible = leaderboard_frame.loc[leaderboard_frame["passes"]]
    selection = {
        "status": "promoted" if not eligible.empty else "rejected",
        "selected_variant": str(eligible.iloc[0]["variant"]) if not eligible.empty else None,
        "baseline_pooled_sharpe": _sharpe(baseline_returns),
        "pbo": pbo,
        "variant_count": len(config["variants"]),
        "development_split_only": True,
    }
    _write_json(output / "selection.json", selection)
    return {"run_id": run_id, "output": str(output), **selection}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_consensus_policy.yaml")
    parser.add_argument("--run-id", default="phase5_consensus_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), indent=2))


if __name__ == "__main__":
    main()
