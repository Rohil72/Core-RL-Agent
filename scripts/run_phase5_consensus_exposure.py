"""Evaluate causal, market-scale-invariant exposure over frozen C0 consensus signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.consensus_policy import consensus_row_filter  # noqa: E402
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _equity_returns(path: Path) -> pd.Series:
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


def _pooled_paths(paths: list[Path]) -> pd.Series:
    series = [_equity_returns(path).rename(str(index)) for index, path in enumerate(paths) if path.exists()]
    return pd.concat(series, axis=1).mean(axis=1, skipna=True).dropna() if series else pd.Series(dtype=float)


def _candidate_paths(output: Path, candidate: str, folds: list[str]) -> list[Path]:
    return [output / "policies" / candidate / fold / "equity_curve.csv" for fold in folds]


def _controller_metrics(decisions: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, Any]:
    if decisions.empty:
        return {
            "mean_target_exposure": 1.0,
            "derisked_session_fraction": 0.0,
            "mean_realized_annualized_volatility": None,
            "mean_evidence_scalar": 1.0,
            "mean_drawdown_scalar": 1.0,
            "mean_volatility_scalar": 1.0,
            "mean_seed_vote_fraction": None,
            "mean_seed_rank_stability": None,
            "rebalance_count": 0,
        }
    target = pd.to_numeric(decisions["target_exposure"], errors="coerce")
    realized = pd.to_numeric(decisions["realized_annualized_volatility"], errors="coerce")
    rebalances = trades.get("is_rebalance", pd.Series(dtype=bool)).fillna(False).astype(bool)
    return {
        "mean_target_exposure": float(target.mean()),
        "derisked_session_fraction": float((target < 0.999).mean()),
        "mean_realized_annualized_volatility": float(realized.mean()) if realized.notna().any() else None,
        "mean_evidence_scalar": float(pd.to_numeric(decisions["evidence_scalar"], errors="coerce").mean()),
        "mean_drawdown_scalar": float(pd.to_numeric(decisions["drawdown_scalar"], errors="coerce").mean()),
        "mean_volatility_scalar": float(pd.to_numeric(decisions["volatility_scalar"], errors="coerce").mean()),
        "mean_seed_vote_fraction": float(pd.to_numeric(decisions["mean_seed_vote_fraction"], errors="coerce").mean()),
        "mean_seed_rank_stability": float(pd.to_numeric(decisions["median_seed_rank_stability"], errors="coerce").mean()),
        "mean_realized_exposure": float(equity["exposure"].mean()) if not equity.empty else 0.0,
        "rebalance_count": int(rebalances.sum()),
    }


def _candidate_stats(
    config: dict[str, Any],
    output: Path,
    summary: pd.DataFrame,
    candidate: str,
) -> dict[str, Any]:
    selected = summary.loc[summary["candidate"] == candidate].copy()
    folds = sorted(selected["fold"].astype(str).tolist())
    pooled = _pooled_paths(_candidate_paths(output, candidate, folds))
    fold_sharpes = pd.to_numeric(selected["sharpe"], errors="coerce").dropna()
    fold_excess = pd.to_numeric(selected["total_return"], errors="coerce") - pd.to_numeric(
        selected["equal_weight_baseline_return"], errors="coerce"
    )
    objective = config["selection"]["objective"]
    minimum_fold_sharpe = float(fold_sharpes.min()) if not fold_sharpes.empty else 0.0
    sharpe_dispersion = float(fold_sharpes.std(ddof=0)) if not fold_sharpes.empty else 0.0
    worst_drawdown = float(pd.to_numeric(selected["max_drawdown"], errors="coerce").min())
    pooled_sharpe = _sharpe(pooled)
    robust_score = (
        float(objective["pooled_sharpe_weight"]) * pooled_sharpe
        + float(objective["minimum_fold_sharpe_weight"]) * minimum_fold_sharpe
        + float(objective["mean_excess_return_weight"]) * float(fold_excess.mean())
        - float(objective["drawdown_weight"]) * abs(worst_drawdown)
        - float(objective["fold_dispersion_weight"]) * sharpe_dispersion
    )
    return {
        "candidate": candidate,
        "fold_count": int(len(selected)),
        "pooled_sharpe": pooled_sharpe,
        "mean_return": float(pd.to_numeric(selected["total_return"], errors="coerce").mean()),
        "minimum_fold_return": float(pd.to_numeric(selected["total_return"], errors="coerce").min()),
        "mean_fold_sharpe": float(fold_sharpes.mean()),
        "minimum_fold_sharpe": minimum_fold_sharpe,
        "fold_sharpe_dispersion": sharpe_dispersion,
        "worst_drawdown": worst_drawdown,
        "mean_excess_return": float(fold_excess.mean()),
        "positive_fold_fraction": float((pd.to_numeric(selected["total_return"], errors="coerce") > 0).mean()),
        "positive_fold_excess_fraction": float((fold_excess > 0).mean()),
        "mean_trade_count": float(pd.to_numeric(selected["trade_count"], errors="coerce").mean()),
        "mean_target_exposure": float(pd.to_numeric(selected["mean_target_exposure"], errors="coerce").mean()),
        "robust_score": float(robust_score),
        "daily_returns": pooled,
    }


def _leave_one_fold_out(
    config: dict[str, Any],
    output: Path,
    summary: pd.DataFrame,
    candidates: list[str],
) -> tuple[pd.DataFrame, float]:
    records: list[dict[str, Any]] = []
    heldout_returns: list[pd.Series] = []
    for fold in sorted(summary["fold"].unique()):
        training = summary.loc[summary["fold"] != fold]
        ranked = sorted(
            (_candidate_stats(config, output, training, candidate) for candidate in candidates),
            key=lambda item: item["robust_score"],
            reverse=True,
        )
        winner = str(ranked[0]["candidate"])
        heldout = summary.loc[(summary["fold"] == fold) & (summary["candidate"] == winner)].iloc[0]
        returns = _equity_returns(output / "policies" / winner / str(fold) / "equity_curve.csv")
        heldout_returns.append(returns.rename(str(fold)))
        records.append(
            {
                "heldout_fold": fold,
                "selected_candidate": winner,
                "training_robust_score": ranked[0]["robust_score"],
                "heldout_return": heldout["total_return"],
                "heldout_sharpe": heldout["sharpe"],
                "heldout_drawdown": heldout["max_drawdown"],
            }
        )
    pooled = pd.concat(heldout_returns, axis=1).mean(axis=1, skipna=True).dropna()
    return pd.DataFrame(records), _sharpe(pooled)


def _selection_report(
    output: Path,
    selection: dict[str, Any],
    leaderboard: pd.DataFrame,
    oof: pd.DataFrame,
) -> None:
    columns = [
        "candidate", "pooled_sharpe", "minimum_fold_sharpe", "fold_sharpe_dispersion",
        "mean_return", "mean_excess_return", "worst_drawdown", "mean_target_exposure",
        "deflated_sharpe_probability", "bootstrap_probability_positive", "pbo", "passes",
    ]
    lines = [
        "# Phase 5 Consensus Exposure Selection",
        "",
        f"- Status: `{selection['status']}`",
        f"- Development candidate: `{selection['selected_candidate']}`",
        f"- Leave-one-fold-out Sharpe: `{selection['cross_fold_oof_sharpe']:.4f}`",
        "- Transformer, memory, consensus ranking, and A2 hold/exit logic remained frozen.",
        "- Cross-market confirmation remains mandatory; this run cannot promote a final policy.",
        "",
        "## Leaderboard",
        "",
        leaderboard[columns].to_markdown(index=False),
        "",
        "## Leave-One-Fold-Out",
        "",
        oof.to_markdown(index=False),
    ]
    (output / "selection_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path: str, run_id: str) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    output = PROJECT_ROOT / experiment["output_root"] / run_id
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty exposure directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source = PROJECT_ROOT / experiment["source_consensus_run"] / "policies" / experiment["source_variant"]
    signal_paths = sorted(source.glob("fold_*/signals.parquet"))
    if len(signal_paths) < int(experiment["minimum_folds"]):
        raise RuntimeError(f"Expected at least {experiment['minimum_folds']} source folds, found {len(signal_paths)}.")
    source_configs = [json.loads((path.parent / "config.json").read_text(encoding="utf-8")) for path in signal_paths]
    if any(item != source_configs[0] for item in source_configs[1:]):
        raise ValueError("Consensus source folds do not share one immutable signal configuration.")
    consensus_config = source_configs[0]
    phase4e_config = yaml.safe_load((PROJECT_ROOT / experiment["phase4e_config"]).read_text(encoding="utf-8"))
    policy = PolicyConfig(**phase4e_config["policy"])
    _write_json(
        output / "source_manifest.json",
        {
            "development_split_only": True,
            "source_variant": experiment["source_variant"],
            "consensus_config": consensus_config,
            "files": [{"path": str(path), "sha256": _sha256(path)} for path in signal_paths],
        },
    )

    rows: list[dict[str, Any]] = []
    for specification in config["controllers"]:
        candidate = str(specification["id"])
        resolved = ExposureControllerConfig(**specification["config"])
        for signal_path in signal_paths:
            fold = signal_path.parent.name
            signals = pd.read_parquet(signal_path)
            decisions: list[dict[str, Any]] = []
            controller = CausalExposureController(resolved) if resolved.enabled else None
            trades, equity = run_long_only_backtest(
                signals,
                policy,
                score_col="consensus_entry_rank",
                exit_score_col="consensus_exit_score",
                row_filter=lambda row, cfg, _score, votes=int(consensus_config["minimum_votes"]): (
                    consensus_row_filter(row, cfg, votes)
                ),
                exposure_controller=controller,
                exposure_log=decisions,
            )
            metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
            metrics.update(equal_weight_baseline(signals, policy.initial_capital))
            metrics.update(_controller_metrics(pd.DataFrame(decisions), equity, trades))
            result_dir = output / "policies" / candidate / fold
            result_dir.mkdir(parents=True, exist_ok=True)
            trades.to_csv(result_dir / "trades.csv", index=False)
            equity.to_csv(result_dir / "equity_curve.csv", index=False)
            pd.DataFrame(decisions).to_csv(result_dir / "exposure_decisions.csv", index=False)
            _write_json(result_dir / "metrics.json", metrics)
            _write_json(result_dir / "controller_snapshot.json", specification)
            rows.append({"candidate": candidate, "fold": fold, **metrics})

    summary = pd.DataFrame(rows)
    summary.to_csv(output / "fold_summary.csv", index=False)
    candidates = [str(item["id"]) for item in config["controllers"]]
    stats = [_candidate_stats(config, output, summary, candidate) for candidate in candidates]
    returns_matrix = pd.concat(
        {item["candidate"]: item.pop("daily_returns") for item in stats}, axis=1
    ).fillna(0.0)
    pbo = probability_of_backtest_overfitting(returns_matrix.to_numpy())
    a2_paths = list(
        (PROJECT_ROOT / experiment["source_phase4e_run"] / "policies" / "val" / "A2_base").glob(
            "fold_*/seed_*/eval/equity_curve.csv"
        )
    )
    a2_returns = _pooled_paths(a2_paths)
    gates = config["selection"]
    for item in stats:
        candidate_returns = returns_matrix[item["candidate"]].dropna()
        candidate_aligned, a2_aligned = candidate_returns.align(a2_returns, join="inner")
        bootstrap = stationary_bootstrap_delta(
            candidate_aligned.to_numpy(),
            a2_aligned.to_numpy(),
            expected_block_length=int(gates["bootstrap_block_length"]),
            samples=int(gates["bootstrap_samples"]),
            seed=7,
        )
        item.update(
            {
                "deflated_sharpe_probability": deflated_sharpe_probability(
                    candidate_returns.to_numpy(), len(candidates)
                ),
                "bootstrap_probability_positive": bootstrap.probability_positive,
                "bootstrap_ci_low": bootstrap.ci_low,
                "bootstrap_ci_high": bootstrap.ci_high,
                "pbo": pbo,
            }
        )
        item["gate_pooled_sharpe"] = item["pooled_sharpe"] >= float(gates["minimum_pooled_sharpe"])
        item["gate_minimum_fold_sharpe"] = item["minimum_fold_sharpe"] >= float(gates["minimum_fold_sharpe"])
        item["gate_drawdown"] = abs(item["worst_drawdown"]) <= float(gates["maximum_worst_drawdown"])
        item["gate_positive_folds"] = item["positive_fold_fraction"] >= float(gates["minimum_positive_fold_fraction"])
        item["gate_excess_return"] = item["mean_excess_return"] > float(gates["minimum_mean_excess_return"])
        item["gate_dsr"] = item["deflated_sharpe_probability"] >= float(gates["minimum_deflated_sharpe_probability"])
        item["gate_bootstrap"] = item["bootstrap_probability_positive"] >= float(gates["minimum_bootstrap_probability"])
        item["gate_pbo"] = pbo <= float(gates["maximum_pbo"])
        item["gate_trade_count"] = item["mean_trade_count"] >= float(gates["minimum_mean_trade_count"])

    oof, oof_sharpe = _leave_one_fold_out(config, output, summary, candidates)
    oof.to_csv(output / "cross_fold_oof.csv", index=False)
    leaderboard = pd.DataFrame(stats).sort_values("robust_score", ascending=False)
    gate_columns = [column for column in leaderboard if column.startswith("gate_")]
    leaderboard["gate_cross_fold_oof"] = oof_sharpe >= float(gates["minimum_cross_fold_oof_sharpe"])
    leaderboard["passes"] = leaderboard[gate_columns + ["gate_cross_fold_oof"]].all(axis=1)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    baseline = str(config["selection"]["baseline_candidate"])
    eligible = leaderboard.loc[leaderboard["passes"] & (leaderboard["candidate"] != baseline)]
    selected = str(eligible.iloc[0]["candidate"]) if not eligible.empty else None
    selection = {
        "status": "cross_market_pending" if selected else "rejected",
        "selected_candidate": selected,
        "promotion_pass": False,
        "cross_market_confirmation_required": True,
        "cross_fold_oof_sharpe": oof_sharpe,
        "pbo": pbo,
        "variant_count": len(candidates),
        "development_split_only": True,
        "output": str(output),
    }
    _write_json(output / "selection.json", selection)
    _selection_report(output, selection, leaderboard, oof)
    return {"run_id": run_id, **selection}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_consensus_exposure.yaml")
    parser.add_argument("--run-id", default="phase5_consensus_exposure_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id), indent=2))


if __name__ == "__main__":
    main()
