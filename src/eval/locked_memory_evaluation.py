from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.consensus_policy import (
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import (
    PolicyConfig,
    compute_backtest_metrics,
    run_long_only_backtest,
)
from src.eval.memory_reliability import (
    ReliabilityConfig,
    build_consensus_frame,
    fit_reliability_model,
    reliability_metrics,
)


@dataclass(frozen=True)
class LockedBaselineConfig:
    """Predeclared baseline suite for a locked market-memory candidate."""

    nominal_coverage: float = 0.25
    momentum_window: int = 21
    random_trials: int = 20
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.nominal_coverage not in {0.25, 0.50, 0.75, 1.0}:
            raise ValueError("nominal_coverage must match a fitted reliability threshold.")
        if self.momentum_window <= 0:
            raise ValueError("momentum_window must be positive.")
        if self.random_trials <= 0:
            raise ValueError("random_trials must be positive.")


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2), encoding="utf-8")


def _run_policy(
    signals: pd.DataFrame,
    policy: PolicyConfig,
    score_col: str,
    *,
    exit_score_col: str | None = None,
    row_filter=None,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    trades, equity = run_long_only_backtest(
        signals,
        policy,
        score_col=score_col,
        exit_score_col=exit_score_col,
        row_filter=row_filter,
        decision_log=decisions,
    )
    return {
        "metrics": compute_backtest_metrics(trades, equity, policy.initial_capital),
        "trades": trades,
        "equity": equity,
        "decisions": pd.DataFrame(decisions),
    }


def _baseline_policy(policy: PolicyConfig) -> PolicyConfig:
    return replace(
        policy,
        min_score=0.0,
        min_expected_upside=0.0,
        max_expected_downside=1e6,
        min_confidence=0.0,
        min_alpha_lcb=None,
        max_downside_cvar=None,
        min_neighbor_count=None,
        require_ood_pass=False,
    )


def _score_frame(source: pd.DataFrame, score: pd.Series, column: str) -> pd.DataFrame:
    frame = source.copy()
    values = pd.to_numeric(score, errors="coerce")
    frame[column] = values
    frame["retrieval_expected_upside"] = values.clip(lower=0.0)
    frame["retrieval_expected_downside"] = values.clip(upper=0.0)
    frame["retrieval_confidence"] = values.notna().astype(float)
    return frame


def _equal_weight_buy_hold(
    signals: pd.DataFrame,
    initial_capital: float,
    slippage_bps: float,
) -> dict[str, Any]:
    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    price_col = next(
        (column for column in ("close", "Close", "adj_close", "open", "Open", "adj_open") if column in frame),
        None,
    )
    if price_col is None:
        raise ValueError("Equal-weight baseline requires a price column.")
    pivot = frame.pivot_table(
        index="timestamp",
        columns="ticker",
        values=price_col,
        aggfunc="last",
    ).sort_index()
    first = pivot.apply(lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan)
    normalized = pivot.divide(first, axis=1)
    relative = normalized.mean(axis=1, skipna=True).dropna()
    cost_factor = 1.0 - float(slippage_bps) / 10000.0
    curve = initial_capital * cost_factor * relative
    if len(curve):
        curve.iloc[-1] *= cost_factor
    equity = pd.DataFrame({"timestamp": curve.index, "equity": curve.to_numpy(dtype=float)})
    equity["return"] = equity["equity"].pct_change().fillna(0.0)
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0
    equity["exposure"] = 1.0
    metrics = compute_backtest_metrics(pd.DataFrame(), equity, initial_capital)
    return {"metrics": metrics, "trades": pd.DataFrame(), "equity": equity, "decisions": pd.DataFrame()}


def _save_policy_result(root: Path, name: str, result: dict[str, Any]) -> None:
    destination = root / "baselines" / name
    destination.mkdir(parents=True, exist_ok=True)
    result["trades"].to_csv(destination / "trades.csv", index=False)
    result["equity"].to_csv(destination / "equity_curve.csv", index=False)
    result["decisions"].to_csv(destination / "decisions.csv", index=False)
    _write_json(destination / "metrics.json", result["metrics"])


def evaluate_locked_market(
    *,
    development_signals: dict[int, pd.DataFrame],
    development_neighbors: dict[int, pd.DataFrame],
    query_signals: dict[int, pd.DataFrame],
    query_neighbors: dict[int, pd.DataFrame],
    policy: PolicyConfig,
    consensus_config: ConsensusSignalConfig,
    reliability_config: ReliabilityConfig,
    useful_alpha_after_costs: float,
    baseline_config: LockedBaselineConfig,
    output: Path,
) -> dict[str, Any]:
    """Evaluate one frozen market with reliability and predeclared baselines."""
    development = build_consensus_frame(
        development_signals,
        development_neighbors,
        useful_alpha_after_costs=useful_alpha_after_costs,
    )
    query = build_consensus_frame(
        query_signals,
        query_neighbors,
        useful_alpha_after_costs=useful_alpha_after_costs,
    )
    model = fit_reliability_model(development, reliability_config)
    predictions = model.predict(query)
    threshold = float(model.calibration_thresholds[baseline_config.nominal_coverage])

    consensus = build_consensus_signals(query_signals, policy, consensus_config)
    reliability_columns = predictions[
        [
            "ticker",
            "timestamp",
            "reliability_probability",
            "predicted_absolute_error",
            "state_shift_score",
        ]
    ]
    consensus = consensus.merge(
        reliability_columns,
        on=["ticker", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    consensus["reliability_pass"] = consensus["reliability_probability"] >= threshold

    locked = _run_policy(
        consensus,
        policy,
        "consensus_entry_rank",
        exit_score_col="consensus_exit_score",
        row_filter=lambda row, cfg, _score: (
            bool(row.get("reliability_pass", False))
            and consensus_row_filter(row, cfg, consensus_config.minimum_votes)
        ),
    )
    raw_memory = _run_policy(
        consensus,
        policy,
        "consensus_entry_rank",
        exit_score_col="consensus_exit_score",
        row_filter=lambda row, cfg, _score: consensus_row_filter(
            row,
            cfg,
            consensus_config.minimum_votes,
        ),
    )

    neutral = _baseline_policy(policy)
    close_col = next(
        (column for column in ("close", "Close", "adj_close", "open", "Open", "adj_open") if column in consensus),
        None,
    )
    if close_col is None:
        raise ValueError("Momentum baseline requires a price column.")
    ordered = consensus.sort_values(["ticker", "timestamp"]).copy()
    momentum = ordered.groupby("ticker")[close_col].pct_change(baseline_config.momentum_window)
    momentum_result = _run_policy(
        _score_frame(ordered, momentum, "momentum_score"),
        neutral,
        "momentum_score",
    )

    if "pred_utility_q50" not in consensus:
        raise ValueError("Direct adapter baseline requires pred_utility_q50.")
    model_head_result = _run_policy(
        _score_frame(consensus, consensus["pred_utility_q50"], "model_head_score"),
        neutral,
        "model_head_score",
    )

    random_rows: list[dict[str, Any]] = []
    random_results: list[dict[str, Any]] = []
    for trial in range(baseline_config.random_trials):
        seed = baseline_config.random_seed + trial
        rng = np.random.default_rng(seed)
        random_result = _run_policy(
            _score_frame(
                consensus,
                pd.Series(rng.random(len(consensus)), index=consensus.index),
                "random_score",
            ),
            neutral,
            "random_score",
        )
        random_results.append(random_result)
        random_rows.append({"trial": trial, "seed": seed, **random_result["metrics"]})
    random_table = pd.DataFrame(random_rows)
    median_index = int(
        (random_table["sharpe"] - random_table["sharpe"].median()).abs().idxmin()
    )
    random_median = random_results[median_index]
    random_summary = {
        "trials": baseline_config.random_trials,
        "median_total_return": float(random_table["total_return"].median()),
        "median_sharpe": float(random_table["sharpe"].median()),
        "sharpe_p05": float(random_table["sharpe"].quantile(0.05)),
        "sharpe_p95": float(random_table["sharpe"].quantile(0.95)),
        "probability_beating_locked_sharpe": float(
            (random_table["sharpe"] >= float(locked["metrics"]["sharpe"])).mean()
        ),
    }
    equal_weight = _equal_weight_buy_hold(
        consensus,
        policy.initial_capital,
        policy.slippage_bps,
    )

    output.mkdir(parents=True, exist_ok=True)
    consensus.to_parquet(output / "consensus_signals.parquet", index=False)
    predictions.to_parquet(output / "confirmation_reliability.parquet", index=False)
    _write_json(output / "reliability_model_audit.json", model.to_dict())
    reliability_audit = reliability_metrics(predictions, model)
    _write_json(output / "reliability_metrics.json", reliability_audit)

    results = {
        "locked_memory": locked,
        "raw_memory": raw_memory,
        "momentum_21": momentum_result,
        "direct_adapter_q50": model_head_result,
        "equal_weight_buy_hold": equal_weight,
        "random_median": random_median,
    }
    for name, result in results.items():
        _save_policy_result(output, name, result)
    random_table.to_csv(output / "baselines" / "random_trials.csv", index=False)
    _write_json(output / "baselines" / "random_summary.json", random_summary)

    payload = {
        "status": "completed",
        "nominal_coverage": baseline_config.nominal_coverage,
        "reliability_threshold": threshold,
        "realized_coverage": float(consensus["reliability_pass"].mean()),
        "useful_alpha_after_costs": useful_alpha_after_costs,
        "policy": asdict(policy),
        "consensus": asdict(consensus_config),
        "reliability": reliability_audit,
        "baselines": {name: result["metrics"] for name, result in results.items()},
        "random": random_summary,
    }
    _write_json(output / "metrics.json", payload)
    return payload
