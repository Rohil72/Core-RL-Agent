from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.backtest.market_memory_backtester import (
    PolicyConfig,
    compute_backtest_metrics,
    run_long_only_backtest,
)


@dataclass(frozen=True)
class BaselineSuiteConfig:
    """Configuration shared by deterministic policy-evaluation baselines."""

    momentum_window: int = 21
    random_trials: int = 20
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.momentum_window <= 0:
            raise ValueError("momentum_window must be positive.")
        if self.random_trials <= 0:
            raise ValueError("random_trials must be positive.")


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    """Write JSON while preserving finite native numeric values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_native(value), indent=2), encoding="utf-8")


def run_scored_policy(
    signals: pd.DataFrame,
    policy: PolicyConfig,
    score_col: str,
    *,
    exit_score_col: str | None = None,
    row_filter: Callable[[pd.Series, PolicyConfig, str], bool] | None = None,
) -> dict[str, Any]:
    """Run one capital-constrained long-only policy and return all artifacts."""

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


def neutral_baseline_policy(policy: PolicyConfig) -> PolicyConfig:
    """Retain execution and risk mechanics while removing memory evidence gates."""

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


def score_frame(source: pd.DataFrame, score: pd.Series, column: str) -> pd.DataFrame:
    """Attach a baseline score and the minimal evidence fields used by the policy."""

    frame = source.copy()
    values = pd.to_numeric(score, errors="coerce")
    frame[column] = values
    frame["retrieval_expected_upside"] = values.clip(lower=0.0)
    frame["retrieval_expected_downside"] = values.clip(upper=0.0)
    frame["retrieval_confidence"] = values.notna().astype(float)
    return frame


def equal_weight_buy_hold(
    signals: pd.DataFrame,
    initial_capital: float,
    slippage_bps: float,
) -> dict[str, Any]:
    """Evaluate one equal-weight buy-and-hold portfolio with round-trip costs."""

    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    price_col = next(
        (
            column
            for column in ("close", "Close", "adj_close", "open", "Open", "adj_open")
            if column in frame
        ),
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
    first = pivot.apply(
        lambda values: values.dropna().iloc[0] if values.notna().any() else np.nan
    )
    relative = pivot.divide(first, axis=1).mean(axis=1, skipna=True).dropna()
    cost_factor = 1.0 - float(slippage_bps) / 10000.0
    curve = initial_capital * cost_factor * relative
    if len(curve):
        curve.iloc[-1] *= cost_factor
    equity = pd.DataFrame({"timestamp": curve.index, "equity": curve.to_numpy(dtype=float)})
    equity["return"] = equity["equity"].pct_change().fillna(0.0)
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0
    equity["exposure"] = 1.0
    return {
        "metrics": compute_backtest_metrics(pd.DataFrame(), equity, initial_capital),
        "trades": pd.DataFrame(),
        "equity": equity,
        "decisions": pd.DataFrame(),
    }


def evaluate_standard_baselines(
    signals: pd.DataFrame,
    policy: PolicyConfig,
    config: BaselineSuiteConfig,
    *,
    reference_sharpe: float,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Evaluate momentum, adapter-head, equal-weight, and repeated-random baselines."""

    neutral = neutral_baseline_policy(policy)
    close_col = next(
        (
            column
            for column in ("close", "Close", "adj_close", "open", "Open", "adj_open")
            if column in signals
        ),
        None,
    )
    if close_col is None:
        raise ValueError("Momentum baseline requires a price column.")
    ordered = signals.sort_values(["ticker", "timestamp"]).copy()
    momentum = ordered.groupby("ticker")[close_col].pct_change(config.momentum_window)
    momentum_result = run_scored_policy(
        score_frame(ordered, momentum, "momentum_score"),
        neutral,
        "momentum_score",
    )

    if "pred_utility_q50" not in signals:
        raise ValueError("Direct adapter baseline requires pred_utility_q50.")
    adapter_result = run_scored_policy(
        score_frame(signals, signals["pred_utility_q50"], "model_head_score"),
        neutral,
        "model_head_score",
    )

    random_rows: list[dict[str, Any]] = []
    random_results: list[dict[str, Any]] = []
    for trial in range(config.random_trials):
        seed = config.random_seed + trial
        rng = np.random.default_rng(seed)
        result = run_scored_policy(
            score_frame(
                signals,
                pd.Series(rng.random(len(signals)), index=signals.index),
                "random_score",
            ),
            neutral,
            "random_score",
        )
        random_results.append(result)
        random_rows.append({"trial": trial, "seed": seed, **result["metrics"]})
    random_table = pd.DataFrame(random_rows)
    median_index = int(
        (random_table["sharpe"] - random_table["sharpe"].median()).abs().idxmin()
    )
    random_summary = {
        "trials": config.random_trials,
        "median_total_return": float(random_table["total_return"].median()),
        "median_sharpe": float(random_table["sharpe"].median()),
        "sharpe_p05": float(random_table["sharpe"].quantile(0.05)),
        "sharpe_p95": float(random_table["sharpe"].quantile(0.95)),
        "probability_beating_reference_sharpe": float(
            (random_table["sharpe"] >= float(reference_sharpe)).mean()
        ),
    }
    results = {
        f"momentum_{config.momentum_window}": momentum_result,
        "direct_adapter_q50": adapter_result,
        "equal_weight_buy_hold": equal_weight_buy_hold(
            signals,
            policy.initial_capital,
            policy.slippage_bps,
        ),
        "random_median": random_results[median_index],
    }
    return results, random_table, random_summary


def save_policy_result(root: Path, name: str, result: dict[str, Any]) -> None:
    """Persist a policy result using the common evaluation artifact contract."""

    destination = root / "baselines" / name
    destination.mkdir(parents=True, exist_ok=True)
    result["trades"].to_csv(destination / "trades.csv", index=False)
    result["equity"].to_csv(destination / "equity_curve.csv", index=False)
    result["decisions"].to_csv(destination / "decisions.csv", index=False)
    write_json(destination / "metrics.json", result["metrics"])
