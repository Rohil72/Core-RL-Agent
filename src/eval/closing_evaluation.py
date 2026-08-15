"""Accounting and inference helpers for the frozen closing evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EquityAuditLimits:
    """Numerical limits used to reject impossible backtest equity curves."""

    minimum_equity: float = 1e-8
    minimum_daily_return: float = -0.95
    reconciliation_tolerance: float = 1e-9


def validate_equity_curve(
    equity: pd.DataFrame,
    initial_capital: float,
    limits: EquityAuditLimits | None = None,
) -> dict[str, float | int | bool]:
    """Validate positivity, finiteness, drawdown bounds, and return reconciliation."""

    cfg = limits or EquityAuditLimits()
    required = {"timestamp", "equity"}
    missing = required.difference(equity.columns)
    if missing:
        raise ValueError(f"Equity curve is missing columns: {sorted(missing)}")
    if equity.empty:
        raise ValueError("Equity curve must not be empty.")

    values = pd.to_numeric(equity["equity"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise FloatingPointError("Equity curve contains non-finite values.")
    if np.any(values <= cfg.minimum_equity):
        raise FloatingPointError("Equity curve contains zero or negative values.")

    derived = pd.Series(values).pct_change()
    derived.iloc[0] = float(values[0]) / float(initial_capital) - 1.0
    derived_returns = derived.fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(derived_returns).all():
        raise FloatingPointError("Equity curve produces non-finite returns.")
    if float(np.min(derived_returns)) <= cfg.minimum_daily_return:
        raise FloatingPointError(
            "Equity curve contains an implausible one-session collapse; inspect calendar marks."
        )

    compounded = float(initial_capital * np.prod(1.0 + derived_returns))
    reconciliation_error = abs(compounded - float(values[-1])) / max(
        abs(float(values[-1])), 1.0
    )
    if reconciliation_error > cfg.reconciliation_tolerance:
        raise FloatingPointError("Equity and compounded returns do not reconcile.")

    peaks = np.maximum.accumulate(values)
    drawdown = values / peaks - 1.0
    if float(np.min(drawdown)) < -1.0 or float(np.max(drawdown)) > cfg.reconciliation_tolerance:
        raise FloatingPointError("Drawdown lies outside the valid [-1, 0] interval.")

    return {
        "passed": True,
        "rows": int(len(equity)),
        "minimum_equity": float(np.min(values)),
        "minimum_daily_return": float(np.min(derived_returns)),
        "maximum_daily_return": float(np.max(derived_returns)),
        "maximum_drawdown": float(np.min(drawdown)),
        "reconciliation_error": float(reconciliation_error),
    }


def aggregate_equal_weight_market_returns(
    curves: Mapping[str, pd.DataFrame],
    initial_capital: float,
) -> pd.DataFrame:
    """Aggregate independent market books as equal-weight normalized daily returns.

    Missing exchange sessions contribute a zero return for that market.  Capital
    is never pooled across currencies; the output is an equal-weight return index.
    """

    if not curves:
        raise ValueError("At least one market equity curve is required.")
    return_series: dict[str, pd.Series] = {}
    exposure_series: dict[str, pd.Series] = {}
    for market, raw in curves.items():
        if raw.empty:
            raise ValueError(f"Market {market!r} has an empty equity curve.")
        frame = raw.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.normalize()
        frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
        values = pd.to_numeric(frame["equity"], errors="coerce")
        returns = values.pct_change()
        returns.iloc[0] = float(values.iloc[0]) / float(initial_capital) - 1.0
        returns = returns.fillna(0.0)
        return_series[str(market)] = pd.Series(
            returns.to_numpy(dtype=float), index=frame["timestamp"]
        )
        exposure = pd.to_numeric(
            frame.get("exposure", pd.Series(0.0, index=frame.index)), errors="coerce"
        ).fillna(0.0)
        exposure_series[str(market)] = pd.Series(
            exposure.to_numpy(dtype=float), index=frame["timestamp"]
        )

    market_returns = pd.concat(return_series, axis=1).sort_index().fillna(0.0)
    market_exposure = pd.concat(exposure_series, axis=1).sort_index().ffill().fillna(0.0)
    pooled_return = market_returns.mean(axis=1)
    pooled_equity = float(initial_capital) * (1.0 + pooled_return).cumprod()
    pooled = pd.DataFrame(
        {
            "timestamp": pooled_return.index,
            "equity": pooled_equity.to_numpy(dtype=float),
            "return": pooled_return.to_numpy(dtype=float),
            "exposure": market_exposure.mean(axis=1).to_numpy(dtype=float),
        }
    )
    peak = pooled["equity"].cummax()
    pooled["drawdown"] = pooled["equity"] / peak - 1.0
    return pooled.reset_index(drop=True)


def stationary_bootstrap_comparison(
    baseline_returns: np.ndarray,
    candidate_returns: np.ndarray,
    *,
    samples: int = 2000,
    mean_block_length: int = 21,
    seed: int = 1701,
) -> dict[str, float | int]:
    """Estimate paired performance-difference intervals with stationary blocks."""

    baseline = np.asarray(baseline_returns, dtype=float)
    candidate = np.asarray(candidate_returns, dtype=float)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("Baseline and candidate returns must be paired one-dimensional arrays.")
    valid = np.isfinite(baseline) & np.isfinite(candidate)
    baseline = baseline[valid]
    candidate = candidate[valid]
    if len(baseline) < 3:
        raise ValueError("At least three paired returns are required for bootstrap inference.")
    if samples <= 0 or mean_block_length <= 0:
        raise ValueError("samples and mean_block_length must be positive.")

    rng = np.random.default_rng(seed)
    restart_probability = 1.0 / float(mean_block_length)
    sharpe_delta = np.empty(samples, dtype=float)
    annualized_delta = np.empty(samples, dtype=float)
    n = len(baseline)
    for sample in range(samples):
        indices = np.empty(n, dtype=int)
        indices[0] = int(rng.integers(0, n))
        for offset in range(1, n):
            if rng.random() < restart_probability:
                indices[offset] = int(rng.integers(0, n))
            else:
                indices[offset] = (indices[offset - 1] + 1) % n
        base_sample = baseline[indices]
        candidate_sample = candidate[indices]
        sharpe_delta[sample] = _annualized_sharpe(candidate_sample) - _annualized_sharpe(
            base_sample
        )
        annualized_delta[sample] = _annualized_return(candidate_sample) - _annualized_return(
            base_sample
        )

    return {
        "samples": int(samples),
        "mean_block_length": int(mean_block_length),
        "paired_observations": int(n),
        "sharpe_delta_p025": float(np.quantile(sharpe_delta, 0.025)),
        "sharpe_delta_median": float(np.median(sharpe_delta)),
        "sharpe_delta_p975": float(np.quantile(sharpe_delta, 0.975)),
        "probability_sharpe_delta_positive": float(np.mean(sharpe_delta > 0.0)),
        "annualized_return_delta_p025": float(np.quantile(annualized_delta, 0.025)),
        "annualized_return_delta_median": float(np.median(annualized_delta)),
        "annualized_return_delta_p975": float(np.quantile(annualized_delta, 0.975)),
        "probability_annualized_return_delta_positive": float(
            np.mean(annualized_delta > 0.0)
        ),
    }


def _annualized_sharpe(returns: np.ndarray) -> float:
    std = float(np.std(returns, ddof=1))
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(np.mean(returns) / std * sqrt(252.0))


def _annualized_return(returns: np.ndarray) -> float:
    total = float(np.prod(1.0 + returns))
    if total <= 0.0:
        return -1.0
    return float(total ** (252.0 / max(len(returns), 1)) - 1.0)
