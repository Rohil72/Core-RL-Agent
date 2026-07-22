from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Stationary-bootstrap evidence for an incremental change."""

    observed_delta: float
    probability_positive: float
    ci_low: float
    ci_high: float
    samples: int


def stationary_bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    expected_block_length: int = 21,
    samples: int = 2000,
    seed: int = 7,
) -> PairedBootstrapResult:
    """Bootstrap the paired mean difference while preserving local dependence."""
    left = np.asarray(candidate, dtype=float)
    right = np.asarray(baseline, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    delta = left[mask] - right[mask]
    if delta.size < 2:
        return PairedBootstrapResult(float("nan"), float("nan"), float("nan"), float("nan"), samples)
    rng = np.random.default_rng(seed)
    restart_probability = 1.0 / max(int(expected_block_length), 1)
    estimates = np.empty(samples, dtype=float)
    n = len(delta)
    for sample in range(samples):
        index = int(rng.integers(n))
        draw = np.empty(n, dtype=float)
        for position in range(n):
            if position > 0 and rng.random() >= restart_probability:
                index = (index + 1) % n
            else:
                index = int(rng.integers(n))
            draw[position] = delta[index]
        estimates[sample] = float(draw.mean())
    return PairedBootstrapResult(
        observed_delta=float(delta.mean()),
        probability_positive=float(np.mean(estimates > 0.0)),
        ci_low=float(np.quantile(estimates, 0.025)),
        ci_high=float(np.quantile(estimates, 0.975)),
        samples=int(samples),
    )


def deflated_sharpe_probability(returns: np.ndarray, trial_count: int) -> float:
    """Approximate the Deflated Sharpe probability after multiple testing."""
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 3 or np.std(values, ddof=1) <= 1e-12:
        return 0.0
    daily_sharpe = float(np.mean(values) / np.std(values, ddof=1))
    annualized_sharpe = daily_sharpe * np.sqrt(252.0)
    trials = max(int(trial_count), 1)
    if trials == 1:
        benchmark = 0.0
    else:
        euler_gamma = 0.5772156649
        expected_max_normal = (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * np.e))
        )
        benchmark = expected_max_normal / np.sqrt(max(values.size - 1, 1)) * np.sqrt(252.0)
    skewness = float(skew(values, bias=False))
    raw_kurtosis = float(kurtosis(values, fisher=False, bias=False))
    denominator = np.sqrt(
        max(1e-12, 1.0 - skewness * daily_sharpe + ((raw_kurtosis - 1.0) / 4.0) * daily_sharpe**2)
    )
    statistic = (annualized_sharpe - benchmark) * np.sqrt(values.size - 1) / (np.sqrt(252.0) * denominator)
    return float(norm.cdf(statistic))


def probability_of_backtest_overfitting(
    performance_matrix: np.ndarray,
    block_count: int = 8,
) -> float:
    """Estimate PBO with combinatorially symmetric contiguous time blocks."""
    matrix = np.asarray(performance_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        return float("nan")
    blocks = max(2, min(int(block_count), matrix.shape[0]))
    if blocks % 2:
        blocks -= 1
    indices = [item for item in np.array_split(np.arange(matrix.shape[0]), blocks) if len(item)]
    if len(indices) < 2 or len(indices) % 2:
        return float("nan")
    half = len(indices) // 2
    failures = []
    all_blocks = set(range(len(indices)))
    for in_sample_blocks in combinations(range(len(indices)), half):
        out_sample_blocks = sorted(all_blocks - set(in_sample_blocks))
        in_rows = np.concatenate([indices[index] for index in in_sample_blocks])
        out_rows = np.concatenate([indices[index] for index in out_sample_blocks])
        in_score = _column_sharpe(matrix[in_rows])
        winner = int(np.nanargmax(in_score))
        out_score = _column_sharpe(matrix[out_rows])
        rank = float(np.mean(out_score <= out_score[winner]))
        failures.append(rank <= 0.5)
    return float(np.mean(failures)) if failures else float("nan")


def _column_sharpe(matrix: np.ndarray) -> np.ndarray:
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    return np.divide(mean, std, out=np.full_like(mean, -np.inf), where=std > 1e-12)
