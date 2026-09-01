"""Statistical Analysis and Dependence-Aware Bootstrap Module.

Implements stationary / moving-block bootstrap (primary block length 21 sessions,
sensitivity at 5 and 63 sessions) for paired metric comparisons across systems,
Holm step-down and False Discovery Rate (FDR) multiplicity adjustments,
and Probability of Backtest Overfitting (PBO) accounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("statistical_bootstrap")


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    standard_error: float
    replications: int
    block_length: int


def moving_block_bootstrap_paired_diff(
    series_a: np.ndarray | pd.Series,
    series_b: np.ndarray | pd.Series,
    metric_func: Callable[[np.ndarray], float],
    block_length: int = 21,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 7,
) -> BootstrapResult:
    """Compute moving-block bootstrap paired difference metric(A) - metric(B).
    
    Preserves serial autocorrelation and heteroskedasticity in financial return series.
    """
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Series lengths must match: {len(a)} vs {len(b)}")

    n = len(a)
    if n < block_length or n < 5:
        # Fallback for very short series
        point = float(metric_func(a) - metric_func(b))
        return BootstrapResult(point, point, point, 1.0, 0.0, 0, block_length)

    point_estimate = float(metric_func(a) - metric_func(b))

    # Construct overlapping blocks
    k = max(1, block_length)
    n_blocks = n - k + 1
    blocks_a = np.array([a[i : i + k] for i in range(n_blocks)])
    blocks_b = np.array([b[i : i + k] for i in range(n_blocks)])

    rng = np.random.default_rng(random_seed)
    num_blocks_needed = int(np.ceil(n / k))

    boot_diffs = np.empty(n_bootstraps, dtype=float)
    for trial in range(n_bootstraps):
        chosen_indices = rng.integers(0, n_blocks, size=num_blocks_needed)
        sample_a = blocks_a[chosen_indices].reshape(-1)[:n]
        sample_b = blocks_b[chosen_indices].reshape(-1)[:n]

        metric_a = metric_func(sample_a)
        metric_b = metric_func(sample_b)
        boot_diffs[trial] = metric_a - metric_b

    # Remove non-finites if any
    valid_diffs = boot_diffs[np.isfinite(boot_diffs)]
    if len(valid_diffs) == 0:
        return BootstrapResult(point_estimate, point_estimate, point_estimate, 1.0, 0.0, 0, block_length)

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(valid_diffs, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(valid_diffs, 100.0 * (1.0 - alpha / 2.0)))
    se = float(np.std(valid_diffs))

    # Two-sided empirical p-value for H0: diff <= 0 (if point > 0)
    if point_estimate >= 0:
        p_val = float(np.mean(valid_diffs <= 0.0)) * 2.0
    else:
        p_val = float(np.mean(valid_diffs >= 0.0)) * 2.0
    p_val = min(1.0, max(0.0, p_val))

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_val,
        standard_error=se,
        replications=len(valid_diffs),
        block_length=block_length,
    )


def holm_bonferroni_correction(p_values: Sequence[float]) -> list[float]:
    """Apply Holm-Bonferroni step-down correction for multiple testing."""
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m <= 1:
        return [float(p) for p in p_vals]

    sort_order = np.argsort(p_vals)
    sorted_p = p_vals[sort_order]

    adjusted = np.empty(m, dtype=float)
    prev = 0.0
    for i in range(m):
        rank = i + 1
        adj = (m - rank + 1) * sorted_p[i]
        adj = max(adj, prev)  # Enforce monotonicity
        adjusted[i] = min(1.0, adj)
        prev = adjusted[i]

    # Reorder to original order
    original_order_adj = np.empty(m, dtype=float)
    original_order_adj[sort_order] = adjusted
    return [float(p) for p in original_order_adj]


def fdr_benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Apply Benjamini-Hochberg False Discovery Rate (FDR) adjustment."""
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m <= 1:
        return [float(p) for p in p_vals]

    sort_order = np.argsort(p_vals)
    sorted_p = p_vals[sort_order]

    adjusted = np.empty(m, dtype=float)
    adjusted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        rank = i + 1
        adj = (float(m) / float(rank)) * sorted_p[i]
        adjusted[i] = min(adjusted[i + 1], adj)  # Enforce monotonicity from right

    adjusted = np.clip(adjusted, 0.0, 1.0)
    original_order_adj = np.empty(m, dtype=float)
    original_order_adj[sort_order] = adjusted
    return [float(p) for p in original_order_adj]


def compute_pbo_from_matrix(
    returns_matrix: np.ndarray,
    n_splits: int = 16,
    max_combinations: int = 2000,
    random_seed: int = 7,
) -> float:
    """Compute Probability of Backtest Overfitting (PBO) from strategy returns matrix [T, N].

    Implements Combinatorially Symmetric Cross-Validation (CSCV) from Bailey et al. (2014).
    The T×N matrix has T time observations and N strategy columns (e.g. P0–P6 daily returns
    stacked as columns).  The series is divided into n_splits non-overlapping subsamples;
    each C(n_splits, n_splits//2) partition assigns half the subsamples as IS and the other
    half as OOS.  For each partition the best IS strategy is identified and its relative rank
    in the OOS Sharpe distribution is recorded.  PBO = fraction of partitions where that
    rank falls below 0.5 (i.e. the IS winner performs below the OOS median).

    When C(n_splits, n_splits//2) exceeds max_combinations, a random subsample of
    max_combinations partitions is evaluated instead to keep runtime tractable.
    """
    T, N = returns_matrix.shape
    if N <= 1 or T < 2 * n_splits:
        return 0.0

    split_size = T // n_splits
    # Build n_splits contiguous, non-overlapping subsamples
    subsamples: list[np.ndarray] = [
        returns_matrix[i * split_size : (i + 1) * split_size]
        for i in range(n_splits)
    ]

    half = n_splits // 2
    total_combos = comb(n_splits, half)

    rng = np.random.default_rng(random_seed)
    below_median = 0
    evaluated = 0

    if total_combos <= max_combinations:
        # Full combinatorial enumeration
        partition_iter = combinations(range(n_splits), half)
    else:
        # Random subsample of partitions — enumerate lazily
        all_is_choices = list(combinations(range(n_splits), half))
        chosen_idx = rng.choice(len(all_is_choices), size=max_combinations, replace=False)
        partition_iter = (all_is_choices[i] for i in chosen_idx)

    for is_indices in partition_iter:
        oos_indices = [j for j in range(n_splits) if j not in set(is_indices)]
        is_data = np.vstack([subsamples[j] for j in is_indices])
        oos_data = np.vstack([subsamples[j] for j in oos_indices])

        is_sharpe = np.nanmean(is_data, axis=0) / (np.nanstd(is_data, axis=0, ddof=1) + 1e-8)
        oos_sharpe = np.nanmean(oos_data, axis=0) / (np.nanstd(oos_data, axis=0, ddof=1) + 1e-8)

        best_is_idx = int(np.nanargmax(is_sharpe))
        # Relative rank of the IS winner in OOS: fraction of strategies that scored BELOW it
        oos_rank = float(np.sum(oos_sharpe < oos_sharpe[best_is_idx])) / float(N)
        if oos_rank < 0.5:
            below_median += 1
        evaluated += 1

    return float(below_median / evaluated) if evaluated > 0 else 0.0
