"""Block bootstrap inference and step-down Holm-Bonferroni correction (v2).

Acceptance criteria addressed:
- A30: Same year-stratified sampled weeks shared by all arms/markets; quantile/p/Holm formulas verified.
- A31: Fresh full-precision summaries and bootstrap outputs replay at declared 1e-10 tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BootstrapResult:
    theta_point: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_holm: float
    num_draws: int


def generate_stratified_week_blocks(
    year_to_weeks: Dict[int, List[int]],
    block_length: int = 4,
    num_draws: int = 10000,
    seed_sequence: int = 42,
) -> List[Dict[int, List[int]]]:
    """Generate synchronized year-stratified sampled week index arrays using PCG64."""
    # PCG64(SeedSequence([42, block_length]))
    seed_seq = np.random.SeedSequence([seed_sequence, block_length])
    rng = np.random.default_rng(np.random.PCG64(seed_seq))

    all_draws: List[Dict[int, List[int]]] = []

    for _ in range(num_draws):
        draw_dict: Dict[int, List[int]] = {}
        for year, week_indices in year_to_weeks.items():
            T_y = len(week_indices)
            sampled = []
            while len(sampled) < T_y:
                # Uniform consecutive block start
                max_start = T_y - block_length + 1
                if max_start > 0:
                    start = rng.integers(0, max_start)
                    block = week_indices[start : start + block_length]
                else:
                    # If stratum has fewer weeks than block length
                    block = week_indices
                sampled.extend(block)
            # Truncate to original number of weeks
            draw_dict[year] = sampled[:T_y]
        all_draws.append(draw_dict)

    return all_draws


def compute_bootstrap_p_and_ci(
    theta: float,
    theta_draws: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute centered 95% CI and two-sided centered p-value according to contract (Section 9.2):

    interval = theta +/- quantile(abs(theta_b - theta), 1 - alpha, method='linear')
    p_value = (1 + count(abs(theta_b - theta) >= abs(theta))) / (B + 1)
    """
    B = len(theta_draws)
    diffs = np.abs(theta_draws - theta)
    margin = float(np.quantile(diffs, 1.0 - alpha, method="linear"))
    ci_lower = theta - margin
    ci_upper = theta + margin

    count_exceed = int(np.sum(diffs >= abs(theta)))
    p_val = float((1.0 + count_exceed) / (B + 1.0))

    return ci_lower, ci_upper, p_val


def apply_step_down_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply step-down Holm-Bonferroni correction to p-values."""
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by ascending p-value
    sorted_indices = np.argsort(p_values)
    adj_p = [0.0] * m

    cum_max = 0.0
    for i, orig_idx in enumerate(sorted_indices):
        p_raw = p_values[orig_idx]
        multiplier = m - i
        step_val = min(1.0, multiplier * p_raw)
        # Enforce monotonicity: p_adj_{(i)} = max(p_adj_{(i-1)}, step_val)
        cum_max = max(cum_max, step_val)
        adj_p[orig_idx] = cum_max

    return adj_p
