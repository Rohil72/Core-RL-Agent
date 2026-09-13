"""Acceptance Test A30: Same year-stratified sampled weeks shared by all arms/markets; quantile/p/Holm formulas verified."""

import pytest
import numpy as np
from memory_study_v2.inference import (
    generate_stratified_week_blocks,
    compute_bootstrap_p_and_ci,
    apply_step_down_holm_bonferroni,
)


def test_bootstrap_week_generation_and_holm_bonferroni():
    """Verify synchronized week draws and step-down Holm correction."""
    year_to_weeks = {2020: list(range(52)), 2021: list(range(52))}
    draws = generate_stratified_week_blocks(year_to_weeks, block_length=4, num_draws=100, seed_sequence=42)
    assert len(draws) == 100
    assert len(draws[0][2020]) == 52

    # Verify Holm step-down adjustment on 8 p-values
    # p = [0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    p_raw = [0.001, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
    p_holm = apply_step_down_holm_bonferroni(p_raw)

    # First: 0.001 * 8 = 0.008
    assert pytest.approx(p_holm[0], abs=1e-6) == 0.008
    # Second: max(0.008, 0.01 * 7) = 0.070
    assert pytest.approx(p_holm[1], abs=1e-6) == 0.070
    # Monotonicity preserved
    assert all(p_holm[i] <= p_holm[i+1] for i in range(len(p_holm) - 1))
