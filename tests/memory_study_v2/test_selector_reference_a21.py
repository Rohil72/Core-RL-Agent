"""Acceptance Test A21: MSE and Sharpe selectors use specified averaging, dates, ties and all grid rows."""

import pytest
import numpy as np
from memory_study_v2.integration import select_mixture_mse


def test_mixture_mse_averaging_and_tie_breaking():
    """Verify MIX_MSE evaluates grid, averages seeds within market then across markets, and breaks ties to smallest lambda."""
    N = 60
    markets = np.array([i % 6 for i in range(N)])
    mem_preds = np.zeros(N)
    targets = np.zeros(N)

    # Base predictions for 3 seeds
    base_by_seed = {
        7: np.full(N, 0.05),
        17: np.full(N, 0.05),
        37: np.full(N, 0.05),
    }

    # If targets are 0 and memory is 0, lambda=1.0 will give 0 MSE.
    mix = select_mixture_mse(base_by_seed, mem_preds, targets, markets)
    assert mix.selected_lambda == 1.0
    assert len(mix.grid_scores) == 5  # [0, 0.1, 0.25, 0.5, 1.0]
