"""Acceptance Test A22: Lambda endpoints alias identical predictions/paths without inflating independence counts."""

import pytest
import numpy as np
from memory_study_v2.integration import select_mixture_mse


def test_mixture_endpoint_aliases():
    """Verify lambda=0.0 aliases BASE and lambda=1.0 aliases MEM_SIM."""
    N = 20
    markets = np.array([i % 2 for i in range(N)])

    # Case 1: Base is perfect (MSE=0) -> lambda=0 selected -> alias of BASE
    base_by_seed = {7: np.zeros(N)}
    mem_preds = np.ones(N)
    targets = np.zeros(N)

    mix_0 = select_mixture_mse(base_by_seed, mem_preds, targets, markets)
    assert mix_0.selected_lambda == 0.0
    assert mix_0.is_alias is True
    assert mix_0.alias_of == "BASE"

    # Case 2: Memory is perfect (MSE=0) -> lambda=1 selected -> alias of MEM_SIM
    base_by_seed_bad = {7: np.ones(N)}
    mix_1 = select_mixture_mse(base_by_seed_bad, dev_mem_preds=np.zeros(N), dev_targets=targets, dev_markets=markets)
    assert mix_1.selected_lambda == 1.0
    assert mix_1.is_alias is True
    assert mix_1.alias_of == "MEM_SIM"
