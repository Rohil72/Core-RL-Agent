"""Acceptance Test A14: Loss weights and microbatch accumulation match macro reference on dropout-disabled fixture."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from memory_study_v2.backbones import MLPAnnual
from memory_study_v2.train import compute_equal_market_weights


def test_microbatch_accumulation_equals_macro_reference():
    """Verify that microbatch gradient accumulation strictly equals macro batch gradient."""
    torch.manual_seed(42)
    model_macro = MLPAnnual(seed=42)
    model_micro = MLPAnnual(seed=42)

    # Macro batch size 32, 4 microbatches of size 8
    macro_size = 32
    micro_size = 8
    num_micro = macro_size // micro_size

    x = torch.randn(macro_size, 966, dtype=torch.float32)
    y = torch.randn(macro_size, dtype=torch.float32)

    # 4 distinct markets with unequal representation
    markets = np.array([0]*16 + [1]*8 + [2]*4 + [3]*4)
    weights = torch.from_numpy(compute_equal_market_weights(markets, target_num_markets=4))

    # --- Macro gradient computation ---
    pred_macro = model_macro(x)
    diff_macro = pred_macro - y
    weighted_loss_macro = torch.sum(weights * (diff_macro ** 2)) / torch.sum(weights)
    weighted_loss_macro.backward()

    # --- Microbatch accumulation ---
    model_micro.zero_grad()
    total_weight = torch.sum(weights)
    for i in range(num_micro):
        idx_slice = slice(i * micro_size, (i + 1) * micro_size)
        x_micro = x[idx_slice]
        y_micro = y[idx_slice]
        w_micro = weights[idx_slice]

        pred_micro = model_micro(x_micro)
        diff_micro = pred_micro - y_micro
        # Micro loss scaled so sum of gradients equals macro gradient
        loss_micro = torch.sum(w_micro * (diff_micro ** 2)) / total_weight
        loss_micro.backward()

    # Compare gradients across all parameters
    for (name_m, p_macro), (name_u, p_micro) in zip(model_macro.named_parameters(), model_micro.named_parameters()):
        assert p_macro.grad is not None
        assert p_micro.grad is not None
        torch.testing.assert_close(
            p_macro.grad,
            p_micro.grad,
            rtol=1e-5,
            atol=1e-6,
            msg=f"Gradient accumulation mismatch in parameter {name_m}"
        )
