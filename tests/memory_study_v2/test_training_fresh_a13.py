"""Acceptance Test A13: Both backbones initialize fresh, receive gradients and update weights; no old Transformer loader."""

import pytest
import torch
import torch.nn as nn
from memory_study_v2.backbones import MLPAnnual, TransformerAnnual


def test_mlp_fresh_initialization_gradients_and_updates():
    """Verify MLP initializes fresh from seed, receives finite gradients, and updates weights."""
    seed = 7
    model = MLPAnnual(seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 16
    x = torch.randn(B, 966, dtype=torch.float32)
    y = torch.randn(B, dtype=torch.float32)

    # Record initial weights
    w_initial = model.fc1.weight.clone().detach()

    # Forward
    pred = model(x)
    assert pred.shape == (B,)

    loss = nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)

    # Backward
    loss.backward()

    # Check all parameters have finite, non-zero gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient is None for {name}"
        assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient in {name}"
        assert torch.sum(torch.abs(p.grad)) > 0.0, f"Zero gradient in {name}"

    # Step
    optimizer.step()
    w_updated = model.fc1.weight.clone().detach()
    assert not torch.allclose(w_initial, w_updated), "Weights were not updated after optimizer step!"


def test_transformer_fresh_initialization_gradients_and_updates():
    """Verify Transformer initializes fresh, receives finite gradients, and updates weights."""
    seed = 17
    model = TransformerAnnual(seed=seed, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    B = 8
    x = torch.randn(B, 42, 23, dtype=torch.float32)
    y = torch.randn(B, dtype=torch.float32)

    w_initial = model.input_proj.weight.clone().detach()

    pred = model(x)
    assert pred.shape == (B,)

    loss = nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)

    loss.backward()

    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient is None for {name}"
        assert torch.all(torch.isfinite(p.grad)), f"Non-finite gradient in {name}"
        assert torch.sum(torch.abs(p.grad)) > 0.0, f"Zero gradient in {name}"

    optimizer.step()
    w_updated = model.input_proj.weight.clone().detach()
    assert not torch.allclose(w_initial, w_updated), "Transformer weights were not updated!"
