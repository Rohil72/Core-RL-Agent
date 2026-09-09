"""Unit tests for Transformer sequence dimension handling and temporal attention.

Verifies:
1. Shape contract for multi-step temporal sequences [B, T, D] with T > 1.
2. Temporal order sensitivity: permuting sequence time steps alters output latents,
   confirming active cross-temporal attention (not bypassed).
3. Shape contract for single-step screening input [B, D].
4. Compatibility with saved checkpoint weights.
"""

from pathlib import Path
import pytest
import torch

from src.models.patch_transformer_model import (
    GlobalTemporalTransformer,
    HierarchicalPatchTransformerCycleModel,
)

CHECKPOINT_PATH = Path("FINAL_SUBMISSION_PACKAGE/models/global_transformer_seed_7.pt")


def test_global_temporal_transformer_multistep_sequence():
    """Verify that multi-step sequences [B, T, D] are properly processed."""
    batch_size = 4
    seq_len = 21
    input_dim = 23
    model = GlobalTemporalTransformer(input_dim=input_dim, embed_dim=64, num_heads=4, latent_dim=128)
    model.eval()

    x = torch.randn(batch_size, seq_len, input_dim)
    with torch.no_grad():
        latent, pred = model(x)

    assert latent.shape == (batch_size, 128), f"Expected (4, 128), got {latent.shape}"
    assert pred.shape == (batch_size, 1), f"Expected (4, 1), got {pred.shape}"
    assert torch.isfinite(latent).all(), "Latent representations contain NaN/Inf"
    assert torch.isfinite(pred).all(), "Predictions contain NaN/Inf"


def test_global_temporal_transformer_temporal_sensitivity():
    """Verify that permuting temporal order alters output latents (active temporal attention)."""
    batch_size = 2
    seq_len = 21
    input_dim = 23
    model = GlobalTemporalTransformer(input_dim=input_dim, embed_dim=64, num_heads=4, latent_dim=128)
    model.eval()

    # Create ordered sequence where time steps have distinct patterns
    time_trend = torch.linspace(-1.0, 1.0, seq_len).view(1, seq_len, 1).expand(batch_size, -1, input_dim)
    x_forward = torch.randn(batch_size, seq_len, input_dim) * 0.1 + time_trend
    # Reverse temporal order
    x_reversed = torch.flip(x_forward, dims=[1])

    with torch.no_grad():
        latent_fwd, pred_fwd = model(x_forward)
        latent_rev, pred_rev = model(x_reversed)

    # If attention is active across time, reversing the order MUST produce different states
    diff = (latent_fwd - latent_rev).abs().max().item()
    assert diff > 1e-4, f"Model is invariant to time reversal (diff={diff:.6f}), indicating temporal attention is degenerate"


def test_global_temporal_transformer_singlestep_backward_compat():
    """Verify single-step screening input [B, D] remains functional for legacy callers."""
    batch_size = 8
    input_dim = 23
    model = GlobalTemporalTransformer(input_dim=input_dim, embed_dim=64, num_heads=4, latent_dim=128)
    model.eval()

    x_2d = torch.randn(batch_size, input_dim)
    with torch.no_grad():
        latent, pred = model(x_2d)

    assert latent.shape == (batch_size, 128)
    assert pred.shape == (batch_size, 1)


def test_global_temporal_transformer_invalid_dimensions():
    """Verify that inputs with dimension < 2 or > 3 raise ValueError."""
    model = GlobalTemporalTransformer()
    with pytest.raises(ValueError):
        model(torch.randn(23))  # 1D
    with pytest.raises(ValueError):
        model(torch.randn(2, 5, 23, 4))  # 4D


def test_checkpoint_weights_loadable():
    """Verify that the official checkpoint weights load cleanly into GlobalTemporalTransformer."""
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Checkpoint not found at {CHECKPOINT_PATH}")

    model = GlobalTemporalTransformer(input_dim=23, embed_dim=64, num_heads=4, latent_dim=128)
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    load_res = model.load_state_dict(state_dict)
    assert len(load_res.missing_keys) == 0, f"Missing keys: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys: {load_res.unexpected_keys}"

    # Test inference with loaded weights
    model.eval()
    x = torch.randn(2, 21, 23)
    with torch.no_grad():
        latent, pred = model(x)
    assert latent.shape == (2, 128)
    assert pred.shape == (2, 1)
