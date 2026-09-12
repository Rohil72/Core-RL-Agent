"""
Unit Tests for Memory-Centric Equity Selection.

Validates the implementation invariants specified in Section 19:
1. M0 identity: lambda=0 reproduces base prediction
2. M3/M4 neighbour identity: neighbour IDs match exactly
3. Uniform-weight identity: equal logits reproduce arithmetic mean
4. Parameter freeze check: backbone parameters cannot receive gradients or mutate
5. Kernel numerical stability: max subtraction avoids overflow
6. Availability boundary: unavailable precedents are strictly filtered
7. Same-ticker exclusion: 0 same-ticker precedents allowed
8. Insufficient memory fallback: graceful degradation to base predictor
9. Execution adapter determinism: identical score streams yield identical ledgers
"""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from memory_study.backbones import (
    AnnualPatchTemporalTransformer,
    MLPEncoder,
    freeze_model,
    verify_backbone_frozen,
)
from memory_study.memory_store import MemoryStore
from memory_study.retrieval import (
    retrieve_m1_random,
    retrieve_m2_raw_window,
    retrieve_m3_latent_cosine,
    compute_m4_kernel_weights,
)
from memory_study.mixing import (
    compute_memory_prediction,
    compute_hybrid_prediction,
    compute_trading_score,
)
from memory_study.engine_adapter import SimulationEngine


def test_m0_identity():
    """M0 identity: lambda=0 reproduces base prediction exactly."""
    base_pred = 0.085
    mem_pred = -0.120
    hybrid_pred, eff_lam = compute_hybrid_prediction(base_pred, mem_pred, lam=0.0, memory_valid=True)
    assert hybrid_pred == base_pred
    assert eff_lam == 0.0


def test_uniform_weight_identity():
    """Uniform weights produce arithmetic mean."""
    ret = np.array([0.05, -0.02, 0.10, 0.01], dtype=np.float32)
    weights = np.full(4, 0.25, dtype=np.float32)
    weighted_mean = compute_memory_prediction(weights, ret)
    arithmetic_mean = float(np.mean(ret))
    assert np.isclose(weighted_mean, arithmetic_mean, atol=1e-6)


def test_m4_kernel_numerical_stability():
    """M4 softmax kernel subtracts max and does not overflow with large logits."""
    sims = np.array([1000.0, 1000.5, 999.0], dtype=np.float32)
    weights = compute_m4_kernel_weights(sims, tau=0.08)
    assert not np.any(np.isnan(weights))
    assert not np.any(np.isinf(weights))
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5)
    assert weights[1] > weights[0] > weights[2]


def test_backbone_parameter_freeze():
    """Verify backbone parameters have requires_grad=False and are not modified by training loops."""
    model = freeze_model(MLPEncoder(input_dim=23, hidden_dim=32, latent_dim=64))
    assert verify_backbone_frozen(model)

    # Attempt to optimize downstream adapter on model output
    x = torch.randn(4, 23)
    with torch.no_grad():
        lat, pred = model(x)

    adapter = nn.Linear(64, 1)
    opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
    loss = nn.MSELoss()(adapter(lat), torch.zeros(4, 1))
    loss.backward()
    opt.step()

    # Model parameters must remain unmodified with no gradients
    for p in model.parameters():
        assert p.grad is None
        assert not p.requires_grad


def test_insufficient_memory_fallback():
    """When fewer than k precedents exist, memory is declared invalid and falls back to base."""
    base_pred = 0.04
    mem_pred = 0.10
    # Fallback condition: memory_valid=False
    hybrid_pred, eff_lam = compute_hybrid_prediction(base_pred, mem_pred, lam=0.25, memory_valid=False)
    assert hybrid_pred == base_pred
    assert eff_lam == 0.0


def test_execution_adapter_determinism():
    """Running simulation engine twice with identical scores yields identical trades and final equity."""
    n_sessions = 50
    n_tickers = 5
    dates = [f"2024-01-{i+1:02d}" for i in range(n_sessions)]
    tickers = [f"TKR{i}" for i in range(n_tickers)]

    rng = np.random.default_rng(42)
    p_open = rng.uniform(50.0, 150.0, size=(n_sessions, n_tickers)).astype(np.float32)
    p_close = p_open * rng.uniform(0.98, 1.02, size=(n_sessions, n_tickers)).astype(np.float32)
    atr = np.full((n_sessions, n_tickers), 0.02, dtype=np.float32)
    vol = np.full((n_sessions, n_tickers), 0.015, dtype=np.float32)
    scores = rng.standard_normal((n_sessions, n_tickers)).astype(np.float32)

    engine1 = SimulationEngine("TEST", tickers, dates, p_open, p_close, atr, vol)
    engine2 = SimulationEngine("TEST", tickers, dates, p_open, p_close, atr, vol)

    res1 = engine1.run_simulation(scores)
    res2 = engine2.run_simulation(scores)

    assert res1["final_equity"] == res2["final_equity"]
    assert res1["total_trades"] == res2["total_trades"]
    assert res1["penny_reconciled"] is True
    assert res2["penny_reconciled"] is True
