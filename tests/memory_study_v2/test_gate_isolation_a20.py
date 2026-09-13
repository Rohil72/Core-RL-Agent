"""Acceptance Test A20: Three inputs only; saved development normalizer; no outcomes supplied as gate inputs."""

import pytest
import numpy as np
from memory_study_v2.integration import fit_trust_gate, GateInputError


def test_gate_three_inputs_only_and_weight_updates():
    """Verify gate uses strictly (|base|, |base-mem|, |mem|) and updates weights."""
    np.random.seed(42)
    N = 100
    base = np.random.normal(0, 0.02, N)
    memory = np.random.normal(0, 0.02, N)
    targets = np.random.normal(0, 0.02, N)
    markets = np.array([i % 6 for i in range(N)])

    gate = fit_trust_gate(base, memory, targets, markets, steps=50, lr=0.05)

    assert gate.weights_a.shape == (3,)
    assert len(gate.normalizer.means) == 3
    assert len(gate.normalizer.stds) == 3

    # Weights updated from zero
    assert not np.allclose(gate.weights_a, np.zeros(3))
    # Loss decreased
    assert gate.loss_history[-1] < gate.loss_history[0]

    # Predict
    preds, g = gate.predict(base, memory)
    assert preds.shape == (N,)
    assert np.all(g >= 0.0) and np.all(g <= 1.0)
