"""Acceptance & Regression Test for R07: Strict seeded attention initialization."""

import numpy as np
import pytest
import torch

from memory_study_v2.backbones import TransformerAnnual


def test_transformer_all_parameters_initialized_from_seed():
    t1 = TransformerAnnual(seed=17)

    # Do unrelated global random calls
    _ = torch.randn(1000)
    _ = np.random.rand(500)

    t2 = TransformerAnnual(seed=17)

    # Assert every single parameter tensor is identical
    for (name1, p1), (name2, p2) in zip(t1.named_parameters(), t2.named_parameters()):
        assert name1 == name2
        assert torch.equal(p1, p2), f"Parameter {name1} differed between same-seed constructions!"


def test_transformer_distinct_seeds_produce_distinct_weights():
    t1 = TransformerAnnual(seed=7)
    t2 = TransformerAnnual(seed=17)

    # Check all randomized linear & attention weight matrices
    randomized_weights_differ = 0
    total_randomized_weights = 0
    for (name1, p1), (name2, p2) in zip(t1.named_parameters(), t2.named_parameters()):
        if "weight" in name1 and "norm" not in name1 and "ln" not in name1:
            total_randomized_weights += 1
            if not torch.equal(p1, p2):
                randomized_weights_differ += 1

    assert total_randomized_weights >= 10
    assert randomized_weights_differ == total_randomized_weights, (
        f"Only {randomized_weights_differ}/{total_randomized_weights} randomized weight tensors differed!"
    )
