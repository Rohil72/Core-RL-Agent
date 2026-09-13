"""Acceptance & Regression Test for R02: Integrated training runner."""

import numpy as np
import pytest
import torch

from memory_study_v2.backbones import MLPAnnual, TransformerAnnual
from memory_study_v2.train import train_backbone_model


def test_train_backbone_model_runner_execution(tmp_path):
    train_x = torch.randn(128, 966)
    train_y = torch.randn(128)
    train_mkts = np.array(["US"] * 64 + ["IN"] * 64)

    val_x = torch.randn(32, 966)
    val_y = torch.randn(32)
    val_mkts = np.array(["US"] * 16 + ["IN"] * 16)

    model = MLPAnnual(seed=7)
    trained_model, summary = train_backbone_model(
        model, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=7, min_epochs=5, max_epochs=6,
        micro_batch_size=32, effective_batch_size=64,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert summary.epochs_trained >= 5
    assert summary.total_macro_steps > 0
    assert summary.samples_per_market == {"US": 64, "IN": 64}
    assert (tmp_path / "checkpoints" / "last_checkpoint.pt").exists()
    assert (tmp_path / "checkpoints" / "best_checkpoint.pt").exists()
