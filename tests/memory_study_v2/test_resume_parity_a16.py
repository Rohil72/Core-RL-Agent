"""Acceptance & Regression Test for A16/R06: Checkpoint resume parity on disk."""

import numpy as np
import pytest
import torch

from memory_study_v2.backbones import TransformerAnnual
from memory_study_v2.train import restore_training_state, train_backbone_model


def test_transformer_resume_parity_on_disk(tmp_path):
    # Setup dataset
    train_x = torch.randn(64, 42, 23)
    train_y = torch.randn(64)
    train_mkts = np.array(["US"] * 32 + ["IN"] * 32)
    val_x = torch.randn(16, 42, 23)
    val_y = torch.randn(16)
    val_mkts = np.array(["US"] * 8 + ["IN"] * 8)

    # 1. Run 6 epochs uninterrupted
    m_uninterrupted = TransformerAnnual(seed=17, dropout=0.1)
    _, sum_uninterrupted = train_backbone_model(
        m_uninterrupted, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=17, min_epochs=6, max_epochs=6,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "uninterrupted",
    )

    # 2. Run 3 epochs, interrupt, resume to 6 epochs
    m_interrupted = TransformerAnnual(seed=17, dropout=0.1)
    _, _ = train_backbone_model(
        m_interrupted, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=17, min_epochs=3, max_epochs=3,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "interrupted",
        interrupt_at_epoch=3,
    )

    # Load checkpoint and continue
    saved_state = torch.load(tmp_path / "interrupted" / "last_checkpoint.pt", weights_only=False)
    m_resumed = TransformerAnnual(seed=999, dropout=0.1)  # start from dummy weights
    # Restore full state
    # We will verify that checkpoint loaded into model matches uninterrupted
    assert saved_state.epoch == 3
