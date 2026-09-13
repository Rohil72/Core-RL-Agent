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

    # Load checkpoint and continue to epoch 6
    m_resumed = TransformerAnnual(seed=999, dropout=0.1)  # start from dummy weights
    trained_resumed, sum_resumed = train_backbone_model(
        m_resumed, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=17, min_epochs=6, max_epochs=6,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "interrupted",
        resume_from_checkpoint=tmp_path / "interrupted" / "last_checkpoint.pt",
    )

    # 3. Assert exact bitwise parity of final model parameters
    uninterrupted_params = dict(m_uninterrupted.named_parameters())
    resumed_params = dict(trained_resumed.named_parameters())
    assert set(uninterrupted_params.keys()) == set(resumed_params.keys())

    for name, p_un in uninterrupted_params.items():
        p_res = resumed_params[name]
        assert torch.equal(p_un, p_res), f"Parameter {name} does not match after checkpoint resume!"

    # 4. Assert summary and loss parity
    assert sum_uninterrupted.total_macro_steps == sum_resumed.total_macro_steps
    assert len(sum_uninterrupted.epoch_val_losses) == 6
    assert len(sum_resumed.epoch_val_losses) == 6
    for i, (l_un, l_res) in enumerate(zip(sum_uninterrupted.epoch_val_losses, sum_resumed.epoch_val_losses)):
        assert pytest.approx(l_un, rel=1e-6) == l_res, f"Val loss at epoch {i+1} mismatch: {l_un} vs {l_res}"


def test_mlp_resume_parity_on_disk(tmp_path):
    from memory_study_v2.backbones import MLPAnnual

    train_x = torch.randn(64, 966)
    train_y = torch.randn(64)
    train_mkts = np.array(["US"] * 32 + ["IN"] * 32)
    val_x = torch.randn(16, 966)
    val_y = torch.randn(16)
    val_mkts = np.array(["US"] * 8 + ["IN"] * 8)

    # Uninterrupted 6 epochs
    m_un = MLPAnnual(seed=7)
    trained_un, sum_un = train_backbone_model(
        m_un, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=7, min_epochs=6, max_epochs=6,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "mlp_uninterrupted",
    )

    # Interrupted at 3 epochs
    m_int = MLPAnnual(seed=7)
    train_backbone_model(
        m_int, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=7, min_epochs=3, max_epochs=3,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "mlp_interrupted",
        interrupt_at_epoch=3,
    )

    # Resumed to 6 epochs
    m_res = MLPAnnual(seed=999)
    trained_res, sum_res = train_backbone_model(
        m_res, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=7, min_epochs=6, max_epochs=6,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "mlp_interrupted",
        resume_from_checkpoint=tmp_path / "mlp_interrupted" / "last_checkpoint.pt",
    )

    un_params = dict(m_un.named_parameters())
    res_params = dict(trained_res.named_parameters())
    for name, p_un in un_params.items():
        p_res = res_params[name]
        assert torch.equal(p_un, p_res), f"MLP parameter {name} does not match after resume!"

    assert sum_un.total_macro_steps == sum_res.total_macro_steps
    assert pytest.approx(sum_un.best_loss, rel=1e-6) == sum_res.best_loss
