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


def test_optimizer_parameter_groups_bind_to_configuration(tmp_path):
    """Verify optimizer parameters strictly bind to approved config (weight_decay 0.0001, lr 0.001) (C3)."""
    train_x = torch.randn(64, 966)
    train_y = torch.randn(64)
    train_mkts = np.array(["US"] * 64)
    val_x = torch.randn(16, 966)
    val_y = torch.randn(16)
    val_markets = np.array(["US"] * 16)

    model = MLPAnnual(seed=7)
    # Default call without overrides should bind to config.proposed.json: weight_decay 0.0001
    trained_model, summary = train_backbone_model(
        model, train_x, train_y, train_mkts,
        val_x, val_y, val_markets,
        seed=7, min_epochs=1, max_epochs=1,
        micro_batch_size=32, effective_batch_size=64,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    # Load last checkpoint and inspect optimizer param groups
    chk = torch.load(tmp_path / "checkpoints" / "last_checkpoint.pt", weights_only=False)
    opt_state = chk.optimizer_state
    for pg in opt_state["param_groups"]:
        assert pg["weight_decay"] == 0.0001, f"Expected weight_decay 0.0001, got {pg['weight_decay']}"
        assert pg["lr"] == 0.001, f"Expected lr 0.001, got {pg['lr']}"
        assert pg["betas"] == (0.9, 0.999), f"Expected betas (0.9, 0.999), got {pg['betas']}"
        assert pg["eps"] == 1e-08, f"Expected eps 1e-8, got {pg['eps']}"


def test_missing_resume_checkpoint_raises_filenotfounderror():
    """Verify missing resume checkpoint raises explicit FileNotFoundError (C3, C5)."""
    train_x = torch.randn(64, 966)
    train_y = torch.randn(64)
    train_mkts = np.array(["US"] * 64)
    val_x = torch.randn(16, 966)
    val_y = torch.randn(16)
    val_markets = np.array(["US"] * 16)

    model = MLPAnnual(seed=7)
    with pytest.raises(FileNotFoundError, match="Requested resume checkpoint does not exist"):
        train_backbone_model(
            model, train_x, train_y, train_mkts,
            val_x, val_y, val_markets,
            resume_from_checkpoint="nonexistent_checkpoint_path.pt",
        )
