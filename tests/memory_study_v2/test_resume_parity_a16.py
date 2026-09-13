"""Acceptance & Regression Test for A16/R06/C5: Checkpoint resume parity on disk."""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from memory_study_v2.backbones import MLPAnnual, TransformerAnnual
from memory_study_v2.train import restore_training_state, train_backbone_model


def assert_optimizer_states_equal(opt_state_1: dict, opt_state_2: dict):
    """Verify bitwise tensor equality of optimizer states (momentum, second moments, steps) (C5)."""
    assert len(opt_state_1["state"]) == len(opt_state_2["state"]), "Optimizer state tensor count mismatch"
    for k in opt_state_1["state"]:
        s1 = opt_state_1["state"][k]
        s2 = opt_state_2["state"][k]
        assert s1["step"] == s2["step"], f"Optimizer step mismatch for param {k}"
        if "exp_avg" in s1:
            assert torch.equal(s1["exp_avg"], s2["exp_avg"]), f"Optimizer exp_avg mismatch for param {k}"
        if "exp_avg_sq" in s1:
            assert torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"]), f"Optimizer exp_avg_sq mismatch for param {k}"


def test_transformer_resume_parity_on_disk(tmp_path):
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

    # 4. Assert optimizer state equality (C5)
    chk_un = torch.load(tmp_path / "uninterrupted" / "last_checkpoint.pt", weights_only=False)
    chk_res = torch.load(tmp_path / "interrupted" / "last_checkpoint.pt", weights_only=False)
    assert_optimizer_states_equal(chk_un.optimizer_state, chk_res.optimizer_state)

    # 5. Assert summary and loss parity
    assert sum_uninterrupted.total_macro_steps == sum_resumed.total_macro_steps
    assert len(sum_uninterrupted.epoch_val_losses) == 6
    assert len(sum_resumed.epoch_val_losses) == 6
    for i, (l_un, l_res) in enumerate(zip(sum_uninterrupted.epoch_val_losses, sum_resumed.epoch_val_losses)):
        assert pytest.approx(l_un, rel=1e-6) == l_res, f"Val loss at epoch {i+1} mismatch: {l_un} vs {l_res}"


def test_mlp_resume_parity_on_disk(tmp_path):
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

    chk_un = torch.load(tmp_path / "mlp_uninterrupted" / "last_checkpoint.pt", weights_only=False)
    chk_res = torch.load(tmp_path / "mlp_interrupted" / "last_checkpoint.pt", weights_only=False)
    assert_optimizer_states_equal(chk_un.optimizer_state, chk_res.optimizer_state)

    assert sum_un.total_macro_steps == sum_res.total_macro_steps
    assert pytest.approx(sum_un.best_loss, rel=1e-6) == sum_res.best_loss


def test_resume_parity_mid_epoch_macro_boundary(tmp_path):
    """Verify bitwise resume parity when interrupted at a mid-epoch macro step (C5)."""
    # 64 samples with effective batch 32 = 2 macro steps per epoch
    train_x = torch.randn(64, 966)
    train_y = torch.randn(64)
    train_mkts = np.array(["US"] * 32 + ["IN"] * 32)
    val_x = torch.randn(16, 966)
    val_y = torch.randn(16)
    val_mkts = np.array(["US"] * 8 + ["IN"] * 8)

    # 1. Run 4 epochs uninterrupted (8 macro steps total)
    m_un = MLPAnnual(seed=42)
    trained_un, sum_un = train_backbone_model(
        m_un, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=42, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "uninterrupted_mid",
    )

    # 2. Run and interrupt at macro step 5 (which is the first macro step of epoch 3)
    m_int = MLPAnnual(seed=42)
    train_backbone_model(
        m_int, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=42, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "interrupted_mid",
        interrupt_at_macro_step=5,
    )
    chk_mid = tmp_path / "interrupted_mid" / "last_checkpoint.pt"
    assert chk_mid.exists()

    # 3. Resume from macro step 5 and complete to epoch 4
    m_res = MLPAnnual(seed=888)
    trained_res, sum_res = train_backbone_model(
        m_res, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=42, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "interrupted_mid",
        resume_from_checkpoint=chk_mid,
    )

    # Verify bitwise parity of model parameters
    for name, p_un in m_un.named_parameters():
        p_res = dict(trained_res.named_parameters())[name]
        assert torch.equal(p_un, p_res), f"Mid-epoch resume parameter {name} mismatch!"

    # Verify optimizer state bitwise equality
    chk_un = torch.load(tmp_path / "uninterrupted_mid" / "last_checkpoint.pt", weights_only=False)
    chk_res = torch.load(tmp_path / "interrupted_mid" / "last_checkpoint.pt", weights_only=False)
    assert_optimizer_states_equal(chk_un.optimizer_state, chk_res.optimizer_state)
    assert sum_un.total_macro_steps == sum_res.total_macro_steps


def test_resume_parity_fresh_process(tmp_path):
    """Verify bitwise resume parity when continued in a completely fresh OS process (C5)."""
    # Setup fixed dataset on disk
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    train_x = torch.tensor(rng.normal(0, 1, (64, 966)), dtype=torch.float32)
    train_y = torch.tensor(rng.normal(0, 0.05, 64), dtype=torch.float32)
    train_mkts = np.array(["US"] * 32 + ["IN"] * 32)
    val_x = torch.tensor(rng.normal(0, 1, (16, 966)), dtype=torch.float32)
    val_y = torch.tensor(rng.normal(0, 0.05, 16), dtype=torch.float32)
    val_markets = np.array(["US"] * 8 + ["IN"] * 8)

    torch.save({
        "train_x": train_x, "train_y": train_y, "train_mkts": train_mkts,
        "val_x": val_x, "val_y": val_y, "val_markets": val_markets
    }, data_dir / "dataset.pt")

    # 1. Run uninterrupted 4 epochs in current process
    m_un = MLPAnnual(seed=7)
    train_backbone_model(
        m_un, train_x, train_y, train_mkts,
        val_x, val_y, val_markets,
        seed=7, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "fresh_uninterrupted",
    )

    # 2. Run 2 epochs and save checkpoint
    m_part = MLPAnnual(seed=7)
    train_backbone_model(
        m_part, train_x, train_y, train_mkts,
        val_x, val_y, val_markets,
        seed=7, min_epochs=2, max_epochs=2,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "fresh_interrupted",
        interrupt_at_epoch=2,
    )

    # 3. Spawn a fresh Python subprocess to resume from checkpoint to epoch 4
    worker_script = tmp_path / "worker_resume.py"
    repo_root = str(Path(__file__).resolve().parents[2])
    with open(worker_script, "w", encoding="utf-8") as f:
        f.write(f'''
import sys
from pathlib import Path
sys.path.insert(0, r"{repo_root}")
import torch
import numpy as np
from memory_study_v2.backbones import MLPAnnual
from memory_study_v2.train import train_backbone_model

data_dir = Path(sys.argv[1])
chk_dir = Path(sys.argv[2])

data = torch.load(data_dir / "dataset.pt", weights_only=False)
model = MLPAnnual(seed=999)  # fresh dummy weights
train_backbone_model(
    model, data["train_x"], data["train_y"], data["train_mkts"],
    data["val_x"], data["val_y"], data["val_markets"],
    seed=7, min_epochs=4, max_epochs=4,
    micro_batch_size=16, effective_batch_size=32,
    checkpoint_dir=chk_dir,
    resume_from_checkpoint=chk_dir / "last_checkpoint.pt",
)
''')

    res = subprocess.run(
        [sys.executable, str(worker_script), str(data_dir), str(tmp_path / "fresh_interrupted")],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert res.returncode == 0, f"Subprocess failed: {res.stderr}"

    # 4. Compare resulting checkpoint state from fresh process against uninterrupted baseline
    chk_un = torch.load(tmp_path / "fresh_uninterrupted" / "last_checkpoint.pt", weights_only=False)
    chk_fresh = torch.load(tmp_path / "fresh_interrupted" / "last_checkpoint.pt", weights_only=False)

    # Parameter equality
    for k in chk_un.model_state:
        assert torch.equal(chk_un.model_state[k], chk_fresh.model_state[k]), f"Fresh process parameter {k} mismatch!"

    # Optimizer state equality
    assert_optimizer_states_equal(chk_un.optimizer_state, chk_fresh.optimizer_state)


def test_resume_parity_final_macro_update_boundary(tmp_path):
    """Verify exact parity when interrupted after the final macro update of an epoch (C5).

    Regression test for auditor finding:
    Interruption on final macro update of epoch must run validation and checkpoint selection
    before declaring epoch checkpoint, ensuring zero skipped validation passes on resume.
    """
    # 64 samples with effective batch 32 = 2 macro steps per epoch
    train_x = torch.randn(64, 966)
    train_y = torch.randn(64)
    train_mkts = np.array(["US"] * 32 + ["IN"] * 32)
    val_x = torch.randn(16, 966)
    val_y = torch.randn(16)
    val_mkts = np.array(["US"] * 8 + ["IN"] * 8)

    # 1. Uninterrupted run for 4 epochs (8 macro steps total)
    m_un = MLPAnnual(seed=42)
    trained_un, sum_un = train_backbone_model(
        m_un, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=42, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "uninterrupted_final_macro",
    )

    # 2. Interrupted after macro step 6 (the FINAL macro update of epoch 3: 2 steps/epoch * 3 = 6)
    m_int = MLPAnnual(seed=42)
    train_backbone_model(
        m_int, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=42, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "interrupted_final_macro",
        interrupt_at_macro_step=6,
    )
    chk_final = tmp_path / "interrupted_final_macro" / "last_checkpoint.pt"
    assert chk_final.exists()

    # Verify state saved at final macro update recorded the completed validation pass
    state_at_int = torch.load(chk_final, weights_only=False)
    assert len(state_at_int.epoch_val_losses) == 3, "Epoch 3 validation loss must be recorded before checkpointing!"

    # 3. Resume from macro step 6 and complete to epoch 4
    m_res = MLPAnnual(seed=888)
    trained_res, sum_res = train_backbone_model(
        m_res, train_x, train_y, train_mkts,
        val_x, val_y, val_mkts,
        seed=42, min_epochs=4, max_epochs=4,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=tmp_path / "interrupted_final_macro",
        resume_from_checkpoint=chk_final,
    )

    # Compare validation history across all 4 epochs
    assert len(sum_res.epoch_val_losses) == len(sum_un.epoch_val_losses) == 4
    for i, (l_un, l_res) in enumerate(zip(sum_un.epoch_val_losses, sum_res.epoch_val_losses)):
        assert pytest.approx(l_un, rel=1e-6) == l_res, f"Validation loss at epoch {i+1} mismatch: {l_un} vs {l_res}"

    # Best epoch and best loss comparison
    assert sum_un.best_epoch == sum_res.best_epoch
    assert pytest.approx(sum_un.best_loss, rel=1e-6) == sum_res.best_loss
    assert sum_un.total_macro_steps == sum_res.total_macro_steps == 8

    # Parameter equality
    for name, p_un in m_un.named_parameters():
        p_res = dict(trained_res.named_parameters())[name]
        assert torch.equal(p_un, p_res), f"Parameter {name} mismatch after final macro resume!"

    # Optimizer state equality
    chk_un = torch.load(tmp_path / "uninterrupted_final_macro" / "last_checkpoint.pt", weights_only=False)
    chk_res = torch.load(tmp_path / "interrupted_final_macro" / "last_checkpoint.pt", weights_only=False)
    assert_optimizer_states_equal(chk_un.optimizer_state, chk_res.optimizer_state)


def test_fail_closed_production_config_enforcement(tmp_path):
    """Verify production mode strictly enforces config existence, authorization, and rejects unapproved overrides (C3)."""
    train_x = torch.randn(32, 966)
    train_y = torch.randn(32)
    train_mkts = np.array(["US"] * 32)
    val_x = torch.randn(16, 966)
    val_y = torch.randn(16)
    val_mkts = np.array(["US"] * 16)

    m = MLPAnnual(seed=7)

    # 1. Missing config in production mode raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="In production mode, configuration file"):
        train_backbone_model(
            m, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=tmp_path / "nonexistent.json",
            execution_mode="production",
        )

    # 2. Production authorized false in config raises PermissionError
    fake_config = tmp_path / "test_config.json"
    with open(fake_config, "w", encoding="utf-8") as f:
        json.dump({"production_authorized": False, "neural": {"learning_rate": 0.001}}, f)

    with pytest.raises(PermissionError, match="production_authorized is false"):
        train_backbone_model(
            m, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=fake_config,
            execution_mode="production",
        )

    # 3. Unapproved hyperparameter override in production mode raises ValueError
    full_neural_test = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "betas": [0.9, 0.999],
        "epsilon": 1e-08,
        "effective_batch": 32,
        "max_epochs": 3,
        "min_epochs": 2,
        "early_stop_patience": 5,
        "minimum_improvement": 1e-06,
        "gradient_norm_clip": 1.0,
        "optimizer": "AdamW",
        "architectures": ["MLP"],
        "seeds": [7],
    }
    with open(fake_config, "w", encoding="utf-8") as f:
        json.dump({"production_authorized": True, "neural": full_neural_test}, f)

    with pytest.raises(ValueError, match="Unapproved hyperparameter override"):
        train_backbone_model(
            m, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=fake_config,
            execution_mode="production",
            lr=0.05,  # Unapproved override
        )


# ---------------------------------------------------------------------------
# Finding 4: Complete production configuration boundary acceptance tests
# ---------------------------------------------------------------------------

def _make_full_neural_config(tmp_path, overrides=None):
    """Build a minimal valid production config file."""
    cfg = {
        "production_authorized": True,
        "neural": {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "betas": [0.9, 0.999],
            "epsilon": 1e-08,
            "effective_batch": 32,
            "max_epochs": 2,
            "min_epochs": 2,
            "early_stop_patience": 5,
            "minimum_improvement": 1e-06,
            "gradient_norm_clip": 1.0,
            "optimizer": "AdamW",
            "architectures": ["MLP"],
            "seeds": [7],
        },
    }
    if overrides:
        cfg["neural"].update(overrides)
    p = tmp_path / "prod_config.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return p


def test_production_missing_neural_section_raises_c4(tmp_path):
    """Finding 4: Missing neural config section raises ValueError before training starts."""
    m = MLPAnnual(seed=7)
    train_x = torch.randn(16, 966)
    train_y = torch.randn(16)
    train_mkts = np.array(["US"] * 16)
    val_x = torch.randn(8, 966)
    val_y = torch.randn(8)
    val_mkts = np.array(["US"] * 8)

    p = tmp_path / "no_neural.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"production_authorized": True}, f)

    with pytest.raises((ValueError, KeyError)):
        train_backbone_model(
            m, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=p,
            execution_mode="production",
        )


def test_production_changed_patience_raises_c4(tmp_path):
    """Finding 4: Changing patience in production mode raises ValueError."""
    m = MLPAnnual(seed=7)
    train_x = torch.randn(16, 966)
    train_y = torch.randn(16)
    train_mkts = np.array(["US"] * 16)
    val_x = torch.randn(8, 966)
    val_y = torch.randn(8)
    val_mkts = np.array(["US"] * 8)

    cfg_path = _make_full_neural_config(tmp_path)

    with pytest.raises(ValueError, match="Unapproved hyperparameter override"):
        train_backbone_model(
            m, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=cfg_path,
            execution_mode="production",
            patience=99,  # Different from config's early_stop_patience=5
        )


def test_production_changed_effective_batch_raises_c4(tmp_path):
    """Finding 4: Changing effective_batch_size in production mode raises ValueError."""
    m = MLPAnnual(seed=7)
    train_x = torch.randn(16, 966)
    train_y = torch.randn(16)
    train_mkts = np.array(["US"] * 16)
    val_x = torch.randn(8, 966)
    val_y = torch.randn(8)
    val_mkts = np.array(["US"] * 8)

    cfg_path = _make_full_neural_config(tmp_path)  # effective_batch=32

    with pytest.raises(ValueError, match="Unapproved hyperparameter override"):
        train_backbone_model(
            m, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=cfg_path,
            execution_mode="production",
            effective_batch_size=512,  # Different from config's effective_batch=32
        )


def test_checkpoint_config_hash_mismatch_raises_c4(tmp_path):
    """Finding 4: Mismatched checkpoint config hash raises ValueError on resume."""
    m = MLPAnnual(seed=7)
    train_x = torch.randn(32, 966)
    train_y = torch.randn(32)
    train_mkts = np.array(["US"] * 32)
    val_x = torch.randn(8, 966)
    val_y = torch.randn(8)
    val_mkts = np.array(["US"] * 8)

    cfg_path = _make_full_neural_config(tmp_path)

    # 1. Train for 2 epochs and save a checkpoint (matches config's min_epochs=2, max_epochs=2)
    m1 = MLPAnnual(seed=7)
    train_backbone_model(
        m1, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
        config_path=cfg_path,
        execution_mode="production",
        checkpoint_dir=tmp_path / "chk",
        min_epochs=2, max_epochs=2,
    )

    # 2. Modify the config (change learning_rate) — this changes config_hash
    cfg_changed_path = _make_full_neural_config(tmp_path / "v2", overrides={"learning_rate": 0.002})

    # 3. Attempt resume with the changed config → must raise ValueError
    m2 = MLPAnnual(seed=7)
    with pytest.raises(ValueError, match="config hash mismatch"):
        train_backbone_model(
            m2, train_x, train_y, train_mkts, val_x, val_y, val_mkts,
            config_path=cfg_changed_path,
            execution_mode="production",
            resume_from_checkpoint=tmp_path / "chk",
            min_epochs=2, max_epochs=2,
        )

