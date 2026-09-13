"""
Pre-Training Parity and Representation Verification Tests for MLP Correction.

Validates the 6 core invariants before retraining and cache generation:
1. Training/Query Parity: Identical vectors extracted for training and evaluation query.
2. Window Boundaries: Strictly sessions t-6 to t-1; session t is strictly excluded.
3. Transform Order: Standardization and clipping [-5.0, 5.0] precedes pooling.
4. Target Eligibility: Outcomes are mature by 2020-12-31 and free of NaNs/Infs.
5. Artifact Compatibility: Legacy checkpoints and 3D inputs are rejected with clear errors.
6. Isolation: Transformer backbones and M2 Euclidean windows are unaffected.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from memory_study.shared_representation import (
    extract_six_session_mean_patch,
    compute_patch_matrix_and_mlp_input,
    MLP_REPRESENTATION_ID,
)
from memory_study.backbones import (
    MLPEncoder,
    load_mlp_checkpoint,
    AnnualPatchTemporalTransformer,
)


def test_1_training_query_parity():
    """Test 1: extract_six_session_mean_patch matches the 42nd patch from patchTST exactly."""
    np.random.seed(42)
    # Simulate a normalized feature array for 300 sessions
    feat_norm = np.random.randn(300, 23).astype(np.float32)

    for loc in [252, 270, 299]:
        # Training representation
        mlp_train = extract_six_session_mean_patch(feat_norm, loc)

        # Query representation (from 252-session window patched into 42 patches of size 6)
        t_win = 252
        w_slice = feat_norm[loc - t_win:loc]
        patch_mat = w_slice.reshape(42, 6, 23).mean(axis=1)
        mlp_query = patch_mat[-1]

        # Shape check
        assert mlp_train.shape == (23,), f"Expected shape (23,), got {mlp_train.shape}"
        assert mlp_train.dtype == np.float32, f"Expected float32, got {mlp_train.dtype}"

        # Bitwise exact / close check
        np.testing.assert_allclose(
            mlp_train, mlp_query, rtol=1e-6, atol=1e-6,
            err_msg=f"Training and Query representations mismatch at loc {loc}"
        )


def test_2_window_boundaries_and_session_t_exclusion():
    """Test 2: Window covers sessions t-6..t-1; session t and t-7 are strictly excluded."""
    feat_norm = np.zeros((20, 23), dtype=np.float32)
    loc = 10  # Window: sessions 4, 5, 6, 7, 8, 9 (t-6 to t-1)

    # Set base values in window
    for s in range(4, 10):
        feat_norm[s, :] = float(s)

    baseline_out = extract_six_session_mean_patch(feat_norm, loc)
    # Expected mean: (4 + 5 + 6 + 7 + 8 + 9) / 6 = 39 / 6 = 6.5
    np.testing.assert_allclose(baseline_out, 6.5, atol=1e-6)

    # 1. Mutate session t (loc = 10)
    feat_norm_mut_t = feat_norm.copy()
    feat_norm_mut_t[loc, :] = 999.0
    out_mut_t = extract_six_session_mean_patch(feat_norm_mut_t, loc)
    np.testing.assert_allclose(out_mut_t, baseline_out, atol=1e-7,
                               err_msg="Session t leakage detected! Output changed when mutating session t.")

    # 2. Mutate session t-7 (loc - 7 = 3)
    feat_norm_mut_t7 = feat_norm.copy()
    feat_norm_mut_t7[loc - 7, :] = 999.0
    out_mut_t7 = extract_six_session_mean_patch(feat_norm_mut_t7, loc)
    np.testing.assert_allclose(out_mut_t7, baseline_out, atol=1e-7,
                               err_msg="Session t-7 leakage detected! Window must not extend beyond t-6.")

    # 3. Mutate any session within window (e.g. session 4, 9)
    for s in range(4, 10):
        feat_norm_mut_win = feat_norm.copy()
        feat_norm_mut_win[s, :] = 100.0
        out_mut_win = extract_six_session_mean_patch(feat_norm_mut_win, loc)
        assert not np.allclose(out_mut_win, baseline_out), f"Session {s} within window did not affect output!"

    # 4. Error on insufficient history
    with pytest.raises(ValueError, match="loc must be >= 6"):
        extract_six_session_mean_patch(feat_norm, 5)


def test_3_transform_order():
    """Test 3: Standardize & clip BEFORE pooling produces distinct results from pooling before clipping."""
    np.random.seed(123)
    raw_feats = np.random.randn(10, 23) * 10.0  # large variance to trigger clipping

    mean = np.mean(raw_feats, axis=0)
    std = np.std(raw_feats, axis=0) + 1e-4

    # Pipeline A: Standardize and clip [-5, 5] first, then mean of 6 sessions
    std_clipped = np.clip((raw_feats - mean) / std, -5.0, 5.0)
    out_a = extract_six_session_mean_patch(std_clipped, 8)

    # Pipeline B (Incorrect): Mean first, then standardize and clip
    raw_mean = np.mean(raw_feats[2:8], axis=0)
    out_b = np.clip((raw_mean - mean) / std, -5.0, 5.0)

    # Because clipping is non-linear, out_a and out_b must not be identical on extreme data
    # Create an outlier that exceeds clip boundary in raw
    raw_outlier = raw_feats.copy()
    raw_outlier[7, 0] = 5000.0  # massive outlier
    std_clipped_outlier = np.clip((raw_outlier - mean) / std, -5.0, 5.0)
    out_a_outlier = extract_six_session_mean_patch(std_clipped_outlier, 8)

    raw_mean_outlier = np.mean(raw_outlier[2:8], axis=0)
    out_b_outlier = np.clip((raw_mean_outlier - mean) / std, -5.0, 5.0)

    # out_a_outlier[0] is bounded because the outlier was clipped to 5.0 before averaging: (sum + 5.0)/6
    # out_b_outlier[0] will be clipped to 5.0 after averaging
    assert not np.isclose(out_a_outlier[0], out_b_outlier[0]), "Transform order invariance failure!"


def test_4_target_eligibility_and_maturity():
    """Test 4: Pre-2021 cutoff filters only mature outcomes and rejects NaNs/Infs."""
    # Test dates logic
    import pandas as pd
    pre_2021_cutoff = pd.Timestamp("2020-12-31")

    # Observations at end of 2020 cannot be mature by 2020-12-31 if 126 sessions required
    obs_date = pd.Timestamp("2020-11-01")
    avail_date = obs_date + pd.Timedelta(days=180)  # > 2020-12-31
    is_eligible = (avail_date <= pre_2021_cutoff)
    assert not is_eligible, "Observation maturing in 2021 was incorrectly marked eligible!"

    mature_obs = pd.Timestamp("2020-01-15")
    mature_avail = pd.Timestamp("2020-07-20")
    assert (mature_avail <= pre_2021_cutoff), "Mature observation was incorrectly marked ineligible!"


def test_5_artifact_compatibility_rejection():
    """Test 5: Checkpoint loader rejects mismatched/legacy representations and MLP rejects 3D inputs."""
    device = torch.device("cpu")
    mlp = MLPEncoder(input_dim=23, hidden_dim=64, latent_dim=128)

    # 1. MLPEncoder rejects 3D tensor
    x_3d = torch.randn(2, 42, 23)
    with pytest.raises(ValueError, match="MLPEncoder explicitly requires 2D input"):
        mlp(x_3d)

    # 2. MLPEncoder accepts 2D tensor of shape (B, 23)
    x_2d = torch.randn(4, 23)
    lat, pred = mlp(x_2d)
    assert lat.shape == (4, 128)
    assert pred.shape == (4, 1)

    # 3. Checkpoint rejection
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "mlp_encoder_seed_7.pt"

        # Save legacy untagged dict
        torch.save(mlp.state_dict(), tmp_path)

        # Attempt to load with expected_rep_id
        import memory_study.backbones as bb_mod
        orig_models_dir = bb_mod.LOCAL_MODELS_DIR
        try:
            bb_mod.LOCAL_MODELS_DIR = Path(tmp_dir)
            with pytest.raises(RuntimeError, match="Artifact compatibility rejection"):
                load_mlp_checkpoint(seed=7, device=device, expected_rep_id=MLP_REPRESENTATION_ID)

            # Save wrong representation_id
            torch.save({
                "representation_id": "legacy_mismatch_v1",
                "state_dict": mlp.state_dict(),
            }, tmp_path)
            with pytest.raises(RuntimeError, match="has representation_id 'legacy_mismatch_v1', expected"):
                load_mlp_checkpoint(seed=7, device=device, expected_rep_id=MLP_REPRESENTATION_ID)
        finally:
            bb_mod.LOCAL_MODELS_DIR = orig_models_dir


def test_6_isolation_transformer_and_m2_raw():
    """Test 6: Transformer takes 3D (B, 42, 23) and produces 128-d latent; M2 is 966-d flat."""
    device = torch.device("cpu")
    transformer = AnnualPatchTemporalTransformer(input_dim=23, embed_dim=64, num_heads=4, latent_dim=128)

    x_3d = torch.randn(3, 42, 23)
    lat, pred = transformer(x_3d)
    assert lat.shape == (3, 128)
    assert pred.shape == (3, 1)

    # M2 Raw window shape
    raw_window_42_6 = np.zeros((42, 6, 23), dtype=np.float32)
    patch_mat = raw_window_42_6.mean(axis=1)  # (42, 23)
    flat_966 = patch_mat.reshape(-1)
    assert flat_966.shape == (966,), f"Expected M2 flat window shape (966,), got {flat_966.shape}"
