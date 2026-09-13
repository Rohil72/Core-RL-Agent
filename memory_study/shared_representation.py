"""
Shared Representation Module for Memory-Centric Equity Selection.

Enforces exact mathematical parity for MLP representations:
x^{MLP}_{i,t} = (1/6) sum_{a=1}^{6} tilde{x}_{i,t-a} in R^{23}

Where:
- tilde{x} are features standardized and clipped to [-5.0, 5.0] before pooling.
- The window strictly covers sessions t-6, ..., t-1 (session t is excluded).
- The resulting vector has shape (23,) and float32 dtype.
"""

import numpy as np
from typing import Tuple

MLP_REPRESENTATION_ID = "mlp_v2_patch6mean_2026"


def extract_six_session_mean_patch(feat_norm: np.ndarray, loc: int) -> np.ndarray:
    """
    Extracts the six-session mean patch ending at session loc - 1.
    Strictly covers sessions [loc - 6 : loc] = {loc-6, loc-5, loc-4, loc-3, loc-2, loc-1}.
    Session loc is strictly excluded.

    Parameters:
        feat_norm: Array of shape (N, 23) of standardized and clipped features.
        loc: Integer index representing decision session t.

    Returns:
        1D float32 array of shape (23,).
    """
    if loc < 6:
        raise ValueError(f"loc must be >= 6 to extract 6-session history, got {loc}")
    if feat_norm.ndim != 2 or feat_norm.shape[1] != 23:
        raise ValueError(f"feat_norm must be 2D with 23 columns, got {feat_norm.shape}")

    slice_6 = feat_norm[loc - 6:loc]  # exactly 6 rows: loc-6 .. loc-1
    return np.mean(slice_6, axis=0).astype(np.float32)


def compute_patch_matrix_and_mlp_input(
    feat_norm: np.ndarray,
    loc: int,
    t_win: int = 252,
    n_patches: int = 42,
    patch_size: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes both the 42-patch temporal matrix (for Transformer / M2 retrieval)
    and the final 6-session mean patch (for MLP).

    Enforces mathematical identity between patch_mat[-1] and extract_six_session_mean_patch.
    """
    if loc < t_win:
        raise ValueError(f"loc must be >= t_win ({t_win}), got {loc}")
    if t_win != n_patches * patch_size:
        raise ValueError(f"t_win ({t_win}) must equal n_patches ({n_patches}) * patch_size ({patch_size})")

    w_slice = feat_norm[loc - t_win:loc]  # (252, 23)
    patch_mat = w_slice.reshape(n_patches, patch_size, 23).mean(axis=1).astype(np.float32)  # (42, 23)
    mlp_vec = extract_six_session_mean_patch(feat_norm, loc)  # (23,)

    # Exact parity assertion
    np.testing.assert_allclose(patch_mat[-1], mlp_vec, rtol=1e-7, atol=1e-7)
    return patch_mat, mlp_vec
