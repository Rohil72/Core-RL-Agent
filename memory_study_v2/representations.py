"""Annual window extraction and temporal representation (v2).

Guarantees:
1. Strict session t exclusion: uses exactly rows t-252 through t-1.
2. Order of operations: Standardization and clipping [-5.0, 5.0] precedes patch pooling.
3. 42 patches of 6 sessions -> (42, 23) matrix for Transformer.
4. Row-major flattened vector of 966 elements for MLP, Ridge, and Retrieval.
5. Absolute parity: training, bank, validation, and evaluation query share the identical function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED
from memory_study_v2.features import FrozenScaler


class RepresentationError(Exception):
    """Raised when an origin does not have 252 contiguous valid sessions or has invalid shapes."""
    pass


@dataclass
class AnnualRepresentation:
    session_t: str  # The decision origin session (excluded from inputs)
    transformer_matrix: np.ndarray  # shape (42, 23), float32
    flattened_vector: np.ndarray    # shape (966,), float32


def extract_annual_representation(
    feature_df: pd.DataFrame,
    session_idx_t: int,
    scaler: FrozenScaler,
) -> AnnualRepresentation:
    """Extract annual representation at decision session t.

    Uses strictly feature rows t-252 through t-1 (session t is excluded).
    Standardizes and clips to [-5.0, 5.0] before pooling into 42 patches of 6 sessions.
    """
    if session_idx_t < 252:
        raise RepresentationError(
            f"Insufficient history for session index {session_idx_t}: requires at least 252 preceding rows."
        )

    # Slice strictly rows t-252 through t-1 (inclusive of t-1, exclusive of t)
    start_idx = session_idx_t - 252
    end_idx = session_idx_t  # slice [start_idx : end_idx] has length 252
    window_df = feature_df.iloc[start_idx:end_idx]

    if len(window_df) != 252:
        raise RepresentationError(f"Window length expected 252, got {len(window_df)}")

    # Check validity mask: all 252 rows must be valid
    if "valid_mask" in window_df.columns:
        if not np.all(window_df["valid_mask"].to_numpy()):
            raise RepresentationError(
                f"Window contains invalid feature rows between index {start_idx} and {end_idx - 1}."
            )

    # Extract 23 raw features
    raw_252 = window_df[EXPECTED_FEATURES_ORDERED].to_numpy(dtype=np.float64)

    # Standardize and clip to [-5.0, 5.0] BEFORE pooling
    clipped_252 = scaler.transform(raw_252)  # shape (252, 23), float32

    # Group into 42 patches of 6 sessions: reshape (42, 6, 23)
    reshaped = clipped_252.reshape(42, 6, 23)

    # Mean across 6 sessions in each patch -> (42, 23)
    patch_matrix = np.mean(reshaped, axis=1).astype(np.float32)

    # Flatten row-major -> (966,)
    flattened = patch_matrix.reshape(966)

    session_t = feature_df.iloc[session_idx_t]["session"]

    return AnnualRepresentation(
        session_t=session_t,
        transformer_matrix=patch_matrix,
        flattened_vector=flattened,
    )
