"""Memory bank storage, exact float64 reference, and accelerated candidate proposal (v2).

Acceptance criteria addressed:
- A17: Optimized selected IDs equal float64 reference on constrained, tied, and buffer-expansion cases.
- Bank records: valid origins >= 2013-01-01, 126-session maturity <= bank cutoff.
- Reference distance: CPU float64 squared Euclidean distance, stable record ID tie-breaking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class MemoryBankError(Exception):
    """Raised when memory bank invariants or acceleration error budgets are violated."""
    pass


@dataclass
class BankRecord:
    record_id: int              # Stable integer record ID (1-indexed or 0-indexed ascending)
    security_id: str            # e.g. "US:AAPL"
    session_origin: str         # YYYY-MM-DD
    session_126_maturity: str   # Availability timestamp of 126th subsequent bar
    vector: np.ndarray          # shape (966,), float32
    target_63: float            # mature 63-session total return


class MemoryBank:
    """Sealed memory bank for a walk-forward fold."""

    def __init__(self, records: List[BankRecord]):
        if not records:
            raise MemoryBankError("Cannot initialize empty MemoryBank")

        # Sort by record_id ascending to guarantee stable ordering
        self.records = sorted(records, key=lambda r: r.record_id)
        self.N = len(self.records)
        self.record_ids = np.array([r.record_id for r in self.records], dtype=np.int64)
        self.security_ids = [r.security_id for r in self.records]
        self.sessions = [r.session_origin for r in self.records]
        self.targets = np.array([r.target_63 for r in self.records], dtype=np.float64)

        # Matrix of stored vectors: shape (N, 966), float32
        self.vectors_f32 = np.vstack([r.vector for r in self.records]).astype(np.float32)
        # Precomputed squared norms in float64 for accelerated proposal
        self.vectors_f64 = self.vectors_f32.astype(np.float64)
        self.bank_norms_sq_f64 = np.sum(self.vectors_f64 ** 2, axis=1)

        # Unconditional bank mean
        self.unconditional_mean = float(np.mean(self.targets))

    def compute_reference_distances(self, query_f32: np.ndarray, candidate_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute exact CPU float64 squared Euclidean distance.

        Subtracts stored float32 promoted to float64, squares elementwise, and sums with float64 reduction.
        """
        q64 = query_f32.astype(np.float64)
        if candidate_indices is not None:
            sub_bank = self.vectors_f64[candidate_indices]
        else:
            sub_bank = self.vectors_f64

        diff = sub_bank - q64[None, :]
        dists = np.sum(diff ** 2, axis=1)
        return dists

    def propose_candidates(
        self,
        query_f32: np.ndarray,
        buffer_size: int,
        query_chunk_size: int = 128,
        bank_chunk_size: int = 16384,
    ) -> np.ndarray:
        """Propose top candidates using chunked float64 matrix products with error verification (A17).

        Formula: ||q||^2 + ||b||^2 - 2 * q.dot(b)
        Error budget: <= 1e-6 absolute. Boundary margin: 2e-6.
        """
        q64 = query_f32.astype(np.float64)
        q_norm_sq = np.sum(q64 ** 2)

        B = min(buffer_size, self.N)
        approx_dists = np.zeros(self.N, dtype=np.float64)

        # Chunked dot product
        for i in range(0, self.N, bank_chunk_size):
            end_i = min(i + bank_chunk_size, self.N)
            chunk_b = self.vectors_f64[i:end_i]
            chunk_norms = self.bank_norms_sq_f64[i:end_i]

            dots = np.dot(chunk_b, q64)
            chunk_dists = q_norm_sq + chunk_norms - 2.0 * dots

            # Check error bound: negative values below -1e-6 fail
            min_val = np.min(chunk_dists)
            if min_val < -1e-6:
                raise MemoryBankError(
                    f"Candidate proposal distance {min_val} violated negative error budget (-1e-6)."
                )
            # Clamp tiny negatives to 0.0 for candidate proposal only
            chunk_dists = np.maximum(chunk_dists, 0.0)
            approx_dists[i:end_i] = chunk_dists

        # Find approximate B-th distance threshold
        # Use partition to find top B candidates efficiently
        partitioned_indices = np.argpartition(approx_dists, B - 1)[:B]
        boundary_dist = np.max(approx_dists[partitioned_indices])

        # Include ALL records whose approximate distance <= boundary_dist + 2e-6
        margin_threshold = boundary_dist + 2e-6
        candidate_indices = np.where(approx_dists <= margin_threshold)[0]

        # Recompute distances for candidates using exact CPU float64 reference
        ref_dists = self.compute_reference_distances(query_f32, candidate_indices)

        # Verify error budget: absolute difference <= 1e-6
        approx_subset = approx_dists[candidate_indices]
        max_error = np.max(np.abs(ref_dists - approx_subset))
        if max_error > 1e-6:
            raise MemoryBankError(
                f"Candidate proposal error {max_error} exceeded absolute budget 1e-6."
            )

        # Sort candidate indices by (ref_distance, record_id)
        candidate_record_ids = self.record_ids[candidate_indices]
        # Lexicographical sort: primary ref_dists, secondary candidate_record_ids
        sort_order = np.lexsort((candidate_record_ids, ref_dists))
        sorted_candidates = candidate_indices[sort_order]

        return sorted_candidates
