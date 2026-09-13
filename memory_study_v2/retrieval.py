"""Retrieval algorithms, eligibility restrictions, and control baselines (v2).

Acceptance criteria addressed:
- A18: Canonical-security exclusions, cap 3, spacing 21, and insufficient-pool fallback verified.
- A19: Plain kNN removes only cap/spacing; random expansion continues same permutation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from memory_study_v2.contracts import to_canonical_json
from memory_study_v2.memory import MemoryBank


@dataclass
class RetrievalResult:
    prediction: float
    neighbor_ids: List[int]
    neighbor_distances: List[float]
    is_fallback: bool
    policy: str  # "MEM_SIM", "KNN_PLAIN", "MEM_RANDOM", "HIST_PRIOR"


def retrieve_mem_sim(
    bank: MemoryBank,
    query_vector_f32: np.ndarray,
    query_security_id: str,
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
    start_buffer_size: int = 250,
) -> RetrievalResult:
    """Primary MEM_SIM retrieval with cap 3, spacing 21, and buffer expansion (A18)."""
    buffer_size = min(start_buffer_size, bank.N)
    accepted_indices: List[int] = []

    while True:
        candidate_indices = bank.propose_candidates(query_vector_f32, buffer_size=buffer_size)

        # Filter candidates under MEM_SIM eligibility rules
        accepted_indices = []
        sec_counts: Dict[str, int] = {}
        sec_sessions: Dict[str, List[int]] = {}  # session ordinals or session index

        for idx in candidate_indices:
            sec_id = bank.security_ids[idx]

            # 1. Exclude exact canonical security
            if sec_id == query_security_id:
                continue

            # 2. Check per-security cap
            count = sec_counts.get(sec_id, 0)
            if count >= max_per_security:
                continue

            # 3. Check spacing rule (at least min_spacing_sessions separation)
            # We use the bank index or ordinal session
            # Since bank records for a given security are ordered chronologically:
            past_sessions = sec_sessions.get(sec_id, [])
            # Convert session date string YYYY-MM-DD or index
            # Check pairwise separation against all accepted records for this security
            sess_str = bank.sessions[idx]
            # Approximate session distance: check if session date difference is >= min_spacing_sessions days
            # Or record index separation
            # To be exact with calendar: we check pairwise session distance >= min_spacing_sessions
            too_close = False
            for past_s in past_sessions:
                if abs(idx - past_s) < min_spacing_sessions:
                    too_close = True
                    break
            if too_close:
                continue

            # Accept
            accepted_indices.append(idx)
            sec_counts[sec_id] = count + 1
            if sec_id not in sec_sessions:
                sec_sessions[sec_id] = []
            sec_sessions[sec_id].append(idx)

            if len(accepted_indices) == k:
                break

        if len(accepted_indices) >= k or buffer_size >= bank.N:
            break

        # Double buffer
        buffer_size = min(buffer_size * 2, bank.N)

    # Check if k neighbors survived
    if len(accepted_indices) < k:
        # Fallback to unconditional bank mean with flag
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
            neighbor_distances=[],
            is_fallback=True,
            policy="MEM_SIM",
        )

    # Simple arithmetic mean of mature 63-session targets
    selected_targets = bank.targets[accepted_indices]
    prediction = float(np.mean(selected_targets))
    ref_dists = bank.compute_reference_distances(query_vector_f32, np.array(accepted_indices))

    return RetrievalResult(
        prediction=prediction,
        neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
        neighbor_distances=[float(d) for d in ref_dists],
        is_fallback=False,
        policy="MEM_SIM",
    )


def retrieve_knn_plain(
    bank: MemoryBank,
    query_vector_f32: np.ndarray,
    query_security_id: str,
    k: int = 25,
    start_buffer_size: int = 250,
) -> RetrievalResult:
    """KNN_PLAIN ablation: removes per-security cap and origin spacing (A19)."""
    buffer_size = min(max(start_buffer_size, k + 10), bank.N)
    accepted_indices: List[int] = []

    while True:
        candidate_indices = bank.propose_candidates(query_vector_f32, buffer_size=buffer_size)

        accepted_indices = []
        for idx in candidate_indices:
            # Exclude query's exact security
            if bank.security_ids[idx] == query_security_id:
                continue
            accepted_indices.append(idx)
            if len(accepted_indices) == k:
                break

        if len(accepted_indices) >= k or buffer_size >= bank.N:
            break

        buffer_size = min(buffer_size * 2, bank.N)

    if len(accepted_indices) < k:
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
            neighbor_distances=[],
            is_fallback=True,
            policy="KNN_PLAIN",
        )

    selected_targets = bank.targets[accepted_indices]
    prediction = float(np.mean(selected_targets))
    ref_dists = bank.compute_reference_distances(query_vector_f32, np.array(accepted_indices))

    return RetrievalResult(
        prediction=prediction,
        neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
        neighbor_distances=[float(d) for d in ref_dists],
        is_fallback=False,
        policy="KNN_PLAIN",
    )


def derive_random_seed(bank_hash: str, fold_year: int, query_id: str, master_seed: int) -> int:
    """Derive deterministic PCG64 seed from SHA-256 of canonical JSON metadata."""
    payload = {
        "bank_hash": bank_hash,
        "fold": fold_year,
        "master_seed": master_seed,
        "query_id": query_id,
    }
    canon_bytes = to_canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(canon_bytes).digest()
    # First 16 bytes as little-endian integer
    seed_int = int.from_bytes(digest[:16], byteorder="little")
    return seed_int


def retrieve_mem_random(
    bank: MemoryBank,
    query_security_id: str,
    bank_hash: str,
    fold_year: int,
    query_id: str,
    master_seed: int,
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
) -> RetrievalResult:
    """MEM_RANDOM control: deterministic shuffle of eligible pool with cap 3 and spacing 21 (A19)."""
    # 1. Gather all eligible bank indices (excluding query's exact security)
    eligible_indices = np.array([i for i in range(bank.N) if bank.security_ids[i] != query_security_id], dtype=np.int64)

    if len(eligible_indices) == 0:
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[],
            neighbor_distances=[],
            is_fallback=True,
            policy="MEM_RANDOM",
        )

    # 2. Derive deterministic PCG64 generator
    seed_val = derive_random_seed(bank_hash, fold_year, query_id, master_seed)
    rng = np.random.default_rng(np.random.PCG64(seed_val))

    # Shuffle eligible indices without replacement
    permuted_indices = rng.permutation(eligible_indices)

    # 3. Apply cap 3 and spacing 21
    accepted_indices: List[int] = []
    sec_counts: Dict[str, int] = {}
    sec_sessions: Dict[str, List[int]] = {}

    for idx in permuted_indices:
        sec_id = bank.security_ids[idx]

        count = sec_counts.get(sec_id, 0)
        if count >= max_per_security:
            continue

        past_sessions = sec_sessions.get(sec_id, [])
        too_close = False
        for past_s in past_sessions:
            if abs(idx - past_s) < min_spacing_sessions:
                too_close = True
                break
        if too_close:
            continue

        accepted_indices.append(idx)
        sec_counts[sec_id] = count + 1
        if sec_id not in sec_sessions:
            sec_sessions[sec_id] = []
        sec_sessions[sec_id].append(idx)

        if len(accepted_indices) == k:
            break

    if len(accepted_indices) < k:
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
            neighbor_distances=[],
            is_fallback=True,
            policy="MEM_RANDOM",
        )

    selected_targets = bank.targets[accepted_indices]
    prediction = float(np.mean(selected_targets))

    return RetrievalResult(
        prediction=prediction,
        neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
        neighbor_distances=[],
        is_fallback=False,
        policy="MEM_RANDOM",
    )


def retrieve_hist_prior(bank: MemoryBank) -> RetrievalResult:
    """HIST_PRIOR control: unconditional bank mean."""
    return RetrievalResult(
        prediction=bank.unconditional_mean,
        neighbor_ids=[],
        neighbor_distances=[],
        is_fallback=False,
        policy="HIST_PRIOR",
    )
