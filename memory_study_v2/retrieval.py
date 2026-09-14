"""Constrained precedent retrieval and control mechanisms (v2).

Acceptance criteria addressed:
- A17: Candidate proposal parity and tie-breaking by record ID ascending.
- A18: Query security exclusion, cap <= 3 per security, spacing >= 21 sessions, fallback to unconditional mean.
- A19: KNN_PLAIN (removes cap/spacing), MEM_RANDOM (deterministic PCG64 shuffling).
- Native exchange session ordinal spacing invariant to bank row interleaving (R03).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from memory_study_v2.memory import MemoryBank


@dataclass
class RetrievalResult:
    prediction: float
    neighbor_ids: List[int]
    neighbor_distances: List[float]
    is_fallback: bool
    policy: str


def retrieve_mem_sim(
    bank: MemoryBank,
    query_vector: np.ndarray,
    query_security_id: str,
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
    start_buffer_size: int = 250,
) -> RetrievalResult:
    """Execute constrained similarity precedent retrieval (MEM_SIM).

    Constraints:
    1. Exclude exact canonical security (query_security_id).
    2. Per-security cap <= 3 precedents.
    3. Spacing >= 21 native exchange sessions within the same security (R03).
    4. Top k=25 nearest neighbors by squared Euclidean distance; ties broken by record ID ascending.
    5. Fallback to unconditional mean if fewer than k valid precedents exist.
    """
    buffer_size = min(start_buffer_size, bank.N)
    accepted_indices: List[int] = []
    sec_counts: Dict[str, int] = {}
    sec_ordinals: Dict[str, List[int]] = {}

    while True:
        candidate_indices = bank.propose_candidates(query_vector, buffer_size=buffer_size)

        accepted_indices.clear()
        sec_counts.clear()
        sec_ordinals.clear()

        for idx in candidate_indices:
            sec_id = bank.security_ids[idx]

            # 1. Exclude exact canonical security
            if sec_id == query_security_id:
                continue

            # 2. Check per-security cap
            count = sec_counts.get(sec_id, 0)
            if count >= max_per_security:
                continue

            # 3. Check spacing rule: native exchange session ordinals within this security (R03)
            cand_ord = int(bank.session_ordinals[idx])
            past_ords = sec_ordinals.get(sec_id, [])
            too_close = any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords)
            if too_close:
                continue

            # Accept candidate
            accepted_indices.append(idx)
            sec_counts[sec_id] = count + 1
            if sec_id not in sec_ordinals:
                sec_ordinals[sec_id] = []
            sec_ordinals[sec_id].append(cand_ord)

            if len(accepted_indices) == k:
                break

        if len(accepted_indices) == k or buffer_size >= bank.N:
            break

        # Expand buffer: double buffer size up to bank size
        buffer_size = min(buffer_size * 2, bank.N)

    # 4. Check if at least k neighbors were accepted
    if len(accepted_indices) < k:
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
            neighbor_distances=[],
            is_fallback=True,
            policy="MEM_SIM",
        )

    # 5. Arithmetic mean of k accepted target values
    targets = bank.targets_63[accepted_indices]
    dist_sq = bank.compute_squared_euclidean_reference(query_vector)[accepted_indices]
    pred = float(np.mean(targets))

    return RetrievalResult(
        prediction=pred,
        neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
        neighbor_distances=[float(d) for d in dist_sq],
        is_fallback=False,
        policy="MEM_SIM",
    )


def retrieve_mem_sim_batch(
    bank: MemoryBank,
    query_vectors: np.ndarray,
    query_security_ids: List[str],
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
    start_buffer_size: int = 250,
    query_chunk_size: int = 256,
    use_gpu: bool = False,
) -> List[RetrievalResult]:
    """Execute constrained similarity precedent retrieval for a batch of queries (Finding 5 / C6).

    Batches candidate distance computation using bank.compute_squared_euclidean_batched,
    while strictly preserving all exact eligibility, spacing, cap, and tie-breaking rules per query.
    Supports query chunking to bound memory and optional GPU distance proposal acceleration.
    """
    Q = len(query_security_ids)
    if Q == 0:
        return []

    q_vecs = np.asarray(query_vectors, dtype=np.float32)
    if len(q_vecs) != Q:
        raise ValueError(f"query_vectors length {len(q_vecs)} != query_security_ids length {Q}")

    # Ensure bank norms are cached
    if not hasattr(bank, "_bank_norms_sq"):
        bank.precompute_bank_norms()

    results: List[RetrievalResult] = []

    # Process queries in chunks to bound peak memory
    for chunk_start in range(0, Q, query_chunk_size):
        chunk_end = min(chunk_start + query_chunk_size, Q)
        chunk_vectors = q_vecs[chunk_start:chunk_end]
        chunk_secs = query_security_ids[chunk_start:chunk_end]
        Q_chunk = chunk_end - chunk_start

        # Precompute candidate proposal distances for the chunk
        dist_matrix = bank.compute_squared_euclidean_batched(chunk_vectors, use_gpu=use_gpu)

        for q_idx in range(Q_chunk):
            query_vec = chunk_vectors[q_idx]
            query_sec = chunk_secs[q_idx]
            dist_row = dist_matrix[q_idx]

            buffer_size = min(start_buffer_size, bank.N)
            accepted_indices: List[int] = []
            accepted_distances: List[float] = []
            sec_counts: Dict[str, int] = {}
            sec_ordinals: Dict[str, List[int]] = {}

            while True:
                # 1. Candidate Proposal Stage: propose candidates from approximate distance
                if buffer_size < bank.N:
                    # Use argpartition to efficiently propose candidate pool without full sort
                    candidate_pool = np.argpartition(dist_row, buffer_size)[:buffer_size]
                else:
                    candidate_pool = np.arange(bank.N)

                # 2. Reference Distance Refinement Step (Finding 5 / C6):
                # Recompute exact direct float64 squared differences for candidate pool
                refined_pool_dists = bank.refine_candidate_distances(query_vec, candidate_pool)

                # 3. Stable sorting of refined candidates: primary dist ascending, secondary record_id ascending
                sort_order = np.lexsort((bank.record_ids[candidate_pool], refined_pool_dists))
                sorted_candidates = candidate_pool[sort_order]
                sorted_refined_dists = refined_pool_dists[sort_order]

                accepted_indices.clear()
                accepted_distances.clear()
                sec_counts.clear()
                sec_ordinals.clear()

                for idx, d_val in zip(sorted_candidates, sorted_refined_dists):
                    sec_id = bank.security_ids[idx]

                    # 1. Exclude exact canonical security
                    if sec_id == query_sec:
                        continue

                    # 2. Check per-security cap
                    count = sec_counts.get(sec_id, 0)
                    if count >= max_per_security:
                        continue

                    # 3. Check spacing rule: native exchange session ordinals within this security (R03)
                    cand_ord = int(bank.session_ordinals[idx])
                    past_ords = sec_ordinals.get(sec_id, [])
                    too_close = any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords)
                    if too_close:
                        continue

                    # Accept candidate
                    accepted_indices.append(idx)
                    accepted_distances.append(float(d_val))
                    sec_counts[sec_id] = count + 1
                    if sec_id not in sec_ordinals:
                        sec_ordinals[sec_id] = []
                    sec_ordinals[sec_id].append(cand_ord)

                    if len(accepted_indices) == k:
                        break

                # 4. Boundary ambiguity verification and expansion:
                if buffer_size >= bank.N:
                    # Entire bank inspected and refined with float64 reference formula
                    break

                if len(accepted_indices) < k:
                    # Need more candidates to fulfill k requirements
                    buffer_size = min(buffer_size * 2, bank.N)
                    continue

                # Boundary ambiguity verification with established mathematical error bound:
                pool_discrepancies = np.abs(dist_row[candidate_pool] - refined_pool_dists)
                max_pool_discrepancy = float(np.max(pool_discrepancies)) if len(pool_discrepancies) > 0 else 0.0

                D_dim = query_vec.shape[0]
                eps_32 = float(np.finfo(np.float32).eps)
                gamma_D = (D_dim * eps_32) / max(1e-12, (1.0 - D_dim * eps_32))
                q_norm_sq = float(np.dot(query_vec.astype(np.float64), query_vec.astype(np.float64)))
                theoretical_slack = 2.0 * gamma_D * (q_norm_sq + float(np.max(bank._bank_norms_sq)))
                err_bound = max(2.0 * max_pool_discrepancy, theoretical_slack, 1e-4)

                max_approx_in_pool = float(np.max(dist_row[candidate_pool]))
                max_accepted = accepted_distances[-1]

                # If an uninspected candidate could have true distance <= max_accepted, expand or fall back:
                if max_approx_in_pool - err_bound <= max_accepted:
                    new_buffer_size = min(buffer_size * 2, bank.N)
                    if new_buffer_size == buffer_size or new_buffer_size >= bank.N:
                        # Fallback to direct reference calculation across all bank records
                        # whenever the truncated ordering cannot be certified within buffer
                        full_ref_dists = bank.refine_candidate_distances(query_vec, np.arange(bank.N))
                        sort_order_full = np.lexsort((bank.record_ids, full_ref_dists))
                        sorted_candidates = np.arange(bank.N)[sort_order_full]
                        sorted_refined_dists = full_ref_dists[sort_order_full]
                        accepted_indices.clear()
                        accepted_distances.clear()
                        sec_counts.clear()
                        sec_ordinals.clear()
                        for idx, d_val in zip(sorted_candidates, sorted_refined_dists):
                            sec_id = bank.security_ids[idx]
                            if sec_id == query_sec:
                                continue
                            count = sec_counts.get(sec_id, 0)
                            if count >= max_per_security:
                                continue
                            cand_ord = int(bank.session_ordinals[idx])
                            past_ords = sec_ordinals.get(sec_id, [])
                            if any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords):
                                continue
                            accepted_indices.append(idx)
                            accepted_distances.append(float(d_val))
                            sec_counts[sec_id] = count + 1
                            if sec_id not in sec_ordinals:
                                sec_ordinals[sec_id] = []
                            sec_ordinals[sec_id].append(cand_ord)
                            if len(accepted_indices) == k:
                                break
                        break
                    else:
                        buffer_size = new_buffer_size
                        continue

                # Certified: all uninspected candidates have true distance > max_accepted
                break

            if len(accepted_indices) < k:
                results.append(RetrievalResult(
                    prediction=bank.unconditional_mean,
                    neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
                    neighbor_distances=[],
                    is_fallback=True,
                    policy="MEM_SIM",
                ))
            else:
                targets = bank.targets_63[accepted_indices]
                pred = float(np.mean(targets))
                results.append(RetrievalResult(
                    prediction=pred,
                    neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
                    neighbor_distances=accepted_distances,
                    is_fallback=False,
                    policy="MEM_SIM",
                ))

    return results


def retrieve_knn_plain(
    bank: MemoryBank,
    query_vector: np.ndarray,
    query_security_id: str,
    k: int = 25,
) -> RetrievalResult:
    """Execute unconstrained k-NN control (KNN_PLAIN): no cap, no spacing."""
    dist_sq = bank.compute_squared_euclidean_reference(query_vector)
    sorted_indices = np.lexsort((bank.record_ids, dist_sq))

    accepted_indices: List[int] = []
    for idx in sorted_indices:
        if bank.security_ids[idx] != query_security_id:
            accepted_indices.append(idx)
        if len(accepted_indices) == k:
            break

    if len(accepted_indices) < k:
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
            neighbor_distances=[],
            is_fallback=True,
            policy="KNN_PLAIN",
        )

    targets = bank.targets_63[accepted_indices]
    dists = dist_sq[accepted_indices]
    pred = float(np.mean(targets))

    return RetrievalResult(
        prediction=pred,
        neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
        neighbor_distances=[float(d) for d in dists],
        is_fallback=False,
        policy="KNN_PLAIN",
    )


def derive_random_seed(bank_hash: str, fold_year: int, query_id: str, master_seed: int) -> int:
    """Derive deterministic PCG64 seed for MEM_RANDOM (Section 6.2)."""
    raw_str = f"{bank_hash}:{fold_year}:{query_id}:{master_seed}"
    digest = hashlib.sha256(raw_str.encode("utf-8")).hexdigest()
    # Take lower 64 bits as unsigned integer
    seed_64 = int(digest[:16], 16)
    return seed_64


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
    """Execute random precedent memory control (MEM_RANDOM).

    Constraints:
    1. Filter out query_security_id.
    2. Deterministic PCG64 shuffling of eligible records seeded by SHA-256 digest.
    3. Apply cap <= 3 per security and spacing >= 21 native sessions (R03).
    4. Top k=25 arithmetic mean; fallback to unconditional mean if fewer than k.
    """
    eligible_indices = np.array([i for i in range(bank.N) if bank.security_ids[i] != query_security_id], dtype=np.int64)

    if len(eligible_indices) == 0:
        return RetrievalResult(
            prediction=bank.unconditional_mean,
            neighbor_ids=[],
            neighbor_distances=[],
            is_fallback=True,
            policy="MEM_RANDOM",
        )

    seed_val = derive_random_seed(bank_hash, fold_year, query_id, master_seed)
    rng = np.random.default_rng(np.random.PCG64(seed_val))
    permuted_indices = rng.permutation(eligible_indices)

    accepted_indices: List[int] = []
    sec_counts: Dict[str, int] = {}
    sec_ordinals: Dict[str, List[int]] = {}

    for idx in permuted_indices:
        sec_id = bank.security_ids[idx]

        count = sec_counts.get(sec_id, 0)
        if count >= max_per_security:
            continue

        cand_ord = int(bank.session_ordinals[idx])
        past_ords = sec_ordinals.get(sec_id, [])
        too_close = any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords)
        if too_close:
            continue

        accepted_indices.append(idx)
        sec_counts[sec_id] = count + 1
        if sec_id not in sec_ordinals:
            sec_ordinals[sec_id] = []
        sec_ordinals[sec_id].append(cand_ord)

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

    targets = bank.targets_63[accepted_indices]
    pred = float(np.mean(targets))

    return RetrievalResult(
        prediction=pred,
        neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices],
        neighbor_distances=[],
        is_fallback=False,
        policy="MEM_RANDOM",
    )


def retrieve_hist_prior(bank: MemoryBank) -> RetrievalResult:
    """Execute unconditional historical prior baseline (HIST_PRIOR)."""
    return RetrievalResult(
        prediction=bank.unconditional_mean,
        neighbor_ids=[],
        neighbor_distances=[],
        is_fallback=False,
        policy="HIST_PRIOR",
    )
