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

        # GPU candidate reduction (Correction 3)
        gpu_cands = None
        dist_matrix = None
        if use_gpu:
            gpu_cands = bank.propose_candidates_gpu(chunk_vectors, buffer_size=start_buffer_size)
        if gpu_cands is None:
            dist_matrix = bank.compute_squared_euclidean_batched(chunk_vectors, use_gpu=use_gpu)

        for q_idx in range(Q_chunk):
            query_vec = chunk_vectors[q_idx]
            query_sec = chunk_secs[q_idx]
            dist_row = dist_matrix[q_idx] if dist_matrix is not None else None

            buffer_size = min(start_buffer_size, bank.N)
            accepted_indices: List[int] = []
            accepted_distances: List[float] = []
            sec_counts: Dict[str, int] = {}
            sec_ordinals: Dict[str, List[int]] = {}

            while True:
                # 1. Candidate Proposal Stage: propose candidates from approximate distance
                if gpu_cands is not None and buffer_size == start_buffer_size:
                    candidate_pool = gpu_cands[0][q_idx][:buffer_size]
                    approx_pool_dists = gpu_cands[1][q_idx][:buffer_size]
                elif buffer_size < bank.N:
                    if dist_row is None:
                        dist_row = bank.compute_squared_euclidean_batched(query_vec[np.newaxis, :], use_gpu=False)[0]
                    candidate_pool = np.argpartition(dist_row, buffer_size)[:buffer_size]
                    approx_pool_dists = dist_row[candidate_pool]
                else:
                    candidate_pool = np.arange(bank.N)
                    approx_pool_dists = None

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
                approx_dists_for_bounds = approx_pool_dists if approx_pool_dists is not None else dist_row[candidate_pool]
                pool_discrepancies = np.abs(approx_dists_for_bounds - refined_pool_dists)
                max_pool_discrepancy = float(np.max(pool_discrepancies)) if len(pool_discrepancies) > 0 else 0.0

                D_dim = query_vec.shape[0]
                eps_32 = float(np.finfo(np.float32).eps)
                gamma_D = (D_dim * eps_32) / max(1e-12, (1.0 - D_dim * eps_32))
                q_norm_sq = float(np.dot(query_vec.astype(np.float64), query_vec.astype(np.float64)))
                theoretical_slack = 2.0 * gamma_D * (q_norm_sq + float(np.max(bank._bank_norms_sq)))
                err_bound = max(2.0 * max_pool_discrepancy, theoretical_slack, 1e-4)

                max_approx_in_pool = float(np.max(approx_dists_for_bounds))
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


def retrieve_similarity_and_knn_batch(
    bank: MemoryBank,
    query_vectors: np.ndarray,
    query_security_ids: List[str],
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
    start_buffer_size: int = 250,
    query_chunk_size: int = 256,
    use_gpu: bool = False,
) -> Tuple[List[RetrievalResult], List[RetrievalResult]]:
    """Execute unified MEM_SIM and KNN_PLAIN retrieval sharing distance proposal and float64 refinement (Phase 5).

    Guarantees:
    1. Proposal and exact float64 candidate refinement are computed once per query chunk.
    2. MEM_SIM and KNN_PLAIN neighbor sets are derived from the exact same certified candidate pool.
    3. Buffer expansion certifies both policies simultaneously.
    4. Bit-for-bit numerical and tie-breaking parity with standalone retrieve_mem_sim_batch and retrieve_knn_plain.
    """
    Q = len(query_security_ids)
    if Q == 0:
        return [], []

    q_vecs = np.asarray(query_vectors, dtype=np.float32)
    if len(q_vecs) != Q:
        raise ValueError(f"query_vectors length {len(q_vecs)} != query_security_ids length {Q}")

    if not hasattr(bank, "bank_norms_float64") or not hasattr(bank, "security_codes_int32"):
        bank.vectors_float32 = bank.vectors
        bank.vectors_float64 = np.ascontiguousarray(bank.vectors, dtype=np.float64)
        bank.targets_float64 = bank.targets_63
        bank.session_ordinals_int32 = np.ascontiguousarray(bank.session_ordinals, dtype=np.int32)
        unique_secs = sorted(list(set(bank.security_ids)))
        bank.security_id_to_code = {s: i for i, s in enumerate(unique_secs)}
        bank.security_codes_int32 = np.array([bank.security_id_to_code[s] for s in bank.security_ids], dtype=np.int32)
        bank.bank_norms_float64 = bank.precompute_bank_norms()

    sec_codes = np.array([bank.security_id_to_code.get(s, -1) for s in query_security_ids], dtype=np.int32)

    mem_results: List[RetrievalResult] = []
    knn_results: List[RetrievalResult] = []

    for chunk_start in range(0, Q, query_chunk_size):
        chunk_end = min(chunk_start + query_chunk_size, Q)
        chunk_vectors = q_vecs[chunk_start:chunk_end]
        chunk_secs = query_security_ids[chunk_start:chunk_end]
        chunk_codes = sec_codes[chunk_start:chunk_end]
        Q_chunk = chunk_end - chunk_start

        # GPU candidate reduction (Correction 3)
        gpu_cands = None
        dist_matrix = None
        if use_gpu:
            gpu_cands = bank.propose_candidates_gpu(chunk_vectors, buffer_size=start_buffer_size)
        if gpu_cands is None:
            dist_matrix = bank.compute_squared_euclidean_batched(chunk_vectors, use_gpu=use_gpu)

        for q_idx in range(Q_chunk):
            query_vec = chunk_vectors[q_idx]
            query_sec = chunk_secs[q_idx]
            query_code = chunk_codes[q_idx]
            dist_row = dist_matrix[q_idx] if dist_matrix is not None else None

            buffer_size = min(start_buffer_size, bank.N)
            accepted_indices_mem: List[int] = []
            accepted_dists_mem: List[float] = []
            accepted_indices_knn: List[int] = []
            accepted_dists_knn: List[float] = []
            sec_counts: Dict[int, int] = {}
            sec_ordinals: Dict[int, List[int]] = {}

            while True:
                if gpu_cands is not None and buffer_size == start_buffer_size:
                    candidate_pool = gpu_cands[0][q_idx][:buffer_size]
                    approx_pool_dists = gpu_cands[1][q_idx][:buffer_size]
                elif buffer_size < bank.N:
                    if dist_row is None:
                        dist_row = bank.compute_squared_euclidean_batched(query_vec[np.newaxis, :], use_gpu=False)[0]
                    candidate_pool = np.argpartition(dist_row, buffer_size)[:buffer_size]
                    approx_pool_dists = dist_row[candidate_pool]
                else:
                    candidate_pool = np.arange(bank.N)
                    approx_pool_dists = None

                refined_pool_dists = bank.refine_candidate_distances(query_vec, candidate_pool)
                sort_order = np.lexsort((bank.record_ids[candidate_pool], refined_pool_dists))
                sorted_candidates = candidate_pool[sort_order]
                sorted_refined_dists = refined_pool_dists[sort_order]

                accepted_indices_mem.clear()
                accepted_dists_mem.clear()
                accepted_indices_knn.clear()
                accepted_dists_knn.clear()
                sec_counts.clear()
                sec_ordinals.clear()

                for idx, d_val in zip(sorted_candidates, sorted_refined_dists):
                    s_code = bank.security_codes_int32[idx]
                    if s_code == query_code:
                        continue

                    if len(accepted_indices_knn) < k:
                        accepted_indices_knn.append(idx)
                        accepted_dists_knn.append(float(d_val))

                    if len(accepted_indices_mem) < k:
                        count = sec_counts.get(s_code, 0)
                        if count < max_per_security:
                            cand_ord = int(bank.session_ordinals_int32[idx])
                            past_ords = sec_ordinals.get(s_code, [])
                            if not any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords):
                                accepted_indices_mem.append(idx)
                                accepted_dists_mem.append(float(d_val))
                                sec_counts[s_code] = count + 1
                                if s_code not in sec_ordinals:
                                    sec_ordinals[s_code] = []
                                sec_ordinals[s_code].append(cand_ord)

                    if len(accepted_indices_mem) == k and len(accepted_indices_knn) == k:
                        break

                if buffer_size >= bank.N:
                    break

                if len(accepted_indices_mem) < k or len(accepted_indices_knn) < k:
                    buffer_size = min(buffer_size * 2, bank.N)
                    continue

                approx_dists_for_bounds = approx_pool_dists if approx_pool_dists is not None else dist_row[candidate_pool]
                pool_discrepancies = np.abs(approx_dists_for_bounds - refined_pool_dists)
                max_pool_discrepancy = float(np.max(pool_discrepancies)) if len(pool_discrepancies) > 0 else 0.0

                D_dim = query_vec.shape[0]
                eps_32 = float(np.finfo(np.float32).eps)
                gamma_D = (D_dim * eps_32) / max(1e-12, (1.0 - D_dim * eps_32))
                q_norm_sq = float(np.dot(query_vec.astype(np.float64), query_vec.astype(np.float64)))
                theoretical_slack = 2.0 * gamma_D * (q_norm_sq + float(np.max(bank._bank_norms_sq)))
                err_bound = max(2.0 * max_pool_discrepancy, theoretical_slack, 1e-4)

                max_approx_in_pool = float(np.max(approx_dists_for_bounds))
                max_accepted_mem = accepted_dists_mem[-1] if accepted_dists_mem else float("inf")
                max_accepted_knn = accepted_dists_knn[-1] if accepted_dists_knn else float("inf")

                # Separate completeness certification for each policy (Correction 6)
                # MEM_SIM: excludes query security, enforces cap <= 3 and spacing >= 21
                # KNN_PLAIN: excludes query security only (no cap, no spacing restrictions)
                mem_certified = (len(accepted_indices_mem) == k) and (max_approx_in_pool - err_bound > max_accepted_mem)
                knn_certified = (len(accepted_indices_knn) == k) and (max_approx_in_pool - err_bound > max_accepted_knn)

                if not (mem_certified and knn_certified):
                    new_buffer_size = min(buffer_size * 2, bank.N)
                    if new_buffer_size == buffer_size or new_buffer_size >= bank.N:
                        full_ref_dists = bank.refine_candidate_distances(query_vec, np.arange(bank.N))
                        sort_order_full = np.lexsort((bank.record_ids, full_ref_dists))
                        sorted_candidates = np.arange(bank.N)[sort_order_full]
                        sorted_refined_dists = full_ref_dists[sort_order_full]
                        accepted_indices_mem.clear()
                        accepted_dists_mem.clear()
                        accepted_indices_knn.clear()
                        accepted_dists_knn.clear()
                        sec_counts.clear()
                        sec_ordinals.clear()
                        for idx, d_val in zip(sorted_candidates, sorted_refined_dists):
                            s_code = bank.security_codes_int32[idx]
                            if s_code == query_code:
                                continue
                            if len(accepted_indices_knn) < k:
                                accepted_indices_knn.append(idx)
                                accepted_dists_knn.append(float(d_val))
                            if len(accepted_indices_mem) < k:
                                count = sec_counts.get(s_code, 0)
                                if count < max_per_security:
                                    cand_ord = int(bank.session_ordinals_int32[idx])
                                    past_ords = sec_ordinals.get(s_code, [])
                                    if not any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords):
                                        accepted_indices_mem.append(idx)
                                        accepted_dists_mem.append(float(d_val))
                                        sec_counts[s_code] = count + 1
                                        if s_code not in sec_ordinals:
                                            sec_ordinals[s_code] = []
                                        sec_ordinals[s_code].append(cand_ord)
                            if len(accepted_indices_mem) == k and len(accepted_indices_knn) == k:
                                break
                        break
                    else:
                        buffer_size = new_buffer_size
                        continue

                break

            if len(accepted_indices_mem) < k:
                mem_results.append(RetrievalResult(
                    prediction=bank.unconditional_mean,
                    neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices_mem],
                    neighbor_distances=[],
                    is_fallback=True,
                    policy="MEM_SIM",
                ))
            else:
                targets_mem = bank.targets_63[accepted_indices_mem]
                mem_results.append(RetrievalResult(
                    prediction=float(np.mean(targets_mem)),
                    neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices_mem],
                    neighbor_distances=accepted_dists_mem,
                    is_fallback=False,
                    policy="MEM_SIM",
                ))

            if len(accepted_indices_knn) < k:
                knn_results.append(RetrievalResult(
                    prediction=bank.unconditional_mean,
                    neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices_knn],
                    neighbor_distances=[],
                    is_fallback=True,
                    policy="KNN_PLAIN",
                ))
            else:
                targets_knn = bank.targets_63[accepted_indices_knn]
                knn_results.append(RetrievalResult(
                    prediction=float(np.mean(targets_knn)),
                    neighbor_ids=[int(bank.record_ids[i]) for i in accepted_indices_knn],
                    neighbor_distances=accepted_dists_knn,
                    is_fallback=False,
                    policy="KNN_PLAIN",
                ))

    return mem_results, knn_results


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


_WORKER_BANK_N: int = 0
_WORKER_BANK_SEC_CODES: Optional[np.ndarray] = None
_WORKER_BANK_ORDINALS: Optional[np.ndarray] = None
_WORKER_BANK_TARGETS: Optional[np.ndarray] = None
_WORKER_BANK_RECORD_IDS: Optional[np.ndarray] = None
_WORKER_BANK_UNCOND_MEAN: float = 0.0


def _init_random_memory_worker(
    memmap_dir: str,
    N: int,
    unconditional_mean: float,
) -> None:
    """Pool initializer: workers open read-only .npy memmaps (Correction 4)."""
    global _WORKER_BANK_N, _WORKER_BANK_SEC_CODES, _WORKER_BANK_ORDINALS
    global _WORKER_BANK_TARGETS, _WORKER_BANK_RECORD_IDS, _WORKER_BANK_UNCOND_MEAN
    import os
    from pathlib import Path
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    p = Path(memmap_dir)
    _WORKER_BANK_N = N
    _WORKER_BANK_UNCOND_MEAN = unconditional_mean
    _WORKER_BANK_SEC_CODES = np.load(p / "sec_codes.npy", mmap_mode="r")
    _WORKER_BANK_ORDINALS = np.load(p / "ordinals.npy", mmap_mode="r")
    _WORKER_BANK_TARGETS = np.load(p / "targets.npy", mmap_mode="r")
    _WORKER_BANK_RECORD_IDS = np.load(p / "record_ids.npy", mmap_mode="r")


def _random_memory_worker_chunk(
    items: List[Tuple[int, str, int]],  # (q_idx, query_id, query_sec_code)
    bank_hash: str,
    fold_year: int,
    master_seed: int,
    k: int,
    max_per_security: int,
    min_spacing_sessions: int,
) -> List[Tuple[int, RetrievalResult]]:
    """Worker task processing a query chunk for MEM_RANDOM using read-only memmaps (Corrections 4 & 7)."""
    global _WORKER_BANK_N, _WORKER_BANK_SEC_CODES, _WORKER_BANK_ORDINALS
    global _WORKER_BANK_TARGETS, _WORKER_BANK_RECORD_IDS, _WORKER_BANK_UNCOND_MEAN

    results: List[Tuple[int, RetrievalResult]] = []
    unique_sec_codes = set(item[2] for item in items)
    # Strictly preserve ascending 0..N-1 index ordering (Correction 7)
    eligible_by_code = {
        code: np.where(_WORKER_BANK_SEC_CODES != code)[0]
        for code in unique_sec_codes
    }

    for q_idx, q_id, query_sec_code in items:
        eligible_indices = eligible_by_code[query_sec_code]
        if len(eligible_indices) == 0:
            results.append((q_idx, RetrievalResult(
                prediction=_WORKER_BANK_UNCOND_MEAN,
                neighbor_ids=[],
                neighbor_distances=[],
                is_fallback=True,
                policy="MEM_RANDOM",
            )))
            continue

        seed_val = derive_random_seed(bank_hash, fold_year, q_id, master_seed)
        rng = np.random.default_rng(np.random.PCG64(seed_val))
        permuted_indices = rng.permutation(eligible_indices)

        accepted_indices: List[int] = []
        sec_counts: Dict[int, int] = {}
        sec_ordinals: Dict[int, List[int]] = {}

        for idx in permuted_indices:
            s_code = int(_WORKER_BANK_SEC_CODES[idx])
            count = sec_counts.get(s_code, 0)
            if count >= max_per_security:
                continue

            cand_ord = int(_WORKER_BANK_ORDINALS[idx])
            past_ords = sec_ordinals.get(s_code, [])
            if any(abs(cand_ord - p_ord) < min_spacing_sessions for p_ord in past_ords):
                continue

            accepted_indices.append(idx)
            sec_counts[s_code] = count + 1
            if s_code not in sec_ordinals:
                sec_ordinals[s_code] = []
            sec_ordinals[s_code].append(cand_ord)

            if len(accepted_indices) == k:
                break

        if len(accepted_indices) < k:
            results.append((q_idx, RetrievalResult(
                prediction=_WORKER_BANK_UNCOND_MEAN,
                neighbor_ids=[int(_WORKER_BANK_RECORD_IDS[i]) for i in accepted_indices],
                neighbor_distances=[],
                is_fallback=True,
                policy="MEM_RANDOM",
            )))
        else:
            targets = _WORKER_BANK_TARGETS[accepted_indices]
            results.append((q_idx, RetrievalResult(
                prediction=float(np.mean(targets)),
                neighbor_ids=[int(_WORKER_BANK_RECORD_IDS[i]) for i in accepted_indices],
                neighbor_distances=[],
                is_fallback=False,
                policy="MEM_RANDOM",
            )))

    return results


def retrieve_mem_random_parallel_multi_seed(
    bank: MemoryBank,
    query_security_ids: List[str],
    query_ids: List[str],
    bank_hash: str,
    fold_year: int,
    seeds: List[int],
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
    max_workers: Optional[int] = None,
    chunk_size: int = 128,
) -> Dict[int, List[RetrievalResult]]:
    """Execute parallel random memory retrieval across multiple seeds reusing a single worker pool and memmap files (Finding 5)."""
    Q = len(query_ids)
    if Q == 0:
        return {s: [] for s in seeds}

    if max_workers is None:
        import os
        max_workers = min(20, os.cpu_count() or 16)

    # Fallback to serial for single worker or small Q
    if max_workers <= 1 or (Q <= chunk_size and len(seeds) == 1):
        out: Dict[int, List[RetrievalResult]] = {}
        for s in seeds:
            out[s] = [
                retrieve_mem_random(
                    bank, query_security_id=query_security_ids[i],
                    bank_hash=bank_hash, fold_year=fold_year,
                    query_id=query_ids[i], master_seed=s,
                    k=k, max_per_security=max_per_security,
                    min_spacing_sessions=min_spacing_sessions,
                )
                for i in range(Q)
            ]
        return out

    import tempfile
    from pathlib import Path
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    # Precompute integer security codes for fast comparisons
    if not hasattr(bank, "security_codes_int32") or not hasattr(bank, "session_ordinals_int32"):
        bank.precompute_bank_norms()

    sec_map = getattr(bank, "security_id_to_code", None)
    if sec_map is None:
        unique_secs = sorted(list(set(bank.security_ids)))
        sec_map = {sec: i for i, sec in enumerate(unique_secs)}
    query_sec_codes = [sec_map.get(s, -1) for s in query_security_ids]

    items = [(i, query_ids[i], query_sec_codes[i]) for i in range(Q)]
    chunks = [items[pos:pos + chunk_size] for pos in range(0, Q, chunk_size)]

    tasks = []
    for s in seeds:
        for chunk in chunks:
            tasks.append((s, chunk))

    effective_workers = min(max_workers, len(tasks))
    results_by_seed: Dict[int, List[Tuple[int, RetrievalResult]]] = {s: [] for s in seeds}

    with tempfile.TemporaryDirectory(prefix="rnd_mem_") as tmp_dir:
        tmp_p = Path(tmp_dir)
        np.save(tmp_p / "sec_codes.npy", bank.security_codes_int32)
        np.save(tmp_p / "ordinals.npy", bank.session_ordinals_int32)
        np.save(tmp_p / "targets.npy", bank.targets_float64)
        np.save(tmp_p / "record_ids.npy", bank.record_ids)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=ctx,
            initializer=_init_random_memory_worker,
            initargs=(str(tmp_p), bank.N, float(bank.unconditional_mean)),
        ) as executor:
            futures = [
                (
                    s,
                    executor.submit(
                        _random_memory_worker_chunk,
                        chunk,
                        bank_hash,
                        fold_year,
                        s,
                        k,
                        max_per_security,
                        min_spacing_sessions,
                    )
                )
                for s, chunk in tasks
            ]
            for s, f in futures:
                results_by_seed[s].extend(f.result())

    final_results: Dict[int, List[RetrievalResult]] = {}
    for s in seeds:
        res_list = results_by_seed[s]
        res_list.sort(key=lambda x: x[0])
        final_results[s] = [r[1] for r in res_list]

    return final_results


def retrieve_mem_random_parallel(
    bank: MemoryBank,
    query_security_ids: List[str],
    query_ids: List[str],
    bank_hash: str,
    fold_year: int,
    master_seed: int,
    k: int = 25,
    max_per_security: int = 3,
    min_spacing_sessions: int = 21,
    max_workers: Optional[int] = None,
    chunk_size: int = 128,
) -> List[RetrievalResult]:
    """Execute parallel random memory precedent retrieval with memmapped pool (Corrections 4 & 7)."""
    res_dict = retrieve_mem_random_parallel_multi_seed(
        bank=bank,
        query_security_ids=query_security_ids,
        query_ids=query_ids,
        bank_hash=bank_hash,
        fold_year=fold_year,
        seeds=[master_seed],
        k=k,
        max_per_security=max_per_security,
        min_spacing_sessions=min_spacing_sessions,
        max_workers=max_workers,
        chunk_size=chunk_size,
    )
    return res_dict[master_seed]


def retrieve_hist_prior(bank: MemoryBank) -> RetrievalResult:
    """Execute unconditional historical prior baseline (HIST_PRIOR)."""
    return RetrievalResult(
        prediction=bank.unconditional_mean,
        neighbor_ids=[],
        neighbor_distances=[],
        is_fallback=False,
        policy="HIST_PRIOR",
    )
