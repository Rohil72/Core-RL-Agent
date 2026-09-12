"""
Retrieval Modes Implementation for Memory-Centric Equity Selection.

High-performance GPU-accelerated and vector-indexed implementations of:
- M1: Random-priority eligible retrieval (seeds 1001, 1002, 1003, uniform weights)
- M2: Standardized input-window Euclidean distance retrieval (uniform weights)
- M3: Frozen latent cosine similarity retrieval (uniform weights)
- M4: Fixed similarity weighting on identical M3 neighbours (tau=0.08 kernel)
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import torch
import torch.nn.functional as F

from memory_study.memory_store import MemoryStore


def compute_m4_kernel_weights(
    similarities: np.ndarray,
    tau: float = 0.08,
) -> np.ndarray:
    """
    M4: Softmax similarity kernel on identical neighbour IDs:
    w_i = exp((s_i - s_max) / tau) / sum_j exp((s_j - s_max) / tau).
    """
    if len(similarities) == 0:
        return np.array([], dtype=np.float32)
    s_max = np.max(similarities)
    exps = np.exp((similarities - s_max) / tau)
    return (exps / (np.sum(exps) + 1e-12)).astype(np.float32)


def batch_retrieve_m1(
    query_tickers: np.ndarray,
    mem_tickers: np.ndarray,
    mem_sessions: np.ndarray,
    mem_returns: np.ndarray,
    retrieval_seeds: List[int] = [1001, 1002, 1003],
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast M1: Random-priority eligible retrieval across retrieval seeds.
    Averages predictions across seeds. Returns (m1_preds, coverages).
    """
    n_queries = len(query_tickers)
    n_mem = len(mem_tickers)
    m1_preds = np.zeros(n_queries, dtype=np.float32)
    coverages = np.ones(n_queries, dtype=np.float32)

    perms = [np.random.default_rng(s).permutation(n_mem) for s in retrieval_seeds]

    for q_i, q_tkr in enumerate(query_tickers):
        seed_preds = []
        for perm in perms:
            accepted = []
            ticker_counts = {}
            ticker_sessions = {}

            for idx in perm:
                t = mem_tickers[idx]
                if t == q_tkr:
                    continue
                if ticker_counts.get(t, 0) >= max_per_ticker:
                    continue
                sess = mem_sessions[idx]
                past_s = ticker_sessions.get(t, [])
                if any(abs(sess - s_prev) < min_separation for s_prev in past_s):
                    continue

                accepted.append(idx)
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
                if t not in ticker_sessions:
                    ticker_sessions[t] = []
                ticker_sessions[t].append(sess)

                if len(accepted) == k:
                    break

            if len(accepted) == k:
                seed_preds.append(float(np.mean(mem_returns[accepted])))

        if seed_preds:
            m1_preds[q_i] = float(np.mean(seed_preds))
            coverages[q_i] = 1.0
        else:
            m1_preds[q_i] = 0.0
            coverages[q_i] = 0.0

    return m1_preds, coverages


def batch_retrieve_m3_m4(
    query_lats: np.ndarray,          # shape (N_Q, 128)
    mem_lats_gpu: torch.Tensor,       # shape (N_M, 128) on GPU
    query_tickers: np.ndarray,
    mem_tickers: np.ndarray,
    mem_sessions: np.ndarray,
    mem_returns: np.ndarray,
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
    tau: float = 0.08,
    batch_size: int = 1000,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated chunked cosine similarity retrieval for M3 and M4.
    Reuses IDENTICAL neighbour IDs for M4 with softmax kernel weighting.
    Returns (m3_preds, m4_preds, coverages).
    """
    n_queries = len(query_lats)
    m3_preds = np.zeros(n_queries, dtype=np.float32)
    m4_preds = np.zeros(n_queries, dtype=np.float32)
    coverages = np.zeros(n_queries, dtype=np.float32)

    q_tensor = torch.tensor(query_lats, dtype=torch.float32)

    for b_start in range(0, n_queries, batch_size):
        b_end = min(b_start + batch_size, n_queries)
        q_chunk = q_tensor[b_start:b_end].to(device)

        with torch.no_grad():
            dots = torch.matmul(q_chunk, mem_lats_gpu.T)  # (B, N_M)
            topk_scores, topk_idx = torch.topk(dots, 250, dim=1, largest=True)
            topk_idx_cpu = topk_idx.cpu().numpy()
            topk_scores_cpu = topk_scores.cpu().numpy()

        for local_i in range(b_end - b_start):
            q_i = b_start + local_i
            q_tkr = query_tickers[q_i]
            cands = topk_idx_cpu[local_i]
            c_scores = topk_scores_cpu[local_i]

            accepted = []
            acc_scores = []
            ticker_counts = {}
            ticker_sessions = {}

            for idx, sc in zip(cands, c_scores):
                t = mem_tickers[idx]
                if t == q_tkr:
                    continue
                if ticker_counts.get(t, 0) >= max_per_ticker:
                    continue
                sess = mem_sessions[idx]
                past_s = ticker_sessions.get(t, [])
                if any(abs(sess - s_prev) < min_separation for s_prev in past_s):
                    continue

                accepted.append(idx)
                acc_scores.append(sc)
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
                if t not in ticker_sessions:
                    ticker_sessions[t] = []
                ticker_sessions[t].append(sess)

                if len(accepted) == k:
                    break

            if len(accepted) == k:
                nbr_rets = mem_returns[accepted]
                # M3: uniform weights
                m3_preds[q_i] = float(np.mean(nbr_rets))

                # M4: kernel weights on IDENTICAL neighbours
                w4 = compute_m4_kernel_weights(np.array(acc_scores, dtype=np.float32), tau=tau)
                m4_preds[q_i] = float(np.sum(w4 * nbr_rets))
                coverages[q_i] = 1.0
            else:
                m3_preds[q_i] = 0.0
                m4_preds[q_i] = 0.0
                coverages[q_i] = 0.0

    return m3_preds, m4_preds, coverages


def batch_retrieve_m2(
    query_windows: np.ndarray,        # shape (N_Q, 966)
    mem_windows_gpu: torch.Tensor,    # shape (N_M, 966) on GPU
    mem_sq_norms_gpu: torch.Tensor,   # shape (N_M,) on GPU
    query_tickers: np.ndarray,
    mem_tickers: np.ndarray,
    mem_sessions: np.ndarray,
    mem_returns: np.ndarray,
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
    batch_size: int = 500,
    device: torch.device = torch.device("cuda"),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated chunked Euclidean distance retrieval for M2.
    Uses expanded identity ||q - m||^2 = ||q||^2 + ||m||^2 - 2 q.m.
    Returns (m2_preds, coverages).
    """
    n_queries = len(query_windows)
    m2_preds = np.zeros(n_queries, dtype=np.float32)
    coverages = np.zeros(n_queries, dtype=np.float32)

    q_tensor = torch.tensor(query_windows, dtype=torch.float32)

    for b_start in range(0, n_queries, batch_size):
        b_end = min(b_start + batch_size, n_queries)
        q_chunk = q_tensor[b_start:b_end].to(device)
        q_sq = torch.sum(q_chunk ** 2, dim=1, keepdim=True)

        with torch.no_grad():
            dots = torch.matmul(q_chunk, mem_windows_gpu.T)  # (B, N_M)
            dists_sq = q_sq + mem_sq_norms_gpu.unsqueeze(0) - 2.0 * dots
            topk_scores, topk_idx = torch.topk(dists_sq, 250, dim=1, largest=False)
            topk_idx_cpu = topk_idx.cpu().numpy()

        for local_i in range(b_end - b_start):
            q_i = b_start + local_i
            q_tkr = query_tickers[q_i]
            cands = topk_idx_cpu[local_i]

            accepted = []
            ticker_counts = {}
            ticker_sessions = {}

            for idx in cands:
                t = mem_tickers[idx]
                if t == q_tkr:
                    continue
                if ticker_counts.get(t, 0) >= max_per_ticker:
                    continue
                sess = mem_sessions[idx]
                past_s = ticker_sessions.get(t, [])
                if any(abs(sess - s_prev) < min_separation for s_prev in past_s):
                    continue

                accepted.append(idx)
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
                if t not in ticker_sessions:
                    ticker_sessions[t] = []
                ticker_sessions[t].append(sess)

                if len(accepted) == k:
                    break

            if len(accepted) == k:
                m2_preds[q_i] = float(np.mean(mem_returns[accepted]))
                coverages[q_i] = 1.0
            else:
                m2_preds[q_i] = 0.0
                coverages[q_i] = 0.0

    return m2_preds, coverages


# Legacy single-query wrappers for unit testing
def retrieve_m1_random(
    eligible_indices: np.ndarray,
    memory_store: MemoryStore,
    retrieval_seed: int,
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(eligible_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), {"valid": False, "fallback_reason": "zero_eligible"}
    rng = np.random.default_rng(retrieval_seed)
    random_priorities = rng.uniform(0.0, 1.0, size=len(eligible_indices))
    accepted_nbrs, stats = memory_store.apply_acceptance_constraints(
        candidate_indices=eligible_indices,
        candidate_scores=random_priorities,
        k=k,
        max_per_ticker=max_per_ticker,
        min_separation=min_separation,
        is_distance=False,
    )
    if not stats["valid"]:
        return accepted_nbrs, np.array([], dtype=np.float32), stats
    weights = np.full(len(accepted_nbrs), 1.0 / len(accepted_nbrs), dtype=np.float32)
    return accepted_nbrs, weights, stats


def retrieve_m2_raw_window(
    query_window: np.ndarray,
    candidate_windows: np.ndarray,
    eligible_indices: np.ndarray,
    memory_store: MemoryStore,
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(eligible_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), {"valid": False, "fallback_reason": "zero_eligible"}
    cand_win = candidate_windows[eligible_indices]
    diff = cand_win - query_window.reshape(1, -1)
    dists = np.linalg.norm(diff, axis=1)
    accepted_nbrs, stats = memory_store.apply_acceptance_constraints(
        candidate_indices=eligible_indices,
        candidate_scores=dists,
        k=k,
        max_per_ticker=max_per_ticker,
        min_separation=min_separation,
        is_distance=True,
    )
    if not stats["valid"]:
        return accepted_nbrs, np.array([], dtype=np.float32), stats
    weights = np.full(len(accepted_nbrs), 1.0 / len(accepted_nbrs), dtype=np.float32)
    return accepted_nbrs, weights, stats


def retrieve_m3_latent_cosine(
    query_latent: np.ndarray,
    candidate_latents: np.ndarray,
    eligible_indices: np.ndarray,
    memory_store: MemoryStore,
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if len(eligible_indices) == 0:
        return np.array([], dtype=int), np.array([], dtype=np.float32), np.array([], dtype=np.float32), {"valid": False, "fallback_reason": "zero_eligible"}
    cand_lats = candidate_latents[eligible_indices]
    sims = np.dot(cand_lats, query_latent)
    accepted_nbrs, stats = memory_store.apply_acceptance_constraints(
        candidate_indices=eligible_indices,
        candidate_scores=sims,
        k=k,
        max_per_ticker=max_per_ticker,
        min_separation=min_separation,
        is_distance=False,
    )
    if not stats["valid"]:
        return accepted_nbrs, np.array([], dtype=np.float32), np.array([], dtype=np.float32), stats
    nbr_sims = np.dot(candidate_latents[accepted_nbrs], query_latent)
    weights = np.full(len(accepted_nbrs), 1.0 / len(accepted_nbrs), dtype=np.float32)
    return accepted_nbrs, weights, nbr_sims, stats
