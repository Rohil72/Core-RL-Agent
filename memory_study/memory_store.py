"""
Memory Store for Memory-Centric Equity Selection.

Manages indexed historical precedents and enforces shared retrieval constraints:
- Temporal availability: precedent available_timestamp <= query decision_timestamp
- Same-ticker entity exclusion: precedent ticker != query ticker
- Same-peer separation: |session_i - session_j| >= 21
- Ticker diversity cap: <= 3 precedents per ticker
- Secondary deterministic tie-breaking
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import pandas as pd


class MemoryStore:
    """
    In-memory representation of the sealed historical precedent bank.
    Provides fast filtering and constraint-satisfying candidate selection.
    """
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.meta_df = pd.read_parquet(cache_dir / "memory_meta.parquet")
        self.n_records = len(self.meta_df)

        self.tickers = self.meta_df["ticker"].values
        self.markets = self.meta_df["market"].values
        self.sessions = self.meta_df["ticker_session_index"].values.astype(int)
        self.dates = self.meta_df["origin_timestamp"].values
        self.available_dates = self.meta_df["available_timestamp"].values
        self.return_63 = self.meta_df["return_63"].values.astype(np.float32)
        self.return_21 = self.meta_df["return_21"].values.astype(np.float32)
        self.drawdown_63 = self.meta_df["drawdown_63"].values.astype(np.float32)

    def get_eligible_indices(
        self,
        query_ticker: str,
        decision_date: str,
        same_ticker_exclusion: bool = True,
    ) -> np.ndarray:
        """
        Returns array indices of precedents satisfying temporal availability
        and entity exclusion.
        """
        mask = self.available_dates <= decision_date
        if same_ticker_exclusion:
            mask = mask & (self.tickers != query_ticker)
        return np.where(mask)[0]

    def apply_acceptance_constraints(
        self,
        candidate_indices: np.ndarray,
        candidate_scores: np.ndarray,
        k: int = 25,
        max_per_ticker: int = 3,
        min_separation: int = 21,
        is_distance: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Scans ranked candidates and enforces:
        1. max_per_ticker cap (e.g. <= 3)
        2. min_separation between entries from the same ticker (e.g. >= 21 sessions)
        3. Deterministic secondary ordering

        Returns (accepted_indices, audit_stats).
        """
        n_cands = len(candidate_indices)
        if n_cands == 0:
            return np.array([], dtype=int), {
                "accepted_count": 0,
                "valid": False,
                "fallback_reason": "zero_eligible",
            }

        # Fast pre-filtering: take top 200 candidates with argpartition before multi-key sorting
        pool_size = min(n_cands, max(200, k * 5))
        if n_cands > pool_size:
            if is_distance:
                part = np.argpartition(candidate_scores, pool_size)[:pool_size]
            else:
                part = np.argpartition(-candidate_scores, pool_size)[:pool_size]
            candidate_indices = candidate_indices[part]
            candidate_scores = candidate_scores[part]

        cand_tickers = self.tickers[candidate_indices]
        cand_sessions = self.sessions[candidate_indices]

        # Multi-key sorting: primary score, secondary session, tertiary index
        if is_distance:
            order = np.lexsort((cand_sessions, candidate_indices, candidate_scores))
        else:
            order = np.lexsort((cand_sessions, candidate_indices, -candidate_scores))

        sorted_indices = candidate_indices[order]

        accepted = []
        ticker_counts: Dict[str, int] = {}
        ticker_sessions: Dict[str, List[int]] = {}

        for idx in sorted_indices:
            tkr = self.tickers[idx]
            sess = self.sessions[idx]

            # Check max_per_ticker
            if ticker_counts.get(tkr, 0) >= max_per_ticker:
                continue

            # Check min_separation
            past_sessions = ticker_sessions.get(tkr, [])
            if any(abs(sess - s_prev) < min_separation for s_prev in past_sessions):
                continue

            # Accept
            accepted.append(idx)
            ticker_counts[tkr] = ticker_counts.get(tkr, 0) + 1
            if tkr not in ticker_sessions:
                ticker_sessions[tkr] = []
            ticker_sessions[tkr].append(sess)

            if len(accepted) == k:
                break

        accepted_arr = np.array(accepted, dtype=int)
        valid = len(accepted_arr) >= k
        fallback_reason = None if valid else f"insufficient_records_{len(accepted_arr)}_of_{k}"

        return accepted_arr, {
            "accepted_count": len(accepted_arr),
            "valid": valid,
            "fallback_reason": fallback_reason,
            "unique_tickers": len(ticker_counts),
        }
