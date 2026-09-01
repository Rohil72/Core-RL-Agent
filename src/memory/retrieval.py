from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RetrievalConfig:
    """Controls temporally safe querying of the market memory database."""

    k: int = 25
    causal_horizon_sessions: int = 126
    minimum_neighbor_separation_sessions: int = 21
    same_ticker_mode: str = "allow"  # "allow", "only", "exclude"
    same_ticker_neighbor_limit: float | None = 0.50
    max_neighbors_per_ticker: int | None = None
    max_distance: float | None = None
    exclude_query_sector: bool = False
    exclude_query_industry: bool = False
    min_confidence: float | None = None
    require_outcome_availability: bool = True
    maximum_memory_age_days: int | None = None


@dataclass(frozen=True)
class RetrievalIndex:
    """Static memory metadata parsed once for many causal queries."""

    available_ns: np.ndarray
    tickers: np.ndarray
    finite_latents: np.ndarray
    sectors: np.ndarray | None
    industries: np.ndarray | None
    confidence: np.ndarray | None
    sessions: np.ndarray | None
    timestamps_ns: np.ndarray


def build_retrieval_index(
    memory: pd.DataFrame,
    memory_x: np.ndarray,
    config: RetrievalConfig,
) -> RetrievalIndex:
    """Precompute immutable filters used by every query against one memory."""
    if "outcome_available_timestamp" in memory.columns:
        availability_source = memory["outcome_available_timestamp"]
    elif config.require_outcome_availability:
        raise ValueError("Memory frame must contain 'outcome_available_timestamp'.")
    else:
        availability_source = memory["timestamp"]

    availability = pd.to_datetime(availability_source, utc=True, errors="coerce")
    # Parquet may preserve timestamps at microsecond precision while Timestamp.value
    # is nanoseconds. Convert explicitly before integer comparison to avoid lookahead.
    available_ns = availability.array.as_unit("ns").asi8.copy()
    available_ns[availability.isna().to_numpy()] = np.iinfo(np.int64).max
    return RetrievalIndex(
        available_ns=available_ns,
        tickers=memory["ticker"].astype(str).to_numpy(),
        finite_latents=np.isfinite(memory_x).all(axis=1),
        sectors=memory["sector"].astype(str).to_numpy() if "sector" in memory else None,
        industries=memory["industry"].astype(str).to_numpy() if "industry" in memory else None,
        confidence=(
            memory["experience_confidence"].fillna(0.0).to_numpy(dtype=float)
            if "experience_confidence" in memory
            else None
        ),
        sessions=memory["session_index"].to_numpy() if "session_index" in memory else None,
        timestamps_ns=(
            pd.to_datetime(memory["timestamp"], utc=True, errors="coerce")
            .array.as_unit("ns")
            .asi8.copy()
        ),
    )


def normalise_latents(memory_x: np.ndarray, query_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardise latent matrices using memory statistics only."""
    mu = np.nanmean(memory_x, axis=0)
    sigma = np.nanstd(memory_x, axis=0)
    sigma = np.where((sigma <= 1e-12) | ~np.isfinite(sigma), 1.0, sigma)
    return (memory_x - mu) / sigma, (query_x - mu) / sigma


def retrieve_neighbors(
    memory: pd.DataFrame,
    memory_x: np.ndarray,
    query_vector: np.ndarray,
    query_timestamp: pd.Timestamp,
    query_ticker: str,
    config: RetrievalConfig,
    query_sector: str | None = None,
    query_industry: str | None = None,
    distance_vector: np.ndarray | None = None,
    retrieval_index: RetrievalIndex | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve legal historical neighbors for one query state."""
    if config.k <= 0:
        raise ValueError("k must be positive.")
    
    index = retrieval_index or build_retrieval_index(memory, memory_x, config)
    if len(index.available_ns) != len(memory):
        raise ValueError("retrieval_index does not match the memory frame.")
    query_ts_val = pd.Timestamp(query_timestamp).value
    legal = (
        (index.available_ns <= query_ts_val)
        & (index.timestamps_ns != np.iinfo(np.int64).min)
        & index.finite_latents
        & np.isfinite(query_vector).all()
    )
    if config.maximum_memory_age_days is not None:
        if config.maximum_memory_age_days <= 0:
            raise ValueError("maximum_memory_age_days must be positive.")
        maximum_age_ns = int(config.maximum_memory_age_days) * 86_400 * 1_000_000_000
        legal &= index.timestamps_ns >= query_ts_val - maximum_age_ns
    tickers = index.tickers
    
    if config.same_ticker_mode == "exclude":
        legal &= tickers != str(query_ticker)
    elif config.same_ticker_mode == "only":
        legal &= tickers == str(query_ticker)
        
    if config.exclude_query_sector:
        if query_sector is None or index.sectors is None:
            # Fail closed when metadata needed for requested isolation mode is absent
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        legal &= index.sectors != str(query_sector)
        
    if config.exclude_query_industry:
        if query_industry is None or index.industries is None:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        legal &= index.industries != str(query_industry)
        
    if index.confidence is not None and config.min_confidence is not None:
        legal &= index.confidence >= float(config.min_confidence)
        
    indices = np.flatnonzero(legal)
    if indices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
        
    all_distances = (
        np.asarray(distance_vector, dtype=float)
        if distance_vector is not None
        else np.linalg.norm(memory_x - query_vector, axis=1)
    )
    if all_distances.shape != (len(memory),):
        raise ValueError("distance_vector must contain one distance per memory row.")
    distances = all_distances[indices]
    if config.max_distance is not None:
        keep = distances <= float(config.max_distance)
        indices = indices[keep]
        distances = distances[keep]
        
    if indices.size == 0:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
        
    capped = _progressive_filtered_neighbors(
        indices,
        distances,
        tickers,
        index.sessions,
        query_ticker,
        config,
    )
    capped_distances = all_distances[capped]
    return capped, capped_distances


def _progressive_filtered_neighbors(
    indices: np.ndarray,
    distances: np.ndarray,
    tickers: np.ndarray,
    sessions: np.ndarray | None,
    query_ticker: str,
    config: RetrievalConfig,
) -> np.ndarray:
    """Find the exact filtered top-k without fully sorting every legal row."""
    count = len(indices)
    pool_size = min(count, max(config.k * 4, 64))
    while True:
        if pool_size >= count:
            candidate_positions = np.arange(count)
        else:
            partition = np.argpartition(distances, pool_size - 1)[:pool_size]
            boundary = float(np.max(distances[partition]))
            candidate_positions = np.flatnonzero(distances <= boundary)
        order = np.lexsort((indices[candidate_positions], distances[candidate_positions]))
        ordered = indices[candidate_positions[order]]
        if config.minimum_neighbor_separation_sessions > 0 and sessions is not None:
            ordered = deduplicate_close_neighbors(
                ordered,
                tickers,
                sessions,
                config.minimum_neighbor_separation_sessions,
            )
        if config.max_neighbors_per_ticker is not None:
            ordered = cap_ticker_concentration(
                ordered,
                tickers,
                config.max_neighbors_per_ticker,
            )
        capped = cap_same_ticker_neighbors(
            ordered,
            tickers,
            query_ticker,
            config.k,
            config.same_ticker_neighbor_limit,
        )
        if len(capped) >= config.k or pool_size >= count:
            return capped
        pool_size = min(count, pool_size * 2)


def cap_ticker_concentration(
    ordered_indices: np.ndarray,
    tickers: np.ndarray,
    maximum_per_ticker: int,
) -> np.ndarray:
    """Cap every ticker's contribution while preserving distance order."""

    if maximum_per_ticker <= 0:
        raise ValueError("maximum_per_ticker must be positive.")
    counts: dict[str, int] = {}
    kept: list[int] = []
    for idx in ordered_indices:
        ticker = str(tickers[idx])
        if counts.get(ticker, 0) >= maximum_per_ticker:
            continue
        kept.append(int(idx))
        counts[ticker] = counts.get(ticker, 0) + 1
    return np.asarray(kept, dtype=int)

def deduplicate_close_neighbors(
    ordered_indices: np.ndarray,
    tickers: np.ndarray,
    session_indices: np.ndarray,
    min_separation: int,
) -> np.ndarray:
    """
    When two selected neighbours have the same ticker and are fewer than 
    min_separation ticker trading rows apart, retain only the closer one.
    """
    kept: list[int] = []
    # keep track of selected session indices per ticker
    selected_sessions_by_ticker: dict[str, list[int]] = {}
    
    for idx in ordered_indices:
        ticker = str(tickers[idx])
        session = session_indices[idx]
        
        # If session is unknown (-1 or NaN), we could fail-closed or just allow it.
        # Assuming they are valid integers from causal_memory.py
        if pd.isna(session) or session < 0:
            kept.append(int(idx))
            continue
            
        sessions = selected_sessions_by_ticker.setdefault(ticker, [])
        
        # Check if there's any existing selected neighbor within min_separation
        conflict = False
        for s in sessions:
            if abs(s - session) < min_separation:
                conflict = True
                break
                
        if not conflict:
            kept.append(int(idx))
            sessions.append(session)
            
    return np.asarray(kept, dtype=int)


def cap_same_ticker_neighbors(
    ordered_indices: np.ndarray,
    tickers: np.ndarray,
    query_ticker: str,
    k: int,
    same_ticker_limit: float | None,
) -> np.ndarray:
    """Apply same-ticker cap while preserving distance order."""
    if same_ticker_limit is None:
        return ordered_indices[:k]
    max_same = int(np.floor(k * same_ticker_limit))
    kept: list[int] = []
    same_count = 0
    for idx in ordered_indices:
        is_same = str(tickers[idx]) == str(query_ticker)
        if is_same and same_count >= max_same:
            continue
        kept.append(int(idx))
        same_count += int(is_same)
        if len(kept) == k:
            break
    return np.asarray(kept, dtype=int)
