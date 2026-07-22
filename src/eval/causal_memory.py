from __future__ import annotations

import glob
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def attach_outcome_availability(
    frame: pd.DataFrame,
    precomputed_glob: str,
    horizon_sessions: int = 126,
) -> pd.DataFrame:
    """
    Attach the exact timestamp at which the outcome becomes available
    for each state in the given frame, using precomputed trading calendars.
    
    A state at row i for a ticker requires the timestamp at row i + horizon_sessions
    to be considered causally mature.
    """
    out = frame.copy()
    if "outcome_available_timestamp" not in out.columns:
        out["outcome_available_timestamp"] = pd.NaT
    if "session_index" not in out.columns:
        out["session_index"] = -1

    if out.empty:
        return out

    is_timestamp_index = isinstance(out.index, pd.DatetimeIndex)
    if is_timestamp_index:
        query_ts_all = out.index
    else:
        query_ts_all = pd.to_datetime(out["timestamp"], utc=True)
    
    tickers = out["ticker"].unique()
    precomputed_files = glob.glob(precomputed_glob)
    if not precomputed_files:
        logger.warning(f"No precomputed files found matching glob: {precomputed_glob}")
        return out

    ticker_to_file = {}
    for fp in precomputed_files:
        stem = Path(fp).stem
        ticker_to_file[stem] = fp

    for ticker in tickers:
        ticker_mask = out["ticker"] == ticker
        if ticker not in ticker_to_file:
            logger.warning(f"No precomputed file found for ticker: {ticker}. Rows will fail-closed (NaT).")
            continue
            
        fp = ticker_to_file[ticker]
        try:
            ticker_df = pd.read_parquet(fp, columns=[]) 
        except Exception as e:
            logger.error(f"Failed to read {fp}: {e}")
            continue

        ticker_df.index = pd.to_datetime(ticker_df.index, utc=True)
        ticker_df = ticker_df.sort_index()
        timestamps = ticker_df.index
        
        ts_to_idx = pd.Series(range(len(timestamps)), index=timestamps)
        query_ts = query_ts_all[ticker_mask]
        
        # Avoid duplicate index issues in ts_to_idx if precomputed files have them, though they shouldn't
        ts_to_idx = ts_to_idx[~ts_to_idx.index.duplicated(keep='first')]
        
        mapped_indices = ts_to_idx.reindex(query_ts).values
        target_indices = mapped_indices + horizon_sessions
        
        out.loc[ticker_mask, "session_index"] = mapped_indices
        
        valid_mask = pd.notna(target_indices) & (target_indices < len(timestamps))
        
        available_ts = pd.Series(pd.NaT, index=out.loc[ticker_mask].index)
        
        valid_query_idx = np.where(valid_mask)[0]
        valid_target_idx = target_indices[valid_mask].astype(int)
        
        available_ts.iloc[valid_query_idx] = timestamps[valid_target_idx]
        out.loc[ticker_mask, "outcome_available_timestamp"] = available_ts

    return out
