"""Vectorized MarketPanel for portfolio simulation (Phase 4).

Prebuilds contiguous 2D NumPy arrays (T x S) for fast session stepping,
eliminating Pandas DataFrame .loc, .iloc, and iterrows scans in inner loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


@dataclass
class MarketPanel:
    """Aligned 2D market panel for a specific market and session sequence."""
    market: str
    sessions: np.ndarray       # 1D shape (T,), dtype '<U10'
    securities: np.ndarray     # 1D shape (S,), dtype object/str
    raw_open: np.ndarray       # 2D shape (T, S), float64
    raw_close: np.ndarray      # 2D shape (T, S), float64
    tradable: np.ndarray       # 2D shape (T, S), bool
    volatility_21: np.ndarray  # 2D shape (T, S), float64
    atr_ratio_14: np.ndarray   # 2D shape (T, S), float64
    sec_to_idx: Dict[str, int]
    sess_to_idx: Dict[str, int]

    @classmethod
    def build_from_sec_info(
        cls,
        market: str,
        sec_info: Dict[str, Any],
        sessions: List[str],
    ) -> MarketPanel:
        """Build MarketPanel from sec_info dictionaries and target sessions."""
        market_secs = sorted([s for s, s_data in sec_info.items() if s_data.get("market", "US") == market])
        T = len(sessions)
        S = len(market_secs)

        sec_to_idx = {s: i for i, s in enumerate(market_secs)}
        sess_to_idx = {s: i for i, s in enumerate(sessions)}

        raw_open = np.full((T, S), np.nan, dtype=np.float64)
        raw_close = np.full((T, S), np.nan, dtype=np.float64)
        tradable = np.zeros((T, S), dtype=bool)
        volatility_21 = np.zeros((T, S), dtype=np.float64)
        atr_ratio_14 = np.zeros((T, S), dtype=np.float64)

        for s_idx, s in enumerate(market_secs):
            s_data = sec_info[s]
            val_df = s_data.get("val_df")
            tr_df = s_data.get("tr_df")
            feats_df = s_data.get("feats_df")

            # Extract numpy arrays from DataFrames if available
            tr_sessions = tr_df["session"].tolist() if tr_df is not None and "session" in tr_df.columns else []
            tr_sess_map = {str(sess): i for i, sess in enumerate(tr_sessions)}
            tr_open_arr = tr_df["raw_open"].to_numpy(dtype=np.float64) if tr_df is not None else None
            tr_close_arr = tr_df["raw_close"].to_numpy(dtype=np.float64) if tr_df is not None else None

            val_sessions = val_df["session"].tolist() if val_df is not None and "session" in val_df.columns else []
            val_sess_map = {str(sess): i for i, sess in enumerate(val_sessions)}
            val_status_arr = val_df["bar_status"].to_numpy(dtype=str) if val_df is not None and "bar_status" in val_df.columns else None
            val_valid_arr = val_df["is_valid_bar"].to_numpy(dtype=bool) if val_df is not None and "is_valid_bar" in val_df.columns else None

            feat_sessions = feats_df["session"].tolist() if feats_df is not None and "session" in feats_df.columns else []
            feat_sess_map = {str(sess): i for i, sess in enumerate(feat_sessions)}
            feat_atr_arr = feats_df["atr_ratio_14"].to_numpy(dtype=np.float64) if feats_df is not None and "atr_ratio_14" in feats_df.columns else None
            feat_vol_arr = feats_df["volatility_21"].to_numpy(dtype=np.float64) if feats_df is not None and "volatility_21" in feats_df.columns else None

            for t_idx, t in enumerate(sessions):
                tr_row_idx = tr_sess_map.get(t)
                if tr_row_idx is not None and tr_open_arr is not None and tr_close_arr is not None:
                    raw_o = tr_open_arr[tr_row_idx]
                    raw_c = tr_close_arr[tr_row_idx]
                    raw_open[t_idx, s_idx] = raw_o
                    raw_close[t_idx, s_idx] = raw_c

                    is_valid = True
                    val_row_idx = val_sess_map.get(t)
                    if val_row_idx is not None:
                        if val_status_arr is not None:
                            is_valid = (val_status_arr[val_row_idx] == "VALID")
                        elif val_valid_arr is not None:
                            is_valid = bool(val_valid_arr[val_row_idx])

                    tradable[t_idx, s_idx] = bool(is_valid and raw_o > 0 and np.isfinite(raw_o))

                feat_row_idx = feat_sess_map.get(t)
                if feat_row_idx is not None and feat_atr_arr is not None and feat_vol_arr is not None:
                    atr_ratio_14[t_idx, s_idx] = feat_atr_arr[feat_row_idx]
                    volatility_21[t_idx, s_idx] = feat_vol_arr[feat_row_idx]

        return cls(
            market=market,
            sessions=np.array(sessions, dtype='<U10'),
            securities=np.array(market_secs, dtype=object),
            raw_open=raw_open,
            raw_close=raw_close,
            tradable=tradable,
            volatility_21=volatility_21,
            atr_ratio_14=atr_ratio_14,
            sec_to_idx=sec_to_idx,
            sess_to_idx=sess_to_idx,
        )
