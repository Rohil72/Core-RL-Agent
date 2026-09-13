"""Target labels and availability timestamps (v2).

Computes 63-session total return targets and 126-session memory bank maturity timestamps.
Enforces continuous valid calendar path and segment identity requirement (R04).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


class LabelPathIntegrityError(Exception):
    """Raised when label computation encounters an unrecoverable data integrity error."""
    pass


@dataclass
class LabelRecord:
    session_origin: str
    target_value: float          # TR_close[t+63] / TR_close[t] - 1
    session_63: str             # Outcome availability session
    session_126: Optional[str]  # 126th session for memory admission
    valid_63: bool
    valid_126: bool


def compute_target_labels(
    tr_bars: pd.DataFrame,
    venue_sessions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute 63-session total return targets and 126-session bank maturity dates.

    Enforces (R04):
    1. Continuous valid calendar path: all intervening sessions must exist, have valid positive closes.
    2. Segment identity: target and origin must belong to the exact same continuous segment.
    3. If venue_sessions is provided, calendar path must exactly match scheduled venue sessions.
    4. Finite target value required.
    """
    n = len(tr_bars)
    if n == 0:
        return pd.DataFrame(columns=[
            "session_origin", "target_value", "session_63", "session_126", "valid_63", "valid_126"
        ])

    sessions = tr_bars["session"].tolist()
    tr_close = tr_bars["tr_close"].to_numpy(dtype=np.float64)

    # Check for validity flags and segments
    has_valid_col = "is_valid_bar" in tr_bars.columns
    is_valid_array = tr_bars["is_valid_bar"].to_numpy(dtype=bool) if has_valid_col else np.ones(n, dtype=bool)

    has_segment = "segment_id" in tr_bars.columns
    segments = tr_bars["segment_id"].tolist() if has_segment else [0] * n

    venue_idx_map: Optional[Dict[str, int]] = None
    if venue_sessions:
        venue_idx_map = {s: i for i, s in enumerate(venue_sessions)}

    target_values = np.full(n, np.nan, dtype=np.float64)
    session_63_list = ["" for _ in range(n)]
    session_126_list = ["" for _ in range(n)]
    valid_63 = np.zeros(n, dtype=bool)
    valid_126 = np.zeros(n, dtype=bool)

    for t in range(n):
        origin_sess = sessions[t]
        origin_close = tr_close[t]

        # Origin must be valid and positive
        if not is_valid_array[t] or not np.isfinite(origin_close) or origin_close <= 0.0:
            continue

        # 1. Evaluate 63-session forward outcome
        if t + 63 < n:
            target_sess = sessions[t + 63]
            target_close = tr_close[t + 63]

            # Check continuous segment
            same_segment_63 = (segments[t] == segments[t + 63])

            # Check all intermediate bars are valid with finite positive close
            all_intermediate_valid_63 = bool(np.all(is_valid_array[t : t + 64]))
            all_closes_positive_63 = bool(np.all(np.isfinite(tr_close[t : t + 64])) and np.all(tr_close[t : t + 64] > 0.0))

            # Check venue calendar continuity if supplied
            calendar_continuous_63 = True
            if venue_idx_map is not None:
                if origin_sess not in venue_idx_map or target_sess not in venue_idx_map:
                    calendar_continuous_63 = False
                elif venue_idx_map[target_sess] - venue_idx_map[origin_sess] != 63:
                    calendar_continuous_63 = False

            if same_segment_63 and all_intermediate_valid_63 and all_closes_positive_63 and calendar_continuous_63:
                val_63 = target_close / origin_close - 1.0
                if np.isfinite(val_63):
                    target_values[t] = val_63
                    session_63_list[t] = target_sess
                    valid_63[t] = True

        # 2. Evaluate 126-session memory bank maturity
        if t + 126 < n:
            mat_sess = sessions[t + 126]
            same_segment_126 = (segments[t] == segments[t + 126])
            all_intermediate_valid_126 = bool(np.all(is_valid_array[t : t + 127]))
            all_closes_positive_126 = bool(np.all(np.isfinite(tr_close[t : t + 127])) and np.all(tr_close[t : t + 127] > 0.0))

            calendar_continuous_126 = True
            if venue_idx_map is not None:
                if origin_sess not in venue_idx_map or mat_sess not in venue_idx_map:
                    calendar_continuous_126 = False
                elif venue_idx_map[mat_sess] - venue_idx_map[origin_sess] != 126:
                    calendar_continuous_126 = False

            if same_segment_126 and all_intermediate_valid_126 and all_closes_positive_126 and calendar_continuous_126:
                session_126_list[t] = mat_sess
                valid_126[t] = True

    df_labels = pd.DataFrame({
        "session_origin": sessions,
        "target_value": target_values,
        "session_63": session_63_list,
        "session_126": session_126_list,
        "valid_63": valid_63,
        "valid_126": valid_126,
    })
    if "security_id" in tr_bars.columns:
        df_labels["security_id"] = tr_bars["security_id"].tolist()

    return df_labels
