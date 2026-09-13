"""Target labels and availability timestamps (v2).

Computes 63-session total return targets and 126-session memory bank maturity timestamps.
Enforces uninterrupted valid segment requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class LabelRecord:
    session_origin: str
    target_value: float       # TR_close[t+63] / TR_close[t] - 1
    session_63: str          # Outcome availability session
    session_126: Optional[str]  # 126th session for memory admission
    valid_63: bool
    valid_126: bool


def compute_target_labels(tr_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute 63-session total return targets and 126-session bank maturity dates."""
    n = len(tr_bars)
    sessions = tr_bars["session"].tolist()
    tr_close = tr_bars["tr_close"].to_numpy(dtype=np.float64)

    target_values = np.full(n, np.nan, dtype=np.float64)
    session_63_list = ["" for _ in range(n)]
    session_126_list = ["" for _ in range(n)]
    valid_63 = np.zeros(n, dtype=bool)
    valid_126 = np.zeros(n, dtype=bool)

    for t in range(n):
        if t + 63 < n:
            target_values[t] = tr_close[t + 63] / tr_close[t] - 1.0
            session_63_list[t] = sessions[t + 63]
            valid_63[t] = True
        if t + 126 < n:
            session_126_list[t] = sessions[t + 126]
            valid_126[t] = True

    df_labels = pd.DataFrame({
        "session_origin": sessions,
        "target_value": target_values,
        "session_63": session_63_list,
        "session_126": session_126_list,
        "valid_63": valid_63,
        "valid_126": valid_126,
    })
    return df_labels
