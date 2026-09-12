"""
Value Field Processing Module for Round 4: Enriched Value Fields.

Implements:
- V0: Single-field 63-session return (baseline)
- V1: Multi-horizon returns (21-session + 63-session)
- V2: Return + Path Risk (63-session return + 63-session max drawdown)
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
CACHE_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "cache"


class ValueFieldProcessor:
    """Combines multi-attribute precedent values into calibrated expected returns."""
    def __init__(self):
        self.v1_coef = np.array([0.05, 0.95], dtype=np.float32)
        self.v1_intercept = 0.0
        self.v2_coef = np.array([1.0, 0.15], dtype=np.float32)
        self.v2_intercept = 0.0
        self.fitted = False

    def fit_on_development_memory(self):
        mm = pd.read_parquet(CACHE_DIR / "memory_meta.parquet")
        dev = mm[(mm["origin_timestamp"] >= "2018-01-01") & (mm["origin_timestamp"] <= "2020-07-07")]

        X_v1 = dev[["return_21", "return_63"]].values
        X_v2 = dev[["return_63", "drawdown_63"]].values
        y = dev["return_63"].values

        reg1 = Ridge(alpha=10.0).fit(X_v1, y)
        self.v1_coef = reg1.coef_.astype(np.float32)
        self.v1_intercept = float(reg1.intercept_)

        reg2 = Ridge(alpha=10.0).fit(X_v2, y)
        self.v2_coef = reg2.coef_.astype(np.float32)
        self.v2_intercept = float(reg2.intercept_)
        self.fitted = True

    def process_v0(self, r63: np.ndarray) -> np.ndarray:
        """V0: 63-session return alone."""
        return r63

    def process_v1(self, r21: np.ndarray, r63: np.ndarray) -> np.ndarray:
        """V1: Multi-horizon return combination."""
        return self.v1_coef[0] * r21 + self.v1_coef[1] * r63 + self.v1_intercept

    def process_v2(self, r63: np.ndarray, dd63: np.ndarray) -> np.ndarray:
        """V2: Return + Maximum Adverse Excursion (Drawdown)."""
        # dd63 is negative, so positive coef adds penalty for deep negative drawdowns
        return self.v2_coef[0] * r63 + self.v2_coef[1] * dd63 + self.v2_intercept
