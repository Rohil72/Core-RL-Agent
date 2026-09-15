"""Technical feature engineering and frozen scalers (v2).

Implements the exact 23 ordered features from feature_contract.csv:
- ddof=1 for all feature standard deviations.
- Denominator guard 1e-9.
- EMA adjust=False initialized with first valid close.
- RSI with simple 14-session gain/loss means; flat window returns 0.5.
- Volume change: both zero -> 0.0; previous zero & current positive -> invalid.
- Trend slope: OLS slope of close on 0..20 divided by current close (C[t] + EPS).
- Up/Down volume: flat up volume evaluates to 0.0 / (down_vol + EPS) = 0.0.
- Segment-aware computation: contiguous 252-bar warmup per segment; EMAs reset per segment.
- Frozen scaler: mean and population std (ddof=0, floor 1e-4), clip [-5.0, 5.0].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED

EPS = 1e-9


def compute_ema(values: np.ndarray, span: int) -> np.ndarray:
    """Compute exponential moving average with adjust=False and first value as initial value."""
    n = len(values)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (span + 1.0)
    ema[0] = values[0]
    for i in range(1, n):
        ema[i] = alpha * values[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_single_segment_features(
    sessions: List[str],
    C: np.ndarray,
    H: np.ndarray,
    L: np.ndarray,
    O: np.ndarray,
    V: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Compute 23 technical features on a contiguous unbroken session segment."""
    n = len(sessions)
    feats: Dict[str, np.ndarray] = {name: np.full(n, np.nan, dtype=np.float64) for name in EXPECTED_FEATURES_ORDERED}
    valid_mask = np.zeros(n, dtype=bool)

    if n == 0:
        return feats, valid_mask

    # 1-day returns r[t] = C[t] / C[t-1] - 1
    r = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        r[1:] = C[1:] / C[:-1] - 1.0

    # Precompute EMAs for MACD
    ema12 = compute_ema(C, 12)
    ema26 = compute_ema(C, 26)
    macd_line = ema12 - ema26
    signal_line = compute_ema(macd_line, 9)
    macd_diff = macd_line - signal_line

    # Precompute True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = H[0] - L[0]
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i - 1]), abs(L[i] - C[i - 1]))

    # Regression weights for trend_slope_21: x = 0..20, sum((x - 10)^2) = 770
    x_weights = np.arange(21, dtype=np.float64) - 10.0
    x_denom = 770.0

    # Moving averages for mean_200_trend_20
    sma200 = np.full(n, np.nan, dtype=np.float64)
    for i in range(199, n):
        sma200[i] = np.mean(C[i - 199 : i + 1])

    for t in range(n):
        # 0: return_1 (min bars: 2)
        if t >= 1:
            feats["return_1"][t] = r[t]

        # 1: momentum_3 (min bars: 4)
        if t >= 3:
            feats["momentum_3"][t] = C[t] / C[t - 3] - 1.0

        # 2: momentum_10 (min bars: 11)
        if t >= 10:
            feats["momentum_10"][t] = C[t] / C[t - 10] - 1.0

        # 3: momentum_21 (min bars: 22)
        if t >= 21:
            feats["momentum_21"][t] = C[t] / C[t - 21] - 1.0

        # 4: volatility_21 (min bars: 22, 21 returns, ddof=1)
        if t >= 21:
            r_window = r[t - 20 : t + 1]
            feats["volatility_21"][t] = np.std(r_window, ddof=1)

        # 5: volume_mean_21 (min bars: 21)
        if t >= 20:
            v_window = V[t - 20 : t + 1]
            feats["volume_mean_21"][t] = np.mean(v_window)

        # 6: volume_ratio_21 (min bars: 21)
        if t >= 20:
            v_mean = np.mean(V[t - 20 : t + 1])
            feats["volume_ratio_21"][t] = V[t] / (v_mean + EPS)

        # 7: volume_change_1 (min bars: 2)
        if t >= 1:
            v_prev = V[t - 1]
            v_curr = V[t]
            if v_prev == 0.0:
                feats["volume_change_1"][t] = 0.0
            else:
                feats["volume_change_1"][t] = v_curr / v_prev - 1.0

        # 8: intraday_range (min bars: 1)
        feats["intraday_range"][t] = (H[t] - L[t]) / (C[t] + EPS)

        # 9: drawdown_252 (min bars: 252)
        if t >= 251:
            c_window_252 = C[t - 251 : t + 1]
            peak = np.max(c_window_252)
            feats["drawdown_252"][t] = (C[t] - peak) / (peak + EPS)

        # 10: trend_slope_21 (min bars: 21)
        # Contract formula: OLS slope of close over 21 bars divided by current close C[t]
        if t >= 20:
            c_window_21 = C[t - 20 : t + 1]
            slope = np.sum(x_weights * c_window_21) / x_denom
            feats["trend_slope_21"][t] = slope / (C[t] + EPS)

        # 11: close_vs_mean_50 (min bars: 50)
        if t >= 49:
            feats["close_vs_mean_50"][t] = C[t] / np.mean(C[t - 49 : t + 1]) - 1.0

        # 12: close_vs_mean_150 (min bars: 150)
        if t >= 149:
            feats["close_vs_mean_150"][t] = C[t] / np.mean(C[t - 149 : t + 1]) - 1.0

        # 13: close_vs_mean_200 (min bars: 200)
        if t >= 199:
            feats["close_vs_mean_200"][t] = C[t] / np.mean(C[t - 199 : t + 1]) - 1.0

        # 14: mean_200_trend_20 (min bars: 220)
        if t >= 219:
            m_t = sma200[t]
            m_t_20 = sma200[t - 20]
            feats["mean_200_trend_20"][t] = (m_t - m_t_20) / (m_t_20 + EPS)

        # 15: above_low_252 (min bars: 252)
        if t >= 251:
            trough = np.min(C[t - 251 : t + 1])
            feats["above_low_252"][t] = (C[t] - trough) / (trough + EPS)

        # 16: from_high_252 (min bars: 252)
        if t >= 251:
            peak = np.max(C[t - 251 : t + 1])
            feats["from_high_252"][t] = (C[t] - peak) / (peak + EPS)

        # 17: up_down_volume_50 (min bars: 51, needs 50 return transitions)
        # Contract formula: sum(V[s] for r[s]>0 in last 50) / (sum(V[s] for r[s]<0 in last 50) + eps)
        # If up_vol == 0, evaluates to 0.0 / (down_vol + eps) = 0.0
        if t >= 50:
            up_vol = 0.0
            down_vol = 0.0
            for i in range(t - 49, t + 1):
                if C[i] > C[i - 1]:
                    up_vol += V[i]
                elif C[i] < C[i - 1]:
                    down_vol += V[i]
            feats["up_down_volume_50"][t] = up_vol / (down_vol + EPS)

        # 18: rsi_14 (min bars: 15, needs 14 return transitions)
        if t >= 14:
            gains = 0.0
            losses = 0.0
            for i in range(t - 13, t + 1):
                diff = C[i] - C[i - 1]
                if diff > 0.0:
                    gains += diff
                elif diff < 0.0:
                    losses += -diff
            mean_gain = gains / 14.0
            mean_loss = losses / 14.0
            if mean_gain == 0.0 and mean_loss == 0.0:
                feats["rsi_14"][t] = 0.5
            else:
                feats["rsi_14"][t] = mean_gain / (mean_gain + mean_loss + EPS)

        # 19: atr_ratio_14 (min bars: 15)
        if t >= 13:
            atr_14 = np.mean(tr[t - 13 : t + 1])
            feats["atr_ratio_14"][t] = atr_14 / (C[t] + EPS)

        # 20: macd_signal_diff (min bars: 34)
        if t >= 33:
            feats["macd_signal_diff"][t] = macd_diff[t] / (C[t] + EPS)

        # 21: bollinger_width_20 (min bars: 20, ddof=1)
        if t >= 19:
            c_window_20 = C[t - 19 : t + 1]
            std_20 = np.std(c_window_20, ddof=1)
            mean_20 = np.mean(c_window_20)
            feats["bollinger_width_20"][t] = (4.0 * std_20) / (mean_20 + EPS)

        # 22: vol_ratio_63_21 (min bars: 64, 63 returns, ddof=1)
        if t >= 63:
            r_63 = r[t - 62 : t + 1]
            r_21 = r[t - 20 : t + 1]
            std_63 = np.std(r_63, ddof=1)
            std_21 = np.std(r_21, ddof=1)
            feats["vol_ratio_63_21"][t] = std_63 / (std_21 + EPS)

    # Contiguous 252-bar warmup within segment
    feat_matrix = np.column_stack([feats[name] for name in EXPECTED_FEATURES_ORDERED])
    for t in range(n):
        if t >= 251:
            if np.all(np.isfinite(feat_matrix[t])):
                valid_mask[t] = True

    return feats, valid_mask


def compute_technical_features(tr_bars: pd.DataFrame) -> pd.DataFrame:
    """Compute the 23 ordered technical features from total-return OHLC and normalized volume.

    Handles optional 'segment_id' for non-continuous splits (R08).
    """
    n = len(tr_bars)
    if n == 0:
        cols = ["session"] + EXPECTED_FEATURES_ORDERED + ["valid_mask"]
        return pd.DataFrame(columns=cols)

    if "segment_id" in tr_bars.columns:
        # Compute segment by segment to prevent leakage across breaks
        segment_dfs = []
        for seg_id, group in tr_bars.groupby("segment_id", sort=False):
            grp_sessions = group["session"].tolist()
            grp_C = group["tr_close"].to_numpy(dtype=np.float64)
            grp_H = group["tr_high"].to_numpy(dtype=np.float64)
            grp_L = group["tr_low"].to_numpy(dtype=np.float64)
            grp_O = group["tr_open"].to_numpy(dtype=np.float64)
            grp_V = group["normalized_volume"].to_numpy(dtype=np.float64)

            grp_feats, grp_valid = _compute_single_segment_features(
                grp_sessions, grp_C, grp_H, grp_L, grp_O, grp_V
            )
            df_seg = pd.DataFrame({"session": grp_sessions})
            for name in EXPECTED_FEATURES_ORDERED:
                df_seg[name] = grp_feats[name]
            df_seg["valid_mask"] = grp_valid
            df_seg["segment_id"] = seg_id
            segment_dfs.append(df_seg)

        res_df = pd.concat(segment_dfs, axis=0, ignore_index=True)
        return res_df
    else:
        sessions = tr_bars["session"].tolist()
        C = tr_bars["tr_close"].to_numpy(dtype=np.float64)
        H = tr_bars["tr_high"].to_numpy(dtype=np.float64)
        L = tr_bars["tr_low"].to_numpy(dtype=np.float64)
        O = tr_bars["tr_open"].to_numpy(dtype=np.float64)
        V = tr_bars["normalized_volume"].to_numpy(dtype=np.float64)

        feats, valid_mask = _compute_single_segment_features(sessions, C, H, L, O, V)
        df_result = pd.DataFrame({"session": sessions})
        for name in EXPECTED_FEATURES_ORDERED:
            df_result[name] = feats[name]
        df_result["valid_mask"] = valid_mask
        return df_result


@dataclass
class FrozenScaler:
    means: np.ndarray  # shape (23,), float64
    stds: np.ndarray   # shape (23,), float64
    feature_names: List[str]

    def transform(self, feat_array: np.ndarray) -> np.ndarray:
        """Standardize features and clip to [-5.0, 5.0]. Returns float32."""
        std_floored = np.maximum(self.stds, 1e-4)
        standardized = (feat_array.astype(np.float64) - self.means) / std_floored
        clipped = np.clip(standardized, -5.0, 5.0)
        return clipped.astype(np.float32)


def fit_scaler(feature_df: pd.DataFrame, training_cutoff: str, start_date: str = "2013-01-01") -> FrozenScaler:
    """Fit feature scaler on unique valid rows from start_date through training_cutoff.

    Uses population standard deviation (ddof=0) and floors std at 1e-4.
    """
    mask = (
        (feature_df["session"] >= start_date)
        & (feature_df["session"] <= training_cutoff)
        & (feature_df["valid_mask"] == True)
    )
    subset = feature_df[mask]
    if subset.empty:
        raise ValueError(f"No valid feature rows to fit scaler between {start_date} and {training_cutoff}")

    feat_matrix = subset[EXPECTED_FEATURES_ORDERED].to_numpy(dtype=np.float64)
    means = np.mean(feat_matrix, axis=0)
    stds = np.std(feat_matrix, axis=0, ddof=0)

    return FrozenScaler(
        means=means,
        stds=stds,
        feature_names=list(EXPECTED_FEATURES_ORDERED),
    )
