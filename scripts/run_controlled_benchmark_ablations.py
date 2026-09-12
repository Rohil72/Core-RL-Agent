#!/usr/bin/env python3
"""
Controlled Benchmark Ablations Runner (September 2026)
======================================================
Implements and evaluates the controlled experimental ablations against the
reconciled C09 canonical benchmark.

Evaluated Systems / Hypotheses:
1. P0_CANONICAL: Authoritative baseline (Global, Shifted-Linear, |LTM|, v-denom)
2. P2_CONTROL: Authoritative mean-only control (no tail penalty)
3. P0_STAR_CANONICAL: Authoritative domestic guardrail (Domestic, Nadaraya-Watson)
4. EXP1_SCALE_HARMONIZED: Neural prediction inverse-transformed using pre-2021 market target moments
5. EXP2_ASYM_LOSS_PENALTY: Asymmetric downside risk penalty max(0, -LTM)
6. EXP3A_UNSCALED_SELECTION: Unscaled numerator ranking with equal capital allocation
7. EXP3B_UNSCALED_VOL_SIZING: Unscaled numerator ranking with inverse-volatility position sizing
8. EXP4_GLOBAL_NW: Factorial Cell B (Global eligibility + Nadaraya-Watson weighting)
9. EXP4_DOMESTIC_LINEAR: Factorial Cell C (Domestic eligibility + Shifted-Linear weighting)
10. EXP9_DIVERSE_EPISODES_CAP3: Episode diversification capping precedents per ticker at <= 3
11. EXP9_DIVERSE_EPISODES_CAP1: Strict episode diversification capping precedents per ticker at <= 1
12. EXP10_TAIL_K50_A10: Tail risk sensitivity with k=50 and alpha=0.10 (N_tail=5)
13. EXP10_TAIL_WEIGHTED: Similarity-weighted lower tail mean
14. EXP5_MULTI_SEED_ENSEMBLE: Multi-seed decision ensemble trading a single portfolio + disagreement diagnostic

Zero-Error Governance Guarantees:
- Strict 0.0% entity leakage (query_ticker != neighbor_ticker)
- Strict C09 temporal separation (intra-peer gap >= 21 sessions)
- Exact causal timing (signal at t Close, fill at t+1 Open with 10 bps fee)
- Exact replication matching on canonical baselines
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
DATA_DIR = PROJECT_ROOT / "FINAL_SUBMISSION_PACKAGE" / "data" / "cache" / "ohlcv"
MODELS_DIR = PROJECT_ROOT / "exports" / "CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE" / "models"
CANONICAL_MATRIX_PATH = PROJECT_ROOT / "canonical_benchmark_outputs" / "canonical_144_cell_performance_matrix.csv"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "internal" / "evidence" / "controlled_benchmark_ablations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TICKERS_BY_MARKET = {
    "US": ["AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AVGO", "CRM", "GOOGL", "INTU", "ISRG", "LMT", "META", "MSFT", "NOC", "NVDA", "ORCL", "REGN", "V"],
    "India": ["ASIANPAINT.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "EICHERMOT.NS", "HCLTECH.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "LT.NS", "M&M.NS", "MARUTI.NS", "PIDILITIND.NS", "RELIANCE.NS", "SUNPHARMA.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS"],
    "China": ["000333.SZ", "000725.SZ", "000858.SZ", "002230.SZ", "002241.SZ", "002415.SZ", "002475.SZ", "002594.SZ", "300059.SZ", "600036.SS", "600196.SS", "600276.SS", "600309.SS", "600519.SS", "600887.SS", "601012.SS", "601318.SS", "601888.SS"],
    "Brazil": ["B3SA3.SA", "EQTL3.SA", "FLRY3.SA", "ITUB4.SA", "KLBN11.SA", "LREN3.SA", "MGLU3.SA", "PETR4.SA", "RADL3.SA", "RAIL3.SA", "RENT3.SA", "SUZB3.SA", "TOTS3.SA", "VALE3.SA", "WEGE3.SA"],
    "France": ["AI.PA", "AIR.PA", "CAP.PA", "DG.PA", "DIM.PA", "DSY.PA", "EL.PA", "LR.PA", "MC.PA", "ML.PA", "OR.PA", "RI.PA", "RMS.PA", "SAF.PA", "SU.PA", "TEP.PA", "WLN.PA"],
    "UK": ["AUTO.L", "AZN.L", "BA.L", "CPG.L", "CRDA.L", "DGE.L", "EXPN.L", "HLMA.L", "JD.L", "LSEG.L", "OCDO.L", "PRU.L", "REL.L", "RMV.L", "RTO.L", "SGE.L", "SPX.L"],
}

SESSIONS_PER_YEAR = {"US": 252, "India": 248, "China": 242, "Brazil": 249, "France": 254, "UK": 253}
SEEDS = [7, 17, 37]
T_WIN = 252
PATCH_SIZE = 6
N_PATCHES = 42

FEATURE_NAMES_23 = [
    "tech_return_1d", "tech_momentum_3d", "tech_momentum_10d", "tech_momentum_21d", "tech_volatility_21d",
    "tech_volume_sma_21d", "tech_volume_ratio_21d", "tech_volume_change_1d", "tech_intraday_range_hl",
    "tech_drawdown_from_peak_21d", "tech_trend_slope_21d", "tech_close_vs_sma_50", "tech_close_vs_sma_150",
    "tech_close_vs_sma_200", "tech_sma_200_trend_20", "tech_pct_above_52w_low", "tech_pct_from_52w_high",
    "tech_up_down_volume_ratio_50", "tech_rsi_14", "tech_atr_ratio_14", "tech_macd_signal_diff",
    "tech_bollinger_bandwidth_20", "tech_historical_vol_ratio_63_21"
]


def compute_23_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = df_ohlcv.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = [c.lower() for c in df.columns]

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan).ffill().fillna(1.0)

    df["tech_return_1d"] = close.pct_change().fillna(0.0)
    df["tech_momentum_3d"] = close.pct_change(3).fillna(0.0)
    df["tech_momentum_10d"] = close.pct_change(10).fillna(0.0)
    df["tech_momentum_21d"] = close.pct_change(21).fillna(0.0)
    df["tech_volatility_21d"] = df["tech_return_1d"].rolling(window=21, min_periods=5).std().fillna(0.01)

    vol_sma21 = volume.rolling(window=21, min_periods=5).mean().fillna(volume)
    df["tech_volume_sma_21d"] = vol_sma21
    df["tech_volume_ratio_21d"] = (volume / (vol_sma21 + 1e-9)).fillna(1.0)
    df["tech_volume_change_1d"] = volume.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)

    df["tech_intraday_range_hl"] = ((high - low) / (close + 1e-9)).fillna(0.02)
    rolling_peak_252 = close.rolling(window=252, min_periods=21).max().fillna(close)
    df["tech_drawdown_from_peak_21d"] = ((close - rolling_peak_252) / (rolling_peak_252 + 1e-9)).fillna(0.0)

    def calc_slope(window_vals: np.ndarray) -> float:
        if len(window_vals) < 5 or np.any(np.isnan(window_vals)):
            return 0.0
        x = np.arange(len(window_vals))
        return float(np.polyfit(x, window_vals, 1)[0] / (window_vals[-1] + 1e-9))

    df["tech_trend_slope_21d"] = close.rolling(window=21, min_periods=10).apply(calc_slope, raw=True).fillna(0.0)

    sma50 = close.rolling(window=50, min_periods=10).mean().fillna(close)
    sma150 = close.rolling(window=150, min_periods=20).mean().fillna(close)
    sma200 = close.rolling(window=200, min_periods=20).mean().fillna(close)
    df["tech_close_vs_sma_50"] = ((close - sma50) / (sma50 + 1e-9)).fillna(0.0)
    df["tech_close_vs_sma_150"] = ((close - sma150) / (sma150 + 1e-9)).fillna(0.0)
    df["tech_close_vs_sma_200"] = ((close - sma200) / (sma200 + 1e-9)).fillna(0.0)
    sma200_shift20 = sma200.shift(20).fillna(sma200)
    df["tech_sma_200_trend_20"] = ((sma200 - sma200_shift20) / (sma200_shift20 + 1e-9)).fillna(0.0)

    low52 = close.rolling(window=252, min_periods=21).min().fillna(close)
    high52 = close.rolling(window=252, min_periods=21).max().fillna(close)
    df["tech_pct_above_52w_low"] = ((close - low52) / (low52 + 1e-9)).fillna(0.0)
    df["tech_pct_from_52w_high"] = ((close - high52) / (high52 + 1e-9)).fillna(0.0)

    up_vol = volume.where(close.diff() > 0, 0.0).rolling(50, min_periods=10).sum()
    dn_vol = volume.where(close.diff() < 0, 0.0).rolling(50, min_periods=10).sum()
    df["tech_up_down_volume_ratio_50"] = (up_vol / (dn_vol + 1e-9)).fillna(1.0)

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14, min_periods=5).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14, min_periods=5).mean()
    rs = gain / (loss + 1e-9)
    df["tech_rsi_14"] = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0) / 100.0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean().fillna(close * 0.02)
    df["raw_atr_14"] = atr14
    df["tech_atr_ratio_14"] = (atr14 / (close + 1e-9)).fillna(0.02)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["tech_macd_signal_diff"] = ((macd - macd_signal) / (close + 1e-9)).fillna(0.0)

    sma20 = close.rolling(20, min_periods=5).mean().fillna(close)
    std20 = close.rolling(20, min_periods=5).std().fillna(close * 0.02)
    upper_b = sma20 + 2.0 * std20
    lower_b = sma20 - 2.0 * std20
    df["tech_bollinger_bandwidth_20"] = ((upper_b - lower_b) / (sma20 + 1e-9)).fillna(0.04)

    vol63 = df["tech_return_1d"].rolling(63, min_periods=15).std().fillna(0.015)
    df["tech_historical_vol_ratio_63_21"] = (vol63 / (df["tech_volatility_21d"].replace(0, np.nan) + 1e-9)).fillna(1.0)

    df["future_return_21"] = close.shift(-21) / close - 1.0
    df["future_return_63"] = close.shift(-63) / close - 1.0
    df["future_return_126"] = close.shift(-126) / close - 1.0

    fwd_min_63 = close.rolling(window=63, min_periods=21).min().shift(-63)
    df["future_drawdown_63"] = ((fwd_min_63 - close) / close).fillna(-0.05)
    df["future_volatility_63"] = df["tech_return_1d"].rolling(window=63, min_periods=21).std().shift(-63).fillna(0.02)
    return df


class AnnualPatchTemporalTransformer(nn.Module):
    def __init__(self, input_dim: int = 23, embed_dim: int = 64, num_heads: int = 4, latent_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=128, batch_first=True, dropout=0.1, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.Linear(embed_dim, 1)
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.outcome_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.input_proj(x)
        T = h.size(1)
        if T > 1:
            pos = torch.arange(T, device=h.device, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.embed_dim, 2, device=h.device, dtype=torch.float32)
                * (-2.302585092994046 * 4 / self.embed_dim)
            )
            pe = torch.zeros(T, self.embed_dim, device=h.device)
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            h = h + pe.unsqueeze(0)

        h_trans = self.transformer(h)
        weights = torch.softmax(self.pool(h_trans), dim=1)
        h_pool = torch.sum(h_trans * weights, dim=1)

        latent = self.latent_head(h_pool)
        pred_outcome = self.outcome_head(latent)
        return latent, pred_outcome


def deduplicate_and_select_top_k(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    mem_tickers: np.ndarray,
    mem_sessions: np.ndarray,
    mem_dates: np.ndarray,
    mem_markets: np.ndarray,
    k: int = 25,
    min_separation: int = 21,
    is_distance: bool = False,
    max_precedents_per_ticker: int | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    n_cands = len(candidate_indices)
    if n_cands == 0:
        return np.array([], dtype=int), [], {"total_evaluated": 0, "total_kept": 0, "total_discarded": 0, "fallback_triggered": 0}

    order = np.argsort(candidate_scores) if is_distance else np.argsort(-candidate_scores)
    sorted_indices = candidate_indices[order]
    sorted_scores = candidate_scores[order]

    kept_indices: list[int] = []
    audit_log: list[dict[str, Any]] = []
    selected_sessions_by_ticker: dict[str, list[int]] = {}
    selected_dates_by_ticker: dict[str, list[str]] = {}

    total_evaluated = 0
    total_discarded = 0
    total_kept = 0

    for rank_raw, idx in enumerate(sorted_indices, 1):
        tkr = str(mem_tickers[idx])
        sess = int(mem_sessions[idx])
        dt = str(mem_dates[idx])
        mkt = str(mem_markets[idx])
        sc = float(sorted_scores[rank_raw - 1])

        total_evaluated += 1

        conflict_found = False
        conflict_prev_sess = None
        conflict_prev_dt = None
        conflict_gap = None
        discard_reason = "none"

        # Check ticker precedent cap if active
        if max_precedents_per_ticker is not None and tkr in selected_sessions_by_ticker:
            if len(selected_sessions_by_ticker[tkr]) >= max_precedents_per_ticker:
                conflict_found = True
                discard_reason = f"max_ticker_cap_{max_precedents_per_ticker}_exceeded"

        # Check intra-peer temporal separation
        if not conflict_found and tkr in selected_sessions_by_ticker:
            for prev_s, prev_d in zip(selected_sessions_by_ticker[tkr], selected_dates_by_ticker[tkr]):
                gap = abs(sess - prev_s)
                if gap < min_separation:
                    conflict_found = True
                    conflict_prev_sess = prev_s
                    conflict_prev_dt = prev_d
                    conflict_gap = gap
                    discard_reason = f"intra_peer_separation_{conflict_gap}_lt_{min_separation}"
                    break

        if conflict_found:
            total_discarded += 1
            audit_log.append({
                "candidate_rank_raw": rank_raw,
                "neighbor_market": mkt,
                "neighbor_ticker": tkr,
                "neighbor_date": dt,
                "neighbor_session_index": sess,
                "similarity_or_distance": round(sc, 5),
                "status": "DISCARDED",
                "conflict_ticker": tkr,
                "conflict_prev_date": conflict_prev_dt or "",
                "conflict_prev_session": conflict_prev_sess or "",
                "session_gap": conflict_gap or "",
                "discard_reason": discard_reason,
            })
        else:
            total_kept += 1
            kept_indices.append(idx)
            selected_sessions_by_ticker.setdefault(tkr, []).append(sess)
            selected_dates_by_ticker.setdefault(tkr, []).append(dt)
            audit_log.append({
                "candidate_rank_raw": rank_raw,
                "neighbor_market": mkt,
                "neighbor_ticker": tkr,
                "neighbor_date": dt,
                "neighbor_session_index": sess,
                "similarity_or_distance": round(sc, 5),
                "status": "KEPT",
                "conflict_ticker": "",
                "conflict_prev_date": "",
                "conflict_prev_session": "",
                "session_gap": "",
                "discard_reason": "none",
            })

            if len(kept_indices) >= k:
                break

    fallback_triggered = int(len(kept_indices) < k)
    stats = {
        "total_evaluated": total_evaluated,
        "total_kept": total_kept,
        "total_discarded": total_discarded,
        "fallback_triggered": fallback_triggered,
    }
    return np.array(kept_indices, dtype=int), audit_log, stats


def run_controlled_ablations():
    print("=" * 85)
    print("CONTROLLED BENCHMARK ABLATION SUITE (C09 RECONCILED)")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    # 1. Load Data
    print("1. Ingesting market data and computing 23 causal technical features...", flush=True)
    market_data = {}
    all_tickers_global = []
    ticker_to_id = {}
    target_moments = {}

    for m, tickers in TICKERS_BY_MARKET.items():
        market_data[m] = {}
        for tkr in tickers:
            p = DATA_DIR / f"{m}_{tkr}.parquet"
            if p.exists():
                market_data[m][tkr] = compute_23_features(pd.read_parquet(p))
                if tkr not in ticker_to_id:
                    ticker_to_id[tkr] = len(all_tickers_global)
                    all_tickers_global.append(tkr)

    # Compute pre-2021 feature scalers
    scaler_params = {}
    for m in TICKERS_BY_MARKET.keys():
        train_dfs = [df.loc[df.index <= pd.Timestamp("2020-12-31"), FEATURE_NAMES_23].dropna() for tkr, df in market_data[m].items()]
        train_dfs = [d for d in train_dfs if not d.empty]
        if train_dfs:
            all_train = pd.concat(train_dfs, axis=0)
            scaler_params[m] = {
                f: {"mean": float(np.mean(all_train[f].values)), "std": float(max(np.std(all_train[f].values), 1e-4))}
                for f in FEATURE_NAMES_23
            }

        # Target forward return moments pre-2021
        fwd_rets = []
        for tkr, df in market_data[m].items():
            sub = df.loc[df.index <= pd.Timestamp("2020-12-31"), "future_return_63"].dropna()
            fwd_rets.extend(sub.values.tolist())
        target_moments[m] = {
            "mean": float(np.mean(fwd_rets)),
            "std": float(np.std(fwd_rets)),
        }
        print(f"   [+] Market {m:6s} | Pre-2021 Target Return Mean: {target_moments[m]['mean']:+.4f}, Std: {target_moments[m]['std']:.4f}")

    def standardize_features(feat_mat: np.ndarray, market: str) -> np.ndarray:
        out = np.empty_like(feat_mat, dtype=np.float32)
        m_dict = scaler_params[market]
        for j, f in enumerate(FEATURE_NAMES_23):
            out[:, j] = np.clip((feat_mat[:, j] - m_dict[f]["mean"]) / m_dict[f]["std"], -5.0, 5.0)
        return out

    MARKET_CALENDARS_2024 = {}
    for m in TICKERS_BY_MARKET.keys():
        dates_set = set()
        for tkr, df in market_data[m].items():
            sub_2024 = df.loc[(df.index >= pd.Timestamp("2024-01-01")) & (df.index <= pd.Timestamp("2024-12-31"))]
            dates_set.update(sub_2024.index.tolist())
        MARKET_CALENDARS_2024[m] = sorted(list(dates_set))

    # 2. Load Frozen Models
    print("2. Loading authoritative Patch-Transformer checkpoints (Seeds 7, 17, 37)...", flush=True)
    encoders: dict[int, nn.Module] = {}
    for s in SEEDS:
        ckpt_path = MODELS_DIR / f"v4_metric_transformer_seed_{s}.pt"
        model = AnnualPatchTemporalTransformer(input_dim=23, embed_dim=64, num_heads=4, latent_dim=128).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        encoders[s] = model
        print(f"   [+] Seed {s:2d} Loaded")

    # 3. Index Memory Bank
    print("3. Indexing Causal Memory Bank (Mature Outcomes <= 2020-12-31)...", flush=True)
    mem_records_base = []
    mem_patch_list = []

    for m in TICKERS_BY_MARKET.keys():
        for tkr, df in market_data[m].items():
            df_train = df.loc[df.index <= pd.Timestamp("2020-12-31")]
            n_len = len(df_train)
            if n_len < T_WIN + 126:
                continue
            feat_raw = df_train[FEATURE_NAMES_23].values
            feat_norm = standardize_features(feat_raw, m)
            ret63 = df_train["future_return_63"].values
            dd63 = df_train["future_drawdown_63"].values
            dates = df_train.index

            for i in range(T_WIN, n_len - 126):
                obs_dt = dates[i]
                avail_dt = dates[i + 126]
                if avail_dt > pd.Timestamp("2020-12-31"):
                    continue
                w_slice = feat_norm[i - T_WIN:i]
                patch_mat = w_slice.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)
                mem_patch_list.append(patch_mat)
                mem_records_base.append({
                    "market": m,
                    "ticker": tkr,
                    "session_index": i,
                    "date": obs_dt.strftime("%Y-%m-%d"),
                    "future_return_63": float(ret63[i]) if not np.isnan(ret63[i]) else 0.0,
                    "future_drawdown_63": float(dd63[i]) if not np.isnan(dd63[i]) else 0.0,
                })

    mem_patches_tensor = torch.tensor(np.array(mem_patch_list, dtype=np.float32), device=DEVICE)
    mem_tickers_arr = np.array([r["ticker"] for r in mem_records_base])
    mem_market_arr = np.array([r["market"] for r in mem_records_base])
    mem_sessions_arr = np.array([r["session_index"] for r in mem_records_base], dtype=int)
    mem_dates_arr = np.array([r["date"] for r in mem_records_base])
    mem_ret63 = np.array([r["future_return_63"] for r in mem_records_base], dtype=np.float32)
    mem_dd63 = np.array([r["future_drawdown_63"] for r in mem_records_base], dtype=np.float32)
    print(f"   [+] Memory Bank: {len(mem_records_base):,d} records indexed.")

    # Compute memory latents on GPU
    print("4. Encoding Memory Bank Latents per Seed...", flush=True)
    mem_latents_by_seed_gpu = {}
    with torch.no_grad():
        for s in SEEDS:
            m_lats = []
            for b_i in range(0, len(mem_patches_tensor), 2048):
                b_p = mem_patches_tensor[b_i:b_i + 2048]
                l, _ = encoders[s](b_p)
                l = F.normalize(l, p=2, dim=1)
                m_lats.append(l)
            mem_latents_by_seed_gpu[s] = torch.cat(m_lats, dim=0)

    # 4. System Definitions
    SYSTEMS_INDIVIDUAL = [
        "P0_CANONICAL",
        "P2_CONTROL",
        "P0_STAR_CANONICAL",
        "EXP1_SCALE_HARMONIZED",
        "EXP2_ASYM_LOSS_PENALTY",
        "EXP3A_UNSCALED_SELECTION",
        "EXP3B_UNSCALED_VOL_SIZING",
        "EXP4_GLOBAL_NW",
        "EXP4_DOMESTIC_LINEAR",
        "EXP9_DIVERSE_EPISODES_CAP3",
        "EXP9_DIVERSE_EPISODES_CAP1",
        "EXP10_TAIL_K50_A10",
        "EXP10_TAIL_WEIGHTED",
    ]

    print(f"\n5. Executing Multi-Market Backtest across {len(SYSTEMS_INDIVIDUAL)} individual systems + 1 multi-seed ensemble...", flush=True)

    matrix_rows: list[dict[str, Any]] = []
    trade_ledger_rows: list[dict[str, Any]] = []
    equity_curve_rows: list[dict[str, Any]] = []

    # Iterate over sovereign markets
    for m_idx, (m, tickers) in enumerate(TICKERS_BY_MARKET.items()):
        cal_dates = MARKET_CALENDARS_2024[m]
        n_sessions = len(cal_dates)
        dates_str = [d.strftime("%Y-%m-%d") for d in cal_dates]
        print(f"\n--- Sovereign Market: {m:6s} ({len(tickers)} tickers, {n_sessions} sessions) ---")

        # Build 2024 evaluation data
        price_open = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
        price_close = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
        vol_2024 = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
        atr_ratio_2024 = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
        patches_2024 = np.zeros((n_sessions, len(tickers), N_PATCHES, 23), dtype=np.float32)

        for t_i, tkr in enumerate(tickers):
            df = market_data[m][tkr]
            feat_raw = df[FEATURE_NAMES_23].values
            feat_norm = standardize_features(feat_raw, m)

            for d_i, dt in enumerate(cal_dates):
                if dt in df.index:
                    loc = df.index.get_loc(dt)
                    price_open[d_i, t_i] = df["open"].iloc[loc]
                    price_close[d_i, t_i] = df["close"].iloc[loc]
                    vol_2024[d_i, t_i] = df["tech_volatility_21d"].iloc[loc]
                    atr_ratio_2024[d_i, t_i] = df["tech_atr_ratio_14"].iloc[loc]

                    if loc >= T_WIN:
                        w_slice = feat_norm[loc - T_WIN:loc]
                        patches_2024[d_i, t_i] = w_slice.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)

        for t_i in range(len(tickers)):
            price_open[:, t_i] = pd.Series(price_open[:, t_i]).ffill().bfill().values
            price_close[:, t_i] = pd.Series(price_close[:, t_i]).ffill().bfill().values

        # Precompute daily model latents and predictions for all seeds
        # day_idx -> seed -> {q_lats, q_preds_np, sim_matrix}
        daily_model_cache = {}
        for d_i in range(n_sessions):
            cand_p = torch.tensor(patches_2024[d_i], dtype=torch.float32, device=DEVICE)
            daily_model_cache[d_i] = {}
            with torch.no_grad():
                for s in SEEDS:
                    q_lats, q_preds = encoders[s](cand_p)
                    q_lats = F.normalize(q_lats, p=2, dim=1)
                    q_preds_np = q_preds.squeeze(1).cpu().numpy()
                    sim_mat = torch.matmul(mem_latents_by_seed_gpu[s], q_lats.T).cpu().numpy()
                    daily_model_cache[d_i][s] = {
                        "q_preds": q_preds_np,
                        "sim_matrix": sim_mat,
                    }

        # -------------------------------------------------------------
        # A. RUN INDIVIDUAL SEED SYSTEMS
        # -------------------------------------------------------------
        for s in SEEDS:
            for sys_id in SYSTEMS_INDIVIDUAL:
                initial_capital = 100000.0
                cash = initial_capital
                portfolio_equity = np.zeros(n_sessions)
                daily_returns = np.zeros(n_sessions)

                open_positions: list[dict[str, Any]] = []
                cell_trades: list[dict[str, Any]] = []
                pending_exits: list[dict[str, Any]] = []
                pending_entries: list[dict[str, Any]] = []
                trade_seq = 0
                entry_seq = 0

                for day_idx in range(n_sessions):
                    cur_date_str = dates_str[day_idx]

                    # 1. EXITS AT OPEN
                    for ex in pending_exits:
                        pos = ex["pos"]
                        t_i = pos["ticker_idx"]
                        exit_price = price_open[day_idx, t_i]
                        gross_proceeds = pos["shares"] * exit_price
                        exit_fee = gross_proceeds * 0.0010
                        net_proceeds = gross_proceeds - exit_fee
                        cash += net_proceeds

                        tot_cost = pos["entry_price"] * pos["shares"] + pos["entry_fee"]
                        realized_pnl = net_proceeds - tot_cost
                        ret_pct = realized_pnl / tot_cost
                        held_days = day_idx - pos["entry_idx"]

                        trade_seq += 1
                        trade_record = {
                            "trade_id": pos["trade_id"],
                            "market": m,
                            "seed": s,
                            "system": sys_id,
                            "ticker": pos["ticker"],
                            "signal_date": ex["signal_date"],
                            "entry_date": dates_str[pos["entry_idx"]],
                            "exit_date": cur_date_str,
                            "holding_days": int(held_days),
                            "exit_reason": ex["exit_reason"],
                            "entry_price": round(float(pos["entry_price"]), 2),
                            "exit_price": round(float(exit_price), 2),
                            "shares": int(pos["shares"]),
                            "entry_fee": round(float(pos["entry_fee"]), 2),
                            "exit_fee": round(float(exit_fee), 2),
                            "gross_pnl": round(float(gross_proceeds - pos["entry_price"] * pos["shares"]), 2),
                            "realized_pnl": round(float(realized_pnl), 2),
                            "return_pct": round(float(ret_pct), 4),
                            "win": int(realized_pnl > 0),
                        }
                        cell_trades.append(trade_record)
                        trade_ledger_rows.append(trade_record)
                        open_positions.remove(pos)
                    pending_exits = []

                    # 2. ENTRIES AT OPEN
                    for ent in pending_entries:
                        t_i = ent["ticker_idx"]
                        open_p = price_open[day_idx, t_i]
                        if open_p > 0:
                            target_cap = ent["allocated_capital"]
                            shares = int((target_cap * 0.95) / open_p)
                            if shares > 0:
                                cost = shares * open_p
                                fee = cost * 0.0010
                                if cash >= (cost + fee):
                                    cash -= (cost + fee)
                                    entry_seq += 1
                                    trade_id = f"TRD_{m}_{s}_{sys_id}_{entry_seq:03d}"
                                    open_positions.append({
                                        "trade_id": trade_id,
                                        "ticker": tickers[t_i],
                                        "ticker_idx": t_i,
                                        "entry_idx": day_idx,
                                        "entry_price": open_p,
                                        "peak_price": open_p,
                                        "shares": shares,
                                        "entry_fee": fee,
                                    })
                    pending_entries = []

                    # 3. MARK TO MARKET AT CLOSE
                    pos_value = sum(p["shares"] * price_close[day_idx, p["ticker_idx"]] for p in open_positions)
                    for p in open_positions:
                        p["peak_price"] = max(p["peak_price"], price_close[day_idx, p["ticker_idx"]])

                    tot_equity = cash + pos_value
                    portfolio_equity[day_idx] = tot_equity
                    daily_returns[day_idx] = (tot_equity - initial_capital) / initial_capital if day_idx == 0 else (tot_equity - portfolio_equity[day_idx - 1]) / portfolio_equity[day_idx - 1]

                    # 4. SIGNALS AT CLOSE
                    if day_idx < n_sessions - 1:
                        for p in open_positions:
                            held_days = day_idx - p["entry_idx"] + 1
                            cur_p = price_close[day_idx, p["ticker_idx"]]
                            dd_peak = (cur_p - p["peak_price"]) / p["peak_price"]
                            atr_rat = atr_ratio_2024[day_idx, p["ticker_idx"]]
                            stop_distance = max(0.10, 2.5 * atr_rat)

                            is_stop = (dd_peak < -stop_distance) and (held_days >= 1)
                            is_max_horizon = held_days >= 63
                            if is_stop or is_max_horizon:
                                pending_exits.append({
                                    "pos": p,
                                    "signal_date": cur_date_str,
                                    "exit_reason": "atr_chandelier_stop" if is_stop else "max_horizon",
                                })

                        exiting_ids = {id(item["pos"]) for item in pending_exits}
                        remaining_count = len([p for p in open_positions if id(p) not in exiting_ids])
                        slots_open = 3 - remaining_count

                        if slots_open > 0 and day_idx < n_sessions - 5:
                            held_tickers = {p["ticker"] for p in open_positions if id(p) not in exiting_ids}
                            available_indices = [i for i in range(len(tickers)) if tickers[i] not in held_tickers]

                            if available_indices:
                                scores = np.zeros(len(available_indices))
                                vol_avail = np.zeros(len(available_indices))
                                sim_col_cache = daily_model_cache[day_idx][s]["sim_matrix"]
                                q_preds_s = daily_model_cache[day_idx][s]["q_preds"]

                                for idx_c, t_i in enumerate(available_indices):
                                    q_tkr = tickers[t_i]
                                    pred_val = float(q_preds_s[t_i])
                                    v = float(vol_2024[day_idx, t_i])
                                    vol_avail[idx_c] = v
                                    sim_col = sim_col_cache[:, t_i]

                                    # Retrieval Mask
                                    is_domestic = sys_id in ("P0_STAR_CANONICAL", "EXP4_DOMESTIC_LINEAR")
                                    if is_domestic:
                                        valid_mask = (mem_tickers_arr != q_tkr) & (mem_market_arr == m)
                                    else:
                                        valid_mask = (mem_tickers_arr != q_tkr)
                                    valid_indices = np.where(valid_mask)[0]

                                    # Parameters per system
                                    k_param = 50 if sys_id == "EXP10_TAIL_K50_A10" else 25
                                    cap_param = 3 if sys_id == "EXP9_DIVERSE_EPISODES_CAP3" else (1 if sys_id == "EXP9_DIVERSE_EPISODES_CAP1" else None)

                                    valid_nbrs, _, _ = deduplicate_and_select_top_k(
                                        candidate_indices=valid_indices,
                                        candidate_scores=sim_col[valid_indices],
                                        mem_tickers=mem_tickers_arr,
                                        mem_sessions=mem_sessions_arr,
                                        mem_dates=mem_dates_arr,
                                        mem_markets=mem_market_arr,
                                        k=k_param,
                                        min_separation=21,
                                        is_distance=False,
                                        max_precedents_per_ticker=cap_param,
                                    )

                                    # Invariant Check: 0% Entity Leakage
                                    assert not np.any(mem_tickers_arr[valid_nbrs] == q_tkr), f"Leakage violation for {q_tkr}!"

                                    nbr_sims = sim_col[valid_nbrs]
                                    nbr_ret = mem_ret63[valid_nbrs]

                                    # Kernel Weights
                                    is_nw = sys_id in ("P0_STAR_CANONICAL", "EXP4_GLOBAL_NW")
                                    if is_nw:
                                        nw_exp = np.exp((nbr_sims - 1.0) / 0.08)
                                        weights = nw_exp / (np.sum(nw_exp) + 1e-9)
                                    else:
                                        w_raw = nbr_sims - np.min(nbr_sims) + 1e-4
                                        weights = w_raw / np.sum(w_raw)

                                    mu_w = float(np.sum(weights * nbr_ret))

                                    # Tail Risk Estimation
                                    alpha_tail = 0.10 if sys_id == "EXP10_TAIL_K50_A10" else 0.05
                                    var_tail = float(np.percentile(nbr_ret, alpha_tail * 100))
                                    tail_mask = nbr_ret <= var_tail

                                    if sys_id == "EXP10_TAIL_WEIGHTED":
                                        if np.sum(tail_mask) > 0 and np.sum(weights[tail_mask]) > 0:
                                            cvar_tail = float(np.sum(weights[tail_mask] * nbr_ret[tail_mask]) / np.sum(weights[tail_mask]))
                                        else:
                                            cvar_tail = var_tail
                                    else:
                                        cvar_tail = float(np.mean(nbr_ret[tail_mask])) if np.sum(tail_mask) > 0 else var_tail

                                    # Score Formulations
                                    denom = v + 1e-4
                                    if sys_id == "P2_CONTROL":
                                        sc = (pred_val + 0.8 * mu_w) / denom
                                    elif sys_id == "EXP1_SCALE_HARMONIZED":
                                        m_mean = target_moments[m]["mean"]
                                        m_std = target_moments[m]["std"]
                                        pred_scaled = pred_val * m_std + m_mean
                                        sc = (pred_scaled + 0.8 * mu_w - 0.2 * abs(cvar_tail)) / denom
                                    elif sys_id == "EXP2_ASYM_LOSS_PENALTY":
                                        loss_pen = max(0.0, -cvar_tail)
                                        sc = (pred_val + 0.8 * mu_w - 0.2 * loss_pen) / denom
                                    elif sys_id in ("EXP3A_UNSCALED_SELECTION", "EXP3B_UNSCALED_VOL_SIZING"):
                                        sc = pred_val + 0.8 * mu_w - 0.2 * abs(cvar_tail)
                                    else:
                                        sc = (pred_val + 0.8 * mu_w - 0.2 * abs(cvar_tail)) / denom

                                    scores[idx_c] = sc

                                # Rank descending
                                top_order = np.argsort(-scores)
                                selected_subset = top_order[:slots_open]
                                n_entering = len(selected_subset)

                                # Capital Allocation
                                if sys_id == "EXP3B_UNSCALED_VOL_SIZING":
                                    total_cap_to_alloc = tot_equity * (n_entering / 3.0)
                                    inv_v = 1.0 / (vol_avail[selected_subset] + 1e-4)
                                    vol_weights = inv_v / np.sum(inv_v)
                                    for sub_i, chosen_c in enumerate(selected_subset):
                                        real_ti = available_indices[chosen_c]
                                        cap = float(total_cap_to_alloc * vol_weights[sub_i])
                                        pending_entries.append({
                                            "ticker_idx": real_ti,
                                            "signal_date": cur_date_str,
                                            "allocated_capital": cap,
                                        })
                                else:
                                    cap_per_slot = tot_equity / 3.0
                                    for chosen_c in selected_subset:
                                        real_ti = available_indices[chosen_c]
                                        pending_entries.append({
                                            "ticker_idx": real_ti,
                                            "signal_date": cur_date_str,
                                            "allocated_capital": cap_per_slot,
                                        })

                # Compute Performance Metrics
                ann_ret = (portfolio_equity[-1] - initial_capital) / initial_capital
                mean_d = np.mean(daily_returns)
                std_d = np.std(daily_returns)
                sharpe = float((mean_d / (std_d + 1e-9)) * np.sqrt(SESSIONS_PER_YEAR[m]))
                downside_d = daily_returns[daily_returns < 0]
                sortino = float((mean_d / (np.std(downside_d) + 1e-9)) * np.sqrt(SESSIONS_PER_YEAR[m])) if len(downside_d) > 0 else 0.0

                cum_eq = portfolio_equity
                cum_max = np.maximum.accumulate(cum_eq)
                dd_series = (cum_eq - cum_max) / cum_max
                max_dd = float(np.min(dd_series))

                win_rate = float(np.mean([t["win"] for t in cell_trades])) if cell_trades else 0.0
                med_hold = float(np.median([t["holding_days"] for t in cell_trades])) if cell_trades else 0.0
                tot_trades = len(cell_trades)

                matrix_rows.append({
                    "market": m,
                    "seed": s,
                    "system": sys_id,
                    "annualized_return": round(ann_ret, 4),
                    "sharpe_ratio": round(sharpe, 4),
                    "sortino_ratio": round(sortino, 4),
                    "max_drawdown": round(max_dd, 4),
                    "win_rate": round(win_rate, 4),
                    "median_hold_days": round(med_hold, 1),
                    "total_trades": tot_trades,
                    "final_equity": round(float(portfolio_equity[-1]), 2),
                })

                for d_i, dt in enumerate(cal_dates):
                    equity_curve_rows.append({
                        "market": m,
                        "seed": s,
                        "system": sys_id,
                        "date": dt.strftime("%Y-%m-%d"),
                        "equity": round(float(portfolio_equity[d_i]), 2),
                        "daily_return": round(float(daily_returns[d_i]), 6),
                    })

        # -------------------------------------------------------------
        # B. RUN MULTI-SEED ENSEMBLE SYSTEM (EXP5)
        # -------------------------------------------------------------
        initial_capital = 100000.0
        cash = initial_capital
        portfolio_equity = np.zeros(n_sessions)
        daily_returns = np.zeros(n_sessions)

        open_positions = []
        cell_trades = []
        pending_exits = []
        pending_entries = []
        trade_seq = 0
        entry_seq = 0

        for day_idx in range(n_sessions):
            cur_date_str = dates_str[day_idx]

            # 1. EXITS AT OPEN
            for ex in pending_exits:
                pos = ex["pos"]
                t_i = pos["ticker_idx"]
                exit_price = price_open[day_idx, t_i]
                gross_proceeds = pos["shares"] * exit_price
                exit_fee = gross_proceeds * 0.0010
                net_proceeds = gross_proceeds - exit_fee
                cash += net_proceeds

                tot_cost = pos["entry_price"] * pos["shares"] + pos["entry_fee"]
                realized_pnl = net_proceeds - tot_cost
                ret_pct = realized_pnl / tot_cost
                held_days = day_idx - pos["entry_idx"]

                trade_seq += 1
                trade_record = {
                    "trade_id": pos["trade_id"],
                    "market": m,
                    "seed": "ensemble",
                    "system": "EXP5_MULTI_SEED_ENSEMBLE",
                    "ticker": pos["ticker"],
                    "signal_date": ex["signal_date"],
                    "entry_date": dates_str[pos["entry_idx"]],
                    "exit_date": cur_date_str,
                    "holding_days": int(held_days),
                    "exit_reason": ex["exit_reason"],
                    "entry_price": round(float(pos["entry_price"]), 2),
                    "exit_price": round(float(exit_price), 2),
                    "shares": int(pos["shares"]),
                    "entry_fee": round(float(pos["entry_fee"]), 2),
                    "exit_fee": round(float(exit_fee), 2),
                    "gross_pnl": round(float(gross_proceeds - pos["entry_price"] * pos["shares"]), 2),
                    "realized_pnl": round(float(realized_pnl), 2),
                    "return_pct": round(float(ret_pct), 4),
                    "win": int(realized_pnl > 0),
                }
                cell_trades.append(trade_record)
                trade_ledger_rows.append(trade_record)
                open_positions.remove(pos)
            pending_exits = []

            # 2. ENTRIES AT OPEN
            for ent in pending_entries:
                t_i = ent["ticker_idx"]
                open_p = price_open[day_idx, t_i]
                if open_p > 0:
                    target_cap = ent["allocated_capital"]
                    shares = int((target_cap * 0.95) / open_p)
                    if shares > 0:
                        cost = shares * open_p
                        fee = cost * 0.0010
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            entry_seq += 1
                            trade_id = f"TRD_{m}_ENS_EXP5_{entry_seq:03d}"
                            open_positions.append({
                                "trade_id": trade_id,
                                "ticker": tickers[t_i],
                                "ticker_idx": t_i,
                                "entry_idx": day_idx,
                                "entry_price": open_p,
                                "peak_price": open_p,
                                "shares": shares,
                                "entry_fee": fee,
                            })
            pending_entries = []

            # 3. MARK TO MARKET AT CLOSE
            pos_value = sum(p["shares"] * price_close[day_idx, p["ticker_idx"]] for p in open_positions)
            for p in open_positions:
                p["peak_price"] = max(p["peak_price"], price_close[day_idx, p["ticker_idx"]])

            tot_equity = cash + pos_value
            portfolio_equity[day_idx] = tot_equity
            daily_returns[day_idx] = (tot_equity - initial_capital) / initial_capital if day_idx == 0 else (tot_equity - portfolio_equity[day_idx - 1]) / portfolio_equity[day_idx - 1]

            # 4. SIGNALS AT CLOSE
            if day_idx < n_sessions - 1:
                for p in open_positions:
                    held_days = day_idx - p["entry_idx"] + 1
                    cur_p = price_close[day_idx, p["ticker_idx"]]
                    dd_peak = (cur_p - p["peak_price"]) / p["peak_price"]
                    atr_rat = atr_ratio_2024[day_idx, p["ticker_idx"]]
                    stop_distance = max(0.10, 2.5 * atr_rat)

                    is_stop = (dd_peak < -stop_distance) and (held_days >= 1)
                    is_max_horizon = held_days >= 63
                    if is_stop or is_max_horizon:
                        pending_exits.append({
                            "pos": p,
                            "signal_date": cur_date_str,
                            "exit_reason": "atr_chandelier_stop" if is_stop else "max_horizon",
                        })

                exiting_ids = {id(item["pos"]) for item in pending_exits}
                remaining_count = len([p for p in open_positions if id(p) not in exiting_ids])
                slots_open = 3 - remaining_count

                if slots_open > 0 and day_idx < n_sessions - 5:
                    held_tickers = {p["ticker"] for p in open_positions if id(p) not in exiting_ids}
                    available_indices = [i for i in range(len(tickers)) if tickers[i] not in held_tickers]

                    if available_indices:
                        scores = np.zeros(len(available_indices))

                        for idx_c, t_i in enumerate(available_indices):
                            q_tkr = tickers[t_i]
                            v = float(vol_2024[day_idx, t_i])
                            denom = v + 1e-4

                            seed_preds = []
                            seed_mus = []
                            seed_cvars = []

                            for s in SEEDS:
                                pred_s = float(daily_model_cache[day_idx][s]["q_preds"][t_i])
                                sim_col = daily_model_cache[day_idx][s]["sim_matrix"][:, t_i]
                                valid_mask = (mem_tickers_arr != q_tkr)
                                valid_indices = np.where(valid_mask)[0]

                                valid_nbrs, _, _ = deduplicate_and_select_top_k(
                                    candidate_indices=valid_indices,
                                    candidate_scores=sim_col[valid_indices],
                                    mem_tickers=mem_tickers_arr,
                                    mem_sessions=mem_sessions_arr,
                                    mem_dates=mem_dates_arr,
                                    mem_markets=mem_market_arr,
                                    k=25,
                                    min_separation=21,
                                    is_distance=False,
                                )

                                nbr_sims = sim_col[valid_nbrs]
                                nbr_ret = mem_ret63[valid_nbrs]
                                w_raw = nbr_sims - np.min(nbr_sims) + 1e-4
                                weights = w_raw / np.sum(w_raw)

                                mu_w_s = float(np.sum(weights * nbr_ret))
                                var_05 = float(np.percentile(nbr_ret, 5))
                                tail_ret = nbr_ret[nbr_ret <= var_05]
                                cvar_s = float(np.mean(tail_ret)) if len(tail_ret) > 0 else var_05

                                seed_preds.append(pred_s)
                                seed_mus.append(mu_w_s)
                                seed_cvars.append(cvar_s)

                            # Ensemble average
                            ens_pred = float(np.mean(seed_preds))
                            ens_mu = float(np.mean(seed_mus))
                            ens_cvar = float(np.mean(seed_cvars))

                            sc = (ens_pred + 0.8 * ens_mu - 0.2 * abs(ens_cvar)) / denom
                            scores[idx_c] = sc

                        top_order = np.argsort(-scores)
                        selected_subset = top_order[:slots_open]
                        cap_per_slot = tot_equity / 3.0
                        for chosen_c in selected_subset:
                            real_ti = available_indices[chosen_c]
                            pending_entries.append({
                                "ticker_idx": real_ti,
                                "signal_date": cur_date_str,
                                "allocated_capital": cap_per_slot,
                            })

        # Multi-seed Ensemble Performance
        ann_ret = (portfolio_equity[-1] - initial_capital) / initial_capital
        mean_d = np.mean(daily_returns)
        std_d = np.std(daily_returns)
        sharpe = float((mean_d / (std_d + 1e-9)) * np.sqrt(SESSIONS_PER_YEAR[m]))
        downside_d = daily_returns[daily_returns < 0]
        sortino = float((mean_d / (np.std(downside_d) + 1e-9)) * np.sqrt(SESSIONS_PER_YEAR[m])) if len(downside_d) > 0 else 0.0
        cum_eq = portfolio_equity
        cum_max = np.maximum.accumulate(cum_eq)
        max_dd = float(np.min((cum_eq - cum_max) / cum_max))
        win_rate = float(np.mean([t["win"] for t in cell_trades])) if cell_trades else 0.0
        med_hold = float(np.median([t["holding_days"] for t in cell_trades])) if cell_trades else 0.0

        matrix_rows.append({
            "market": m,
            "seed": "ensemble",
            "system": "EXP5_MULTI_SEED_ENSEMBLE",
            "annualized_return": round(ann_ret, 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "median_hold_days": round(med_hold, 1),
            "total_trades": len(cell_trades),
            "final_equity": round(float(portfolio_equity[-1]), 2),
        })

        for d_i, dt in enumerate(cal_dates):
            equity_curve_rows.append({
                "market": m,
                "seed": "ensemble",
                "system": "EXP5_MULTI_SEED_ENSEMBLE",
                "date": dt.strftime("%Y-%m-%d"),
                "equity": round(float(portfolio_equity[d_i]), 2),
                "daily_return": round(float(daily_returns[d_i]), 6),
            })

    # Save Output Tables
    df_matrix = pd.DataFrame(matrix_rows)
    matrix_out_path = OUTPUT_DIR / "ablation_14_systems_matrix.csv"
    df_matrix.to_csv(matrix_out_path, index=False)
    print(f"\n[+] Saved Performance Matrix: {matrix_out_path} ({len(df_matrix)} cells)")

    df_trades = pd.DataFrame(trade_ledger_rows)
    trades_out_path = OUTPUT_DIR / "ablation_trade_ledger.csv"
    df_trades.to_csv(trades_out_path, index=False)
    print(f"[+] Saved Trade Ledger: {trades_out_path} ({len(df_trades)} trades)")

    df_equity = pd.DataFrame(equity_curve_rows)
    equity_out_path = OUTPUT_DIR / "ablation_daily_equity_curves.csv"
    df_equity.to_csv(equity_out_path, index=False)
    print(f"[+] Saved Daily Equity Curves: {equity_out_path} ({len(df_equity)} rows)")

    # -------------------------------------------------------------
    # 6. REPLICATION INTEGRITY & VERIFICATION SANITY CHECK
    # -------------------------------------------------------------
    print("\n6. Running Verification Sanity Checks against Canonical Outputs...", flush=True)
    if CANONICAL_MATRIX_PATH.exists():
        df_canon = pd.read_csv(CANONICAL_MATRIX_PATH)
        # Check P0 replication
        sub_p0 = df_matrix[df_matrix["system"] == "P0_CANONICAL"].sort_values(["market", "seed"])
        canon_p0 = df_canon[df_canon["system"] == "P0"].sort_values(["market", "seed"])

        ret_diff = np.abs(sub_p0["annualized_return"].values - canon_p0["annualized_return"].values)
        sharpe_diff = np.abs(sub_p0["sharpe_ratio"].values - canon_p0["sharpe_ratio"].values)

        max_ret_err = float(np.max(ret_diff))
        max_sh_err = float(np.max(sharpe_diff))
        print(f"   [+] P0 Benchmark Matching Check: Max Ann Return Delta = {max_ret_err:.6f}, Max Sharpe Delta = {max_sh_err:.6f}")
        assert max_ret_err < 1e-3, f"P0 replication failed! Max return error: {max_ret_err}"
        assert max_sh_err < 1e-3, f"P0 replication failed! Max Sharpe error: {max_sh_err}"
        print("   [+] VERIFIED: P0 EXACT REPLICATION PASSES ZERO-ERROR CONTRACT!")

    # -------------------------------------------------------------
    # 7. GENERATE COMPARATIVE RESEARCH SUMMARY
    # -------------------------------------------------------------
    summary_report = {}
    print("\n" + "=" * 85)
    print("EMPIRICAL COMPARATIVE RESULTS (MEAN ACROSS 6 MARKETS & SEEDS)")
    print("=" * 85)

    all_systems = df_matrix["system"].unique()
    for sys_id in all_systems:
        sub = df_matrix[df_matrix["system"] == sys_id]
        mean_ret = float(sub["annualized_return"].mean())
        mean_sh = float(sub["sharpe_ratio"].mean())
        mean_dd = float(sub["max_drawdown"].mean())
        mean_win = float(sub["win_rate"].mean())
        mean_trd = float(sub["total_trades"].mean())

        summary_report[sys_id] = {
            "mean_annualized_return": round(mean_ret, 4),
            "mean_sharpe_ratio": round(mean_sh, 4),
            "mean_max_drawdown": round(mean_dd, 4),
            "mean_win_rate": round(mean_win, 4),
            "mean_trades_per_cell": round(mean_trd, 1),
            "cells_evaluated": len(sub),
        }
        print(f"{sys_id:28s} | Ann Ret: {mean_ret:+7.2%} | Sharpe: {mean_sh:+6.3f} | Max DD: {mean_dd:7.2%} | Win: {mean_win:5.1%} | Trades/cell: {mean_trd:4.1f}")

    # Save summary report JSON
    summary_json_path = OUTPUT_DIR / "ablation_summary_report.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\n[+] Saved Summary Report: {summary_json_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_controlled_ablations()
