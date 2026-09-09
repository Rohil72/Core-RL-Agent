"""Authoritative V4 Retrospective Multi-Year Robustness Panel (2021–2025)
========================================================================
Loads pre-trained V4 metric checkpoints and evaluates across 5 out-of-sample calendar years:
2021, 2022, 2023, 2024, 2025.

Contract:
- 252 x 23 annual context (42 patches x 6 sessions).
- Outcome-structured S^127 latent geometry from trained v4_metric_transformer_seed_{s}.pt.
- Causal memory bank strictly <= 2020-12-31.
- Signal at session t Close, execution at session t+1 Open (10 bps fee).
- Adaptive 2.5 x ATR_14 Chandelier exit (10% floor), H_min=1, H_max=63.
- True Expected Shortfall (CVaR_0.05 = E[R | R <= VaR_0.05]).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
DATA_DIR = PROJECT_ROOT / "data" / "cache" / "ohlcv"
EVIDENCE_DIR = PROJECT_ROOT / "paper" / "internal" / "evidence" / "research_defense_extract" / "v4_annual_252_evidence"
MODELS_DIR = EVIDENCE_DIR / "models"
OUTPUT_DIR = EVIDENCE_DIR / "retrospective_2021_2025"
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
EVAL_YEARS = [2021, 2022, 2023, 2024, 2025]

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
    
    df["future_return_63"] = close.shift(-63) / close - 1.0
    return df

print("1. Loading raw market data...", flush=True)
market_data = {}
for m, tickers in TICKERS_BY_MARKET.items():
    market_data[m] = {}
    for tkr in tickers:
        p = DATA_DIR / f"{m}_{tkr}.parquet"
        if p.exists():
            market_data[m][tkr] = compute_23_features(pd.read_parquet(p))

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

def standardize_features(feat_mat: np.ndarray, market: str) -> np.ndarray:
    out = np.empty_like(feat_mat, dtype=np.float32)
    m_dict = scaler_params[market]
    for j, f in enumerate(FEATURE_NAMES_23):
        out[:, j] = np.clip((feat_mat[:, j] - m_dict[f]["mean"]) / m_dict[f]["std"], -5.0, 5.0)
    return out

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

print("2. Loading Pre-Trained V4 Metric Encoders from checkpoints...", flush=True)
encoders = {}
for s in SEEDS:
    ckpt_p = MODELS_DIR / f"v4_metric_transformer_seed_{s}.pt"
    m = AnnualPatchTemporalTransformer().to(DEVICE)
    m.load_state_dict(torch.load(ckpt_p, map_location=DEVICE, weights_only=True))
    m.eval()
    encoders[s] = m
    print(f"   [+] Loaded {ckpt_p.name}")

print("3. Indexing Causal Memory Bank (<= 2020-12-31)...", flush=True)
mem_records_base = []
mem_patch_list = []

for m in TICKERS_BY_MARKET.keys():
    for tkr, df in market_data[m].items():
        df_train = df.loc[df.index <= pd.Timestamp("2020-12-31")]
        n_len = len(df_train)
        if n_len < T_WIN + 63:
            continue
        feat_raw = df_train[FEATURE_NAMES_23].values
        feat_norm = standardize_features(feat_raw, m)
        dates = df_train.index
        ret63 = df_train["future_return_63"].values

        for i in range(T_WIN, n_len - 63):
            if dates[i + 63] > pd.Timestamp("2020-12-31") or np.isnan(ret63[i]):
                continue
            raw_win = feat_norm[i - T_WIN : i]
            patch_mat = raw_win.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)
            mem_patch_list.append(patch_mat)
            mem_records_base.append({
                "market": m,
                "ticker": tkr,
                "future_return_63": float(ret63[i]),
            })

all_mem_patches = torch.tensor(np.array(mem_patch_list, dtype=np.float32))
mem_latents_by_seed_gpu = {}
loader_mem = torch.utils.data.DataLoader(all_mem_patches, batch_size=2048, shuffle=False)
for s in SEEDS:
    model = encoders[s]
    lat_list = []
    with torch.no_grad():
        for bx in loader_mem:
            bx = bx.to(DEVICE)
            lat, _ = model(bx)
            lat_list.append(F.normalize(lat, p=2, dim=1))
    mem_latents_by_seed_gpu[s] = torch.cat(lat_list, dim=0)

mem_tickers_arr = np.array([r["ticker"] for r in mem_records_base])
mem_market_arr = np.array([r["market"] for r in mem_records_base])
mem_ret63 = np.array([r["future_return_63"] for r in mem_records_base], dtype=np.float32)
print(f"   [+] Indexed {len(mem_records_base)} causal episodes on GPU.")

def run_retrospective():
    print(f"\n[*] Starting 2021–2025 Multi-Regime Robustness Evaluation on: {DEVICE}")
    systems = ["P0", "P0*", "P1", "P2", "P4", "P6"]
    retrospective_rows = []
    
    for yr in EVAL_YEARS:
        print(f"\n>>> Running Evaluation for Calendar Year: {yr} <<<")
        start_dt = pd.Timestamp(f"{yr}-01-01")
        end_dt = pd.Timestamp(f"{yr}-12-31")
        
        for m in TICKERS_BY_MARKET.keys():
            tickers = TICKERS_BY_MARKET[m]
            dates_set = set()
            for tkr in tickers:
                if tkr in market_data[m]:
                    sub_y = market_data[m][tkr].loc[(market_data[m][tkr].index >= start_dt) & (market_data[m][tkr].index <= end_dt)]
                    dates_set.update(sub_y.index.tolist())
            calendar = sorted(list(dates_set))
            n_sessions = len(calendar)
            if n_sessions < 50:
                print(f"   [!] Skipping market {m} in {yr} (insufficient sessions: {n_sessions})")
                continue
                
            dates_str = [d.strftime("%Y-%m-%d") for d in calendar]
            n_ann = SESSIONS_PER_YEAR[m]
            
            price_open = np.full((n_sessions, len(tickers)), np.nan)
            price_close = np.full((n_sessions, len(tickers)), np.nan)
            atr_ratio_yr = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
            mom21_yr = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
            vol_yr = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
            patches_yr = np.zeros((n_sessions, len(tickers), N_PATCHES, 23), dtype=np.float32)
            
            for t_i, tkr in enumerate(tickers):
                if tkr in market_data[m]:
                    df_tkr = market_data[m][tkr]
                    feat_norm = standardize_features(df_tkr[FEATURE_NAMES_23].values, m)
                    for d_i, dt in enumerate(calendar):
                        match_dt = dt.strftime("%Y-%m-%d")
                        rows = df_tkr.loc[df_tkr.index.strftime("%Y-%m-%d") == match_dt]
                        if not rows.empty:
                            row = rows.iloc[0]
                            price_open[d_i, t_i] = float(row["open"]) if "open" in row and not np.isnan(row["open"]) else float(row["close"])
                            price_close[d_i, t_i] = float(row["close"])
                            atr_ratio_yr[d_i, t_i] = float(row["tech_atr_ratio_14"])
                            mom21_yr[d_i, t_i] = float(row["tech_momentum_21d"])
                            vol_yr[d_i, t_i] = float(row["tech_volatility_21d"]) if "tech_volatility_21d" in row and not np.isnan(row["tech_volatility_21d"]) else 0.015
                            
                            loc_idx = df_tkr.index.get_loc(rows.index[0])
                            if loc_idx >= T_WIN:
                                w_slice = feat_norm[loc_idx - T_WIN : loc_idx]
                                patches_yr[d_i, t_i] = w_slice.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)

            for t_i in range(len(tickers)):
                price_open[:, t_i] = pd.Series(price_open[:, t_i]).ffill().bfill().values
                price_close[:, t_i] = pd.Series(price_close[:, t_i]).ffill().bfill().values

            for s in SEEDS:
                model = encoders[s]
                mem_lat_gpu = mem_latents_by_seed_gpu[s]
                
                for sys_id in systems:
                    initial_capital = 100000.0
                    cash = initial_capital
                    portfolio_equity = np.zeros(n_sessions)
                    daily_returns = np.zeros(n_sessions)
                    
                    open_positions = []
                    cell_trades = []
                    pending_exits = []
                    pending_entries = []
                    
                    if sys_id == "P6":
                        cap_per_sec = initial_capital / len(tickers)
                        p6_h = []
                        for t_i in range(len(tickers)):
                            p_op = price_open[0, t_i]
                            sh = int((cap_per_sec * 0.999) / p_op)
                            if sh > 0:
                                f_in = sh * p_op * 0.0010
                                cash -= (sh * p_op + f_in)
                                p6_h.append({"t_i": t_i, "shares": sh, "entry_p": p_op, "entry_f": f_in})
                        for d_i in range(n_sessions):
                            tot_v = cash + sum(h["shares"] * price_close[d_i, h["t_i"]] for h in p6_h)
                            portfolio_equity[d_i] = tot_v
                            daily_returns[d_i] = (tot_v - initial_capital) / initial_capital if d_i == 0 else (tot_v - portfolio_equity[d_i - 1]) / portfolio_equity[d_i - 1]
                    else:
                        for day_idx in range(n_sessions):
                            cur_date_str = dates_str[day_idx]
                            
                            # 1. Fill pending exits at Open
                            for ex in pending_exits:
                                pos = ex["pos"]
                                t_i = pos["ticker_idx"]
                                ex_p = price_open[day_idx, t_i]
                                proc = pos["shares"] * ex_p * 0.9990
                                cash += proc
                                pnl = proc - (pos["entry_price"] * pos["shares"] + pos["entry_fee"])
                                cell_trades.append({"win": int(pnl > 0), "hold": day_idx - pos["entry_idx"]})
                                open_positions.remove(pos)
                            pending_exits = []
                            
                            # 2. Fill pending entries at Open
                            for ent in pending_entries:
                                t_i = ent["ticker_idx"]
                                op_p = price_open[day_idx, t_i]
                                if op_p > 0:
                                    sh = int((ent["capital"] * 0.95) / op_p)
                                    if sh > 0:
                                        fee = sh * op_p * 0.0010
                                        if cash >= (sh * op_p + fee):
                                            cash -= (sh * op_p + fee)
                                            open_positions.append({
                                                "ticker": tickers[t_i], "ticker_idx": t_i, "entry_idx": day_idx,
                                                "entry_price": op_p, "peak_price": op_p, "shares": sh, "entry_fee": fee
                                            })
                            pending_entries = []
                            
                            # 3. Mark-to-market at Close
                            for p in open_positions:
                                p["peak_price"] = max(p["peak_price"], price_close[day_idx, p["ticker_idx"]])
                            tot_eq = cash + sum(p["shares"] * price_close[day_idx, p["ticker_idx"]] for p in open_positions)
                            portfolio_equity[day_idx] = tot_eq
                            daily_returns[day_idx] = (tot_eq - initial_capital)/initial_capital if day_idx == 0 else (tot_eq - portfolio_equity[day_idx - 1]) / portfolio_equity[day_idx - 1]
                            
                            # 4. Generate signals at Close for tomorrow
                            if day_idx < n_sessions - 1:
                                for p in open_positions:
                                    h_d = day_idx - p["entry_idx"] + 1
                                    c_p = price_close[day_idx, p["ticker_idx"]]
                                    dd = (c_p - p["peak_price"]) / p["peak_price"]
                                    stop_d = max(0.10, 2.5 * atr_ratio_yr[day_idx, p["ticker_idx"]])
                                    if (dd < -stop_d and h_d >= 1) or (h_d >= 63):
                                        pending_exits.append({"pos": p})
                                        
                                ex_ids = {id(item["pos"]) for item in pending_exits}
                                remaining = len([p for p in open_positions if id(p) not in ex_ids])
                                slots = 3 - remaining
                                if slots > 0 and day_idx < n_sessions - 5:
                                    held_tkrs = {p["ticker"] for p in open_positions if id(p) not in ex_ids}
                                    avail = [i for i in range(len(tickers)) if tickers[i] not in held_tkrs]
                                    if avail:
                                        c_patches = torch.tensor(patches_yr[day_idx, avail], dtype=torch.float32, device=DEVICE)
                                        with torch.no_grad():
                                            q_lats, q_preds = model(c_patches)
                                            q_lats = F.normalize(q_lats, p=2, dim=1)
                                            q_preds_np = q_preds.squeeze(1).cpu().numpy()
                                            sim_mat = torch.matmul(mem_lat_gpu, q_lats.T).cpu().numpy()
                                            
                                        scores = np.zeros(len(avail))
                                        for idx_c, t_i in enumerate(avail):
                                            q_tkr = tickers[t_i]
                                            v = vol_yr[day_idx, t_i]
                                            pred_v = q_preds_np[idx_c]
                                            
                                            if sys_id in ("P0", "P0*", "P2"):
                                                s_col = sim_mat[:, idx_c]
                                                v_mask = (mem_tickers_arr != q_tkr) & (mem_market_arr == m) if sys_id == "P0*" else (mem_tickers_arr != q_tkr)
                                                v_idx = np.where(v_mask)[0]
                                                top25 = v_idx[np.argpartition(-s_col[v_idx], 25)[:25]]
                                                n_ret = mem_ret63[top25]
                                                n_sim = s_col[top25]
                                                if sys_id == "P0*":
                                                    nw = np.exp((n_sim - 1.0) / 0.08)
                                                    w = nw / (np.sum(nw) + 1e-9)
                                                else:
                                                    w = n_sim - np.min(n_sim) + 1e-4
                                                    w /= np.sum(w)
                                                mu_w = float(np.sum(w * n_ret))
                                                var05 = float(np.percentile(n_ret, 5))
                                                tail = n_ret[n_ret <= var05]
                                                cvar05 = float(np.mean(tail)) if len(tail) > 0 else var05
                                                scores[idx_c] = (pred_v + 0.8 * mu_w - 0.2 * abs(cvar05)) / (v + 1e-4) if sys_id in ("P0", "P0*") else (pred_v + 0.8 * mu_w) / (v + 1e-4)
                                            elif sys_id == "P1":
                                                scores[idx_c] = pred_v / (v + 1e-4)
                                            elif sys_id == "P4":
                                                scores[idx_c] = mom21_yr[day_idx, t_i]
                                                
                                        ranked = np.argsort(-scores)
                                        for c_i in ranked[:slots]:
                                            pending_entries.append({"ticker_idx": avail[c_i], "capital": tot_eq / 3.0})
                            else:
                                for p in list(open_positions):
                                    c_p = price_close[day_idx, p["ticker_idx"]]
                                    proc = p["shares"] * c_p * 0.9990
                                    pnl = proc - (p["entry_price"] * p["shares"] + p["entry_fee"])
                                    cell_trades.append({"win": int(pnl > 0), "hold": day_idx - p["entry_idx"]})
                                    open_positions.remove(p)

                    final_eq = portfolio_equity[-1]
                    tot_r = (final_eq - initial_capital) / initial_capital
                    mean_d = np.mean(daily_returns)
                    std_d = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-4
                    ann_r = (1.0 + tot_r) ** (n_ann / n_sessions) - 1.0
                    sh = np.sqrt(n_ann) * mean_d / (std_d + 1e-8)
                    
                    running_p = np.maximum.accumulate(portfolio_equity)
                    dd_arr = (portfolio_equity - running_p) / running_p
                    m_dd = float(np.min(dd_arr))
                    
                    retrospective_rows.append({
                        "year": yr,
                        "market": m,
                        "seed": s,
                        "system": sys_id,
                        "annualized_return": round(float(ann_r), 4),
                        "sharpe": round(float(sh), 3),
                        "max_drawdown": round(float(m_dd), 4),
                        "win_rate": round(float(np.mean([t["win"] for t in cell_trades])), 4) if cell_trades else 0.0,
                        "trades": len(cell_trades),
                    })

    df_retro = pd.DataFrame(retrospective_rows)
    df_retro.to_csv(OUTPUT_DIR / "v4_retrospective_2021_2025_cells.csv", index=False)
    
    # Aggregate by Year x System
    summary_yr = df_retro.groupby(["year", "system"])[["annualized_return", "sharpe", "max_drawdown"]].mean().reset_index()
    summary_yr["ann_return_pct"] = (summary_yr["annualized_return"] * 100).round(2)
    summary_yr["sharpe"] = summary_yr["sharpe"].round(3)
    summary_yr["max_dd_pct"] = (summary_yr["max_drawdown"] * 100).round(2)
    summary_yr.to_csv(OUTPUT_DIR / "v4_retrospective_2021_2025_summary.csv", index=False)
    
    print("\n" + "="*80)
    print("RETROSPECTIVE MULTI-YEAR ROBUSTNESS SUMMARY (2021–2025)")
    print("="*80)
    pivot_sh = df_retro.groupby(["year", "system"])["sharpe"].mean().unstack()
    print("\nSharpe Ratio Across Regimes (2021–2025):")
    print(pivot_sh[["P0", "P0*", "P1", "P2", "P4", "P6"]].round(3).to_string())
    
    pivot_ret = (df_retro.groupby(["year", "system"])["annualized_return"].mean().unstack() * 100).round(2)
    print("\nAnnual Return (%) Across Regimes (2021–2025):")
    print(pivot_ret[["P0", "P0*", "P1", "P2", "P4", "P6"]].to_string())

if __name__ == "__main__":
    run_retrospective()
