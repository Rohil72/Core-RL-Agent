"""Authoritative Corrected V4 Pipeline: Causal Historical Memory Across Six Markets
================================================================================
Contract Specifications & Architectural Corrections:
1. 252 x 23 historical observation window (Full annual calendar year).
2. PatchTST-style genuine temporal processing: 42 weekly patches (P=6 sessions).
3. 128-dimensional outcome-structured latent representation S^127 with joint Huber + metric geometry loss.
4. Causal memory bank strictly <= 2020-12-31 with mature 63/126-session horizons and ticker self-exclusion.
5. Strict causal execution timing: Signal at session t Close, Execution at session t+1 Open (10 bps fee).
6. Adaptive 2.5 x ATR_14 Chandelier exit (10% floor) with H_min = 1, H_max = 63 sessions.
7. True Expected Shortfall: CVaR_0.05 = E[R | R <= VaR_0.05].
8. P6 implemented as true passive equal-weight buy-and-hold context across the universe.
9. Deterministic seeding and cuDNN settings configured BEFORE model instantiation.
10. Panel-preserving moving-block bootstrap (B=10,000, L=21) testing Delta Sharpe with Benjamini-Hochberg FDR.
11. Comprehensive Query-Neighbor-Decision ledger for all executed trades.
12. Evaluates both Canonical P0 and Guardrailed P0* (Domestic Guardrail + Nadaraya-Watson kernel).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
DATA_DIR = PROJECT_ROOT / "data" / "cache" / "ohlcv"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "internal" / "evidence" / "research_defense_extract" / "v4_annual_252_evidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t_pipeline_start = time.time()
print(f"[*] Initializing Corrected V4 Pipeline on: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

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

# Contract observation parameters
T_WIN = 252        # Full annual observation window
PATCH_SIZE = 6     # 6 sessions per patch (approx 1.2 trading weeks)
N_PATCHES = 42     # 42 x 6 = 252 sessions exactly

FEATURE_NAMES_23 = [
    "tech_return_1d", "tech_momentum_3d", "tech_momentum_10d", "tech_momentum_21d", "tech_volatility_21d",
    "tech_volume_sma_21d", "tech_volume_ratio_21d", "tech_volume_change_1d", "tech_intraday_range_hl",
    "tech_drawdown_from_peak_21d", "tech_trend_slope_21d", "tech_close_vs_sma_50", "tech_close_vs_sma_150",
    "tech_close_vs_sma_200", "tech_sma_200_trend_20", "tech_pct_above_52w_low", "tech_pct_from_52w_high",
    "tech_up_down_volume_ratio_50", "tech_rsi_14", "tech_atr_ratio_14", "tech_macd_signal_diff",
    "tech_bollinger_bandwidth_20", "tech_historical_vol_ratio_63_21"
]

# ----------------------------------------------------------------------
# 1. DATA INGESTION, FEATURE ENGINEERING & CAUSAL STANDARDIZATION
# ----------------------------------------------------------------------

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

    # ATR 14 calculation
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
    
    # Dual forward opportunity horizons (quarterly and semi-annual)
    df["future_return_21"] = close.shift(-21) / close - 1.0
    df["future_return_63"] = close.shift(-63) / close - 1.0
    df["future_return_126"] = close.shift(-126) / close - 1.0
    
    # Forward path risk
    fwd_min_63 = close.rolling(window=63, min_periods=21).min().shift(-63)
    df["future_drawdown_63"] = ((fwd_min_63 - close) / close).fillna(-0.05)
    df["future_volatility_63"] = df["tech_return_1d"].rolling(window=63, min_periods=21).std().shift(-63).fillna(0.02)
    return df

print("1. Loading raw market data and computing features...", flush=True)
market_data = {}
all_tickers_global = []
ticker_to_id = {}

for m, tickers in TICKERS_BY_MARKET.items():
    market_data[m] = {}
    for tkr in tickers:
        p = DATA_DIR / f"{m}_{tkr}.parquet"
        if p.exists():
            market_data[m][tkr] = compute_23_features(pd.read_parquet(p))
            if tkr not in ticker_to_id:
                ticker_to_id[tkr] = len(all_tickers_global)
                all_tickers_global.append(tkr)

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

# 2024 Exchange Calendars
MARKET_CALENDARS_2024 = {}
for m in TICKERS_BY_MARKET.keys():
    dates_set = set()
    for tkr, df in market_data[m].items():
        sub_2024 = df.loc[(df.index >= pd.Timestamp("2024-01-01")) & (df.index <= pd.Timestamp("2024-12-31"))]
        dates_set.update(sub_2024.index.tolist())
    cal = sorted(list(dates_set))
    MARKET_CALENDARS_2024[m] = cal
    print(f"   [+] Market {m:6s}: {len(cal)} trading sessions in 2024 calendar.")

# ----------------------------------------------------------------------
# 2. TEMPORAL PATCH TRANSFORMER BACKBONE (T=42 Patches, 128-d Latent)
# ----------------------------------------------------------------------

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
        self.pool = nn.Linear(embed_dim, 1) # Attention pooling
        self.latent_head = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.outcome_head = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N_patches=42, F=23]
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

        h_trans = self.transformer(h) # [B, 42, 64]
        weights = torch.softmax(self.pool(h_trans), dim=1) # [B, 42, 1]
        h_pool = torch.sum(h_trans * weights, dim=1)       # [B, 64]
        
        latent = self.latent_head(h_pool)                  # [B, 128]
        pred_outcome = self.outcome_head(latent)           # [B, 1]
        return latent, pred_outcome

def continuous_outcome_geometry_loss(
    latents: torch.Tensor,
    targets: torch.Tensor,
    ticker_ids: torch.Tensor | None = None,
    tau_z: float = 0.5,
    tau_y: float = 0.1,
) -> torch.Tensor:
    """Supervised metric alignment with entity debiasing.
    Aligns latent distances on S^127 with outcome distances via KL divergence.
    """
    z_norm = F.normalize(latents, p=2, dim=1)
    diff_z = z_norm.unsqueeze(1) - z_norm.unsqueeze(0)
    D_z = (diff_z ** 2).sum(dim=-1) # [B, B]
    
    diff_y = targets.unsqueeze(1) - targets.unsqueeze(0)
    D_y = diff_y.abs().squeeze(-1)  # [B, B]
    
    B = latents.size(0)
    bias = torch.zeros((B, B), device=latents.device)
    bias.fill_diagonal_(-1e9)
    
    if ticker_ids is not None:
        same_ticker = (ticker_ids.unsqueeze(1) == ticker_ids.unsqueeze(0))
        bias = bias.masked_fill(same_ticker, -1e9)
        
    log_P_z = F.log_softmax(-D_z / tau_z + bias, dim=-1)
    P_y = F.softmax(-D_y / tau_y + bias, dim=-1)
    
    valid_mask = (bias > -1e8)
    P_y = torch.where(valid_mask, P_y, torch.zeros_like(P_y))
    row_sum = P_y.sum(dim=-1, keepdim=True)
    valid_rows = (row_sum.squeeze(-1) > 0)
    
    P_y = P_y / (row_sum + 1e-9)
    kl_per_row = (P_y * (torch.log(P_y + 1e-9) - log_P_z)).sum(dim=-1)
    loss = torch.where(valid_rows, kl_per_row, torch.zeros_like(kl_per_row)).mean()
    return loss

# ----------------------------------------------------------------------
# 3. BUILD 252-SESSION ANNUAL TRAINING DATASET (<= 2020-12-31)
# ----------------------------------------------------------------------

print("2. Building 252-Session Annual Training Dataset (42 Patches x 6 Sessions <= 2020)...", flush=True)

train_X_patches = []
train_Y_targets = []
train_ticker_ids = []

for m in TICKERS_BY_MARKET.keys():
    for tkr, df in market_data[m].items():
        feat_raw = df[FEATURE_NAMES_23].values
        feat_norm = standardize_features(feat_raw, m)
        targets = df["future_return_63"].values
        dates = df.index
        n_len = len(df)
        t_id = ticker_to_id[tkr]

        for i in range(T_WIN, n_len - 63):
            obs_dt = dates[i]
            if obs_dt > pd.Timestamp("2020-12-31") or dates[i + 63] > pd.Timestamp("2020-12-31") or np.isnan(targets[i]):
                continue
            
            raw_win = feat_norm[i - T_WIN : i]
            patch_mat = raw_win.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)
            train_X_patches.append(patch_mat)
            train_Y_targets.append(targets[i])
            train_ticker_ids.append(t_id)

X_train_arr = np.array(train_X_patches, dtype=np.float32)
Y_train_arr = np.array(train_Y_targets, dtype=np.float32)
Tkr_train_arr = np.array(train_ticker_ids, dtype=np.int64)
print(f"   [+] Training sequences collected: {len(X_train_arr)} samples | Shape: {X_train_arr.shape}")

torch_X = torch.tensor(X_train_arr, dtype=torch.float32)
torch_Y = torch.tensor(Y_train_arr, dtype=torch.float32).unsqueeze(1)
torch_Tkr = torch.tensor(Tkr_train_arr, dtype=torch.int64)

# ----------------------------------------------------------------------
# 4. DETERMINISTIC TRAINING ACROSS SEEDS [7, 17, 37] WITH METRIC LOSS
# ----------------------------------------------------------------------

encoders: dict[int, nn.Module] = {}
print("\n3. Training Annual Patch Temporal Transformers with Continuous Metric Loss on CUDA...", flush=True)

EPOCHS = 10
BATCH_SIZE = 256
LAMBDA_GEOM = 0.35

for s in SEEDS:
    t_seed_start = time.time()
    
    # Fully deterministic setup BEFORE model creation
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = AnnualPatchTemporalTransformer(input_dim=23, embed_dim=64, num_heads=4, latent_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    dataset = torch.utils.data.TensorDataset(torch_X, torch_Y, torch_Tkr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_huber = 0.0
        total_geom = 0.0
        n_b = 0
        
        for bx, by, btkr in loader:
            bx, by, btkr = bx.to(DEVICE), by.to(DEVICE), btkr.to(DEVICE)
            optimizer.zero_grad()
            latents, pred = model(bx)
            
            loss_huber = F.smooth_l1_loss(pred, by)
            loss_geom = continuous_outcome_geometry_loss(latents, by, btkr, tau_z=0.5, tau_y=0.1)
            loss = loss_huber + LAMBDA_GEOM * loss_geom
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_huber += loss_huber.item()
            total_geom += loss_geom.item()
            n_b += 1
            
        scheduler.step()
        
    model.eval()
    
    # Verify temporal sensitivity
    sample_x = torch_X[:10].to(DEVICE)
    sample_rev = torch.flip(sample_x, dims=[1])
    with torch.no_grad():
        lat_fwd, _ = model(sample_x)
        lat_rev, _ = model(sample_rev)
    temp_diff = (lat_fwd - lat_rev).abs().max().item()
    assert temp_diff > 1e-4, f"Temporal attention collapsed on seed {s} (diff={temp_diff:.6f})"
    
    # Metric alignment diagnostic: Spearman rank correlation between latent distance & outcome distance
    with torch.no_grad():
        test_bx = torch_X[:500].to(DEVICE)
        test_by = torch_Y[:500].to(DEVICE)
        test_lat, _ = model(test_bx)
        test_lat = F.normalize(test_lat, p=2, dim=1)
        pair_dz = torch.cdist(test_lat, test_lat, p=2).cpu().numpy().flatten()
        pair_dy = torch.cdist(test_by, test_by, p=1).cpu().numpy().flatten()
        sp_corr, _ = spearmanr(pair_dz, pair_dy)
        
    encoders[s] = model
    ckpt_path = MODELS_DIR / f"v4_metric_transformer_seed_{s}.pt"
    torch.save(model.state_dict(), ckpt_path)
    
    elapsed = time.time() - t_seed_start
    print(f"   [+] Seed {s} Trained in {elapsed:.1f}s | Loss: {total_loss/n_b:.4f} (Huber: {total_huber/n_b:.4f}, Geom: {total_geom/n_b:.4f}) | Rank-Corr: {sp_corr:+.3f} | Temp Sens: {temp_diff:.4f} (Active)")

# ----------------------------------------------------------------------
# 5. INDEXING CAUSAL MEMORY BANK (252-Session Annual Regimes <= 2020)
# ----------------------------------------------------------------------

print("\n4. Indexing Causal Memory Bank (Mature Outcomes <= 2020-12-31) on CUDA...", flush=True)
t_mem_start = time.time()

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
        dates = df_train.index
        
        ret21 = df_train["future_return_21"].values
        ret63 = df_train["future_return_63"].values
        ret126 = df_train["future_return_126"].values
        dd63 = df_train["future_drawdown_63"].values
        vol63 = df_train["future_volatility_63"].values

        for i in range(T_WIN, n_len - 126):
            if dates[i + 126] > pd.Timestamp("2020-12-31") or np.isnan(ret63[i]):
                continue
                
            raw_win = feat_norm[i - T_WIN : i]
            patch_mat = raw_win.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)
            mem_patch_list.append(patch_mat)
            
            mem_records_base.append({
                "date": dates[i].strftime("%Y-%m-%d"),
                "market": m,
                "ticker": tkr,
                "raw_features": feat_norm[i - 1],
                "future_return_21": float(ret21[i]) if not np.isnan(ret21[i]) else 0.0,
                "future_return_63": float(ret63[i]) if not np.isnan(ret63[i]) else 0.0,
                "future_return_126": float(ret126[i]) if not np.isnan(ret126[i]) else 0.0,
                "future_drawdown_63": float(dd63[i]) if not np.isnan(dd63[i]) else -0.05,
                "future_volatility_63": float(vol63[i]) if not np.isnan(vol63[i]) else 0.02,
            })

all_mem_patches = torch.tensor(np.array(mem_patch_list, dtype=np.float32))
mem_latents_by_seed_gpu: dict[int, torch.Tensor] = {}

loader_mem = torch.utils.data.DataLoader(all_mem_patches, batch_size=2048, shuffle=False)
for s in SEEDS:
    model = encoders[s]
    model.eval()
    lat_list = []
    with torch.no_grad():
        for bx in loader_mem:
            bx = bx.to(DEVICE)
            lat, _ = model(bx)
            lat_list.append(F.normalize(lat, p=2, dim=1))
    mem_latents_by_seed_gpu[s] = torch.cat(lat_list, dim=0)

mem_tickers_arr = np.array([r["ticker"] for r in mem_records_base])
mem_market_arr = np.array([r["market"] for r in mem_records_base])
mem_raw_feats = np.array([r["raw_features"] for r in mem_records_base], dtype=np.float32)
mem_ret63 = np.array([r["future_return_63"] for r in mem_records_base], dtype=np.float32)

t_mem_elapsed = time.time() - t_mem_start
print(f"   [+] Indexed {len(mem_records_base)} causal annual episodes in {t_mem_elapsed:.1f}s on GPU.")

# ----------------------------------------------------------------------
# 6. MATCHED 126-CELL 2024 BACKTEST ENGINE (CAUSAL TIMING & ATR EXITS)
# ----------------------------------------------------------------------

SYSTEM_CONFIGS = {
    "P0": {"name": "Full learned-state distributional memory", "claim": "Reference"},
    "P0*": {"name": "Guardrailed memory (Domestic + Nadaraya-Watson)", "claim": "Exploratory / Guardrailed"},
    "P1": {"name": "No external memory", "claim": "H2 (No Memory)"},
    "P2": {"name": "Same-neighbour mean-only memory", "claim": "H3 (Mean-Only)"},
    "P3": {"name": "Raw-feature kNN memory", "claim": "H1/H2 Control"},
    "P4": {"name": "Momentum-21 ranking", "claim": "Momentum Baseline"},
    "P5": {"name": "Random ranking", "claim": "Random Baseline"},
    "P6": {"name": "Equal-weight buy-and-hold context", "claim": "Equal Weight Context"},
}

print("\n" + "="*85)
print("5. Running Matched 126-Cell Backtest Engine with Causal Timing (Signal t -> Fill t+1 Open)...")
print("="*85)

matrix_rows = []
all_equity_curves = []
all_trade_ledgers = []
all_decision_ledgers = []

for m_idx, (m, calendar) in enumerate(MARKET_CALENDARS_2024.items()):
    tickers = TICKERS_BY_MARKET[m]
    n_sessions = len(calendar)
    dates_str = [d.strftime("%Y-%m-%d") for d in calendar]
    n_ann_sessions = SESSIONS_PER_YEAR[m]

    price_open = np.full((n_sessions, len(tickers)), np.nan)
    price_close = np.full((n_sessions, len(tickers)), np.nan)
    atr_ratio_2024 = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
    mom21_2024 = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
    vol_2024 = np.zeros((n_sessions, len(tickers)), dtype=np.float32)
    patches_2024 = np.zeros((n_sessions, len(tickers), N_PATCHES, 23), dtype=np.float32)
    raw_feats_2024 = np.zeros((n_sessions, len(tickers), 23), dtype=np.float32)

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
                    atr_ratio_2024[d_i, t_i] = float(row["tech_atr_ratio_14"])
                    mom21_2024[d_i, t_i] = float(row["tech_momentum_21d"])
                    vol_2024[d_i, t_i] = float(row["tech_volatility_21d"]) if "tech_volatility_21d" in row and not np.isnan(row["tech_volatility_21d"]) else 0.015
                    
                    loc_idx = df_tkr.index.get_loc(rows.index[0])
                    raw_feats_2024[d_i, t_i] = feat_norm[loc_idx]
                    if loc_idx >= T_WIN:
                        w_slice = feat_norm[loc_idx - T_WIN : loc_idx]
                        patches_2024[d_i, t_i] = w_slice.reshape(N_PATCHES, PATCH_SIZE, 23).mean(axis=1)

    for t_i in range(len(tickers)):
        price_open[:, t_i] = pd.Series(price_open[:, t_i]).ffill().bfill().values
        price_close[:, t_i] = pd.Series(price_close[:, t_i]).ffill().bfill().values

    for s in SEEDS:
        model = encoders[s]
        mem_lat_gpu = mem_latents_by_seed_gpu[s]
        rng_sim = np.random.default_rng(s * 10000 + m_idx * 100 + 42)

        for sys_id, sys_cfg in SYSTEM_CONFIGS.items():
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

            # ----------------------------------------------------------
            # P6: PASSIVE EQUAL-WEIGHT BUY-AND-HOLD CONTEXT (FULL UNIVERSE)
            # ----------------------------------------------------------
            if sys_id == "P6":
                cap_per_sec = initial_capital / len(tickers)
                p6_holdings = []
                for t_i in range(len(tickers)):
                    p_op = price_open[0, t_i]
                    sh = int((cap_per_sec * 0.999) / p_op)
                    if sh > 0:
                        f_entry = sh * p_op * 0.0010
                        cash -= (sh * p_op + f_entry)
                        p6_holdings.append({"t_i": t_i, "shares": sh, "entry_p": p_op, "entry_f": f_entry})
                
                for d_i in range(n_sessions):
                    tot_v = cash + sum(h["shares"] * price_close[d_i, h["t_i"]] for h in p6_holdings)
                    portfolio_equity[d_i] = tot_v
                    daily_returns[d_i] = (tot_v - initial_capital) / initial_capital if d_i == 0 else (tot_v - portfolio_equity[d_i - 1]) / portfolio_equity[d_i - 1]
                
                # Final day liquidation trade records
                for h in p6_holdings:
                    c_p = price_close[-1, h["t_i"]]
                    f_exit = h["shares"] * c_p * 0.0010
                    pnl = (c_p * h["shares"] - f_exit) - (h["entry_p"] * h["shares"] + h["entry_f"])
                    trade_seq += 1
                    cell_trades.append({
                        "trade_id": f"TRD_{m}_{s}_{sys_id}_{trade_seq:03d}",
                        "market": m,
                        "seed": s,
                        "system": sys_id,
                        "ticker": tickers[h["t_i"]],
                        "signal_date": dates_str[0],
                        "entry_date": dates_str[0],
                        "exit_date": dates_str[-1],
                        "holding_days": n_sessions,
                        "exit_reason": "passive_hold",
                        "entry_price": round(float(h["entry_p"]), 2),
                        "exit_price": round(float(c_p), 2),
                        "shares": int(h["shares"]),
                        "entry_fee": round(float(h["entry_f"]), 2),
                        "exit_fee": round(float(f_exit), 2),
                        "gross_pnl": round(float((c_p - h["entry_p"]) * h["shares"]), 2),
                        "realized_pnl": round(float(pnl), 2),
                        "return_pct": round(float(pnl / (h["entry_p"] * h["shares"] + h["entry_f"])), 4),
                        "win": int(pnl > 0),
                    })

            # ----------------------------------------------------------
            # ACTIVE SYSTEMS P0–P5: STRICT CAUSAL EXECUTION & ATR CHANDELIER
            # ----------------------------------------------------------
            else:
                for day_idx in range(n_sessions):
                    cur_date_str = dates_str[day_idx]

                    # 1. EXECUTE PENDING FILLS AT TODAY'S OPEN (t Open)
                    # Exits executed first to free cash
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
                        cell_trades.append({
                            "trade_id": pos.get("trade_id", f"TRD_{m}_{s}_{sys_id}_{trade_seq:03d}"),
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
                        })
                        open_positions.remove(pos)
                    pending_exits = []

                    # Entries executed at today's open
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
                                    # Record entry decision ledger
                                    all_decision_ledgers.append({
                                        "trade_id": trade_id,
                                        "market": m,
                                        "seed": s,
                                        "system": sys_id,
                                        "ticker": tickers[t_i],
                                        "signal_date": ent["signal_date"],
                                        "execution_date": cur_date_str,
                                        "pred_utility": round(float(ent.get("pred_utility", 0.0)), 4),
                                        "neighbor_mu": round(float(ent.get("neighbor_mu", 0.0)), 4),
                                        "neighbor_cvar": round(float(ent.get("neighbor_cvar", 0.0)), 4),
                                        "decision_score": round(float(ent.get("decision_score", 0.0)), 4),
                                        "top_neighbors": ent.get("top_neighbors", ""),
                                        "neighbor_weights": ent.get("neighbor_weights", ""),
                                    })
                    pending_entries = []

                    # 2. MARK-TO-MARKET AT TODAY'S CLOSE (t Close)
                    pos_value = sum(p["shares"] * price_close[day_idx, p["ticker_idx"]] for p in open_positions)
                    for p in open_positions:
                        cur_p = price_close[day_idx, p["ticker_idx"]]
                        p["peak_price"] = max(p["peak_price"], cur_p)

                    tot_equity = cash + pos_value
                    portfolio_equity[day_idx] = tot_equity
                    daily_returns[day_idx] = (tot_equity - initial_capital) / initial_capital if day_idx == 0 else (tot_equity - portfolio_equity[day_idx - 1]) / portfolio_equity[day_idx - 1]

                    # 3. GENERATE SIGNALS AT TODAY'S CLOSE FOR TOMORROW'S OPEN (t -> t+1)
                    if day_idx < n_sessions - 1:
                        # A. Evaluate Exits for Open Positions
                        for p in open_positions:
                            held_days = day_idx - p["entry_idx"] + 1
                            cur_p = price_close[day_idx, p["ticker_idx"]]
                            dd_peak = (cur_p - p["peak_price"]) / p["peak_price"]
                            
                            # Adaptive 2.5 x ATR_14 Chandelier Exit Contract (10% floor)
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

                        # B. Calculate Open Slots for Tomorrow Open
                        exiting_pos_ids = {id(item["pos"]) for item in pending_exits}
                        remaining_count = len([p for p in open_positions if id(p) not in exiting_pos_ids])
                        slots_open = 3 - remaining_count
                        
                        if slots_open > 0 and day_idx < n_sessions - 5:
                            cap_per_slot = tot_equity / 3.0
                            held_tickers = {p["ticker"] for p in open_positions if id(p) not in exiting_pos_ids}
                            available_indices = [
                                i for i in range(len(tickers))
                                if tickers[i] not in held_tickers
                            ]
                            
                            if available_indices:
                                scores = np.zeros(len(available_indices))
                                decision_details: list[dict[str, Any]] = []
                                
                                cand_patches = torch.tensor(patches_2024[day_idx, available_indices], dtype=torch.float32, device=DEVICE)
                                with torch.no_grad():
                                    q_lats, q_preds = model(cand_patches)
                                    q_lats = F.normalize(q_lats, p=2, dim=1)
                                    q_preds_np = q_preds.squeeze(1).cpu().numpy()
                                    sim_matrix = torch.matmul(mem_lat_gpu, q_lats.T).cpu().numpy()

                                for idx_c, t_i in enumerate(available_indices):
                                    q_tkr = tickers[t_i]
                                    pred_val = q_preds_np[idx_c]
                                    v = vol_2024[day_idx, t_i]

                                    if sys_id in ("P0", "P0*", "P2"):
                                        sim_col = sim_matrix[:, idx_c]
                                        
                                        if sys_id == "P0*":
                                            # Domestic guardrail: strictly query from same domestic market
                                            valid_mask = (mem_tickers_arr != q_tkr) & (mem_market_arr == m)
                                        else:
                                            # Global unconstrained retrieval
                                            valid_mask = (mem_tickers_arr != q_tkr)
                                            
                                        valid_indices = np.where(valid_mask)[0]
                                        top_25_sub = np.argpartition(-sim_col[valid_indices], 25)[:25]
                                        valid_nbrs = valid_indices[top_25_sub]
                                        
                                        nbr_sims = sim_col[valid_nbrs]
                                        nbr_ret = mem_ret63[valid_nbrs]
                                        
                                        if sys_id == "P0*":
                                            # Nadaraya-Watson distance kernel (tau=0.08)
                                            nw_exp = np.exp((nbr_sims - 1.0) / 0.08)
                                            weights = nw_exp / (np.sum(nw_exp) + 1e-9)
                                        else:
                                            # Linear min-subtracted weights
                                            weights = nbr_sims - np.min(nbr_sims) + 1e-4
                                            weights /= np.sum(weights)
                                            
                                        mu_w = float(np.sum(weights * nbr_ret))
                                        
                                        # True Expected Shortfall (CVaR_0.05)
                                        var_05 = float(np.percentile(nbr_ret, 5))
                                        tail_ret = nbr_ret[nbr_ret <= var_05]
                                        cvar_05 = float(np.mean(tail_ret)) if len(tail_ret) > 0 else var_05

                                        if sys_id in ("P0", "P0*"):
                                            final_sc = (pred_val + 0.8 * mu_w - 0.2 * abs(cvar_05)) / (v + 1e-4)
                                        else: # P2 mean-only
                                            final_sc = (pred_val + 0.8 * mu_w) / (v + 1e-4)
                                            
                                        scores[idx_c] = final_sc
                                        
                                        top5_idx = np.argsort(-weights)[:5]
                                        top_nbr_str = ",".join([f"{mem_market_arr[valid_nbrs[n]]}:{mem_tickers_arr[valid_nbrs[n]]}" for n in top5_idx])
                                        top_wt_str = ",".join([f"{weights[n]:.3f}" for n in top5_idx])
                                        
                                        decision_details.append({
                                            "pred_utility": pred_val,
                                            "neighbor_mu": mu_w,
                                            "neighbor_cvar": cvar_05,
                                            "decision_score": final_sc,
                                            "top_neighbors": top_nbr_str,
                                            "neighbor_weights": top_wt_str,
                                        })

                                    elif sys_id == "P1":
                                        sc = pred_val / (v + 1e-4)
                                        scores[idx_c] = sc
                                        decision_details.append({
                                            "pred_utility": pred_val,
                                            "neighbor_mu": 0.0,
                                            "neighbor_cvar": 0.0,
                                            "decision_score": sc,
                                            "top_neighbors": "none",
                                            "neighbor_weights": "none",
                                        })

                                    elif sys_id == "P3":
                                        q_raw = raw_feats_2024[day_idx, t_i]
                                        dist = np.linalg.norm(mem_raw_feats - q_raw, axis=1)
                                        valid_mask = (mem_tickers_arr != q_tkr)
                                        valid_indices = np.where(valid_mask)[0]
                                        top_25_sub = np.argpartition(dist[valid_indices], 25)[:25]
                                        valid_nbrs = valid_indices[top_25_sub]
                                        mu_raw = float(np.mean(mem_ret63[valid_nbrs]))
                                        sc = (pred_val + 0.8 * mu_raw) / (v + 1e-4)
                                        scores[idx_c] = sc
                                        decision_details.append({
                                            "pred_utility": pred_val,
                                            "neighbor_mu": mu_raw,
                                            "neighbor_cvar": 0.0,
                                            "decision_score": sc,
                                            "top_neighbors": "raw_knn",
                                            "neighbor_weights": "uniform",
                                        })

                                    elif sys_id == "P4":
                                        sc = mom21_2024[day_idx, t_i]
                                        scores[idx_c] = sc
                                        decision_details.append({
                                            "pred_utility": sc,
                                            "neighbor_mu": 0.0,
                                            "neighbor_cvar": 0.0,
                                            "decision_score": sc,
                                            "top_neighbors": "momentum_rank",
                                            "neighbor_weights": "none",
                                        })

                                    elif sys_id == "P5":
                                        sc = float(rng_sim.uniform(-1.0, 1.0))
                                        scores[idx_c] = sc
                                        decision_details.append({
                                            "pred_utility": sc,
                                            "neighbor_mu": 0.0,
                                            "neighbor_cvar": 0.0,
                                            "decision_score": sc,
                                            "top_neighbors": "random_null",
                                            "neighbor_weights": "none",
                                        })

                                ranked_cands = np.argsort(-scores)
                                selected_cands = [available_indices[c] for c in ranked_cands[:slots_open]]
                                
                                for c_rank, t_i in enumerate(selected_cands):
                                    d_info = decision_details[ranked_cands[c_rank]]
                                    pending_entries.append({
                                        "ticker_idx": t_i,
                                        "allocated_capital": cap_per_slot,
                                        "signal_date": cur_date_str,
                                        **d_info
                                    })
                    else:
                        # Final session close: Liquidate all remaining positions
                        for p in list(open_positions):
                            c_p = price_close[day_idx, p["ticker_idx"]]
                            f_exit = p["shares"] * c_p * 0.0010
                            tot_cost = p["entry_price"] * p["shares"] + p["entry_fee"]
                            realized_pnl = (c_p * p["shares"] - f_exit) - tot_cost
                            held_days = day_idx - p["entry_idx"]
                            trade_seq += 1
                            cell_trades.append({
                                "trade_id": p.get("trade_id", f"TRD_{m}_{s}_{sys_id}_{trade_seq:03d}"),
                                "market": m,
                                "seed": s,
                                "system": sys_id,
                                "ticker": p["ticker"],
                                "signal_date": cur_date_str,
                                "entry_date": dates_str[p["entry_idx"]],
                                "exit_date": cur_date_str,
                                "holding_days": int(held_days),
                                "exit_reason": "calendar_end",
                                "entry_price": round(float(p["entry_price"]), 2),
                                "exit_price": round(float(c_p), 2),
                                "shares": int(p["shares"]),
                                "entry_fee": round(float(p["entry_fee"]), 2),
                                "exit_fee": round(float(f_exit), 2),
                                "gross_pnl": round(float((c_p - p["entry_price"]) * p["shares"]), 2),
                                "realized_pnl": round(float(realized_pnl), 2),
                                "return_pct": round(float(realized_pnl / tot_cost), 4),
                                "win": int(realized_pnl > 0),
                            })
                            open_positions.remove(p)

            # ----------------------------------------------------------
            # CELL-LEVEL PERFORMANCE METRICS COMPUTATION
            # ----------------------------------------------------------
            final_eq = portfolio_equity[-1]
            tot_ret = (final_eq - initial_capital) / initial_capital
            mean_d_ret = np.mean(daily_returns)
            std_d_ret = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-4
            ann_ret = (1.0 + tot_ret) ** (n_ann_sessions / n_sessions) - 1.0
            sharpe = np.sqrt(n_ann_sessions) * mean_d_ret / (std_d_ret + 1e-8)
            
            neg_returns = daily_returns[daily_returns < 0]
            down_std = np.std(neg_returns, ddof=1) if len(neg_returns) > 1 else 1e-4
            sortino = np.sqrt(n_ann_sessions) * mean_d_ret / (down_std + 1e-8)
            
            running_peak = np.maximum.accumulate(portfolio_equity)
            drawdowns = (portfolio_equity - running_peak) / running_peak
            max_dd = float(np.min(drawdowns))
            
            win_rate = float(np.mean([t["win"] for t in cell_trades])) if cell_trades else 0.0
            median_hold = float(np.median([t["holding_days"] for t in cell_trades])) if cell_trades else 0.0
            mean_hold = float(np.mean([t["holding_days"] for t in cell_trades])) if cell_trades else 0.0

            matrix_rows.append({
                "market": m,
                "seed": s,
                "system": sys_id,
                "system_name": sys_cfg["name"],
                "claim": sys_cfg["claim"],
                "total_return": round(float(tot_ret), 4),
                "annualized_return": round(float(ann_ret), 4),
                "sharpe": round(float(sharpe), 3),
                "sortino": round(float(sortino), 3),
                "max_drawdown": round(float(max_dd), 4),
                "win_rate": round(float(win_rate), 4),
                "cell_median_hold": round(float(median_hold), 1),
                "cell_mean_hold": round(float(mean_hold), 1),
                "trade_count": len(cell_trades),
                "final_equity": round(float(final_eq), 2),
            })

            for d_i in range(n_sessions):
                all_equity_curves.append({
                    "date": dates_str[d_i],
                    "market": m,
                    "seed": s,
                    "system": sys_id,
                    "equity": round(float(portfolio_equity[d_i]), 2),
                    "daily_return": round(float(daily_returns[d_i]), 6),
                })
            all_trade_ledgers.extend(cell_trades)

df_matrix = pd.DataFrame(matrix_rows)
df_curves = pd.DataFrame(all_equity_curves)
df_trades = pd.DataFrame(all_trade_ledgers)
df_decisions = pd.DataFrame(all_decision_ledgers)

# ----------------------------------------------------------------------
# 7. SUMMARY REPORTING: POOLED TRADE METRICS VS CROSS-CELL AVERAGES
# ----------------------------------------------------------------------

print("\n" + "="*95)
print("AUDITABLE 126-CELL SCORECARD (CORRECTED TIMING, ATR CHANDELIER, METRIC S^127)")
print("="*95)

summary_table = []
for sys_id, sys_cfg in SYSTEM_CONFIGS.items():
    sub_m = df_matrix[df_matrix["system"] == sys_id]
    sub_t = df_trades[df_trades["system"] == sys_id]
    
    # Exact pooled trade statistics across all executions
    pooled_med_hold = float(sub_t["holding_days"].median()) if not sub_t.empty else 0.0
    pooled_mean_hold = float(sub_t["holding_days"].mean()) if not sub_t.empty else 0.0
    pooled_win_rate = float(sub_t["win"].mean()) if not sub_t.empty else 0.0
    
    # Profit factor: gross wins / gross losses
    gross_wins = sub_t.loc[sub_t["realized_pnl"] > 0, "realized_pnl"].sum()
    gross_losses = abs(sub_t.loc[sub_t["realized_pnl"] < 0, "realized_pnl"].sum())
    profit_factor = round(float(gross_wins / (gross_losses + 1e-9)), 2)

    summary_table.append({
        "System": sys_id,
        "System Name": sys_cfg["name"],
        "Ann. Return": f"{sub_m['annualized_return'].mean()*100:+.2f}%",
        "Sharpe": f"{sub_m['sharpe'].mean():.3f}",
        "Sortino": f"{sub_m['sortino'].mean():.3f}",
        "Max Drawdown": f"{sub_m['max_drawdown'].mean()*100:.2f}%",
        "Pooled Win%": f"{pooled_win_rate*100:.1f}%",
        "Pooled Med Hold": f"{pooled_med_hold:.1f}d",
        "Pooled Mean Hold": f"{pooled_mean_hold:.1f}d",
        "Profit Factor": profit_factor,
        "Trade Count": len(sub_t),
    })

df_summary = pd.DataFrame(summary_table)
print(df_summary.to_string(index=False))

print("\n" + "="*95)
print("PER-MARKET SHARPE RATIO COMPARISON (P0 vs P0* vs P1 vs P2 vs P4)")
print("="*95)
mkt_sharpe = df_matrix.groupby(["market", "system"])["sharpe"].mean().unstack()
cols_to_print = [c for c in ["P0", "P0*", "P1", "P2", "P3", "P4", "P6"] if c in mkt_sharpe.columns]
print(mkt_sharpe[cols_to_print].round(3).to_string())

print("\n" + "="*95)
print("PER-MARKET ANNUALIZED RETURN (%) COMPARISON (P0 vs P0* vs P1 vs P2 vs P4)")
print("="*95)
mkt_ret = df_matrix.groupby(["market", "system"])["annualized_return"].mean().unstack() * 100
print(mkt_ret[cols_to_print].round(2).to_string())

# ----------------------------------------------------------------------
# 8. PANEL-PRESERVING MOVING-BLOCK BOOTSTRAP (B=10,000, L=21)
# ----------------------------------------------------------------------

print("\n" + "="*95)
print("6. Computing Panel-Preserving Moving-Block Bootstrap for Delta Sharpe (B=10,000, L=21)...")
print("="*95)

def run_panel_bootstrap(target_sys: str, comparator_sys: str, B: int = 10000, L: int = 21) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    boot_delta_sharpes = []
    
    # Extract synchronized cell return series
    cells = []
    for m in MARKET_CALENDARS_2024.keys():
        for s in SEEDS:
            sub_tgt = df_curves[(df_curves["market"] == m) & (df_curves["seed"] == s) & (df_curves["system"] == target_sys)].sort_values("date")
            sub_cmp = df_curves[(df_curves["market"] == m) & (df_curves["seed"] == s) & (df_curves["system"] == comparator_sys)].sort_values("date")
            
            r_tgt = sub_tgt["daily_return"].values
            r_cmp = sub_cmp["daily_return"].values
            n_t = min(len(r_tgt), len(r_cmp))
            cells.append((r_tgt[:n_t], r_cmp[:n_t], SESSIONS_PER_YEAR[m]))

    # Actual sample delta Sharpe
    all_tgt = np.concatenate([c[0] for c in cells])
    all_cmp = np.concatenate([c[1] for c in cells])
    actual_delta_sh = float((np.mean(all_tgt) / (np.std(all_tgt, ddof=1) + 1e-8) - np.mean(all_cmp) / (np.std(all_cmp, ddof=1) + 1e-8)) * np.sqrt(252.0))

    # Resample strictly WITHIN each cell timeline
    for _ in range(B):
        resamp_tgt = []
        resamp_cmp = []
        
        for r_tgt, r_cmp, ann_f in cells:
            n_len = len(r_tgt)
            n_blocks = int(np.ceil(n_len / L))
            starts = rng.integers(0, max(1, n_len - L + 1), size=n_blocks)
            sampled_idx = np.concatenate([np.arange(st, min(st + L, n_len)) for st in starts])[:n_len]
            
            resamp_tgt.append(r_tgt[sampled_idx])
            resamp_cmp.append(r_cmp[sampled_idx])
            
        boot_tgt = np.concatenate(resamp_tgt)
        boot_cmp = np.concatenate(resamp_cmp)
        
        sh_t = np.mean(boot_tgt) / (np.std(boot_tgt, ddof=1) + 1e-8) * np.sqrt(252.0)
        sh_c = np.mean(boot_cmp) / (np.std(boot_cmp, ddof=1) + 1e-8) * np.sqrt(252.0)
        boot_delta_sharpes.append(sh_t - sh_c)

    boot_arr = np.array(boot_delta_sharpes)
    p_two_sided = float(2.0 * min(np.mean(boot_arr <= 0), np.mean(boot_arr >= 0)))
    p_two_sided = min(p_two_sided, 1.0)
    
    ci_lower = float(np.percentile(boot_arr, 2.5))
    ci_upper = float(np.percentile(boot_arr, 97.5))
    
    return {
        "actual_delta_sharpe": round(actual_delta_sh, 4),
        "ci_95": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
        "p_value": p_two_sided,
    }

bootstrap_results = []
comparators = [
    ("P0", "P1", "P0 vs P1 (H2: No Memory)"),
    ("P0", "P2", "P0 vs P2 (H3: Mean-Only)"),
    ("P0", "P3", "P0 vs P3 (H1/H2 Control)"),
    ("P0", "P4", "P0 vs P4 (Momentum Baseline)"),
    ("P0", "P5", "P0 vs P5 (Random Ranking)"),
    ("P0", "P6", "P0 vs P6 (Equal Weight Context)"),
    ("P0*", "P1", "P0* (Guardrail) vs P1 (No Memory)"),
    ("P0*", "P0", "P0* (Guardrail) vs P0 (Global Memory)"),
]

raw_p_vals = []
for tgt_s, cmp_s, label in comparators:
    res = run_panel_bootstrap(tgt_s, cmp_s, B=10000, L=21)
    raw_p_vals.append(res["p_value"])
    bootstrap_results.append({
        "Comparison": label,
        "Delta Sharpe": res["actual_delta_sharpe"],
        "95% CI": res["ci_95"],
        "p_value": res["p_value"],
    })

# Benjamini-Hochberg FDR
m_tests = len(raw_p_vals)
sorted_idx = np.argsort(raw_p_vals)
q_values = np.zeros(m_tests)
for rank, idx in enumerate(sorted_idx, 1):
    q_values[idx] = min(raw_p_vals[idx] * m_tests / rank, 1.0)

for i in range(m_tests - 2, -1, -1):
    q_values[sorted_idx[i]] = min(q_values[sorted_idx[i]], q_values[sorted_idx[i+1]])

for i in range(len(bootstrap_results)):
    bootstrap_results[i]["FDR q_value"] = round(float(q_values[i]), 4)
    bootstrap_results[i]["Verdict"] = "Reject H0 (p < 0.05)" if q_values[i] < 0.05 else "Fail to Reject H0"

df_bootstrap = pd.DataFrame(bootstrap_results)
print(df_bootstrap.to_string(index=False))

# ----------------------------------------------------------------------
# 9. EXPORT COMPLETE EVIDENCE REPOSITORY
# ----------------------------------------------------------------------

df_matrix.to_csv(OUTPUT_DIR / "v4_primary_systems_126_cell_matrix.csv", index=False)
df_curves.to_csv(OUTPUT_DIR / "v4_daily_equity_curves_p0_p6.csv", index=False)
df_trades.to_csv(OUTPUT_DIR / "v4_trade_ledgers_p0_p6.csv", index=False)
df_decisions.to_csv(OUTPUT_DIR / "v4_query_neighbor_decision_ledger.csv", index=False)
df_bootstrap.to_csv(OUTPUT_DIR / "v4_statistical_bootstrap.csv", index=False)

# Environment and configuration metadata
env_meta = {
    "execution_time_seconds": round(time.time() - t_pipeline_start, 1),
    "python_version": sys.version,
    "torch_version": torch.__version__,
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "platform": platform.platform(),
    "contract": {
        "observation_window": "252 sessions x 23 features",
        "patch_architecture": "42 patches x 6 sessions",
        "latent_dimension": 128,
        "loss_function": "SmoothL1Loss(pred, y) + 0.35 * continuous_outcome_geometry_loss(latents, y)",
        "holding_period_contract": "H_min = 1 session, H_max = 63 sessions",
        "exit_mechanism": "Adaptive 2.5 x ATR_14 Chandelier exit (10% floor)",
        "causal_timing": "Signal at session t Close, Execution at session t+1 Open (10 bps fee)",
        "cvar_definition": "Expected Shortfall E[R | R <= VaR_0.05]",
        "training_cutoff": "<= 2020-12-31",
        "evaluation_year": 2024,
    },
    "scorecard_summary": summary_table,
    "bootstrap_results": bootstrap_results,
}

with open(OUTPUT_DIR / "v4_experiment_metadata.json", "w") as f:
    json.dump(env_meta, f, indent=2)

print(f"\n[+] Corrected V4 Pipeline successfully executed and sealed in: {OUTPUT_DIR}")
