"""
Cache Builder for Memory-Centric Equity Selection.

Builds and persists:
1. Reusable Query Cache (2021-2024)
2. Sealed Memory Bank Cache (<= 2020-12-31)
3. Future Realized Outcomes Table

Saves arrays in .npy and metadata in .parquet / .csv.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from memory_study.backbones import (
    AnnualPatchTemporalTransformer,
    MLPEncoder,
    load_transformer_checkpoint,
    load_mlp_checkpoint,
    ensure_mlp_checkpoints,
)

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
DATA_DIR = PROJECT_ROOT / "FINAL_SUBMISSION_PACKAGE" / "data" / "cache" / "ohlcv"
CACHE_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "cache"
CONFIG_PATH = PROJECT_ROOT / "memory_study" / "configs" / "common.json"

FEATURE_NAMES_23 = [
    "tech_return_1d", "tech_momentum_3d", "tech_momentum_10d", "tech_momentum_21d", "tech_volatility_21d",
    "tech_volume_sma_21d", "tech_volume_ratio_21d", "tech_volume_change_1d", "tech_intraday_range_hl",
    "tech_drawdown_from_peak_21d", "tech_trend_slope_21d", "tech_close_vs_sma_50", "tech_close_vs_sma_150",
    "tech_close_vs_sma_200", "tech_sma_200_trend_20", "tech_pct_above_52w_low", "tech_pct_from_52w_high",
    "tech_up_down_volume_ratio_50", "tech_rsi_14", "tech_atr_ratio_14", "tech_macd_signal_diff",
    "tech_bollinger_bandwidth_20", "tech_historical_vol_ratio_63_21"
]


def compute_23_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Computes the 23 technical features and target labels."""
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

    df["future_return_21"] = close.shift(-21) / close - 1.0
    df["future_return_63"] = close.shift(-63) / close - 1.0
    fwd_min_63 = close.rolling(window=63, min_periods=21).min().shift(-63)
    df["future_drawdown_63"] = ((fwd_min_63 - close) / close).fillna(-0.05)
    return df


def build_caches(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Constructs and persists all query, memory, and outcome caches.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    markets = cfg["markets"]
    backbone_seeds = cfg["backbone_seeds"]
    t_win = cfg["window_sessions"]
    patch_size = cfg["patch_size"]
    n_patches = cfg["n_patches"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check if already built
    mem_meta_p = CACHE_DIR / "memory_meta.parquet"
    query_meta_p = CACHE_DIR / "query_meta.parquet"
    if not force_rebuild and mem_meta_p.exists() and query_meta_p.exists():
        print(f"[+] Reusing existing caches from {CACHE_DIR}")
        return {"cache_dir": str(CACHE_DIR), "reused": True}

    print("=" * 80)
    print(f"BUILDING CACHES FOR MEMORY-CENTRIC EQUITY SELECTION (Device: {device})")
    print("=" * 80)

    # 1. Ingest market data
    print("\n[Step 1] Ingesting sovereign parquet files and computing 23 features...")
    t0 = time.time()
    market_data = {}
    for m, tickers in markets.items():
        market_data[m] = {}
        for tkr in tickers:
            p = DATA_DIR / f"{m}_{tkr}.parquet"
            if p.exists():
                market_data[m][tkr] = compute_23_features(pd.read_parquet(p))
            else:
                print(f"   [!] Missing parquet: {p}")
    print(f"   [+] Ingested data for {sum(len(v) for v in market_data.values())} tickers in {time.time() - t0:.2f}s.")

    # 2. Compute pre-2021 feature scalers
    print("\n[Step 2] Computing pre-2021 per-market feature scalers...")
    scaler_params = {}
    for m in markets.keys():
        train_dfs = [df.loc[df.index <= pd.Timestamp("2020-12-31"), FEATURE_NAMES_23].dropna() for tkr, df in market_data[m].items()]
        train_dfs = [d for d in train_dfs if not d.empty]
        if train_dfs:
            all_train = pd.concat(train_dfs, axis=0)
            scaler_params[m] = {
                f: {"mean": float(np.mean(all_train[f].values)), "std": float(max(np.std(all_train[f].values), 1e-4))}
                for f in FEATURE_NAMES_23
            }

    def standardize(feat_mat: np.ndarray, m_name: str) -> np.ndarray:
        out = np.empty_like(feat_mat, dtype=np.float32)
        m_dict = scaler_params[m_name]
        for j, f in enumerate(FEATURE_NAMES_23):
            out[:, j] = np.clip((feat_mat[:, j] - m_dict[f]["mean"]) / m_dict[f]["std"], -5.0, 5.0)
        return out

    # 3. Index Sealed Memory Bank (<= 2020-12-31)
    print("\n[Step 3] Indexing sealed memory bank precedents (<= 2020-12-31)...")
    mem_records = []
    mem_raw_windows = []
    mem_raw_last_list = []
    mem_ret63_list = []

    for m in markets.keys():
        for tkr, df in market_data[m].items():
            df_train = df.loc[df.index <= pd.Timestamp("2020-12-31")]
            n_len = len(df_train)
            if n_len < t_win + 126:
                continue
            feat_raw = df_train[FEATURE_NAMES_23].values
            feat_norm = standardize(feat_raw, m)
            ret21 = df_train["future_return_21"].values
            ret63 = df_train["future_return_63"].values
            dd63 = df_train["future_drawdown_63"].values
            dates = df_train.index

            for i in range(t_win, n_len - 126):
                obs_dt = dates[i]
                avail_dt = dates[i + 126]
                if avail_dt > pd.Timestamp("2020-12-31"):
                    continue

                w_slice = feat_norm[i - t_win:i]
                patch_mat = w_slice.reshape(n_patches, patch_size, 23).mean(axis=1)  # shape (42, 23)
                flat_window = patch_mat.reshape(-1)  # shape (966,)

                r_idx = len(mem_records)
                mem_raw_windows.append(flat_window)
                mem_raw_last_list.append(feat_norm[i - 1])
                r63_val = float(ret63[i]) if not np.isnan(ret63[i]) else 0.0
                mem_ret63_list.append(r63_val)

                mem_records.append({
                    "record_id": r_idx,
                    "market": m,
                    "ticker": tkr,
                    "origin_timestamp": obs_dt.strftime("%Y-%m-%d"),
                    "available_timestamp": avail_dt.strftime("%Y-%m-%d"),
                    "ticker_session_index": i,
                    "latent_array_index": r_idx,
                    "raw_window_array_index": r_idx,
                    "return_21": float(ret21[i]) if not np.isnan(ret21[i]) else 0.0,
                    "return_63": r63_val,
                    "drawdown_63": float(dd63[i]) if not np.isnan(dd63[i]) else 0.0,
                })

    mem_df = pd.DataFrame(mem_records)
    mem_raw_arr = np.array(mem_raw_windows, dtype=np.float32)
    print(f"   [+] Indexed {len(mem_df):,d} memory precedents. Saving memory metadata and raw windows...")
    mem_df.to_parquet(CACHE_DIR / "memory_meta.parquet", index=False)
    np.save(CACHE_DIR / "memory_raw_windows.npy", mem_raw_arr)

    # 4. Ensure MLP Checkpoints on pre-2021 data
    print("\n[Step 4] Ensuring persistent MLP checkpoints for seeds 7, 17, 37...")
    ensure_mlp_checkpoints(
        train_x=np.array(mem_raw_last_list, dtype=np.float32),
        train_y=np.array(mem_ret63_list, dtype=np.float32),
        seeds=backbone_seeds,
        device=device,
    )

    # 5. Encode Memory Latents on GPU for Transformer and MLP
    print("\n[Step 5] Encoding memory latents on GPU...")
    mem_patches_tensor = torch.tensor(mem_raw_arr.reshape(-1, n_patches, 23), dtype=torch.float32, device=device)
    mem_last_feat_tensor = torch.tensor(np.array(mem_raw_last_list, dtype=np.float32), dtype=torch.float32, device=device)

    for s in backbone_seeds:
        # Transformer
        print(f"   * Encoding Transformer latents (seed {s})...")
        trans_model = load_transformer_checkpoint(s, device)
        trans_lats = []
        with torch.no_grad():
            for b_i in range(0, len(mem_patches_tensor), 4096):
                b_p = mem_patches_tensor[b_i:b_i + 4096]
                lat, _ = trans_model(b_p)
                trans_lats.append(F.normalize(lat, p=2, dim=1).cpu())
        trans_lat_arr = torch.cat(trans_lats, dim=0).numpy()
        np.save(CACHE_DIR / f"memory_latents_transformer_seed_{s}.npy", trans_lat_arr)

        # MLP
        print(f"   * Encoding MLP latents (seed {s})...")
        mlp_model = load_mlp_checkpoint(s, device)
        mlp_lats = []
        with torch.no_grad():
            for b_i in range(0, len(mem_last_feat_tensor), 4096):
                b_x = mem_last_feat_tensor[b_i:b_i + 4096]
                lat, _ = mlp_model(b_x)
                mlp_lats.append(F.normalize(lat, p=2, dim=1).cpu())
        mlp_lat_arr = torch.cat(mlp_lats, dim=0).numpy()
        np.save(CACHE_DIR / f"memory_latents_mlp_seed_{s}.npy", mlp_lat_arr)

    # 6. Index Evaluation Queries (2021-2024)
    print("\n[Step 6] Indexing query records across 2021-2024...")
    query_records = []
    outcome_records = []
    query_raw_windows = []

    # Map dates per market
    market_calendars = {}
    for m in markets.keys():
        dates_set = set()
        for tkr, df in market_data[m].items():
            sub = df.loc[(df.index >= "2021-01-01") & (df.index <= "2024-12-31")]
            dates_set.update(sub.index.tolist())
        market_calendars[m] = sorted(list(dates_set))

    # Collect queries
    for m in markets.keys():
        cal_dates = market_calendars[m]
        for dt in cal_dates:
            dt_str = dt.strftime("%Y-%m-%d")
            for tkr in markets[m]:
                if tkr not in market_data[m]:
                    continue
                df = market_data[m][tkr]
                if dt not in df.index:
                    continue
                loc = df.index.get_loc(dt)
                if loc < t_win:
                    continue

                feat_raw = df[FEATURE_NAMES_23].values
                feat_norm = standardize(feat_raw, m)
                w_slice = feat_norm[loc - t_win:loc]
                patch_mat = w_slice.reshape(n_patches, patch_size, 23).mean(axis=1)
                flat_window = patch_mat.reshape(-1)

                q_idx = len(query_records)
                query_raw_windows.append(flat_window)

                vol_21d = float(df["tech_volatility_21d"].iloc[loc])
                r63_realized = float(df["future_return_63"].iloc[loc]) if not np.isnan(df["future_return_63"].iloc[loc]) else 0.0
                dd63_realized = float(df["future_drawdown_63"].iloc[loc]) if not np.isnan(df["future_drawdown_63"].iloc[loc]) else 0.0
                avail_session_idx = loc + 63
                avail_dt = df.index[avail_session_idx].strftime("%Y-%m-%d") if avail_session_idx < len(df) else "9999-12-31"

                q_id = f"Q_{m}_{tkr}_{dt_str}"
                query_records.append({
                    "query_id": q_id,
                    "market": m,
                    "ticker": tkr,
                    "decision_timestamp": dt_str,
                    "information_cutoff": dt_str,
                    "array_index": q_idx,
                    "daily_volatility": vol_21d,
                    "feature_valid": True,
                })

                outcome_records.append({
                    "query_id": q_id,
                    "realized_return_63": r63_realized,
                    "realized_drawdown_63": dd63_realized,
                    "outcome_available_timestamp": avail_dt,
                })

    query_df = pd.DataFrame(query_records)
    outcome_df = pd.DataFrame(outcome_records)
    query_raw_arr = np.array(query_raw_windows, dtype=np.float32)

    print(f"   [+] Indexed {len(query_df):,d} queries across 2021-2024. Saving query metadata and outcomes...")
    query_df.to_parquet(CACHE_DIR / "query_meta.parquet", index=False)
    outcome_df.to_parquet(CACHE_DIR / "outcomes.parquet", index=False)
    np.save(CACHE_DIR / "query_raw_windows.npy", query_raw_arr)

    # 7. Compute Base Predictions and Latents for Queries
    print("\n[Step 7] Computing base predictions and latents for all queries...")
    q_patches_tensor = torch.tensor(query_raw_arr.reshape(-1, n_patches, 23), dtype=torch.float32, device=device)
    # Extract last patch feature for MLP
    q_last_feat_tensor = q_patches_tensor[:, -1, :]

    for s in backbone_seeds:
        # Transformer
        print(f"   * Query inference for Transformer (seed {s})...")
        trans_model = load_transformer_checkpoint(s, device)
        trans_preds = []
        trans_lats = []
        with torch.no_grad():
            for b_i in range(0, len(q_patches_tensor), 4096):
                b_p = q_patches_tensor[b_i:b_i + 4096]
                lat, pred = trans_model(b_p)
                trans_lats.append(F.normalize(lat, p=2, dim=1).cpu())
                trans_preds.append(pred.squeeze(1).cpu())
        np.save(CACHE_DIR / f"query_latents_transformer_seed_{s}.npy", torch.cat(trans_lats, dim=0).numpy())
        np.save(CACHE_DIR / f"query_preds_transformer_seed_{s}.npy", torch.cat(trans_preds, dim=0).numpy())

        # MLP
        print(f"   * Query inference for MLP (seed {s})...")
        mlp_model = load_mlp_checkpoint(s, device)
        mlp_preds = []
        mlp_lats = []
        with torch.no_grad():
            for b_i in range(0, len(q_last_feat_tensor), 4096):
                b_x = q_last_feat_tensor[b_i:b_i + 4096]
                lat, pred = mlp_model(b_x)
                mlp_lats.append(F.normalize(lat, p=2, dim=1).cpu())
                mlp_preds.append(pred.squeeze(1).cpu())
        np.save(CACHE_DIR / f"query_latents_mlp_seed_{s}.npy", torch.cat(mlp_lats, dim=0).numpy())
        np.save(CACHE_DIR / f"query_preds_mlp_seed_{s}.npy", torch.cat(mlp_preds, dim=0).numpy())

    print(f"\n[+] All caches successfully built and persisted in {CACHE_DIR}")
    return {"cache_dir": str(CACHE_DIR), "reused": False, "n_memory": len(mem_df), "n_queries": len(query_df)}


if __name__ == "__main__":
    build_caches(force_rebuild=True)
