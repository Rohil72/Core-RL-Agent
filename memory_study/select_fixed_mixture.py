"""
Development-Period Fixed Mixture Selection Module.

Evaluates lambda in {0.0, 0.10, 0.25, 0.50, 1.00} on the designated H2 2021 development window.
Selects optimal lambda* per backbone based on mean net Sharpe across sovereign markets.
Saves:
- research_runs/memory_study/final_comparison/development_mixture_grid.csv
- research_runs/memory_study/final_comparison/selected_mixture_coefficients.json
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch

from memory_study.retrieval import batch_retrieve_m2
from memory_study.mixing import compute_trading_score
from memory_study.engine_adapter import SimulationEngine
from memory_study.evaluation import compute_forecast_metrics

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
CACHE_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "cache"
CONFIG_PATH = PROJECT_ROOT / "memory_study" / "configs" / "final_comparison.yaml"
OUTPUT_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "final_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_mixture_selection(dev_start: str = "2021-07-01", dev_end: str = "2021-12-31") -> Dict[str, Any]:
    print("=" * 85)
    print("DEVELOPMENT-PERIOD CONSTANT MIXTURE SELECTION (H2 2021)")
    print(f"Interval: {dev_start} to {dev_end}")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 85)

    import yaml
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    markets = cfg["markets"]
    backbone_seeds = cfg["backbones"]["seeds"]
    candidate_lambdas = cfg["mixture_selection"]["candidate_lambdas"]
    k_val = cfg["retrieval"]["k"]

    # 1. Load Caches
    print("\n[Step 1] Loading precomputed caches...")
    query_meta = pd.read_parquet(CACHE_DIR / "query_meta.parquet")
    outcomes_df = pd.read_parquet(CACHE_DIR / "outcomes.parquet")
    outcomes_map = dict(zip(outcomes_df["query_id"], outcomes_df["realized_return_63"]))

    mem_meta = pd.read_parquet(CACHE_DIR / "memory_meta.parquet")
    mem_tickers = mem_meta["ticker"].values
    mem_sessions = mem_meta["ticker_session_index"].values.astype(int)
    mem_returns = mem_meta["return_63"].values.astype(np.float32)

    mem_raw_arr = np.load(CACHE_DIR / "memory_raw_windows.npy")
    mem_windows_gpu = torch.tensor(mem_raw_arr, dtype=torch.float32, device=DEVICE)
    mem_sq_norms_gpu = torch.sum(mem_windows_gpu ** 2, dim=1)

    query_raw_windows = np.load(CACHE_DIR / "query_raw_windows.npy")
    query_preds = {}
    for bb in ["transformer", "mlp"]:
        query_preds[bb] = {}
        for s in backbone_seeds:
            query_preds[bb][s] = np.load(CACHE_DIR / f"query_preds_{bb}_seed_{s}.npy")

    # Ingest OHLCV
    data_dir = PROJECT_ROOT / "FINAL_SUBMISSION_PACKAGE" / "data" / "cache" / "ohlcv"
    market_dfs = {}
    for m, tickers in markets.items():
        market_dfs[m] = {}
        for tkr in tickers:
            p = data_dir / f"{m}_{tkr}.parquet"
            df = pd.read_parquet(p)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = [c.lower() for c in df.columns]
            market_dfs[m][tkr] = df

    # Filter H2 2021 queries
    h2_mask = (query_meta["decision_timestamp"] >= dev_start) & (query_meta["decision_timestamp"] <= dev_end)
    h2_queries = query_meta[h2_mask].copy()
    h2_indices = h2_queries["array_index"].values
    h2_tickers = h2_queries["ticker"].values
    h2_qids = h2_queries["query_id"].values
    h2_windows = query_raw_windows[h2_indices]
    y_h2 = np.array([outcomes_map.get(qid, 0.0) for qid in h2_qids], dtype=np.float32)

    print(f"\n[Step 2] Retrieving M2 precedents for {len(h2_queries)} H2 2021 queries...")
    t0 = time.time()
    h2_mem_preds, _ = batch_retrieve_m2(
        query_windows=h2_windows,
        mem_windows_gpu=mem_windows_gpu,
        mem_sq_norms_gpu=mem_sq_norms_gpu,
        query_tickers=h2_tickers,
        mem_tickers=mem_tickers,
        mem_sessions=mem_sessions,
        mem_returns=mem_returns,
        k=k_val,
        batch_size=500,
        device=DEVICE,
    )
    print(f"   [+] Precedents retrieved in {time.time() - t0:.2f}s.")

    # 2. Evaluate Grid on H2 2021
    print("\n[Step 3] Simulating mixture grid across backbones, seeds, and lambdas...")
    grid_rows = []

    backbones = ["MLP", "Transformer"]

    for bb in backbones:
        bb_id = bb.lower()
        for lam in candidate_lambdas:
            sharpes_by_seed = []
            returns_by_seed = []
            mses_by_seed = []
            rank_ics_by_seed = []

            for s in backbone_seeds:
                b_preds = query_preds[bb_id][s][h2_indices]
                hyb_preds = (1.0 - lam) * b_preds + lam * h2_mem_preds
                preds_dict = dict(zip(h2_qids, hyb_preds))

                f_metrics = compute_forecast_metrics(y_h2, hyb_preds)
                mses_by_seed.append(f_metrics["mse"])
                rank_ics_by_seed.append(f_metrics["rank_ic"])

                # Market simulations
                mkt_sharpes = []
                mkt_returns = []

                for m in markets.keys():
                    tickers = markets[m]
                    n_tickers = len(tickers)

                    cal_dates_set = set()
                    for tkr in tickers:
                        sub = market_dfs[m][tkr].loc[
                            (market_dfs[m][tkr].index >= dev_start) & (market_dfs[m][tkr].index <= dev_end)
                        ]
                        cal_dates_set.update(sub.index.tolist())
                    cal_dates = sorted(list(cal_dates_set))
                    n_sessions = len(cal_dates)
                    dates_str = [d.strftime("%Y-%m-%d") for d in cal_dates]

                    price_open = np.zeros((n_sessions, n_tickers), dtype=np.float32)
                    price_close = np.zeros((n_sessions, n_tickers), dtype=np.float32)
                    atr_ratio = np.zeros((n_sessions, n_tickers), dtype=np.float32)
                    vol_21d = np.zeros((n_sessions, n_tickers), dtype=np.float32)

                    for t_i, tkr in enumerate(tickers):
                        df_t = market_dfs[m][tkr]
                        c = df_t["close"]
                        h = df_t["high"]
                        l = df_t["low"]
                        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
                        atr = tr.rolling(14, min_periods=5).mean().fillna(c * 0.02)
                        atr_rat = (atr / (c + 1e-9)).fillna(0.02)
                        vol = c.pct_change().rolling(21, min_periods=5).std().fillna(0.01)

                        for d_i, dt in enumerate(cal_dates):
                            if dt in df_t.index:
                                loc = df_t.index.get_loc(dt)
                                price_open[d_i, t_i] = df_t["open"].iloc[loc]
                                price_close[d_i, t_i] = df_t["close"].iloc[loc]
                                atr_ratio[d_i, t_i] = atr_rat.iloc[loc]
                                vol_21d[d_i, t_i] = vol.iloc[loc]

                    for t_i in range(n_tickers):
                        price_open[:, t_i] = pd.Series(price_open[:, t_i]).ffill().bfill().values
                        price_close[:, t_i] = pd.Series(price_close[:, t_i]).ffill().bfill().values
                        atr_ratio[:, t_i] = pd.Series(atr_ratio[:, t_i]).ffill().bfill().values
                        vol_21d[:, t_i] = pd.Series(vol_21d[:, t_i]).ffill().bfill().values

                    score_matrix = np.zeros((n_sessions, n_tickers), dtype=np.float32)
                    for d_i, dt_str in enumerate(dates_str):
                        for t_i, tkr in enumerate(tickers):
                            qid = f"Q_{m}_{tkr}_{dt_str}"
                            if qid in preds_dict:
                                pred_val = preds_dict[qid]
                                v = float(vol_21d[d_i, t_i])
                                score_matrix[d_i, t_i] = compute_trading_score(pred_val, v)

                    engine = SimulationEngine(
                        market=m,
                        tickers=tickers,
                        calendar_dates=dates_str,
                        price_open=price_open,
                        price_close=price_close,
                        atr_ratio_14=atr_ratio,
                        daily_vol=vol_21d,
                        initial_capital=cfg["risk_and_execution"]["initial_capital"],
                        max_slots=cfg["risk_and_execution"]["max_slots"],
                        fee_rate=cfg["risk_and_execution"]["fee_rate"],
                        cash_buffer=cfg["risk_and_execution"]["cash_buffer_ratio"],
                        chandelier_mult=cfg["risk_and_execution"]["chandelier_multiplier"],
                        chandelier_min=cfg["risk_and_execution"]["chandelier_min_dist"],
                        max_holding_days=cfg["risk_and_execution"]["max_holding_sessions"],
                    )
                    cell_res = engine.run_simulation(score_matrix, run_tag=f"DEV_{bb}_lam_{lam}_{s}")
                    mkt_sharpes.append(cell_res["sharpe_ratio"])
                    mkt_returns.append(cell_res["annualized_return"])

                sharpes_by_seed.append(float(np.mean(mkt_sharpes)))
                returns_by_seed.append(float(np.mean(mkt_returns)))

            mean_sr = float(np.mean(sharpes_by_seed))
            mean_ret = float(np.mean(returns_by_seed))
            mean_mse = float(np.mean(mses_by_seed))
            mean_ic = float(np.mean(rank_ics_by_seed))

            grid_rows.append({
                "backbone": bb,
                "lambda": lam,
                "dev_mean_sharpe": round(mean_sr, 4),
                "dev_mean_return": round(mean_ret, 4),
                "dev_forecast_mse": round(mean_mse, 6),
                "dev_rank_ic": round(mean_ic, 4),
            })
            print(f"   [+] Backbone: {bb:<12} | Lambda: {lam:.2f} | Sharpe: {mean_sr:+.4f} | Ret: {mean_ret:+.2%} | MSE: {mean_mse:.6f} | Rank IC: {mean_ic:+.4f}")

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(OUTPUT_DIR / "development_mixture_grid.csv", index=False)
    print(f"\n[+] Saved development grid to {OUTPUT_DIR / 'development_mixture_grid.csv'}")

    # Select optimal lambda* per backbone
    selected_lambdas = {}
    for bb in backbones:
        sub = grid_df[grid_df["backbone"] == bb]
        best_row = sub.loc[sub["dev_mean_sharpe"].idxmax()]
        selected_lambdas[bb.lower()] = {
            "selected_lambda": float(best_row["lambda"]),
            "dev_mean_sharpe": float(best_row["dev_mean_sharpe"]),
            "dev_mean_return": float(best_row["dev_mean_return"]),
            "dev_forecast_mse": float(best_row["dev_forecast_mse"]),
            "dev_rank_ic": float(best_row["dev_rank_ic"]),
        }
        print(f"\n*** SELECTED OPTIMAL MIXTURE FOR {bb.upper()}: lambda* = {best_row['lambda']:.2f} (Dev Sharpe = {best_row['dev_mean_sharpe']:+.4f}) ***")

    with open(OUTPUT_DIR / "selected_mixture_coefficients.json", "w") as f:
        json.dump(selected_lambdas, f, indent=2)
    print(f"[+] Saved selected mixture coefficients to {OUTPUT_DIR / 'selected_mixture_coefficients.json'}")

    return {
        "grid_df": grid_df,
        "selected_lambdas": selected_lambdas,
    }


if __name__ == "__main__":
    run_mixture_selection()
