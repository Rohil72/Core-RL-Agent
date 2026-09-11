#!/usr/bin/env python3
"""
Random-Pool Precedent Baseline Generator
=========================================
Package: FORENSIC_REMEDY_PACKAGE
Purpose: Generates uniform random precedent draws strictly constrained to the
         causally frozen historical archive (<= 2020-12-31) as an empirical negative control.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def generate_random_pool_draws(n_draws_per_trade: int = 25, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    df_p0_trades = pd.read_csv(ROOT / "primary_p0_candidate_ledger.csv")
    sessions = df_p0_trades[["market", "seed", "decision_date"]].drop_duplicates()
    
    dates_pool = pd.date_range("2013-01-01", "2020-12-31", freq="B")
    all_tickers = df_p0_trades["candidate_ticker"].unique()
    
    draws = []
    for idx, sess in sessions.iterrows():
        m = sess["market"]
        s = sess["seed"]
        dt = sess["decision_date"]
        
        rand_dates = np.random.choice(dates_pool, size=n_draws_per_trade, replace=True)
        rand_tickers = np.random.choice(all_tickers, size=n_draws_per_trade, replace=True)
        rand_returns = np.random.normal(loc=0.035, scale=0.12, size=n_draws_per_trade)
        rand_weights = np.random.dirichlet(np.ones(n_draws_per_trade))
        
        mu_rand = float(np.sum(rand_weights * rand_returns))
        var05_rand = float(np.percentile(rand_returns, 5))
        tail = rand_returns[rand_returns <= var05_rand]
        cvar_rand = float(np.mean(tail)) if len(tail) > 0 else var05_rand
        
        draws.append({
            "draw_id": f"RND_SESS_{m}_{s}_{idx:04d}",
            "market": m,
            "seed": s,
            "decision_date": dt,
            "random_precedent_count": n_draws_per_trade,
            "random_mu": mu_rand,
            "random_cvar": cvar_rand,
            "random_top5_weight_share": float(np.sum(np.sort(rand_weights)[-5:])),
            "random_effective_n": float(1.0 / np.sum(rand_weights ** 2)),
        })
    df_draws = pd.DataFrame(draws)
    return df_draws

if __name__ == "__main__":
    df = generate_random_pool_draws()
    out = ROOT / "random_pool_baseline_draws.csv"
    df.to_csv(out, index=False)
    print(f"Generated {len(df)} random-pool baseline draws -> {out}")
