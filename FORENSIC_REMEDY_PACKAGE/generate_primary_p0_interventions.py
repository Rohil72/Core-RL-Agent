#!/usr/bin/env python3
"""
Primary-P0 Candidate Ledger and Evidentiary Intervention Generator
==================================================================
Package: FORENSIC_REMEDY_PACKAGE
Purpose: Evaluates counterfactual interventions specifically on the Primary P0
         (Global Learned Memory) candidate evaluation ledger across 243 active sessions.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

def run_primary_p0_interventions(csv_path: Path = None) -> dict:
    if csv_path is None:
        csv_path = ROOT / "primary_p0_candidate_ledger.csv"
    
    df = pd.read_csv(csv_path)
    df["direct_score"] = df["pred_utility"] / df["volatility_21d"]
    
    mem_adj = 0.8 * df["baseline_mu"] - 0.2 * df["baseline_cvar"].abs()
    ratio = (mem_adj.abs() / (df["pred_utility"].abs().replace(0, np.nan))).dropna()
    
    dir_sign = np.sign(df["direct_score"])
    base_sign = np.sign(df["baseline_score"])
    sign_overrides = int(np.sum((dir_sign != base_sign) & (dir_sign != 0) & (base_sign != 0)))
    sign_override_rate = sign_overrides / len(df)
    
    sessions = df.groupby(["market", "seed", "decision_date"])
    top1_changes = []
    top3_changes = []
    rank_corrs_spearman = []
    rank_corrs_kendall = []
    
    for (m, s, dt), g in sessions:
        if len(g) < 2:
            continue
        base_sorted = g.sort_values("baseline_score", ascending=False)["candidate_ticker"].tolist()
        dir_sorted = g.sort_values("direct_score", ascending=False)["candidate_ticker"].tolist()
        
        top1_changes.append(base_sorted[0] != dir_sorted[0])
        k = min(3, len(g))
        top3_changes.append(set(base_sorted[:k]) != set(dir_sorted[:k]))
        
        rho, _ = stats.spearmanr(g["baseline_score"], g["direct_score"])
        tau, _ = stats.kendalltau(g["baseline_score"], g["direct_score"])
        if not np.isnan(rho):
            rank_corrs_spearman.append(rho)
        if not np.isnan(tau):
            rank_corrs_kendall.append(tau)
            
    summary = {
        "system": "Primary P0 (Global Learned Memory)",
        "active_decision_sessions": int(len(top1_changes)),
        "total_candidate_evaluations": int(len(df)),
        "top1_candidate_change_count": int(np.sum(top1_changes)),
        "top1_candidate_change_rate": float(np.mean(top1_changes)),
        "top3_candidate_set_change_count": int(np.sum(top3_changes)),
        "top3_candidate_set_change_rate": float(np.mean(top3_changes)),
        "mean_rank_correlation_spearman": float(np.mean(rank_corrs_spearman)),
        "std_rank_correlation_spearman": float(np.std(rank_corrs_spearman)),
        "mean_rank_correlation_kendall": float(np.mean(rank_corrs_kendall)),
        "sign_override_count": sign_overrides,
        "sign_override_rate": float(sign_override_rate),
        "median_memory_to_direct_ratio": float(ratio.median()),
        "mean_memory_to_direct_ratio": float(ratio.mean()),
    }
    return summary

if __name__ == "__main__":
    res = run_primary_p0_interventions()
    print("=" * 70)
    print("PRIMARY P0 EVIDENTIARY INTERVENTION AUDIT")
    print("=" * 70)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:45s}: {v:.4f}")
        else:
            print(f"  {k:45s}: {v}")
