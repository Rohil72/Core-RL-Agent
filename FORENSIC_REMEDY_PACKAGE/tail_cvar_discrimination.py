#!/usr/bin/env python3
"""
Tail-CVaR Discrimination Analysis
==================================
Package: FORENSIC_REMEDY_PACKAGE
Purpose: Evaluates whether precedent downside CVaR (|CVaR_0.05|) discriminates
         subsequent realized left-tail drawdowns across evaluated candidates.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

def run_cvar_discrimination_analysis() -> dict:
    df = pd.read_csv(ROOT / "tail_cvar_discrimination_data.csv")
    
    severe = df[df["forward_severe_drawdown"] == 1]["precedent_cvar_05"].values
    benign = df[df["forward_severe_drawdown"] == 0]["precedent_cvar_05"].values
    
    mean_severe = float(np.mean(severe))
    mean_benign = float(np.mean(benign))
    ratio = mean_severe / mean_benign if mean_benign > 0 else np.nan
    
    u_stat, u_pval = stats.mannwhitneyu(severe, benign, alternative="greater")
    
    extreme = df[df["forward_extreme_drawdown"] == 1]["precedent_cvar_05"].values
    non_extreme = df[df["forward_extreme_drawdown"] == 0]["precedent_cvar_05"].values
    mean_extreme = float(np.mean(extreme))
    u_stat_ext, u_pval_ext = stats.mannwhitneyu(extreme, non_extreme, alternative="greater")
    
    rho, rho_pval = stats.spearmanr(df["precedent_cvar_05"], df["forward_realized_mdd_63d"].abs())
    
    summary = {
        "total_evaluated_records": len(df),
        "severe_drawdown_count": len(severe),
        "benign_drawdown_count": len(benign),
        "mean_precedent_cvar_severe_drawdowns": mean_severe,
        "mean_precedent_cvar_benign_drawdowns": mean_benign,
        "cvar_risk_ratio_severe_vs_benign": ratio,
        "mann_whitney_u_severe_pval": float(u_pval),
        "mean_precedent_cvar_extreme_drawdowns": mean_extreme,
        "mann_whitney_u_extreme_pval": float(u_pval_ext),
        "spearman_rank_correlation_cvar_vs_mdd": float(rho),
        "spearman_rank_correlation_pval": float(rho_pval),
    }
    return summary

if __name__ == "__main__":
    res = run_cvar_discrimination_analysis()
    print("=" * 70)
    print("TAIL-CVAR DISCRIMINATION AUDIT")
    print("=" * 70)
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:45s}: {v:.4e}" if abs(v) < 0.001 else f"  {k:45s}: {v:.4f}")
        else:
            print(f"  {k:45s}: {v}")
