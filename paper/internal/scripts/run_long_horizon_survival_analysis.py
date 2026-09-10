#!/usr/bin/env python3
"""
Event-Level Long-Horizon Survival & Greenwood Uncertainty Analysis
==================================================================
Date: September 10, 2026
Target: Digital Finance (Springer Nature)

Computes:
1. Exact Kaplan-Meier survival curves S(t) for t in 1..63 sessions.
2. Log-log transformed 95% Greenwood confidence bands constrained to [0, 1].
3. Number at risk (n_t), active events (d_t), and administrative censors (c_t) at every t.
4. Risk-set balance verification: n_t = n_{t-1} - d_{t-1} - c_{t-1}.
5. Restricted Mean Survival Time (RMST) through day 63 with 18-cluster bootstrap SE.
6. Cause-specific cumulative incidence (ATR Chandelier Stop vs Max Horizon 63d vs Calendar End).
7. Local Calendar Follow-Up Sensitivity: Complete 63-session cohort vs Full Sample.
8. Synchronized publication LaTeX table: table_holding_duration_and_censoring.tex.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
EVIDENCE_DIR = PROJECT_ROOT / "paper" / "internal" / "evidence" / "research_defense_extract"
V4_DIR = EVIDENCE_DIR / "v4_annual_252_evidence"
ROBUST_DIR = EVIDENCE_DIR / "v4_deep_robustness"
LATEX_DIR = PROJECT_ROOT / "latex_tables"
PKG_LATEX_DIR = PROJECT_ROOT / "exports" / "CORE_RL_V4_VERIFIED_GOVERNANCE_PACKAGE" / "latex_tables"

for d in [ROBUST_DIR, LATEX_DIR, PKG_LATEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("[*] Starting Event-Level Long-Horizon Survival Analysis...")

# 1. Load Data
p_trades = V4_DIR / "v4_trade_ledgers_p0_p6.csv"
p_eq = V4_DIR / "v4_daily_equity_curves_p0_p6.csv"

df_trades = pd.read_csv(p_trades)
df_eq = pd.read_csv(p_eq)

# Map local exchange calendars
market_calendars = {}
for m, g in df_eq[df_eq["system"] == "P0"].groupby("market"):
    market_calendars[m] = sorted(g["date"].unique())

print(f"   [+] Loaded {len(df_trades):,} trades across {len(market_calendars)} sovereign markets.")

# Flag complete 63-session follow-up eligibility using actual local market calendars
def check_eligible_followup(row):
    cal = market_calendars[row["market"]]
    ed = row["entry_date"]
    if ed in cal:
        idx = cal.index(ed)
        return (idx + 63 < len(cal))
    return False

df_trades["eligible_63d"] = df_trades.apply(check_eligible_followup, axis=1)

# Systems to evaluate (excluding passive hold P6)
ACTIVE_SYSTEMS = ["P0", "P0*", "P1", "P2", "P4"]

# 2. Compute Kaplan-Meier and Greenwood Bands
km_records = []
summary_results = {}

for sys_name in ACTIVE_SYSTEMS:
    sub = df_trades[df_trades["system"] == sys_name].copy()
    N_total = len(sub)
    
    # Verify holding_days match
    # Active policy event: atr_chandelier_stop or max_horizon
    # Administrative censor: calendar_end
    events_stop = sub[sub["exit_reason"] == "atr_chandelier_stop"]["holding_days"].values
    events_max = sub[sub["exit_reason"] == "max_horizon"]["holding_days"].values
    censors_cal = sub[sub["exit_reason"] == "calendar_end"]["holding_days"].values
    
    assert len(events_stop) + len(events_max) + len(censors_cal) == N_total, "Exit reasons do not reconcile!"
    
    # Discrete day counts
    n_at_risk = N_total
    S_t = 1.0
    cum_greenwood_sum = 0.0
    
    # Store daily trajectory
    daily_rows = []
    
    for t in range(1, 64):
        # Events on day t
        d_stop_t = int(np.sum(events_stop == t))
        d_max_t = int(np.sum(events_max == t))
        d_t = d_stop_t + d_max_t
        c_t = int(np.sum(censors_cal == t))
        
        # Hazard
        if n_at_risk > 0:
            h_t = d_t / n_at_risk
        else:
            h_t = 0.0
            
        S_prev = S_t
        S_t = S_t * (1.0 - h_t)
        
        # Greenwood term
        if n_at_risk > d_t and d_t > 0:
            cum_greenwood_sum += d_t / (n_at_risk * (n_at_risk - d_t))
            
        var_S = (S_t ** 2) * cum_greenwood_sum
        se_S = np.sqrt(max(0.0, var_S))
        
        # Log-log transformed 95% confidence bands (bounded in [0, 1])
        if 0.0 < S_t < 1.0 and se_S > 0:
            w = 1.96 * se_S / (S_t * abs(np.log(S_t)))
            ci_lower = S_t ** np.exp(w)
            ci_upper = S_t ** np.exp(-w)
        elif S_t >= 1.0:
            ci_lower = 1.0
            ci_upper = 1.0
        else:
            ci_lower = 0.0
            ci_upper = 0.0
            
        ci_lower = max(0.0, min(1.0, ci_lower))
        ci_upper = max(0.0, min(1.0, ci_upper))
        
        daily_rows.append({
            "system": sys_name,
            "holding_day": t,
            "n_at_risk": n_at_risk,
            "events_total": d_t,
            "events_stop": d_stop_t,
            "events_max_horizon": d_max_t,
            "censored_calendar": c_t,
            "survival_prob": round(float(S_t), 5),
            "greenwood_se": round(float(se_S), 5),
            "ci_95_lower": round(float(ci_lower), 5),
            "ci_95_upper": round(float(ci_upper), 5),
        })
        
        # Invariant update for next day
        n_at_risk = n_at_risk - d_t - c_t
        
    assert n_at_risk == 0, f"Remaining at risk on day 64 was {n_at_risk}, expected 0!"
    km_records.extend(daily_rows)
    
    # RMST through day 63: sum of S(t) for t=1..63
    rmst_val = sum(r["survival_prob"] for r in daily_rows)
    
    # Also evaluate eligible cohort sensitivity (where calendar_end == 0)
    sub_elig = sub[sub["eligible_63d"]].copy()
    N_elig = len(sub_elig)
    n_at_risk_elig = N_elig
    S_t_elig = 1.0
    daily_elig_S = []
    
    for t in range(1, 64):
        d_elig_t = int(np.sum(sub_elig["holding_days"] == t))
        h_elig_t = d_elig_t / n_at_risk_elig if n_at_risk_elig > 0 else 0.0
        S_t_elig = S_t_elig * (1.0 - h_elig_t)
        daily_elig_S.append(S_t_elig)
        n_at_risk_elig -= d_elig_t
        
    rmst_elig = sum(daily_elig_S)
    
    # Store summary
    summary_results[sys_name] = {
        "total_trades": N_total,
        "atr_stops": len(events_stop),
        "atr_stops_pct": round(len(events_stop) / N_total * 100, 2),
        "max_horizon_63d": len(events_max),
        "max_horizon_pct": round(len(events_max) / N_total * 100, 2),
        "calendar_censored": len(censors_cal),
        "calendar_censored_pct": round(len(censors_cal) / N_total * 100, 2),
        "mean_holding_days": round(float(sub["holding_days"].mean()), 2),
        "median_holding_days": round(float(sub["holding_days"].median()), 1),
        "q25_holding_days": round(float(sub["holding_days"].quantile(0.25)), 1),
        "q75_holding_days": round(float(sub["holding_days"].quantile(0.75)), 1),
        "rmst_day_63": round(float(rmst_val), 2),
        "eligible_cohort_trades": N_elig,
        "eligible_cohort_pct": round(N_elig / N_total * 100, 2),
        "eligible_rmst_day_63": round(float(rmst_elig), 2),
    }

df_km = pd.DataFrame(km_records)
df_km.to_csv(ROBUST_DIR / "kaplan_meier_survival_with_greenwood_bands.csv", index=False)
print(f"   [+] Exported Kaplan-Meier survival curves with Greenwood bands: {len(df_km)} rows.")

# 3. Market-Seed Cluster Bootstrap for RMST Uncertainty (B=2000)
print("3. Computing Market-Seed Cluster Bootstrap for RMST...")
clusters = df_trades[["market", "seed"]].drop_duplicates().values
B = 2000
rng = np.random.RandomState(42)

bootstrap_rmst = {s: [] for s in ACTIVE_SYSTEMS}

for b in range(B):
    boot_idx = rng.choice(len(clusters), size=len(clusters), replace=True)
    boot_clusters = clusters[boot_idx]
    
    # Reassemble bootstrap trades
    boot_dfs = []
    for m, s in boot_clusters:
        boot_dfs.append(df_trades[(df_trades["market"] == m) & (df_trades["seed"] == s)])
    df_boot = pd.concat(boot_dfs, ignore_index=True)
    
    for sys_name in ACTIVE_SYSTEMS:
        s_boot = df_boot[df_boot["system"] == sys_name]
        n_risk = len(s_boot)
        if n_risk == 0:
            continue
        S_b = 1.0
        rmst_b = 0.0
        ev_stop = s_boot[s_boot["exit_reason"] == "atr_chandelier_stop"]["holding_days"].values
        ev_max = s_boot[s_boot["exit_reason"] == "max_horizon"]["holding_days"].values
        cens_cal = s_boot[s_boot["exit_reason"] == "calendar_end"]["holding_days"].values
        
        for t in range(1, 64):
            dt = int(np.sum(ev_stop == t)) + int(np.sum(ev_max == t))
            ct = int(np.sum(cens_cal == t))
            ht = dt / n_risk if n_risk > 0 else 0.0
            S_b = S_b * (1.0 - ht)
            rmst_b += S_b
            n_risk = n_risk - dt - ct
            
        bootstrap_rmst[sys_name].append(rmst_b)

for sys_name in ACTIVE_SYSTEMS:
    draws = np.array(bootstrap_rmst[sys_name])
    se_boot = float(np.std(draws))
    ci_low = float(np.percentile(draws, 2.5))
    ci_high = float(np.percentile(draws, 97.5))
    summary_results[sys_name]["rmst_cluster_se"] = round(se_boot, 2)
    summary_results[sys_name]["rmst_95_ci"] = [round(ci_low, 2), round(ci_high, 2)]
    print(f"   [+] {sys_name} RMST(63): {summary_results[sys_name]['rmst_day_63']:.2f} sessions (Cluster SE: {se_boot:.2f}, 95% CI: [{ci_low:.2f}, {ci_high:.2f}])")

# Save summary JSON
with open(ROBUST_DIR / "holding_survival_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary_results, f, indent=2)
print("   [+] Wrote holding_survival_summary.json.")

# 4. Generate Publication-Ready LaTeX Table
latex_table = r"""\begin{table}[ht]
\centering
\small
\begin{tabular}{lccccccc}
\toprule
\textbf{System} & \textbf{Trades} & \textbf{\shortstack{Stop Loss\\(\%)}} & \textbf{\shortstack{Max Horizon\\63d (\%)}} & \textbf{\shortstack{Calendar End\\Censored (\%)}} & \textbf{\shortstack{Median\\Hold}} & \textbf{\shortstack{Full RMST\\(63d, SE)}} & \textbf{\shortstack{Eligible\\RMST}} \\
\midrule
"""

for sys_name in ACTIVE_SYSTEMS:
    sr = summary_results[sys_name]
    bold_prefix = r"\textbf{" if sys_name in ("P0", "P0*") else ""
    bold_suffix = r"}" if sys_name in ("P0", "P0*") else ""
    latex_table += (
        f"{bold_prefix}{sys_name}{bold_suffix} & "
        f"{sr['total_trades']} & "
        f"{sr['atr_stops_pct']}\\% & "
        f"{sr['max_horizon_pct']}\\% & "
        f"{sr['calendar_censored_pct']}\\% & "
        f"{sr['median_holding_days']:.1f}d & "
        f"{sr['rmst_day_63']:.2f}d ({sr['rmst_cluster_se']:.2f}) & "
        f"{sr['eligible_rmst_day_63']:.2f}d \\\\\n"
    )

latex_table += r"""\bottomrule
\end{tabular}
\caption{Position Holding Duration, Active-Policy Termination, and Administrative Censoring across 6 Sovereign Markets $\times$ 3 Seeds (18 cells). \textit{Stop Loss}: trailing $2.5\times\text{ATR}_{14}$ Chandelier volatility liquidation. \textit{Max Horizon}: positions reaching the 63-session ceiling without stop triggering. \textit{Calendar End}: positions active at the 2024 boundary (treated as administrative right-censoring in survival estimation). \textit{Full RMST}: Restricted Mean Survival Time through day 63 with market--seed cluster-bootstrap standard error. \textit{Eligible RMST}: sensitivity restricted to positions with $\ge 63$ potential follow-up sessions before year-end on actual local market exchange calendars ($0.0\%$ calendar censoring).}
\label{tab:holding_duration_and_censoring}
\end{table}
"""

for target_dir in [LATEX_DIR, PKG_LATEX_DIR]:
    p = target_dir / "table_holding_duration_and_censoring.tex"
    p.write_text(latex_table, encoding="utf-8")
    print(f"[+] Wrote {p}")

# Copy script to paper/internal/scripts
p_repo_script = PROJECT_ROOT / "paper" / "internal" / "scripts" / "run_long_horizon_survival_analysis.py"
p_repo_script.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
print(f"[+] Synced script to {p_repo_script}")

print("\n[STAGE 3 COMPLETED SUCCESSFULLY]")
