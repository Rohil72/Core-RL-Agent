"""Standalone, exact Market x Seed Stratified Panel Moving-Block Bootstrap Generator.

Loads paired daily returns / equity curves across 6 markets x 3 seeds (18 independent cells),
executes synchronous moving-block panel resampling (L in {5, 21, 63} sessions, 1,000 resamples),
computes 95% panel confidence intervals, empirical discrete p-values, Holm-Bonferroni, and FDR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def run_panel_stratified_bootstrap(
    df_piv_table: pd.DataFrame,
    sys_a: str,
    sys_b: str,
    block_len: int = 21,
    n_bootstraps: int = 1000,
    random_seed: int = 7,
) -> dict[str, Any]:
    """Execute stratified moving-block panel bootstrap across all 18 market-seed cells."""
    cells = df_piv_table.groupby(["market", "seed"])
    cell_series = {}
    for (m, s), grp in cells:
        cell_series[(m, s)] = (grp[sys_a].values, grp[sys_b].values)

    # Point estimate: mean delta Sharpe across all 18 cells
    cell_diffs = []
    for (m, s), (ra, rb) in cell_series.items():
        sa = np.sqrt(252) * np.mean(ra) / (np.std(ra, ddof=1) + 1e-8)
        sb = np.sqrt(252) * np.mean(rb) / (np.std(rb, ddof=1) + 1e-8)
        cell_diffs.append(sa - sb)
    point_estimate = float(np.mean(cell_diffs))

    rng = np.random.default_rng(random_seed)
    boot_estimates = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        b_diffs = []
        for (m, s), (ra, rb) in cell_series.items():
            n = len(ra)
            k = max(1, block_len)
            n_blocks = n - k + 1
            blocks_a = np.array([ra[i : i + k] for i in range(n_blocks)])
            blocks_b = np.array([rb[i : i + k] for i in range(n_blocks)])

            n_needed = int(np.ceil(n / k))
            chosen = rng.integers(0, n_blocks, size=n_needed)
            samp_a = blocks_a[chosen].reshape(-1)[:n]
            samp_b = blocks_b[chosen].reshape(-1)[:n]

            sha = np.sqrt(252) * np.mean(samp_a) / (np.std(samp_a, ddof=1) + 1e-8)
            shb = np.sqrt(252) * np.mean(samp_b) / (np.std(samp_b, ddof=1) + 1e-8)
            b_diffs.append(sha - shb)
        boot_estimates[b] = np.mean(b_diffs)

    ci_low = float(np.percentile(boot_estimates, 2.5))
    ci_high = float(np.percentile(boot_estimates, 97.5))
    if point_estimate >= 0:
        raw_p = (1.0 + np.sum(boot_estimates <= 0.0)) / (n_bootstraps + 1.0)
    else:
        raw_p = (1.0 + np.sum(boot_estimates >= 0.0)) / (n_bootstraps + 1.0)

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "raw_p_value": float(raw_p),
        "replications": n_bootstraps,
        "block_length": block_len,
        "cell_count": len(cell_series),
    }

def main():
    parser = argparse.ArgumentParser(description="Run panel stratified bootstrap.")
    parser.add_argument("--equity-file", default="paper/internal/evidence/research_defense_extract/raw_experimental_evidence/equity_curves_and_trades/daily_equity_curves_p0_p6.csv")
    parser.add_argument("--output-json", default="paper/internal/evidence/research_defense_extract/raw_experimental_evidence/paired_returns_bootstrap/statistical_significance_tests.json")
    args = parser.parse_args()

    eq_path = Path(args.equity_file)
    if not eq_path.exists():
        print(f"Error: {eq_path} not found.")
        return

    df_eq = pd.read_csv(eq_path)
    df_piv = df_eq.pivot_table(index=["date", "market", "seed"], columns="system", values="daily_return").reset_index()

    bootstrap_results = {}
    comparisons = [
        ("P0 vs P1 (H2 Memory Benefit)", "P0", "P1"),
        ("P0 vs P2 (H3 Distributional Benefit)", "P0", "P2"),
        ("P0 vs P3 (H1 Representation Benefit)", "P0", "P3"),
        ("P0 vs P4 (Momentum Superiority)", "P0", "P4"),
        ("P0 vs P5 (Random Superiority)", "P0", "P5"),
    ]

    for b_len in [5, 21, 63]:
        b_rows = []
        p_vals = []
        for label, sa, sb in comparisons:
            res = run_panel_stratified_bootstrap(df_piv, sa, sb, block_len=b_len, n_bootstraps=1000)
            p_vals.append(res["raw_p_value"])
            b_rows.append({
                "comparison": label,
                "point_estimate": round(res["point_estimate"], 3),
                "ci_lower": round(res["ci_lower"], 3),
                "ci_upper": round(res["ci_upper"], 3),
                "raw_p_value": round(res["raw_p_value"], 4),
                "block_length": b_len,
                "replications": 1000,
                "cell_count": res["cell_count"],
            })
        p_arr = np.array(p_vals)
        m = len(p_arr)
        order = np.argsort(p_arr)
        p_holm = np.zeros(m)
        prev = 0.0
        for rank, idx in enumerate(order):
            adj = (m - rank) * p_arr[idx]
            adj = max(adj, prev)
            p_holm[idx] = min(1.0, adj)
            prev = p_holm[idx]

        q_fdr = np.zeros(m)
        for rank, idx in enumerate(order):
            q_fdr[idx] = min(1.0, (m / (rank + 1)) * p_arr[idx])

        for i, r in enumerate(b_rows):
            r["p_holm"] = round(float(p_holm[i]), 4)
            r["q_fdr"] = round(float(q_fdr[i]), 5)

        bootstrap_results[f"block_length_{b_len}"] = b_rows

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bootstrap_results, f, indent=2)

    print("=" * 70)
    print("STRATIFIED PANEL MOVING-BLOCK BOOTSTRAP RESULTS (L=21 sessions):")
    print("=" * 70)
    for r in bootstrap_results["block_length_21"]:
        print(f"{r['comparison']:<40} DeltaSharpe={r['point_estimate']:+.3f} 95% CI=[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]  p_Holm={r['p_holm']:.4f}  q_FDR={r['q_fdr']:.5f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
