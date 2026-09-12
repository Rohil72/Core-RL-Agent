"""
Reporting and Table Formatting for Memory-Centric Equity Selection.

Generates:
- Table A: Basic Memory Matrix (Backbone x Mode)
- Table B: Incremental Contribution & Contrasts (M3-M1, M3-M2, M4-M3, M4-M0)
- Table C: Consistency across markets and years
- Formatted Markdown summaries
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def generate_table_a(cell_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Table A: Basic Memory Matrix.
    Aggregates performance across cells by (Backbone, Memory Mode).
    """
    rows = []
    groups = cell_df.groupby(["backbone", "mode"])

    for (backbone, mode), grp in groups:
        ann_ret = grp["annualized_return"].mean()
        sharpe = grp["sharpe_ratio"].mean()
        max_dd = grp["max_drawdown"].mean()
        turnover = grp["turnover"].mean()
        exposure = grp["avg_exposure"].mean()
        coverage = grp["memory_coverage"].mean() if "memory_coverage" in grp.columns else 1.0

        mse = grp["forecast_mse"].mean() if "forecast_mse" in grp.columns else np.nan
        rank_ic = grp["rank_ic"].mean() if "rank_ic" in grp.columns else np.nan

        # Seed dispersion
        seed_means = grp.groupby("seed")["annualized_return"].mean()
        seed_disp = float(seed_means.std()) if len(seed_means) > 1 else 0.0

        rows.append({
            "Backbone": backbone,
            "Mode": mode,
            "Forecast MSE": round(float(mse), 6) if not np.isnan(mse) else "N/A",
            "Rank IC": round(float(rank_ic), 4) if not np.isnan(rank_ic) else "N/A",
            "Ann. Return": f"{ann_ret * 100.0:+.2f}%",
            "Sharpe": f"{sharpe:+.3f}",
            "Max DD": f"{max_dd * 100.0:.2f}%",
            "Turnover": f"{turnover:.2f}x",
            "Exposure": f"{exposure * 100.0:.1f}%",
            "Coverage": f"{coverage * 100.0:.1f}%",
            "Seed Dispersion": f"{seed_disp * 100.0:.2f}%",
            "_raw_return": ann_ret,
            "_raw_sharpe": sharpe,
        })

    df = pd.DataFrame(rows)
    # Order nicely
    mode_order = {"M0": 0, "M1": 1, "M2": 2, "M3": 3, "M4": 4}
    df["_order"] = df["Mode"].map(lambda m: mode_order.get(m, 99))
    df = df.sort_values(by=["Backbone", "_order"]).drop(columns=["_order", "_raw_return", "_raw_sharpe"])
    return df


def generate_table_b(contrast_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs Table B: Incremental Contribution.
    """
    rows = []
    for r in contrast_results:
        ci_sr = r["ci95_delta_sharpe"]
        ci_str = f"[{ci_sr[0]:+.3f}, {ci_sr[1]:+.3f}]"

        rows.append({
            "Candidate": r["candidate"],
            "Comparator": r["comparator"],
            "Contrast Label": r["label"],
            "N Pairs": r["n_pairs"],
            "Mean Delta Return": f"{r['mean_delta_return'] * 100.0:+.2f}%",
            "Mean Delta Sharpe": f"{r['mean_delta_sharpe']:+.3f}",
            "95% CI (Delta Sharpe)": ci_str,
            "Raw p": f"{r['p_value_raw']:.4f}",
            "Holm p": f"{r.get('p_value_holm', r['p_value_raw']):.4f}",
            "Significant?": "YES" if r.get("statistically_significant", False) else "No",
        })
    return pd.DataFrame(rows)


def save_reports(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    output_dir: Path,
    run_summary_md: str,
):
    """Saves tables and summaries to disk in CSV and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    table_a.to_csv(output_dir / "table_a_basic_matrix.csv", index=False)
    table_b.to_csv(output_dir / "table_b_incremental_contrasts.csv", index=False)

    with open(output_dir / "run_summary.md", "w") as f:
        f.write(run_summary_md)
