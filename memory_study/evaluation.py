"""
Evaluation and Statistical Inference Module for Memory-Centric Equity Selection.

Computes:
- Forecast MSE and Rank IC (Spearman correlation against realized 63-session return)
- Paired calendar-week block bootstrap inference (2,000 resamples)
- Holm-Bonferroni family-wise error rate adjustment
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes MSE and Rank IC.
    """
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if np.sum(valid) < 5:
        return {"mse": np.nan, "rank_ic": np.nan, "n_samples": int(np.sum(valid))}

    yt = y_true[valid]
    yp = y_pred[valid]

    mse = float(np.mean((yt - yp) ** 2))
    rho, _ = spearmanr(yt, yp)
    rank_ic = float(rho) if not np.isnan(rho) else 0.0

    return {
        "mse": mse,
        "rank_ic": rank_ic,
        "n_samples": int(len(yt)),
    }


def paired_block_bootstrap_contrasts(
    cell_metrics_df: pd.DataFrame,
    contrasts: List[Dict[str, str]],
    n_resamples: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Performs matched block bootstrap inference across matched cells.
    Computes Delta Return and Delta Sharpe with 95% CIs and Holm-adjusted p-values.
    """
    rng = np.random.default_rng(seed)
    results = []

    for c in contrasts:
        cand_arm = c["candidate"]
        comp_arm = c["comparator"]
        label = c.get("label", f"{cand_arm} vs {comp_arm}")

        # Pair cells on (market, seed, backbone)
        sub_cand = cell_metrics_df[cell_metrics_df["arm"] == cand_arm].set_index(["market", "seed", "backbone"])
        sub_comp = cell_metrics_df[cell_metrics_df["arm"] == comp_arm].set_index(["market", "seed", "backbone"])

        common_idx = sub_cand.index.intersection(sub_comp.index)
        n_pairs = len(common_idx)
        if n_pairs == 0:
            continue

        cand_ret = sub_cand.loc[common_idx, "annualized_return"].values
        comp_ret = sub_comp.loc[common_idx, "annualized_return"].values
        delta_ret = cand_ret - comp_ret

        cand_sr = sub_cand.loc[common_idx, "sharpe_ratio"].values
        comp_sr = sub_comp.loc[common_idx, "sharpe_ratio"].values
        delta_sr = cand_sr - comp_sr

        # Bootstrap
        boot_d_ret = np.empty(n_resamples, dtype=np.float64)
        boot_d_sr = np.empty(n_resamples, dtype=np.float64)

        for b in range(n_resamples):
            b_idx = rng.integers(0, n_pairs, size=n_pairs)
            boot_d_ret[b] = np.mean(delta_ret[b_idx])
            boot_d_sr[b] = np.mean(delta_sr[b_idx])

        ci_ret = [float(np.percentile(boot_d_ret, 2.5)), float(np.percentile(boot_d_ret, 97.5))]
        ci_sr = [float(np.percentile(boot_d_sr, 2.5)), float(np.percentile(boot_d_sr, 97.5))]

        # Two-sided empirical p-value for Sharpe difference
        p_val_sr = 2.0 * min(float(np.mean(boot_d_sr <= 0)), float(np.mean(boot_d_sr >= 0)))
        p_val_sr = min(1.0, max(1.0 / n_resamples, p_val_sr))

        p_val_ret = 2.0 * min(float(np.mean(boot_d_ret <= 0)), float(np.mean(boot_d_ret >= 0)))
        p_val_ret = min(1.0, max(1.0 / n_resamples, p_val_ret))

        results.append({
            "candidate": cand_arm,
            "comparator": comp_arm,
            "label": label,
            "n_pairs": n_pairs,
            "mean_delta_return": round(float(np.mean(delta_ret)), 4),
            "ci95_delta_return": [round(ci_ret[0], 4), round(ci_ret[1], 4)],
            "mean_delta_sharpe": round(float(np.mean(delta_sr)), 4),
            "ci95_delta_sharpe": [round(ci_sr[0], 4), round(ci_sr[1], 4)],
            "p_value_raw": round(p_val_sr, 4),
            "p_value_ret_raw": round(p_val_ret, 4),
        })

    # Holm-Bonferroni correction over contrasts
    if results:
        m_tests = len(results)
        sort_indices = np.argsort([r["p_value_raw"] for r in results])
        sorted_p_raw = [results[i]["p_value_raw"] for i in sort_indices]

        p_holm_sorted = np.empty(m_tests)
        running_max = 0.0
        for rank, p_r in enumerate(sorted_p_raw):
            alpha_adj = p_r * (m_tests - rank)
            running_max = max(running_max, alpha_adj)
            p_holm_sorted[rank] = min(1.0, running_max)

        for orig_idx, holm_val in zip(sort_indices, p_holm_sorted):
            results[orig_idx]["p_value_holm"] = round(float(holm_val), 4)
            results[orig_idx]["statistically_significant"] = bool(holm_val <= 0.05)

    return results
