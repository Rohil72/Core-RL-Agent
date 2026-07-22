"""
Latent Space Audit
==================
Five-part diagnostic suite for comparing "old" vs "new" encoder embeddings:

  1. UMAP / t-SNE visualization – do profitable market states form coherent
     neighbourhoods, or has everything collapsed into target-fitting clusters?
  2. Neighbourhood consistency – variance of future outcomes in the 25 nearest
     neighbours.  Lower variance <=> more retrieval-friendly.
  3. Neighbour overlap – how many of the k=25 nearest neighbours changed?
     >80% change <=> fundamentally different memory structure.
  4. Objective ablation tracker – compare retrieval quality across training
     checkpoints that progressively add richer targets.
  5. Trading-harness checkpoint selection – rank checkpoints by Phase-1 trading
     performance (profitable_cycle_rate / average_cycle_return), not by MSE.

Usage
-----
  python src/visualization/latent_space_audit.py \\
      --old-latents  reports/latent_analysis/latents_20260616_073741.parquet \\
      --new-latents  reports/latent_analysis/test_latents_20260621_184044.parquet \\
      --output-dir   reports/latent_space_audit

  # Optional: pass multiple checkpoint latents for ablation tracking
  python src/visualization/latent_space_audit.py \\
      --old-latents  <old.parquet> \\
      --new-latents  <new.parquet> \\
      --ablation-latents <ckpt1.parquet> <ckpt2.parquet> <ckpt3.parquet> \\
      --ablation-labels  "baseline" "add_horizon" "add_path_quality" \\
      --trading-reports  <report1.json> <report2.json> <report3.json>
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ── optional dimensionality-reduction libs ────────────────────────────────────
try:
    import umap  # type: ignore
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

# ── project root on sys.path ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("latent_space_audit")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── constants ─────────────────────────────────────────────────────────────────
K_NEIGHBORS = 25
TARGET_UPSIDE = "future_max_return_63"
TARGET_DOWNSIDE = "future_min_return_63"
TARGET_PATH = "event_upside_before_drawdown_126"
PROFITABLE_RETURN_THRESHOLD = 0.10   # 10 % upside -> "profitable"
RANDOM_SEED = 42

# colour palette (colourblind-safe)
C_OLD = "#4C72B0"    # muted blue
C_NEW = "#DD8452"    # muted orange
C_PROFIT = "#2CA02C" # green
C_LOSS = "#D62728"   # red
C_NEUTRAL = "#7F7F7F" # grey


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers  (mirror patterns in run_final_direction.py)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_latent_array_col(df: pd.DataFrame) -> pd.DataFrame:
    """If a 'latent' column holds numpy arrays/lists, expand to latent_0...N cols."""
    if "latent" not in df.columns:
        return df
    sample = df["latent"].dropna().iloc[0]
    if not isinstance(sample, (list, np.ndarray)):
        return df
    arr = np.vstack(
        df["latent"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy()
    )
    latent_df = pd.DataFrame(
        arr,
        columns=[f"latent_{i}" for i in range(arr.shape[1])],
        index=df.index,
    )
    return pd.concat([df.drop(columns=["latent"]), latent_df], axis=1)


def _extract_latent_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = sorted(
        [c for c in df.columns if c.startswith("latent_")],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if not cols:
        raise RuntimeError("No latent_* columns found.")
    return df[cols].to_numpy(dtype=float)


def _resolve_target(df: pd.DataFrame, col: str) -> "str | None":
    if col in df.columns:
        return col
    true_name = f"true_{col}"
    if true_name in df.columns:
        return true_name
    matches = sorted(c for c in df.columns if c.endswith(col) and c != col)
    return matches[0] if matches else None


def load_latents(path: "str | Path") -> pd.DataFrame:
    """Load a latent parquet file and normalise it."""
    path = Path(path)
    logger.info("Loading latents: %s", path)
    df = pd.read_parquet(path)
    df = _expand_latent_array_col(df)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str)
    # resolve target columns using true_ prefix convention
    for col in [TARGET_UPSIDE, TARGET_DOWNSIDE, TARGET_PATH]:
        resolved = _resolve_target(df, col)
        if resolved and resolved != col:
            df[col] = df[resolved]
        elif resolved is None:
            df[col] = np.nan
    n_latent_cols = len([c for c in df.columns if c.startswith("latent_")])
    target_counts = {c: int(df[c].notna().sum()) for c in [TARGET_UPSIDE, TARGET_DOWNSIDE, TARGET_PATH]}
    logger.info("  rows=%d  latent_dim=%d  targets: %s", len(df), n_latent_cols, target_counts)
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Dimensionality reduction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _reduce_umap(X: np.ndarray, n_components: int = 2, seed: int = RANDOM_SEED) -> np.ndarray:
    if not HAS_UMAP:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=30,
        min_dist=0.1,
        metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(X)


def _reduce_tsne(X: np.ndarray, n_components: int = 2, seed: int = RANDOM_SEED):
    """Returns (coords, subsample_idx_or_None)."""
    if not HAS_TSNE:
        raise ImportError("scikit-learn not installed.")
    max_tsne_rows = 8_000
    if X.shape[0] > max_tsne_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=max_tsne_rows, replace=False)
        logger.info("t-SNE: subsampling %d -> %d rows", X.shape[0], max_tsne_rows)
        coords = TSNE(
            n_components=n_components,
            perplexity=40,
            n_iter=1000,
            random_state=seed,
            init="pca",
        ).fit_transform(X[idx])
        return coords, idx
    coords = TSNE(
        n_components=n_components,
        perplexity=40,
        n_iter=1000,
        random_state=seed,
        init="pca",
    ).fit_transform(X)
    return coords, None


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 – UMAP / t-SNE Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _profit_labels(df: pd.DataFrame, threshold: float = PROFITABLE_RETURN_THRESHOLD) -> np.ndarray:
    """3-class label: 0=loss  1=neutral  2=profitable, based on TARGET_UPSIDE."""
    col = TARGET_UPSIDE
    if col not in df.columns:
        return np.full(len(df), 1, dtype=int)
    vals = df[col].to_numpy(dtype=float)
    labels = np.where(vals >= threshold, 2, np.where(vals <= -threshold, 0, 1))
    labels[~np.isfinite(vals)] = 1
    return labels


def part1_latent_visualisation(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    output_dir: Path,
    method: str = "auto",
) -> None:
    """
    Run UMAP or t-SNE on both old and new embeddings, coloured by:
      (a) profitable / neutral / loss outcome
      (b) ticker identity
    """
    logger.info("-- Part 1: Latent Space Visualisation --")

    # choose method
    if method == "auto":
        method = "umap" if HAS_UMAP else "tsne"
    if method == "umap" and not HAS_UMAP:
        logger.warning("umap-learn not found; falling back to t-SNE")
        method = "tsne"
    if method not in {"umap", "tsne"}:
        logger.error("Unknown method '%s'; skipping Part 1", method)
        return

    results: dict = {}
    for label, df in [("old", old_df), ("new", new_df)]:
        X = _extract_latent_matrix(df)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        logger.info("  %s encoder: %d x %d", label, *X_s.shape)

        tsne_idx = None
        if method == "umap":
            coords = _reduce_umap(X_s)
        else:
            coords, tsne_idx = _reduce_tsne(X_s)

        # Align labels to subsample if t-SNE subsampled
        if tsne_idx is not None:
            sub_df = df.iloc[tsne_idx].reset_index(drop=True)
        else:
            sub_df = df.reset_index(drop=True)

        profit_lab = _profit_labels(sub_df)
        tickers = sub_df["ticker"].to_numpy(str) if "ticker" in sub_df.columns else np.full(len(sub_df), "UNK")

        results[label] = {
            "coords": coords,
            "profit_lab": profit_lab,
            "tickers": tickers,
            "n_rows": len(df),
            "tsne_idx": tsne_idx,
        }

    method_up = method.upper()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Latent Space Comparison - Old vs New Encoder  ({method_up})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    profit_colours = {0: C_LOSS, 1: C_NEUTRAL, 2: C_PROFIT}
    profit_names = {0: "Loss (<-10%)", 1: "Neutral", 2: "Profitable (>+10%)"}

    for row_idx, (label, res) in enumerate(results.items()):
        coords = res["coords"]
        profit_lab = res["profit_lab"]
        tickers = res["tickers"]
        n = res["n_rows"]

        # left panel: coloured by profitability
        ax = axes[row_idx, 0]
        for cls_id, colour in profit_colours.items():
            mask = profit_lab == cls_id
            if mask.any():
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    c=colour,
                    alpha=0.35,
                    s=4,
                    rasterized=True,
                    label=profit_names[cls_id],
                )
        ax.set_title(f"{label.upper()} encoder - profitability  (n={n:,})", fontsize=11)
        ax.set_xlabel(f"{method_up}-1")
        ax.set_ylabel(f"{method_up}-2")
        ax.legend(fontsize=8, markerscale=3, loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])

        # right panel: coloured by ticker
        ax2 = axes[row_idx, 1]
        unique_tickers = np.unique(tickers)
        cmap = plt.cm.get_cmap("tab20", len(unique_tickers))
        for ti, tick in enumerate(unique_tickers):
            mask = tickers == tick
            ax2.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[cmap(ti)],
                alpha=0.35,
                s=4,
                rasterized=True,
                label=tick,
            )
        ax2.set_title(f"{label.upper()} encoder - ticker identity  (n={n:,})", fontsize=11)
        ax2.set_xlabel(f"{method_up}-1")
        ax2.set_ylabel(f"{method_up}-2")
        if len(unique_tickers) <= 22:
            ax2.legend(fontsize=6, markerscale=3, ncol=3, loc="upper right")
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = output_dir / f"part1_latent_viz_{method}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    # supplemental: upside return heatmap
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7))
    fig2.suptitle(
        f"Latent Space - Upside Return Heatmap ({method_up})\n"
        "Coherent regions = encoder is retrieval-friendly",
        fontsize=13,
        fontweight="bold",
    )
    for col_idx, (label, res) in enumerate(results.items()):
        ax = axes2[col_idx]
        source_df = old_df if label == "old" else new_df
        tsne_idx = res.get("tsne_idx")
        if tsne_idx is not None:
            vals = source_df.iloc[tsne_idx][TARGET_UPSIDE].to_numpy(dtype=float)
        else:
            vals = source_df[TARGET_UPSIDE].to_numpy(dtype=float)

        coords = res["coords"]
        finite = np.isfinite(vals)
        vmin, vmax = -0.30, 0.50
        sc = ax.scatter(
            coords[finite, 0],
            coords[finite, 1],
            c=vals[finite],
            cmap="RdYlGn",
            vmin=vmin,
            vmax=vmax,
            alpha=0.5,
            s=4,
            rasterized=True,
        )
        plt.colorbar(sc, ax=ax, label="future_max_return_63")
        ax.set_title(f"{label.upper()} encoder - upside return heatmap", fontsize=11)
        ax.set_xlabel(f"{method_up}-1")
        ax.set_ylabel(f"{method_up}-2")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path2 = output_dir / f"part1_upside_heatmap_{method}.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("  Saved: %s", out_path2)


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 – Neighbourhood Consistency
# ─────────────────────────────────────────────────────────────────────────────

def _knn_outcome_variance(
    X: np.ndarray,
    outcomes: np.ndarray,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    For each point, compute the variance of the TARGET outcome in its k nearest
    neighbours (excluding self).  Returns an array of per-point variances.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    # drop self (index 0 is always the point itself when fitting on same data)
    neighbor_indices = indices[:, 1: k + 1]

    variances = np.full(len(X), np.nan)
    for i, nb_idx in enumerate(neighbor_indices):
        nb_outcomes = outcomes[nb_idx]
        valid = nb_outcomes[np.isfinite(nb_outcomes)]
        if len(valid) >= 2:
            variances[i] = float(np.var(valid, ddof=1))
    return variances


def part2_neighbourhood_consistency(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    output_dir: Path,
    k: int = K_NEIGHBORS,
) -> dict:
    """
    Compute outcome-variance in k-NN neighbourhoods for old vs new encoder.
    Lower variance -> encoder is genuinely retrieval-friendly for that target.
    """
    logger.info("-- Part 2: Neighbourhood Consistency (k=%d) --", k)

    target_cols = [TARGET_UPSIDE, TARGET_DOWNSIDE, TARGET_PATH]
    results: dict = {}

    fig, axes = plt.subplots(1, len(target_cols), figsize=(6 * len(target_cols), 6))
    if len(target_cols) == 1:
        axes = [axes]

    summary_rows = []

    for col_idx, target in enumerate(target_cols):
        ax = axes[col_idx]
        target_data: dict = {}

        for label, df in [("old", old_df), ("new", new_df)]:
            X = _extract_latent_matrix(df)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            outcomes = df[target].to_numpy(dtype=float)

            n_finite = int(np.isfinite(outcomes).sum())
            if n_finite < k + 1:
                logger.warning(
                    "  %s/%s: only %d finite target values (need >%d); skipping",
                    label, target, n_finite, k,
                )
                target_data[label] = {"variances": np.array([np.nan]), "median": np.nan, "mean": np.nan}
                continue

            variances = _knn_outcome_variance(X_s, outcomes, k=k)
            finite_var = variances[np.isfinite(variances)]
            median_var = float(np.median(finite_var)) if len(finite_var) > 0 else np.nan
            mean_var = float(np.mean(finite_var)) if len(finite_var) > 0 else np.nan

            target_data[label] = {
                "variances": finite_var,
                "median": median_var,
                "mean": mean_var,
                "n_computed": len(finite_var),
            }
            logger.info(
                "  %s / %s: median_var=%.5f  mean_var=%.5f  (n=%d)",
                label, target, median_var, mean_var, len(finite_var),
            )
            summary_rows.append({
                "encoder": label,
                "target": target,
                "median_neighbor_variance": median_var,
                "mean_neighbor_variance": mean_var,
                "n_points": len(finite_var),
            })

        # plot
        for label, colour in [("old", C_OLD), ("new", C_NEW)]:
            if label not in target_data:
                continue
            finite_var = target_data[label].get("variances", np.array([]))
            if len(finite_var) == 0:
                continue
            log_var = np.log1p(finite_var)
            med = target_data[label].get("median", np.nan)
            ax.hist(
                log_var,
                bins=60,
                color=colour,
                alpha=0.55,
                label=f"{label} (med={med:.4f})",
                density=True,
            )
            if np.isfinite(med):
                ax.axvline(
                    np.log1p(med),
                    color=colour,
                    linestyle="--",
                    linewidth=1.5,
                )

        ax.set_title(f"k-NN outcome variance\n{target}", fontsize=10)
        ax.set_xlabel("log(1 + variance)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        results[target] = target_data

    fig.suptitle(
        f"Part 2 - Neighbourhood Consistency (k={k})\n"
        "Lower variance = more retrieval-friendly embedding",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = output_dir / "part2_neighbourhood_consistency.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    # summary bar chart
    if summary_rows:
        sumdf = pd.DataFrame(summary_rows)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(target_cols))
        width = 0.35
        for i, (label, colour) in enumerate([("old", C_OLD), ("new", C_NEW)]):
            sub = sumdf[sumdf["encoder"] == label]
            vals = []
            for t in target_cols:
                row = sub[sub["target"] == t]
                vals.append(float(row["median_neighbor_variance"].values[0]) if len(row) > 0 else 0.0)
            bars = ax2.bar(x + i * width - width / 2, vals, width, label=f"{label} encoder", color=colour, alpha=0.8)
            for bar, v in zip(bars, vals):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                         f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax2.set_xticks(x)
        ax2.set_xticklabels([t.replace("_", "\n") for t in target_cols], fontsize=8)
        ax2.set_ylabel("Median k-NN outcome variance")
        ax2.set_title("Neighbourhood Consistency Summary\n(lower = more retrieval-friendly)")
        ax2.legend()
        plt.tight_layout()
        out_path2 = output_dir / "part2_consistency_summary.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        logger.info("  Saved: %s", out_path2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Part 3 – Neighbour Overlap
# ─────────────────────────────────────────────────────────────────────────────

def _compute_knn_sets(X: np.ndarray, k: int) -> "list[set[int]]":
    """Return a list of k-NN index-sets (excluding self) for each point."""
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=-1)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)
    return [set(row[1: k + 1]) for row in indices]


def part3_neighbour_overlap(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    output_dir: Path,
    k: int = K_NEIGHBORS,
) -> dict:
    """
    For each query point present in BOTH old and new latent files (matched by
    ticker+timestamp), compute the Jaccard overlap of its k nearest neighbours.

    >80% neighbours different -> the memory structure has fundamentally changed.
    """
    logger.info("-- Part 3: Neighbour Overlap (k=%d) --", k)

    if "ticker" not in old_df.columns or "timestamp" not in old_df.columns:
        logger.warning("  Skipping Part 3: missing ticker/timestamp columns.")
        return {}

    # align rows between old and new
    old_ts = old_df["timestamp"].astype(str)
    new_ts = new_df["timestamp"].astype(str)
    old_key_vals = old_df["ticker"].astype(str) + "__" + old_ts
    new_key_vals = new_df["ticker"].astype(str) + "__" + new_ts

    shared_keys = set(old_key_vals) & set(new_key_vals)
    n_shared = len(shared_keys)
    logger.info("  Shared rows (ticker+timestamp match): %d", n_shared)

    if n_shared < k + 1:
        logger.warning(
            "  Only %d shared rows found (need >%d); overlap analysis skipped.", n_shared, k
        )
        return {"n_shared": n_shared, "error": "too_few_shared_rows"}

    old_mask = old_key_vals.isin(shared_keys).to_numpy()
    new_mask = new_key_vals.isin(shared_keys).to_numpy()

    old_sub = old_df[old_mask].copy()
    new_sub = new_df[new_mask].copy()
    old_sub["_key"] = old_key_vals[old_mask].values
    new_sub["_key"] = new_key_vals[new_mask].values

    old_sub = old_sub.sort_values("_key").reset_index(drop=True)
    new_sub = new_sub.sort_values("_key").reset_index(drop=True)

    X_old = _extract_latent_matrix(old_sub)
    X_new = _extract_latent_matrix(new_sub)

    scaler_old = StandardScaler()
    scaler_new = StandardScaler()
    X_old_s = scaler_old.fit_transform(X_old)
    X_new_s = scaler_new.fit_transform(X_new)

    logger.info("  Computing k-NN sets for old encoder ...")
    knn_old = _compute_knn_sets(X_old_s, k=k)
    logger.info("  Computing k-NN sets for new encoder ...")
    knn_new = _compute_knn_sets(X_new_s, k=k)

    # per-point overlap
    jaccard_scores = []
    pct_changed = []
    for s_old, s_new in zip(knn_old, knn_new):
        intersection = len(s_old & s_new)
        union = len(s_old | s_new)
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)
        pct_changed.append(1.0 - intersection / k if k > 0 else 1.0)

    jaccard_arr = np.array(jaccard_scores)
    pct_changed_arr = np.array(pct_changed)

    median_jaccard = float(np.median(jaccard_arr))
    median_pct_changed = float(np.median(pct_changed_arr))
    pct_80plus_changed = float(np.mean(pct_changed_arr >= 0.80))

    logger.info("  Median Jaccard overlap: %.3f  (1 = identical, 0 = no overlap)", median_jaccard)
    logger.info("  Median %% neighbours changed: %.1f%%", median_pct_changed * 100)
    logger.info("  Fraction of points with >=80%% neighbours changed: %.1f%%", pct_80plus_changed * 100)

    # verdict
    if pct_80plus_changed >= 0.80:
        verdict = "CRITICAL: >=80% of points have >=80% neighbour turnover -- memory structure fundamentally changed"
        verdict_colour = C_LOSS
    elif pct_80plus_changed >= 0.50:
        verdict = "WARNING: significant neighbour turnover -- retrieval reliability degraded"
        verdict_colour = "#FF7F0E"
    else:
        verdict = "OK: neighbour structure largely preserved"
        verdict_colour = C_PROFIT

    logger.info("  Verdict: %s", verdict)

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(jaccard_arr, bins=50, color=C_OLD, alpha=0.75, edgecolor="white")
    axes[0].axvline(median_jaccard, color="k", linestyle="--", label=f"Median={median_jaccard:.3f}")
    axes[0].set_xlabel("Jaccard overlap (old & new) / (old | new)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Neighbour Jaccard Overlap\n(k={k}, n={n_shared:,} shared rows)")
    axes[0].legend()

    axes[1].hist(pct_changed_arr * 100, bins=50, color=C_NEW, alpha=0.75, edgecolor="white")
    axes[1].axvline(median_pct_changed * 100, color="k", linestyle="--",
                    label=f"Median={median_pct_changed * 100:.1f}%")
    axes[1].axvline(80, color=C_LOSS, linestyle=":", linewidth=2, label="80% threshold")
    axes[1].set_xlabel("% Neighbours Changed")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Neighbour Turnover per Point")
    axes[1].legend()

    sorted_pct = np.sort(pct_changed_arr * 100)
    cdf = np.arange(1, len(sorted_pct) + 1) / len(sorted_pct)
    axes[2].plot(sorted_pct, cdf, color=C_OLD, linewidth=2)
    axes[2].axvline(80, color=C_LOSS, linestyle=":", linewidth=2, label="80% threshold")
    axes[2].axhline(pct_80plus_changed, color=C_LOSS, linestyle="--", linewidth=1.5,
                    label=f"{pct_80plus_changed * 100:.1f}% of points")
    axes[2].set_xlabel("% Neighbours Changed")
    axes[2].set_ylabel("CDF")
    axes[2].set_title("Cumulative Distribution")
    axes[2].legend()
    axes[2].set_xlim([0, 100])
    axes[2].set_ylim([0, 1])

    fig.suptitle(
        f"Part 3 - Neighbour Overlap Analysis\n{verdict}",
        fontsize=12,
        fontweight="bold",
        color=verdict_colour,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = output_dir / "part3_neighbour_overlap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    return {
        "n_shared": n_shared,
        "median_jaccard": median_jaccard,
        "median_pct_changed": median_pct_changed,
        "pct_80plus_changed": pct_80plus_changed,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part 4 – Objective Ablation Tracker
# ─────────────────────────────────────────────────────────────────────────────

def _retrieval_quality_score(df: pd.DataFrame, k: int = K_NEIGHBORS) -> dict:
    """
    Scalar retrieval-quality proxy: measure how much the k-NN mean outcome
    varies ACROSS clusters (between-cluster variance) vs within neighbourhoods
    (within-neighbourhood variance).

    High between / low within = good retrieval structure.
    """
    from sklearn.cluster import KMeans

    X = _extract_latent_matrix(df)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    outcomes = df[TARGET_UPSIDE].to_numpy(dtype=float)
    finite = np.isfinite(outcomes)
    if finite.sum() < 50:
        return {"within_var": np.nan, "between_var": np.nan, "retrieval_score": np.nan}

    variances = _knn_outcome_variance(X_s, outcomes, k=k)
    within_var = float(np.nanmedian(variances))

    km = KMeans(n_clusters=8, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X_s)
    cluster_means = []
    for cid in range(8):
        mask = (labels == cid) & finite
        if mask.sum() >= 2:
            cluster_means.append(outcomes[mask].mean())
    between_var = float(np.var(cluster_means)) if len(cluster_means) >= 2 else 0.0

    retrieval_score = between_var / (within_var + 1e-9)
    return {
        "within_var": within_var,
        "between_var": between_var,
        "retrieval_score": retrieval_score,
    }


def part4_objective_ablation(
    ablation_latents: "list[pd.DataFrame]",
    ablation_labels: "list[str]",
    output_dir: Path,
    k: int = K_NEIGHBORS,
) -> dict:
    """
    Progressively evaluate retrieval quality as richer targets are added.
    Expects a list of latent DataFrames (one per ablation checkpoint) and
    matching human-readable labels.
    """
    logger.info("-- Part 4: Objective Ablation Tracker (%d checkpoints) --", len(ablation_latents))

    if not ablation_latents:
        logger.info("  No ablation latents provided; skipping Part 4.")
        return {}

    rows = []
    for label, df in zip(ablation_labels, ablation_latents):
        logger.info("  Evaluating checkpoint: %s", label)
        metrics = _retrieval_quality_score(df, k=k)
        rows.append({"checkpoint": label, **metrics})
        logger.info(
            "    within_var=%.5f  between_var=%.5f  retrieval_score=%.4f",
            metrics["within_var"], metrics["between_var"], metrics["retrieval_score"],
        )

    df_results = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(rows))

    metric_specs = [
        ("within_var", "Median k-NN outcome variance (lower=better)", False),
        ("between_var", "Cluster-mean variance (higher=better)", True),
        ("retrieval_score", "Between / Within composite (higher=better)", True),
    ]

    for ax, (col, ylabel, higher_better) in zip(axes, metric_specs):
        vals = df_results[col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) == 0:
            ax.set_title(f"{ylabel}\n(no data)")
            continue
        best_val = np.nanmax(vals) if higher_better else np.nanmin(vals)
        colours = [C_PROFIT if abs(v - best_val) < 1e-12 else C_OLD for v in vals]
        bars = ax.bar(x, vals, color=colours, alpha=0.85, edgecolor="white")
        val_range = np.nanmax(finite_vals) - np.nanmin(finite_vals) if len(finite_vals) > 1 else 1.0
        offset = val_range * 0.01 if val_range > 0 else 0.001
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{v:.4f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(df_results["checkpoint"].tolist(), rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    fig.suptitle(
        "Part 4 - Objective Ablation: Retrieval Quality vs Training Objective\n"
        "Green bar = best checkpoint for retrieval",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = output_dir / "part4_objective_ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    return {"checkpoints": df_results.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────────────────────
# Part 5 – Trading-Harness Checkpoint Selection
# ─────────────────────────────────────────────────────────────────────────────

def _parse_trading_report(path: "str | Path") -> dict:
    """
    Parse a training report JSON (produced by train_cycle_model.write_report).
    Extracts validation/test trading metrics.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    report = payload.get("report", payload)

    def _get_split(split: str) -> dict:
        if split in report:
            return report[split]
        for v in report.values():
            if isinstance(v, dict) and split in v:
                return v[split]
        return {}

    for split in ("test", "val", "train"):
        m = _get_split(split)
        if m:
            return {
                "split_used": split,
                "future_target_mae": float(m.get("future_target_mae", float("nan"))),
                "profitable_cycle_rate": float(m.get("profitable_cycle_rate", float("nan"))),
                "average_cycle_return": float(m.get("average_cycle_return", float("nan"))),
                "cycle_f1": float(m.get("cycle_f1", float("nan"))),
                "catastrophic_cycle_rate": float(m.get("catastrophic_cycle_rate", float("nan"))),
                "moving_avg_cycle_return": float(m.get("moving_avg_cycle_return", float("nan"))),
            }
    return {}


def _trading_score(metrics: dict) -> float:
    """
    Composite trading score for checkpoint selection.
    Combines profitable_cycle_rate and average_cycle_return.
    Penalises catastrophic_cycle_rate.
    """
    pcr = metrics.get("profitable_cycle_rate", 0.0)
    acr = metrics.get("average_cycle_return", 0.0)
    ccr = metrics.get("catastrophic_cycle_rate", 1.0)
    if not all(np.isfinite([pcr, acr, ccr])):
        return float("-inf")
    return float(0.5 * pcr + 0.5 * float(np.tanh(acr * 5)) - 0.3 * ccr)


def part5_checkpoint_selection(
    trading_report_paths: "list[str | Path]",
    checkpoint_labels: "list[str]",
    output_dir: Path,
) -> dict:
    """
    Compare checkpoints by trading-harness performance (Phase-1 criterion),
    NOT by training loss.
    """
    logger.info("-- Part 5: Trading-Harness Checkpoint Selection --")

    if not trading_report_paths:
        logger.info("  No trading reports provided; skipping Part 5.")
        return {}

    rows = []
    for path, label in zip(trading_report_paths, checkpoint_labels):
        try:
            metrics = _parse_trading_report(path)
        except Exception as exc:
            logger.warning("  Failed to parse %s: %s", path, exc)
            metrics = {}
        metrics["checkpoint"] = label
        metrics["trading_score"] = _trading_score(metrics)
        rows.append(metrics)
        logger.info(
            "  %s: mae=%.4f  profitable_rate=%.3f  avg_return=%.4f  trading_score=%.4f",
            label,
            metrics.get("future_target_mae", float("nan")),
            metrics.get("profitable_cycle_rate", float("nan")),
            metrics.get("average_cycle_return", float("nan")),
            metrics.get("trading_score", float("nan")),
        )

    df_ckpt = pd.DataFrame(rows).sort_values("trading_score", ascending=False)
    best = df_ckpt.iloc[0]
    logger.info("  Best checkpoint by trading score: %s (score=%.4f)", best["checkpoint"], best["trading_score"])

    # plot
    metric_pairs = [
        ("future_target_mae", "Future-target MAE (lower=old criterion)", False),
        ("profitable_cycle_rate", "Profitable Cycle Rate (higher=better)", True),
        ("average_cycle_return", "Avg Cycle Return (higher=better)", True),
        ("trading_score", "Composite Trading Score (higher=best)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes_flat = axes.flatten()
    labels_x = [r["checkpoint"] for r in rows]
    x = np.arange(len(rows))

    for ax, (metric, ylabel, is_higher_better) in zip(axes_flat, metric_pairs):
        vals = np.array([r.get(metric, float("nan")) for r in rows], dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals) == 0:
            ax.set_title(f"{ylabel}\n(no data)")
            continue
        best_idx = int(np.nanargmax(vals) if is_higher_better else np.nanargmin(vals))
        colours = [C_PROFIT if i == best_idx else C_OLD for i in range(len(vals))]
        bars = ax.bar(x, vals, color=colours, alpha=0.85, edgecolor="white")
        val_range = np.nanmax(finite_vals) - np.nanmin(finite_vals) if len(finite_vals) > 1 else 1.0
        offset = val_range * 0.01 if val_range > 0 else 0.001
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                        f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        best_label = labels_x[best_idx]
        ax.set_title(f"{ylabel}\nBest: {best_label}")
        ax.axhline(0, color="grey", linewidth=0.5)

    fig.suptitle(
        "Part 5 - Checkpoint Selection: Training Loss vs Trading Performance\n"
        "Green bar = winner under each criterion",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = output_dir / "part5_checkpoint_selection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", out_path)

    table_path = output_dir / "part5_checkpoint_ranking.csv"
    df_ckpt.to_csv(table_path, index=False)
    logger.info("  Ranking saved: %s", table_path)

    return {
        "ranked_checkpoints": df_ckpt.to_dict(orient="records"),
        "best_checkpoint": str(best["checkpoint"]),
        "best_trading_score": float(best["trading_score"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(
    output_dir: Path,
    part2_results: dict,
    part3_results: dict,
    part4_results: dict,
    part5_results: dict,
) -> None:
    """Write a markdown summary of all findings."""
    lines = [
        "# Latent Space Audit - Summary Report",
        "",
        "> Generated by `src/visualization/latent_space_audit.py`",
        "",
        "---",
        "",
        "## Part 1 - Latent Visualisation (UMAP / t-SNE)",
        "",
        "See `part1_latent_viz_*.png` and `part1_upside_heatmap_*.png`.",
        "",
        "**How to interpret:**",
        "- If profitable states (green) form **coherent compact regions** -> encoder is retrieval-friendly.",
        "- If profitable states are **scattered uniformly** -> embedding is dominated by target-fitting noise.",
        "- If new encoder shows **worse profitability clustering** than old -> richer objectives degraded retrieval.",
        "",
        "---",
        "",
        "## Part 2 - Neighbourhood Consistency",
        "",
        "| Target | Old Median Var | New Median Var | Direction |",
        "|--------|---------------|---------------|-----------|",
    ]

    for target, data in part2_results.items():
        old_med = data.get("old", {}).get("median", float("nan"))
        new_med = data.get("new", {}).get("median", float("nan"))
        if np.isfinite(old_med) and np.isfinite(new_med):
            direction = "improved (new < old)" if new_med < old_med else "degraded (new > old)"
        else:
            direction = "insufficient data"
        lines.append(f"| `{target}` | {old_med:.5f} | {new_med:.5f} | {direction} |")

    lines += [
        "",
        "**Lower k-NN outcome variance = encoder groups similar-outcome states more tightly = better for retrieval.**",
        "",
        "---",
        "",
        "## Part 3 - Neighbour Overlap",
        "",
    ]
    if part3_results and "error" not in part3_results:
        n_shared = part3_results.get("n_shared", 0)
        jac = part3_results.get("median_jaccard", float("nan"))
        pct_chg = part3_results.get("median_pct_changed", float("nan"))
        pct_80 = part3_results.get("pct_80plus_changed", float("nan"))
        verdict = part3_results.get("verdict", "N/A")
        lines += [
            f"- Shared rows matched by ticker+timestamp: **{n_shared:,}**",
            f"- Median Jaccard overlap: **{jac:.3f}** (1 = identical, 0 = completely different)",
            f"- Median % neighbours changed: **{pct_chg * 100:.1f}%**",
            f"- Points with >=80% neighbour turnover: **{pct_80 * 100:.1f}%**",
            f"- **Verdict:** {verdict}",
        ]
    else:
        lines.append("Skipped (no shared ticker+timestamp rows found or too few rows).")

    lines += [
        "",
        "---",
        "",
        "## Part 4 - Objective Ablation",
        "",
    ]
    checkpoints = part4_results.get("checkpoints", [])
    if checkpoints:
        lines.append("| Checkpoint | Within-var | Between-var | Retrieval Score |")
        lines.append("|------------|-----------|------------|----------------|")
        for r in checkpoints:
            lines.append(
                f"| {r['checkpoint']} | {r['within_var']:.5f} | {r['between_var']:.5f} | {r['retrieval_score']:.4f} |"
            )
        lines.append("")
        lines.append("The objective variant that **destroys** retrieval will show a sudden drop in retrieval_score.")
    else:
        lines.append("No ablation latents provided.")

    lines += [
        "",
        "---",
        "",
        "## Part 5 - Checkpoint Selection by Trading Performance",
        "",
    ]
    if part5_results:
        best = part5_results.get("best_checkpoint", "N/A")
        score = part5_results.get("best_trading_score", float("nan"))
        lines += [
            f"**Best checkpoint (trading harness):** `{best}` (composite score = {score:.4f})",
            "",
            "Full ranking saved in `part5_checkpoint_ranking.csv`.",
            "",
            "**Rule:** Select checkpoints by the Phase-1 trading harness",
            "(profitable_cycle_rate, average_cycle_return), NOT by future_target_mae.",
            "A checkpoint that loses 2% on MAE but gains 15% in trading",
            "performance is the better encoder for this project.",
        ]
    else:
        lines.append("No trading reports provided.")

    lines += ["", "---", ""]

    out_path = output_dir / "latent_space_audit_summary.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Summary written: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _default_latent_paths():
    """Auto-detect most recent old and new latent files in the standard directory."""
    latent_dir = _PROJECT_ROOT / "reports" / "latent_analysis"
    if not latent_dir.exists():
        return None, None

    train_files = sorted(latent_dir.glob("latents_*.parquet"))
    test_files = sorted(latent_dir.glob("test_latents_*.parquet"))

    old_path = train_files[-1] if train_files else None
    new_path = test_files[-1] if test_files else None
    return old_path, new_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latent Space Audit: compare old vs new encoder embeddings."
    )
    parser.add_argument(
        "--old-latents",
        default=None,
        help="Path to old encoder latent parquet (defaults to most recent latents_*.parquet)",
    )
    parser.add_argument(
        "--new-latents",
        default=None,
        help="Path to new encoder latent parquet (defaults to most recent test_latents_*.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "reports" / "latent_space_audit"),
        help="Directory for output plots and reports",
    )
    parser.add_argument(
        "--method",
        choices=["umap", "tsne", "auto"],
        default="auto",
        help="Dimensionality-reduction method (default: auto -> umap if available, else tsne)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_NEIGHBORS,
        help=f"Number of nearest neighbours (default: {K_NEIGHBORS})",
    )
    parser.add_argument(
        "--ablation-latents",
        nargs="*",
        default=[],
        help="Paths to ablation-checkpoint latent parquets (Part 4)",
    )
    parser.add_argument(
        "--ablation-labels",
        nargs="*",
        default=[],
        help="Human-readable labels for each ablation checkpoint (same order as --ablation-latents)",
    )
    parser.add_argument(
        "--trading-reports",
        nargs="*",
        default=[],
        help="Paths to training report JSON files for checkpoint selection (Part 5)",
    )
    parser.add_argument(
        "--trading-labels",
        nargs="*",
        default=[],
        help="Labels for trading report JSON files (same order as --trading-reports)",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip Part 1 (expensive UMAP/t-SNE) and run only Parts 2-5",
    )
    args = parser.parse_args()

    # resolve paths
    if args.old_latents is None or args.new_latents is None:
        auto_old, auto_new = _default_latent_paths()
        if args.old_latents is None:
            if auto_old is None:
                parser.error("No --old-latents provided and no latents_*.parquet found.")
            args.old_latents = str(auto_old)
            logger.info("Auto-detected old latents: %s", args.old_latents)
        if args.new_latents is None:
            if auto_new is None:
                parser.error("No --new-latents provided and no test_latents_*.parquet found.")
            args.new_latents = str(auto_new)
            logger.info("Auto-detected new latents: %s", args.new_latents)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # load data
    old_df = load_latents(args.old_latents)
    new_df = load_latents(args.new_latents)

    # Part 1
    if not args.skip_viz:
        try:
            part1_latent_visualisation(old_df, new_df, output_dir, method=args.method)
        except Exception as exc:
            logger.exception("Part 1 failed: %s", exc)
    else:
        logger.info("Part 1 skipped (--skip-viz).")

    # Part 2
    try:
        part2_results = part2_neighbourhood_consistency(old_df, new_df, output_dir, k=args.k)
    except Exception as exc:
        logger.exception("Part 2 failed: %s", exc)
        part2_results = {}

    # Part 3
    try:
        part3_results = part3_neighbour_overlap(old_df, new_df, output_dir, k=args.k)
    except Exception as exc:
        logger.exception("Part 3 failed: %s", exc)
        part3_results = {}

    # Part 4
    ablation_dfs = []
    ablation_labels = list(args.ablation_labels)
    for i, p in enumerate(args.ablation_latents):
        try:
            adf = load_latents(p)
            ablation_dfs.append(adf)
            if i >= len(ablation_labels):
                ablation_labels.append(Path(p).stem)
        except Exception as exc:
            logger.warning("Failed to load ablation latent %s: %s", p, exc)

    try:
        part4_results = part4_objective_ablation(ablation_dfs, ablation_labels, output_dir, k=args.k)
    except Exception as exc:
        logger.exception("Part 4 failed: %s", exc)
        part4_results = {}

    # Part 5
    trading_labels = list(args.trading_labels)
    for i, p in enumerate(args.trading_reports):
        if i >= len(trading_labels):
            trading_labels.append(Path(p).stem)

    try:
        part5_results = part5_checkpoint_selection(
            args.trading_reports, trading_labels, output_dir
        )
    except Exception as exc:
        logger.exception("Part 5 failed: %s", exc)
        part5_results = {}

    # summary
    write_summary(output_dir, part2_results, part3_results, part4_results, part5_results)
    logger.info("Latent Space Audit complete. Results in: %s", output_dir)


if __name__ == "__main__":
    main()
