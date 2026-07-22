"""Phase 1 analysis utilities

Produces latent projections, neighbor-similarity statistics, and opportunity ranking
reports under reports/research_phase1/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

try:
    import umap
    _has_umap = True
except Exception:
    _has_umap = False


OUT_DIR = Path("reports/research_phase1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_latents(latent_dir: Path = Path("reports/latent_analysis")) -> pd.DataFrame:
    files = sorted(latent_dir.glob("latents_*.parquet"))
    if not files:
        raise FileNotFoundError("No latent parquet files found in reports/latent_analysis")
    df = pd.read_parquet(files[-1])
    return df


def run_pca(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    latents = np.vstack(df["latent"].values)
    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(latents)
    for i in range(proj.shape[1]):
        df[f"pca_{i}"] = proj[:, i]
    df.attrs["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    return df


def run_umap(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    if not _has_umap:
        raise RuntimeError("umap-learn not installed")
    latents = np.vstack(df["latent"].values)
    reducer = umap.UMAP(n_components=n_components, random_state=7)
    proj = reducer.fit_transform(latents)
    for i in range(proj.shape[1]):
        df[f"umap_{i}"] = proj[:, i]
    return df


def neighbor_future_correlation(df: pd.DataFrame, k: int = 10, target_col: str = "true_future_max_return_63") -> dict:
    latents = np.vstack(df["latent"].values)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(latents)
    distances, indices = nbrs.kneighbors(latents)
    # ignore self neighbor at index 0
    neighbor_inds = indices[:, 1:]
    target_vals = df[target_col].fillna(0.0).to_numpy()
    corrs = []
    for i in range(len(df)):
        neigh_targets = target_vals[neighbor_inds[i]]
        if np.std(neigh_targets) == 0 or np.std([target_vals[i]]) == 0:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef([target_vals[i]], neigh_targets.mean(axis=0))[0, 1])
    mean_corr = float(np.nanmean(corrs))
    # random baseline: shuffle targets
    rnd = np.random.RandomState(7)
    shuffled = rnd.permutation(target_vals)
    rnd_corrs = []
    for i in range(len(df)):
        neigh_targets = shuffled[neighbor_inds[i]]
        if np.std(neigh_targets) == 0 or np.std([shuffled[i]]) == 0:
            rnd_corrs.append(0.0)
        else:
            rnd_corrs.append(np.corrcoef([shuffled[i]], neigh_targets.mean(axis=0))[0, 1])
    mean_rnd = float(np.nanmean(rnd_corrs))
    out = {"k": k, "mean_neighbor_corr": mean_corr, "random_baseline": mean_rnd}
    return out


def ranking_test(df: pd.DataFrame, score_col: str = "pred_future_max_return_63") -> dict:
    # compute bucket stats for top deciles
    df2 = df.copy()
    # Accept several common score column name variants emitted by the trainer
    candidate_scores = [score_col, "future_pred_max_return_63", "pred_future_max_63", "pred_future_max"]
    found_score = None
    for c in candidate_scores:
        if c in df2.columns:
            found_score = c
            break
    if found_score is None:
        raise KeyError(
            f"None of the expected score columns found. Checked: {candidate_scores}. Available columns: {list(df2.columns)}"
        )
    # ensure required true target columns exist
    required_targets = ["true_future_max_return_63", "true_future_min_return_63"]
    for t in required_targets:
        if t not in df2.columns:
            raise KeyError(f"Required target column missing: {t}")
    df2 = df2.dropna(subset=[found_score, "true_future_max_return_63", "true_future_min_return_63"])
    df2 = df2.sort_values(found_score, ascending=False)
    out = {}
    for pct in [0.1, 0.2, 0.3]:
        n = max(1, int(len(df2) * pct))
        top = df2.head(n)
        out[f"top_{int(pct*100)}_mean_max"] = float(top["true_future_max_return_63"].mean())
        out[f"top_{int(pct*100)}_mean_min"] = float(top["true_future_min_return_63"].mean())
        out[f"top_{int(pct*100)}_median_max"] = float(top["true_future_max_return_63"].median())
        out[f"top_{int(pct*100)}_win_rate"] = float((top["true_future_max_return_63"] > 0).mean())
    # overall baseline
    out["overall_mean_max"] = float(df2["true_future_max_return_63"].mean())
    out["overall_mean_min"] = float(df2["true_future_min_return_63"].mean())
    return out


def main():
    df = load_latest_latents()
    # ensure proper column names exist
    # try common variants
    if "true_future_max_return_63" not in df.columns:
        for c in df.columns:
            if c.endswith("future_max_return_63"):
                df[f"true_future_max_return_63"] = df[c]
    if "true_future_min_return_63" not in df.columns:
        for c in df.columns:
            if c.endswith("future_min_return_63"):
                df[f"true_future_min_return_63"] = df[c]

    df = run_pca(df, n_components=10)
    if _has_umap:
        try:
            df = run_umap(df, n_components=2)
        except Exception:
            pass

    neigh_res = neighbor_future_correlation(df, k=10)
    rank_res = ranking_test(df)

    summary = {"neighbor": neigh_res, "ranking": rank_res}
    with open(OUT_DIR / "phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_parquet(OUT_DIR / "latent_projections.parquet", index=False)
    print("Wrote analysis outputs to", OUT_DIR)


if __name__ == "__main__":
    main()

