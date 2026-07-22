"""
Analyze future-outcome-only test-set latent vectors.

The audit tests whether nearby latent vectors have similar realized future
opportunity profiles and whether clusters expose distinct market states.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2" / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_LATENT_GLOB = str(ROOT / "reports" / "latent_analysis" / "test_latents_*.parquet")
TARGETS = [
    "future_max_return_63",
    "future_min_return_63",
    "event_upside_before_drawdown_126",
]


def load_latents(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise SystemExit("No latent files found for pattern: " + glob_pattern)
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _target_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in TARGETS if col not in df.columns]
    if missing:
        raise SystemExit("Missing required future target columns: " + ", ".join(missing))
    return df.dropna(subset=TARGETS).copy()


def _mean_abs_target_distance(y: np.ndarray, indices: np.ndarray) -> float:
    distances = []
    for row, neighbors in enumerate(indices):
        distances.append(np.mean(np.abs(y[neighbors] - y[row]), axis=0))
    return float(np.mean(np.vstack(distances)))


def neighbor_similarity(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> dict[str, float]:
    n_neighbors = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    _, idx = nn.kneighbors(X)
    knn_idx = idx[:, 1:]
    rng = np.random.default_rng(seed)
    random_idx = rng.integers(0, len(X), size=knn_idx.shape)

    metrics: dict[str, float] = {
        "knn_mean_abs_target_distance": _mean_abs_target_distance(y, knn_idx),
        "random_mean_abs_target_distance": _mean_abs_target_distance(y, random_idx),
    }
    metrics["knn_distance_improvement"] = (
        metrics["random_mean_abs_target_distance"]
        - metrics["knn_mean_abs_target_distance"]
    )

    for target_idx, target in enumerate(TARGETS):
        knn_mean = np.mean(y[knn_idx, target_idx], axis=1)
        random_mean = np.mean(y[random_idx, target_idx], axis=1)
        metrics[f"{target}_knn_corr"] = _safe_corr(y[:, target_idx], knn_mean)
        metrics[f"{target}_random_corr"] = _safe_corr(y[:, target_idx], random_mean)
    return metrics


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cluster_analysis(df: pd.DataFrame, X: np.ndarray, y: np.ndarray, k: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    k = min(k, max(2, len(df) - 1))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(X)
    clustered = df.copy()
    clustered["cluster"] = labels

    summary = (
        clustered.groupby("cluster")[TARGETS]
        .agg(["count", "mean", "median", "std"])
        .sort_index()
    )
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()

    metrics: dict[str, float] = {
        "cluster_count": float(k),
        "silhouette_score": float(silhouette_score(X, labels)) if len(set(labels)) > 1 else 0.0,
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else 0.0,
    }
    for target in TARGETS:
        groups = [g[target].dropna().to_numpy() for _, g in clustered.groupby("cluster")]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) > 1:
            stat, p_value = f_oneway(*groups)
            metrics[f"{target}_cluster_anova_f"] = float(stat)
            metrics[f"{target}_cluster_anova_p"] = float(p_value)
        metrics[f"{target}_cluster_mean_range"] = float(
            summary[f"{target}_mean"].max() - summary[f"{target}_mean"].min()
        )
    return summary, metrics


def write_projection(df: pd.DataFrame, X: np.ndarray) -> None:
    n_components = min(3, X.shape[1], len(df))
    pca = PCA(n_components=n_components)
    z = pca.fit_transform(X)
    out = df[["ticker", "timestamp", *TARGETS]].copy()
    for idx in range(n_components):
        out[f"pca_{idx}"] = z[:, idx]
    out.to_parquet(OUT / "future_only_latent_projection.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", default=DEFAULT_LATENT_GLOB)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = _target_frame(load_latents(args.latent_dir))
    latent_cols = sorted(
        [col for col in df.columns if col.startswith("latent_")],
        key=lambda col: int(col.split("_", 1)[1]),
    )
    if not latent_cols:
        raise SystemExit("No latent_ columns found in latent files")

    X = StandardScaler().fit_transform(df[latent_cols].to_numpy(dtype=float))
    y = df[TARGETS].to_numpy(dtype=float)

    summary, cluster_metrics = cluster_analysis(df, X, y, args.clusters, args.seed)
    neighbor_metrics = neighbor_similarity(X, y, args.k_neighbors, args.seed)
    write_projection(df, X)

    metrics = {
        "sample_count": int(len(df)),
        "latent_dim": int(len(latent_cols)),
        "targets": TARGETS,
        **neighbor_metrics,
        **cluster_metrics,
    }
    summary.to_csv(OUT / "future_only_cluster_profiles.csv", index=False)
    with open(OUT / "future_only_latent_state_evidence.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "# Future-Only Latent State Evidence",
        "",
        f"- Samples: {metrics['sample_count']}",
        f"- Latent dimensions: {metrics['latent_dim']}",
        f"- k-NN target distance: {metrics['knn_mean_abs_target_distance']:.6f}",
        f"- Random target distance: {metrics['random_mean_abs_target_distance']:.6f}",
        f"- Distance improvement: {metrics['knn_distance_improvement']:.6f}",
        f"- Silhouette score: {metrics['silhouette_score']:.6f}",
        f"- Calinski-Harabasz score: {metrics['calinski_harabasz_score']:.6f}",
        "",
        "## Target Neighbor Correlations",
    ]
    for target in TARGETS:
        lines.append(
            f"- {target}: k-NN={metrics[f'{target}_knn_corr']:.6f}, "
            f"random={metrics[f'{target}_random_corr']:.6f}"
        )
    lines.append("")
    lines.append("## Cluster Target Separation")
    for target in TARGETS:
        lines.append(
            f"- {target}: mean range={metrics[f'{target}_cluster_mean_range']:.6f}, "
            f"ANOVA p={metrics.get(f'{target}_cluster_anova_p', 1.0):.6g}"
        )
    (OUT / "future_only_latent_state_evidence.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print("Future-only latent audit artifacts saved to", OUT)


if __name__ == "__main__":
    main()
