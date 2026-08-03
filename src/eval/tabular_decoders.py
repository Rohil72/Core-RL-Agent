from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DecoderSuiteConfig:
    """Frozen-state decoder controls for strong non-memory baselines."""

    target_column: str = "future_blended_alpha_63"
    elastic_alpha: float = 0.001
    elastic_l1_ratio: float = 0.25
    tree_iterations: int = 200
    tree_leaf_nodes: int = 15
    knn_neighbors: int = 25
    pca_components: int = 32
    random_seed: int = 7


def embedding_columns(frame: pd.DataFrame, prefix: str = "latent_") -> list[str]:
    """Return numerically ordered embedding columns only."""

    columns = [
        column
        for column in frame
        if column.startswith(prefix) and column.removeprefix(prefix).isdigit()
    ]
    return sorted(columns, key=lambda value: int(value.removeprefix(prefix)))


def _models(config: DecoderSuiteConfig, feature_count: int) -> dict[str, object]:
    components = max(1, min(config.pca_components, feature_count))
    return {
        "elastic_net": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    ElasticNet(
                        alpha=config.elastic_alpha,
                        l1_ratio=config.elastic_l1_ratio,
                        max_iter=5000,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=config.tree_iterations,
            max_leaf_nodes=config.tree_leaf_nodes,
            l2_regularization=1.0,
            random_state=config.random_seed,
        ),
        "pca_knn": Pipeline(
            [
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=components, random_state=config.random_seed)),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=config.knn_neighbors,
                        weights="distance",
                    ),
                ),
            ]
        ),
    }


def fit_predict_frozen_decoders(
    training: pd.DataFrame,
    query: pd.DataFrame,
    config: DecoderSuiteConfig | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Fit causal conventional decoders on the same frozen encoder states."""

    cfg = config or DecoderSuiteConfig()
    columns = embedding_columns(training)
    if not columns:
        raise ValueError("Training frame contains no latent embedding columns.")
    if columns != embedding_columns(query):
        raise ValueError("Training and query embedding contracts differ.")
    if cfg.target_column not in training:
        raise ValueError(f"Training frame lacks target: {cfg.target_column}")
    train = training.copy()
    query_start = pd.to_datetime(query["timestamp"], utc=True).min()
    if "outcome_available_timestamp" in train:
        available = pd.to_datetime(
            train["outcome_available_timestamp"], utc=True, errors="coerce"
        )
        train = train.loc[available <= query_start].copy()
    features = train[columns].apply(pd.to_numeric, errors="coerce")
    target = pd.to_numeric(train[cfg.target_column], errors="coerce")
    legal = np.isfinite(features.to_numpy(dtype=float)).all(axis=1) & np.isfinite(
        target.to_numpy(dtype=float)
    )
    features = features.loc[legal]
    target = target.loc[legal]
    query_features = query[columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(query_features.to_numpy(dtype=float)).all():
        raise FloatingPointError("Query embeddings contain non-finite values.")
    if len(features) <= max(cfg.knn_neighbors, cfg.pca_components):
        raise ValueError("Too few causal training rows for the decoder suite.")

    predictions: dict[str, np.ndarray] = {}
    for name, model in _models(cfg, len(columns)).items():
        model.fit(features, target)
        values = np.asarray(model.predict(query_features), dtype=float)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} produced non-finite predictions.")
        predictions[name] = values
    audit = {
        "target": cfg.target_column,
        "features": len(columns),
        "training_rows": int(len(features)),
        "query_rows": int(len(query_features)),
        "query_start": query_start.isoformat(),
        "models": sorted(predictions),
    }
    return predictions, audit
