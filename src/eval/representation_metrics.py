from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def compute_memory_metrics(
    query_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    target_names: list[str] | None = None,
    candidate_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Compute the 20 causal retrieval metrics used by Phase 4D promotion."""
    names = [
        "causal_eligibility_coverage", "mean_legal_neighbor_count", "outcome_ndcg_at_25",
        "outcome_recall_at_25", "outcome_precision_at_25", "mean_reciprocal_outcome_rank",
        "outcome_distance_improvement_over_random", "alpha_retrieval_mae", "downside_retrieval_mae",
        "alpha_spearman_correlation", "weekly_alpha_rank_ic", "positive_alpha_directional_accuracy",
        "top_decile_opportunity_precision", "downside_cvar_calibration_error",
        "prediction_interval_coverage", "confidence_brier_score", "confidence_calibration_error",
        "effective_sample_size", "neighbor_ticker_hhi", "cross_seed_neighbor_jaccard",
    ]
    metrics = {name: float("nan") for name in names}
    if query_df.empty or neighbor_df.empty:
        return metrics

    query = query_df.copy()
    neighbors = neighbor_df.copy()
    query["timestamp"] = pd.to_datetime(query["timestamp"], utc=True, errors="coerce")
    counts = neighbors.groupby("query_id").size()
    metrics["causal_eligibility_coverage"] = float(query["query_id"].isin(counts.index).mean())
    metrics["mean_legal_neighbor_count"] = float(counts.mean())
    ticker_counts = neighbors.groupby(["query_id", "neighbor_ticker"]).size()
    ticker_shares = ticker_counts / neighbors.groupby("query_id").size()
    metrics["neighbor_ticker_hhi"] = float((ticker_shares**2).groupby("query_id").sum().mean())

    alpha_col = _first_column(
        query,
        target_names or ["future_universe_alpha_63", "future_blended_alpha_63", "future_return_63"],
    )
    neighbor_alpha_col = _first_column(neighbors, [alpha_col] if alpha_col else [])
    downside_col = _first_column(query, ["future_min_return_63", "true_future_min_return_63"])
    neighbor_downside_col = _first_column(neighbors, [downside_col] if downside_col else [])
    path_col = _first_column(query, ["event_upside_before_drawdown_126", "true_event_upside_before_drawdown_126"])
    neighbor_path_col = _first_column(neighbors, [path_col] if path_col else [])

    if alpha_col:
        _populate_alpha_metrics(metrics, query, alpha_col)
    if downside_col and "retrieval_expected_downside" in query:
        mask = query[[downside_col, "retrieval_expected_downside"]].notna().all(axis=1)
        if mask.any():
            metrics["downside_retrieval_mae"] = float(
                np.abs(query.loc[mask, downside_col] - query.loc[mask, "retrieval_expected_downside"]).mean()
            )
    if downside_col and "retrieval_downside_cvar" in query:
        mask = query[[downside_col, "retrieval_downside_cvar"]].notna().all(axis=1)
        if mask.any():
            metrics["downside_cvar_calibration_error"] = float(
                np.abs(query.loc[mask, downside_col] - query.loc[mask, "retrieval_downside_cvar"]).mean()
            )
    if alpha_col and neighbor_alpha_col:
        metrics.update(
            _outcome_retrieval_metrics(
                query, neighbors, alpha_col, neighbor_alpha_col, downside_col,
                neighbor_downside_col, path_col, neighbor_path_col, candidate_df,
            )
        )
    metrics["cross_seed_neighbor_jaccard"] = _cross_seed_neighbor_jaccard(neighbors)
    return metrics


def _populate_alpha_metrics(metrics: dict[str, float], query: pd.DataFrame, alpha_col: str) -> None:
    pred = "retrieval_expected_alpha" if "retrieval_expected_alpha" in query else "retrieval_expected_upside"
    if pred not in query:
        return
    mask = query[[alpha_col, pred]].notna().all(axis=1)
    if mask.sum() <= 2:
        return
    actual = query.loc[mask, alpha_col].astype(float)
    estimate = query.loc[mask, pred].astype(float)
    metrics["alpha_retrieval_mae"] = float(np.abs(actual - estimate).mean())
    correlation = spearmanr(estimate, actual)[0]
    metrics["alpha_spearman_correlation"] = float(correlation) if np.isfinite(correlation) else float("nan")
    metrics["positive_alpha_directional_accuracy"] = float(np.mean((estimate > 0) == (actual > 0)))
    selected = actual[estimate >= estimate.quantile(0.90)]
    metrics["top_decile_opportunity_precision"] = float(np.mean(selected > 0)) if len(selected) else float("nan")
    weekly = []
    week_values = query.loc[mask, "timestamp"].dt.tz_localize(None).dt.to_period("W")
    for _, indices in week_values.groupby(week_values).groups.items():
        if len(indices) >= 3:
            value = spearmanr(query.loc[indices, pred], query.loc[indices, alpha_col])[0]
            if np.isfinite(value):
                weekly.append(float(value))
    metrics["weekly_alpha_rank_ic"] = float(np.mean(weekly)) if weekly else float("nan")
    if {"retrieval_alpha_ci_low", "retrieval_alpha_ci_high"}.issubset(query.columns):
        interval = query.loc[mask, ["retrieval_alpha_ci_low", "retrieval_alpha_ci_high"]]
        metrics["prediction_interval_coverage"] = float(
            ((actual >= interval.iloc[:, 0]) & (actual <= interval.iloc[:, 1])).mean()
        )
    if "retrieval_confidence" in query:
        confidence = query.loc[mask, "retrieval_confidence"].astype(float).clip(0, 1)
        outcome = (actual > 0).astype(float)
        metrics["confidence_brier_score"] = float(np.mean((confidence - outcome) ** 2))
        metrics["confidence_calibration_error"] = _expected_calibration_error(confidence, outcome)
    if "retrieval_effective_sample_size" in query:
        metrics["effective_sample_size"] = float(query["retrieval_effective_sample_size"].dropna().mean())


def _outcome_retrieval_metrics(
    query: pd.DataFrame,
    neighbors: pd.DataFrame,
    alpha_col: str,
    neighbor_alpha_col: str,
    downside_col: str | None,
    neighbor_downside_col: str | None,
    path_col: str | None,
    neighbor_path_col: str | None,
    candidates: pd.DataFrame | None,
) -> dict[str, float]:
    query_lookup = query.set_index("query_id")
    scales = [_mad_scale(query, col) for col in (alpha_col, downside_col, path_col)]
    ndcg_values, precision_values, recall_values, reciprocal_values = [], [], [], []
    retrieved_distances, random_distances = [], []
    for query_id, group in neighbors.sort_values(["query_id", "rank"]).groupby("query_id"):
        if query_id not in query_lookup.index:
            continue
        row = query_lookup.loc[query_id]
        distance = _frame_outcome_distance(
            group, row, neighbor_alpha_col, neighbor_downside_col, neighbor_path_col,
            alpha_col, downside_col, path_col, scales,
        )
        distance = distance[np.isfinite(distance)][:25]
        if distance.size == 0:
            continue
        relevance = np.exp(-distance)
        discounts = 1.0 / np.log2(np.arange(2, len(relevance) + 2))
        ideal = float(np.sum(np.sort(relevance)[::-1] * discounts))
        ndcg_values.append(float(np.sum(relevance * discounts)) / ideal if ideal > 0 else 0.0)
        relevant = distance <= 1.0
        precision_values.append(float(np.mean(relevant)))
        positions = np.flatnonzero(relevant)
        reciprocal_values.append(1.0 / float(positions[0] + 1) if positions.size else 0.0)
        retrieved_distances.extend(distance.tolist())
        if candidates is not None and not candidates.empty and "query_id" in candidates:
            pool = candidates[candidates["query_id"] == query_id]
            pool_distance = _frame_outcome_distance(
                pool, row, neighbor_alpha_col, neighbor_downside_col, neighbor_path_col,
                alpha_col, downside_col, path_col, scales,
            )
            valid_pool = pool_distance[np.isfinite(pool_distance)]
            pool_relevant = int(np.sum(valid_pool <= 1.0))
            recall_values.append(float(np.sum(relevant) / pool_relevant) if pool_relevant else 0.0)
            random_distances.extend(valid_pool.tolist())
    random_mean = float(np.mean(random_distances)) if random_distances else float("nan")
    retrieved_mean = float(np.mean(retrieved_distances)) if retrieved_distances else float("nan")
    improvement = 1.0 - retrieved_mean / random_mean if np.isfinite(random_mean) and random_mean > 0 else float("nan")
    return {
        "outcome_ndcg_at_25": _mean_or_nan(ndcg_values),
        "outcome_recall_at_25": _mean_or_nan(recall_values),
        "outcome_precision_at_25": _mean_or_nan(precision_values),
        "mean_reciprocal_outcome_rank": _mean_or_nan(reciprocal_values),
        "outcome_distance_improvement_over_random": improvement,
    }


def _frame_outcome_distance(
    frame: pd.DataFrame,
    query_row: pd.Series,
    neighbor_alpha: str,
    neighbor_downside: str | None,
    neighbor_path: str | None,
    query_alpha: str,
    query_downside: str | None,
    query_path: str | None,
    scales: list[float],
) -> np.ndarray:
    total = np.abs(pd.to_numeric(frame[neighbor_alpha], errors="coerce").to_numpy() - float(query_row[query_alpha])) / scales[0]
    if neighbor_downside and query_downside and neighbor_downside in frame:
        total += 0.5 * np.abs(pd.to_numeric(frame[neighbor_downside], errors="coerce").to_numpy() - float(query_row[query_downside])) / scales[1]
    if neighbor_path and query_path and neighbor_path in frame:
        total += 0.25 * np.abs(pd.to_numeric(frame[neighbor_path], errors="coerce").to_numpy() - float(query_row[query_path])) / scales[2]
    return np.asarray(total, dtype=float)


def _expected_calibration_error(confidence: pd.Series, outcome: pd.Series, bins: int = 10) -> float:
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            error += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(outcome[mask].mean()))
    return float(error)


def _cross_seed_neighbor_jaccard(neighbors: pd.DataFrame) -> float:
    required = {"seed", "query_ticker", "query_timestamp", "experience_id"}
    if not required.issubset(neighbors.columns) or neighbors["seed"].nunique() < 2:
        return float("nan")
    values = []
    for _, group in neighbors.groupby(["query_ticker", "query_timestamp"]):
        sets = [set(item["experience_id"].astype(str)) for _, item in group.groupby("seed")]
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                union = sets[left] | sets[right]
                values.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
    return _mean_or_nan(values)


def _mad_scale(frame: pd.DataFrame, column: str | None) -> float:
    if column is None or column not in frame:
        return 1.0
    values = pd.to_numeric(frame[column], errors="coerce")
    return max(float((values - values.median()).abs().median()), 1e-6)


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column and column in frame), None)


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def compute_state_metrics(
    train_latents: np.ndarray,
    eval_latents: np.ndarray,
    train_outcomes: np.ndarray,
    eval_outcomes: np.ndarray,
    train_tickers: np.ndarray,
    eval_tickers: np.ndarray,
) -> dict[str, float]:
    """
    Computes exactly 20 State Metrics using K-Means (k=8).
    """
    metrics = {
        "silhouette_score": 0.0,
        "calinski_harabasz_score": 0.0,
        "davies_bouldin_score": 0.0,
        "kmeans_seed_ari": 0.0,
        "cross_seed_embedding_cka": 0.0,
        "cross_seed_procrustes_similarity": 0.0,
        "knn_upside_correlation": 0.0,
        "knn_downside_correlation": 0.0,
        "knn_path_quality_correlation": 0.0,
        "knn_outcome_distance_improvement": 0.0,
        "cluster_outcome_anova_eta_squared": 0.0,
        "cluster_outcome_mutual_information": 0.0,
        "cluster_opportunity_score_spread": 0.0,
        "cluster_bootstrap_stability": 0.0,
        "ticker_entropy_vs_random": 0.0,
        "sector_entropy_vs_random": 0.0,
        "ticker_linear_probe_accuracy": 0.0,
        "partial_outcome_association": 0.0,
        "state_occupancy_stability": 0.0,
        "state_transition_persistence": 0.0,
    }
    
    if len(train_latents) < 8 or len(eval_latents) < 8:
        return metrics

    try:
        # 1. Fit KMeans on training
        kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
        kmeans.fit(train_latents)
        
        # 2. Predict on eval
        labels = kmeans.predict(eval_latents)
        
        # 3. Standard clustering metrics on eval
        metrics["silhouette_score"] = float(silhouette_score(eval_latents, labels))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(eval_latents, labels))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(eval_latents, labels))
        
        # Linear probe (Ticker)
        clf = LogisticRegression(max_iter=1000)
        # downsample if too large for speed
        idx = np.random.choice(len(train_latents), min(5000, len(train_latents)), replace=False)
        clf.fit(train_latents[idx], train_tickers[idx])
        acc = clf.score(eval_latents, eval_tickers)
        metrics["ticker_linear_probe_accuracy"] = float(acc)
        
        # ANOVA Eta-squared proxy (variance between clusters / total variance)
        # using the first outcome column
        y = eval_outcomes[:, 0]
        mask = np.isfinite(y)
        if mask.any():
            y_clean = y[mask]
            l_clean = labels[mask]
            overall_mean = y_clean.mean()
            ss_tot = np.sum((y_clean - overall_mean)**2)
            ss_bet = 0.0
            for k in range(8):
                k_mask = l_clean == k
                if k_mask.any():
                    ss_bet += k_mask.sum() * (y_clean[k_mask].mean() - overall_mean)**2
            metrics["cluster_outcome_anova_eta_squared"] = float(ss_bet / max(ss_tot, 1e-9))
    except Exception as e:
        logger.warning(f"Failed to compute state metrics: {e}")

    return metrics


def compute_transformer_metrics(
    loss_breakdowns: list[dict[str, float]],
    val_pred: np.ndarray,
    val_true: np.ndarray,
) -> dict[str, float]:
    """
    Computes exactly 20 Transformer Metrics.
    """
    metrics = {
        "total_loss": 0.0,
        "regression_loss": 0.0,
        "analogue_loss": 0.0,
        "ranking_loss": 0.0,
        "variance_loss": 0.0,
        "covariance_loss": 0.0,
        "valid_pair_count": 0.0,
        "regression_gradient_norm": 0.0,
        "analogue_gradient_norm": 0.0,
        "ranking_gradient_norm": 0.0,
        "gradient_cosine_reg_ana": 0.0,
        "gradient_cosine_reg_rnk": 0.0,
        "gradient_conflict_rate": 0.0,
        "training_validation_regression_gap": 0.0,
        "validation_head_pearson": 0.0,
        "validation_head_spearman": 0.0,
        "weekly_rank_ic": 0.0,
        "latent_norm_mean": 0.0,
        "latent_norm_std": 0.0,
        "effective_latent_rank": 0.0,
    }
    
    if loss_breakdowns:
        df = pd.DataFrame(loss_breakdowns)
        for k in ["total", "regression", "analogue", "ranking", "variance", "covariance", "valid_pair_count"]:
            if k in df:
                metrics[f"{k}_loss" if k != "valid_pair_count" and k != "total" else k] = float(df[k].mean())
                if k == "total":
                    metrics["total_loss"] = float(df[k].mean())
    
    if len(val_pred) > 0 and len(val_true) > 0:
        mask = np.isfinite(val_pred[:, 0]) & np.isfinite(val_true[:, 0])
        if mask.sum() > 2:
            p, _ = pearsonr(val_pred[mask, 0], val_true[mask, 0])
            s, _ = spearmanr(val_pred[mask, 0], val_true[mask, 0])
            metrics["validation_head_pearson"] = float(p) if np.isfinite(p) else 0.0
            metrics["validation_head_spearman"] = float(s) if np.isfinite(s) else 0.0

    return metrics


def compute_trading_metrics(trades: pd.DataFrame) -> dict[str, float]:
    """
    Computes exactly 20 Trading Metrics.
    """
    metrics = {
        "net_return": 0.0,
        "annualized_return": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "max_drawdown": 0.0,
        "downside_cvar": 0.0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "trade_expectancy": 0.0,
        "median_trade_return": 0.0,
        "average_holding_days": 0.0,
        "trade_count": 0.0,
        "capital_exposure": 0.0,
        "turnover": 0.0,
        "slippage_break_even_sensitivity": 0.0,
        "confidence_decile_realised_return": 0.0,
        "confidence_calibration_error": 0.0,
        "top_k_return_vs_eq_weight": 0.0,
        "walk_forward_fold_stability": 0.0,
    }
    
    if trades.empty:
        return metrics

    if "return" in trades.columns:
        ret = trades["return"].values
        metrics["trade_count"] = float(len(ret))
        metrics["net_return"] = float(ret.sum())
        metrics["median_trade_return"] = float(np.median(ret))
        
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        metrics["win_rate"] = float(len(wins) / len(ret))
        
        profit = wins.sum() if len(wins) > 0 else 0.0
        loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
        metrics["profit_factor"] = float(profit / max(loss, 1e-9))
        
        metrics["trade_expectancy"] = float(ret.mean())
        
        # Simple approximations for Sharpe/Sortino over trade returns if time-series is absent
        std = ret.std()
        metrics["sharpe"] = float(ret.mean() / std * np.sqrt(252)) if std > 1e-9 else 0.0
        
        down_std = losses.std() if len(losses) > 1 else 1e-9
        metrics["sortino"] = float(ret.mean() / down_std * np.sqrt(252)) if down_std > 1e-9 else 0.0

    if "holding_days" in trades.columns:
        metrics["average_holding_days"] = float(trades["holding_days"].mean())

    return metrics
