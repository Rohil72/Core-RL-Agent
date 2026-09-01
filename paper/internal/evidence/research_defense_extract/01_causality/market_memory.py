from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.memory.aggregator import (
    AggregationConfig,
    combine_evidence_weights,
    distance_weights,
)
from src.memory.confidence import ConfidenceConfig
from src.memory.evidence import EvidenceSummary, build_evidence_summary
from src.memory.experience import ExperienceMemory, ExperienceSchema, latent_columns, resolve_column
from src.memory.retrieval import RetrievalConfig, build_retrieval_index, normalise_latents, retrieve_neighbors
from src.memory.metric import RetrievalMetric


@dataclass(frozen=True)
class MarketMemoryConfig:
    """Configuration for retrieval, evidence aggregation, and opportunity scoring."""

    k: int = 25
    causal_horizon_sessions: int = 126
    minimum_neighbor_separation_sessions: int = 21
    same_ticker_mode: str = "allow"
    exclude_query_sector: bool = False
    exclude_query_industry: bool = False
    same_ticker_neighbor_limit: float | None = 0.50
    max_neighbors_per_ticker: int | None = None
    target_upside: str = "future_max_return_63"
    target_alpha: str | None = None
    target_absolute_return: str | None = "decision_return_63"
    target_downside: str = "future_min_return_63"
    target_path_quality: str = "event_upside_before_drawdown_126"
    target_holding_period: str | None = "event_peak_offset_63"
    expected_upside_weight: float = 1.0
    path_quality_weight: float = 0.10
    downside_weight: float = 0.80
    uncertainty_weight: float = 0.20
    confidence_weight: float = 0.10
    disagreement_weight: float = 0.10
    aggregation_method: str = "gaussian"
    gaussian_bandwidth: float | None = None
    trim_fraction: float = 0.10
    tail_fraction: float = 0.25
    max_distance: float | None = None
    min_memory_confidence: float | None = None
    min_neighbors: int = 1
    max_median_distance: float | None = None
    confidence_reference_distance: float | None = None
    score_mode: str = "legacy"
    neighbor_scales: tuple[int, ...] | None = None
    multiscale_disagreement_weight: float = 0.25
    require_outcome_availability: bool = False
    retrieval_batch_size: int = 128
    maximum_memory_age_days: int | None = None
    temporal_half_life_days: float | None = None
    balance_group_column: str | None = None


def load_latent_frame(path: str | Path) -> pd.DataFrame:
    """Load a latent parquet table and normalize core identity columns."""
    df = pd.read_parquet(path)
    df = expand_latent_array_col(df)
    if "timestamp" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Latent table must contain ticker and timestamp columns.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ticker"] = df["ticker"].astype(str)
    return df.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def expand_latent_array_col(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a list-valued latent column into latent_0...latent_n columns."""
    if "latent" not in df.columns:
        return df
    out = df.copy()
    sample = out["latent"].dropna().iloc[0] if out["latent"].notna().any() else None
    if sample is None:
        return out.drop(columns=["latent"])
    arr = np.vstack(out["latent"].apply(lambda x: np.asarray(x, dtype=float)).to_numpy())
    latent_df = pd.DataFrame(arr, columns=[f"latent_{i}" for i in range(arr.shape[1])], index=out.index)
    return pd.concat([out.drop(columns=["latent"]), latent_df], axis=1).copy()


def score_market_memory(
    memory_df: pd.DataFrame,
    query_df: pd.DataFrame,
    config: MarketMemoryConfig | None = None,
    keep_query_columns: Iterable[str] | None = None,
    retrieval_metric: RetrievalMetric | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score query states using historical evidence instead of simple neighbor averaging."""
    cfg = config or MarketMemoryConfig()
    if cfg.k <= 0:
        raise ValueError("k must be positive.")

    memory = memory_df.copy()
    query = query_df.copy()
    memory["timestamp"] = pd.to_datetime(memory["timestamp"], utc=True)
    query["timestamp"] = pd.to_datetime(query["timestamp"], utc=True)
    memory["ticker"] = memory["ticker"].astype(str)
    query["ticker"] = query["ticker"].astype(str)

    schema = ExperienceSchema(
        target_upside=cfg.target_upside,
        target_alpha=cfg.target_alpha,
        target_absolute_return=cfg.target_absolute_return,
        target_downside=cfg.target_downside,
        target_path_quality=cfg.target_path_quality,
        target_holding_period=cfg.target_holding_period,
    )
    experience_memory = ExperienceMemory.from_frame(memory, schema)
    missing = [c for c in experience_memory.latent_cols if c not in query.columns]
    if missing:
        raise ValueError(f"Query table missing latent columns: {missing[:5]}")

    memory = experience_memory.frame
    query = query.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    query_matrix = query[experience_memory.latent_cols].to_numpy(dtype=float)
    if retrieval_metric is None:
        mem_x, qry_x = normalise_latents(experience_memory.latent_matrix, query_matrix)
    else:
        mem_x = retrieval_metric.transform(experience_memory.latent_matrix)
        qry_x = retrieval_metric.transform(query_matrix)
    scales = tuple(sorted(set(int(value) for value in (cfg.neighbor_scales or ()))))
    if any(value <= 0 for value in scales):
        raise ValueError("neighbor_scales must contain positive integers.")
    retrieval_k = max((cfg.k, *scales))
    retrieval_cfg = RetrievalConfig(
        k=retrieval_k,
        causal_horizon_sessions=cfg.causal_horizon_sessions,
        minimum_neighbor_separation_sessions=cfg.minimum_neighbor_separation_sessions,
        same_ticker_mode=cfg.same_ticker_mode,
        exclude_query_sector=cfg.exclude_query_sector,
        exclude_query_industry=cfg.exclude_query_industry,
        same_ticker_neighbor_limit=cfg.same_ticker_neighbor_limit,
        max_neighbors_per_ticker=cfg.max_neighbors_per_ticker,
        max_distance=cfg.max_distance,
        min_confidence=cfg.min_memory_confidence,
        require_outcome_availability=cfg.require_outcome_availability,
        maximum_memory_age_days=cfg.maximum_memory_age_days,
    )
    aggregation_cfg = AggregationConfig(
        method=cfg.aggregation_method,
        gaussian_bandwidth=cfg.gaussian_bandwidth,
        trim_fraction=cfg.trim_fraction,
        tail_fraction=cfg.tail_fraction,
    )
    confidence_cfg = ConfidenceConfig(reference_distance=cfg.confidence_reference_distance)
    score_weights = {
        "expected_upside_weight": cfg.expected_upside_weight,
        "path_quality_weight": cfg.path_quality_weight,
        "downside_weight": cfg.downside_weight,
        "uncertainty_weight": cfg.uncertainty_weight,
        "confidence_weight": cfg.confidence_weight,
        "disagreement_weight": cfg.disagreement_weight,
        "score_mode": cfg.score_mode,
    }

    signal_rows: list[dict] = []
    neighbor_rows: list[dict] = []
    base_keep = ["ticker", "timestamp"]
    if keep_query_columns:
        base_keep.extend(c for c in keep_query_columns if c in query.columns and c not in base_keep)

    if cfg.retrieval_batch_size <= 0:
        raise ValueError("retrieval_batch_size must be positive.")
    mem_x = np.asarray(mem_x, dtype=np.float32)
    qry_x = np.asarray(qry_x, dtype=np.float32)
    retrieval_index = build_retrieval_index(memory, mem_x, retrieval_cfg)
    distance_block = np.empty((0, len(memory)), dtype=np.float32)
    block_start = 0

    for qpos, qrow in query.iterrows():
        if qpos == block_start + len(distance_block):
            block_start = qpos
            batch = qry_x[qpos : qpos + cfg.retrieval_batch_size]
            squared = (
                np.sum(batch * batch, axis=1, keepdims=True)
                + np.sum(mem_x * mem_x, axis=1)[None, :]
                - 2.0 * (batch @ mem_x.T)
            )
            distance_block = np.sqrt(np.maximum(squared, 0.0)).astype(np.float32, copy=False)
        row = {c: qrow[c] for c in base_keep if c in query.columns}
        row["query_id"] = int(qpos)
        neighbor_idx, neighbor_dist = retrieve_neighbors(
            memory=memory,
            memory_x=mem_x,
            query_vector=qry_x[qpos],
            query_timestamp=qrow["timestamp"],
            query_ticker=str(qrow["ticker"]),
            config=retrieval_cfg,
            query_sector=qrow.get("sector"),
            query_industry=qrow.get("industry"),
            distance_vector=distance_block[qpos - block_start],
            retrieval_index=retrieval_index,
        )
        rejection_reason = None
        if neighbor_idx.size < cfg.min_neighbors:
            rejection_reason = "insufficient_neighbors"
        elif cfg.max_median_distance is not None and float(np.nanmedian(neighbor_dist)) > cfg.max_median_distance:
            rejection_reason = "ood"
        if rejection_reason is not None:
            row.update(_empty_signal_payload(rejection_reason))
            signal_rows.append(row)
            continue

        neighbors = memory.iloc[neighbor_idx]
        evidence, evidence_payload = _build_evidence_payload(
            neighbors=neighbors,
            distances=neighbor_dist,
            query_ticker=str(qrow["ticker"]),
            scales=scales,
            disagreement_weight=cfg.multiscale_disagreement_weight,
            upside_col=experience_memory.upside_col,
            alpha_col=experience_memory.alpha_col,
            downside_col=experience_memory.downside_col,
            path_quality_col=experience_memory.path_quality_col,
            holding_period_col=experience_memory.holding_period_col,
            absolute_return_col=experience_memory.absolute_return_col,
            aggregation=aggregation_cfg,
            confidence=confidence_cfg,
            score_weights=score_weights,
            query_timestamp=qrow["timestamp"],
            temporal_half_life_days=cfg.temporal_half_life_days,
            balance_group_column=cfg.balance_group_column,
        )
        row.update(evidence_payload)
        signal_rows.append(row)

        neighbor_priors = _evidence_prior_weights(
            neighbors,
            qrow["timestamp"],
            cfg.temporal_half_life_days,
            cfg.balance_group_column,
        )
        neighbor_evidence_weights = combine_evidence_weights(
            distance_weights(neighbor_dist, aggregation_cfg),
            neighbor_priors,
        )

        for rank, (midx, dist, prior, evidence_weight) in enumerate(
            zip(
                neighbor_idx,
                neighbor_dist,
                neighbor_priors,
                neighbor_evidence_weights,
            ),
            start=1,
        ):
            mrow = memory.iloc[int(midx)]
            relative_outcomes = {
                column: _safe_float(float(mrow[column]))
                for column in ("future_universe_alpha_63", "future_blended_alpha_63")
                if column in memory and pd.notna(mrow[column])
            }
            neighbor_rows.append(
                {
                    "query_id": int(qpos),
                    "query_ticker": str(qrow["ticker"]),
                    "query_timestamp": qrow["timestamp"],
                    "rank": rank,
                    "neighbor_index": int(midx),
                    "neighbor_ticker": str(mrow["ticker"]),
                    "neighbor_timestamp": mrow["timestamp"],
                    "neighbor_age_days": float(
                        (pd.Timestamp(qrow["timestamp"]) - pd.Timestamp(mrow["timestamp"])).total_seconds()
                        / 86_400.0
                    ),
                    "neighbor_market": _optional_str(mrow, "market"),
                    "neighbor_outcome_available_timestamp": mrow.get("outcome_available_timestamp"),
                    "experience_id": f"{mrow['ticker']}|{pd.Timestamp(mrow['timestamp']).isoformat()}",
                    "distance": float(dist),
                    "neighbor_prior_weight": float(prior),
                    "neighbor_evidence_weight": float(evidence_weight),
                    "neighbor_sector": _optional_str(mrow, "sector"),
                    "neighbor_industry": _optional_str(mrow, "industry"),
                    cfg.target_upside: _safe_float(float(mrow[experience_memory.upside_col])),
                    **(
                        {cfg.target_alpha: _safe_float(float(mrow[experience_memory.alpha_col]))}
                        if cfg.target_alpha and experience_memory.alpha_col is not None
                        else {}
                    ),
                    cfg.target_downside: _safe_float(float(mrow[experience_memory.downside_col])),
                    cfg.target_path_quality: (
                        _safe_float(float(mrow[experience_memory.path_quality_col]))
                        if experience_memory.path_quality_col is not None
                        else None
                    ),
                    **(
                        {cfg.target_holding_period: _safe_float(float(mrow[experience_memory.holding_period_col]))}
                        if cfg.target_holding_period and experience_memory.holding_period_col is not None
                        else {}
                    ),
                    **relative_outcomes,
                    "evidence_confidence": evidence.confidence.confidence,
                    "evidence_agreement_score": evidence.confidence.agreement_score,
                    "evidence_disagreement_score": evidence.confidence.disagreement_score,
                    "evidence_effective_sample_size": evidence.confidence.effective_sample_size,
                }
            )

    return pd.DataFrame(signal_rows), pd.DataFrame(neighbor_rows)


def _build_evidence_payload(
    *,
    neighbors: pd.DataFrame,
    distances: np.ndarray,
    query_ticker: str,
    scales: tuple[int, ...],
    disagreement_weight: float,
    upside_col: str,
    alpha_col: str | None,
    downside_col: str,
    path_quality_col: str | None,
    holding_period_col: str | None,
    absolute_return_col: str | None = None,
    aggregation: AggregationConfig,
    confidence: ConfidenceConfig,
    score_weights: dict,
    query_timestamp: pd.Timestamp,
    temporal_half_life_days: float | None,
    balance_group_column: str | None,
) -> tuple[EvidenceSummary, dict]:
    """Build one evidence payload, optionally requiring stability across scales."""

    usable_scales = [scale for scale in scales if scale <= len(neighbors)]
    if not usable_scales:
        usable_scales = [len(neighbors)]
    summaries = [
        build_evidence_summary(
            neighbors.iloc[:scale],
            distances[:scale],
            query_ticker,
            upside_col=upside_col,
            alpha_col=alpha_col,
            downside_col=downside_col,
            path_quality_col=path_quality_col,
            holding_period_col=holding_period_col,
            absolute_return_col=absolute_return_col,
            aggregation=aggregation,
            confidence=confidence,
            score_weights=score_weights,
            prior_weights=_evidence_prior_weights(
                neighbors.iloc[:scale],
                query_timestamp,
                temporal_half_life_days,
                balance_group_column,
            ),
        )
        for scale in usable_scales
    ]
    if len(summaries) == 1:
        payload = summaries[0].to_signal_payload()
        payload.update(
            {
                "retrieval_scale_count": 1,
                "retrieval_scale_score_std": 0.0,
                "retrieval_scale_sign_agreement": 1.0,
            }
        )
        return summaries[0], payload

    payloads = [summary.to_signal_payload() for summary in summaries]
    combined: dict[str, object] = {}
    for key in payloads[0]:
        values = [
            float(payload[key])
            for payload in payloads
            if isinstance(payload.get(key), (int, float))
            and not isinstance(payload.get(key), bool)
            and np.isfinite(float(payload[key]))
        ]
        combined[key] = float(np.median(values)) if values else payloads[-1].get(key)

    scores = np.asarray(
        [summary.score for summary in summaries if summary.score is not None],
        dtype=float,
    )
    signs = scores >= 0.0
    sign_agreement = (
        float(max(signs.mean(), 1.0 - signs.mean())) if len(signs) else 0.0
    )
    score_std = float(np.std(scores, ddof=0)) if len(scores) else float("nan")
    if len(scores):
        combined["opportunity_score"] = float(
            np.median(scores) - disagreement_weight * score_std
        )
    alpha_lows = [
        value for value in (summary.alpha.ci_low for summary in summaries) if value is not None
    ]
    alpha_highs = [
        value for value in (summary.alpha.ci_high for summary in summaries) if value is not None
    ]
    combined["retrieval_alpha_ci_low"] = min(alpha_lows) if alpha_lows else None
    combined["retrieval_alpha_lcb"] = combined["retrieval_alpha_ci_low"]
    combined["retrieval_alpha_ci_high"] = max(alpha_highs) if alpha_highs else None
    abs_lows = [
        value for value in (summary.absolute_return.ci_low for summary in summaries if summary.absolute_return) if value is not None
    ]
    abs_highs = [
        value for value in (summary.absolute_return.ci_high for summary in summaries if summary.absolute_return) if value is not None
    ]
    combined["retrieval_absolute_return_ci_low"] = min(abs_lows) if abs_lows else None
    combined["retrieval_absolute_return_lcb"] = combined["retrieval_absolute_return_ci_low"]
    combined["retrieval_absolute_return_ci_high"] = max(abs_highs) if abs_highs else None
    combined["retrieval_confidence"] = float(
        np.median([summary.confidence.confidence for summary in summaries])
        * sign_agreement
    )
    combined["retrieval_agreement_score"] = float(
        np.median([summary.confidence.agreement_score for summary in summaries])
        * sign_agreement
    )
    combined["retrieval_disagreement_score"] = (
        1.0 - float(combined["retrieval_agreement_score"])
    )
    combined["retrieval_neighbor_count"] = int(max(usable_scales))
    combined["retrieval_ood_pass"] = all(summary.ood_pass for summary in summaries)
    combined["retrieval_scale_count"] = len(summaries)
    combined["retrieval_scale_score_std"] = score_std
    combined["retrieval_scale_sign_agreement"] = sign_agreement
    return summaries[-1], combined


def _evidence_prior_weights(
    neighbors: pd.DataFrame,
    query_timestamp: pd.Timestamp,
    temporal_half_life_days: float | None,
    balance_group_column: str | None,
) -> np.ndarray:
    """Build causal recency and group-balance priors for retrieved evidence."""

    weights = np.ones(len(neighbors), dtype=float)
    if temporal_half_life_days is not None:
        if temporal_half_life_days <= 0.0:
            raise ValueError("temporal_half_life_days must be positive.")
        timestamps = pd.to_datetime(neighbors["timestamp"], utc=True, errors="coerce")
        ages = (
            pd.Timestamp(query_timestamp) - timestamps
        ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
        ages = np.maximum(np.nan_to_num(ages, nan=np.inf), 0.0)
        weights *= np.exp2(-ages / float(temporal_half_life_days))
    if balance_group_column is not None:
        if balance_group_column not in neighbors:
            raise ValueError(
                f"Memory lacks configured balance group column: {balance_group_column}"
            )
        groups = neighbors[balance_group_column].fillna("unknown").astype(str)
        counts = groups.value_counts()
        weights *= groups.map(lambda value: 1.0 / float(counts[value])).to_numpy(dtype=float)
    return weights


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _optional_str(row: pd.Series, col: str) -> str | None:
    return str(row[col]) if col in row and pd.notna(row[col]) else None


def _empty_signal_payload(reason: str | None = None) -> dict:
    return {
        "retrieval_expected_upside": None,
        "retrieval_expected_alpha": None,
        "retrieval_alpha_p10": None,
        "retrieval_alpha_p90": None,
        "retrieval_alpha_ci_low": None,
        "retrieval_alpha_ci_high": None,
        "retrieval_alpha_lcb": None,
        "retrieval_alpha_std": None,
        "retrieval_expected_absolute_return": None,
        "retrieval_absolute_return_lcb": None,
        "retrieval_absolute_return_p10": None,
        "retrieval_absolute_return_p90": None,
        "retrieval_absolute_return_ci_low": None,
        "retrieval_absolute_return_ci_high": None,
        "retrieval_absolute_return_std": None,
        "retrieval_expected_downside": None,
        "retrieval_calibrated_downside": None,
        "retrieval_upside_p10": None,
        "retrieval_upside_p90": None,
        "retrieval_upside_ci_low": None,
        "retrieval_upside_ci_high": None,
        "retrieval_upside_before_drawdown_prob": None,
        "retrieval_downside_cvar": None,
        "retrieval_outcome_std": None,
        "retrieval_expected_holding_period": None,
        "retrieval_confidence": 0.0,
        "retrieval_agreement_score": 0.0,
        "retrieval_disagreement_score": 1.0,
        "retrieval_effective_sample_size": 0.0,
        "retrieval_entropy": 0.0,
        "retrieval_distance_weighted_confidence": 0.0,
        "retrieval_historical_diversity": 0.0,
        "retrieval_neighbor_count": 0,
        "retrieval_same_ticker_rate": None,
        "retrieval_cross_ticker_rate": None,
        "retrieval_median_distance": None,
        "retrieval_ood_pass": reason != "ood",
        "retrieval_rejection_reason": reason,
        "retrieval_scale_count": 0,
        "retrieval_scale_score_std": None,
        "retrieval_scale_sign_agreement": 0.0,
        "opportunity_quality": None,
        "opportunity_score": None,
    }
