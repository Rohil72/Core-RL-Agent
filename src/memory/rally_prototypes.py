"""Causal rally-start, neutral, and failure prototype memories over frozen latent states."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from src.memory.experience import latent_columns
from src.memory.retrieval import (
    RetrievalConfig,
    RetrievalIndex,
    build_retrieval_index,
    normalise_latents,
    retrieve_neighbors,
)


@dataclass(frozen=True)
class RallyPrototypeConfig:
    """Define fixed path-dependent rally labels and prototype retrieval rules."""

    upside_threshold: float = 0.10
    drawdown_threshold: float = -0.10
    horizon_sessions: int = 63
    maturity_sessions: int = 126
    k: int = 25
    minimum_neighbors: int = 5
    minimum_neighbor_separation_sessions: int = 21
    same_ticker_mode: str = "exclude"
    exclude_query_sector: bool = False
    exclude_query_industry: bool = False
    retrieval_batch_size: int = 128
    include_neutral: bool = False
    neutral_sampling_stride_sessions: int = 21
    maximum_nearest_distance_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.horizon_sessions <= 0 or self.maturity_sessions < self.horizon_sessions:
            raise ValueError("maturity_sessions must be at least horizon_sessions and both must be positive.")
        if self.upside_threshold <= 0 or self.drawdown_threshold >= 0:
            raise ValueError("upside_threshold must be positive and drawdown_threshold must be negative.")
        if self.k <= 0 or self.minimum_neighbors <= 0 or self.minimum_neighbors > self.k:
            raise ValueError("Require 0 < minimum_neighbors <= k.")
        if self.retrieval_batch_size <= 0:
            raise ValueError("retrieval_batch_size must be positive.")
        if self.neutral_sampling_stride_sessions <= 0:
            raise ValueError("neutral_sampling_stride_sessions must be positive.")
        if self.maximum_nearest_distance_ratio is not None and self.maximum_nearest_distance_ratio <= 0:
            raise ValueError("maximum_nearest_distance_ratio must be positive when configured.")


@dataclass(frozen=True)
class RallyPrototypeSet:
    """Deduplicated successful, neutral, and failed historical rally experiences."""

    success: pd.DataFrame
    failure: pd.DataFrame
    audit: dict[str, float | int | str | bool]
    neutral: pd.DataFrame | None = None


def build_rally_prototypes(
    frame: pd.DataFrame,
    config: RallyPrototypeConfig | None = None,
) -> RallyPrototypeSet:
    """Create causal successful, neutral, and failed prototype banks from mature states."""
    cfg = config or RallyPrototypeConfig()
    required = {
        "ticker",
        "timestamp",
        "future_max_return_63",
        "future_min_return_63",
        "event_peak_offset_63",
        "event_drawdown_offset_63",
        "event_upside_before_drawdown_126",
        "event_upside_hit_126",
        "event_drawdown_hit_126",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Rally prototype labels require columns: {', '.join(missing)}")
    if not latent_columns(frame):
        raise ValueError("Rally prototype labels require latent_* columns.")

    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
    source["ticker"] = source["ticker"].astype(str)
    if "outcome_available_timestamp" in source:
        source["outcome_available_timestamp"] = pd.to_datetime(
            source["outcome_available_timestamp"], utc=True, errors="coerce"
        )
        source = source.loc[source["outcome_available_timestamp"].notna()].copy()
    if source.empty:
        return RallyPrototypeSet(source.copy(), source.copy(), _empty_audit(cfg), source.copy() if cfg.include_neutral else None)

    source["_session"] = _session_index(source)
    success_mask = (
        pd.to_numeric(source["future_max_return_63"], errors="coerce") >= cfg.upside_threshold
    ) & (
        pd.to_numeric(source["event_upside_before_drawdown_126"], errors="coerce") == 1.0
    )
    failure_mask = (
        pd.to_numeric(source["event_drawdown_hit_126"], errors="coerce") == 1.0
    ) & (
        pd.to_numeric(source["event_upside_before_drawdown_126"], errors="coerce") == 0.0
    ) & (
        pd.to_numeric(source["future_min_return_63"], errors="coerce") <= cfg.drawdown_threshold
    )

    success_candidates = source.loc[success_mask].copy()
    failure_candidates = source.loc[failure_mask].copy()
    neutral_candidates = source.loc[~success_mask & ~failure_mask].copy()
    success = _deduplicate_episodes(success_candidates, cfg, role="success")
    failure = _deduplicate_episodes(failure_candidates, cfg, role="failure")
    neutral = _sample_neutral_prototypes(neutral_candidates, cfg) if cfg.include_neutral else None
    total_events = len(success_candidates) + len(failure_candidates)
    if cfg.include_neutral:
        total_events += len(neutral_candidates)
    audit: dict[str, float | int | str | bool] = {
        "upside_threshold": cfg.upside_threshold,
        "drawdown_threshold": cfg.drawdown_threshold,
        "horizon_sessions": cfg.horizon_sessions,
        "maturity_sessions": cfg.maturity_sessions,
        "source_rows": int(len(source)),
        "success_candidate_rows": int(len(success_candidates)),
        "failure_candidate_rows": int(len(failure_candidates)),
        "neutral_candidate_rows": int(len(neutral_candidates)),
        "success_prototype_rows": int(len(success)),
        "failure_prototype_rows": int(len(failure)),
        "neutral_prototype_rows": int(len(neutral)) if neutral is not None else 0,
        "success_prior": float(len(success_candidates) / total_events) if total_events else 0.5,
        "failure_prior": float(len(failure_candidates) / total_events) if total_events else 0.5,
        "neutral_prior": float(len(neutral_candidates) / total_events) if cfg.include_neutral and total_events else 0.0,
        "include_neutral": cfg.include_neutral,
    }
    return RallyPrototypeSet(success.reset_index(drop=True), failure.reset_index(drop=True), audit, neutral)


def score_rally_prototype_membership(
    reference_memory: pd.DataFrame,
    prototypes: RallyPrototypeSet,
    query: pd.DataFrame,
    config: RallyPrototypeConfig | None = None,
    *,
    bandwidth: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate whether a query resembles causal successful rather than failed rally starts."""
    cfg = config or RallyPrototypeConfig()
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be a positive finite distance from training-memory calibration.")
    columns = latent_columns(reference_memory)
    if not columns or any(column not in query for column in columns):
        raise ValueError("Reference memory and query must have matching latent_* columns.")
    success = prototypes.success.copy()
    failure = prototypes.failure.copy()
    neutral = prototypes.neutral.copy() if prototypes.neutral is not None else None
    if success.empty or failure.empty or (cfg.include_neutral and (neutral is None or neutral.empty)):
        return _empty_membership(query), pd.DataFrame()

    reference_x = reference_memory[columns].to_numpy(dtype=np.float32)
    banks: list[tuple[str, pd.DataFrame]] = [("success", success), ("failure", failure)]
    if cfg.include_neutral and neutral is not None:
        banks.append(("neutral", neutral))
    bank_matrices = [frame[columns].to_numpy(dtype=np.float32) for _, frame in banks]
    stacked = np.vstack([*bank_matrices, query[columns].to_numpy(dtype=np.float32)])
    _, normalised = normalise_latents(reference_x, stacked)
    normalised_banks: list[tuple[str, pd.DataFrame, np.ndarray]] = []
    cursor = 0
    for (role, frame), matrix in zip(banks, bank_matrices):
        stop = cursor + len(matrix)
        normalised_banks.append((role, frame, normalised[cursor:stop]))
        cursor = stop
    query_x = normalised[cursor:]
    retrieval_cfg = RetrievalConfig(
        k=cfg.k,
        minimum_neighbor_separation_sessions=cfg.minimum_neighbor_separation_sessions,
        same_ticker_mode=cfg.same_ticker_mode,
        exclude_query_sector=cfg.exclude_query_sector,
        exclude_query_industry=cfg.exclude_query_industry,
        require_outcome_availability=True,
    )
    evidence: dict[str, pd.DataFrame] = {}
    neighbor_frames: list[pd.DataFrame] = []
    for role, frame, vectors in normalised_banks:
        rows, neighbors = _bank_evidence(
            frame,
            vectors,
            build_retrieval_index(frame, vectors, retrieval_cfg),
            query,
            query_x,
            retrieval_cfg,
            cfg,
            bandwidth,
            role,
        )
        evidence[role] = rows
        neighbor_frames.append(neighbors)
    out = query[["ticker", "timestamp"]].copy().reset_index(drop=True)
    out["query_id"] = np.arange(len(out), dtype=np.int64)
    for role in ("success", "failure", "neutral"):
        if role in evidence:
            out = out.merge(evidence[role], on="query_id", how="left")
        else:
            out[f"{role}_neighbor_count"] = 0
            out[f"{role}_nearest_distance"] = np.nan
            out[f"{role}_median_distance"] = np.nan
            out[f"{role}_kernel_density"] = 0.0
    for column in ("success_kernel_density", "failure_kernel_density", "neutral_kernel_density"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    success_prior = float(prototypes.audit.get("success_prior", 0.5))
    failure_prior = float(prototypes.audit.get("failure_prior", 0.5))
    neutral_prior = float(prototypes.audit.get("neutral_prior", 0.0))
    success_mass = success_prior * out["success_kernel_density"].to_numpy(dtype=float)
    failure_mass = failure_prior * out["failure_kernel_density"].to_numpy(dtype=float)
    neutral_mass = neutral_prior * out["neutral_kernel_density"].to_numpy(dtype=float)
    opposing_mass = failure_mass + neutral_mass
    total_mass = success_mass + opposing_mass
    out["rally_start_probability"] = np.divide(
        success_mass,
        total_mass,
        out=np.full(len(out), np.nan, dtype=float),
        where=total_mass > 0,
    )
    out["rally_start_log_odds"] = np.log((success_mass + 1e-12) / (opposing_mass + 1e-12))
    out["rally_evidence_mass"] = total_mass
    distance_columns = ["success_nearest_distance", "failure_nearest_distance"]
    if cfg.include_neutral:
        distance_columns.append("neutral_nearest_distance")
    out["rally_nearest_prototype_distance"] = out[distance_columns].min(axis=1, skipna=True)
    threshold = (
        float(cfg.maximum_nearest_distance_ratio) * bandwidth
        if cfg.maximum_nearest_distance_ratio is not None
        else np.inf
    )
    out["rally_support_distance_threshold"] = threshold
    out["rally_support_pass"] = out["rally_nearest_prototype_distance"] <= threshold
    required_roles = ["success", "failure", *( ["neutral"] if cfg.include_neutral else [])]
    out["rally_start_eligible"] = out["rally_support_pass"]
    for role in required_roles:
        out["rally_start_eligible"] &= out[f"{role}_neighbor_count"].fillna(0).astype(int) >= cfg.minimum_neighbors
    neighbors = pd.concat(neighbor_frames, ignore_index=True)
    return out, neighbors


def prototype_config_dict(config: RallyPrototypeConfig) -> dict[str, float | int | str | bool | None]:
    """Return a serialisable configuration audit for experiment artifacts."""
    return asdict(config)


def _deduplicate_episodes(frame: pd.DataFrame, cfg: RallyPrototypeConfig, *, role: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    offset_column = "event_peak_offset_63" if role == "success" else "event_drawdown_offset_63"
    offset = pd.to_numeric(frame[offset_column], errors="coerce")
    event_session = frame["_session"].to_numpy(dtype=np.int64) + np.rint(
        offset.fillna(0.0).to_numpy(dtype=float) * cfg.horizon_sessions
    ).astype(np.int64)
    out = frame.assign(_event_session=event_session)
    out["prototype_role"] = role
    out["rally_episode_id"] = out["ticker"].astype(str) + "|" + out["_event_session"].astype(str)
    out = out.sort_values(["ticker", "_event_session", "timestamp"], kind="stable")
    out = out.drop_duplicates("rally_episode_id", keep="first").copy()
    out["prototype_timestamp"] = out["timestamp"]
    out["prototype_event_session"] = out["_event_session"].astype(np.int64)
    return out.drop(columns=["_session", "_event_session"], errors="ignore").reset_index(drop=True)


def _sample_neutral_prototypes(frame: pd.DataFrame, cfg: RallyPrototypeConfig) -> pd.DataFrame:
    """Thin ordinary states by ticker and session so neutral memory cannot dominate event banks."""
    if frame.empty:
        return frame.copy()
    out = frame.sort_values(["ticker", "timestamp"], kind="stable").copy()
    out["_neutral_bucket"] = (out["_session"].to_numpy(dtype=np.int64) // cfg.neutral_sampling_stride_sessions)
    out["prototype_role"] = "neutral"
    out["rally_episode_id"] = out["ticker"].astype(str) + "|neutral|" + out["_neutral_bucket"].astype(str)
    out = out.drop_duplicates("rally_episode_id", keep="first").copy()
    out["prototype_timestamp"] = out["timestamp"]
    out["prototype_event_session"] = out["_session"].astype(np.int64)
    return out.drop(columns=["_session", "_neutral_bucket"], errors="ignore").reset_index(drop=True)


def _session_index(frame: pd.DataFrame) -> pd.Series:
    if "session_index" in frame:
        values = pd.to_numeric(frame["session_index"], errors="coerce")
        if values.notna().all():
            return values.astype(np.int64)
    ordered = frame.sort_values(["ticker", "timestamp"], kind="stable")
    sequence = ordered.groupby("ticker", sort=False).cumcount().astype(np.int64)
    return sequence.reindex(frame.index).astype(np.int64)


def _bank_evidence(
    memory: pd.DataFrame,
    memory_x: np.ndarray,
    index: RetrievalIndex,
    query: pd.DataFrame,
    query_x: np.ndarray,
    retrieval_cfg: RetrievalConfig,
    prototype_cfg: RallyPrototypeConfig,
    bandwidth: float,
    role: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, float | int]] = []
    neighbors: list[dict[str, object]] = []
    query_frame = query.reset_index(drop=True).copy()
    for start in range(0, len(query_frame), prototype_cfg.retrieval_batch_size):
        stop = min(start + prototype_cfg.retrieval_batch_size, len(query_frame))
        batch = query_x[start:stop]
        squared = (
            np.sum(batch * batch, axis=1, keepdims=True)
            + np.sum(memory_x * memory_x, axis=1)[None, :]
            - 2.0 * (batch @ memory_x.T)
        )
        distances = np.sqrt(np.maximum(squared, 0.0)).astype(np.float32, copy=False)
        for local, (_, qrow) in enumerate(query_frame.iloc[start:stop].iterrows()):
            query_id = start + local
            indices, selected_distances = retrieve_neighbors(
                memory,
                memory_x,
                batch[local],
                qrow["timestamp"],
                str(qrow["ticker"]),
                retrieval_cfg,
                query_sector=qrow.get("sector"),
                query_industry=qrow.get("industry"),
                distance_vector=distances[local],
                retrieval_index=index,  # type: ignore[arg-type]
            )
            kernel = np.exp(-0.5 * (selected_distances / bandwidth) ** 2)
            count = int(len(indices))
            records.append(
                {
                    "query_id": query_id,
                    f"{role}_neighbor_count": count,
                    f"{role}_nearest_distance": float(selected_distances[0]) if count else np.nan,
                    f"{role}_median_distance": float(np.median(selected_distances)) if count else np.nan,
                    f"{role}_kernel_density": float(np.mean(kernel)) if count else 0.0,
                }
            )
            for rank, (idx, distance, weight) in enumerate(zip(indices, selected_distances, kernel), start=1):
                row = memory.iloc[int(idx)]
                neighbors.append(
                    {
                        "query_id": query_id,
                        "prototype_role": role,
                        "query_ticker": str(qrow["ticker"]),
                        "query_timestamp": qrow["timestamp"],
                        "rank": rank,
                        "neighbor_ticker": str(row["ticker"]),
                        "neighbor_timestamp": row["timestamp"],
                        "rally_episode_id": row.get("rally_episode_id"),
                        "distance": float(distance),
                        "kernel_weight": float(weight),
                        "future_max_return_63": _finite(row.get("future_max_return_63")),
                        "future_min_return_63": _finite(row.get("future_min_return_63")),
                        "event_upside_before_drawdown_126": _finite(row.get("event_upside_before_drawdown_126")),
                    }
                )
    return pd.DataFrame(records), pd.DataFrame(neighbors)


def _empty_membership(query: pd.DataFrame) -> pd.DataFrame:
    out = query[["ticker", "timestamp"]].copy().reset_index(drop=True)
    out["query_id"] = np.arange(len(out), dtype=np.int64)
    for role in ("success", "failure", "neutral"):
        out[f"{role}_neighbor_count"] = 0
        out[f"{role}_nearest_distance"] = np.nan
        out[f"{role}_median_distance"] = np.nan
        out[f"{role}_kernel_density"] = 0.0
    out["rally_start_probability"] = np.nan
    out["rally_start_log_odds"] = np.nan
    out["rally_evidence_mass"] = np.nan
    out["rally_nearest_prototype_distance"] = np.nan
    out["rally_support_distance_threshold"] = np.nan
    out["rally_support_pass"] = False
    out["rally_start_eligible"] = False
    return out


def _empty_audit(config: RallyPrototypeConfig) -> dict[str, float | int | str | bool]:
    return {
        "upside_threshold": config.upside_threshold,
        "drawdown_threshold": config.drawdown_threshold,
        "horizon_sessions": config.horizon_sessions,
        "maturity_sessions": config.maturity_sessions,
        "source_rows": 0,
        "success_candidate_rows": 0,
        "failure_candidate_rows": 0,
        "neutral_candidate_rows": 0,
        "success_prototype_rows": 0,
        "failure_prototype_rows": 0,
        "neutral_prototype_rows": 0,
        "success_prior": 0.5,
        "failure_prior": 0.5,
        "neutral_prior": 0.0,
        "include_neutral": config.include_neutral,
    }


def _finite(value: object) -> float | None:
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None
