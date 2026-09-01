from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression


@dataclass(frozen=True)
class OpportunityAllocatorConfig:
    """Configuration for scale-free, per-opportunity allocation inference."""

    downside_weight: float = 0.35
    learning_rate: float = 0.05
    max_iter: int = 200
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 50
    l2_regularization: float = 0.10
    random_state: int = 7
    action_rank_thresholds: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90)
    absolute_utility_thresholds: tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
    action_levels: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    support_floor_quantile: float = 0.20
    support_reference_quantile: float = 0.50
    minimum_support_multiplier: float = 0.50

    def __post_init__(self) -> None:
        if self.downside_weight < 0:
            raise ValueError("downside_weight cannot be negative.")
        if tuple(sorted(self.action_rank_thresholds)) != self.action_rank_thresholds:
            raise ValueError("action_rank_thresholds must be sorted.")
        if tuple(sorted(self.absolute_utility_thresholds)) != self.absolute_utility_thresholds:
            raise ValueError("absolute_utility_thresholds must be sorted.")
        if len(self.action_levels) != len(self.action_rank_thresholds) + 1:
            raise ValueError("action_levels must contain one more value than thresholds.")
        if len(self.action_levels) != len(self.absolute_utility_thresholds) + 1:
            raise ValueError("action_levels must contain one more value than utility thresholds.")
        if any(not 0.0 <= value <= 1.0 for value in (*self.action_rank_thresholds, *self.action_levels)):
            raise ValueError("Allocation thresholds and levels must lie in [0, 1].")
        if not 0.0 <= self.support_floor_quantile < self.support_reference_quantile <= 1.0:
            raise ValueError("Support quantiles must satisfy 0 <= floor < reference <= 1.")
        if not 0.0 <= self.minimum_support_multiplier <= 1.0:
            raise ValueError("minimum_support_multiplier must lie in [0, 1].")


@dataclass
class CalibratedObviousAllocator:
    """Fitted monotonic utility calibration and causal support references."""

    calibrator: IsotonicRegression
    market_support_floor: float
    market_support_reference: float
    evidence_support_floor: float
    evidence_support_reference: float
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp


RANK_SOURCES: dict[str, tuple[str, bool]] = {
    "economic_score": ("consensus_economic_score", True),
    "entry_rank": ("consensus_entry_rank", True),
    "expected_alpha": ("retrieval_expected_alpha", True),
    "alpha_lcb": ("retrieval_alpha_ci_low", True),
    "alpha_p10": ("retrieval_alpha_p10", True),
    "expected_upside": ("retrieval_expected_upside", True),
    "downside_safety": ("retrieval_downside_cvar", False),
    "outcome_stability": ("retrieval_outcome_std", False),
    "distance_safety": ("retrieval_median_distance", False),
}


BOUNDED_SOURCES = (
    "retrieval_confidence",
    "retrieval_agreement_score",
    "retrieval_distance_weighted_confidence",
    "retrieval_historical_diversity",
    "retrieval_entropy",
    "retrieval_cross_ticker_rate",
    "retrieval_upside_before_drawdown_prob",
    "seed_vote_fraction",
    "seed_score_positive_fraction",
)


def build_opportunity_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build currency- and universe-scale-invariant features from retrieval evidence."""
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    output = pd.DataFrame(index=data.index)
    feature_names: list[str] = []

    for name, (column, higher_is_better) in RANK_SOURCES.items():
        values = _numeric(data, column)
        if not higher_is_better:
            values = -values.abs()
        feature = f"rank_{name}"
        output[feature] = values.groupby(data["timestamp"]).rank(method="average", pct=True).fillna(0.0)
        feature_names.append(feature)

    for column in BOUNDED_SOURCES:
        output[column] = _numeric(data, column).clip(0.0, 1.0).fillna(0.0)
        feature_names.append(column)

    ess = _numeric(data, "retrieval_effective_sample_size")
    output["effective_sample_fraction"] = (ess / 25.0).clip(0.0, 1.0).fillna(0.0)
    output["seed_rank_stability"] = (
        1.0 - _numeric(data, "seed_rank_std") / 0.20
    ).clip(0.0, 1.0).fillna(0.0)
    output["alpha_lcb_positive"] = (_numeric(data, "retrieval_alpha_ci_low") > 0.0).astype(float)
    output["ood_pass"] = data.get("retrieval_ood_pass", False)
    output["ood_pass"] = output["ood_pass"].fillna(False).astype(bool).astype(float)

    alpha = _numeric(data, "retrieval_expected_alpha")
    alpha_std = _numeric(data, "retrieval_alpha_std").abs()
    downside = _numeric(data, "retrieval_downside_cvar").abs()
    upside = _numeric(data, "retrieval_expected_upside")
    output["alpha_signal_to_noise"] = np.tanh(alpha / alpha_std.replace(0.0, np.nan)).fillna(0.0)
    output["alpha_reward_risk"] = np.tanh(alpha / downside.replace(0.0, np.nan)).fillna(0.0)
    output["upside_downside_balance"] = np.tanh(upside / downside.replace(0.0, np.nan)).fillna(0.0)
    feature_names.extend(
        [
            "effective_sample_fraction",
            "seed_rank_stability",
            "alpha_lcb_positive",
            "ood_pass",
            "alpha_signal_to_noise",
            "alpha_reward_risk",
            "upside_downside_balance",
        ]
    )

    grouped = data.groupby("timestamp", sort=False)
    market_features = {
        "market_alpha_breadth": grouped["retrieval_alpha_ci_low"].transform(
            lambda values: float((pd.to_numeric(values, errors="coerce") > 0.0).mean())
        ),
        "market_confidence": grouped["retrieval_confidence"].transform("median"),
        "market_agreement": grouped["retrieval_agreement_score"].transform("median"),
        "market_seed_breadth": grouped["seed_score_positive_fraction"].transform("mean"),
    }
    for name, values in market_features.items():
        output[name] = pd.to_numeric(values, errors="coerce").clip(0.0, 1.0).fillna(0.0)
        feature_names.append(name)
    return output.astype(np.float32), feature_names


def allocation_utility(frame: pd.DataFrame, downside_weight: float = 0.35) -> pd.Series:
    """Return the long-horizon, benchmark-relative reward used only as a training label."""
    alpha = _numeric(frame, "decision_net_alpha")
    adverse = _numeric(frame, "decision_mae").abs()
    return alpha - float(downside_weight) * adverse


def obvious_signal_score(features: pd.DataFrame) -> pd.Series:
    """Transparent no-fit score used to test whether an obvious rule is sufficient."""
    return (
        0.25 * features["rank_alpha_lcb"]
        + 0.20 * features["rank_expected_alpha"]
        + 0.20 * features["rank_downside_safety"]
        + 0.10 * features["retrieval_confidence"]
        + 0.10 * features["retrieval_agreement_score"]
        + 0.10 * features["seed_vote_fraction"]
        + 0.05 * features["seed_rank_stability"]
    )


def allocator_support(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize existing market-regime and retrieval reliability evidence."""
    market_columns = (
        "market_alpha_breadth",
        "market_confidence",
        "market_agreement",
        "market_seed_breadth",
    )
    evidence_columns = (
        "retrieval_confidence",
        "retrieval_agreement_score",
        "seed_vote_fraction",
        "seed_rank_stability",
        "effective_sample_fraction",
        "ood_pass",
    )
    return pd.DataFrame(
        {
            "market_support": features.loc[:, market_columns].mean(axis=1),
            "evidence_support": features.loc[:, evidence_columns].mean(axis=1),
        },
        index=features.index,
        dtype=float,
    ).clip(0.0, 1.0)


def fit_calibrated_obvious_allocator(
    frame: pd.DataFrame,
    config: OpportunityAllocatorConfig | None = None,
) -> CalibratedObviousAllocator:
    """Fit a monotonic B1-to-utility map using only mature, eligible history."""
    cfg = config or OpportunityAllocatorConfig()
    features, _ = build_opportunity_features(frame)
    score = obvious_signal_score(features)
    target = allocation_utility(frame, cfg.downside_weight)
    mature = pd.Series(frame.get("decision_is_mature", True), index=frame.index).fillna(False).astype(bool)
    eligible = pd.Series(frame.get("allocator_eligible", True), index=frame.index).fillna(False).astype(bool)
    mask = mature & eligible & score.notna() & target.notna()
    if int(mask.sum()) < 200:
        raise ValueError("Not enough mature causal examples to calibrate obvious utility.")

    calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
    calibrator.fit(score.loc[mask], target.loc[mask])
    support = allocator_support(features.loc[mask])
    timestamps = pd.to_datetime(frame.loc[mask, "timestamp"], utc=True)
    return CalibratedObviousAllocator(
        calibrator=calibrator,
        market_support_floor=float(support["market_support"].quantile(cfg.support_floor_quantile)),
        market_support_reference=float(
            support["market_support"].quantile(cfg.support_reference_quantile)
        ),
        evidence_support_floor=float(support["evidence_support"].quantile(cfg.support_floor_quantile)),
        evidence_support_reference=float(
            support["evidence_support"].quantile(cfg.support_reference_quantile)
        ),
        fit_start=timestamps.min(),
        fit_end=timestamps.max(),
    )


def predict_calibrated_obvious_allocator(
    model: CalibratedObviousAllocator,
    frame: pd.DataFrame,
    config: OpportunityAllocatorConfig | None = None,
) -> pd.DataFrame:
    """Produce expected utility and support-aware absolute entry allocations."""
    cfg = config or OpportunityAllocatorConfig()
    features, _ = build_opportunity_features(frame)
    obvious = obvious_signal_score(features)
    expected_utility = pd.Series(
        model.calibrator.predict(obvious.to_numpy(dtype=float)),
        index=frame.index,
        dtype=float,
    )
    thresholds = np.asarray(cfg.absolute_utility_thresholds, dtype=float)
    bins = np.searchsorted(thresholds, expected_utility.fillna(-np.inf), side="right")
    base_allocation = pd.Series(np.asarray(cfg.action_levels, dtype=float)[bins], index=frame.index)

    support = allocator_support(features)
    market_ratio = support["market_support"] / max(model.market_support_reference, 1e-12)
    evidence_ratio = support["evidence_support"] / max(model.evidence_support_reference, 1e-12)
    multiplier = pd.concat([market_ratio, evidence_ratio], axis=1).min(axis=1).clip(
        cfg.minimum_support_multiplier, 1.0
    )
    jointly_weak = (
        (support["market_support"] < model.market_support_floor)
        & (support["evidence_support"] < model.evidence_support_floor)
    )
    veto = jointly_weak | (features["ood_pass"] <= 0.0)
    allocation = (base_allocation * multiplier).mask(veto, 0.0).clip(0.0, 1.0)
    effective_score = (expected_utility * multiplier).mask(veto, 0.0)
    return pd.DataFrame(
        {
            "allocator_score": effective_score,
            "calibrated_utility": expected_utility,
            "entry_allocation_fraction": allocation,
            "base_allocation_fraction": base_allocation,
            "support_multiplier": multiplier.mask(veto, 0.0),
            "market_support": support["market_support"],
            "evidence_support": support["evidence_support"],
            "support_veto": veto,
        },
        index=frame.index,
    )


def fit_nonlinear_allocator(
    frame: pd.DataFrame,
    config: OpportunityAllocatorConfig | None = None,
) -> tuple[HistGradientBoostingRegressor, list[str]]:
    """Fit a compact nonlinear reward model on mature causal retrieval examples."""
    cfg = config or OpportunityAllocatorConfig()
    features, feature_names = build_opportunity_features(frame)
    target = allocation_utility(frame, cfg.downside_weight)
    mature = frame.get("decision_is_mature", True)
    mask = pd.Series(mature, index=frame.index).fillna(False).astype(bool) & target.notna()
    if "allocator_eligible" in frame:
        mask &= frame["allocator_eligible"].fillna(False).astype(bool)
    if int(mask.sum()) < max(cfg.min_samples_leaf * 4, 200):
        raise ValueError("Not enough mature causal examples to fit the opportunity allocator.")
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=cfg.learning_rate,
        max_iter=cfg.max_iter,
        max_leaf_nodes=cfg.max_leaf_nodes,
        min_samples_leaf=cfg.min_samples_leaf,
        l2_regularization=cfg.l2_regularization,
        early_stopping=False,
        random_state=cfg.random_state,
    )
    model.fit(features.loc[mask, feature_names], target.loc[mask])
    return model, feature_names


def predict_nonlinear_score(
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    feature_names: list[str],
) -> pd.Series:
    """Predict allocation utility using the immutable fitted feature contract."""
    features, available = build_opportunity_features(frame)
    if available != feature_names:
        raise ValueError("Opportunity feature contract changed between fit and inference.")
    return pd.Series(model.predict(features[feature_names]), index=frame.index, dtype=float)


def scores_to_allocations(
    frame: pd.DataFrame,
    scores: pd.Series,
    config: OpportunityAllocatorConfig | None = None,
) -> pd.Series:
    """Map same-date score ranks onto the fixed avoid-to-full allocation grid."""
    cfg = config or OpportunityAllocatorConfig()
    ranks = pd.to_numeric(scores, errors="coerce").groupby(frame["timestamp"]).rank(
        method="average", pct=True
    )
    bins = np.searchsorted(np.asarray(cfg.action_rank_thresholds), ranks.fillna(-1.0), side="right")
    levels = np.asarray(cfg.action_levels, dtype=float)
    return pd.Series(levels[bins], index=frame.index, dtype=float)


def decile_diagnostics(
    frame: pd.DataFrame,
    scores: pd.Series,
    downside_weight: float = 0.35,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure out-of-time ordering, tail separation, and monotonicity."""
    target = allocation_utility(frame, downside_weight)
    valid = target.notna() & pd.to_numeric(scores, errors="coerce").notna()
    work = pd.DataFrame(
        {
            "score": pd.to_numeric(scores, errors="coerce")[valid],
            "allocation_utility": target[valid],
            "net_alpha": _numeric(frame, "decision_net_alpha")[valid],
            "mae": _numeric(frame, "decision_mae")[valid],
        }
    )
    if work.empty:
        return pd.DataFrame(), {"spearman": None, "top_decile_lift": None, "monotonicity": None}
    ranked = work["score"].rank(method="average", pct=True)
    work["decile"] = np.minimum((ranked * 10).astype(int), 9)
    table = work.groupby("decile", as_index=False).agg(
        sample_count=("allocation_utility", "size"),
        mean_utility=("allocation_utility", "mean"),
        mean_net_alpha=("net_alpha", "mean"),
        mean_mae=("mae", "mean"),
        positive_utility_fraction=("allocation_utility", lambda values: float((values > 0.0).mean())),
    )
    decile_number = table["decile"].astype(float)
    metrics = {
        "spearman": float(work["score"].rank().corr(work["allocation_utility"].rank())),
        "top_decile_lift": float(table.iloc[-1]["mean_utility"] - work["allocation_utility"].mean()),
        "top_bottom_spread": float(table.iloc[-1]["mean_utility"] - table.iloc[0]["mean_utility"]),
        "monotonicity": float(decile_number.corr(table["mean_utility"].rank())),
    }
    return table, metrics


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")
