from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


BASE_RELIABILITY_FEATURES = (
    "retrieval_confidence",
    "retrieval_agreement_score",
    "retrieval_effective_sample_size",
    "retrieval_entropy",
    "retrieval_distance_weighted_confidence",
    "retrieval_historical_diversity",
    "retrieval_median_distance",
    "retrieval_alpha_ci_low",
    "retrieval_alpha_std",
    "retrieval_downside_cvar",
    "retrieval_outcome_std",
    "retrieval_expected_alpha",
    "opportunity_score",
    "alpha_interval_width",
    "positive_alpha_breadth",
    "score_dispersion",
    "cross_seed_expected_alpha_std",
    "cross_seed_confidence_std",
    "cross_seed_score_std",
    "cross_seed_neighbor_overlap",
    "seed_count",
)

MODEL_FEATURES = (*BASE_RELIABILITY_FEATURES, "state_shift_score")

OUTCOME_COLUMNS = (
    "future_blended_alpha_63",
    "future_universe_alpha_63",
    "future_return_63",
    "future_min_return_63",
    "future_max_return_63",
)

IDENTITY_COLUMNS = ("ticker", "timestamp", "open", "close")


@dataclass(frozen=True)
class ReliabilityConfig:
    """Fixed training and calibration contract for memory reliability."""

    fit_fraction: float = 0.70
    calibration_embargo_sessions: int = 63
    minimum_fit_rows: int = 500
    minimum_calibration_rows: int = 200
    logistic_c: float = 0.10
    useful_alpha_after_costs: float = 0.002
    isotonic_minimum_unique_scores: int = 10
    random_seed: int = 7

    def __post_init__(self) -> None:
        if not 0.50 <= self.fit_fraction < 1.0:
            raise ValueError("fit_fraction must lie in [0.50, 1.0).")
        if self.calibration_embargo_sessions < 0:
            raise ValueError("calibration_embargo_sessions cannot be negative.")
        if self.minimum_fit_rows <= 0 or self.minimum_calibration_rows <= 0:
            raise ValueError("Minimum row counts must be positive.")
        if self.logistic_c <= 0:
            raise ValueError("logistic_c must be positive.")


@dataclass
class ReliabilityModel:
    """Regularized correctness and error models with held-out calibration."""

    classifier: Pipeline
    error_model: Pipeline
    calibrator: IsotonicRegression | None
    shift_median: np.ndarray
    shift_scale: np.ndarray
    calibration_thresholds: dict[float, float]
    base_rate: float
    error_prediction_cap: float
    config: ReliabilityConfig
    fit_end: pd.Timestamp
    calibration_start: pd.Timestamp

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach reliability, expected error, and train-relative shift diagnostics."""
        out = frame.copy()
        matrix = _base_matrix(out)
        shift = _shift_score(matrix, self.shift_median, self.shift_scale)
        out["state_shift_score"] = shift
        model_matrix = np.column_stack([matrix, shift])
        raw = self.classifier.predict_proba(model_matrix)[:, 1]
        calibrated = self.calibrator.predict(raw) if self.calibrator is not None else raw
        out["reliability_probability"] = np.clip(calibrated, 0.0, 1.0)
        out["predicted_absolute_error"] = np.clip(
            self.error_model.predict(model_matrix),
            0.0,
            self.error_prediction_cap,
        )
        return out

    def to_dict(self) -> dict[str, Any]:
        """Return a portable model audit without serializing executable objects."""
        classifier = self.classifier.named_steps["model"]
        error_model = self.error_model.named_steps["model"]
        payload: dict[str, Any] = {
            "features": list(MODEL_FEATURES),
            "config": asdict(self.config),
            "fit_end": self.fit_end.isoformat(),
            "calibration_start": self.calibration_start.isoformat(),
            "base_rate": self.base_rate,
            "error_prediction_cap": self.error_prediction_cap,
            "calibration_thresholds": {
                str(key): value for key, value in self.calibration_thresholds.items()
            },
            "classifier_coefficients": classifier.coef_.ravel().tolist(),
            "classifier_intercept": classifier.intercept_.ravel().tolist(),
            "error_coefficients": error_model.coef_.ravel().tolist(),
            "error_intercept": float(error_model.intercept_),
            "shift_median": self.shift_median.tolist(),
            "shift_scale": self.shift_scale.tolist(),
        }
        if self.calibrator is not None:
            payload["isotonic_x_thresholds"] = self.calibrator.X_thresholds_.tolist()
            payload["isotonic_y_thresholds"] = self.calibrator.y_thresholds_.tolist()
        return payload


def build_consensus_frame(
    seed_frames: Mapping[int, pd.DataFrame],
    neighbor_frames: Mapping[int, pd.DataFrame] | None = None,
    *,
    useful_alpha_after_costs: float = 0.002,
) -> pd.DataFrame:
    """Collapse seed-specific memory estimates into one state-level reliability table."""
    if not seed_frames:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for seed, source in seed_frames.items():
        frame = source.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["ticker"] = frame["ticker"].astype(str)
        frame["seed"] = int(seed)
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    keys = ["ticker", "timestamp"]
    signal_columns = [
        column
        for column in combined.columns
        if column.startswith("retrieval_") or column in {"opportunity_score", "opportunity_quality"}
    ]
    numeric_signals = [
        column for column in signal_columns if pd.api.types.is_numeric_dtype(combined[column])
    ]
    grouped = combined.groupby(keys, sort=True, observed=True)
    consensus = grouped[numeric_signals].mean().reset_index()
    for column in (*IDENTITY_COLUMNS[2:], *OUTCOME_COLUMNS):
        if column in combined:
            consensus[column] = grouped[column].first().to_numpy()
    consensus["seed_count"] = grouped["seed"].nunique().to_numpy(dtype=float)
    consensus["cross_seed_expected_alpha_std"] = _group_std(grouped, "retrieval_expected_alpha")
    consensus["cross_seed_confidence_std"] = _group_std(grouped, "retrieval_confidence")
    consensus["cross_seed_score_std"] = _group_std(grouped, "opportunity_score")
    consensus["cross_seed_neighbor_overlap"] = 0.0
    if neighbor_frames:
        overlap = compute_neighbor_overlap(neighbor_frames)
        consensus = consensus.merge(overlap, on=keys, how="left", suffixes=("", "_computed"))
        computed = consensus.pop("cross_seed_neighbor_overlap_computed")
        consensus["cross_seed_neighbor_overlap"] = computed.fillna(
            consensus["cross_seed_neighbor_overlap"]
        )

    consensus["alpha_interval_width"] = (
        consensus.get("retrieval_alpha_ci_high", np.nan)
        - consensus.get("retrieval_alpha_ci_low", np.nan)
    )
    positive = (
        pd.to_numeric(consensus.get("retrieval_alpha_ci_low"), errors="coerce") > 0.0
    )
    if "retrieval_ood_pass" in consensus:
        consensus["retrieval_ood_pass"] = consensus["retrieval_ood_pass"] >= 0.5
        positive &= consensus["retrieval_ood_pass"]
    consensus["positive_alpha_breadth"] = positive.groupby(consensus["timestamp"]).transform("mean")
    consensus["score_dispersion"] = consensus.groupby("timestamp")["opportunity_score"].transform("std").fillna(0.0)
    realized = pd.to_numeric(consensus["future_blended_alpha_63"], errors="coerce")
    expected = pd.to_numeric(consensus["retrieval_expected_alpha"], errors="coerce")
    consensus["direction_correct"] = ((expected >= 0.0) == (realized >= 0.0)).astype(int)
    consensus["prediction_error"] = (expected - realized).abs()
    consensus["useful_trade"] = (realized > float(useful_alpha_after_costs)).astype(int)
    consensus["alpha_interval_covered"] = (
        (realized >= consensus["retrieval_alpha_ci_low"])
        & (realized <= consensus["retrieval_alpha_ci_high"])
    ).astype(int)
    return consensus.sort_values(["timestamp", "ticker"]).reset_index(drop=True)


def compute_neighbor_overlap(neighbor_frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Compute mean pairwise Jaccard overlap of retrieved experiences across seeds."""
    by_query: dict[tuple[str, pd.Timestamp], dict[int, set[str]]] = {}
    for seed, source in neighbor_frames.items():
        if source.empty:
            continue
        frame = source.copy()
        frame["query_timestamp"] = pd.to_datetime(frame["query_timestamp"], utc=True)
        identity = (
            frame["experience_id"].astype(str)
            if "experience_id" in frame
            else frame["neighbor_ticker"].astype(str) + "|" + frame["neighbor_timestamp"].astype(str)
        )
        frame = frame.assign(_experience=identity)
        for (ticker, timestamp), group in frame.groupby(["query_ticker", "query_timestamp"], sort=False):
            by_query.setdefault((str(ticker), timestamp), {})[int(seed)] = set(group["_experience"])
    records: list[dict[str, Any]] = []
    for (ticker, timestamp), seed_sets in by_query.items():
        scores: list[float] = []
        for left, right in combinations(seed_sets.values(), 2):
            union = left | right
            scores.append(len(left & right) / len(union) if union else 0.0)
        records.append(
            {
                "ticker": ticker,
                "timestamp": timestamp,
                "cross_seed_neighbor_overlap": float(np.mean(scores)) if scores else 1.0,
            }
        )
    return pd.DataFrame(records)


def fit_reliability_model(
    train: pd.DataFrame,
    config: ReliabilityConfig | None = None,
) -> ReliabilityModel:
    """Fit on an early window and calibrate on an embargoed chronological tail."""
    cfg = config or ReliabilityConfig()
    frame = train.dropna(subset=["future_blended_alpha_63"]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    dates = np.asarray(sorted(frame["timestamp"].unique()))
    if len(dates) < 3:
        raise ValueError("Reliability training requires at least three sessions.")
    split_index = min(max(int(len(dates) * cfg.fit_fraction), 1), len(dates) - 2)
    calibration_index = min(split_index + cfg.calibration_embargo_sessions, len(dates) - 1)
    fit_end = pd.Timestamp(dates[split_index - 1])
    calibration_start = pd.Timestamp(dates[calibration_index])
    fit_rows = frame.loc[frame["timestamp"] <= fit_end].copy()
    calibration_rows = frame.loc[frame["timestamp"] >= calibration_start].copy()
    if len(fit_rows) < cfg.minimum_fit_rows:
        raise ValueError(f"Reliability fit has {len(fit_rows)} rows; requires {cfg.minimum_fit_rows}.")
    if len(calibration_rows) < cfg.minimum_calibration_rows:
        raise ValueError(
            f"Reliability calibration has {len(calibration_rows)} rows; requires {cfg.minimum_calibration_rows}."
        )
    if fit_rows["useful_trade"].nunique() < 2:
        raise ValueError("Reliability fit requires both useful and non-useful outcomes.")

    fit_matrix = _base_matrix(fit_rows)
    shift_median = np.nanmedian(fit_matrix, axis=0)
    shift_scale = np.nanpercentile(fit_matrix, 75, axis=0) - np.nanpercentile(fit_matrix, 25, axis=0)
    shift_scale = np.where(np.isfinite(shift_scale) & (shift_scale > 1e-9), shift_scale, 1.0)
    fit_shift = _shift_score(fit_matrix, shift_median, shift_scale)
    model_matrix = np.column_stack([fit_matrix, fit_shift])
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    C=cfg.logistic_c,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=cfg.random_seed,
                ),
            ),
        ]
    )
    classifier.fit(model_matrix, fit_rows["useful_trade"].to_numpy(dtype=int))
    error_rows = fit_rows.loc[np.isfinite(pd.to_numeric(fit_rows["prediction_error"], errors="coerce"))]
    if len(error_rows) < max(20, cfg.minimum_fit_rows // 4):
        raise ValueError("Reliability error model has too few finite prediction-error rows.")
    error_matrix = _base_matrix(error_rows)
    error_shift = _shift_score(error_matrix, shift_median, shift_scale)
    error_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", HuberRegressor(alpha=1.0, max_iter=1000)),
        ]
    )
    error_model.fit(
        np.column_stack([error_matrix, error_shift]),
        error_rows["prediction_error"].to_numpy(dtype=float),
    )

    calibration_matrix = _base_matrix(calibration_rows)
    calibration_shift = _shift_score(calibration_matrix, shift_median, shift_scale)
    raw = classifier.predict_proba(np.column_stack([calibration_matrix, calibration_shift]))[:, 1]
    calibrator: IsotonicRegression | None = None
    if (
        np.unique(raw).size >= cfg.isotonic_minimum_unique_scores
        and calibration_rows["useful_trade"].nunique() == 2
    ):
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, calibration_rows["useful_trade"].to_numpy(dtype=int))
        calibrated = calibrator.predict(raw)
    else:
        calibrated = raw
    thresholds = {1.0: float("-inf")}
    thresholds.update(
        {
            coverage: float(np.quantile(calibrated, 1.0 - coverage))
            for coverage in (0.75, 0.50, 0.25)
        }
    )
    return ReliabilityModel(
        classifier=classifier,
        error_model=error_model,
        calibrator=calibrator,
        shift_median=shift_median,
        shift_scale=shift_scale,
        calibration_thresholds=thresholds,
        base_rate=float(fit_rows["useful_trade"].mean()),
        error_prediction_cap=float(error_rows["prediction_error"].quantile(0.99)),
        config=cfg,
        fit_end=fit_end,
        calibration_start=calibration_start,
    )


def reliability_metrics(predictions: pd.DataFrame, model: ReliabilityModel) -> dict[str, float | None]:
    """Measure calibration, ordering, interval coverage, and error awareness."""
    frame = predictions.dropna(subset=["reliability_probability", "future_blended_alpha_63"])
    if frame.empty:
        return {}
    y = frame["useful_trade"].to_numpy(dtype=int)
    probability = frame["reliability_probability"].to_numpy(dtype=float)
    baseline_brier = float(np.mean((y - model.base_rate) ** 2))
    brier = float(brier_score_loss(y, probability))
    confidence = np.clip(frame["retrieval_confidence"].to_numpy(dtype=float), 0.0, 1.0)
    confidence_brier = float(brier_score_loss(y, confidence))
    realised = frame["future_blended_alpha_63"].to_numpy(dtype=float)
    error = frame["prediction_error"].to_numpy(dtype=float)
    finite_error = error[np.isfinite(error)]
    top = frame.loc[frame["reliability_probability"] >= frame["reliability_probability"].quantile(0.75)]
    bottom = frame.loc[frame["reliability_probability"] <= frame["reliability_probability"].quantile(0.25)]
    return {
        "row_count": float(len(frame)),
        "useful_trade_rate": float(np.mean(y)),
        "brier": brier,
        "brier_skill": 1.0 - brier / baseline_brier if baseline_brier > 1e-12 else 0.0,
        "confidence_brier": confidence_brier,
        "log_loss": float(log_loss(y, np.clip(probability, 1e-6, 1.0 - 1e-6))),
        "roc_auc": _binary_metric(roc_auc_score, y, probability),
        "average_precision": _binary_metric(average_precision_score, y, probability),
        "expected_calibration_error": _expected_calibration_error(y, probability),
        "reliability_alpha_spearman": _spearman(probability, realised),
        "reliability_negative_error_spearman": _spearman(probability, -error),
        "top_quartile_realized_alpha": float(top["future_blended_alpha_63"].mean()),
        "bottom_quartile_realized_alpha": float(bottom["future_blended_alpha_63"].mean()),
        "top_bottom_alpha_spread": float(
            top["future_blended_alpha_63"].mean() - bottom["future_blended_alpha_63"].mean()
        ),
        "direction_accuracy": float(frame["direction_correct"].mean()),
        "interval_coverage": float(frame["alpha_interval_covered"].mean()),
        "mean_prediction_error": float(np.mean(finite_error)) if finite_error.size else None,
        "mean_predicted_error": float(frame["predicted_absolute_error"].mean()),
        "mean_shift_score": float(frame["state_shift_score"].mean()),
    }


def risk_coverage_table(
    predictions: pd.DataFrame,
    model: ReliabilityModel,
) -> pd.DataFrame:
    """Evaluate fixed calibration-derived participation levels without validation quantiles."""
    records: list[dict[str, Any]] = []
    for nominal, threshold in sorted(model.calibration_thresholds.items(), reverse=True):
        selected = predictions.loc[predictions["reliability_probability"] >= threshold]
        records.append(
            {
                "nominal_coverage": nominal,
                "threshold": threshold,
                "realized_coverage": len(selected) / len(predictions) if len(predictions) else 0.0,
                "row_count": len(selected),
                "useful_trade_rate": float(selected["useful_trade"].mean()) if len(selected) else None,
                "mean_realized_alpha": float(selected["future_blended_alpha_63"].mean()) if len(selected) else None,
                "mean_prediction_error": float(selected["prediction_error"].mean()) if len(selected) else None,
                "downside_breach_rate": float((selected["future_min_return_63"] < -0.12).mean())
                if len(selected)
                else None,
            }
        )
    return pd.DataFrame(records)


def univariate_reliability_table(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...] = BASE_RELIABILITY_FEATURES,
    bins: int = 10,
) -> pd.DataFrame:
    """Measure whether train-defined feature bins retain outcome ordering on validation."""
    records: list[dict[str, Any]] = []
    for feature in features:
        if feature not in train or feature not in validation:
            continue
        reference = pd.to_numeric(train[feature], errors="coerce").dropna()
        if reference.nunique() < 2:
            continue
        edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        labels = pd.cut(pd.to_numeric(validation[feature], errors="coerce"), bins=edges, labels=False)
        for bin_index, group in validation.assign(_bin=labels).dropna(subset=["_bin"]).groupby("_bin"):
            records.append(
                {
                    "feature": feature,
                    "bin": int(bin_index),
                    "row_count": len(group),
                    "feature_mean": float(group[feature].mean()),
                    "useful_trade_rate": float(group["useful_trade"].mean()),
                    "mean_realized_alpha": float(group["future_blended_alpha_63"].mean()),
                    "mean_prediction_error": float(group["prediction_error"].mean()),
                    "interval_coverage": float(group["alpha_interval_covered"].mean()),
                }
            )
    return pd.DataFrame(records)


def domain_shift_auc(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    random_seed: int = 7,
    maximum_rows_per_domain: int = 10000,
) -> float | None:
    """Diagnose whether validation states are distinguishable from training states."""
    rng = np.random.default_rng(random_seed)
    left = _base_matrix(train)
    right = _base_matrix(validation)
    if len(left) > maximum_rows_per_domain:
        left = left[rng.choice(len(left), maximum_rows_per_domain, replace=False)]
    if len(right) > maximum_rows_per_domain:
        right = right[rng.choice(len(right), maximum_rows_per_domain, replace=False)]
    matrix = np.vstack([left, right])
    labels = np.concatenate([np.zeros(len(left), dtype=int), np.ones(len(right), dtype=int)])
    if len(np.unique(labels)) < 2 or len(matrix) < 20:
        return None
    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_seed,
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(C=0.10, max_iter=2000, random_state=random_seed),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return float(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))


def _group_std(grouped: Any, column: str) -> np.ndarray:
    if column not in grouped.obj:
        return np.zeros(len(grouped), dtype=float)
    return grouped[column].std(ddof=0).fillna(0.0).to_numpy(dtype=float)


def _base_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for feature in BASE_RELIABILITY_FEATURES:
        if feature in frame:
            columns.append(pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float))
        else:
            columns.append(np.full(len(frame), np.nan, dtype=float))
    return np.column_stack(columns)


def _shift_score(matrix: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    robust_z = np.abs((matrix - median) / scale)
    robust_z = np.clip(robust_z, 0.0, 10.0)
    valid = np.isfinite(robust_z)
    count = valid.sum(axis=1)
    total = np.where(valid, robust_z, 0.0).sum(axis=1)
    return np.divide(total, count, out=np.full(len(matrix), 10.0), where=count > 0)


def _binary_metric(function: Any, y: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(y).size < 2:
        return None
    return float(function(y, score))


def _expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    observed, predicted = calibration_curve(y, probability, n_bins=bins, strategy="quantile")
    return float(np.mean(np.abs(observed - predicted))) if len(observed) else 0.0


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return None
    return float(spearmanr(left[mask], right[mask]).statistic)
