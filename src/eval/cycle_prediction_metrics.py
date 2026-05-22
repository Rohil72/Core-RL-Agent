from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.cycle.oracle import CycleSpan


def compute_action_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )
    labels = [0, 1, 2, 3]
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    true_counts = matrix.sum(axis=1)
    pred_counts = matrix.sum(axis=0)
    total = int(matrix.sum())
    return {
        "action_accuracy": accuracy,
        "action_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
        if len(y_true)
        else 0.0,
        "action_macro_precision": float(macro_precision),
        "action_macro_recall": float(macro_recall),
        "action_macro_f1": float(macro_f1),
        "action_weighted_precision": float(weighted_precision),
        "action_weighted_recall": float(weighted_recall),
        "action_weighted_f1": float(weighted_f1),
        "action_confusion_matrix": matrix.astype(int).tolist(),
        "action_true_distribution": {
            str(label): float(true_counts[idx] / total) if total else 0.0
            for idx, label in enumerate(labels)
        },
        "action_pred_distribution": {
            str(label): float(pred_counts[idx] / total) if total else 0.0
            for idx, label in enumerate(labels)
        },
        "action_per_class": {
            str(label): {
                "precision": float(per_precision[idx]),
                "recall": float(per_recall[idx]),
                "f1": float(per_f1[idx]),
                "support": float(per_support[idx]),
            }
            for idx, label in enumerate(labels)
        },
    }


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def compute_future_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Sequence[str],
) -> dict[str, float | dict[str, dict[str, float]]]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "future_target_mae": 0.0,
            "future_target_rmse": 0.0,
            "future_target_r2": 0.0,
            "future_target_pearson": 0.0,
            "future_target_valid_count": 0.0,
            "future_target_metrics": {},
        }

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not valid_mask.any():
        return {
            "future_target_mae": 0.0,
            "future_target_rmse": 0.0,
            "future_target_r2": 0.0,
            "future_target_pearson": 0.0,
            "future_target_valid_count": 0.0,
            "future_target_metrics": {},
        }

    diff = y_pred[valid_mask] - y_true[valid_mask]
    per_target: dict[str, dict[str, float]] = {}
    r2_values = []
    pearson_values = []
    for idx, name in enumerate(target_names):
        target_mask = valid_mask[:, idx]
        if not target_mask.any():
            per_target[name] = {
                "mae": 0.0,
                "rmse": 0.0,
                "r2": 0.0,
                "pearson": 0.0,
                "valid_count": 0.0,
            }
            continue
        target_true = y_true[target_mask, idx]
        target_pred = y_pred[target_mask, idx]
        target_diff = target_pred - target_true
        target_r2 = _safe_r2(target_true, target_pred)
        target_pearson = _safe_pearson(target_true, target_pred)
        r2_values.append(target_r2)
        pearson_values.append(target_pearson)
        per_target[name] = {
            "mae": float(np.mean(np.abs(target_diff))),
            "rmse": float(np.sqrt(np.mean(target_diff**2))),
            "r2": target_r2,
            "pearson": target_pearson,
            "valid_count": float(target_mask.sum()),
        }

    return {
        "future_target_mae": float(np.mean(np.abs(diff))),
        "future_target_rmse": float(np.sqrt(np.mean(diff**2))),
        "future_target_r2": float(np.mean(r2_values)) if r2_values else 0.0,
        "future_target_pearson": float(np.mean(pearson_values))
        if pearson_values
        else 0.0,
        "future_target_valid_count": float(valid_mask.sum()),
        "future_target_metrics": per_target,
    }


def _overlap_days(a: CycleSpan, b: CycleSpan) -> int:
    return max(0, min(a.end_idx, b.end_idx) - max(a.start_idx, b.start_idx) + 1)


def match_cycles(
    predicted: Sequence[CycleSpan],
    oracle: Sequence[CycleSpan],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, float, int, int]] = []
    for pred_idx, pred_cycle in enumerate(predicted):
        for oracle_idx, oracle_cycle in enumerate(oracle):
            overlap = _overlap_days(pred_cycle, oracle_cycle)
            if overlap <= 0:
                continue
            union = (
                max(pred_cycle.end_idx, oracle_cycle.end_idx)
                - min(pred_cycle.start_idx, oracle_cycle.start_idx)
                + 1
            )
            iou = overlap / union if union > 0 else 0.0
            candidates.append((overlap, iou, pred_idx, oracle_idx))

    candidates.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_oracle: set[int] = set()
    matches: list[tuple[int, int]] = []

    for _, _, pred_idx, oracle_idx in candidates:
        if pred_idx in matched_pred or oracle_idx in matched_oracle:
            continue
        matched_pred.add(pred_idx)
        matched_oracle.add(oracle_idx)
        matches.append((pred_idx, oracle_idx))

    return matches


def score_exit_alignment(
    predicted: CycleSpan,
    oracle: CycleSpan,
    late_penalty_factor: float = 1.5,
) -> float:
    duration = max(oracle.duration, 1)
    day_error = predicted.end_idx - oracle.end_idx
    normalized = abs(day_error) / duration
    if day_error > 0:
        normalized *= late_penalty_factor
    return float(min(normalized, 1.0))


def compute_cycle_metrics(
    predicted: Sequence[CycleSpan],
    oracle: Sequence[CycleSpan],
    catastrophic_return: float = -0.10,
    late_penalty_factor: float = 1.5,
) -> dict[str, float]:
    matches = match_cycles(predicted, oracle)
    matched_pred = {pred_idx for pred_idx, _ in matches}
    matched_oracle = {oracle_idx for _, oracle_idx in matches}

    predicted_returns = np.array([cycle.return_pct for cycle in predicted], dtype=float)
    profitable = predicted_returns > 0
    catastrophic = predicted_returns <= catastrophic_return

    lead_days = []
    lead_ratio = []
    exit_error = []
    matched_returns = []

    for pred_idx, oracle_idx in matches:
        pred_cycle = predicted[pred_idx]
        oracle_cycle = oracle[oracle_idx]
        matched_returns.append(pred_cycle.return_pct)
        if oracle_cycle.peak_idx is not None:
            lead = oracle_cycle.peak_idx - pred_cycle.start_idx
            lead_days.append(float(lead))
            lead_ratio.append(float(lead / max(oracle_cycle.duration, 1)))
        exit_error.append(
            score_exit_alignment(
                predicted=pred_cycle,
                oracle=oracle_cycle,
                late_penalty_factor=late_penalty_factor,
            )
        )

    unmatched_negative = [
        predicted[idx].return_pct <= 0
        for idx in range(len(predicted))
        if idx not in matched_pred
    ]
    unmatched_catastrophic = [
        predicted[idx].return_pct <= catastrophic_return
        for idx in range(len(predicted))
        if idx not in matched_pred
    ]

    predicted_count = len(predicted)
    oracle_count = len(oracle)
    match_count = len(matches)
    precision = float(match_count / predicted_count) if predicted_count else 0.0
    recall = float(match_count / oracle_count) if oracle_count else 0.0
    cycle_f1 = (
        float(2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "predicted_cycles": float(predicted_count),
        "oracle_cycles": float(oracle_count),
        "cycle_precision": precision,
        "cycle_recall": recall,
        "cycle_f1": cycle_f1,
        "profitable_cycle_rate": float(np.mean(profitable)) if predicted_count else 0.0,
        "average_cycle_return": float(np.mean(predicted_returns))
        if predicted_count
        else 0.0,
        "median_cycle_return": float(np.median(predicted_returns))
        if predicted_count
        else 0.0,
        "cycle_return_std": float(np.std(predicted_returns))
        if predicted_count
        else 0.0,
        "average_positive_return": float(np.mean(np.clip(predicted_returns, 0.0, None)))
        if predicted_count
        else 0.0,
        "catastrophic_cycle_rate": float(np.mean(catastrophic))
        if predicted_count
        else 0.0,
        "matched_average_return": float(np.mean(matched_returns))
        if matched_returns
        else 0.0,
        "mean_lead_time_days": float(np.mean(lead_days)) if lead_days else 0.0,
        "mean_lead_time_ratio": float(np.mean(lead_ratio)) if lead_ratio else 0.0,
        "bounded_exit_error": float(np.mean(exit_error)) if exit_error else 1.0,
        "hard_negative_false_positive_rate": float(np.mean(unmatched_negative))
        if unmatched_negative
        else 0.0,
        "catastrophic_false_positive_rate": float(np.mean(unmatched_catastrophic))
        if unmatched_catastrophic
        else 0.0,
        "matched_predicted_share": float(len(matched_pred) / predicted_count)
        if predicted_count
        else 0.0,
        "matched_oracle_share": float(len(matched_oracle) / oracle_count)
        if oracle_count
        else 0.0,
    }


def compute_cycle_moving_average_return(cycles: Sequence[CycleSpan], window: int = 50, method: str = "ema") -> dict[str, float]:
    """Compute a moving-average summary of cycle returns (independent of prediction).

    Returns a small dict with the computed scalar moving average and metadata.

    - cycles: sequence of CycleSpan (uses CycleSpan.return_pct)
    - window: lookback window for the moving average (number of cycles)
    - method: 'ema' for exponential moving average, otherwise simple rolling mean over last window
    """
    returns = np.array([c.return_pct for c in cycles], dtype=float)
    out: dict[str, float] = {"moving_avg_cycle_return": 0.0, "moving_avg_window": int(window), "moving_avg_count": float(len(returns))}
    if returns.size == 0:
        return out

    if method == "ema":
        # standard EMA alpha
        alpha = 2.0 / (float(window) + 1.0) if window > 0 else 1.0
        ema = float(returns[0])
        for r in returns[1:]:
            ema = float(alpha * r + (1.0 - alpha) * ema)
        out["moving_avg_cycle_return"] = float(ema)
    else:
        if window <= 0:
            ma = float(np.mean(returns))
        else:
            ma = float(np.mean(returns[-window:]))
        out["moving_avg_cycle_return"] = ma
    return out

