from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.cycle.oracle import CycleSpan


def compute_action_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else 0.0
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "action_accuracy": accuracy,
        "action_macro_precision": float(precision),
        "action_macro_recall": float(recall),
        "action_macro_f1": float(f1),
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

    return {
        "predicted_cycles": float(predicted_count),
        "oracle_cycles": float(oracle_count),
        "cycle_precision": float(match_count / predicted_count)
        if predicted_count
        else 0.0,
        "cycle_recall": float(match_count / oracle_count) if oracle_count else 0.0,
        "profitable_cycle_rate": float(np.mean(profitable)) if predicted_count else 0.0,
        "average_cycle_return": float(np.mean(predicted_returns))
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
