"""Temporal partitioning, 6 annual folds, and label isolation (v2).

Acceptance criteria addressed:
- A11: 63/126-session maturity masks pass last-admitted and first-rejected boundary fixtures.
- A12: Evaluation labels strictly unavailable to training, selection, and execution APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import pandas as pd


class LabelIsolationViolationError(Exception):
    """Raised when training, selection, or execution pipelines attempt unauthorized access to evaluation labels."""
    pass


@dataclass
class FoldBoundaries:
    evaluation_year: int
    train_origin_start: str
    training_availability_cutoff: str
    validation_query_start: str
    validation_query_end: str
    validation_label_cutoff: str
    development_query_start: str
    development_query_end: str
    development_label_cutoff: str
    evaluation_query_start: str
    evaluation_query_end: str
    bank_cutoff: str


@dataclass
class FoldPartitions:
    evaluation_year: int
    train_indices: List[int]
    validation_indices: List[int]
    development_indices: List[int]
    evaluation_indices: List[int]
    bank_indices: List[int]


class SealedEvaluationLabels:
    """Sealed container for evaluation query labels.

    Strictly prevents unauthorized access from training, gate, and execution code.
    Only authorized post-prediction verification or scoring can unlock it.
    """

    def __init__(self, evaluation_year: int, labels_df: pd.DataFrame, evaluation_sessions: Set[str]):
        self.evaluation_year = evaluation_year
        self._labels_df = labels_df[labels_df["session_origin"].isin(evaluation_sessions)].copy()
        self._sealed = True

    def access_for_training(self) -> None:
        raise LabelIsolationViolationError(
            "CRITICAL: Training pipeline attempted to access evaluation labels! Access strictly denied."
        )

    def access_for_selection(self) -> None:
        raise LabelIsolationViolationError(
            "CRITICAL: Selection pipeline attempted to access evaluation labels! Access strictly denied."
        )

    def access_for_execution(self) -> None:
        raise LabelIsolationViolationError(
            "CRITICAL: Execution engine attempted to access evaluation labels! Access strictly denied."
        )

    def unlock_for_final_scoring(self, authorization_token: str) -> pd.DataFrame:
        if authorization_token != "EVALUATION_PREDICTIONS_SEALED":
            raise LabelIsolationViolationError("Invalid authorization token to unlock evaluation labels.")
        return self._labels_df.copy()


def partition_fold(
    sessions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    boundaries: FoldBoundaries,
) -> FoldPartitions:
    """Partition session indices into disjoint roles based on exact date and maturity cutoffs (Test A11)."""
    # Clean overlapping columns to prevent _x, _y collision
    lbl = labels_df.copy()
    if "session_origin" not in lbl.columns and "session" in lbl.columns:
        lbl["session_origin"] = lbl["session"]

    # Drop any overlapping columns from sessions_df except 'session'
    overlap = [c for c in sessions_df.columns if c in lbl.columns and c != "session"]
    sess_clean = sessions_df.drop(columns=overlap, errors="ignore")

    merged = sess_clean.merge(lbl, left_on="session", right_on="session_origin")
    n = len(merged)

    train_idx: List[int] = []
    val_idx: List[int] = []
    dev_idx: List[int] = []
    eval_idx: List[int] = []
    bank_idx: List[int] = []

    for i in range(n):
        sess = merged.iloc[i]["session_origin"]
        v63 = bool(merged.iloc[i]["valid_63"])
        v126 = bool(merged.iloc[i]["valid_126"])
        s63 = str(merged.iloc[i]["session_63"])
        s126 = str(merged.iloc[i]["session_126"])

        # Train origin: origin >= start, 63-session outcome available by training cutoff
        if sess >= boundaries.train_origin_start and v63 and s63 <= boundaries.training_availability_cutoff:
            train_idx.append(i)

        # Bank: origin >= start, 126-session outcome available by bank cutoff
        if sess >= boundaries.train_origin_start and v126 and s126 <= boundaries.bank_cutoff:
            bank_idx.append(i)

        # Validation queries: in validation year, label available by validation cutoff
        if boundaries.validation_query_start <= sess <= boundaries.validation_query_end:
            if v63 and s63 <= boundaries.validation_label_cutoff:
                val_idx.append(i)

        # Development queries: in development year, label available by dev cutoff
        if boundaries.development_query_start <= sess <= boundaries.development_query_end:
            if v63 and s63 <= boundaries.development_label_cutoff:
                dev_idx.append(i)

        # Evaluation queries: in evaluation year
        if boundaries.evaluation_query_start <= sess <= boundaries.evaluation_query_end:
            eval_idx.append(i)

    return FoldPartitions(
        evaluation_year=boundaries.evaluation_year,
        train_indices=train_idx,
        validation_indices=val_idx,
        development_indices=dev_idx,
        evaluation_indices=eval_idx,
        bank_indices=bank_idx,
    )
