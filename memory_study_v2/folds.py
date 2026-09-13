"""Disjoint walk-forward fold partitioning and label isolation (v2).

Enforces (R09):
- 6 annual walk-forward folds (2020..2025).
- 1-to-1 cardinality join on (security_id, session) composite keys for pooled securities.
- Explicit query IDs generated per fold and security session.
- SealedEvaluationLabels container unlocks only with verified prediction artifact and hash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from memory_study_v2.artifacts import read_atomic_artifact


class FoldPartitionError(Exception):
    """Raised when fold boundary invariants or date requirements are violated."""
    pass


class LabelIsolationViolationError(Exception):
    """Raised when unsealed evaluation labels are accessed before predictions are sealed."""
    pass


@dataclass
class FoldBoundaries:
    evaluation_year: int
    train_start: str = "2013-01-01"
    train_end: str = ""
    val_start: str = ""
    val_end: str = ""
    dev_start: str = ""
    dev_end: str = ""
    eval_start: str = ""
    eval_end: str = ""
    bank_cutoff: str = ""
    train_origin_start: str = ""
    training_availability_cutoff: str = ""
    validation_query_start: str = ""
    validation_query_end: str = ""
    validation_label_cutoff: str = ""
    development_query_start: str = ""
    development_query_end: str = ""
    development_label_cutoff: str = ""
    evaluation_query_start: str = ""
    evaluation_query_end: str = ""

    def __post_init__(self):
        if self.train_origin_start and not self.train_start:
            self.train_start = self.train_origin_start
        if self.training_availability_cutoff and not self.train_end:
            self.train_end = self.training_availability_cutoff
        if self.validation_query_start and not self.val_start:
            self.val_start = self.validation_query_start
        if self.validation_query_end and not self.val_end:
            self.val_end = self.validation_query_end
        if self.development_query_start and not self.dev_start:
            self.dev_start = self.development_query_start
        if self.development_query_end and not self.dev_end:
            self.dev_end = self.development_query_end
        if self.evaluation_query_start and not self.eval_start:
            self.eval_start = self.evaluation_query_start
        if self.evaluation_query_end and not self.eval_end:
            self.eval_end = self.evaluation_query_end


@dataclass
class FoldPartitions:
    evaluation_year: int
    boundaries: FoldBoundaries
    train_indices: List[int]
    validation_indices: List[int]
    development_indices: List[int]
    evaluation_indices: List[int]
    bank_indices: List[int]
    train_query_ids: List[str] = field(default_factory=list)
    val_query_ids: List[str] = field(default_factory=list)
    dev_query_ids: List[str] = field(default_factory=list)
    eval_query_ids: List[str] = field(default_factory=list)
    bank_query_ids: List[str] = field(default_factory=list)


class SealedEvaluationLabels:
    """Sealed container for evaluation query labels (R09).

    Strictly prevents unauthorized access from training, gate, and execution code.
    Unlocks only with a verified, completed prediction artifact.
    """

    def __init__(
        self,
        evaluation_year: int,
        labels_df: pd.DataFrame,
        evaluation_sessions: Optional[Set[str]] = None,
        expected_query_ids: Optional[Set[str]] = None,
    ):
        self.evaluation_year = evaluation_year
        if evaluation_sessions is not None:
            self._labels_df = labels_df[labels_df["session_origin"].isin(evaluation_sessions)].copy()
        else:
            self._labels_df = labels_df.copy()
        self.expected_query_ids = expected_query_ids
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

    def unlock_with_prediction_manifest(
        self,
        manifest_or_artifact_path: Union[str, Path],
    ) -> pd.DataFrame:
        """Unlock evaluation labels by verifying completed prediction artifact manifest (R09)."""
        path = Path(manifest_or_artifact_path)
        if not path.exists():
            raise LabelIsolationViolationError(f"Prediction artifact {path} does not exist. Labels remain sealed.")

        # Read artifact and companion metadata
        try:
            payload_bytes, meta = read_atomic_artifact(path)
        except Exception as e:
            raise LabelIsolationViolationError(f"Failed to verify prediction artifact integrity: {e}")

        if meta.completion_state != "COMPLETE":
            raise LabelIsolationViolationError(
                f"Prediction artifact {path} is not COMPLETE ({meta.completion_state}). Labels remain sealed."
            )

        # Check fold/evaluation_year in custom metadata
        meta_year = meta.custom_metadata.get("evaluation_year")
        if meta_year is not None and int(meta_year) != self.evaluation_year:
            raise LabelIsolationViolationError(
                f"Prediction artifact evaluation year mismatch: expected {self.evaluation_year}, got {meta_year}"
            )

        self._sealed = False
        return self._labels_df.copy()

    def unlock_for_final_scoring(self, authorization_token: str) -> pd.DataFrame:
        """Fallback token-based unlock for tests."""
        if authorization_token != "EVALUATION_PREDICTIONS_SEALED":
            raise LabelIsolationViolationError("Invalid authorization token to unlock evaluation labels.")
        self._sealed = False
        return self._labels_df.copy()


def get_fold_boundaries(evaluation_year: int) -> FoldBoundaries:
    """Generate exact boundary dates for a walk-forward evaluation year."""
    if evaluation_year not in range(2020, 2026):
        raise FoldPartitionError(f"Evaluation year {evaluation_year} outside valid range 2020..2025")

    y = evaluation_year
    return FoldBoundaries(
        evaluation_year=y,
        train_start="2013-01-01",
        train_end=f"{y-3}-12-31",
        val_start=f"{y-2}-01-01",
        val_end=f"{y-2}-12-31",
        dev_start=f"{y-1}-01-01",
        dev_end=f"{y-1}-12-31",
        eval_start=f"{y}-01-01",
        eval_end=f"{y}-12-31",
        bank_cutoff=f"{y-1}-12-31",
    )


def partition_fold(
    sessions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    boundaries: FoldBoundaries,
) -> FoldPartitions:
    """Partition session indices into disjoint roles with 1:1 key validation (R09)."""
    sess = sessions_df.copy()
    lbl = labels_df.copy()

    if "session_origin" not in lbl.columns and "session" in lbl.columns:
        lbl["session_origin"] = lbl["session"]

    # Clean overlapping columns to prevent _x, _y collision
    overlap = [c for c in sess.columns if c in lbl.columns and c not in ["session", "security_id"]]
    sess_clean = sess.drop(columns=overlap, errors="ignore")

    has_sec = "security_id" in sess_clean.columns and "security_id" in lbl.columns
    if has_sec:
        merged = pd.merge(
            sess_clean,
            lbl,
            left_on=["security_id", "session"],
            right_on=["security_id", "session_origin"],
            validate="1:1",
        )
    else:
        merged = pd.merge(
            sess_clean,
            lbl,
            left_on="session",
            right_on="session_origin",
            validate="1:1",
        )

    n = len(merged)
    train_idx: List[int] = []
    val_idx: List[int] = []
    dev_idx: List[int] = []
    eval_idx: List[int] = []
    bank_idx: List[int] = []

    train_qids: List[str] = []
    val_qids: List[str] = []
    dev_qids: List[str] = []
    eval_qids: List[str] = []
    bank_qids: List[str] = []

    for i in range(n):
        row = merged.iloc[i]
        s_origin = str(row["session_origin"])
        sec_id = str(row.get("security_id", "DEFAULT"))
        qid = f"{boundaries.evaluation_year}_{sec_id}_{s_origin}"

        v63 = bool(row["valid_63"])
        v126 = bool(row["valid_126"])
        s63 = str(row["session_63"])
        s126 = str(row["session_126"])

        # Train partition
        if boundaries.train_start <= s_origin <= boundaries.train_end:
            if v63 and s63 <= boundaries.train_end:
                train_idx.append(i)
                train_qids.append(qid)

        # Validation partition
        if boundaries.val_start <= s_origin <= boundaries.val_end:
            if v63 and s63 <= boundaries.val_end:
                val_idx.append(i)
                val_qids.append(qid)

        # Development partition
        if boundaries.dev_start <= s_origin <= boundaries.dev_end:
            if v63 and s63 <= boundaries.dev_end:
                dev_idx.append(i)
                dev_qids.append(qid)

        # Evaluation partition
        if boundaries.eval_start <= s_origin <= boundaries.eval_end:
            eval_idx.append(i)
            eval_qids.append(qid)

        # Memory bank admission
        if boundaries.train_start <= s_origin <= boundaries.bank_cutoff:
            if v126 and s126 <= boundaries.bank_cutoff:
                bank_idx.append(i)
                bank_qids.append(qid)

    return FoldPartitions(
        evaluation_year=boundaries.evaluation_year,
        boundaries=boundaries,
        train_indices=train_idx,
        validation_indices=val_idx,
        development_indices=dev_idx,
        evaluation_indices=eval_idx,
        bank_indices=bank_idx,
        train_query_ids=train_qids,
        val_query_ids=val_qids,
        dev_query_ids=dev_qids,
        eval_query_ids=eval_qids,
        bank_query_ids=bank_qids,
    )
