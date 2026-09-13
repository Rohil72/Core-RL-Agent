"""Acceptance & Regression Test for R09: Fold assembly pooled join and sealed prediction unlock."""

import json
from pathlib import Path
import pandas as pd
import pytest

from memory_study_v2.folds import (
    FoldBoundaries,
    LabelIsolationViolationError,
    SealedEvaluationLabels,
    get_fold_boundaries,
    partition_fold,
)
from memory_study_v2.predict import PredictionQuery, generate_and_seal_predictions


def test_fold_assembly_two_securities_same_date():
    boundaries = get_fold_boundaries(2020)
    # Sessions with 2 securities on the same date
    sessions_df = pd.DataFrame({
        "security_id": ["US_AAPL", "US_MSFT"],
        "session": ["2015-05-01", "2015-05-01"],
    })
    labels_df = pd.DataFrame({
        "security_id": ["US_AAPL", "US_MSFT"],
        "session_origin": ["2015-05-01", "2015-05-01"],
        "target_value": [0.05, 0.03],
        "session_63": ["2015-08-01", "2015-08-01"],
        "session_126": ["2015-11-01", "2015-11-01"],
        "valid_63": [True, True],
        "valid_126": [True, True],
    })
    partitions = partition_fold(sessions_df, labels_df, boundaries)
    # Exactly 2 train records, no Cartesian duplication
    assert len(partitions.train_indices) == 2
    assert "2020_US_AAPL_2015-05-01" in partitions.train_query_ids
    assert "2020_US_MSFT_2015-05-01" in partitions.train_query_ids


def test_sealed_evaluation_labels_unlock_with_verified_manifest(tmp_path):
    labels_df = pd.DataFrame({
        "session_origin": ["2020-01-02", "2020-01-03"],
        "target_value": [0.01, 0.02],
    })
    sealed = SealedEvaluationLabels(evaluation_year=2020, labels_df=labels_df)

    # 1. Direct access fails
    with pytest.raises(LabelIsolationViolationError):
        sealed.access_for_training()
    with pytest.raises(LabelIsolationViolationError):
        sealed.access_for_execution()

    # 2. Missing manifest fails
    with pytest.raises(LabelIsolationViolationError, match="does not exist"):
        sealed.unlock_with_prediction_manifest(tmp_path / "nonexistent.json")

    # 3. Create valid prediction artifact
    queries = [
        PredictionQuery(2020, "SEC_0", "2020-01-02", 0.01, 0.01, 0.02),
        PredictionQuery(2020, "SEC_0", "2020-01-03", 0.02, 0.01, 0.02),
    ]
    pred_path, _ = generate_and_seal_predictions(
        "MEM_SIM", 2020, queries, tmp_path / "preds.json"
    )

    unlocked = sealed.unlock_with_prediction_manifest(pred_path)
    assert len(unlocked) == 2
