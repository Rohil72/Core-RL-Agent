"""Acceptance Test A12: Evaluation labels unavailable to training, selection and execution APIs."""

import pytest
import pandas as pd
from memory_study_v2.folds import SealedEvaluationLabels, LabelIsolationViolationError


def test_sealed_evaluation_labels_reject_training_and_execution_access():
    """Verify evaluation query labels strictly deny access to training, selection, and execution."""
    df_labels = pd.DataFrame({
        "session_origin": ["2020-01-02", "2020-01-03"],
        "target_value": [0.05, -0.02],
    })
    vault = SealedEvaluationLabels(
        evaluation_year=2020,
        labels_df=df_labels,
        evaluation_sessions={"2020-01-02", "2020-01-03"},
    )

    # Training access must raise
    with pytest.raises(LabelIsolationViolationError, match="Training pipeline attempted to access evaluation labels"):
        vault.access_for_training()

    # Selection access must raise
    with pytest.raises(LabelIsolationViolationError, match="Selection pipeline attempted to access evaluation labels"):
        vault.access_for_selection()

    # Execution access must raise
    with pytest.raises(LabelIsolationViolationError, match="Execution engine attempted to access evaluation labels"):
        vault.access_for_execution()

    # Direct token method must not exist
    assert not hasattr(vault, "unlock_for_final_scoring"), "Token-based bypass must be completely removed!"


def test_sealed_evaluation_labels_unlock_with_verified_manifest_and_query_ids(tmp_path):
    """Verify evaluation labels unlock only with verified complete predictions covering expected query IDs."""
    from memory_study_v2.predict import PredictionQuery, generate_and_seal_predictions

    df_labels = pd.DataFrame({
        "session_origin": ["2020-01-02", "2020-01-03"],
        "target_value": [0.05, -0.02],
    })
    expected_qids = ["2020_SEC_0_2020-01-02", "2020_SEC_0_2020-01-03"]
    vault = SealedEvaluationLabels(
        evaluation_year=2020,
        labels_df=df_labels,
        evaluation_sessions={"2020-01-02", "2020-01-03"},
        expected_query_ids=expected_qids,
    )

    # 1. Non-existent artifact fails
    with pytest.raises(LabelIsolationViolationError, match="does not exist"):
        vault.unlock_with_prediction_manifest(tmp_path / "nonexistent.json")

    # 2. Incomplete predictions (missing a query ID) fails
    partial_queries = [
        PredictionQuery(2020, "SEC_0", "2020-01-02", 0.05, 0.01, 0.02),
    ]
    partial_path, _ = generate_and_seal_predictions(
        "MEM_SIM", 2020, partial_queries, tmp_path / "partial_preds.json"
    )
    with pytest.raises(LabelIsolationViolationError, match="missing 1 required query IDs"):
        vault.unlock_with_prediction_manifest(partial_path)

    # 3. Complete predictions covering all expected queries succeeds
    full_queries = [
        PredictionQuery(2020, "SEC_0", "2020-01-02", 0.05, 0.01, 0.02),
        PredictionQuery(2020, "SEC_0", "2020-01-03", -0.02, 0.01, 0.02),
    ]
    full_path, _ = generate_and_seal_predictions(
        "MEM_SIM", 2020, full_queries, tmp_path / "full_preds.json"
    )
    unlocked = vault.unlock_with_prediction_manifest(full_path)
    assert len(unlocked) == 2

