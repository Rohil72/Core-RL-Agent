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

    # Unlocking without valid token must raise
    with pytest.raises(LabelIsolationViolationError, match="Invalid authorization token"):
        vault.unlock_for_final_scoring("UNAUTHORIZED_TOKEN")

    # Unlocking with authorized token succeeds
    unlocked = vault.unlock_for_final_scoring("EVALUATION_PREDICTIONS_SEALED")
    assert len(unlocked) == 2
