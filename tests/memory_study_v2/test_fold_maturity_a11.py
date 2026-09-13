"""Acceptance Test A11: 63/126-session maturity masks pass last-admitted and first-rejected boundary fixtures."""

import pytest
import pandas as pd
from memory_study_v2.folds import FoldBoundaries, partition_fold


def test_fold_maturity_boundary_cutoffs():
    """Verify exact cutoffs for 63-session training labels and 126-session bank admission."""
    boundaries = FoldBoundaries(
        evaluation_year=2020,
        train_origin_start="2013-01-01",
        training_availability_cutoff="2017-12-31",
        validation_query_start="2018-01-01",
        validation_query_end="2018-12-31",
        validation_label_cutoff="2018-12-31",
        development_query_start="2019-01-01",
        development_query_end="2019-12-31",
        development_label_cutoff="2019-12-31",
        evaluation_query_start="2020-01-01",
        evaluation_query_end="2020-12-31",
        bank_cutoff="2017-12-31",
    )

    # Synthetic session list with boundary cases
    sessions = [
        # Origin admitted to train: 63-session outcome available on 2017-12-29 (<= 2017-12-31)
        {"session": "2017-09-01", "session_origin": "2017-09-01", "session_63": "2017-12-29", "session_126": "2018-03-30", "valid_63": True, "valid_126": True},
        # Origin rejected from train: 63-session outcome available on 2018-01-02 (> 2017-12-31)
        {"session": "2017-10-01", "session_origin": "2017-10-01", "session_63": "2018-01-02", "session_126": "2018-04-05", "valid_63": True, "valid_126": True},
        # Origin admitted to bank: 126-session outcome available on 2017-12-30 (<= 2017-12-31)
        {"session": "2017-06-01", "session_origin": "2017-06-01", "session_63": "2017-09-01", "session_126": "2017-12-30", "valid_63": True, "valid_126": True},
    ]
    df_sess = pd.DataFrame(sessions)
    df_labels = pd.DataFrame(sessions)

    part = partition_fold(df_sess, df_labels, boundaries)

    # 2017-09-01 (index 0) has 63-day on 2017-12-29 <= cutoff -> admitted to train
    assert 0 in part.train_indices
    # 2017-10-01 (index 1) has 63-day on 2018-01-02 > cutoff -> REJECTED from train
    assert 1 not in part.train_indices

    # 2017-06-01 (index 2) has 126-day on 2017-12-30 <= cutoff -> admitted to bank
    assert 2 in part.bank_indices
    # 2017-09-01 (index 0) has 126-day on 2018-03-30 > cutoff -> REJECTED from bank
    assert 0 not in part.bank_indices
