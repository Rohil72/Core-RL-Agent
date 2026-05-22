import numpy as np

from src.cycle.oracle import CycleSpan
from src.eval.cycle_prediction_metrics import (
    compute_action_metrics,
    compute_cycle_metrics,
    compute_future_target_metrics,
)


def test_action_metrics_include_standard_classification_summaries():
    y_true = np.array([0, 0, 1, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    metrics = compute_action_metrics(y_true, y_pred)

    assert metrics["action_accuracy"] == 4 / 6
    assert "action_balanced_accuracy" in metrics
    assert "action_weighted_f1" in metrics
    assert metrics["action_confusion_matrix"] == [
        [1, 1, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
    ]
    assert metrics["action_per_class"]["1"]["support"] == 2.0


def test_future_target_metrics_include_per_target_regression_stats():
    y_true = np.array(
        [
            [1.0, 0.0],
            [2.0, np.nan],
            [3.0, 1.0],
        ]
    )
    y_pred = np.array(
        [
            [1.5, 0.0],
            [2.5, 0.3],
            [2.5, 0.5],
        ]
    )

    metrics = compute_future_target_metrics(y_true, y_pred, ["return", "hit"])

    assert np.isclose(metrics["future_target_mae"], 0.4)
    assert metrics["future_target_valid_count"] == 5.0
    assert set(metrics["future_target_metrics"]) == {"return", "hit"}
    assert metrics["future_target_metrics"]["return"]["valid_count"] == 3.0
    assert "rmse" in metrics["future_target_metrics"]["hit"]


def test_cycle_metrics_include_f1_and_return_dispersion():
    oracle = [
        CycleSpan(
            ticker="T",
            start_idx=1,
            end_idx=5,
            start_date=np.datetime64("2024-01-01"),
            end_date=np.datetime64("2024-01-05"),
            start_price=10.0,
            end_price=12.0,
            peak_idx=5,
        )
    ]
    predicted = [
        CycleSpan(
            ticker="T",
            start_idx=2,
            end_idx=5,
            start_date=np.datetime64("2024-01-02"),
            end_date=np.datetime64("2024-01-05"),
            start_price=10.0,
            end_price=11.0,
        ),
        CycleSpan(
            ticker="T",
            start_idx=20,
            end_idx=22,
            start_date=np.datetime64("2024-02-01"),
            end_date=np.datetime64("2024-02-03"),
            start_price=10.0,
            end_price=9.0,
        ),
    ]

    metrics = compute_cycle_metrics(predicted, oracle)

    assert metrics["cycle_precision"] == 0.5
    assert metrics["cycle_recall"] == 1.0
    assert np.isclose(metrics["cycle_f1"], 2 / 3)
    assert "median_cycle_return" in metrics
    assert "cycle_return_std" in metrics
