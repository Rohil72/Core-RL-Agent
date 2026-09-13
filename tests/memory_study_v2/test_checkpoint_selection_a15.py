"""Acceptance Test A15: Early stopping and earliest-best tie rule match deterministic loss-sequence fixtures."""

import pytest
from memory_study_v2.train import EarlyStoppingSelector


def test_early_stopping_and_earliest_best_tie_rule():
    """Verify earliest-best tie rule (improvement must exceed 1e-6) and patience 5."""
    selector = EarlyStoppingSelector(
        min_epochs=5,
        max_epochs=50,
        patience=5,
        min_improvement=1e-6,
    )

    # Synthetic sequence of validation losses:
    # Epoch 0: 0.100000 -> Best: epoch 0 (0.10)
    # Epoch 1: 0.080000 -> Best: epoch 1 (0.08)
    # Epoch 2: 0.070000 -> Best: epoch 2 (0.07)
    # Epoch 3: 0.0700005 -> NOT an improvement (> 0.07 - 1e-6). Best remains epoch 2.
    # Epoch 4: 0.0699998 -> Improvement of only 2e-7 (<= 1e-6). NOT a new best. Best remains epoch 2.
    # Epoch 5: 0.070100 -> Worse.
    # Epoch 6: 0.070200 -> Worse.
    # Epoch 7: 0.070300 -> 5th non-improving epoch after epoch 2 -> triggers should_stop at epoch 7!

    losses = [0.10, 0.08, 0.07, 0.0700005, 0.0699998, 0.070100, 0.070200, 0.070300]
    expected_best_epoch = [0, 1, 2, 2, 2, 2, 2, 2]
    expected_stop = [False, False, False, False, False, False, False, True]

    for ep, loss in enumerate(losses):
        selector.step(ep, loss)
        assert selector.best_epoch == expected_best_epoch[ep], f"Epoch {ep}: expected best {expected_best_epoch[ep]}, got {selector.best_epoch}"
        assert selector.should_stop == expected_stop[ep], f"Epoch {ep}: expected stop {expected_stop[ep]}, got {selector.should_stop}"

    assert selector.best_epoch == 2
    assert selector.best_loss == 0.07
