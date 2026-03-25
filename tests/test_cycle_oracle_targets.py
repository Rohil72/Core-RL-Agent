import numpy as np
import pandas as pd

from src.cycle.cycle_detector import Cycle
from src.cycle.oracle import (
    ACTION_ENTER,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_NEUTRAL,
    annotate_cycle_targets,
    decode_action_spans,
    extract_oracle_spans,
)


def test_oracle_targets_create_enter_hold_exit_labels():
    index = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.linspace(10, 18, len(index)),
            "high": np.linspace(11, 19, len(index)),
            "low": np.linspace(9, 17, len(index)),
            "close": np.linspace(10, 18, len(index)),
            "volume": np.linspace(1_000, 2_000, len(index)),
            "ticker": ["TEST"] * len(index),
        },
        index=index,
    )
    cycles = [
        Cycle(
            start_date=index[2],
            end_date=index[5],
            start_idx=2,
            end_idx=5,
            duration_days=3,
            net_return=0.4,
            peak_date=index[5],
            peak_idx=5,
        )
    ]

    annotated = annotate_cycle_targets(frame, cycles, future_horizons=(2, 3))

    assert annotated["oracle_action"].tolist() == [
        ACTION_NEUTRAL,
        ACTION_NEUTRAL,
        ACTION_ENTER,
        ACTION_HOLD,
        ACTION_HOLD,
        ACTION_EXIT,
        ACTION_NEUTRAL,
        ACTION_NEUTRAL,
    ]
    spans = extract_oracle_spans(annotated)
    assert len(spans) == 1
    assert spans[0].start_idx == 2
    assert spans[0].end_idx == 5


def test_decode_action_spans_respects_cooldown():
    index = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": np.linspace(10, 20, len(index)),
            "tech_minervini_gate": np.ones(len(index), dtype=float),
            "ticker": ["TEST"] * len(index),
        },
        index=index,
    )
    actions = [0, 1, 2, 3, 1, 2, 3, 0, 0, 0]
    spans = decode_action_spans(frame, actions, cooldown_days=30)

    assert len(spans) == 1
    assert spans[0].start_idx == 1
    assert spans[0].end_idx == 3
