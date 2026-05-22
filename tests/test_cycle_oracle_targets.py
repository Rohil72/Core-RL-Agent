import numpy as np
import pandas as pd

from src.cycle.cycle_detector import Cycle
from src.cycle.oracle import (
    ACTION_ENTER,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_NEUTRAL,
    ORACLE_TIME_IDX_COL,
    annotate_cycle_targets,
    decode_action_spans,
    ensure_event_outcome_targets,
    extract_oracle_spans,
)
from src.data.io_utils import write_dataframe
from src.trainers.train_cycle_model import load_precomputed_frame


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


def test_decode_action_spans_allows_immediate_reentry():
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
    spans = decode_action_spans(frame, actions)

    assert len(spans) == 2
    assert spans[0].start_idx == 1
    assert spans[0].end_idx == 3
    assert spans[1].start_idx == 4
    assert spans[1].end_idx == 6


def test_extract_oracle_spans_keeps_full_cycle_on_truncated_split():
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
            duration_days=4,
            net_return=0.4,
            peak_date=index[5],
            peak_idx=5,
        )
    ]

    annotated = annotate_cycle_targets(frame, cycles, future_horizons=(2, 3))
    truncated = annotated.loc[index[3] :].copy()
    spans = extract_oracle_spans(truncated)

    assert len(spans) == 1
    assert spans[0].start_idx == 2
    assert spans[0].end_idx == 5
    assert spans[0].start_date == index[2]
    assert spans[0].end_date == index[5]


def test_decode_action_spans_uses_precomputed_time_idx():
    index = pd.date_range("2024-01-01", periods=6, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "close": np.linspace(10, 20, len(index)),
            "tech_minervini_gate": np.ones(len(index), dtype=float),
            "ticker": ["TEST"] * len(index),
            ORACLE_TIME_IDX_COL: np.arange(100, 106, dtype=np.int32),
        },
        index=index,
    )
    actions = [0, 1, 2, 3, 0, 0]

    spans = decode_action_spans(frame, actions)

    assert len(spans) == 1
    assert spans[0].start_idx == 101
    assert spans[0].end_idx == 103


def test_event_targets_are_derived_from_future_prices():
    index = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
    close = np.array([100, 105, 112, 95, 90, 120, 121, 80], dtype=float)
    frame = pd.DataFrame({"close": close}, index=index)

    out = ensure_event_outcome_targets(
        frame,
        peak_horizon=3,
        drawdown_horizon=3,
        ordering_horizon=5,
        upside_threshold=0.10,
        drawdown_threshold=-0.08,
    )

    assert np.isclose(out.loc[index[0], "event_peak_offset_63"], 2 / 3)
    assert np.isclose(out.loc[index[0], "event_drawdown_offset_63"], 1.0)
    assert out.loc[index[0], "event_upside_before_drawdown_126"] == 1.0
    assert out.loc[index[0], "event_upside_hit_126"] == 1.0
    assert out.loc[index[0], "event_drawdown_hit_126"] == 1.0


def test_load_precomputed_frame_preserves_existing_oracle_targets(tmp_path):
    index = pd.date_range("2024-01-01", periods=64, freq="B", tz="UTC")
    close = np.linspace(100, 160, len(index))
    frame = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.linspace(1_000_000, 1_200_000, len(index)),
            "ticker": ["TEST"] * len(index),
            "tech_return_1": pd.Series(close).pct_change().fillna(0.0).to_numpy(),
            "oracle_action": np.zeros(len(index), dtype=np.int8),
            "oracle_cycle_id": np.full(len(index), -1, dtype=np.int32),
            "future_return_21": np.zeros(len(index), dtype=np.float32),
        },
        index=index,
    )
    data_path = tmp_path / "TEST.parquet"
    write_dataframe(frame, data_path)

    config = {
        "data": {"precomputed_dir": str(tmp_path / "*.parquet")},
        "oracle": {
            "min_duration_days": 10,
            "max_duration_days": 63,
            "min_return": 0.10,
            "catastrophic_return": -0.10,
        },
        "features": {
            "sequence": ["tech_return_1"],
            "future_targets": ["future_return_21"],
        },
    }

    loaded = load_precomputed_frame(config)

    assert (loaded["oracle_action"] == 0).all()
    assert (loaded["oracle_cycle_id"] == -1).all()
    assert ORACLE_TIME_IDX_COL in loaded.columns
    assert "event_peak_offset_63" in loaded.columns
