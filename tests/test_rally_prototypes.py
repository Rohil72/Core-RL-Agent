import numpy as np
import pandas as pd

from src.memory.rally_prototypes import (
    RallyPrototypeConfig,
    build_rally_prototypes,
    score_rally_prototype_membership,
)


def _frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB", "CCC", "DDD"],
            "timestamp": [dates[0], dates[1], dates[0], dates[1], dates[10], dates[2]],
            "outcome_available_timestamp": [dates[5], dates[5], dates[5], dates[5], dates[11], dates[5]],
            "session_index": [0, 1, 0, 1, 10, 2],
            "future_max_return_63": [0.20, 0.18, 0.02, 0.01, 0.0, 0.0],
            "future_min_return_63": [-0.02, -0.03, -0.20, -0.18, 0.0, 0.0],
            "event_peak_offset_63": [2 / 63, 1 / 63, 0.0, 0.0, 0.0, 0.0],
            "event_drawdown_offset_63": [0.0, 0.0, 2 / 63, 1 / 63, 0.0, 0.0],
            "event_upside_before_drawdown_126": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "event_upside_hit_126": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "event_drawdown_hit_126": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "latent_0": [0.0, 0.0, 5.0, 5.0, 0.1, 0.1],
            "latent_1": [0.0, 0.0, 5.0, 5.0, 0.1, 0.1],
        }
    )


def test_rally_prototypes_keep_the_earliest_state_per_event():
    prototypes = build_rally_prototypes(_frame())

    assert len(prototypes.success) == 1
    assert len(prototypes.failure) == 1
    assert prototypes.success.iloc[0]["timestamp"] < prototypes.success.iloc[0]["outcome_available_timestamp"]
    assert prototypes.success.iloc[0]["prototype_role"] == "success"
    assert prototypes.failure.iloc[0]["prototype_role"] == "failure"
    assert prototypes.audit["success_candidate_rows"] == 2
    assert prototypes.audit["failure_candidate_rows"] == 2


def test_rally_membership_prefers_nearby_success_and_respects_maturity():
    frame = _frame()
    config = RallyPrototypeConfig(k=1, minimum_neighbors=1, retrieval_batch_size=2)
    prototypes = build_rally_prototypes(frame, config)
    query = frame.loc[frame["ticker"] == "CCC"].copy()

    signals, neighbors = score_rally_prototype_membership(
        frame.iloc[:4],
        prototypes,
        query,
        config,
        bandwidth=1.0,
    )

    assert bool(signals.iloc[0]["rally_start_eligible"])
    assert signals.iloc[0]["rally_start_probability"] > 0.5
    assert set(neighbors["prototype_role"]) == {"success", "failure"}

    delayed = prototypes.success.copy()
    delayed["outcome_available_timestamp"] = query.iloc[0]["timestamp"] + pd.Timedelta(days=1)
    delayed_set = type(prototypes)(success=delayed, failure=prototypes.failure, audit=prototypes.audit)
    delayed_signals, _ = score_rally_prototype_membership(
        frame.iloc[:4],
        delayed_set,
        query,
        config,
        bandwidth=1.0,
    )
    assert delayed_signals.iloc[0]["success_neighbor_count"] == 0
    assert not bool(delayed_signals.iloc[0]["rally_start_eligible"])


def test_rally_prototype_config_rejects_immature_horizon():
    with np.testing.assert_raises(ValueError):
        RallyPrototypeConfig(horizon_sessions=126, maturity_sessions=63)


def test_neutral_memory_and_absolute_support_reject_unknown_states():
    frame = _frame()
    config = RallyPrototypeConfig(
        include_neutral=True,
        k=1,
        minimum_neighbors=1,
        maximum_nearest_distance_ratio=1.0,
    )
    prototypes = build_rally_prototypes(frame, config)
    assert prototypes.neutral is not None
    assert len(prototypes.neutral) == 2
    assert prototypes.audit["neutral_candidate_rows"] == 2

    query = frame.loc[frame["ticker"] == "CCC"].copy()
    known, neighbors = score_rally_prototype_membership(frame, prototypes, query, config, bandwidth=1.0)
    assert bool(known.iloc[0]["rally_support_pass"])
    assert known.iloc[0]["neutral_neighbor_count"] == 1
    assert set(neighbors["prototype_role"]) == {"success", "failure", "neutral"}

    unknown = query.copy()
    unknown["ticker"] = "ZZZ"
    unknown["latent_0"] = 100.0
    unknown["latent_1"] = 100.0
    rejected, _ = score_rally_prototype_membership(frame, prototypes, unknown, config, bandwidth=1.0)
    assert not bool(rejected.iloc[0]["rally_support_pass"])
    assert not bool(rejected.iloc[0]["rally_start_eligible"])
