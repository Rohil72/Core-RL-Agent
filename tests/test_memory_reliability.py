import numpy as np
import pandas as pd

from src.eval.memory_reliability import (
    BASE_RELIABILITY_FEATURES,
    ReliabilityConfig,
    build_consensus_frame,
    fit_reliability_model,
    reliability_metrics,
    risk_coverage_table,
)


def _seed_frame(seed: int, dates: int = 4) -> pd.DataFrame:
    rows = []
    for index in range(dates):
        timestamp = pd.Timestamp("2022-01-03", tz="UTC") + pd.offsets.BDay(index)
        realised = 0.04 if index % 2 == 0 else -0.02
        rows.append(
            {
                "ticker": "AAA",
                "timestamp": timestamp,
                "open": 100.0,
                "close": 101.0,
                "future_blended_alpha_63": realised,
                "future_universe_alpha_63": realised,
                "future_return_63": realised + 0.01,
                "future_min_return_63": -0.03,
                "future_max_return_63": 0.08,
                "retrieval_expected_alpha": realised + 0.001 * seed,
                "retrieval_alpha_ci_low": realised - 0.02,
                "retrieval_alpha_ci_high": realised + 0.02,
                "retrieval_alpha_std": 0.03,
                "retrieval_confidence": 0.7,
                "retrieval_agreement_score": 0.6,
                "retrieval_effective_sample_size": 20.0,
                "retrieval_entropy": 0.9,
                "retrieval_distance_weighted_confidence": 0.5,
                "retrieval_historical_diversity": 0.8,
                "retrieval_median_distance": 5.0,
                "retrieval_downside_cvar": -0.08,
                "retrieval_outcome_std": 0.1,
                "retrieval_ood_pass": True,
                "opportunity_score": realised,
            }
        )
    return pd.DataFrame(rows)


def _neighbors(seed: int) -> pd.DataFrame:
    shared = ["A", "B"]
    unique = [f"seed-{seed}"]
    return pd.DataFrame(
        {
            "query_ticker": ["AAA"] * 3,
            "query_timestamp": [pd.Timestamp("2022-01-03", tz="UTC")] * 3,
            "experience_id": shared + unique,
        }
    )


def test_consensus_tracks_cross_seed_dispersion_and_neighbor_overlap():
    consensus = build_consensus_frame(
        {7: _seed_frame(7), 17: _seed_frame(17)},
        {7: _neighbors(7), 17: _neighbors(17)},
    )

    first = consensus.iloc[0]
    assert first["seed_count"] == 2
    assert first["cross_seed_expected_alpha_std"] > 0
    assert np.isclose(first["cross_seed_neighbor_overlap"], 0.5)
    assert first["alpha_interval_covered"] == 1
    assert set(("useful_trade", "prediction_error", "positive_alpha_breadth")).issubset(consensus)


def _training_frame() -> pd.DataFrame:
    frames = []
    for seed in (7, 17, 37):
        frame = _seed_frame(seed, dates=180)
        frame["ticker"] = [f"T{index % 5}" for index in range(len(frame))]
        wave = np.sin(np.arange(len(frame)) / 8.0)
        frame["future_blended_alpha_63"] = 0.03 * wave
        frame["future_universe_alpha_63"] = frame["future_blended_alpha_63"]
        frame["retrieval_expected_alpha"] = frame["future_blended_alpha_63"] + 0.01 * np.cos(
            np.arange(len(frame)) / 7.0 + seed
        )
        frame["retrieval_alpha_ci_low"] = frame["retrieval_expected_alpha"] - 0.02
        frame["retrieval_alpha_ci_high"] = frame["retrieval_expected_alpha"] + 0.02
        frame["retrieval_confidence"] = np.clip(0.5 + 0.4 * np.abs(wave), 0.0, 1.0)
        frames.append(frame)
    consensus = build_consensus_frame({seed: frame for seed, frame in zip((7, 17, 37), frames)})
    for feature in BASE_RELIABILITY_FEATURES:
        if feature not in consensus:
            consensus[feature] = 0.0
    return consensus


def test_reliability_fit_is_chronological_and_full_coverage_is_unfiltered():
    train = _training_frame()
    train.loc[train.index[:3], "prediction_error"] = np.nan
    config = ReliabilityConfig(
        fit_fraction=0.55,
        calibration_embargo_sessions=5,
        minimum_fit_rows=50,
        minimum_calibration_rows=30,
    )

    model = fit_reliability_model(train, config)
    predictions = model.predict(train.tail(40))
    metrics = reliability_metrics(predictions, model)
    coverage = risk_coverage_table(predictions, model)

    assert model.fit_end < model.calibration_start
    assert model.calibration_thresholds[1.0] == float("-inf")
    assert predictions["reliability_probability"].between(0.0, 1.0).all()
    assert (predictions["predicted_absolute_error"] >= 0.0).all()
    assert metrics["row_count"] == 40
    assert coverage.loc[coverage["nominal_coverage"] == 1.0, "realized_coverage"].iloc[0] == 1.0
