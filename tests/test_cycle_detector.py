import numpy as np
import pandas as pd

from src.cycle.cycle_detector import detect_cycles


def _feature_frame(index, gate: float, template: float, fund: float, reports: float):
    return pd.DataFrame(
        {
            "tech_minervini_gate": np.full(len(index), gate, dtype=float),
            "tech_minervini_template_score": np.full(len(index), template, dtype=float),
            "fund_minervini_score": np.full(len(index), fund, dtype=float),
            "fund_report_available": np.full(len(index), reports, dtype=float),
        },
        index=index,
    )


def test_detect_cycles_accepts_soft_pullback_when_profit_is_strong():
    index = pd.date_range("2024-01-01", periods=11, freq="B", tz="UTC")
    prices = pd.Series(
        [100, 99, 97, 94, 98, 102, 108, 115, 121, 127, 132],
        index=index,
        dtype=float,
    )

    cycles = detect_cycles(
        prices,
        min_duration_days=10,
        max_duration_days=10,
        min_return=0.30,
        soft_pullback_limit=0.05,
        hard_pullback_limit=0.12,
        volatility_multiplier=0.0,
        min_cycle_score=0.40,
        min_quality_score=0.0,
    )

    assert len(cycles) == 1
    assert cycles[0].net_return >= 0.30
    assert cycles[0].start_idx == 0
    assert -0.12 <= cycles[0].max_drawdown_from_start <= -0.05


def test_detect_cycles_rejects_deep_reversion_even_if_return_recovers():
    index = pd.date_range("2024-01-01", periods=11, freq="B", tz="UTC")
    prices = pd.Series(
        [100, 95, 88, 81, 84, 89, 95, 103, 112, 122, 131],
        index=index,
        dtype=float,
    )

    cycles = detect_cycles(
        prices,
        min_duration_days=10,
        max_duration_days=10,
        min_return=0.30,
        soft_pullback_limit=0.05,
        hard_pullback_limit=0.12,
        volatility_multiplier=0.0,
        min_cycle_score=0.40,
        min_quality_score=0.0,
    )

    assert cycles == []


def test_detect_cycles_uses_quality_filter_when_features_are_available():
    index = pd.date_range("2024-01-01", periods=14, freq="B", tz="UTC")
    prices = pd.Series(
        [100, 102, 104, 103, 106, 110, 115, 121, 127, 133, 139, 144, 148, 151],
        index=index,
        dtype=float,
    )
    weak_features = _feature_frame(index, gate=0.0, template=2.0, fund=0.0, reports=1.0)
    strong_features = _feature_frame(
        index, gate=1.0, template=7.0, fund=1.0, reports=1.0
    )

    weak_cycles = detect_cycles(
        prices,
        min_duration_days=5,
        max_duration_days=12,
        min_return=0.30,
        feature_frame=weak_features,
        min_cycle_score=0.40,
        min_quality_score=0.50,
    )
    strong_cycles = detect_cycles(
        prices,
        min_duration_days=5,
        max_duration_days=12,
        min_return=0.30,
        feature_frame=strong_features,
        min_cycle_score=0.40,
        min_quality_score=0.50,
    )

    assert weak_cycles == []
    assert len(strong_cycles) == 1


def test_detect_cycles_prefers_best_scoring_window_over_first_threshold_cross():
    index = pd.date_range("2024-01-01", periods=16, freq="B", tz="UTC")
    prices = pd.Series(
        [
            100,
            101,
            103,
            105,
            108,
            112,
            118,
            126,
            133,
            141,
            149,
            158,
            166,
            174,
            181,
            179,
        ],
        index=index,
        dtype=float,
    )

    cycles = detect_cycles(
        prices,
        min_duration_days=5,
        max_duration_days=14,
        min_return=0.30,
        min_cycle_score=0.40,
        min_quality_score=0.0,
    )

    assert len(cycles) == 1
    assert cycles[0].end_idx > 7


def test_detect_cycles_cuts_cycle_at_post_peak_breakdown():
    index = pd.date_range("2024-01-01", periods=15, freq="B", tz="UTC")
    prices = pd.Series(
        [
            100,
            103,
            106,
            110,
            116,
            123,
            131,
            140,
            150,
            158,
            160,
            156,
            149,
            138,
            132,
        ],
        index=index,
        dtype=float,
    )

    cycles = detect_cycles(
        prices,
        min_duration_days=5,
        max_duration_days=14,
        min_return=0.30,
        soft_pullback_limit=0.05,
        hard_pullback_limit=0.12,
        volatility_multiplier=0.0,
        min_cycle_score=0.40,
        min_quality_score=0.0,
    )

    assert len(cycles) == 1
    assert cycles[0].peak_idx == 10
    assert cycles[0].end_idx == 13


def test_detect_cycles_walks_forward_without_a_cooldown_gap():
    index = pd.date_range("2024-01-01", periods=20, freq="B", tz="UTC")
    prices = pd.Series(
        [
            100,
            103,
            107,
            112,
            118,
            125,
            133,
            140,
            136,
            121,
            123,
            127,
            132,
            138,
            145,
            153,
            160,
            171,
            168,
            149,
        ],
        index=index,
        dtype=float,
    )

    cycles = detect_cycles(
        prices,
        min_duration_days=4,
        max_duration_days=9,
        min_return=0.20,
        soft_pullback_limit=0.05,
        hard_pullback_limit=0.12,
        volatility_multiplier=0.0,
        min_cycle_score=0.35,
        min_quality_score=0.0,
    )

    assert len(cycles) == 2
    assert cycles[0].start_idx == 0
    assert cycles[0].end_idx == 9
    assert cycles[1].start_idx == 10
    assert cycles[1].end_idx == 19


def test_detect_cycles_enforces_two_month_maximum_window():
    index = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    prices = pd.Series(np.linspace(100, 170, len(index)), index=index, dtype=float)

    cycles = detect_cycles(
        prices,
        min_duration_days=21,
        max_duration_days=252,
        min_return=0.20,
        volatility_multiplier=0.0,
        min_cycle_score=0.35,
        min_quality_score=0.0,
    )

    assert len(cycles) >= 1
    assert max(cycle.duration_days for cycle in cycles) <= 43


def test_detect_cycles_uses_adaptive_return_hurdle_for_volatile_names():
    index = pd.date_range("2024-01-01", periods=14, freq="B", tz="UTC")
    stable_prices = pd.Series(
        [100.0, 100.4, 100.7, 101.0, 101.3, 101.7, 102.0, 102.5, 103.0, 103.7, 104.3, 105.0, 106.0, 112.0],
        index=index,
        dtype=float,
    )
    volatile_prices = pd.Series(
        [100.0, 94.0, 104.0, 97.0, 108.0, 96.0, 111.0, 99.0, 115.0, 103.0, 118.0, 106.0, 121.0, 112.0],
        index=index,
        dtype=float,
    )

    stable_cycles = detect_cycles(
        stable_prices,
        min_duration_days=5,
        max_duration_days=14,
        min_return=0.10,
        volatility_multiplier=0.0,
        min_cycle_score=0.35,
        min_quality_score=0.0,
    )
    volatile_cycles = detect_cycles(
        volatile_prices,
        min_duration_days=5,
        max_duration_days=14,
        min_return=0.10,
        volatility_multiplier=0.0,
        min_cycle_score=0.35,
        min_quality_score=0.0,
    )

    assert len(stable_cycles) == 1
    assert volatile_cycles == []
