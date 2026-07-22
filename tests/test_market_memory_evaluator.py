import pandas as pd
import numpy as np

from src.backtest.market_memory_backtester import PolicyConfig
from src.backtest.market_memory_evaluator import (
    momentum_score_frame,
    random_score_frame,
    run_policy_suite,
)


def _signals():
    rows = []
    for date, aaa, bbb in [
        ("2024-01-01", 10.0, 20.0),
        ("2024-01-02", 11.0, 21.0),
        ("2024-01-03", 12.0, 19.0),
        ("2024-01-04", 13.0, 22.0),
    ]:
        rows.append(
            {
                "ticker": "AAA",
                "timestamp": pd.Timestamp(date, tz="UTC"),
                "open": aaa,
                "close": aaa,
                "opportunity_score": 0.20,
                "retrieval_expected_upside": 0.15,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 1.0,
                "pred_future_max_return_63": 0.10,
                "pred_future_min_return_63": -0.02,
            }
        )
        rows.append(
            {
                "ticker": "BBB",
                "timestamp": pd.Timestamp(date, tz="UTC"),
                "open": bbb,
                "close": bbb,
                "opportunity_score": 0.10,
                "retrieval_expected_upside": 0.12,
                "retrieval_expected_downside": -0.03,
                "retrieval_confidence": 1.0,
                "pred_future_max_return_63": 0.06,
                "pred_future_min_return_63": -0.02,
            }
        )
    return pd.DataFrame(rows)


def test_policy_suite_runs_retrieval_and_baselines():
    suite = run_policy_suite(
        _signals(),
        PolicyConfig(top_k=1, min_hold_days=1, max_hold_days=2, slippage_bps=0, initial_capital=1000),
        {"baselines": "model_head,momentum,random", "momentum_window": 1, "random_seed": 7},
    )

    assert {"retrieval", "model_head", "momentum", "random"}.issubset(set(suite))
    assert all("metrics" in result for result in suite.values())


def test_random_score_frame_is_reproducible():
    left = random_score_frame(_signals(), seed=11)
    right = random_score_frame(_signals(), seed=11)

    assert left["random_score"].tolist() == right["random_score"].tolist()


def test_momentum_score_frame_uses_past_prices_per_ticker():
    frame = momentum_score_frame(_signals(), window=1)

    aaa = frame[frame["ticker"] == "AAA"].sort_values("timestamp")
    assert pd.isna(aaa.iloc[0]["momentum_score"])
    assert np.isclose(aaa.iloc[1]["momentum_score"], 0.10)
