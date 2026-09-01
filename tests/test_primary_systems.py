"""Tests for primary matched systems (P0–P6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.market_memory_backtester import PolicyConfig
from src.eval.policy_baselines import BaselineSuiteConfig
from src.eval.primary_systems import (
    PRIMARY_SPECS,
    build_mean_only_signals_from_p0,
    run_all_primary_systems,
    summarize_primary_systems,
)


def _mock_signals() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=60, freq="B", tz="UTC")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "timestamp": d,
                "ticker": t,
                "open": 100.0 + np.random.randn() * 2,
                "close": 101.0 + np.random.randn() * 2,
                "retrieval_expected_upside": 0.08 + np.random.randn() * 0.02,
                "retrieval_expected_downside": -0.04 + np.random.randn() * 0.01,
                "retrieval_expected_alpha": 0.04 + np.random.randn() * 0.02,
                "retrieval_confidence": 0.8,
                "retrieval_downside_cvar": -0.05,
                "opportunity_score": 0.06 + np.random.randn() * 0.02,
                "pred_utility_q50": 0.05 + np.random.randn() * 0.02,
            })
    return pd.DataFrame(rows)


def test_primary_specs_definition():
    assert len(PRIMARY_SPECS) == 7
    for pid in ("P0", "P1", "P2", "P3", "P4", "P5", "P6"):
        assert pid in PRIMARY_SPECS


def test_mean_only_signals_construction():
    signals = _mock_signals()
    p2 = build_mean_only_signals_from_p0(signals)
    assert "mean_only_score" in p2.columns
    assert p2["retrieval_upside_before_drawdown_prob"].eq(0.5).all()
    assert p2["retrieval_downside_cvar"].eq(0.0).all()


def test_run_all_primary_systems():
    signals = _mock_signals()
    policy = PolicyConfig(
        top_k=3,
        min_score=0.01,
        min_expected_upside=0.02,
        max_expected_downside=0.12,
        initial_capital=100000.0,
        slippage_bps=10.0,
    )
    baseline_cfg = BaselineSuiteConfig(momentum_window=5, random_trials=3, random_seed=7)

    results = run_all_primary_systems(
        p0_signals=signals,
        direct_model_signals=signals,
        raw_knn_signals=signals,
        policy_config=policy,
        baseline_config=baseline_cfg,
    )

    for pid in ("P0", "P1", "P2", "P3", "P4", "P5", "P6"):
        assert pid in results
        assert "metrics" in results[pid]
        assert "equity" in results[pid]
        assert "trades" in results[pid]

    summary_df = summarize_primary_systems(results)
    assert len(summary_df) == 7
    assert set(summary_df["System"]) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}
