"""Automated Causality and Leakage Invariant Tests.

Verifies the 14 mandatory invariants:
1. Memory-field maturity
2. 252-session target isolation
3. Training-label maturity
4. Scaler isolation
5. Same-ticker exclusion
6. Temporal separation
7. Outcome availability
8. Neighbour count
9. Weight normalization
10. Decision/execution ordering
11. Fundamental release ordering
12. Cross-market cutoff
13. Corporate action handling
14. Final liquidation accounting
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.causality_audit import (
    TARGET_HORIZONS,
    generate_target_lineage_table,
    run_full_causality_audit_suite,
    verify_252_session_target_isolation,
    verify_decision_execution_ordering,
    verify_encoder_training_label_maturity,
    verify_memory_field_maturity,
    verify_same_ticker_exclusion,
    verify_scaler_isolation,
    verify_temporal_separation,
    verify_weight_normalization,
)
from src.memory.retrieval import deduplicate_close_neighbors


def test_target_lineage_table_integrity():
    lineage = generate_target_lineage_table()
    assert len(lineage) == 11
    fields = [r.field for r in lineage]
    assert "future_max_return_252" in fields
    assert "future_max_return_63" in fields
    assert "event_upside_before_drawdown_126" in fields

    # Verify 252-session target has explicit non-use in memory, reliability, scoring, exits
    r252 = next(r for r in lineage if r.field == "future_max_return_252")
    assert r252.encoder_target is True
    assert r252.stored_in_memory is False
    assert r252.used_in_reliability is False
    assert r252.used_in_scoring is False
    assert r252.used_in_exits is False
    assert r252.proven_status == "PASS"


def test_252_target_isolation_check():
    # Clean case
    res_clean = verify_252_session_target_isolation(
        memory_schema_columns=["future_max_return_63", "decision_return_63"],
        reliability_features=["retrieval_confidence", "opportunity_score"],
        scoring_formula_terms=["expected_upside", "downside"],
        exit_logic_terms=["stop_loss", "exit_score_fraction"],
    )
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Leakage case
    res_leak = verify_252_session_target_isolation(
        memory_schema_columns=["future_max_return_252"],
        reliability_features=["retrieval_confidence"],
        scoring_formula_terms=["expected_upside"],
        exit_logic_terms=["stop_loss"],
    )
    assert res_leak.status == "FAIL"
    assert res_leak.violations == 1
    assert "memory_schema_columns" in res_leak.details["found_in"]


def test_memory_field_maturity_invariant():
    query_ts = pd.Timestamp("2024-06-01", tz="UTC")
    # All mature before query_ts
    avail_clean = pd.Series(
        pd.date_range("2023-01-01", "2024-05-15", periods=10, tz="UTC")
    )
    res_clean = verify_memory_field_maturity(
        retrieved_timestamps=avail_clean,
        outcome_available_timestamps=avail_clean,
        query_timestamp=query_ts,
    )
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Violating case where an outcome matures after query_ts
    avail_violating = pd.Series([
        pd.Timestamp("2024-05-01", tz="UTC"),
        pd.Timestamp("2024-06-02", tz="UTC"),  # Leakage!
    ])
    res_violating = verify_memory_field_maturity(
        retrieved_timestamps=avail_violating,
        outcome_available_timestamps=avail_violating,
        query_timestamp=query_ts,
    )
    assert res_violating.status == "FAIL"
    assert res_violating.violations == 1


def test_same_ticker_exclusion_invariant():
    query_ticker = "AAPL"
    # Clean neighbours
    neighbours_clean = ["MSFT", "GOOGL", "AMZN", "NVDA"]
    res_clean = verify_same_ticker_exclusion(query_ticker, neighbours_clean)
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Contaminated neighbours
    neighbours_bad = ["MSFT", "AAPL", "NVDA"]
    res_bad = verify_same_ticker_exclusion(query_ticker, neighbours_bad)
    assert res_bad.status == "FAIL"
    assert res_bad.violations == 1


def test_temporal_separation_invariant():
    # 21 sessions min separation
    sessions_clean = [10, 40, 75, 120]
    tickers = ["AAPL", "AAPL", "AAPL", "AAPL"]
    res_clean = verify_temporal_separation(
        sessions_clean, min_separation=21, tickers=tickers
    )
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Too close same-ticker sessions (e.g. session 10 and 15)
    sessions_close = [10, 15, 75]
    res_close = verify_temporal_separation(
        sessions_close, min_separation=21, tickers=["AAPL", "AAPL", "AAPL"]
    )
    assert res_close.status == "FAIL"
    assert res_close.violations == 1


def test_weight_normalization_invariant():
    # Normalized weights
    w_clean = np.array([0.4, 0.35, 0.25])
    res_clean = verify_weight_normalization(w_clean)
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Non-normalized weights
    w_bad = np.array([0.5, 0.6])
    res_bad = verify_weight_normalization(w_bad)
    assert res_bad.status == "FAIL"
    assert res_bad.violations == 1

    # Negative weights
    w_neg = np.array([1.2, -0.2])
    res_neg = verify_weight_normalization(w_neg)
    assert res_neg.status == "FAIL"
    assert res_neg.violations == 1


def test_decision_execution_ordering_invariant():
    signals = pd.Series([
        pd.Timestamp("2024-01-02 16:00:00", tz="UTC"),
        pd.Timestamp("2024-01-03 16:00:00", tz="UTC"),
    ])
    fills = pd.Series([
        pd.Timestamp("2024-01-03 09:30:00", tz="UTC"),
        pd.Timestamp("2024-01-04 09:30:00", tz="UTC"),
    ])
    res_clean = verify_decision_execution_ordering(signals, fills)
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Same-bar fill (look-ahead)
    fills_bad = pd.Series([
        pd.Timestamp("2024-01-02 16:00:00", tz="UTC"),  # Same timestamp!
        pd.Timestamp("2024-01-04 09:30:00", tz="UTC"),
    ])
    res_bad = verify_decision_execution_ordering(signals, fills_bad)
    assert res_bad.status == "FAIL"
    assert res_bad.violations == 1


def test_scaler_isolation_invariant():
    train_dates = pd.date_range("2013-01-01", "2020-12-31", periods=50, tz="UTC")
    val_dates = pd.date_range("2021-01-01", "2021-12-31", periods=10, tz="UTC")

    # Scaler fitted exclusively on train
    scaler_clean = train_dates[:30]
    res_clean = verify_scaler_isolation(train_dates, scaler_clean, val_dates)
    assert res_clean.status == "PASS"
    assert res_clean.violations == 0

    # Scaler fitted with val contamination
    scaler_dirty = train_dates[:20].append(val_dates[:2])
    res_dirty = verify_scaler_isolation(train_dates, scaler_dirty, val_dates)
    assert res_dirty.status == "FAIL"
    assert res_dirty.violations > 0


def test_full_causality_audit_suite_run(tmp_path):
    report = run_full_causality_audit_suite(output_dir=tmp_path)
    assert report["all_invariants_pass"] is True
    assert len(report["target_lineage"]) == 11
    assert (tmp_path / "target_lineage_and_causality_audit.json").exists()
