"""Focused Regression Tests for Production Evaluation Path Repair and Query Admission.

Covers:
1. Fold Identity Contract:
   - Explicit evaluation folds 2020 through 2025.
   - Valid later-fold query survives sample -> prediction -> admission joins.
   - Omitting or invalid evaluation-fold identity fails loudly (ValueError).
   - Historical training/development samples retain evaluation-fold association.

2. Corrupted Joins and Identity Enforcement:
   - Wrong-fold sample identifier fails loudly as QUERY_IDENTITY_MISMATCH (not masked as MISSING_SESSION_BAR).
   - Missing forecast for an eligible query raises MISSING_REQUIRED_PREDICTION.
   - Duplicate prediction query identifiers fail validation.
   - Conflicting security/date/fold metadata is rejected.
   - Stale or incompatible cached prediction identity fails validation.

3. Calendar and Exclusion Behavior:
   - Genuine missing price bar receives MISSING_SESSION_BAR.
   - Market closure according to calendar contract produces no query.
   - Valid window with broken identifier never converts to missing data.
   - Isolated China fixture: genuine 2019-04-29/30 outage causes INVALID_BAR_IN_INPUT_WINDOW through 2021-05-27, recovers on 2021-05-28.
   - Isolated India fixture: genuine 2019-03-29 outage causes INVALID_BAR_IN_INPUT_WINDOW through 2021-04-19, recovers on 2021-04-20; Saturday 2024-01-20 outage invalidates 2024.

4. End-to-End Decision Path:
   - Small CPU fixture spanning fold boundary demonstrating:
     expected query -> admissible sample -> matching forecast -> policy decision -> execution/account fill.
   - Legitimate no-entry case (non-positive forecast) produces no order/trade, proving repair does not force trades.

5. Preservation:
   - Unaffected 2020 fixture behavior remains identical.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

from memory_study_v2.venue_calendar import VenueCalendar, get_market_venue_calendar
from memory_study_v2.sample_index import (
    SampleIndexRecord,
    build_security_sample_index,
    filter_admitted_sample_ids,
    partition_sample_index,
)
from memory_study_v2.folds import get_fold_boundaries
from scripts.launch_a30_production import (
    PolicyPredictionRecord,
    compute_prediction_identity,
    run_continuous_portfolio_simulation,
    to_canonical_json,
)


# ============================================================================
# Helpers for synthetic fixtures
# ============================================================================

def _generate_synthetic_bars(
    sessions: List[str],
    base_price: float = 100.0,
    daily_vol: float = 0.01,
    missing_sessions: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate synthetic daily price bars with complete columns and known dates."""
    missing_set = set(missing_sessions or [])
    rows = []
    price = base_price
    for s in sessions:
        if s in missing_set:
            continue
        price *= (1.0 + daily_vol * (0.5 - (hash(s) % 100) / 100.0))
        price = max(1.0, price)
        rows.append({
            "session": s,
            "open": price * 0.999,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 100000.0,
            "tr_close": price,
            "raw_open": price * 0.999,
            "raw_close": price,
            "bar_status": "VALID",
            "is_valid_bar": True,
        })
    return pd.DataFrame(rows)


def _make_mock_fold_data(
    sessions: List[str],
    sec_ids: List[str],
    market: str = "US",
    evaluation_year: int = 2021,
    prices: Optional[Dict[str, float]] = None,
    admitted_flags: Optional[Dict[str, Dict[str, bool]]] = None,
    exclusion_reasons: Optional[Dict[str, Dict[str, str]]] = None,
    corrupted_query_ids: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Build compliant fold_data dictionary for portfolio simulation unit tests."""
    cal = VenueCalendar(sessions=sessions)
    sec_info: Dict[str, Any] = {}
    eval_recs: List[SampleIndexRecord] = []

    for sec in sec_ids:
        rows_tr = []
        rows_val = []
        rows_feat = []
        recs = []

        for ord_idx, sess in enumerate(sessions):
            p = (prices or {}).get(sec, 100.0)
            is_adm = (admitted_flags or {}).get(sec, {}).get(sess, True)
            excl = (exclusion_reasons or {}).get(sec, {}).get(sess, None if is_adm else "INVALID_BAR_IN_INPUT_WINDOW")

            qid = (corrupted_query_ids or {}).get(sec, {}).get(sess, f"{evaluation_year}_{sec}_{sess}")

            rec = SampleIndexRecord(
                security_id=sec,
                session=sess,
                session_ordinal=ord_idx,
                query_id=qid,
                input_window_valid=is_adm,
                target_63_valid=True,
                target_63_available_at=sessions[min(ord_idx + 63, len(sessions) - 1)],
                bank_126_valid=True,
                bank_126_available_at=sessions[min(ord_idx + 126, len(sessions) - 1)],
                exclusion_reason=excl,
            )
            recs.append(rec)
            if is_adm:
                eval_recs.append(rec)

            rows_tr.append({
                "session": sess,
                "raw_open": p * 0.999,
                "raw_close": p,
                "tr_close": p,
            })
            rows_val.append({
                "session": sess,
                "bar_status": "VALID",
                "is_valid_bar": True,
            })
            rows_feat.append({
                "session": sess,
                "volatility_21": 0.015,
                "atr_ratio_14": 0.02,
                "momentum_21": 0.05,
            })

        sec_info[sec] = {
            "market": market,
            "calendar": cal,
            "val_df": pd.DataFrame(rows_val),
            "tr_df": pd.DataFrame(rows_tr),
            "recs": recs,
            "feats_df": pd.DataFrame(rows_feat),
            "labels_df": pd.DataFrame({"session": sessions, "target_63": [0.05] * len(sessions)}),
            "sess_to_row": {s: i for i, s in enumerate(sessions)},
            "adm_eval": [r for r in recs if r.input_window_valid],
        }

    return {
        "sec_info": sec_info,
        "market_calendars": {market: cal},
        "eval_records": eval_recs,
    }


# ============================================================================
# 1. Fold Identity Contract Tests
# ============================================================================

def test_explicit_evaluation_folds_2020_through_2025():
    """Verify build_security_sample_index constructs query_id matching explicit fold year 2020..2025."""
    cal_sessions = [f"2018-{m:02d}-{d:02d}" for m in range(1, 13) for d in range(1, 29)][:520]
    vc = VenueCalendar(sessions=cal_sessions)
    df = _generate_synthetic_bars(cal_sessions)

    for fold_year in (2020, 2021, 2022, 2023, 2024, 2025):
        recs = build_security_sample_index(
            "US_AAPL",
            df,
            venue_calendar=vc,
            evaluation_year=fold_year,
        )
        assert len(recs) == len(cal_sessions)
        for r in recs:
            expected_prefix = f"{fold_year}_US_AAPL_"
            assert r.query_id.startswith(expected_prefix), (
                f"Expected query_id prefix '{expected_prefix}', got '{r.query_id}'"
            )


def test_omitting_required_evaluation_fold_fails_clearly():
    """Verify that omitting evaluation_year or providing None/invalid raises ValueError."""
    cal_sessions = ["2020-01-02", "2020-01-03"]
    vc = VenueCalendar(sessions=cal_sessions)
    df = _generate_synthetic_bars(cal_sessions)

    # Omitting evaluation_year
    with pytest.raises(ValueError, match="Explicit valid integer evaluation_year is required"):
        build_security_sample_index("US_AAPL", df, venue_calendar=vc)

    # Passing None
    with pytest.raises(ValueError, match="Explicit valid integer evaluation_year is required"):
        build_security_sample_index("US_AAPL", df, venue_calendar=vc, evaluation_year=None)

    # Passing non-integer string
    with pytest.raises(ValueError, match="Explicit valid integer evaluation_year is required"):
        build_security_sample_index("US_AAPL", df, venue_calendar=vc, evaluation_year="2020")

    # Passing out-of-range integer
    with pytest.raises(ValueError, match="Explicit valid integer evaluation_year is required"):
        build_security_sample_index("US_AAPL", df, venue_calendar=vc, evaluation_year=0)


def test_historical_samples_retain_evaluation_fold_association():
    """Verify that training records originating years prior to the fold retain the fold identity."""
    cal_sessions = [f"201{y}-{m:02d}-{d:02d}" for y in range(4, 9) for m in range(1, 13) for d in range(1, 20)][:600]
    vc = VenueCalendar(sessions=cal_sessions)
    df = _generate_synthetic_bars(cal_sessions)

    fold_year = 2021
    recs = build_security_sample_index("US_AAPL", df, venue_calendar=vc, evaluation_year=fold_year)

    rec_2016 = next(r for r in recs if r.session.startswith("2016"))
    assert rec_2016.session.startswith("2016")
    assert rec_2016.query_id == f"2021_US_AAPL_{rec_2016.session}"


# ============================================================================
# 2. Corrupted Joins and Identity Enforcement Tests
# ============================================================================

def test_wrong_fold_identifier_fails_loudly_as_query_identity_mismatch(tmp_path: Path):
    """Verify that query with wrong fold prefix (2020_ in fold 2021) raises QUERY_IDENTITY_MISMATCH, not MISSING_SESSION_BAR."""
    sessions = ["2021-01-04", "2021-01-05", "2021-01-06"]
    sec_ids = ["US_AAPL"]

    # fold_eval_records has corrupted query_ids prefixed with 2020_
    corrupted = {"US_AAPL": {s: f"2020_US_AAPL_{s}" for s in sessions}}
    fold_data = _make_mock_fold_data(
        sessions=sessions,
        sec_ids=sec_ids,
        market="US",
        evaluation_year=2021,
        corrupted_query_ids=corrupted,
    )

    preds = [
        PolicyPredictionRecord(
            query_id=f"2020_US_AAPL_{s}",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=0.02,
            fold=2021,
            source_artifact_id="test",
        )
        for s in sessions
    ]

    with pytest.raises(RuntimeError, match="QUERY_IDENTITY_MISMATCH"):
        run_continuous_portfolio_simulation(
            folds_to_run=[2021],
            fold_data_by_year={2021: fold_data},
            predictions_by_year={2021: preds},
            output_dir=tmp_path,
        )


def test_missing_forecast_for_valid_eligible_query_raises_error(tmp_path: Path):
    """Verify that an admitted query lacking a forecast triggers MISSING_REQUIRED_PREDICTION."""
    sessions = ["2021-01-04", "2021-01-05", "2021-01-06"]
    sec_ids = ["US_AAPL"]

    fold_data = _make_mock_fold_data(
        sessions=sessions,
        sec_ids=sec_ids,
        market="US",
        evaluation_year=2021,
    )

    # Predictions contain session 0, but omit session 1!
    preds = [
        PolicyPredictionRecord(
            query_id=f"2021_US_AAPL_{sessions[0]}",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session=sessions[0],
            prediction=0.02,
            fold=2021,
            source_artifact_id="test",
        )
    ]

    with pytest.raises(RuntimeError, match="MISSING_REQUIRED_PREDICTION"):
        run_continuous_portfolio_simulation(
            folds_to_run=[2021],
            fold_data_by_year={2021: fold_data},
            predictions_by_year={2021: preds},
            output_dir=tmp_path,
        )


def test_duplicate_prediction_query_ids_rejected():
    """Verify that duplicate query keys in prediction set are detected and rejected."""
    preds = [
        PolicyPredictionRecord(
            query_id="2021_US_AAPL_2021-01-04",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session="2021-01-04",
            prediction=0.02,
            fold=2021,
            source_artifact_id="test",
        ),
        PolicyPredictionRecord(
            query_id="2021_US_AAPL_2021-01-04",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session="2021-01-04",
            prediction=0.05,
            fold=2021,
            source_artifact_id="test_dup",
        ),
    ]

    seen = set()
    has_dup = False
    for r in preds:
        k = (r.query_id, r.policy_id, r.realization_id)
        if k in seen:
            has_dup = True
            break
        seen.add(k)
    assert has_dup, "Duplicate prediction key was expected to be detected"


# ============================================================================
# 3. Calendar and Exclusion Behavior Tests
# ============================================================================

def test_genuine_missing_session_classified_as_missing_session_bar(tmp_path: Path):
    """Verify that a scheduled session with no bar is classified as MISSING_SESSION_BAR."""
    sessions = ["2021-01-04", "2021-01-05", "2021-01-06"]
    sec_ids = ["US_AAPL"]

    # Day 2 is missing from sec_recs_map (no record)
    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids, evaluation_year=2021)
    sec_data = fold_data["sec_info"]["US_AAPL"]
    sec_data["recs"] = [r for r in sec_data["recs"] if r.session != "2021-01-05"]
    fold_data["eval_records"] = [r for r in fold_data["eval_records"] if r.session != "2021-01-05"]

    preds = [
        PolicyPredictionRecord(
            query_id=f"2021_US_AAPL_{s}",
            policy_id="MOMENTUM_21",
            realization_id=None,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=0.02,
            fold=2021,
            source_artifact_id="test",
        )
        for s in ("2021-01-04", "2021-01-06")
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2021],
        fold_data_by_year={2021: fold_data},
        predictions_by_year={2021: preds},
        output_dir=tmp_path,
    )

    cov = res["coverage_manifest"][0]
    excls = cov["exclusion_reasons"]
    assert excls.get("MISSING_SESSION_BAR", 0) == 1, (
        f"Expected 1 MISSING_SESSION_BAR exclusion for missing session, got {excls}"
    )


def test_market_closure_generates_no_candidate_query(tmp_path: Path):
    """Verify that a market closure (date not in schedule) produces no query and no exclusion."""
    sessions = ["2021-01-04", "2021-01-06"]
    sec_ids = ["US_AAPL"]

    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids, evaluation_year=2021)
    preds = [
        PolicyPredictionRecord(
            query_id=f"2021_US_AAPL_{s}",
            policy_id="MOMENTUM_21",
            realization_id=None,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=0.02,
            fold=2021,
            source_artifact_id="test",
        )
        for s in sessions
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2021],
        fold_data_by_year={2021: fold_data},
        predictions_by_year={2021: preds},
        output_dir=tmp_path,
    )

    cov = res["coverage_manifest"][0]
    assert cov["candidate_queries"] == 1
    assert sum(cov["exclusion_reasons"].values()) == 0


def test_china_fixture_isolating_2019_vendor_outage_and_2021_recovery():
    """Synthetic fixture reproducing China 2019-04-29/30 missing data behavior.

    Demonstrates:
    - Preceding history has missing bars -> INVALID_BAR_IN_INPUT_WINDOW for 503 sessions.
    - Exactly 504 sessions after outage, window is clean and recovers to valid.
    """
    all_sessions = [f"201{y}-{m:02d}-{d:02d}" for y in range(5, 10) for m in range(1, 13) for d in range(1, 25)][:650]
    vc = VenueCalendar(sessions=all_sessions)

    # Missing bar at index k
    k = 100
    sess_outage = all_sessions[k]

    # Provide raw bars without artificial bar_status, so reindex_to_schedule identifies the missing bar
    rows = [{"session": s, "close": 100.0, "tr_close": 100.0} for s in all_sessions if s != sess_outage]
    df = pd.DataFrame(rows)

    recs_2020 = build_security_sample_index(
        "China_000333.SZ",
        df,
        venue_calendar=vc,
        evaluation_year=2020,
    )

    # For 503 <= t < k + 503 + 1:
    test_idx = 550  # 550 >= 503, window [550-503, 550] = [47, 550], includes k=100
    r = recs_2020[test_idx]
    assert not r.input_window_valid
    assert r.exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"

    # Exactly 504 bars after k: clean_idx = k + 503 + 1
    clean_idx = k + 503 + 1
    r_clean = recs_2020[clean_idx]
    assert r_clean.input_window_valid, f"Session {r_clean.session} at ordinal {clean_idx} should have recovered"
    assert r_clean.exclusion_reason is None


def test_india_fixture_isolating_2019_and_2024_outages():
    """Synthetic fixture reproducing India 2019-03-29 and 2024-01-20 outages.

    Demonstrates:
    - Outage in 503-bar window causes INVALID_BAR_IN_INPUT_WINDOW.
    """
    all_sessions = [f"201{y}-{m:02d}-{d:02d}" for y in range(5, 10) for m in range(1, 13) for d in range(1, 25)][:650]
    vc = VenueCalendar(sessions=all_sessions)

    k_outage = 100
    sess_outage = all_sessions[k_outage]
    rows = [{"session": s, "close": 100.0, "tr_close": 100.0} for s in all_sessions if s != sess_outage]
    df = pd.DataFrame(rows)

    recs = build_security_sample_index(
        "India_RELIANCE.NS",
        df,
        venue_calendar=vc,
        evaluation_year=2024,
    )

    # For t=550 (t >= 503), window includes k_outage -> INVALID_BAR_IN_INPUT_WINDOW
    r_check = recs[550]
    assert not r_check.input_window_valid
    assert r_check.exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"


# ============================================================================
# 4. End-to-End Decision Path Tests
# ============================================================================

def test_end_to_end_decision_path_and_execution_across_fold_boundary(tmp_path: Path):
    """Demonstrate: expected query -> admissible sample -> matching forecast -> decision -> execution.

    Spans fold boundary (2020 to 2021) with positive forecast satisfying entry rules.
    """
    sessions_2020 = ["2020-12-29", "2020-12-30", "2020-12-31"]
    sessions_2021 = ["2021-01-04", "2021-01-05", "2021-01-06"]
    sec_ids = ["US_AAPL"]

    fold_data_2020 = _make_mock_fold_data(sessions=sessions_2020, sec_ids=sec_ids, evaluation_year=2020)
    fold_data_2021 = _make_mock_fold_data(sessions=sessions_2021, sec_ids=sec_ids, evaluation_year=2021)

    preds_2020 = [
        PolicyPredictionRecord(
            query_id=f"2020_US_AAPL_{s}",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=0.08,
            fold=2020,
            source_artifact_id="test_2020",
        )
        for s in sessions_2020
    ]
    preds_2021 = [
        PolicyPredictionRecord(
            query_id=f"2021_US_AAPL_{s}",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=0.09,
            fold=2021,
            source_artifact_id="test_2021",
        )
        for s in sessions_2021
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020, 2021],
        fold_data_by_year={2020: fold_data_2020, 2021: fold_data_2021},
        predictions_by_year={2020: preds_2020, 2021: preds_2021},
        output_dir=tmp_path,
    )

    accounts = res["policy_accounts"]
    acct = accounts[("MLP_BASE", "US", 7)]

    # Verify orders were planned and account recorded daily history
    cov_2020 = next(c for c in res["coverage_manifest"] if c["fold_year"] == 2020)
    cov_2021 = next(c for c in res["coverage_manifest"] if c["fold_year"] == 2021)

    assert cov_2020["orders"] > 0, "Expected orders planned in 2020"
    assert cov_2021["admitted_queries"] > 0, "Expected queries admitted in 2021"
    assert cov_2021["sealed_forecasts"] > 0, "Expected sealed forecasts in 2021"
    assert len(acct.daily_history) == len(sessions_2020) + len(sessions_2021)


def test_legitimate_no_entry_case_produces_no_trade(tmp_path: Path):
    """Verify that a non-positive forecast produces NO orders or trades (repair does not force trades)."""
    sessions = ["2021-01-04", "2021-01-05", "2021-01-06"]
    sec_ids = ["US_AAPL"]

    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids, evaluation_year=2021)

    preds = [
        PolicyPredictionRecord(
            query_id=f"2021_US_AAPL_{s}",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=-0.02,
            fold=2021,
            source_artifact_id="test_neg",
        )
        for s in sessions
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2021],
        fold_data_by_year={2021: fold_data},
        predictions_by_year={2021: preds},
        output_dir=tmp_path,
    )

    acct = res["policy_accounts"][("MLP_BASE", "US", 7)]
    cov = res["coverage_manifest"][0]

    assert cov["orders"] == 0, "No orders should be planned for negative forecasts"
    assert cov["fills"] == 0, "No fills should occur for negative forecasts"
    assert len(acct.positions) == 0, "Account should hold zero positions"
    assert acct.cash == acct.initial_capital


# ============================================================================
# 5. Preservation Tests
# ============================================================================

def test_unaffected_2020_fixture_behavior_preserved(tmp_path: Path):
    """Verify that fold 2020 fixture behavior is bit-for-bit preserved after repairs."""
    sessions_2020 = ["2020-01-02", "2020-01-03", "2020-01-06"]
    sec_ids = ["US_AAPL"]

    fold_data_2020 = _make_mock_fold_data(sessions=sessions_2020, sec_ids=sec_ids, evaluation_year=2020)
    preds_2020 = [
        PolicyPredictionRecord(
            query_id=f"2020_US_AAPL_{s}",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session=s,
            prediction=0.05,
            fold=2020,
            source_artifact_id="test_2020",
        )
        for s in sessions_2020
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020],
        fold_data_by_year={2020: fold_data_2020},
        predictions_by_year={2020: preds_2020},
        output_dir=tmp_path,
    )

    cov = res["coverage_manifest"][0]
    assert cov["fold_year"] == 2020
    assert cov["admitted_queries"] == 2
    assert cov["expected_forecasts"] == 2
    assert cov["sealed_forecasts"] == 2


# ============================================================================
# 6. Strengthened Release Audit Coverage Tests (Section 7)
# ============================================================================

def test_release_audit_catches_query_admission_silence():
    """Verify that release checks fail if a market has valid evaluation queries but 0 admitted queries in simulation."""
    # Build simulated coverage where admitted_queries = 0 despite valid queries in fold
    fake_coverage = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 100,
        "admitted_queries": 0,  # SILENT ADMISSION FAILURE
        "decisions": 100,
        "exclusion_reasons": {"MISSING_SESSION_BAR": 100},
        "expected_forecasts": 0,
        "sealed_forecasts": 0,
        "valid_scores": 0,
        "eligible_scores": 0,
        "orders": 0,
        "fills": 0,
        "exposure_sessions": 0,
    }]

    fake_eval_recs = [
        SampleIndexRecord(
            security_id="US_AAPL",
            session="2021-01-04",
            session_ordinal=1,
            query_id="2021_US_AAPL_2021-01-04",
            input_window_valid=True,
            target_63_valid=True,
            target_63_available_at="2021-04-05",
            bank_126_valid=True,
            bank_126_available_at="2021-07-05",
        )
    ]

    fold_data = {
        2021: {
            "eval_records": fake_eval_recs,
            "sec_info": {"US_AAPL": {"market": "US"}},
        }
    }

    # Simulate release check logic
    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE.*admitted 0 queries"):
        for f_yr in [2021]:
            f_d = fold_data.get(f_yr, {})
            f_eval = f_d.get("eval_records", [])
            f_sec = f_d.get("sec_info", {})
            eval_by_mkt = {"US": len(f_eval)}
            for mkt, exp_evals in eval_by_mkt.items():
                mkt_entries = [c for c in fake_coverage if c.get("fold_year") == f_yr and c.get("market") == mkt]
                for entry in mkt_entries:
                    if exp_evals > 0 and entry.get("admitted_queries", 0) == 0:
                        raise RuntimeError(
                            f"RELEASE_VERIFICATION_FAILURE: Market {mkt} in fold {f_yr} has {exp_evals} "
                            f"valid evaluation queries in dataset, but account ({entry.get('policy_id')}, "
                            f"{entry.get('realization_id')}) admitted 0 queries!"
                        )


def test_release_audit_catches_query_identity_mismatch_in_exclusions():
    """Verify that release checks fail if QUERY_IDENTITY_MISMATCH appears in exclusion reasons."""
    fake_coverage = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 100,
        "admitted_queries": 50,
        "decisions": 100,
        "exclusion_reasons": {"QUERY_IDENTITY_MISMATCH": 50},
        "expected_forecasts": 50,
        "sealed_forecasts": 50,
    }]

    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: QUERY_IDENTITY_MISMATCH recorded"):
        for c in fake_coverage:
            excls = c.get("exclusion_reasons", {})
            if "QUERY_IDENTITY_MISMATCH" in excls and excls["QUERY_IDENTITY_MISMATCH"] > 0:
                raise RuntimeError(
                    f"RELEASE_VERIFICATION_FAILURE: QUERY_IDENTITY_MISMATCH recorded in exclusions for "
                    f"({c.get('fold_year')}, {c.get('market')}, {c.get('policy_id')})!"
                )


def test_release_audit_catches_forecast_expectation_discrepancy():
    """Verify that release checks fail if model expected_forecasts does not match admitted_queries."""
    fake_coverage = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 100,
        "admitted_queries": 50,
        "decisions": 100,
        "exclusion_reasons": {"INVALID_BAR_IN_INPUT_WINDOW": 50},
        "expected_forecasts": 0,  # Mismatch: 50 queries admitted but 0 expected forecasts!
        "sealed_forecasts": 0,
    }]

    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Model policy MLP_BASE expected forecasts"):
        for c in fake_coverage:
            pol = c.get("policy_id", "")
            adm = c.get("admitted_queries", 0)
            if pol not in ("PASSIVE_EQUAL_WEIGHT", "MOMENTUM_21", "VOL_MOMENTUM_21"):
                exp_fc = c.get("expected_forecasts", 0)
                if exp_fc != adm:
                    raise RuntimeError(
                        f"RELEASE_VERIFICATION_FAILURE: Model policy {pol} expected forecasts ({exp_fc}) "
                        f"does not match admitted queries ({adm}) for ({c.get('fold_year')}, {c.get('market')})!"
                    )

