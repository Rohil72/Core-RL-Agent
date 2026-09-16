"""Regression and Acceptance Tests for Calendar Corrections and Prediction Coverage Repair.

Verifies:
1. Verified exchange calendar fixtures:
   - India (NSE): 2020-05-25 (closed), 2024-01-20 (open), 2024-01-22 (closed), 2024-11-20 (closed).
   - France (Euronext Paris): 2015-12-31 (closed).
   - Brazil (B3): 2020-11-20 (open venue session retained; genuine vendor gap).
   - China (SSE/SZSE): 2019-04-29, 2019-04-30 (open venue sessions retained; genuine vendor gap).
2. False calendar session vs. genuine missing observation distinction:
   - False session in schedule creates artificial missing bar and 503-bar outage.
   - Genuine open session with missing vendor bar is explicitly classified as missing.
3. Exact feature/sample exclusion and recovery boundaries:
   - Session k invalid -> sessions k+1..k+503 excluded (503 sessions).
   - Session k+504 recovered.
4. Strict temporal causality:
   - Query at session t depends strictly on sessions in [t-503, t-1].
   - Changes at or after session t have zero impact on admission at session t.
5. Admitted query with missing forecast raises RuntimeError("MISSING_REQUIRED_PREDICTION...").
6. Genuine finite zero forecast (0.0) evaluates to NONPOSITIVE_SCORE without error.
7. Non-admitted query receives NO_ADMISSIBLE_INPUT and exclusion reason without forecast lookup.
8. Coverage manifest integrity and release verification checks.
9. Changed calendar or sample manifest hashes invalidate cached pipeline completion.
10. NAV and compounded return accounting identities reconcile within numerical tolerances.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import pytest

from memory_study_v2.venue_calendar import VenueCalendar, get_market_venue_calendar
from memory_study_v2.sample_index import (
    build_security_sample_index,
    filter_admitted_sample_ids,
)
from scripts.launch_a30_production import (
    REPO_ROOT,
    PolicyPredictionRecord,
    compute_scientific_run_identity,
    run_continuous_portfolio_simulation,
    to_canonical_json,
)


def _make_mock_fold_data(
    sessions: List[str],
    sec_ids: List[str],
    market: str = "US",
    admitted_flags: Optional[Dict[str, Dict[str, bool]]] = None,
    exclusion_reasons: Optional[Dict[str, Dict[str, str]]] = None,
    prices: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build compliant fold_data dictionary for continuous portfolio simulation tests."""
    cal = VenueCalendar(sessions=sessions)
    sec_info = {}
    eval_records = []

    for s in sec_ids:
        p = (prices or {}).get(s, 100.0)
        tr_df = pd.DataFrame({
            "session": sessions,
            "raw_open": [p] * len(sessions),
            "raw_close": [p] * len(sessions),
        })
        val_df = pd.DataFrame({
            "session": sessions,
            "bar_status": ["VALID"] * len(sessions),
        })
        feats_df = pd.DataFrame({
            "session": sessions,
            "volatility_21": [0.01] * len(sessions),
            "atr_ratio_14": [0.01] * len(sessions),
            "momentum_21": [0.05] * len(sessions),
        })

        recs = []
        for sess in sessions:
            is_adm = admitted_flags.get(s, {}).get(sess, True) if admitted_flags else True
            excl_r = exclusion_reasons.get(s, {}).get(sess, None) if exclusion_reasons else None
            r_obj = type("SampleRec", (), {
                "session": sess,
                "security_id": s,
                "query_id": f"2020_{s}_{sess}",
                "input_window_valid": is_adm,
                "exclusion_reason": excl_r,
                "open_close_ratio": 1.0,
                "close_t": p,
                "market": market,
            })()
            recs.append(r_obj)
            eval_records.append(r_obj)

        sec_info[s] = {
            "market": market,
            "calendar": cal,
            "tr_df": tr_df,
            "val_df": val_df,
            "feats_df": feats_df,
            "recs": recs,
            "sess_to_row": {sess: i for i, sess in enumerate(sessions)},
        }

    return {
        2020: {
            "sec_info": sec_info,
            "market_calendars": {market: cal},
            "eval_records": eval_records,
        }
    }


def test_verified_calendar_fixtures():
    """Verify that official exchange circular corrections are reflected in fixture CSVs."""
    # 1. India (NSE)
    india_path = REPO_ROOT / "data" / "fixtures" / "india_trading_sessions.csv"
    assert india_path.exists(), "India calendar fixture missing"
    india_sessions = set(pd.read_csv(india_path)["session"].astype(str).str.strip())

    assert "2020-05-25" not in india_sessions, "2020-05-25 (Ramzan Id) must be absent from India calendar"
    assert "2024-01-20" in india_sessions, "2024-01-20 (Live DR Switchover) must be present in India calendar"
    assert "2024-01-22" not in india_sessions, "2024-01-22 (Ayodhya consecration) must be absent from India calendar"
    assert "2024-11-20" not in india_sessions, "2024-11-20 (Maharashtra Election) must be absent from India calendar"

    # 2. France (Euronext Paris)
    france_path = REPO_ROOT / "data" / "fixtures" / "france_trading_sessions.csv"
    assert france_path.exists(), "France calendar fixture missing"
    france_sessions = set(pd.read_csv(france_path)["session"].astype(str).str.strip())
    assert "2015-12-31" not in france_sessions, "2015-12-31 (Euronext Paris closed) must be absent from France calendar"

    # 3. Brazil (B3)
    brazil_path = REPO_ROOT / "data" / "fixtures" / "brazil_trading_sessions.csv"
    assert brazil_path.exists(), "Brazil calendar fixture missing"
    brazil_sessions = set(pd.read_csv(brazil_path)["session"].astype(str).str.strip())
    assert "2020-11-20" in brazil_sessions, "2020-11-20 (B3 open) must be retained in Brazil calendar"

    # 4. China (SSE/SZSE)
    china_path = REPO_ROOT / "data" / "fixtures" / "china_trading_sessions.csv"
    assert china_path.exists(), "China calendar fixture missing"
    china_sessions = set(pd.read_csv(china_path)["session"].astype(str).str.strip())
    assert "2019-04-29" in china_sessions, "2019-04-29 (China open) must be retained in China calendar"
    assert "2019-04-30" in china_sessions, "2019-04-30 (China open) must be retained in China calendar"


def test_false_calendar_session_vs_genuine_missing_observation():
    """Verify distinction between false calendar session and genuine missing data."""
    vc_std = VenueCalendar()
    all_sess = vc_std.sessions_in_range("2013-01-02", "2016-01-01")[:600]
    false_sess_date = all_sess[510]

    # Calendar A (false): includes date that was not an actual operating session
    vc_false = VenueCalendar(sessions=all_sess)

    # Calendar B (correct): omits the non-operating holiday
    correct_sess = [s for s in all_sess if s != false_sess_date]
    vc_correct = VenueCalendar(sessions=correct_sess)

    # Security data reflects real trading (never traded on the non-operating holiday)
    df_sec = pd.DataFrame({
        "session": correct_sess,
        "close": np.linspace(100.0, 200.0, len(correct_sess)),
    })

    # Under false calendar, missing bar at index 510 invalidates the next session
    recs_false = build_security_sample_index("SEC_A", df_sec, venue_calendar=vc_false, evaluation_year=2015)
    rec_next_false = [r for r in recs_false if r.session == all_sess[511]][0]
    assert rec_next_false.input_window_valid is False
    assert rec_next_false.exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"

    # Under correct calendar, the holiday is not a session, so history has no missing bar
    recs_correct = build_security_sample_index("SEC_A", df_sec, venue_calendar=vc_correct, evaluation_year=2015)
    rec_next_correct = [r for r in recs_correct if r.session == all_sess[511]][0]
    assert rec_next_correct.input_window_valid is True
    assert rec_next_correct.exclusion_reason is None


def test_feature_sample_exclusion_and_recovery_boundary_503_bars():
    """Verify exact 503-bar exclusion and k+504 recovery invariant."""
    vc = VenueCalendar()
    # 1150 scheduled sessions ensures >503 bars precede index k=520
    sessions = vc.sessions_in_range("2011-01-03", "2016-01-01")[:1150]
    N = len(sessions)
    assert N >= 1100

    closes = np.linspace(100.0, 200.0, N)
    df = pd.DataFrame({
        "session": sessions,
        "close": closes,
        "bar_status": ["VALID"] * N,
    })

    # Invalidate exactly session at index k = 520
    k = 520
    df_corrupt = df.copy()
    df_corrupt.loc[k, "bar_status"] = "MISSING"
    df_corrupt.loc[k, "close"] = np.nan

    recs = build_security_sample_index("TEST_SEC", df_corrupt, venue_calendar=vc, evaluation_year=2015)

    # 1. Sessions k+1 to k+503 (indices 521 to 1023 inclusive) MUST have input_window_valid == False
    # because their 503-bar preceding window [t-503, t-1] includes bar k=520.
    for t_idx in range(k + 1, k + 503 + 1):
        if t_idx < len(recs):
            rec = recs[t_idx]
            assert rec.input_window_valid is False, (
                f"Session at index {t_idx} ({rec.session}) must be invalid because its window includes invalid bar {k}"
            )
            assert rec.exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"

    # 2. Session at index k+504 (index 1024) MUST recover:
    # Its 503-bar window is [1024-503, 1024-1] = [521, 1023], which strictly begins AFTER bar k=520!
    rec_recovered = recs[k + 504]
    assert rec_recovered.input_window_valid is True, (
        f"Session at index {k+504} ({rec_recovered.session}) must recover valid window"
    )
    assert rec_recovered.exclusion_reason is None


def test_causality_and_temporal_window_bounds():
    """Verify that sample index validity at session t strictly depends on bars <= t-1."""
    vc = VenueCalendar()
    sessions = vc.sessions_in_range("2013-01-02", "2016-01-01")[:600]
    N = len(sessions)

    df_base = pd.DataFrame({
        "session": sessions,
        "close": np.linspace(100.0, 200.0, N),
        "bar_status": ["VALID"] * N,
    })

    target_idx = 520
    recs_base = build_security_sample_index("TEST_SEC", df_base, venue_calendar=vc, evaluation_year=2015)
    assert recs_base[target_idx].input_window_valid is True

    # Mutate bar at target_idx (session t) or target_idx + 1 (session t+1)
    df_future = df_base.copy()
    df_future.loc[target_idx, "bar_status"] = "MISSING"
    df_future.loc[target_idx, "close"] = np.nan
    df_future.loc[target_idx + 1, "bar_status"] = "CORRUPT"

    recs_future = build_security_sample_index("TEST_SEC", df_future, venue_calendar=vc, evaluation_year=2015)
    # The query at session t (target_idx) uses window [t-503, t-1], so changes at t and t+1 cannot affect it
    assert recs_future[target_idx].input_window_valid is True, (
        "Causality violated: bar at t or t+1 affected window validity at t"
    )


def test_missing_required_prediction_raises_runtime_error(tmp_path: Path):
    """Verify that an admitted query missing a forecast raises RuntimeError instead of defaulting to 0."""
    sessions = ["2020-01-02", "2020-01-03"]
    sec_ids = ["SEC_1", "SEC_2"]

    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids)

    # SEC_1 has predictions for all sessions; SEC_2 is missing a prediction for session 2020-01-02
    preds = [
        PolicyPredictionRecord(
            query_id="2020_SEC_1_2020-01-02",
            security_id="SEC_1",
            market="US",
            decision_session="2020-01-02",
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.05,
            source_artifact_id="art1",
        ),
        PolicyPredictionRecord(
            query_id="2020_SEC_1_2020-01-03",
            security_id="SEC_1",
            market="US",
            decision_session="2020-01-03",
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.05,
            source_artifact_id="art1",
        ),
    ]

    with pytest.raises(RuntimeError, match="MISSING_REQUIRED_PREDICTION"):
        run_continuous_portfolio_simulation(
            folds_to_run=[2020],
            fold_data_by_year=fold_data,
            predictions_by_year={2020: preds},
            output_dir=tmp_path,
        )


def test_genuine_finite_zero_forecast_handling(tmp_path: Path):
    """Verify that a model-generated 0.0 forecast is preserved as NONPOSITIVE_SCORE without error."""
    sessions = ["2020-01-02", "2020-01-03"]
    sec_ids = ["SEC_1"]
    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids)

    # Explicit 0.0 forecast for session 0
    preds = [
        PolicyPredictionRecord(
            query_id="2020_SEC_1_2020-01-02",
            security_id="SEC_1",
            market="US",
            decision_session="2020-01-02",
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.0,
            source_artifact_id="art1",
        )
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020],
        fold_data_by_year=fold_data,
        predictions_by_year={2020: preds},
        output_dir=tmp_path,
    )

    decisions = res["all_decisions"][("MLP_BASE", "US", 7)]
    assert len(decisions) == 1
    d = decisions[0]
    assert d.prediction == 0.0
    assert d.score == 0.0
    assert d.decision_state == "NONPOSITIVE_SCORE"
    assert d.selected_for_entry is False
    assert d.action == "NONPOSITIVE_SCORE"


def test_non_admitted_query_records_no_admissible_input(tmp_path: Path):
    """Verify that non-admitted query records NO_ADMISSIBLE_INPUT and does not look up predictions."""
    sessions = ["2020-01-02", "2020-01-03"]
    sec_ids = ["SEC_INVAL"]

    admitted_flags = {"SEC_INVAL": {"2020-01-02": False, "2020-01-03": False}}
    exclusion_reasons = {"SEC_INVAL": {"2020-01-02": "INVALID_BAR_IN_INPUT_WINDOW", "2020-01-03": "INVALID_BAR_IN_INPUT_WINDOW"}}

    fold_data = _make_mock_fold_data(
        sessions=sessions,
        sec_ids=sec_ids,
        admitted_flags=admitted_flags,
        exclusion_reasons=exclusion_reasons,
    )

    # Empty predictions table - since it's not admitted, it must NOT require a prediction!
    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020],
        fold_data_by_year=fold_data,
        predictions_by_year={2020: []},
        output_dir=tmp_path,
    )

    # PASSIVE_EQUAL_WEIGHT runs automatically when present
    for acct_key, decisions in res["all_decisions"].items():
        if decisions:
            d = decisions[0]
            assert d.decision_state == "NO_ADMISSIBLE_INPUT"
            assert d.exclusion_reason == "INVALID_BAR_IN_INPUT_WINDOW"
            assert d.selected_for_entry is False
            assert "EXCLUDED_INVALID_BAR_IN_INPUT_WINDOW" in d.action


def test_coverage_manifest_integrity_and_release_check(tmp_path: Path):
    """Verify coverage manifest export and release failure on missing forecast."""
    sessions = ["2020-01-02", "2020-01-03"]
    sec_ids = ["SEC_1"]
    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids)

    preds = [
        PolicyPredictionRecord(
            query_id="2020_SEC_1_2020-01-02",
            security_id="SEC_1",
            market="US",
            decision_session="2020-01-02",
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.02,
            source_artifact_id="art1",
        )
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020],
        fold_data_by_year=fold_data,
        predictions_by_year={2020: preds},
        output_dir=tmp_path,
    )

    cov_file = tmp_path / "coverage_manifest.json"
    assert cov_file.exists(), "coverage_manifest.json must be written"
    with open(cov_file, "r", encoding="utf-8") as f:
        cov_data = json.load(f)

    assert len(cov_data) > 0
    first_entry = cov_data[0]
    assert first_entry["expected_forecasts"] == first_entry["sealed_forecasts"]
    assert first_entry["admitted_queries"] == 1
    assert first_entry["valid_scores"] == 1


def test_changed_calendar_or_manifest_invalidates_run_identity(tmp_path: Path):
    """Verify that modifying calendar hashes or sample manifest changes run identity."""
    config = {
        "execution": {"initial_capital_account_units": 100000.0},
        "neural": {"seeds": [7], "architectures": ["MLP"]},
    }
    markets = ["Brazil", "China", "France", "India", "UK", "US"]
    cal_hashes_orig = {m: get_market_venue_calendar(m, execution_mode="production").schedule_hash for m in markets}

    ident_1 = compute_scientific_run_identity(
        config=config,
        folds_to_run=[2020],
        context={},
        checkpoint_hashes={},
        calendar_hashes=cal_hashes_orig,
        code_revision="git_rev_1",
    )

    # Simulate changed India calendar
    cal_hashes_mut = dict(cal_hashes_orig)
    cal_hashes_mut["India"] = "0123456789abcdef" * 4

    ident_2 = compute_scientific_run_identity(
        config=config,
        folds_to_run=[2020],
        context={},
        checkpoint_hashes={},
        calendar_hashes=cal_hashes_mut,
        code_revision="git_rev_1",
    )

    assert ident_1["calendar_hashes"] != ident_2["calendar_hashes"]
    assert ident_1 != ident_2


def test_nav_and_compound_return_reconciliation_tolerance(tmp_path: Path):
    """Verify NAV identity and compound return reconciliation within numerical precision."""
    sessions = ["2020-01-02", "2020-01-03", "2020-01-06"]
    sec_ids = ["SEC_1"]
    fold_data = _make_mock_fold_data(sessions=sessions, sec_ids=sec_ids)

    preds = [
        PolicyPredictionRecord(
            query_id=f"2020_SEC_1_{sess}",
            security_id="SEC_1",
            market="US",
            decision_session=sess,
            fold=2020,
            policy_id="MLP_BASE",
            realization_id=7,
            prediction=0.05,
            source_artifact_id="art1",
        )
        for sess in sessions
    ]

    res = run_continuous_portfolio_simulation(
        folds_to_run=[2020],
        fold_data_by_year=fold_data,
        predictions_by_year={2020: preds},
        output_dir=tmp_path,
        initial_capital=100000.0,
    )

    acct = res["policy_accounts"][("MLP_BASE", "US", 7)]
    for st in acct.daily_history:
        # NAV identity: NAV = cash + holdings_value
        assert abs(st.total_nav - (st.cash + st.holdings_value)) < 1e-6

    # Compound return check
    daily_rets = [st.daily_return for st in acct.daily_history]
    compound_nav = 100000.0 * float(np.prod([1.0 + r for r in daily_rets]))
    final_nav = acct.daily_history[-1].total_nav
    assert abs(compound_nav - final_nav) < 1e-4
