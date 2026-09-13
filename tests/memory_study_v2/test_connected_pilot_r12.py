"""Acceptance & Regression Test for R12: Connected pre-2020 restricted pilot runner."""

import json
import pytest

from memory_study_v2.connected_pilot import run_connected_restricted_pilot


def test_connected_restricted_pilot_runs_end_to_end(tmp_path):
    report = run_connected_restricted_pilot(tmp_path / "pilot_out")
    assert report["status"] == "CONNECTED_PILOT_SUCCESS"
    expected_phases = [
        "data_canonicalization",
        "feature_engineering",
        "backbone_training",
        "memory_bank_construction",
        "prediction_sealing",
        "portfolio_execution",
        "inference_contrasts",
        "release_verification",
    ]
    assert report["phases_executed"] == expected_phases
    assert report["release_verification"]["status"] == "RELEASE_VERIFIED"
    assert report["replay_verification"]["status"] == "REPLAY_VERIFIED"

    # Verify input manifest (C1)
    input_manifest_path = tmp_path / "pilot_out" / "input_manifest.json"
    assert input_manifest_path.exists()
    with open(input_manifest_path, "r", encoding="utf-8") as f:
        input_manifest = json.load(f)
    assert input_manifest["imputation_policy"]["imputed_targets_count"] == 0
    assert input_manifest["imputation_policy"]["fallback_targets_count"] == 0
    assert input_manifest["imputation_policy"]["substitute_policy_returns_count"] == 0
    assert len(input_manifest["input_files"]) == 2
    for file_info in input_manifest["input_files"]:
        assert len(file_info["sha256"]) == 64

    # Verify chronological integrity assertions (C1)
    assertions = input_manifest["chronological_integrity_assertions"]
    assert assertions["training_targets_mature_before_validation"] is True
    assert assertions["validation_targets_mature_before_evaluation"] is True
    assert assertions["bank_records_mature_before_evaluation"] is True
    assert assertions["evaluation_origins_in_fold_2016"] is True
    assert assertions["scaler_fitted_strictly_on_training"] is True
    assert input_manifest["scaler_training_cutoff"] == "2015-03-25"
    assert input_manifest["price_adjustment_mode"] == "ADJUSTED_PRICE_CACHE_PILOT_MODE"

    # Verify output manifest (C1)
    output_manifest_path = tmp_path / "pilot_out" / "output_manifest.json"
    assert output_manifest_path.exists()
    with open(output_manifest_path, "r", encoding="utf-8") as f:
        output_manifest = json.load(f)
    assert output_manifest["status"] == "CONNECTED_PILOT_SUCCESS"
    assert output_manifest["policies_evaluated"] == ["MEM_SIM", "MLP_BASE", "TRANS_BASE"]
    assert output_manifest["policies_unrun_status"] == "NOT_RUN"
    assert output_manifest["verification"]["replay_verification_status"] == "REPLAY_VERIFIED"

    # Verify per-policy sealed predictions, trade ledgers, and NAV histories (C1)
    for pol in ["MEM_SIM", "MLP_BASE", "TRANS_BASE"]:
        pol_lower = pol.lower()
        pred_path = tmp_path / "pilot_out" / f"predictions_2016_{pol_lower}.json"
        assert pred_path.exists(), f"Sealed predictions missing for {pol}"
        with open(pred_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        queries = payload.get("queries", [])
        assert len(queries) > 0
        assert all(p["fold_year"] == 2016 for p in queries), f"All queries for {pol} must be fold 2016"
        assert all(p["session_origin"] >= "2016-01-01" for p in queries), f"All queries for {pol} must be in 2016"

        trade_path = tmp_path / "pilot_out" / f"trades_{pol}.json"
        assert trade_path.exists(), f"Trades ledger missing for {pol}"

        nav_path = tmp_path / "pilot_out" / f"daily_nav_{pol}.json"
        assert nav_path.exists(), f"Daily NAV history missing for {pol}"
        with open(nav_path, "r", encoding="utf-8") as f:
            nav_data = json.load(f)
        assert len(nav_data) > 0


def test_connected_pilot_contrasts_distinguish_evaluated_vs_unrun(tmp_path):
    """Verify primary contrasts distinguish evaluated arms from un-run comparison arms without fake noise (C1)."""
    run_connected_restricted_pilot(tmp_path / "pilot_out")
    contrasts_path = tmp_path / "pilot_out" / "release_analysis" / "primary_contrasts.json"
    assert contrasts_path.exists()
    with open(contrasts_path, "r", encoding="utf-8") as f:
        contrasts = json.load(f)

    # 8 primary contrasts
    assert len(contrasts) == 8
    contrasts_by_id = {c["contrast_id"]: c for c in contrasts}

    # P5 (MEM_SIM vs MLP_BASE) and P6 (MEM_SIM vs TRANS_BASE) were both evaluated
    assert contrasts_by_id["P5"]["status"] == "COMPLETED"
    assert contrasts_by_id["P6"]["status"] == "COMPLETED"

    # Arms not run in this restricted pilot (e.g. MEM_RANDOM, KNN_PLAIN) are marked NOT_RUN
    assert contrasts_by_id["P1"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P2"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P3"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P4"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P7"]["status"] == "NOT_RUN"
    assert contrasts_by_id["P8"]["status"] == "NOT_RUN"


def test_independent_venue_calendar_c1():
    """Finding 1: Venue calendar is independent of price file contents.

    - Session ordinals come from the VenueCalendar fixture, not from price-file intersection.
    - Deleting one security's bar does NOT shift any other session's ordinal.
    - The affected security's window/label for the deleted session is invalidated; other
      securities' windows are unaffected.
    """
    from memory_study_v2.venue_calendar import VenueCalendar

    vc = VenueCalendar()

    # 1. Calendar is independent of price file length.
    sessions_2013_2017 = vc.sessions_in_range("2013-01-01", "2017-12-31")
    assert len(sessions_2013_2017) > 1000, (
        f"Expected >1000 scheduled sessions 2013-2017, got {len(sessions_2013_2017)}"
    )

    # 2. Ordinals are stable: deleting one security's bar does NOT move any other ordinal.
    first_10 = sessions_2013_2017[:10]
    ordinals_before = [vc.get_ordinal(s) for s in first_10]

    session_to_delete = sessions_2013_2017[4]  # 5th session
    ordinals_remaining = [vc.get_ordinal(s) for s in first_10 if s != session_to_delete]
    expected_remaining = [o for s, o in zip(first_10, ordinals_before) if s != session_to_delete]
    assert ordinals_remaining == expected_remaining, (
        "Deleting one security's bar must NOT shift other sessions' ordinals."
    )

    # 3. Window invalidation: missing AAPL bar invalidates AAPL windows; MSFT unaffected.
    test_sessions = sessions_2013_2017[260:270]
    missing_idx = 3
    aapl_status = {s: ("MISSING" if i == missing_idx else "VALID") for i, s in enumerate(test_sessions)}
    msft_status = {s: "VALID" for s in test_sessions}

    aapl_valid = vc.invalidate_windows_with_missing(test_sessions, aapl_status, window_size=3)
    msft_valid = vc.invalidate_windows_with_missing(test_sessions, msft_status, window_size=3)

    assert aapl_valid[missing_idx] is False, "Session with MISSING bar must be invalid"
    assert aapl_valid[missing_idx + 1] is False, "Session +1 after MISSING must also be invalid (window=3)"
    assert aapl_valid[missing_idx + 2] is False, "Session +2 after MISSING must also be invalid (window=3)"
    assert aapl_valid[missing_idx - 1] is True, "Session before MISSING bar must be valid"

    assert all(msft_valid), "MSFT windows must all be valid when MSFT has no missing bars"


def test_bar_status_three_states_c1():
    """Finding 1: VenueCalendar.bar_status returns exactly CLOSED, VALID, or MISSING."""
    from memory_study_v2.venue_calendar import VenueCalendar

    vc = VenueCalendar()

    # CLOSED: a Saturday
    assert vc.bar_status("2015-01-03", has_valid_bar=False) == "CLOSED"
    assert vc.bar_status("2015-01-03", has_valid_bar=True) == "CLOSED"

    # CLOSED: NYSE holiday (2015-01-01 New Year's Day is Thursday - confirmed closed)
    assert vc.bar_status("2015-01-01", has_valid_bar=False) == "CLOSED"

    # VALID: regular trading day with a bar
    assert vc.bar_status("2015-01-02", has_valid_bar=True) == "VALID"

    # MISSING: regular trading day with no bar
    assert vc.bar_status("2015-01-02", has_valid_bar=False) == "MISSING"


def test_reindex_to_schedule_preserves_ordinals_c1():
    """Finding 1: reindex_to_schedule gives each scheduled session its correct ordinal,
    independent of which bars are actually present."""
    import pandas as pd
    from memory_study_v2.venue_calendar import VenueCalendar

    vc = VenueCalendar()

    sessions = vc.sessions_in_range("2015-01-02", "2015-01-09")
    # Simulate AAPL missing the 3rd session
    data_sessions = [s for i, s in enumerate(sessions) if i != 2]
    df = pd.DataFrame({
        "session": data_sessions,
        "close": [100.0 + i for i in range(len(data_sessions))],
    })

    reindexed = vc.reindex_to_schedule(df, ("2015-01-02", "2015-01-09"))

    assert len(reindexed) == len(sessions), (
        f"Expected {len(sessions)} rows after reindex, got {len(reindexed)}"
    )

    for _, row in reindexed.iterrows():
        sess = str(row["session"])
        expected_ord = vc.get_ordinal(sess)
        assert int(row["session_ordinal"]) == expected_ord, (
            f"Session {sess}: expected ordinal {expected_ord}, got {row['session_ordinal']}"
        )

    missing_sess = sessions[2]
    missing_rows = reindexed[reindexed["session"] == missing_sess]
    assert len(missing_rows) == 1
    assert missing_rows.iloc[0]["bar_status"] == "MISSING"

    present_sess = sessions[0]
    present_rows = reindexed[reindexed["session"] == present_sess]
    assert len(present_rows) == 1
    assert present_rows.iloc[0]["bar_status"] == "VALID"
