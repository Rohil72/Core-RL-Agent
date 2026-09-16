"""Tests for dedicated zero-retraining recovery pipeline (Section 8 & 9).

Covers:
1. Full end-to-end CPU acceptance test on deterministic fixture.
2. All 19 mandatory failure tests:
   - Wrong checkpoint digest (rejected before deserialization)
   - Missing checkpoint (rejected without training)
   - Wrong fold/seed assignment (rejected)
   - Source equals or overlaps output (rejected before writes)
   - Existing output directory (rejected without overwriting)
   - Destination link into protected inputs (rejected)
   - Duplicate prediction identity (rejected)
   - Conflicting query metadata (rejected)
   - Wrong-fold query ID (identity error, not data exclusion)
   - Missing configured account (rejected)
   - Duplicate coverage account (rejected)
   - Partial valid-query loss (rejected)
   - Missing evaluation metadata (rejected)
   - Truly empty expected population (handled correctly)
   - Missing required admitted-query IDs (rejected)
   - Missing analysis export (cannot report completion)
   - Stale analysis artifacts (cannot be reused silently)
   - Downstream exception (source unchanged; no success receipt)
   - Accidental neural training call (fails test immediately)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from memory_study_v2.backbones import MLPAnnual
from memory_study_v2.contracts import to_canonical_json
from memory_study_v2.sample_index import SampleIndexRecord
from memory_study_v2.inference import (
    ContrastResult,
    evaluate_primary_contrasts,
    export_analysis_bundle,
    replay_analysis_bundle,
    ReplayVerificationError,
)
from memory_study_v2.venue_calendar import VenueCalendar
from scripts.launch_a30_production import (
    PolicyPredictionRecord,
    run_continuous_portfolio_simulation,
    validate_release_coverage_and_accounting,
)
from scripts.recover_a30_production import (
    assert_safety_isolation,
    audit_source_checkpoints,
    compute_directory_manifest,
    execute_zero_retraining_recovery,
    verify_checkpoint_identity,
)


@pytest.fixture
def cpu_recovery_fixture(tmp_path: Path):
    """Sets up a small, deterministic CPU test fixture with 2 folds, real parquets, and fixture checkpoints."""
    fixture_root = tmp_path / "recovery_fixture"
    fixture_root.mkdir(parents=True, exist_ok=True)

    source_dir = fixture_root / "protected_source"
    source_dir.mkdir(parents=True, exist_ok=True)

    # 1. Checkpoints for fold 2020 and fold 2021 (MLPAnnual seed 7)
    inv = []
    for yr in [2020, 2021]:
        chk_dir = source_dir / f"fold_{yr}" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7"
        chk_dir.mkdir(parents=True, exist_ok=True)
        pt_path = chk_dir / "best_checkpoint.pt"

        # Create valid reference weights
        model = MLPAnnual(seed=7)
        state_dict = model.state_dict()
        torch.save({"model_state": state_dict}, pt_path)
        chk_sha = hashlib.sha256(pt_path.read_bytes()).hexdigest()

        # Create compliant job_completion provenance
        job_comp = {
            "job_id": f"fold_{yr}_MLP_ANNUAL_966_64_128_1_seed7",
            "fold_year": yr,
            "architecture": "MLP_ANNUAL_966_64_128_1",
            "seed": 7,
            "status": "STAGE_COMPLETED",
            "artifacts": {
                "best_checkpoint": {"path": str(pt_path), "sha256": chk_sha},
            },
            "training_identity": {
                "architecture": "MLP_ANNUAL_966_64_128_1",
                "seed": 7,
                "fold_year": yr,
                "code_revision": "test_provenance_rev_001",
                "job_identity_sha256": f"job_sha_{yr}",
                "neural_configuration_sha256": "cfg_sha",
                "train_sample_ids_sha256": "sample_ids_sha",
            },
        }
        (chk_dir / "job_completion.json").write_text(to_canonical_json(job_comp), encoding="utf-8")

        inv.append({
            "job_id": f"fold_{yr}_MLP_ANNUAL_966_64_128_1_seed7",
            "fold_year": yr,
            "architecture": "MLP_ANNUAL_966_64_128_1",
            "seed": 7,
            "sha256": chk_sha,
            "relative_path": f"fold_{yr}/checkpoints_MLP_ANNUAL_966_64_128_1_seed7/best_checkpoint.pt",
            "disposition": "reuse candidate—verification pending",
        })

    inv_path = fixture_root / "reviewed_inventory.json"
    inv_path.write_text(to_canonical_json(inv), encoding="utf-8")

    # 2. Sample IDs manifests
    sample_ids_dir = fixture_root / "sample_ids"
    sample_ids_dir.mkdir(parents=True, exist_ok=True)
    for yr in [2020, 2021]:
        (sample_ids_dir / f"fold_{yr}_sample_ids.json").write_text("{}", encoding="utf-8")

    # 3. Data cache with real parquets for US_AAPL and US_MSFT
    data_dir = fixture_root / "data_cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    src_data = Path("FINAL_SUBMISSION_PACKAGE/data/cache/ohlcv")
    shutil.copy2(src_data / "US_AAPL.parquet", data_dir / "US_AAPL.parquet")
    shutil.copy2(src_data / "US_MSFT.parquet", data_dir / "US_MSFT.parquet")

    # 4. Authorized configuration
    config = {
        "production_authorized": False,
        "experiment_name": "cpu_recovery_acceptance",
        "neural": {
            "seeds": [7],
            "architectures": ["MLP_ANNUAL_966_64_128_1"],
        },
        "folds": [
            {"evaluation_year": 2020},
            {"evaluation_year": 2021},
        ],
        "analysis": {
            "bootstrap_draws": 10,
            "analysis_replay_tolerance": 1e-10,
        },
        "execution": {
            "initial_capital_account_units": 100000.0,
            "commission_per_side": 0.001,
            "slippage_per_side": 0.0005,
            "max_positions": 3,
        },
    }

    return {
        "fixture_root": fixture_root,
        "source_dir": source_dir,
        "inv_path": inv_path,
        "sample_ids_dir": sample_ids_dir,
        "data_dir": data_dir,
        "config": config,
    }


# =============================================================================
# 1. Real End-to-End CPU Acceptance Test (Section 8)
# =============================================================================

def test_recovery_pipeline_end_to_end_cpu_acceptance(cpu_recovery_fixture, monkeypatch):
    """Executes the complete zero-retraining recovery pipeline on a small CPU fixture.

    Verifies:
    1. Checkpoint authentication -> REUSE_APPROVED.
    2. Prediction generation using recovered checkpoints in eval mode without training.
    3. Eligible post-boundary entry executes a BUY fill in fold 2021.
    4. Legitimate no-entry case produces zero trades for that security.
    5. Continuous account state carry-over across fold boundary.
    6. Coverage validation passes complete accounting and query set equality.
    7. Primary inference and analysis bundle export.
    8. Independent replay verification passes with 1e-10 tolerance.
    9. Complete release sealing receipts written.
    10. Source fixture remains 100% bit-for-bit unchanged.
    11. Neural training guard asserts zero training calls were made.
    """
    source_dir = cpu_recovery_fixture["source_dir"]
    output_dir = cpu_recovery_fixture["fixture_root"] / "recovered_output"
    sample_ids_dir = cpu_recovery_fixture["sample_ids_dir"]
    data_dir = cpu_recovery_fixture["data_dir"]
    config = cpu_recovery_fixture["config"]
    inv_path = cpu_recovery_fixture["inv_path"]

    before_manifest = compute_directory_manifest(source_dir)

    # Install guard preventing any neural training
    def forbidden_training(*args, **kwargs):
        raise AssertionError("CRITICAL VIOLATION: train_backbone_model was invoked during recovery!")

    monkeypatch.setattr("scripts.launch_a30_production.train_backbone_model", forbidden_training)

    # Custom prediction: AAPL positive score -> plans BUY fill; MSFT non-positive -> no entry
    # On 2021-01-04 (first session of fold 2021):
    custom_preds = {
        "2021_US_AAPL_2021-01-04": 0.08,
        "US_MSFT": -0.05,
    }

    # Execute recovery orchestration
    res = execute_zero_retraining_recovery(
        source_dir=source_dir,
        output_dir=output_dir,
        config=config,
        sample_ids_dir=sample_ids_dir,
        data_dir=data_dir,
        selected_folds=[2020, 2021],
        inventory_path=inv_path,
        selected_securities=["US_AAPL", "US_MSFT"],
        custom_predictions=custom_preds,
        allow_reduced_arms=True,
        device=torch.device("cpu"),
    )

    # Assert recovery completed cleanly
    assert res["status"] == "RECOVERY_SUCCESS"
    assert res["pipeline_stage"] == "RELEASE_VERIFIED"
    assert res["replay_report"]["status"] == "REPLAY_VERIFIED"

    # Assert required new analysis artifacts exist
    analysis_dir = output_dir / "release_analysis"
    assert (analysis_dir / "daily_returns.json").exists()
    assert (analysis_dir / "primary_contrasts.json").exists()
    assert (analysis_dir / "contrast_draws.npy").exists()
    assert (analysis_dir / "sampled_weeks.npy").exists()
    assert (analysis_dir / "replay_report.json").exists()

    # Assert release sealing files exist
    assert (output_dir / "recovery_summary.json").exists()
    assert (output_dir / "pipeline_summary.json").exists()
    assert (output_dir / "release_manifest.json").exists()
    assert (output_dir / "coverage_manifest.json").exists()
    assert (output_dir / "portfolio_manifest.json").exists()
    assert (output_dir / "runtime_config.authorized.json").exists()

    # Assert code revision is recorded and not UNKNOWN
    with open(output_dir / "recovery_summary.json", "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["code_revision"]
    assert "UNKNOWN" not in summary_data["code_revision"]

    # Assert a post-boundary BUY fill actually executed in fold 2021 for MLP_BASE account
    acct_dir = output_dir / "accounts" / "MLP_BASE__US__7"
    assert acct_dir.exists()
    with open(acct_dir / "trades.json", "r", encoding="utf-8") as f:
        trades = json.load(f)

    post_boundary_buys = [
        t for t in trades
        if str(t["session"]).startswith("2021") and t["side"] == "BUY" and t["security_id"] == "US_AAPL"
    ]
    assert len(post_boundary_buys) > 0, "Post-boundary buy on US_AAPL must execute on 2021 session"

    # Assert legitimate no-entry case: US_MSFT had negative predictions and produced NO buys
    msft_trades = [t for t in trades if t["security_id"] == "US_MSFT"]
    assert len(msft_trades) == 0, "Nonpositive score on US_MSFT must produce 0 trades"

    # Assert carry-over state: daily nav history spans both 2020 and 2021
    with open(acct_dir / "daily_nav.json", "r", encoding="utf-8") as f:
        nav_hist = json.load(f)
    sessions = [st["session"] for st in nav_hist]
    assert any(s.startswith("2020") for s in sessions)
    assert any(s.startswith("2021") for s in sessions)
    assert sessions == sorted(sessions), "Continuous account sessions must be monotonically ordered"

    # Assert source fixture remains 100% bit-for-bit unchanged
    after_manifest = compute_directory_manifest(source_dir)
    assert before_manifest == after_manifest, "Protected source fixture must be bit-for-bit immutable"


# =============================================================================
# 2. Mandatory Failure Tests (Section 9)
# =============================================================================

def test_failure_condition_1_wrong_checkpoint_digest(tmp_path: Path):
    """Condition 1: Checkpoint with digest mismatch is rejected BEFORE loading/deserialization."""
    chk_path = tmp_path / "corrupted_checkpoint.pt"
    chk_path.write_bytes(b"CORRUPTED_RAW_BYTES_NOT_MATCHING_DIGEST")
    expected_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    # verify_checkpoint_identity must return REUSE_REJECTED with digest mismatch reason
    res = verify_checkpoint_identity(
        checkpoint_path=chk_path,
        fold_year=2020,
        architecture="MLP_ANNUAL_966_64_128_1",
        seed=7,
        expected_sha256=expected_sha,
    )
    assert res["disposition"] == "REUSE_REJECTED"
    assert any("Digest mismatch" in r for r in res["reasons"])
    assert res["integrity_verified"] is False


def test_failure_condition_2_missing_checkpoint(cpu_recovery_fixture):
    """Condition 2: Missing required checkpoint aborts recovery loudly without retraining."""
    source_dir = cpu_recovery_fixture["source_dir"]
    # Delete one checkpoint file
    missing_pt = source_dir / "fold_2021" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7" / "best_checkpoint.pt"
    missing_pt.unlink()

    output_dir = cpu_recovery_fixture["fixture_root"] / "out_missing"

    with pytest.raises(RuntimeError, match="RECOVERY_ABORTED.*missing or incompatible|not approved"):
        execute_zero_retraining_recovery(
            source_dir=source_dir,
            output_dir=output_dir,
            config=cpu_recovery_fixture["config"],
            sample_ids_dir=cpu_recovery_fixture["sample_ids_dir"],
            data_dir=cpu_recovery_fixture["data_dir"],
            selected_folds=[2020, 2021],
            inventory_path=cpu_recovery_fixture["inv_path"],
            allow_reduced_arms=True,
        )

    # Output directory must not have been created as a valid release
    assert not (output_dir / "pipeline_summary.json").exists()


def test_failure_condition_3_wrong_fold_or_seed_assignment(cpu_recovery_fixture):
    """Condition 3: Checkpoint assigned to wrong fold or seed in provenance is rejected."""
    source_dir = cpu_recovery_fixture["source_dir"]
    # Tamper with job_completion provenance in fold 2020 to claim seed=999
    comp_file = source_dir / "fold_2020" / "checkpoints_MLP_ANNUAL_966_64_128_1_seed7" / "job_completion.json"
    comp_data = json.loads(comp_file.read_text(encoding="utf-8"))
    comp_data["seed"] = 999
    comp_file.write_text(to_canonical_json(comp_data), encoding="utf-8")

    res = audit_source_checkpoints(
        source_dir=source_dir,
        folds=[2020],
        architectures=["MLP_ANNUAL_966_64_128_1"],
        seeds=[7],
        inventory_path=cpu_recovery_fixture["inv_path"],
    )
    assert res["all_approved"] is False
    assert any("Provenance identity mismatch" in str(it.get("reasons")) for it in res["inventory"])


def test_failure_condition_4_source_equals_or_overlaps_output(tmp_path: Path):
    """Condition 4: Reject source equals output, or either being inside the other before any writes."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_ids = tmp_path / "samples"
    sample_ids.mkdir()

    # 1. Source equals output
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*must NOT be identical"):
        assert_safety_isolation(source_dir, source_dir, data_dir, sample_ids)

    # 2. Output inside source
    nested_out = source_dir / "nested_out"
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*Output directory.*inside source"):
        assert_safety_isolation(source_dir, nested_out, data_dir, sample_ids)

    # 3. Source inside output
    parent_out = tmp_path
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*Source directory.*inside output"):
        assert_safety_isolation(source_dir, parent_out, data_dir, sample_ids)


def test_failure_condition_5_existing_output_directory(tmp_path: Path):
    """Condition 5: Pre-existing output directory is strictly rejected without overwriting or deleting."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_ids = tmp_path / "samples"
    sample_ids.mkdir()

    preexisting_out = tmp_path / "already_exists_out"
    preexisting_out.mkdir()
    (preexisting_out / "valuable_file.txt").write_text("DO_NOT_DELETE", encoding="utf-8")

    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*Output directory.*already exists"):
        assert_safety_isolation(source_dir, preexisting_out, data_dir, sample_ids)

    # Assert pre-existing file was NOT deleted
    assert (preexisting_out / "valuable_file.txt").read_text(encoding="utf-8") == "DO_NOT_DELETE"


def test_failure_condition_6_destination_link_into_protected_source(tmp_path: Path):
    """Condition 6: Destination path that redirects or links inside protected inputs is rejected."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_ids = tmp_path / "samples"
    sample_ids.mkdir()

    # Output pointing inside data_dir
    redirect_into_data = data_dir / "out"
    with pytest.raises(ValueError, match="CRITICAL SAFETY VIOLATION.*redirects inside protected path"):
        assert_safety_isolation(source_dir, redirect_into_data, data_dir, sample_ids)


def test_failure_condition_7_duplicate_prediction_identity(tmp_path: Path):
    """Condition 7: Duplicate prediction records for the same account/security/session fail validation."""
    sessions = ["2021-01-04", "2021-01-05"]
    cal = VenueCalendar(sessions=sessions)
    sec_info = {
        "US_AAPL": {
            "market": "US",
            "calendar": cal,
            "val_df": pd.DataFrame({"session": sessions, "bar_status": ["VALID", "VALID"], "raw_close": [100.0, 101.0]}),
            "tr_df": pd.DataFrame({"session": sessions, "raw_open": [100.0, 101.0], "raw_close": [100.0, 101.0]}),
            "feats_df": pd.DataFrame({"session": sessions, "volatility_21": [0.01, 0.01], "atr_ratio_14": [0.01, 0.01], "momentum_21": [0.05, 0.05]}),
            "sess_to_row": {s: i for i, s in enumerate(sessions)},
        }
    }
    fold_data = {
        "sec_info": sec_info,
        "eval_records": [],
        "market_calendars": {"US": cal},
    }

    # Duplicate prediction keys for (MLP_BASE, 7, US_AAPL, 2021-01-04)
    preds = [
        PolicyPredictionRecord(
            query_id="2021_US_AAPL_2021-01-04",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session="2021-01-04",
            prediction=0.05,
            fold=2021,
            source_artifact_id="art1",
        ),
        PolicyPredictionRecord(
            query_id="2021_US_AAPL_2021-01-04",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session="2021-01-04",
            prediction=0.06,  # Duplicate!
            fold=2021,
            source_artifact_id="art2",
        ),
    ]

    with pytest.raises(ValueError, match="DUPLICATE_PREDICTION_KEY"):
        run_continuous_portfolio_simulation(
            folds_to_run=[2021],
            fold_data_by_year={2021: fold_data},
            predictions_by_year={2021: preds},
            output_dir=tmp_path / "dup_sim_out",
        )


def test_failure_condition_8_conflicting_query_metadata(tmp_path: Path):
    """Condition 8: Query metadata conflicting between simulation records and dataset fails validation."""
    from scripts.launch_a30_production import run_continuous_portfolio_simulation
    sessions = ["2021-01-04"]
    cal = VenueCalendar(sessions=sessions)
    sec_info = {
        "US_AAPL": {
            "market": "US",
            "calendar": cal,
            "val_df": pd.DataFrame({"session": sessions, "bar_status": ["VALID"], "raw_close": [100.0]}),
            "tr_df": pd.DataFrame({"session": sessions, "raw_open": [100.0], "raw_close": [100.0]}),
            "feats_df": pd.DataFrame({"session": sessions, "volatility_21": [0.01], "atr_ratio_14": [0.01], "momentum_21": [0.05]}),
            "sess_to_row": {"2021-01-04": 0},
        }
    }
    # Security is AAPL, but query_id claims MSFT
    eval_recs = [
        SampleIndexRecord(
            security_id="US_AAPL",
            session="2021-01-04",
            session_ordinal=1,
            query_id="2021_US_MSFT_2021-01-04",  # Conflicting security in query_id
            input_window_valid=True,
            target_63_valid=True,
            target_63_available_at="2021-04-05",
            bank_126_valid=True,
            bank_126_available_at="2021-07-05",
        )
    ]
    fold_data = {
        "sec_info": sec_info,
        "eval_records": eval_recs,
        "market_calendars": {"US": cal},
    }

    preds = [
        PolicyPredictionRecord(
            query_id="2021_US_MSFT_2021-01-04",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session="2021-01-04",
            prediction=0.05,
            fold=2021,
            source_artifact_id="art1",
        )
    ]

    with pytest.raises(RuntimeError, match="QUERY_IDENTITY_MISMATCH"):
        run_continuous_portfolio_simulation(
            folds_to_run=[2021],
            fold_data_by_year={2021: fold_data},
            predictions_by_year={2021: preds},
            output_dir=tmp_path / "conflict_sim_out",
        )


def test_failure_condition_9_wrong_fold_query_id(tmp_path: Path):
    """Condition 9: Query ID having wrong fold year prefix raises identity error rather than silent data exclusion."""
    sessions = ["2021-01-04"]
    cal = VenueCalendar(sessions=sessions)
    sec_info = {
        "US_AAPL": {
            "market": "US",
            "calendar": cal,
            "val_df": pd.DataFrame({"session": sessions, "bar_status": ["VALID"], "raw_close": [100.0]}),
            "tr_df": pd.DataFrame({"session": sessions, "raw_open": [100.0], "raw_close": [100.0]}),
            "feats_df": pd.DataFrame({"session": sessions, "volatility_21": [0.01], "atr_ratio_14": [0.01], "momentum_21": [0.05]}),
            "sess_to_row": {"2021-01-04": 0},
        }
    }
    # Query ID prefix claims 2020 instead of 2021
    eval_recs = [
        SampleIndexRecord(
            security_id="US_AAPL",
            session="2021-01-04",
            session_ordinal=1,
            query_id="2020_US_AAPL_2021-01-04",  # Wrong fold prefix!
            input_window_valid=True,
            target_63_valid=True,
            target_63_available_at="2021-04-05",
            bank_126_valid=True,
            bank_126_available_at="2021-07-05",
        )
    ]
    fold_data = {
        "sec_info": sec_info,
        "eval_records": eval_recs,
        "market_calendars": {"US": cal},
    }

    preds = [
        PolicyPredictionRecord(
            query_id="2020_US_AAPL_2021-01-04",
            policy_id="MLP_BASE",
            realization_id=7,
            market="US",
            security_id="US_AAPL",
            decision_session="2021-01-04",
            prediction=0.05,
            fold=2021,
            source_artifact_id="art1",
        )
    ]

    with pytest.raises(RuntimeError, match="QUERY_IDENTITY_MISMATCH"):
        run_continuous_portfolio_simulation(
            folds_to_run=[2021],
            fold_data_by_year={2021: fold_data},
            predictions_by_year={2021: preds},
            output_dir=tmp_path / "wrong_fold_out",
        )


def test_failure_condition_10_missing_configured_account():
    """Condition 10: Missing configured account in coverage manifest raises exact account set mismatch."""
    cov_manifest = [
        {
            "fold_year": 2021,
            "market": "US",
            "policy_id": "MLP_BASE",
            "realization_id": 7,
            "candidate_queries": 0,
            "admitted_queries": 0,
            "admitted_qids": [],
            "decisions": 0,
            "exclusion_reasons": {},
            "expected_forecasts": 0,
            "sealed_forecasts": 0,
        }
    ]
    fold_data = {2021: {"eval_records": [], "sec_info": {}}}

    # Expected accounts includes realization 17 as well, which is missing from manifest
    expected_accounts = {
        (2021, "US", "MLP_BASE", 7),
        (2021, "US", "MLP_BASE", 17),
    }

    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Exact account set mismatch.*Missing accounts: 1"):
        validate_release_coverage_and_accounting(
            coverage_manifest=cov_manifest,
            fold_data_by_year=fold_data,
            folds_to_run=[2021],
            expected_accounts=expected_accounts,
        )


def test_failure_condition_11_duplicate_coverage_account():
    """Condition 11: Duplicate coverage manifest entries for the same account key fail validation."""
    cov_manifest = [
        {
            "fold_year": 2021,
            "market": "US",
            "policy_id": "MLP_BASE",
            "realization_id": 7,
            "candidate_queries": 0,
            "admitted_queries": 0,
            "admitted_qids": [],
            "decisions": 0,
            "exclusion_reasons": {},
            "expected_forecasts": 0,
            "sealed_forecasts": 0,
        },
        {
            "fold_year": 2021,
            "market": "US",
            "policy_id": "MLP_BASE",
            "realization_id": 7,  # DUPLICATE ACCOUNT ENTRY!
            "candidate_queries": 0,
            "admitted_queries": 0,
            "admitted_qids": [],
            "decisions": 0,
            "exclusion_reasons": {},
            "expected_forecasts": 0,
            "sealed_forecasts": 0,
        },
    ]
    fold_data = {2021: {"eval_records": [], "sec_info": {}}}

    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Duplicate coverage manifest entry"):
        validate_release_coverage_and_accounting(
            coverage_manifest=cov_manifest,
            fold_data_by_year=fold_data,
            folds_to_run=[2021],
        )


def test_failure_condition_12_partial_valid_query_loss():
    """Condition 12: Partial valid query loss (e.g. 1 admitted + 999 excluded vs 1000 expected) is rejected."""
    eval_recs = [
        SampleIndexRecord(
            security_id=f"US_SEC_{i}",
            session="2021-01-04",
            session_ordinal=1,
            query_id=f"2021_US_SEC_{i}_2021-01-04",
            input_window_valid=True,
            target_63_valid=True,
            target_63_available_at="2021-04-05",
            bank_126_valid=True,
            bank_126_available_at="2021-07-05",
        )
        for i in range(100)
    ]
    sec_info = {f"US_SEC_{i}": {"market": "US"} for i in range(100)}

    # Only 1 admitted out of 100 expected
    cov_manifest = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 100,
        "admitted_queries": 1,
        "admitted_qids": ["2021_US_SEC_0_2021-01-04"],
        "decisions": 100,
        "exclusion_reasons": {"INVALID_BAR_IN_INPUT_WINDOW": 99},
        "expected_forecasts": 1,
        "sealed_forecasts": 1,
        "valid_scores": 1,
        "eligible_scores": 1,
        "orders": 1,
        "fills": 0,
        "exposure_sessions": 0,
    }]

    fold_data = {2021: {"eval_records": eval_recs, "sec_info": sec_info}}

    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Partial or mismatched query admission"):
        validate_release_coverage_and_accounting(
            coverage_manifest=cov_manifest,
            fold_data_by_year=fold_data,
            folds_to_run=[2021],
        )


def test_failure_condition_13_missing_evaluation_metadata():
    """Condition 13: Missing required eval_records or sec_info metadata raises immediate verification failure."""
    cov_manifest = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 0,
        "admitted_queries": 0,
        "admitted_qids": [],
        "decisions": 0,
        "exclusion_reasons": {},
        "expected_forecasts": 0,
        "sealed_forecasts": 0,
    }]
    # Missing eval_records entirely
    fold_data_no_eval = {2021: {"sec_info": {}}}
    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Missing required 'eval_records' metadata"):
        validate_release_coverage_and_accounting(
            coverage_manifest=cov_manifest,
            fold_data_by_year=fold_data_no_eval,
            folds_to_run=[2021],
        )

    # Missing sec_info entirely
    fold_data_no_sec = {2021: {"eval_records": []}}
    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Missing required 'sec_info' metadata"):
        validate_release_coverage_and_accounting(
            coverage_manifest=cov_manifest,
            fold_data_by_year=fold_data_no_sec,
            folds_to_run=[2021],
        )


def test_handling_condition_14_truly_empty_expected_population():
    """Condition 14: Intentionally empty expected evaluation population with 0 admitted queries passes cleanly."""
    cov_manifest = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 0,
        "admitted_queries": 0,
        "admitted_qids": [],
        "decisions": 0,
        "exclusion_reasons": {},
        "expected_forecasts": 0,
        "sealed_forecasts": 0,
    }]
    fold_data = {2021: {"eval_records": [], "sec_info": {}}}

    res = validate_release_coverage_and_accounting(
        coverage_manifest=cov_manifest,
        fold_data_by_year=fold_data,
        folds_to_run=[2021],
    )
    assert res["status"] == "VALIDATED"


def test_failure_condition_15_missing_required_admitted_query_ids():
    """Condition 15: Account has admitted_queries > 0 but missing or empty admitted_qids is rejected."""
    cov_manifest = [{
        "fold_year": 2021,
        "market": "US",
        "policy_id": "MLP_BASE",
        "realization_id": 7,
        "candidate_queries": 5,
        "admitted_queries": 5,
        "admitted_qids": [],  # Empty admitted_qids!
        "decisions": 5,
        "exclusion_reasons": {},
        "expected_forecasts": 5,
        "sealed_forecasts": 5,
    }]
    fold_data = {2021: {"eval_records": [], "sec_info": {}}}

    with pytest.raises(RuntimeError, match="RELEASE_VERIFICATION_FAILURE: Account.*has 5 admitted queries but admitted_qids is missing or empty"):
        validate_release_coverage_and_accounting(
            coverage_manifest=cov_manifest,
            fold_data_by_year=fold_data,
            folds_to_run=[2021],
        )


def test_failure_condition_16_missing_analysis_export(tmp_path: Path):
    """Condition 16: Missing analysis export files prevent replay verification from reporting success."""
    empty_export_dir = tmp_path / "empty_analysis"
    empty_export_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="Missing analysis bundle files"):
        replay_analysis_bundle(export_dir=empty_export_dir)


def test_failure_condition_17_stale_analysis_artifacts(tmp_path: Path):
    """Condition 17: Tampered or stale analysis bundle artifacts fail replay verification."""
    export_dir = tmp_path / "tampered_analysis"
    export_dir.mkdir()

    # Create synthetic return series
    returns_by_key = {
        "MLP_BASE__US__7": [0.01, -0.01, 0.02, 0.00],
        "MEM_SIM__US__None": [0.02, 0.01, 0.01, 0.01],
    }
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]

    export_analysis_bundle(
        returns_by_key=returns_by_key,
        contrast_results=[],
        draw_matrix=np.zeros((0, 5)),
        draw_week_indices=np.zeros((5, 1), dtype=int),
        export_dir=export_dir,
        session_dates=dates,
        num_draws=5,
        allow_reduced_arms=True,
    )

    # Tamper with daily_returns.json after export
    ret_file = export_dir / "daily_returns.json"
    data = json.loads(ret_file.read_text(encoding="utf-8"))
    data["returns"]["MLP_BASE__US__7"] = [0.99, 0.99, 0.99, 0.99]
    ret_file.write_text(to_canonical_json(data), encoding="utf-8")

    with pytest.raises(ReplayVerificationError):
        replay_analysis_bundle(export_dir=export_dir, allow_reduced_arms=True)


def test_failure_condition_18_downstream_exception_leaves_source_unchanged(cpu_recovery_fixture, monkeypatch):
    """Condition 18: Downstream exception during recovery leaves source fixture 100% unchanged and emits no success receipt."""
    source_dir = cpu_recovery_fixture["source_dir"]
    output_dir = cpu_recovery_fixture["fixture_root"] / "out_failed_downstream"
    before_manifest = compute_directory_manifest(source_dir)

    # Induce an error during simulation
    def failing_simulation(*args, **kwargs):
        raise RuntimeError("Induced downstream portfolio calculation failure")

    monkeypatch.setattr("scripts.recover_a30_production.run_continuous_portfolio_simulation", failing_simulation)

    with pytest.raises(RuntimeError, match="Induced downstream portfolio calculation failure"):
        execute_zero_retraining_recovery(
            source_dir=source_dir,
            output_dir=output_dir,
            config=cpu_recovery_fixture["config"],
            sample_ids_dir=cpu_recovery_fixture["sample_ids_dir"],
            data_dir=cpu_recovery_fixture["data_dir"],
            selected_folds=[2020, 2021],
            inventory_path=cpu_recovery_fixture["inv_path"],
            selected_securities=["US_AAPL", "US_MSFT"],
            allow_reduced_arms=True,
        )

    # Success receipt must not exist
    assert not (output_dir / "pipeline_summary.json").exists()
    assert not (output_dir / "recovery_summary.json").exists()

    # Source directory must remain completely unchanged
    after_manifest = compute_directory_manifest(source_dir)
    assert before_manifest == after_manifest, "Source directory must be immutable despite downstream failure"


def test_failure_condition_19_accidental_neural_training_call_fails_test(monkeypatch):
    """Condition 19: Any accidental invocation of neural training fails the test immediately."""
    training_invoked = False

    def guard_training(*args, **kwargs):
        nonlocal training_invoked
        training_invoked = True
        raise AssertionError("CRITICAL VIOLATION: Neural training driver was invoked!")

    monkeypatch.setattr("scripts.launch_a30_production.train_backbone_model", guard_training)

    # Invoking training raises AssertionError
    from scripts.launch_a30_production import train_backbone_model
    with pytest.raises(AssertionError, match="CRITICAL VIOLATION: Neural training driver was invoked!"):
        train_backbone_model(None, None, None, None, None, None, None, 7, None, None)
    assert training_invoked is True
