"""
Comprehensive Validation & Negative Test Suite for Public Release.
Verifies that:
1. Exporting does not overwrite corrected tech_* feature definitions (regression test).
2. validate_release passes on genuine canonical bundle (standalone and with replay).
3. validate_release fails on corrupted confidence intervals (with refreshed checksums, asserting named check).
4. validate_release fails on altered p-values / Holm ordering (with refreshed checksums, asserting named check).
5. validate_release fails on NaN in secondary contrasts (with refreshed checksums, asserting named check).
6. validate_release fails on duplicated run IDs in run manifest (with refreshed checksums, asserting named check).
7. validate_release fails on empty checksum list (asserting named checksum check).
8. validate_release fails on unrefreshed checksum mismatch (asserting named checksum check).
9. validate_release fails on missing mandatory inputs (asserting named file presence check).
10. validate_release fails on replayed output mismatch (asserting named replay check).
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from memory_study.cache_builder import FEATURE_NAMES_23
from memory_study.export_release_bundle import generate_evaluated_feature_definitions, export_bundle
from memory_study.validate_release import validate_release

REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = Path(os.environ.get("RELEASE_BUNDLE_PATH", REPO_ROOT.parent / "historical-memory-equity-data"))
REPLAY_PATH = BUNDLE_PATH / "validation" / "replay_output"


def refresh_fixture_checksums(bundle_dir: Path):
    """Recomputes SHA256SUMS.txt for all files in bundle_dir so cryptographic checks pass."""
    sha_file = bundle_dir / "SHA256SUMS.txt"
    lines = []
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file():
            rel_str = str(p.relative_to(bundle_dir)).replace("\\", "/")
            if rel_str in ["SHA256SUMS.txt", "validation/report.json"]:
                continue
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {rel_str}\n")
    sha_file.write_text("".join(lines), encoding="utf-8")


def test_feature_export_regression(tmp_path):
    """Regression test: export must NOT overwrite corrected tech_* dictionary with different representation."""
    df_feat = generate_evaluated_feature_definitions()
    assert len(df_feat) == 23, f"Expected 23 features, got {len(df_feat)}"
    assert list(df_feat["name"]) == FEATURE_NAMES_23, "Feature names do not match FEATURE_NAMES_23 in exact order"
    for name in df_feat["name"]:
        assert name.startswith("tech_"), f"Feature {name} does not start with tech_"
    
    legacy_forbidden = ["returns_1d", "returns_5d", "ma_ratio_5", "volatility_21", "bollinger_upper", "high_distance_63"]
    for leg in legacy_forbidden:
        assert leg not in df_feat["name"].values, f"Legacy feature {leg} detected in export definitions!"

    out_dir = tmp_path / "bundle_export_test"
    source_dir = REPO_ROOT / "research_runs" / "memory_study" / "final_comparison"
    export_bundle(source_dir=source_dir, output_dir=out_dir)
    
    exported_csv = out_dir / "data" / "feature_definitions.csv"
    assert exported_csv.exists(), "feature_definitions.csv was not created by export_bundle"
    exported_df = pd.read_csv(exported_csv)
    assert len(exported_df) == 23
    assert list(exported_df["name"]) == FEATURE_NAMES_23
    for leg in legacy_forbidden:
        assert leg not in exported_df["name"].values, f"Legacy feature {leg} found in exported CSV!"


def test_validation_fails_on_corrupted_interval(tmp_path):
    """Negative test: corrupted confidence interval (ci_lower > ci_upper) must trigger CI failure while checksums pass."""
    temp_bundle = tmp_path / "corrupted_ci_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    prim_file = temp_bundle / "data" / "primary_contrasts.csv"
    df = pd.read_csv(prim_file)
    df.loc[0, "ci_lower"] = 2.5
    df.loc[0, "ci_upper"] = -1.0
    df.to_csv(prim_file, index=False)
    
    # Refresh checksums so failure is strictly statistical
    refresh_fixture_checksums(temp_bundle)
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed on inverted confidence interval!"
    assert any(c["name"].startswith("Cryptographic Checksums") and c["passed"] for c in res["checks"]), "Checksums should pass when refreshed"
    assert any("Confidence Intervals" in name for name in res["failed_checks"]), f"Expected CI check failure, got: {res['failed_checks']}"


def test_validation_fails_on_altered_p_value(tmp_path):
    """Negative test: altered p-value violating Holm adjustment ordering must fail p-value check while checksums pass."""
    temp_bundle = tmp_path / "altered_p_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    prim_file = temp_bundle / "data" / "primary_contrasts.csv"
    df = pd.read_csv(prim_file)
    df.loc[0, "p_holm"] = 0.0001
    df.to_csv(prim_file, index=False)
    
    refresh_fixture_checksums(temp_bundle)
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed on altered Holm p-value!"
    assert any(c["name"].startswith("Cryptographic Checksums") and c["passed"] for c in res["checks"]), "Checksums should pass when refreshed"
    assert any("P-Values" in name for name in res["failed_checks"]), f"Expected P-Value failure, got: {res['failed_checks']}"


def test_validation_fails_on_nan_secondary_result(tmp_path):
    """Negative test: unexpected NaN in secondary contrasts must fail numeric validity check while checksums pass."""
    temp_bundle = tmp_path / "nan_sec_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    sec_file = temp_bundle / "data" / "secondary_contrasts.csv"
    df = pd.read_csv(sec_file)
    df.loc[0, "delta_original"] = np.nan
    df.to_csv(sec_file, index=False)
    
    refresh_fixture_checksums(temp_bundle)
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed on NaN in secondary contrasts!"
    assert any(c["name"].startswith("Cryptographic Checksums") and c["passed"] for c in res["checks"]), "Checksums should pass when refreshed"
    assert any("Numeric Validity" in name for name in res["failed_checks"]), f"Expected Numeric Validity failure, got: {res['failed_checks']}"


def test_validation_fails_on_duplicated_run(tmp_path):
    """Negative test: duplicate run_id in run manifest must fail run membership check while checksums pass."""
    temp_bundle = tmp_path / "dup_run_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    run_file = temp_bundle / "data" / "run_manifest.csv"
    df = pd.read_csv(run_file)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.to_csv(run_file, index=False)
    
    refresh_fixture_checksums(temp_bundle)
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed on duplicate run ID in run manifest!"
    assert any(c["name"].startswith("Cryptographic Checksums") and c["passed"] for c in res["checks"]), "Checksums should pass when refreshed"
    assert any("Run Membership" in name for name in res["failed_checks"]), f"Expected Run Membership failure, got: {res['failed_checks']}"


def test_validation_fails_on_empty_checksum_list(tmp_path):
    """Negative test: empty SHA256SUMS.txt must specifically fail checksum verification."""
    temp_bundle = tmp_path / "empty_sha_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    sha_file = temp_bundle / "SHA256SUMS.txt"
    sha_file.write_text("# Empty checksum list\n", encoding="utf-8")
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed on empty SHA256SUMS.txt!"
    assert any("Cryptographic Checksums" in name for name in res["failed_checks"]), f"Expected Checksums failure, got: {res['failed_checks']}"


def test_validation_fails_on_unrefreshed_checksum_mismatch(tmp_path):
    """Negative test: unrefreshed checksum mismatch must specifically fail cryptographic checksums check."""
    temp_bundle = tmp_path / "mismatch_sha_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    # Tamper with file without refreshing SHA256SUMS.txt
    prim_file = temp_bundle / "data" / "primary_contrasts.csv"
    prim_file.write_text(prim_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed on tampered file without refreshed checksum!"
    assert any("Cryptographic Checksums" in name for name in res["failed_checks"]), f"Expected Checksums failure, got: {res['failed_checks']}"


def test_validation_fails_on_missing_required_input(tmp_path):
    """Negative test: missing mandatory daily returns file must fail data presence check."""
    temp_bundle = tmp_path / "missing_input_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    (temp_bundle / "data" / "daily_returns.csv").unlink(missing_ok=True)
    (temp_bundle / "data" / "daily_returns.parquet").unlink(missing_ok=True)
    
    res = validate_release(temp_bundle)
    assert not res, "Validator erroneously passed when daily_returns was missing!"
    assert any("Required Data Files" in name for name in res["failed_checks"]), f"Expected file presence failure, got: {res['failed_checks']}"


def test_validation_fails_on_mismatched_replay_output(tmp_path):
    """Negative test: corrupted replay output must specifically fail replay agreement at <= 1e-10."""
    temp_replay = tmp_path / "corrupted_replay"
    shutil.copytree(REPLAY_PATH, temp_replay)
    
    # Corrupt replay primary delta by 0.05
    rep_file = temp_replay / "primary_contrasts_full_precision.csv"
    df = pd.read_csv(rep_file)
    df.loc[0, "delta_original"] += 0.05
    df.to_csv(rep_file, index=False)
    
    res = validate_release(BUNDLE_PATH, replay_dir=temp_replay)
    assert not res, "Validator erroneously passed on mismatched replay delta!"
    assert res.bundle_passed is True, "Bundle consistency should pass"
    assert res.replay_passed is False, "Replay verification should fail"
    assert any("Primary Contrasts Full-Precision Replay Agreement" in name for name in res["failed_checks"])


def test_validation_passes_on_genuine_bundle():
    """Positive test: genuine release bundle must pass bundle consistency with 100% success."""
    res = validate_release(BUNDLE_PATH)
    assert res.all_passed is True, f"Validator failed on genuine bundle: {res['failed_checks']}"
    assert res.bundle_passed is True


def test_validation_passes_on_genuine_bundle_with_replay():
    """Positive test: genuine release bundle with genuine replay output must pass both bundle consistency and replay verification."""
    res = validate_release(BUNDLE_PATH, replay_dir=REPLAY_PATH)
    assert res.all_passed is True, f"Validator failed on genuine bundle + replay: {res['failed_checks']}"
    assert res.bundle_passed is True
    assert res.replay_passed is True
