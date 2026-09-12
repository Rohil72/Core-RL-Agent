"""
Comprehensive Validation & Negative Test Suite for Public Release.
Verifies that:
1. Exporting does not overwrite corrected tech_* feature definitions (regression test).
2. validate_release passes on genuine canonical bundle.
3. validate_release fails on corrupted confidence intervals.
4. validate_release fails on altered p-values / Holm ordering.
5. validate_release fails on NaN in secondary contrasts.
6. validate_release fails on duplicated run IDs in run manifest.
7. validate_release fails on empty checksum list.
8. validate_release fails on missing mandatory inputs.
"""

import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from memory_study.cache_builder import FEATURE_NAMES_23
from memory_study.export_release_bundle import generate_evaluated_feature_definitions, export_bundle
from memory_study.validate_release import validate_release

BUNDLE_PATH = Path("c:/Users/rohil/OneDrive/Desktop/historical-memory-equity-data")


def test_feature_export_regression(tmp_path):
    """Regression test: export must NOT overwrite corrected tech_* dictionary with different representation."""
    # 1. Verify in-memory generator
    df_feat = generate_evaluated_feature_definitions()
    assert len(df_feat) == 23, f"Expected 23 features, got {len(df_feat)}"
    assert list(df_feat["name"]) == FEATURE_NAMES_23, "Feature names do not match FEATURE_NAMES_23 in exact order"
    for name in df_feat["name"]:
        assert name.startswith("tech_"), f"Feature {name} does not start with tech_"
    
    # Check no legacy names present
    legacy_forbidden = ["returns_1d", "returns_5d", "ma_ratio_5", "volatility_21", "bollinger_upper", "high_distance_63"]
    for leg in legacy_forbidden:
        assert leg not in df_feat["name"].values, f"Legacy feature {leg} detected in export definitions!"

    # 2. Test export_bundle into a temp directory
    out_dir = tmp_path / "bundle_export_test"
    export_bundle(source_dir=Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent/research_runs/memory_study/final_comparison"),
                  output_dir=out_dir)
    
    exported_csv = out_dir / "data" / "feature_definitions.csv"
    assert exported_csv.exists(), "feature_definitions.csv was not created by export_bundle"
    exported_df = pd.read_csv(exported_csv)
    assert len(exported_df) == 23
    assert list(exported_df["name"]) == FEATURE_NAMES_23
    for leg in legacy_forbidden:
        assert leg not in exported_df["name"].values, f"Legacy feature {leg} found in exported CSV!"


def test_validation_fails_on_corrupted_interval(tmp_path):
    """Negative test: corrupted confidence interval (ci_lower > ci_upper) must trigger validation failure."""
    temp_bundle = tmp_path / "corrupted_ci_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    # Corrupt primary contrasts CI
    prim_file = temp_bundle / "data" / "primary_contrasts.csv"
    df = pd.read_csv(prim_file)
    df.loc[0, "ci_lower"] = 2.5
    df.loc[0, "ci_upper"] = -1.0
    df.to_csv(prim_file, index=False)
    
    passed = validate_release(temp_bundle)
    assert not passed, "Validator erroneously passed on inverted confidence interval!"


def test_validation_fails_on_altered_p_value(tmp_path):
    """Negative test: altered p-value violating Holm adjustment ordering must fail."""
    temp_bundle = tmp_path / "altered_p_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    prim_file = temp_bundle / "data" / "primary_contrasts.csv"
    df = pd.read_csv(prim_file)
    df.loc[0, "p_holm"] = 0.0001  # Incorrect Holm value
    df.to_csv(prim_file, index=False)
    
    passed = validate_release(temp_bundle)
    assert not passed, "Validator erroneously passed on altered Holm p-value!"


def test_validation_fails_on_nan_secondary_result(tmp_path):
    """Negative test: unexpected NaN in secondary contrasts must fail numeric validity check."""
    temp_bundle = tmp_path / "nan_sec_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    sec_file = temp_bundle / "data" / "secondary_contrasts.csv"
    df = pd.read_csv(sec_file)
    df.loc[0, "delta_original"] = np.nan
    df.to_csv(sec_file, index=False)
    
    passed = validate_release(temp_bundle)
    assert not passed, "Validator erroneously passed on NaN in secondary contrasts!"


def test_validation_fails_on_duplicated_run(tmp_path):
    """Negative test: duplicate run_id in run manifest must fail run membership check."""
    temp_bundle = tmp_path / "dup_run_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    run_file = temp_bundle / "data" / "run_manifest.csv"
    df = pd.read_csv(run_file)
    # Duplicate row 0 to create duplicate run_id
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.to_csv(run_file, index=False)
    
    passed = validate_release(temp_bundle)
    assert not passed, "Validator erroneously passed on duplicate run ID in run manifest!"


def test_validation_fails_on_empty_checksum_list(tmp_path):
    """Negative test: empty SHA256SUMS.txt must fail checksum verification."""
    temp_bundle = tmp_path / "empty_sha_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    sha_file = temp_bundle / "SHA256SUMS.txt"
    sha_file.write_text("# Empty checksum list\n", encoding="utf-8")
    
    passed = validate_release(temp_bundle)
    assert not passed, "Validator erroneously passed on empty SHA256SUMS.txt!"


def test_validation_fails_on_missing_required_input(tmp_path):
    """Negative test: missing mandatory daily returns file must fail presence check."""
    temp_bundle = tmp_path / "missing_input_bundle"
    shutil.copytree(BUNDLE_PATH, temp_bundle)
    
    # Remove mandatory files
    (temp_bundle / "data" / "daily_returns.csv").unlink(missing_ok=True)
    (temp_bundle / "data" / "daily_returns.parquet").unlink(missing_ok=True)
    
    passed = validate_release(temp_bundle)
    assert not passed, "Validator erroneously passed when daily_returns was missing!"


def test_validation_passes_on_genuine_bundle():
    """Positive test: genuine release bundle must pass all validation checks with 100% success."""
    passed = validate_release(BUNDLE_PATH)
    assert passed, "Validator failed on genuine release bundle!"
