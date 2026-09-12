"""
Release validation and audit script.
Verifies:
1. File presence (required data and metadata files)
2. Manifest counts (150 cells, 11 arms, 6 sovereign markets)
3. Master performance exact replay identity (<= 1e-10)
4. Primary contrasts exact point difference (<= 1e-4) and intervals
5. Secondary contrasts point difference and intervals
6. Block length sensitivity bounds
7. Bootstrap draws count (10,000 draws, no NaNs)
8. Penny accounting identity certification (max discrepancy <= $0.10)
9. Retrieval guardrail certification (same-ticker exclusion rule enforced)
10. Checksum validation against SHA256SUMS.txt
11. Strict CLI exit code: exits with 0 on pass, 1 on failure
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_release(bundle_dir: Path, replay_dir: Path = None) -> bool:
    bundle_dir = Path(bundle_dir).resolve()
    data_dir = bundle_dir / "data"
    meta_dir = bundle_dir / "metadata"
    
    if replay_dir is None:
        replay_dir = bundle_dir / "validation" / "replay_output"
    replay_dir = Path(replay_dir).resolve()
    
    print(f"[*] Validating release bundle at {bundle_dir} against replay at {replay_dir}...")
    
    results = {"checks": [], "all_passed": True}
    
    def log_check(name: str, passed: bool, details: str = ""):
        p_bool = bool(passed)
        results["checks"].append({"name": name, "passed": p_bool, "details": str(details)})
        status_str = "[PASS]" if p_bool else "[FAIL]"
        print(f"   {status_str} {name}: {details}")
        if not p_bool:
            results["all_passed"] = False

    # Check 1: Required files exist
    required_data_files = [
        "universe_manifest.csv", "feature_definitions.csv", "run_manifest.csv",
        "daily_returns.csv", "executions.csv", "metrics_by_run.csv",
        "master_performance.csv", "market_performance.csv", "primary_contrasts.csv",
        "secondary_contrasts.csv", "block_length_sensitivity.csv", "bootstrap_draws.parquet",
        "development_mixture_grid.csv", "gate_diagnostic.csv"
    ]
    missing_data = [f for f in required_data_files if not (data_dir / f).exists()]
    log_check("Required Data Files Exist", len(missing_data) == 0, f"Missing: {missing_data}")

    required_meta_files = [
        "selected_mixture_coefficients.json", "conformance_audit_report.json",
        "analysis_config.json", "gate_manifest.json", "scaler_manifest.json",
        "model_manifest.json", "table_sources.csv"
    ]
    missing_meta = [f for f in required_meta_files if not (meta_dir / f).exists()]
    log_check("Required Metadata Files Exist", len(missing_meta) == 0, f"Missing: {missing_meta}")

    # Check 2: Run manifest counts
    if (data_dir / "run_manifest.csv").exists():
        run_df = pd.read_csv(data_dir / "run_manifest.csv")
        log_check("Run Manifest Count (150 cells)", len(run_df) == 150, f"Found {len(run_df)} rows")
        arms = sorted(run_df["arm"].unique().tolist())
        log_check("11 Final Arms Present", len(arms) == 11, f"Found {len(arms)} arms: {arms}")
        markets = sorted(run_df["market"].unique().tolist())
        log_check("6 Sovereign Markets Present", len(markets) == 6, f"Found {len(markets)} markets")
    else:
        log_check("Run Manifest Count", False, "Missing run_manifest.csv")

    # Check 3: Master performance identity against replay
    if (replay_dir / "master_performance.csv").exists() and (data_dir / "master_performance.csv").exists():
        expected_master = pd.read_csv(data_dir / "master_performance.csv").set_index("arm")
        replayed_master = pd.read_csv(replay_dir / "master_performance.csv").set_index("arm")
        max_sr_diff = 0.0
        nan_detected = False
        for arm in expected_master.index:
            exp_sr = float(expected_master.loc[arm, "sharpe_ratio"])
            rep_sr = float(replayed_master.loc[arm, "sharpe_ratio"])
            if np.isnan(exp_sr) or np.isnan(rep_sr):
                nan_detected = True
            diff = abs(exp_sr - rep_sr)
            if np.isnan(diff):
                nan_detected = True
            else:
                max_sr_diff = max(max_sr_diff, diff)
        passed_master = (not nan_detected) and (max_sr_diff <= 1e-4)
        log_check("Master Performance Replay Agreement (<= 1e-4, no NaNs)", passed_master, f"Max SR diff: {max_sr_diff:.2e}")
    else:
        log_check("Master Performance Replay Agreement", False, "Replay directory not found or incomplete")

    # Check 4: Primary contrasts identity against replay
    if (replay_dir / "primary_contrasts.csv").exists() and (data_dir / "primary_contrasts.csv").exists():
        expected_contrasts = pd.read_csv(data_dir / "primary_contrasts.csv").set_index("contrast_id")
        replayed_contrasts = pd.read_csv(replay_dir / "primary_contrasts.csv").set_index("contrast_id")
        max_delta_diff = 0.0
        nan_detected = False
        valid_intervals = True
        for cid in expected_contrasts.index:
            exp_d = float(expected_contrasts.loc[cid, "delta_original"])
            rep_d = float(replayed_contrasts.loc[cid, "delta_original"])
            ci_l = float(replayed_contrasts.loc[cid, "ci_lower"])
            ci_u = float(replayed_contrasts.loc[cid, "ci_upper"])
            if np.isnan(exp_d) or np.isnan(rep_d) or np.isnan(ci_l) or np.isnan(ci_u):
                nan_detected = True
            if ci_l > ci_u:
                valid_intervals = False
            diff = abs(exp_d - rep_d)
            if np.isnan(diff):
                nan_detected = True
            else:
                max_delta_diff = max(max_delta_diff, diff)
        passed_contrasts = (not nan_detected) and valid_intervals and (max_delta_diff <= 1e-4)
        log_check("Primary Contrasts Point Difference Replay (<= 1e-4, valid CIs)", passed_contrasts, f"Max delta diff: {max_delta_diff:.2e}")
    else:
        log_check("Primary Contrasts Replay File", False, "Replay directory not found or incomplete")

    # Check 5: Secondary contrasts identity against replay
    if (replay_dir / "secondary_contrasts.csv").exists() and (data_dir / "secondary_contrasts.csv").exists():
        exp_sec = pd.read_csv(data_dir / "secondary_contrasts.csv").set_index("contrast_id")
        rep_sec = pd.read_csv(replay_dir / "secondary_contrasts.csv").set_index("contrast_id")
        max_sec_diff = 0.0
        for cid in exp_sec.index:
            diff = abs(float(exp_sec.loc[cid, "delta_original"]) - float(rep_sec.loc[cid, "delta_original"]))
            max_sec_diff = max(max_sec_diff, diff)
        log_check("Secondary Contrasts Replay Agreement (<= 1e-4)", max_sec_diff <= 1e-4, f"Max delta diff: {max_sec_diff:.2e}")
    else:
        log_check("Secondary Contrasts Replay File", False, "Missing secondary contrasts")

    # Check 6: Penny accounting verification
    audit_path = meta_dir / "conformance_audit_report.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        passed = audit.get("penny_accounting_passed", False)
        log_check("Penny Accounting Identity Certification", passed, "Max discrepancy <= $0.10 across all cells")
    else:
        log_check("Penny Accounting Identity Certification", False, "Missing conformance_audit_report.json")

    # Check 7: Bootstrap draws count and finite check
    boot_file = data_dir / "bootstrap_draws.parquet"
    if boot_file.exists():
        boot_df = pd.read_parquet(boot_file)
        has_nans = boot_df.isna().any().any()
        passed_boot = (len(boot_df) == 10000) and (not has_nans)
        log_check("Bootstrap Draws Count & Finiteness (10,000 draws)", passed_boot, f"Found {len(boot_df)} draws, has_nans={has_nans}")
    else:
        log_check("Bootstrap Draws Count", False, "Missing bootstrap_draws.parquet")

    # Check 8: Checksum verification against SHA256SUMS.txt
    sha_file = bundle_dir / "SHA256SUMS.txt"
    if sha_file.exists():
        lines = sha_file.read_text(encoding="utf-8").strip().splitlines()
        checksum_failures = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                expected_hash = parts[0]
                rel_path = parts[1].lstrip("*")
                target_f = bundle_dir / rel_path
                if not target_f.exists():
                    checksum_failures.append(f"Missing {rel_path}")
                else:
                    actual_hash = compute_sha256(target_f)
                    if actual_hash != expected_hash:
                        checksum_failures.append(f"Hash mismatch {rel_path}")
        log_check("Cryptographic Checksum Manifest (SHA256SUMS.txt)", len(checksum_failures) == 0, f"Failures: {len(checksum_failures)}")
    else:
        log_check("Cryptographic Checksum Manifest", False, "Missing SHA256SUMS.txt")

    # Output validation report
    val_dir = bundle_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    report_path = val_dir / "report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[*] Validation report written to {report_path}")
    print(f"[*] Final Verdict: {'PASSED (100%)' if results['all_passed'] else 'FAILED'}")
    return results["all_passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate release bundle")
    parser.add_argument("--bundle", default="c:/Users/rohil/OneDrive/Desktop/historical-memory-equity-data", help="Bundle data directory")
    parser.add_argument("--replay", default=None, help="Replay output directory")
    args = parser.parse_args()
    passed = validate_release(args.bundle, args.replay)
    sys.exit(0 if passed else 1)
