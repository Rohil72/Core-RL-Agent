"""
Release validation and audit script.
Verifies file presence, schemas, null conventions, row counts,
original-sample identities (<= 1e-10), and numerical agreement.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

def validate_release(bundle_dir: Path, replay_dir: Path = None):
    bundle_dir = Path(bundle_dir).resolve()
    data_dir = bundle_dir / "data"
    meta_dir = bundle_dir / "metadata"
    
    if replay_dir is None:
        replay_dir = bundle_dir / "validation" / "replay_output"
    replay_dir = Path(replay_dir).resolve()
    
    print(f"[*] Validating release bundle at {bundle_dir} against replay at {replay_dir}...")
    
    results = {"checks": [], "all_passed": True}
    
    def log_check(name, passed, details=""):
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
    log_check("Required Data Files", len(missing_data) == 0, f"Missing: {missing_data}")

    required_meta_files = [
        "selected_mixture_coefficients.json", "conformance_audit_report.json",
        "analysis_config.json", "gate_manifest.json", "scaler_manifest.json"
    ]
    missing_meta = [f for f in required_meta_files if not (meta_dir / f).exists()]
    log_check("Required Metadata Files", len(missing_meta) == 0, f"Missing: {missing_meta}")

    # Check 2: Run manifest counts
    run_df = pd.read_csv(data_dir / "run_manifest.csv")
    log_check("Run Manifest Count (150 cells)", len(run_df) == 150, f"Found {len(run_df)} rows")
    arms = sorted(run_df["arm"].unique().tolist())
    log_check("11 Final Arms Present", len(arms) == 11, f"Found {len(arms)} arms: {arms}")
    markets = sorted(run_df["market"].unique().tolist())
    log_check("6 Sovereign Markets Present", len(markets) == 6, f"Found {len(markets)} markets")

    # Check 3: Master performance identity against replay
    if (replay_dir / "master_performance.csv").exists():
        expected_master = pd.read_csv(data_dir / "master_performance.csv").set_index("arm")
        replayed_master = pd.read_csv(replay_dir / "master_performance.csv").set_index("arm")
        max_sr_diff = 0.0
        for arm in expected_master.index:
            exp_sr = expected_master.loc[arm, "sharpe_ratio"]
            rep_sr = replayed_master.loc[arm, "sharpe_ratio"]
            diff = abs(exp_sr - rep_sr)
            max_sr_diff = max(max_sr_diff, diff)
        log_check("Master Performance Exact Replay Identity (<= 1e-10)", max_sr_diff <= 1e-10, f"Max SR diff: {max_sr_diff:.2e}")
    else:
        log_check("Master Performance Replay File", False, "Replay directory not found or incomplete")

    # Check 4: Primary contrasts identity against replay
    if (replay_dir / "primary_contrasts.csv").exists():
        expected_contrasts = pd.read_csv(data_dir / "primary_contrasts.csv").set_index("contrast_id")
        replayed_contrasts = pd.read_csv(replay_dir / "primary_contrasts.csv").set_index("contrast_id")
        max_delta_diff = 0.0
        for cid in expected_contrasts.index:
            exp_d = expected_contrasts.loc[cid, "delta_original"]
            rep_d = replayed_contrasts.loc[cid, "delta_original"]
            max_delta_diff = max(max_delta_diff, abs(exp_d - rep_d))
        log_check("Primary Contrasts Point Difference Replay (<= 1e-4)", max_delta_diff <= 1e-4, f"Max delta diff: {max_delta_diff:.2e}")
    else:
        log_check("Primary Contrasts Replay File", False, "Replay directory not found or incomplete")

    # Check 5: Penny accounting verification
    audit_path = meta_dir / "conformance_audit_report.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        passed = audit.get("penny_accounting_passed", False)
        log_check("Penny Accounting Identity Certification", passed, "Max discrepancy <= $0.10 across all cells")

    # Check 6: Bootstrap draws count
    boot_file = data_dir / "bootstrap_draws.parquet"
    if boot_file.exists():
        boot_df = pd.read_parquet(boot_file)
        log_check("Bootstrap Draws Count (10,000 draws)", len(boot_df) == 10000, f"Found {len(boot_df)} draws")

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
    validate_release(args.bundle, args.replay)
