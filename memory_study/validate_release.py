"""
Release validation and audit script.
Comprehensive verification of public release bundle and replay reproduction.

Two-tier verification:
1. Bundle Consistency Validation (standalone, checks frozen data, metadata, exact cell membership, schemas, finiteness, checksums)
2. Replay Verification (requires complete replay directory, evaluates exact 1e-10 unrounded agreement on primary, secondary, sensitivity, bootstrap draws, and performance)

CLI exits with code 0 on complete success, 1 on any failure.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd
import jsonschema


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def step_down_holm_bonferroni(raw_p_values: List[float]) -> List[float]:
    m = len(raw_p_values)
    if m <= 1:
        return [float(p) for p in raw_p_values]
    p_vals = np.array(raw_p_values, dtype=float)
    sort_order = np.argsort(p_vals)
    sorted_p = p_vals[sort_order]
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for i in range(m):
        rank = i + 1
        adj = (m - rank + 1) * sorted_p[i]
        running_max = max(running_max, adj)
        adjusted[i] = min(1.0, running_max)
    original_order = np.empty(m, dtype=float)
    original_order[sort_order] = adjusted
    return [float(p) for p in original_order]


class ValidationResult:
    def __init__(self, bundle_passed: bool, replay_passed: Optional[bool], report: Dict[str, Any]):
        self.bundle_passed = bundle_passed
        self.replay_passed = replay_passed
        self.report = report

    @property
    def all_passed(self) -> bool:
        if self.replay_passed is None:
            return self.bundle_passed
        return self.bundle_passed and self.replay_passed

    def __bool__(self) -> bool:
        return self.all_passed

    def __getitem__(self, item: str) -> Any:
        return self.report[item]

    def get(self, k: str, default: Any = None) -> Any:
        return self.report.get(k, default)

    def __repr__(self) -> str:
        rep_status = "SKIPPED" if self.replay_passed is None else ("PASSED" if self.replay_passed else "FAILED")
        return f"<ValidationResult bundle_passed={self.bundle_passed} replay={rep_status} overall={self.all_passed}>"


def build_expected_cell_grid() -> Set[Tuple[str, str, str]]:
    markets = ["Brazil", "China", "France", "India", "UK", "US"]
    grid = set()
    
    # 4 deterministic arms with seed fixed
    det_arms = ["BENCH_EQUAL_WEIGHT", "BENCH_MOMENTUM_21", "HIST_PRIOR", "MEM_SIM"]
    for arm in det_arms:
        for m in markets:
            grid.add((arm, m, "fixed"))
            
    # 1 random arm with seeds 1001, 1002, 1003
    for s in ["1001", "1002", "1003"]:
        for m in markets:
            grid.add(("MEM_RANDOM", m, s))
            
    # 6 model arms with seeds 7, 17, 37
    model_arms = ["MLP_BASE", "MLP_GATE", "MLP_MIX_SELECTED", "TRANS_BASE", "TRANS_GATE", "TRANS_MIX_SELECTED"]
    for arm in model_arms:
        for s in ["7", "17", "37"]:
            for m in markets:
                grid.add((arm, m, s))
                
    return grid


def validate_release(bundle_dir: Path, replay_dir: Optional[Path] = None) -> ValidationResult:
    bundle_dir = Path(bundle_dir).resolve()
    data_dir = bundle_dir / "data"
    meta_dir = bundle_dir / "metadata"
    schema_dir = bundle_dir / "schema"
    
    # Check if replay directory is specified
    replay_requested = replay_dir is not None
    replay_path: Optional[Path] = None
    if replay_dir is not None:
        replay_path = Path(replay_dir).resolve()
    
    print(f"[*] Validating release bundle at: {bundle_dir}")
    if replay_path and replay_path.exists():
        print(f"[*] Replay verification target: {replay_path}")
    elif replay_requested:
        print(f"[*] Replay directory requested but does not exist: {replay_path}")
    else:
        print("[*] Standalone mode: Replay directory not supplied. Validating bundle consistency only.")

    results: Dict[str, Any] = {
        "checks": [],
        "bundle_checks": [],
        "replay_checks": [],
        "failed_checks": [],
        "bundle_consistency_passed": True,
        "replay_verified": None,
        "all_passed": True
    }
    
    def log_check(name: str, passed: bool, details: str = "", is_replay: bool = False):
        p_bool = bool(passed)
        rec = {"name": name, "passed": p_bool, "details": str(details), "tier": "replay" if is_replay else "bundle"}
        results["checks"].append(rec)
        if is_replay:
            results["replay_checks"].append(rec)
        else:
            results["bundle_checks"].append(rec)
            
        status_str = "[PASS]" if p_bool else "[FAIL]"
        print(f"   {status_str} {name}: {details}")
        if not p_bool:
            results["failed_checks"].append(name)
            if is_replay:
                results["replay_verified"] = False
            else:
                results["bundle_consistency_passed"] = False

    # =========================================================================
    # PART 1: BUNDLE CONSISTENCY CHECKS
    # =========================================================================

    # Check 1: Required Data Files Exist
    required_data_files = [
        "universe_manifest.csv", "feature_definitions.csv", "run_manifest.csv",
        "daily_returns.csv", "executions.csv", "metrics_by_run.csv",
        "master_performance.csv", "master_performance_full_precision.csv",
        "market_performance.csv", "primary_contrasts.csv", "primary_contrasts_full_precision.csv",
        "secondary_contrasts.csv", "secondary_contrasts_full_precision.csv",
        "block_length_sensitivity.csv", "bootstrap_draws.parquet",
        "development_mixture_grid.csv", "gate_diagnostic.csv"
    ]
    missing_data = [f for f in required_data_files if not (data_dir / f).exists()]
    log_check("Required Data Files Exist", len(missing_data) == 0, f"Missing: {missing_data}")

    # Check 2: Required Metadata Files Exist
    required_meta_files = [
        "selected_mixture_coefficients.json", "conformance_audit_report.json",
        "analysis_config.json", "gate_manifest.json", "scaler_manifest.json",
        "model_manifest.json", "table_sources.csv"
    ]
    missing_meta = [f for f in required_meta_files if not (meta_dir / f).exists()]
    log_check("Required Metadata Files Exist", len(missing_meta) == 0, f"Missing: {missing_meta}")

    # Check 3: Run Membership & Exact Grid Combinations (150 cells)
    run_file = data_dir / "run_manifest.csv"
    if run_file.exists():
        run_df = pd.read_csv(run_file)
        if "run_id" not in run_df.columns:
            run_df.insert(0, "run_id", run_df["arm"] + "_" + run_df["market"] + "_" + run_df["seed"].astype(str))
            
        actual_cells = set(zip(run_df["arm"].astype(str), run_df["market"].astype(str), run_df["seed"].astype(str)))
        expected_cells = build_expected_cell_grid()
        
        diff_missing = expected_cells - actual_cells
        diff_extra = actual_cells - expected_cells
        exact_grid_ok = (len(diff_missing) == 0 and len(diff_extra) == 0)
        unique_run_ok = (len(run_df["run_id"].unique()) == 150) and (len(run_df) == 150)
        
        all_run_ok = exact_grid_ok and unique_run_ok
        log_check("Run Membership & Exact Cell Coverage (150 cells: 11 arms, 6 markets, exact seeds)", all_run_ok,
                  f"150 rows: {len(run_df) == 150}, Unique run_id: {unique_run_ok}, Exact grid: {exact_grid_ok}" +
                  (f", missing={diff_missing}" if diff_missing else "") +
                  (f", extra={diff_extra}" if diff_extra else ""))
    else:
        log_check("Run Membership & Exact Cell Coverage", False, "Missing run_manifest.csv")

    # Check 4: Record Schema & Full Daily Returns Validation
    schema_file = schema_dir / "tables.schema.json"
    if not schema_file.exists():
        schema_file = bundle_dir.parent / "Core-RL-Agent" / "release" / "schema" / "tables.schema.json"
    if schema_file.exists():
        try:
            schema_data = json.loads(schema_file.read_text(encoding="utf-8"))
            definitions = schema_data.get("definitions", {})
            schema_failures = []
            
            # Primary contrasts schema
            if (data_dir / "primary_contrasts.csv").exists():
                p_df = pd.read_csv(data_dir / "primary_contrasts.csv")
                prim_schema = definitions.get("PrimaryContrastRecord", {})
                for idx, row in p_df.iterrows():
                    rec = row.to_dict()
                    try:
                        jsonschema.validate(rec, prim_schema)
                    except Exception as e:
                        schema_failures.append(f"primary_contrasts row {idx}: {e.message}")
                        
            # Secondary contrasts schema
            if (data_dir / "secondary_contrasts.csv").exists():
                s_df = pd.read_csv(data_dir / "secondary_contrasts.csv")
                sec_schema = definitions.get("SecondaryContrastRecord", {})
                for idx, row in s_df.iterrows():
                    rec = row.to_dict()
                    try:
                        jsonschema.validate(rec, sec_schema)
                    except Exception as e:
                        schema_failures.append(f"secondary_contrasts row {idx}: {e.message}")

            # Master performance schema
            if (data_dir / "master_performance.csv").exists():
                m_df = pd.read_csv(data_dir / "master_performance.csv")
                master_schema = definitions.get("MasterPerformanceRecord", {})
                for idx, row in m_df.iterrows():
                    rec = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
                    try:
                        jsonschema.validate(rec, master_schema)
                    except Exception as e:
                        schema_failures.append(f"master_performance row {idx}: {e.message}")

            # Feature definitions schema
            if (data_dir / "feature_definitions.csv").exists():
                f_df = pd.read_csv(data_dir / "feature_definitions.csv")
                feat_schema = definitions.get("FeatureDefinitionRecord", {})
                for idx, row in f_df.iterrows():
                    rec = row.to_dict()
                    try:
                        jsonschema.validate(rec, feat_schema)
                    except Exception as e:
                        schema_failures.append(f"feature_definitions row {idx}: {e.message}")

            # Daily returns: COMPLETE validation across all rows (no 100 row cap)
            daily_file = data_dir / "daily_returns.parquet" if (data_dir / "daily_returns.parquet").exists() else data_dir / "daily_returns.csv"
            if daily_file.exists():
                if str(daily_file).endswith(".parquet"):
                    d_df = pd.read_parquet(daily_file)
                else:
                    d_df = pd.read_csv(daily_file)
                    
                # Full dataset vectorized validity
                n_daily = len(d_df)
                if n_daily < 30000:
                    schema_failures.append(f"daily_returns row count {n_daily} suspiciously low (<30,000)")
                
                # Check finiteness
                ret_finite = np.all(np.isfinite(d_df["return"].values))
                eq_finite = np.all(np.isfinite(d_df["equity"].values))
                if not ret_finite:
                    schema_failures.append("Non-finite values found in daily_returns return column")
                if not eq_finite:
                    schema_failures.append("Non-finite values found in daily_returns equity column")
                    
                # Check bounds
                if (d_df["return"] < -1.0).any() or (d_df["return"] > 10.0).any():
                    schema_failures.append("Daily return values out of realistic bounds [-1.0, 10.0]")
                if (d_df["equity"] <= 0.0).any():
                    schema_failures.append("Equity values non-positive found in daily_returns")
                    
                # Check date pattern across all rows
                date_str = d_df["date"].astype(str)
                date_match = date_str.str.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$").all()
                if not date_match:
                    schema_failures.append("Invalid date format in daily_returns (must match YYYY-MM-DD)")
                    
                # Check calendar week pattern across all rows
                if "calendar_week" in d_df.columns:
                    week_str = d_df["calendar_week"].astype(str)
                    week_match = week_str.str.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$").all()
                    if not week_match:
                        schema_failures.append("Invalid calendar_week format (must match YYYY-MM-DD)")
            else:
                schema_failures.append("Missing daily_returns.csv or parquet")

            log_check("Record Schema & Full Daily Returns Validation", len(schema_failures) == 0,
                      f"Schema errors: {len(schema_failures)}" + (f" ({schema_failures[:2]})" if schema_failures else ""))
        except Exception as e:
            log_check("Record Schema & Full Daily Returns Validation", False, f"Schema parsing exception: {e}")
    else:
        log_check("Record Schema & Full Daily Returns Validation", False, f"Missing schema at {schema_file}")

    # Check 5: Numeric Validity (Finite value checks on primary and secondary contrasts)
    num_valid = True
    nan_reasons = []
    if (data_dir / "primary_contrasts.csv").exists():
        prim_df = pd.read_csv(data_dir / "primary_contrasts.csv")
        check_cols = ["candidate_metric", "comparator_metric", "delta_original", "ci_lower", "ci_upper", "p_raw", "p_holm"]
        for c in check_cols:
            if c in prim_df.columns:
                vals = prim_df[c].astype(float).values
                if not np.all(np.isfinite(vals)):
                    num_valid = False
                    nan_reasons.append(f"Non-finite in primary_contrasts.{c}")
                    
    if (data_dir / "secondary_contrasts.csv").exists():
        sec_df = pd.read_csv(data_dir / "secondary_contrasts.csv")
        check_cols_sec = ["candidate_metric", "comparator_metric", "delta_original", "ci_lower", "ci_upper", "p_raw_unadjusted"]
        for c in check_cols_sec:
            if c in sec_df.columns:
                vals = sec_df[c].astype(float).values
                if not np.all(np.isfinite(vals)):
                    num_valid = False
                    nan_reasons.append(f"Non-finite in secondary_contrasts.{c}")
                    
    log_check("Numeric Validity (Finite checks for primary & secondary)", num_valid,
              "All required contrast metrics finite" if num_valid else f"Errors: {nan_reasons}")

    # Check 6: Point estimates & mathematical identity in bundle (candidate - comparator == delta_original)
    identity_ok = True
    identity_reasons = []
    if (data_dir / "primary_contrasts_full_precision.csv").exists():
        pf_df = pd.read_csv(data_dir / "primary_contrasts_full_precision.csv")
        for _, row in pf_df.iterrows():
            diff = float(row["candidate_metric"]) - float(row["comparator_metric"])
            delta = float(row["delta_original"])
            if abs(diff - delta) > 1e-10:
                identity_ok = False
                identity_reasons.append(f"Primary {row['contrast_id']} delta identity: diff={diff}, delta={delta}")
    else:
        identity_ok = False
        identity_reasons.append("Missing primary_contrasts_full_precision.csv")

    if (data_dir / "secondary_contrasts_full_precision.csv").exists():
        sf_df = pd.read_csv(data_dir / "secondary_contrasts_full_precision.csv")
        for _, row in sf_df.iterrows():
            diff = float(row["candidate_metric"]) - float(row["comparator_metric"])
            delta = float(row["delta_original"])
            if abs(diff - delta) > 1e-10:
                identity_ok = False
                identity_reasons.append(f"Secondary {row['contrast_id']} delta identity: diff={diff}, delta={delta}")

    log_check("Point Estimates & Mathematical Identity (<= 1e-10)", identity_ok,
              "Mathematical identities verified to <= 1e-10" if identity_ok else f"Failures: {identity_reasons}")

    # Check 7: Confidence interval validation (logical ordering: ci_lower <= delta <= ci_upper and ci_lower < ci_upper)
    ci_ok = True
    ci_reasons = []
    if (data_dir / "primary_contrasts.csv").exists():
        p_df = pd.read_csv(data_dir / "primary_contrasts.csv")
        for _, row in p_df.iterrows():
            d = float(row["delta_original"])
            l = float(row["ci_lower"])
            u = float(row["ci_upper"])
            if not (l <= d <= u) or not (l < u):
                ci_ok = False
                ci_reasons.append(f"Invalid primary CI bounds for {row['contrast_id']}: [{l}, {d}, {u}]")
                
    if (data_dir / "secondary_contrasts.csv").exists():
        s_df = pd.read_csv(data_dir / "secondary_contrasts.csv")
        for _, row in s_df.iterrows():
            d = float(row["delta_original"])
            l = float(row["ci_lower"])
            u = float(row["ci_upper"])
            if not (l <= d <= u) or not (l < u):
                ci_ok = False
                ci_reasons.append(f"Invalid secondary CI bounds for {row['contrast_id']}: [{l}, {d}, {u}]")

    log_check("Confidence Intervals Validation (logical bounds and ordering)", ci_ok,
              "All confidence intervals logically ordered (ci_lower <= delta <= ci_upper)" if ci_ok else f"Failures: {ci_reasons}")

    # Check 8: P-Values & Step-Down Holm Adjustment (C1-C5 step-down Holm-Bonferroni ordering)
    p_ok = True
    p_reasons = []
    if (data_dir / "primary_contrasts.csv").exists():
        p_df = pd.read_csv(data_dir / "primary_contrasts.csv")
        raw_p = p_df["p_raw"].astype(float).tolist()
        holm_p = p_df["p_holm"].astype(float).tolist()
        
        if any(p < 0.0 or p > 1.0 for p in raw_p):
            p_ok = False
            p_reasons.append("raw p-values outside [0, 1]")
            
        expected_holm = step_down_holm_bonferroni(raw_p)
        for cid, act_h, exp_h in zip(p_df["contrast_id"], holm_p, expected_holm):
            if abs(act_h - exp_h) > 1e-4:
                p_ok = False
                p_reasons.append(f"Holm mismatch for {cid}: actual={act_h}, expected={exp_h:.4f}")

    log_check("P-Values & Step-Down Holm Adjustment (C1-C5 family)", p_ok,
              "P-values in [0, 1] and step-down Holm multipliers verified" if p_ok else f"Failures: {p_reasons}")

    # Check 9: Sensitivity analysis reported block lengths (2w, 4w, 8w)
    sens_ok = True
    sens_reasons = []
    sens_file = data_dir / "block_length_sensitivity.csv"
    if sens_file.exists():
        sens_df = pd.read_csv(sens_file)
        expected_cols = ["contrast_id", "delta_original", "p_2w", "ci_2w", "p_4w_primary", "ci_4w_primary", "p_8w", "ci_8w"]
        for c in expected_cols:
            if c not in sens_df.columns:
                sens_ok = False
                sens_reasons.append(f"Missing column {c} in block_length_sensitivity.csv")
        for _, row in sens_df.iterrows():
            for p_col in ["p_2w", "p_4w_primary", "p_8w"]:
                val = float(row[p_col])
                if not (0.0 <= val <= 1.0):
                    sens_ok = False
                    sens_reasons.append(f"Invalid p-value {val} in {row['contrast_id']}.{p_col}")
    else:
        sens_ok = False
        sens_reasons.append("Missing block_length_sensitivity.csv")

    log_check("Block Length Sensitivity (L in {2, 4, 8} weeks reported)", sens_ok,
              "All block length columns and valid p-values verified" if sens_ok else f"Failures: {sens_reasons}")

    # Check 10: Bootstrap draws count & strict finiteness (10,000 draws, neither NaN nor inf)
    boot_ok = True
    boot_reasons = []
    boot_file = data_dir / "bootstrap_draws.parquet"
    if boot_file.exists():
        boot_df = pd.read_parquet(boot_file)
        if len(boot_df) != 10000:
            boot_ok = False
            boot_reasons.append(f"Expected 10,000 draws, found {len(boot_df)}")
        num_vals = boot_df.select_dtypes(include=[np.number]).values
        if not np.all(np.isfinite(num_vals)):
            boot_ok = False
            boot_reasons.append("Non-finite values (NaN, +inf, or -inf) detected in bootstrap draws")
    else:
        boot_ok = False
        boot_reasons.append("Missing bootstrap_draws.parquet")

    log_check("Bootstrap Draws Count & Strict Finiteness (10,000 draws, finite)", boot_ok,
              "10,000 draws verified, all numeric draws strictly finite" if boot_ok else f"Failures: {boot_reasons}")

    # Check 11: Cryptographic checksum manifest (SHA256SUMS.txt)
    sha_file = bundle_dir / "SHA256SUMS.txt"
    checksum_ok = True
    checksum_reasons = []
    if sha_file.exists():
        lines = sha_file.read_text(encoding="utf-8").strip().splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((parts[0], parts[1].lstrip("*")))
        if len(entries) == 0:
            checksum_ok = False
            checksum_reasons.append("SHA256SUMS.txt is empty")
        else:
            paths = [p for _, p in entries]
            if len(paths) != len(set(paths)):
                checksum_ok = False
                checksum_reasons.append("Duplicate file paths in SHA256SUMS.txt")
                
            mandatory_in_sha = ["data/primary_contrasts.csv", "data/master_performance.csv", "data/daily_returns.parquet", "data/bootstrap_draws.parquet", "data/feature_definitions.csv"]
            for mf in mandatory_in_sha:
                if mf not in paths:
                    checksum_ok = False
                    checksum_reasons.append(f"Mandatory entry {mf} missing from SHA256SUMS.txt")
                    
            for exp_hash, rel_path in entries:
                if rel_path in ["validation/report.json", "SHA256SUMS.txt"]:
                    checksum_ok = False
                    checksum_reasons.append(f"Self-referential file {rel_path} included in SHA256SUMS.txt")
                    continue
                target_f = bundle_dir / rel_path
                if not target_f.exists():
                    checksum_ok = False
                    checksum_reasons.append(f"Missing file on disk: {rel_path}")
                else:
                    act_hash = compute_sha256(target_f)
                    if act_hash != exp_hash:
                        checksum_ok = False
                        checksum_reasons.append(f"Hash mismatch for {rel_path}: exp={exp_hash[:8]}, act={act_hash[:8]}")
    else:
        checksum_ok = False
        checksum_reasons.append("Missing SHA256SUMS.txt")

    log_check("Cryptographic Checksums (SHA256SUMS.txt: no empty, no duplicates, exact bytes)", checksum_ok,
              "All hashes verified" if checksum_ok else f"Failures: {checksum_reasons}")

    # Check 12: Penny accounting certification (author-reported) & retrieval guardrails
    audit_path = meta_dir / "conformance_audit_report.json"
    accounting_ok = False
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        accounting_ok = bool(audit.get("penny_accounting_passed", False))
    log_check("Penny Accounting Certification (Author-Reported)", accounting_ok,
              "Author-reported in conformance_audit_report.json (max discrepancy <= $0.10 across all cells; not independently verified from raw trade ticks)")

    # =========================================================================
    # PART 2: REPLAY VERIFICATION CHECKS (Require Complete Replay Directory)
    # =========================================================================
    if replay_path is None:
        results["replay_verified"] = None
        print("\n[*] Replay Verification: SKIPPED (no replay directory supplied; reporting bundle consistency only)")
    else:
        # Check completeness of replay directory
        required_replay_files = [
            "primary_contrasts_full_precision.csv",
            "secondary_contrasts_full_precision.csv",
            "block_length_sensitivity.csv",
            "bootstrap_draws.parquet",
            "master_performance_full_precision.csv"
        ]
        missing_replay = [f for f in required_replay_files if not (replay_path / f).exists()]
        if len(missing_replay) > 0:
            results["replay_verified"] = False
            log_check("Replay Directory Completeness", False,
                      f"Replay directory incomplete. Missing: {missing_replay}", is_replay=True)
        else:
            results["replay_verified"] = True
            
            # Check R1: Primary Contrasts Full-Precision Replay Agreement (<= 1e-10)
            r1_ok = True
            r1_reasons = []
            exp_p = pd.read_csv(data_dir / "primary_contrasts_full_precision.csv").set_index("contrast_id")
            rep_p = pd.read_csv(replay_path / "primary_contrasts_full_precision.csv").set_index("contrast_id")
            cols_to_check = ["candidate_metric", "comparator_metric", "delta_original", "ci_lower", "ci_upper", "p_raw", "p_holm"]
            for cid in exp_p.index:
                for c in cols_to_check:
                    diff = abs(float(exp_p.loc[cid, c]) - float(rep_p.loc[cid, c]))
                    if diff > 1e-10:
                        r1_ok = False
                        r1_reasons.append(f"{cid}.{c} diff={diff:.2e} > 1e-10 (exp={exp_p.loc[cid, c]}, rep={rep_p.loc[cid, c]})")
            log_check("Primary Contrasts Full-Precision Replay Agreement (<= 1e-10)", r1_ok,
                      "All primary metrics, CIs, and p-values match replayed outputs to <= 1e-10" if r1_ok else f"Mismatches: {r1_reasons}", is_replay=True)

            # Check R2: Secondary Contrasts Full-Precision Replay Agreement (<= 1e-10)
            r2_ok = True
            r2_reasons = []
            exp_s = pd.read_csv(data_dir / "secondary_contrasts_full_precision.csv").set_index("contrast_id")
            rep_s = pd.read_csv(replay_path / "secondary_contrasts_full_precision.csv").set_index("contrast_id")
            cols_sec = ["candidate_metric", "comparator_metric", "delta_original", "ci_lower", "ci_upper", "p_raw_unadjusted"]
            for cid in exp_s.index:
                for c in cols_sec:
                    diff = abs(float(exp_s.loc[cid, c]) - float(rep_s.loc[cid, c]))
                    if diff > 1e-10:
                        r2_ok = False
                        r2_reasons.append(f"{cid}.{c} diff={diff:.2e} > 1e-10 (exp={exp_s.loc[cid, c]}, rep={rep_s.loc[cid, c]})")
            log_check("Secondary Contrasts Full-Precision Replay Agreement (<= 1e-10)", r2_ok,
                      "All secondary metrics, CIs, and p-values match replayed outputs to <= 1e-10" if r2_ok else f"Mismatches: {r2_reasons}", is_replay=True)

            # Check R3: Block Length Sensitivity Replay Agreement (<= 1e-10)
            r3_ok = True
            r3_reasons = []
            exp_b = pd.read_csv(data_dir / "block_length_sensitivity.csv").set_index("contrast_id")
            rep_b = pd.read_csv(replay_path / "block_length_sensitivity.csv").set_index("contrast_id")
            sens_num_cols = ["delta_original", "p_2w", "p_4w_primary", "p_8w"]
            for cid in exp_b.index:
                for c in sens_num_cols:
                    diff = abs(float(exp_b.loc[cid, c]) - float(rep_b.loc[cid, c]))
                    if diff > 1e-10:
                        r3_ok = False
                        r3_reasons.append(f"{cid}.{c} diff={diff:.2e} > 1e-10 (exp={exp_b.loc[cid, c]}, rep={rep_b.loc[cid, c]})")
                for c in ["ci_2w", "ci_4w_primary", "ci_8w"]:
                    if str(exp_b.loc[cid, c]) != str(rep_b.loc[cid, c]):
                        r3_ok = False
                        r3_reasons.append(f"{cid}.{c} string mismatch: exp={exp_b.loc[cid, c]}, rep={rep_b.loc[cid, c]}")
            log_check("Block Length Sensitivity Replay Agreement (<= 1e-10)", r3_ok,
                      "All sensitivity p-values (2w, 4w, 8w) and intervals match replayed outputs to <= 1e-10" if r3_ok else f"Mismatches: {r3_reasons}", is_replay=True)

            # Check R4: Bootstrap Draws Replay Agreement & Finiteness (<= 1e-10)
            r4_ok = True
            r4_reasons = []
            boot_ref = pd.read_parquet(data_dir / "bootstrap_draws.parquet")
            boot_rep = pd.read_parquet(replay_path / "bootstrap_draws.parquet")
            if len(boot_rep) != 10000:
                r4_ok = False
                r4_reasons.append(f"Replayed bootstrap draws length {len(boot_rep)} != 10,000")
            rep_nums = boot_rep.select_dtypes(include=[np.number]).values
            if not np.all(np.isfinite(rep_nums)):
                r4_ok = False
                r4_reasons.append("Non-finite values detected in replayed bootstrap draws")
            common_cols = [c for c in boot_ref.columns if c in boot_rep.columns and c != "draw_idx"]
            for c in common_cols:
                diff = np.max(np.abs(boot_ref[c].values - boot_rep[c].values))
                if diff > 1e-10:
                    r4_ok = False
                    r4_reasons.append(f"Draw column {c} max diff={diff:.2e} > 1e-10")
            log_check("Bootstrap Draws Replay Agreement & Finiteness (<= 1e-10)", r4_ok,
                      f"All 10,000 draws across {len(common_cols)} contrast columns match replayed draws to <= 1e-10 and are strictly finite" if r4_ok else f"Failures: {r4_reasons}", is_replay=True)

            # Check R5: Master Performance Recomputed Metrics Agreement (<= 1e-10)
            r5_ok = True
            r5_reasons = []
            exp_m = pd.read_csv(data_dir / "master_performance_full_precision.csv").set_index("arm")
            rep_m = pd.read_csv(replay_path / "master_performance_full_precision.csv").set_index("arm")
            for arm in exp_m.index:
                for c in ["annualized_return", "sharpe_ratio"]:
                    diff = abs(float(exp_m.loc[arm, c]) - float(rep_m.loc[arm, c]))
                    if diff > 1e-10:
                        r5_ok = False
                        r5_reasons.append(f"{arm}.{c} diff={diff:.2e} > 1e-10")
            log_check("Master Performance Recomputed Metrics Agreement (<= 1e-10)", r5_ok,
                      "Annualized return and Sharpe ratio match replayed metrics to <= 1e-10" if r5_ok else f"Failures: {r5_reasons}", is_replay=True)

    # Output validation report outside frozen evidence
    val_dir = bundle_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    report_path = val_dir / "report.json"
    
    validation_result = ValidationResult(
        bundle_passed=results["bundle_consistency_passed"],
        replay_passed=results["replay_verified"],
        report=results
    )
    results["all_passed"] = validation_result.all_passed
    
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[*] Validation report written to: {report_path}")
    print(f"[*] Bundle Consistency: {'PASSED (100%)' if results['bundle_consistency_passed'] else 'FAILED'}")
    if results['replay_verified'] is not None:
        print(f"[*] Replay Verification: {'PASSED (100%)' if results['replay_verified'] else 'FAILED'}")
    print(f"[*] Overall Verdict: {'PASSED' if validation_result.all_passed else 'FAILED'}")
    
    return validation_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate release bundle")
    parser.add_argument("--bundle", default="../historical-memory-equity-data", help="Bundle data directory")
    parser.add_argument("--replay", default=None, help="Replay output directory (required for replay verification)")
    args = parser.parse_args()
    result = validate_release(args.bundle, args.replay)
    sys.exit(0 if result.all_passed else 1)
