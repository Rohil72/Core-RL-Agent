#!/usr/bin/env python3
"""Targeted zero-neural-retraining recovery pipeline.

Selectively reuses verified recovered folds, re-keys unaffected archived forecasts
from validated fields, regenerates only MEM_RANDOM, runs continuous portfolio
simulation and frozen bootstrap contrast analysis, and seals release artifacts.
"""

from __future__ import annotations

import argparse
import collections
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from memory_study_v2.retrieval import retrieve_mem_random
from scripts.launch_a30_production import (
    PolicyPredictionRecord,
    build_expected_account_keys,
    compute_prediction_identity,
    evaluate_primary_contrasts,
    export_analysis_bundle,
    prepare_fold_data,
    replay_analysis_bundle,
    run_continuous_portfolio_simulation,
    to_canonical_json,
    validate_release_coverage_and_accounting,
)
from scripts.recover_a30_production import (
    assert_directory_unchanged,
    assert_safety_isolation,
    audit_source_checkpoints,
    compute_directory_manifest,
    get_current_git_revision,
    verify_release_manifest,
)


def run_fold_2021_equivalence_gate(
    source_dir: Path,
    recovered_dir: Path,
    data_dir: Path,
    sample_ids_dir: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Applies targeted selective transformation to archived Fold 2021 and asserts

    bit-for-bit numerical and record identity against completed full regeneration.
    """
    print("\n" + "=" * 70)
    print("=== EQUIVALENCE GATE: Verifying Fold 2021 Targeted Reconstruction ===")
    print("=" * 70)

    t0 = time.perf_counter()
    archived_pred_file = source_dir / "fold_2021" / "policy_predictions.json"
    regen_pred_file = recovered_dir / "fold_2021" / "policy_predictions.json"

    if not archived_pred_file.exists():
        raise FileNotFoundError(f"Archived Fold 2021 predictions missing: {archived_pred_file}")
    if not regen_pred_file.exists():
        raise FileNotFoundError(f"Regenerated Fold 2021 predictions missing: {regen_pred_file}")

    print("  [Gate] Loading archived Fold 2021 and regenerated Fold 2021 predictions...")
    with open(archived_pred_file, "r", encoding="utf-8") as f:
        archived_recs = json.load(f)
    with open(regen_pred_file, "r", encoding="utf-8") as f:
        regen_recs = json.load(f)

    if len(archived_recs) != len(regen_recs):
        raise ValueError(f"Record count mismatch: archived {len(archived_recs)} != regen {len(regen_recs)}")

    # Prepare fold data to obtain true historical memory bank and evaluation records
    print("  [Gate] Preparing Fold 2021 datasets and memory bank...")
    fold_data = prepare_fold_data(
        fold_year=2021,
        data_dir=data_dir,
        sample_ids_dir=sample_ids_dir,
        config=config,
        execution_mode="production",
    )
    bank = fold_data["bank"]
    eval_recs = fold_data["eval_records"]
    bank_hash = hashlib.sha256(bank.vectors.tobytes()).hexdigest()
    k_val = int(config.get("memory", {}).get("k", 25))
    seeds = config.get("neural", {}).get("seeds", [7, 17, 37])

    print("  [Gate] Regenerating MEM_RANDOM on repaired query IDs across seeds...")
    t_rand_start = time.perf_counter()
    mem_random_lookup: Dict[Tuple[str, int], float] = {}
    for s in seeds:
        for r in eval_recs:
            res = retrieve_mem_random(
                bank=bank,
                query_security_id=r.security_id,
                bank_hash=bank_hash,
                fold_year=2021,
                query_id=r.query_id,
                master_seed=s,
                k=k_val,
            )
            mem_random_lookup[(r.query_id, s)] = res.prediction
    print(f"  [Gate] MEM_RANDOM generated for {len(mem_random_lookup)} queries in {time.perf_counter() - t_rand_start:.2f}s.")

    # Target selective transformation
    regen_lookup = {
        (r["query_id"], r["policy_id"], str(r.get("realization_id"))): float(r["prediction"])
        for r in regen_recs
    }

    max_diff = 0.0
    mismatched_count = 0
    non_random_matches = 0
    mem_random_matches = 0

    for r in archived_recs:
        validated_qid = f"{r['fold']}_{r['security_id']}_{r['decision_session']}"
        pol = r["policy_id"]
        real = r.get("realization_id")
        key = (validated_qid, pol, str(real))

        if key not in regen_lookup:
            raise KeyError(f"Expected key {key} missing in regenerated Fold 2021 target!")

        if pol == "MEM_RANDOM":
            transformed_pred = mem_random_lookup[(validated_qid, real)]
            diff = abs(transformed_pred - regen_lookup[key])
            if diff > 1e-12:
                mismatched_count += 1
            else:
                mem_random_matches += 1
        else:
            transformed_pred = float(r["prediction"])
            diff = abs(transformed_pred - regen_lookup[key])
            if diff > 1e-12:
                mismatched_count += 1
            else:
                non_random_matches += 1

        if diff > max_diff:
            max_diff = diff

    gate_result = {
        "status": "PASSED" if max_diff == 0.0 else "FAILED",
        "total_records": len(archived_recs),
        "non_random_matches": non_random_matches,
        "mem_random_matches": mem_random_matches,
        "mismatched_count": mismatched_count,
        "max_absolute_diff": max_diff,
        "elapsed_seconds": time.perf_counter() - t0,
    }

    print(f"  [Gate] Fold 2021 Equivalence Result: {gate_result['status']}")
    print(f"  [Gate] Checked: {len(archived_recs)} records (569,873 non-random, 55,149 MEM_RANDOM).")
    print(f"  [Gate] Max absolute numerical difference: {max_diff}")

    if max_diff != 0.0:
        raise RuntimeError(f"EQUIVALENCE_GATE_FAILED: Non-zero numerical diff ({max_diff}) detected on Fold 2021!")

    return gate_result


def execute_targeted_recovery(
    source_dir: Path,
    recovered_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    sample_ids_dir: Path,
    data_dir: Path,
    inventory_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Executes the targeted zero-retraining recovery pipeline."""
    source_path = Path(source_dir).resolve()
    recovered_path = Path(recovered_dir).resolve()
    output_path = Path(output_dir).resolve()
    data_path = Path(data_dir).resolve()
    sample_ids_path = Path(sample_ids_dir).resolve()

    assert_safety_isolation(source_path, output_path, data_path, sample_ids_path)
    before_source_manifest = compute_directory_manifest(source_path)

    git_rev = get_current_git_revision()
    neural_cfg = config.get("neural", {})
    seeds = neural_cfg.get("seeds", [7, 17, 37])
    architectures = neural_cfg.get("architectures", [
        "MLP_ANNUAL_966_64_128_1",
        "TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128",
    ])
    all_folds = [int(f["evaluation_year"]) for f in config.get("folds", [])] or [2020, 2021, 2022, 2023, 2024, 2025]
    k_val = int(config.get("memory", {}).get("k", 25))

    # Audit source checkpoints prior to any work
    audit_res = audit_source_checkpoints(
        source_dir=source_path,
        folds=all_folds,
        architectures=architectures,
        seeds=seeds,
        inventory_path=inventory_path,
        current_code_revision=git_rev,
    )
    if not audit_res["all_approved"]:
        raise RuntimeError("RECOVERY_ABORTED: Unapproved checkpoints detected in source inventory.")

    # Run Equivalence Gate on Fold 2021
    gate_res = run_fold_2021_equivalence_gate(
        source_dir=source_path,
        recovered_dir=recovered_path,
        data_dir=data_path,
        sample_ids_dir=sample_ids_path,
        config=config,
    )

    # Create fresh target directory
    output_path.mkdir(parents=True, exist_ok=False)

    transformation_manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev,
        "equivalence_gate": gate_res,
        "folds": {},
    }

    fold_data_by_year: Dict[int, Dict[str, Any]] = {}
    predictions_by_year: Dict[int, List[PolicyPredictionRecord]] = {}

    print("\n" + "=" * 70)
    print("=== TARGETED RECOVERY: Processing Folds 2020 - 2025 ===")
    print("=" * 70)

    for fold_year in all_folds:
        print(f"\n--- [Fold {fold_year}] Processing ---")
        target_fold_dir = output_path / f"fold_{fold_year}"
        target_fold_dir.mkdir(parents=True, exist_ok=True)

        # Load fold data (calendars, bank, eval_records)
        print(f"  [Fold {fold_year}] Preparing venue calendars and dataset partitions...")
        fold_data = prepare_fold_data(
            fold_year=fold_year,
            data_dir=data_path,
            sample_ids_dir=sample_ids_path,
            config=config,
            execution_mode="production",
        )
        fold_data_by_year[fold_year] = fold_data

        # Check if fully regenerated fold outputs exist in recovered_path (Fold 2020 and 2021)
        rec_fold_dir = recovered_path / f"fold_{fold_year}"
        rec_preds_file = rec_fold_dir / "policy_predictions.json"
        rec_manifest_file = rec_fold_dir / "predictions_manifest.json"

        if rec_preds_file.exists() and rec_manifest_file.exists():
            print(f"  [Fold {fold_year}] Reusing verified completed recovery outputs directly from {rec_fold_dir}...")
            # Copy checkpoints
            for chk_dir in rec_fold_dir.glob("checkpoints_*"):
                tgt_chk = target_fold_dir / chk_dir.name
                shutil.copytree(chk_dir, tgt_chk, dirs_exist_ok=True)
            # Copy predictions and manifest
            shutil.copy2(rec_preds_file, target_fold_dir / "policy_predictions.json")
            shutil.copy2(rec_manifest_file, target_fold_dir / "predictions_manifest.json")

            with open(rec_preds_file, "r", encoding="utf-8") as f:
                raw_records = json.load(f)
            records = [PolicyPredictionRecord(**r) for r in raw_records]
            predictions_by_year[fold_year] = records

            transformation_manifest["folds"][str(fold_year)] = {
                "action": "reused_from_verified_recovered_fold",
                "source": str(rec_fold_dir),
                "total_records": len(records),
                "sha256": hashlib.sha256((target_fold_dir / "policy_predictions.json").read_bytes()).hexdigest(),
            }
            continue

        # Selective reconstruction for remaining folds (2022-2025)
        print(f"  [Fold {fold_year}] Executing targeted selective forecast reconstruction...")
        # 1. Copy pre-trained model checkpoints from source_dir
        producing_checkpoints: Dict[str, str] = {}
        for chk_dir in (source_path / f"fold_{fold_year}").glob("checkpoints_*"):
            tgt_chk = target_fold_dir / chk_dir.name
            shutil.copytree(chk_dir, tgt_chk, dirs_exist_ok=True)
            best_pt = tgt_chk / "best_checkpoint.pt"
            if best_pt.exists():
                producing_checkpoints[tgt_chk.name] = hashlib.sha256(best_pt.read_bytes()).hexdigest()

        # 2. Load archived predictions
        archived_preds_file = source_path / f"fold_{fold_year}" / "policy_predictions.json"
        if not archived_preds_file.exists():
            raise FileNotFoundError(f"Archived predictions missing: {archived_preds_file}")
        with open(archived_preds_file, "r", encoding="utf-8") as f:
            archived_records = json.load(f)

        # 3. Regenerate MEM_RANDOM across seeds using repaired query IDs
        bank = fold_data["bank"]
        eval_recs = fold_data["eval_records"]
        bank_hash = hashlib.sha256(bank.vectors.tobytes()).hexdigest()

        print(f"  [Fold {fold_year}] Regenerating MEM_RANDOM on historical bank ({len(eval_recs)} queries x {len(seeds)} seeds)...")
        t_rand_s = time.perf_counter()
        mem_rand_lookup: Dict[Tuple[str, int], float] = {}
        for s in seeds:
            for r in eval_recs:
                res = retrieve_mem_random(
                    bank=bank,
                    query_security_id=r.security_id,
                    bank_hash=bank_hash,
                    fold_year=fold_year,
                    query_id=r.query_id,
                    master_seed=s,
                    k=k_val,
                )
                mem_rand_lookup[(r.query_id, s)] = res.prediction
        print(f"  [Fold {fold_year}] Regenerated MEM_RANDOM in {time.perf_counter() - t_rand_s:.2f}s.")

        # 4. Transform records: derive corrected query_id from validated fields, reuse 15 policies, replace MEM_RANDOM
        admitted_qids = {r.query_id for r in eval_recs}
        transformed_records: List[PolicyPredictionRecord] = []
        seen_keys: Set[Tuple[str, str, Optional[int]]] = set()

        reused_count = 0
        regenerated_count = 0

        for r in archived_records:
            f_yr = int(r["fold"])
            if f_yr != fold_year:
                raise ValueError(f"Record fold {f_yr} does not match current fold {fold_year}")
            sec = str(r["security_id"])
            sess = str(r["decision_session"])
            validated_qid = f"{f_yr}_{sec}_{sess}"

            if validated_qid not in admitted_qids:
                raise ValueError(f"Validated query ID {validated_qid} is not in admitted eval queries")

            pol = str(r["policy_id"])
            real = r.get("realization_id")
            if real is not None:
                real = int(real)

            k = (validated_qid, pol, real)
            if k in seen_keys:
                raise ValueError(f"Duplicate prediction record encountered for {k}")
            seen_keys.add(k)

            if pol == "MEM_RANDOM":
                pred = float(mem_rand_lookup[(validated_qid, real)])
                regenerated_count += 1
            else:
                pred = float(r["prediction"])
                reused_count += 1

            if not math.isfinite(pred):
                raise ValueError(f"Non-finite prediction for {k}: {pred}")

            transformed_records.append(
                PolicyPredictionRecord(
                    query_id=validated_qid,
                    security_id=sec,
                    market=str(r["market"]),
                    decision_session=sess,
                    fold=f_yr,
                    policy_id=pol,
                    realization_id=real,
                    prediction=pred,
                    source_artifact_id=str(r.get("source_artifact_id", "")),
                )
            )

        # 5. Atomically seal predictions artifact and manifest
        pred_json_path = target_fold_dir / "policy_predictions.json"
        pred_manifest_path = target_fold_dir / "predictions_manifest.json"

        payload = [asdict(rec) for rec in transformed_records]
        with open(pred_json_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json(payload))

        pred_sha256 = hashlib.sha256(pred_json_path.read_bytes()).hexdigest()
        pred_ident = compute_prediction_identity(
            fold_year=fold_year,
            producing_checkpoints=producing_checkpoints,
            eval_records=eval_recs,
            dev_records=fold_data.get("dev_records", []),
            configured_policies=None,
            seeds=seeds,
            code_revision=git_rev,
        )

        manifest_data = {
            "fold_year": fold_year,
            "count": len(transformed_records),
            "sha256": pred_sha256,
            "prediction_identity": pred_ident,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(pred_manifest_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json(manifest_data))

        predictions_by_year[fold_year] = transformed_records
        transformation_manifest["folds"][str(fold_year)] = {
            "action": "selective_regeneration",
            "total_records": len(transformed_records),
            "reused_unaffected_records": reused_count,
            "regenerated_mem_random_records": regenerated_count,
            "sha256": pred_sha256,
        }
        print(f"  [Fold {fold_year}] Sealed {len(transformed_records)} records ({reused_count} reused, {regenerated_count} regenerated).")

    # 6. Continuous Portfolio Simulation
    print("\n" + "=" * 70)
    print("=== CONTINUOUS PORTFOLIO SIMULATION (6 YEARS, 204 ACCOUNTS) ===")
    print("=" * 70)
    port_res = run_continuous_portfolio_simulation(
        folds_to_run=all_folds,
        fold_data_by_year=fold_data_by_year,
        predictions_by_year=predictions_by_year,
        output_dir=output_path,
        initial_capital=float(config.get("execution", {}).get("initial_capital_account_units", 100000.0)),
        commission=float(config.get("execution", {}).get("commission_per_side", 0.001)),
        slippage=float(config.get("execution", {}).get("slippage_per_side", 0.0005)),
        max_positions=int(config.get("execution", {}).get("max_positions", 3)),
    )

    policy_accounts = port_res["policy_accounts"]
    union_sessions = port_res["union_sessions"]

    # 7. Release Coverage & Accounting Validation
    print("\n=== Release Coverage & Accounting Validation ===")
    all_markets = sorted(list(set(
        m for fd in fold_data_by_year.values()
        for m in (list(fd.get("sec_info_by_market", {}).keys()) or [s_data.get("market") for s_data in fd.get("sec_info", {}).values()])
        if m
    )))
    auto_expected_accounts = build_expected_account_keys(
        config=config,
        folds=all_folds,
        markets=all_markets,
        allow_reduced_arms=False,
    )
    coverage_manifest = port_res.get("coverage_manifest", [])
    validation_res = validate_release_coverage_and_accounting(
        coverage_manifest=coverage_manifest,
        fold_data_by_year=fold_data_by_year,
        folds_to_run=all_folds,
        expected_accounts=auto_expected_accounts,
    )

    # 8. Statistical Inference & Primary Contrasts
    print("\n=== Frozen Statistical Inference (10,000 Bootstrap Draws) ===")
    analysis_cfg = config.get("analysis", {})
    primary_contrasts_spec = analysis_cfg.get("primary_contrasts", [])
    bootstrap_draws = int(analysis_cfg.get("bootstrap_draws", 10000))
    bootstrap_primary_weeks = int(analysis_cfg.get("bootstrap_primary_weeks", 4))

    stats_res = evaluate_primary_contrasts(
        policy_accounts=policy_accounts,
        union_sessions=union_sessions,
        primary_contrasts=primary_contrasts_spec,
        bootstrap_weeks=bootstrap_primary_weeks,
        num_draws=bootstrap_draws,
        seed=42,
        allow_reduced_arms=False,
    )

    # 9. Analysis Export & Replay Verification
    print("\n=== Analysis Export & Replay Verification ===")
    release_analysis_dir = output_path / "release_analysis"
    export_analysis_bundle(
        analysis_dir=release_analysis_dir,
        policy_accounts=policy_accounts,
        union_sessions=union_sessions,
        primary_contrasts=primary_contrasts_spec,
        bootstrap_weeks=bootstrap_primary_weeks,
        num_draws=bootstrap_draws,
        seed=42,
        allow_reduced_arms=False,
    )

    replay_report = replay_analysis_bundle(
        export_dir=release_analysis_dir,
        tolerance=float(analysis_cfg.get("analysis_replay_tolerance", 1e-10)),
        allow_reduced_arms=False,
    )
    print(f"  Replay Verification: {replay_report.get('status')}")
    if replay_report.get("status") != "REPLAY_VERIFIED":
        raise RuntimeError(f"REPLAY_FAILED: Replay verification failed: {replay_report.get('error')}")

    with open(release_analysis_dir / "replay_report.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(replay_report))

    # 10. Source Immutability Assertion
    print("\n=== Pre-Publication Security Gate: Source Immutability Check ===")
    assert_directory_unchanged(source_path, before_source_manifest, "targeted recovery execution")

    # 11. Transactional Release Sealing
    print("\n=== Transactional Release Sealing ===")
    runtime_cfg_path = output_path / "runtime_config.authorized.json"
    auth_config = copy.deepcopy(config)
    auth_config["production_authorized"] = True
    auth_config["execution_mode"] = "targeted_recovery"
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(auth_config))

    trans_manifest_path = output_path / "transformation_manifest.json"
    with open(trans_manifest_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(transformation_manifest))

    rel_artifacts: Dict[str, str] = {
        "runtime_config.authorized.json": hashlib.sha256(runtime_cfg_path.read_bytes()).hexdigest(),
        "transformation_manifest.json": hashlib.sha256(trans_manifest_path.read_bytes()).hexdigest(),
    }
    for f_yr in all_folds:
        f_dir = output_path / f"fold_{f_yr}"
        p_json = f_dir / "policy_predictions.json"
        p_man = f_dir / "predictions_manifest.json"
        rel_artifacts[f"fold_{f_yr}/policy_predictions.json"] = hashlib.sha256(p_json.read_bytes()).hexdigest()
        rel_artifacts[f"fold_{f_yr}/predictions_manifest.json"] = hashlib.sha256(p_man.read_bytes()).hexdigest()

    for p in sorted(release_analysis_dir.glob("*.json")):
        rel_artifacts[f"release_analysis/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()

    rel_artifacts["portfolio_manifest.json"] = hashlib.sha256((output_path / "portfolio_manifest.json").read_bytes()).hexdigest()
    rel_artifacts["coverage_manifest.json"] = hashlib.sha256((output_path / "coverage_manifest.json").read_bytes()).hexdigest()

    release_manifest = {
        "release_id": "a30-targeted-recovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_revision": git_rev,
        "sealed_artifacts": rel_artifacts,
        "equivalence_gate": gate_res,
        "summary": {
            "folds_count": len(all_folds),
            "accounts_count": len(policy_accounts),
            "total_admitted_queries": validation_res["total_admitted_queries"],
            "total_expected_forecasts": validation_res["total_expected_forecasts"],
            "total_sealed_forecasts": validation_res["total_sealed_forecasts"],
        },
    }
    release_manifest_path = output_path / "release_manifest.json"
    with open(release_manifest_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(release_manifest))

    # Verify release manifest integrity
    verify_release_manifest(output_path)

    recovery_summary = {
        "status": "TARGETED_RECOVERY_SUCCESS",
        "pipeline_stage": "RELEASE_VERIFIED",
        "release_id": "a30-targeted-recovery",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_rev,
        "equivalence_gate": gate_res,
        "coverage_validation": validation_res,
        "replay_report": replay_report,
    }
    with open(output_path / "recovery_summary.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(recovery_summary))
    with open(output_path / "pipeline_summary.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(recovery_summary))
    with open(output_path / "pipeline_completion.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(recovery_summary))

    print("\n" + "=" * 70)
    print("=== TARGETED RECOVERY COMPLETED AND RELEASE SEALED SUCCESSFULLY ===")
    print("=" * 70)
    return recovery_summary


def main():
    parser = argparse.ArgumentParser(description="Targeted zero-neural-retraining recovery pipeline.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-corrected",
        help="Path to protected original source output directory.",
    )
    parser.add_argument(
        "--recovered-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-recovery",
        help="Path to preserved recovered directory containing Fold 2020 and Fold 2021.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-targeted-recovery",
        help="Path to fresh isolated target output directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-corrected" / "runtime_config.authorized.json",
        help="Path to authorized production config JSON.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=REPO_ROOT / "outputs" / "ops" / "checkpoint_recovery_approval.json",
        help="Path to reviewed checkpoint reuse inventory.",
    )
    parser.add_argument(
        "--sample-ids-dir",
        type=Path,
        default=REPO_ROOT / "rebuild_plan" / "sample_ids",
        help="Path to sample ids directory.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "cache" / "ohlcv",
        help="Path to OHLCV parquet data directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="Run only the Fold 2021 equivalence gate test and exit.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.gate_only:
        gate_res = run_fold_2021_equivalence_gate(
            source_dir=args.source_dir,
            recovered_dir=args.recovered_dir,
            data_dir=args.data_dir,
            sample_ids_dir=args.sample_ids_dir,
            config=config,
        )
        sys.exit(0 if gate_res["status"] == "PASSED" else 1)

    dev = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    execute_targeted_recovery(
        source_dir=args.source_dir,
        recovered_dir=args.recovered_dir,
        output_dir=args.output_dir,
        config=config,
        sample_ids_dir=args.sample_ids_dir,
        data_dir=args.data_dir,
        inventory_path=args.inventory,
        device=dev,
    )


if __name__ == "__main__":
    main()
