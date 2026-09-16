#!/usr/bin/env python3
"""Dedicated Zero-Retraining Recovery Entry Point for A30 Production Runs.

Guarantees:
1. STRICT READ-ONLY SOURCE: Source artifacts directory is opened read-only, never modified,
   and verified with pre/post-execution byte-level directory digests.
2. DESTINATION ISOLATION: All recovery artifacts are written strictly to an isolated, fresh,
   non-existent output directory. Existing output directories are strictly rejected.
3. ZERO TRAINING CAPABILITY: No training loops, backpropagation, or optimizer steps are imported or invoked.
   If any checkpoint is missing, unapproved, pending, or incompatible, the recovery aborts immediately.
4. TWO-TIER CHECKPOINT AUTHENTICATION:
   - Artifact Integrity: Validates byte-level SHA-256 digest against reviewed inventory BEFORE deserialization.
   - Scientific Compatibility: Verifies strict parameter names/shapes, finite floating-point weights,
     provenance (fold, seed, architecture), training identity, and records training vs recovery git revisions.
   - Dispositions: REUSE_APPROVED, REUSE_PENDING, REUSE_REJECTED. Recovery strictly requires REUSE_APPROVED.
5. REPAIRED EVALUATION & COMPLETE RECONCILIATION:
   - Evaluates models in model.eval(), torch.no_grad().
   - Fits Ridge baseline on training set, and trust gates/mixture weights on development set (never evaluation).
   - Validates accounting, exact query set equality, permitted exclusions, and expected account set equality.
6. STAGE ORDERING & INDEPENDENT REPLAY:
   - Corrected predictions -> continuous account ledgers -> validated coverage -> new daily returns ->
     primary inference -> exported analysis bundle -> replay comparison -> release sealing.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Explicitly ensure repo root is in Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.launch_a30_production import (
    MLPAnnual,
    TransformerAnnual,
    fit_ridge_model,
    generate_and_seal_policy_predictions,
    prepare_fold_data,
    run_continuous_portfolio_simulation,
    to_canonical_json,
    validate_release_coverage_and_accounting,
    PolicyPredictionRecord,
)
from memory_study_v2.inference import (
    evaluate_primary_contrasts,
    export_analysis_bundle,
    replay_analysis_bundle,
)


def get_current_git_revision() -> str:
    """Retrieve current committed git revision HEAD SHA without returning UNKNOWN."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )
        rev = res.stdout.strip()
        if rev:
            return rev
    except Exception:
        pass
    # Fallback to checking git head file if subprocess fails
    git_head = REPO_ROOT / ".git" / "HEAD"
    if git_head.exists():
        head_ref = git_head.read_text(encoding="utf-8").strip()
        if head_ref.startswith("ref:"):
            ref_path = REPO_ROOT / ".git" / head_ref.split()[1]
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
        else:
            return head_ref
    raise RuntimeError("Unable to determine current committed git revision via git rev-parse HEAD.")


def compute_directory_manifest(dir_path: Path) -> Dict[str, str]:
    """Compute relative path -> sha256 digest mapping for all files in a directory."""
    manifest: Dict[str, str] = {}
    if not dir_path.exists():
        return manifest
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(dir_path))
            manifest[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return manifest


def assert_directory_unchanged(dir_path: Path, before_manifest: Dict[str, str], stage_name: str = ""):
    """Assert directory contents are bit-for-bit identical to before_manifest."""
    after_manifest = compute_directory_manifest(dir_path)
    if before_manifest != after_manifest:
        added = set(after_manifest.keys()) - set(before_manifest.keys())
        removed = set(before_manifest.keys()) - set(after_manifest.keys())
        modified = [k for k in before_manifest if k in after_manifest and before_manifest[k] != after_manifest[k]]
        raise RuntimeError(
            f"CRITICAL SAFETY VIOLATION: Source directory {dir_path} was modified during {stage_name}! "
            f"Added files: {added}, Removed files: {removed}, Modified files: {modified}."
        )


def verify_checkpoint_identity(
    checkpoint_path: Path,
    fold_year: int,
    architecture: str,
    seed: int,
    expected_sha256: Optional[str] = None,
    provenance_dir: Optional[Path] = None,
    current_code_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Verifies checkpoint artifact integrity, scientific compatibility, and provenance.

    Returns a dict with disposition:
      - REUSE_APPROVED: required integrity, architecture, weights, and provenance established.
      - REUSE_PENDING: expected digest or required provenance receipt missing/inconclusive.
      - REUSE_REJECTED: explicit incompatibility demonstrated (digest mismatch, non-finite weights,
        parameter shape mismatch, or identity mismatch).
    """
    result: Dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "fold_year": fold_year,
        "architecture": architecture,
        "seed": seed,
        "disposition": "REUSE_REJECTED",
        "integrity_verified": False,
        "compatibility_verified": False,
        "provenance_verified": False,
        "reasons": [],
    }

    # 1. Require expected digest from reviewed inventory or trusted receipt
    if not expected_sha256:
        result["disposition"] = "REUSE_PENDING"
        result["reasons"].append("No expected digest provided from reviewed inventory or trusted receipt")
        return result

    # 2. Check file existence and non-empty
    if not checkpoint_path.exists():
        result["disposition"] = "REUSE_REJECTED"
        result["reasons"].append(f"Checkpoint file does not exist: {checkpoint_path}")
        return result

    raw_bytes = checkpoint_path.read_bytes()
    size_bytes = len(raw_bytes)
    result["size_bytes"] = size_bytes
    if size_bytes == 0:
        result["disposition"] = "REUSE_REJECTED"
        result["reasons"].append(f"Checkpoint file is empty (0 bytes): {checkpoint_path}")
        return result

    # 3. Verify bytes against expected digest BEFORE deserializing
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    result["actual_sha256"] = actual_sha256
    result["expected_sha256"] = expected_sha256
    if actual_sha256 != expected_sha256:
        result["disposition"] = "REUSE_REJECTED"
        result["reasons"].append(
            f"Digest mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
        return result
    result["integrity_verified"] = True

    # 4. Deserialization & Architecture Parameter Compatibility
    try:
        chk_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if hasattr(chk_data, "model_state"):
            state_dict = chk_data.model_state
        elif isinstance(chk_data, dict) and "model_state" in chk_data:
            state_dict = chk_data["model_state"]
        else:
            state_dict = chk_data
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} state dict is not a dictionary")
    except Exception as e:
        result["disposition"] = "REUSE_REJECTED"
        result["reasons"].append(f"Deserialization failure: {e}")
        return result

    # Reference architecture instantiation
    try:
        if "MLP" in architecture.upper():
            ref_model: nn.Module = MLPAnnual(seed=seed)
        elif "TRANSFORMER" in architecture.upper():
            ref_model = TransformerAnnual(seed=seed)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        ref_state = ref_model.state_dict()
        ref_keys = set(ref_state.keys())
        chk_keys = set(state_dict.keys())

        # Strict parameter key match: disallow strict=False or discarding parameters
        if ref_keys != chk_keys:
            missing_keys = ref_keys - chk_keys
            unexpected_keys = chk_keys - ref_keys
            raise ValueError(
                f"Parameter keys mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
            )

        # Check shapes and finite values across parameters and buffers
        param_count = 0
        for name, ref_tensor in ref_state.items():
            chk_tensor = state_dict[name]
            if chk_tensor.shape != ref_tensor.shape:
                raise ValueError(
                    f"Parameter shape mismatch for '{name}': expected {ref_tensor.shape}, got {chk_tensor.shape}"
                )
            if not torch.all(torch.isfinite(chk_tensor)):
                raise ValueError(f"Non-finite weights found in tensor '{name}'")
            param_count += chk_tensor.numel()

        # Load with strict=True
        ref_model.load_state_dict(state_dict, strict=True)
        result["compatibility_verified"] = True
        result["parameter_count"] = param_count
    except Exception as e:
        result["disposition"] = "REUSE_REJECTED"
        result["reasons"].append(f"Architecture compatibility failure: {e}")
        return result

    # 5. Provenance verification from job_completion.json / training_identity.json
    prov_dir = provenance_dir or checkpoint_path.parent
    job_comp_file = prov_dir / "job_completion.json"
    if not job_comp_file.exists():
        result["disposition"] = "REUSE_PENDING"
        result["reasons"].append(f"Missing provenance file: {job_comp_file}")
        return result

    try:
        job_comp = json.loads(job_comp_file.read_text(encoding="utf-8"))
        prov_fold = job_comp.get("fold_year")
        prov_seed = job_comp.get("seed")
        prov_arch = job_comp.get("architecture")
        if prov_fold != fold_year or prov_seed != seed or prov_arch != architecture:
            result["disposition"] = "REUSE_REJECTED"
            result["reasons"].append(
                f"Provenance identity mismatch: fold={prov_fold} (expected {fold_year}), "
                f"seed={prov_seed} (expected {seed}), arch={prov_arch} (expected {architecture})"
            )
            return result

        training_ident = job_comp.get("training_identity", {})
        result["training_revision"] = training_ident.get("code_revision", "NOT_RECORDED")
        result["recovery_revision"] = current_code_revision
        result["job_identity_sha256"] = training_ident.get("job_identity_sha256")
        result["neural_configuration_sha256"] = training_ident.get("neural_configuration_sha256")
        result["train_sample_ids_sha256"] = training_ident.get("train_sample_ids_sha256")
        result["provenance_verified"] = True
    except Exception as e:
        result["disposition"] = "REUSE_REJECTED"
        result["reasons"].append(f"Provenance parsing failure: {e}")
        return result

    # 6. Approved disposition
    result["disposition"] = "REUSE_APPROVED"
    return result


# Backwards compatibility alias
verify_checkpoint_compatibility = verify_checkpoint_identity


def audit_source_checkpoints(
    source_dir: Path,
    folds: List[int],
    architectures: List[str],
    seeds: List[int],
    inventory_path: Optional[Path] = None,
    current_code_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Audits all required checkpoints in source_dir against reviewed inventory and provenance receipts."""
    inventory: List[Dict[str, Any]] = []
    total_expected = len(folds) * len(architectures) * len(seeds)
    approved_count = 0
    pending_count = 0
    rejected_count = 0

    # Load inventory if provided or if default exists
    expected_digests: Dict[str, str] = {}
    if inventory_path and Path(inventory_path).exists():
        inv_data = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
        if isinstance(inv_data, list):
            for item in inv_data:
                jid = item.get("job_id")
                sha = item.get("sha256")
                if jid and sha:
                    expected_digests[jid] = sha
    elif (REPO_ROOT / "outputs" / "ops" / "checkpoint_reuse_inventory.json").exists():
        inv_data = json.loads((REPO_ROOT / "outputs" / "ops" / "checkpoint_reuse_inventory.json").read_text(encoding="utf-8"))
        if isinstance(inv_data, list):
            for item in inv_data:
                jid = item.get("job_id")
                sha = item.get("sha256")
                if jid and sha:
                    expected_digests[jid] = sha

    print(f"\n[Audit] Inspecting {total_expected} required model checkpoints in {source_dir}...")

    for f_yr in folds:
        fold_dir = source_dir / f"fold_{f_yr}"
        for arch in architectures:
            for seed in seeds:
                job_id = f"fold_{f_yr}_{arch}_seed{seed}"
                chk_dir = fold_dir / f"checkpoints_{arch}_seed{seed}"
                best_pt = chk_dir / "best_checkpoint.pt"

                # Expected digest from inventory or original receipt
                expected_sha = expected_digests.get(job_id)
                if not expected_sha and (chk_dir / "job_completion.json").exists():
                    try:
                        jc = json.loads((chk_dir / "job_completion.json").read_text(encoding="utf-8"))
                        expected_sha = jc.get("artifacts", {}).get("best_checkpoint", {}).get("sha256")
                    except Exception:
                        pass

                item = verify_checkpoint_identity(
                    checkpoint_path=best_pt,
                    fold_year=f_yr,
                    architecture=arch,
                    seed=seed,
                    expected_sha256=expected_sha,
                    provenance_dir=chk_dir,
                    current_code_revision=current_code_revision,
                )
                item["job_id"] = job_id
                item["relative_path"] = str(best_pt.relative_to(source_dir)) if best_pt.exists() else None

                disp = item["disposition"]
                if disp == "REUSE_APPROVED":
                    approved_count += 1
                elif disp == "REUSE_PENDING":
                    pending_count += 1
                else:
                    rejected_count += 1

                inventory.append(item)

    all_approved = (approved_count == total_expected and pending_count == 0 and rejected_count == 0)
    print(
        f"[Audit] Result: {approved_count}/{total_expected} REUSE_APPROVED, "
        f"{pending_count} REUSE_PENDING, {rejected_count} REUSE_REJECTED "
        f"({'ALL APPROVED' if all_approved else 'FAILURES/PENDING DETECTED'})."
    )

    return {
        "total_expected": total_expected,
        "approved_count": approved_count,
        "pending_count": pending_count,
        "rejected_count": rejected_count,
        "all_approved": all_approved,
        "inventory": inventory,
    }


def assert_safety_isolation(
    source_path: Path,
    output_path: Path,
    data_path: Path,
    sample_ids_path: Path,
):
    """Enforces strict path isolation, non-overlap, and fresh non-existent output directory."""
    source_res = source_path.resolve()
    output_res = output_path.resolve()
    data_res = data_path.resolve()
    sample_res = sample_ids_path.resolve()

    if not source_res.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_res}")

    if source_res == output_res:
        raise ValueError(
            f"CRITICAL SAFETY VIOLATION: Source dir ({source_res}) and output dir ({output_res}) "
            f"must NOT be identical! Recovery requires an isolated output directory to preserve source integrity."
        )

    # Reject either directory being inside the other
    try:
        output_res.relative_to(source_res)
        raise ValueError(
            f"CRITICAL SAFETY VIOLATION: Output directory {output_res} is located inside source directory {source_res}!"
        )
    except ValueError as e:
        if "CRITICAL SAFETY VIOLATION" in str(e):
            raise

    try:
        source_res.relative_to(output_res)
        raise ValueError(
            f"CRITICAL SAFETY VIOLATION: Source directory {source_res} is located inside output directory {output_res}!"
        )
    except ValueError as e:
        if "CRITICAL SAFETY VIOLATION" in str(e):
            raise

    # Reject if output directory already exists or is a symlink (Fresh directory required)
    if output_path.is_symlink() or output_res.exists():
        raise ValueError(
            f"CRITICAL SAFETY VIOLATION: Output directory {output_res} already exists or is a symlink! "
            f"Recovery strictly requires a fresh, nonexistent output directory. "
            f"Automatic resume or overwriting is strictly disallowed."
        )

    # Reject output path pointing into protected data or sample IDs
    for protected in [data_res, sample_res]:
        try:
            output_res.relative_to(protected)
            raise ValueError(
                f"CRITICAL SAFETY VIOLATION: Output directory {output_res} redirects inside protected path {protected}!"
            )
        except ValueError as e:
            if "CRITICAL SAFETY VIOLATION" in str(e):
                raise


def execute_zero_retraining_recovery(
    source_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    sample_ids_dir: Path,
    data_dir: Path,
    selected_folds: Optional[List[int]] = None,
    inventory_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    allow_reduced_arms: bool = False,
    selected_securities: Optional[List[str]] = None,
    custom_predictions: Optional[Dict[str, float]] = None,
    expected_accounts: Optional[Set[Tuple[int, str, str, Optional[int]]]] = None,
    custom_calendars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Executes zero-retraining recovery pipeline: infers from source checkpoints, replays simulation and stats."""
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()
    data_path = Path(data_dir).resolve()
    sample_ids_path = Path(sample_ids_dir).resolve()

    # 1. Enforce safety isolation before any action
    assert_safety_isolation(source_path, output_path, data_path, sample_ids_path)

    # Capture source directory manifest before execution for byte-level immutability guarantee
    before_source_manifest = compute_directory_manifest(source_path)

    try:
        git_rev = get_current_git_revision()
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        neural_cfg = config.get("neural", {})
        seeds = neural_cfg.get("seeds", [7, 17, 37])
        architectures = neural_cfg.get("architectures", [
            "MLP_ANNUAL_966_64_128_1",
            "TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128",
        ])
        all_folds = [int(f["evaluation_year"]) for f in config.get("folds", [])] or [2020, 2021, 2022, 2023, 2024, 2025]
        folds_to_run = [int(f) for f in selected_folds] if selected_folds else all_folds
        active_securities = selected_securities or config.get("selected_securities")

        # 2. Audit and authenticate all source checkpoints prior to doing any work
        audit_res = audit_source_checkpoints(
            source_dir=source_path,
            folds=folds_to_run,
            architectures=architectures,
            seeds=seeds,
            inventory_path=inventory_path,
            current_code_revision=git_rev,
        )
        if not audit_res["all_approved"]:
            unapproved = [
                f"{it['job_id']} ({it['disposition']}: {'; '.join(it['reasons'])})"
                for it in audit_res["inventory"]
                if it.get("disposition") != "REUSE_APPROVED"
            ]
            raise RuntimeError(
                f"RECOVERY_ABORTED: {len(unapproved)} required checkpoints in {source_path} are not approved for reuse: "
                f"{unapproved}. Production recovery strictly refuses pending or rejected checkpoints. Zero retraining is permitted."
            )

        # Create fresh output directory after preflight and audit pass
        output_path.mkdir(parents=True, exist_ok=False)

        # 3. Prepare fold datasets (using repaired query admission) and generate predictions
        fold_data_by_year: Dict[int, Dict[str, Any]] = {}
        predictions_by_year: Dict[int, List[PolicyPredictionRecord]] = {}
        recovered_checkpoint_hashes: Dict[str, str] = {}

        print("\n=== Stage 2 (Recovery): Policy Forecast Generation via Pre-trained Checkpoints ===")
        for fold_year in folds_to_run:
            target_fold_dir = output_path / f"fold_{fold_year}"
            target_fold_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [Fold {fold_year}] Preparing fold datasets with repaired admission from {data_path}...")
            fold_data = prepare_fold_data(
                fold_year=fold_year,
                data_dir=data_path,
                sample_ids_dir=sample_ids_path,
                config=config,
                selected_securities=active_securities,
                custom_calendars=custom_calendars,
                execution_mode="production" if not allow_reduced_arms else "pilot",
            )
            fold_data_by_year[fold_year] = fold_data

            # Load trained neural backbones from source checkpoints
            trained_models: Dict[str, nn.Module] = {}
            for arch in architectures:
                for seed in seeds:
                    job_id = f"fold_{fold_year}_{arch}_seed{seed}"
                    src_pt = source_path / f"fold_{fold_year}" / f"checkpoints_{arch}_seed{seed}" / "best_checkpoint.pt"
                    tgt_chk_dir = target_fold_dir / f"checkpoints_{arch}_seed{seed}"
                    tgt_chk_dir.mkdir(parents=True, exist_ok=True)
                    tgt_pt = tgt_chk_dir / "best_checkpoint.pt"

                    # Copy checkpoint to target output directory for self-contained artifact bundle (never hard-link)
                    shutil.copy2(src_pt, tgt_pt)

                    sha = hashlib.sha256(tgt_pt.read_bytes()).hexdigest()
                    recovered_checkpoint_hashes[job_id] = sha

                    if "MLP" in arch.upper():
                        m = MLPAnnual(seed=seed)
                    else:
                        m = TransformerAnnual(seed=seed, dropout=float(neural_cfg.get("transformer_dropout", 0.1)))

                    chk_data = torch.load(tgt_pt, map_location="cpu", weights_only=False)
                    if hasattr(chk_data, "model_state"):
                        state_dict = chk_data.model_state
                    elif isinstance(chk_data, dict) and "model_state" in chk_data:
                        state_dict = chk_data["model_state"]
                    else:
                        state_dict = chk_data
                    m.load_state_dict(state_dict, strict=True)
                    m.to(device)
                    m.eval()
                    trained_models[f"{arch}_seed{seed}"] = m

            # Fit analytical Ridge baseline on historical training set (designated training data, never evaluation)
            ridge_model = fit_ridge_model(
                fold_data["train_x_mlp"],
                fold_data["train_y"],
                market_labels=fold_data["train_markets"],
                l2_lambda=0.001,
            )

            # Generate and seal policy predictions using recovered models and historical dev set for mixtures/gates
            p_records = generate_and_seal_policy_predictions(
                fold_year=fold_year,
                fold_dir=target_fold_dir,
                fold_data=fold_data,
                trained_models=trained_models,
                ridge_model=ridge_model,
                device=device,
                config=config,
                seeds=seeds,
                checkpoint_hashes=recovered_checkpoint_hashes,
                custom_predictions=custom_predictions,
                code_revision=git_rev,
            )
            predictions_by_year[fold_year] = p_records

        # 4. Continuous Portfolio Simulation
        print("\n=== Stage 3 (Recovery): Continuous Portfolio Simulation & Ledger Execution ===")
        port_res = run_continuous_portfolio_simulation(
            folds_to_run=folds_to_run,
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
        aligned_returns = port_res["aligned_returns"]
        market_open_mask = port_res["market_open_mask"]

        # 5. Strict Release Coverage & Accounting Validation
        print("\n=== Stage 4 (Recovery): Release Coverage & Accounting Validation ===")
        coverage_manifest = port_res.get("coverage_manifest", [])
        validation_res = validate_release_coverage_and_accounting(
            coverage_manifest=coverage_manifest,
            fold_data_by_year=fold_data_by_year,
            folds_to_run=folds_to_run,
            expected_accounts=expected_accounts,
        )
        print(f"  Coverage Validation: {validation_res['status']}")

        # 6. Statistical Inference & Primary Contrasts
        print("\n=== Stage 5 (Recovery): Statistical Inference & Primary Contrasts ===")
        analysis_cfg = config.get("analysis", {})
        unique_markets = sorted(list(set(k[1] for k in policy_accounts.keys())))

        if not allow_reduced_arms:
            expected_markets = {"Brazil", "China", "France", "India", "UK", "US"}
            actual_markets = set(unique_markets)
            missing_m = expected_markets - actual_markets
            if missing_m:
                raise ValueError(f"Full production requires complete market coverage. Missing: {missing_m}")
            if len(policy_accounts) < 204:
                raise ValueError(f"Full production requires 204 continuous market paths. Found: {len(policy_accounts)}")

        bootstrap_draws = int(analysis_cfg.get("bootstrap_draws", 100))
        contrast_results, draw_matrix, sampled_weeks = evaluate_primary_contrasts(
            returns_by_arm_market_realization=aligned_returns,
            markets=unique_markets,
            session_dates=union_sessions,
            valid_mask=market_open_mask,
            allow_reduced_arms=allow_reduced_arms,
            num_draws=bootstrap_draws,
            seed=42,
        )

        # 7. Export Analysis Bundle (starts fresh)
        print("\n=== Stage 6 (Recovery): Export Analysis Bundle ===")
        release_analysis_dir = output_path / "release_analysis"
        if release_analysis_dir.exists():
            shutil.rmtree(release_analysis_dir)
        release_analysis_dir.mkdir(parents=True, exist_ok=True)

        export_analysis_bundle(
            returns_by_key={f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in aligned_returns.items()},
            contrast_results=contrast_results,
            draw_matrix=draw_matrix,
            draw_week_indices=sampled_weeks,
            export_dir=release_analysis_dir,
            session_dates=union_sessions,
            valid_mask=market_open_mask,
            markets=unique_markets,
            num_draws=bootstrap_draws,
            seed=42,
            allow_reduced_arms=allow_reduced_arms,
        )

        # 8. Independent Replay Verification
        print("\n=== Stage 7 (Recovery): Independent Replay Verification ===")
        replay_report = replay_analysis_bundle(
            export_dir=release_analysis_dir,
            tolerance=float(analysis_cfg.get("analysis_replay_tolerance", 1e-10)),
            allow_reduced_arms=allow_reduced_arms,
        )
        print(f"  Replay Verification: {replay_report.get('status')}")
        if replay_report.get("status") != "REPLAY_VERIFIED":
            raise RuntimeError(f"REPLAY_FAILED: Replay verification failed: {replay_report.get('error')}")

        with open(release_analysis_dir / "replay_report.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(replay_report))

        # 9. Release Sealing & Completion Receipt
        print("\n=== Stage 8 (Recovery): Release Sealing & Completion Receipt ===")
        total_admitted_queries = validation_res["total_admitted_queries"]
        total_expected_forecasts = validation_res["total_expected_forecasts"]
        total_sealed_forecasts = validation_res["total_sealed_forecasts"]
        missing_forecasts = total_expected_forecasts - total_sealed_forecasts

        recovery_summary = {
            "status": "RECOVERY_SUCCESS",
            "pipeline_stage": "RELEASE_VERIFIED",
            "source_directory": str(source_path),
            "output_directory": str(output_path),
            "code_revision": git_rev,
            "executed_folds": folds_to_run,
            "recovered_checkpoint_hashes": recovered_checkpoint_hashes,
            "coverage_summary": {
                "total_admitted_queries": total_admitted_queries,
                "total_expected_forecasts": total_expected_forecasts,
                "total_sealed_forecasts": total_sealed_forecasts,
                "missing_forecasts": missing_forecasts,
            },
            "coverage_validation": validation_res,
            "replay_report": replay_report,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        with open(output_path / "recovery_summary.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(recovery_summary))
        with open(output_path / "pipeline_summary.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(recovery_summary))
        with open(output_path / "recovery_completion.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(recovery_summary))

        runtime_cfg_path = output_path / "runtime_config.authorized.json"
        auth_config = copy.deepcopy(config)
        auth_config["production_authorized"] = True
        auth_config["execution_mode"] = "recovery"
        with open(runtime_cfg_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json(auth_config))

        cov_path = output_path / "coverage_manifest.json"
        rel_artifacts = {
            "recovery_summary.json": hashlib.sha256((output_path / "recovery_summary.json").read_bytes()).hexdigest(),
            "pipeline_summary.json": hashlib.sha256((output_path / "pipeline_summary.json").read_bytes()).hexdigest(),
            "recovery_completion.json": hashlib.sha256((output_path / "recovery_completion.json").read_bytes()).hexdigest(),
            "runtime_config.authorized.json": hashlib.sha256(runtime_cfg_path.read_bytes()).hexdigest(),
        }
        if cov_path.exists():
            rel_artifacts["coverage_manifest.json"] = hashlib.sha256(cov_path.read_bytes()).hexdigest()

        with open(output_path / "release_manifest.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json({
                "status": "RELEASE_VERIFIED",
                "code_revision": git_rev,
                "artifacts": rel_artifacts,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }))

        # Final assertion of source directory immutability
        assert_directory_unchanged(source_path, before_source_manifest, "zero-retraining recovery execution")
        print("\n[SUCCESS] Recovery pipeline completed and verified cleanly.")
        return recovery_summary

    finally:
        # Guarantee source directory immutability even on unhandled failure
        assert_directory_unchanged(source_path, before_source_manifest, "recovery preflight / execution teardown")


def main():
    parser = argparse.ArgumentParser(
        description="Dedicated Zero-Retraining Recovery Entry Point for A30 Production Runs."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-corrected",
        help="Path to source production run containing pre-trained best_checkpoint.pt files (read-only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-recovery",
        help="Path to isolated target output directory for recovered artifacts (must not exist).",
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
        default=REPO_ROOT / "outputs" / "ops" / "checkpoint_reuse_inventory.json",
        help="Path to reviewed machine-readable checkpoint reuse inventory.",
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
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Optional subset of folds to process (e.g. 2020 2021).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Perform non-destructive checkpoint compatibility audit and inventory without running recovery simulation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--allow-reduced-arms",
        action="store_true",
        help="Allow running with reduced arms/markets for small tests and fixtures.",
    )

    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    if not args.config.exists():
        print(f"Error: Config file {args.config} does not exist.", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    neural_cfg = config.get("neural", {})
    seeds = neural_cfg.get("seeds", [7, 17, 37])
    architectures = neural_cfg.get("architectures", [
        "MLP_ANNUAL_966_64_128_1",
        "TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128",
    ])
    all_folds = [int(f["evaluation_year"]) for f in config.get("folds", [])] or [2020, 2021, 2022, 2023, 2024, 2025]
    folds_to_run = args.folds if args.folds else all_folds

    git_rev = get_current_git_revision()

    # Audit mode
    if args.audit_only:
        audit_res = audit_source_checkpoints(
            source_dir=source_dir,
            folds=folds_to_run,
            architectures=architectures,
            seeds=seeds,
            inventory_path=args.inventory,
            current_code_revision=git_rev,
        )
        if audit_res["all_approved"]:
            print("\n[SUCCESS] Checkpoint audit completed: all checkpoints approved for zero-retraining recovery.")
            sys.exit(0)
        else:
            print("\n[FAILURE] Checkpoint audit detected unapproved, pending, or incompatible checkpoints.", file=sys.stderr)
            sys.exit(1)

    # Recovery mode
    dev = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    execute_zero_retraining_recovery(
        source_dir=source_dir,
        output_dir=output_dir,
        config=config,
        sample_ids_dir=args.sample_ids_dir,
        data_dir=args.data_dir,
        selected_folds=folds_to_run,
        inventory_path=args.inventory,
        device=dev,
        allow_reduced_arms=args.allow_reduced_arms,
    )


if __name__ == "__main__":
    main()
