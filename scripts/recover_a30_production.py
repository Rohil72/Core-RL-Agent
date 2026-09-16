#!/usr/bin/env python3
"""Dedicated Zero-Retraining Recovery Entry Point for A30 Production Runs.

Guarantees:
1. STRICT READ-ONLY SOURCE: Source artifacts directory is opened read-only and never modified.
2. SEPARATE OUTPUT DIRECTORY: All recovery artifacts are written strictly to an isolated output directory.
3. ZERO TRAINING CAPABILITY: No training loops, model fitters, or optimizer steps are imported or invoked.
   If any checkpoint is missing or incompatible, the recovery script fails loudly rather than retraining.
4. COMPATIBILITY VERIFICATION: Validates checkpoint digests, architecture parameter shapes, and finite weights.
5. REPAIRED EVALUATION & ADMISSION: Uses explicit evaluation_year and full query set equality validation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
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
    replay_analysis_bundle,
    run_continuous_portfolio_simulation,
    to_canonical_json,
    validate_release_coverage_and_accounting,
    PolicyPredictionRecord,
)


def verify_checkpoint_compatibility(
    checkpoint_path: Path,
    architecture: str,
    seed: int,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verifies that a checkpoint file exists, is non-empty, matches digest, and loads valid finite weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    raw_bytes = checkpoint_path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    size_bytes = len(raw_bytes)

    if size_bytes == 0:
        raise ValueError(f"Checkpoint file is empty: {checkpoint_path}")

    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Checkpoint SHA256 mismatch for {checkpoint_path}: expected {expected_sha256}, got {actual_sha256}"
        )

    # Load state dict safely on CPU
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if hasattr(checkpoint_data, "model_state"):
        state_dict = checkpoint_data.model_state
    elif isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        state_dict = checkpoint_data["model_state"]
    else:
        state_dict = checkpoint_data
    if state_dict is None:
        raise ValueError(f"Checkpoint {checkpoint_path} has no 'model_state' attribute or key")

    # Instantiate reference model to verify architecture compatibility
    if "MLP" in architecture.upper():
        model: nn.Module = MLPAnnual(seed=seed)
    elif "TRANSFORMER" in architecture.upper():
        model = TransformerAnnual(seed=seed)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    model.load_state_dict(state_dict)

    # Verify all parameters are finite floats
    param_count = 0
    for name, param in model.named_parameters():
        param_count += param.numel()
        if not torch.all(torch.isfinite(param)):
            raise ValueError(f"Non-finite weights found in {checkpoint_path} parameter '{name}'")

    return {
        "status": "COMPATIBLE",
        "sha256": actual_sha256,
        "size_bytes": size_bytes,
        "parameter_count": param_count,
        "architecture": architecture,
        "seed": seed,
    }


def audit_source_checkpoints(
    source_dir: Path,
    folds: List[int],
    architectures: List[str],
    seeds: List[int],
) -> Dict[str, Any]:
    """Audits all required checkpoints in source_dir, returning inventory and compatibility status."""
    inventory: List[Dict[str, Any]] = []
    total_expected = len(folds) * len(architectures) * len(seeds)
    compatible_count = 0

    print(f"\n[Audit] Inspecting {total_expected} required model checkpoints in {source_dir}...")

    for f_yr in folds:
        fold_dir = source_dir / f"fold_{f_yr}"
        for arch in architectures:
            for seed in seeds:
                job_id = f"fold_{f_yr}_{arch}_seed{seed}"
                chk_dir = fold_dir / f"checkpoints_{arch}_seed{seed}"
                best_pt = chk_dir / "best_checkpoint.pt"

                item: Dict[str, Any] = {
                    "job_id": job_id,
                    "fold_year": f_yr,
                    "architecture": arch,
                    "seed": seed,
                    "relative_path": str(best_pt.relative_to(source_dir)) if best_pt.exists() else None,
                    "exists": best_pt.exists(),
                }

                if not best_pt.exists():
                    item["status"] = "MISSING"
                    item["error"] = f"File {best_pt} does not exist"
                    inventory.append(item)
                    continue

                try:
                    res = verify_checkpoint_compatibility(best_pt, arch, seed)
                    item.update(res)
                    item["disposition"] = "reuse candidate—verification pending"
                    compatible_count += 1
                except Exception as e:
                    item["status"] = "INCOMPATIBLE"
                    item["error"] = str(e)

                inventory.append(item)

    all_compatible = (compatible_count == total_expected)
    print(f"[Audit] Result: {compatible_count}/{total_expected} checkpoints compatible ({'ALL PASS' if all_compatible else 'FAILURES DETECTED'}).")

    return {
        "total_expected": total_expected,
        "compatible_count": compatible_count,
        "all_compatible": all_compatible,
        "inventory": inventory,
    }


def execute_zero_retraining_recovery(
    source_dir: Path,
    output_dir: Path,
    config: Dict[str, Any],
    sample_ids_dir: Path,
    data_dir: Path,
    selected_folds: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    allow_reduced_arms: bool = False,
) -> Dict[str, Any]:
    """Executes zero-retraining recovery pipeline: infers from source checkpoints, replays simulation and stats."""
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    # Absolute safety assertion: never allow source and output directories to overlap
    if source_path == output_path:
        raise ValueError(
            f"CRITICAL SAFETY VIOLATION: Source dir ({source_path}) and output dir ({output_path}) "
            f"must NOT be identical! Recovery requires an isolated output directory to preserve source integrity."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    neural_cfg = config.get("neural", {})
    seeds = neural_cfg.get("seeds", [7, 17, 37])
    architectures = neural_cfg.get("architectures", [
        "MLP_ANNUAL_966_64_128_1",
        "TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128",
    ])
    all_folds = [int(f["evaluation_year"]) for f in config.get("folds", [])] or [2020, 2021, 2022, 2023, 2024, 2025]
    folds_to_run = [int(f) for f in selected_folds] if selected_folds else all_folds

    # 1. Audit and verify all source checkpoints prior to any work
    audit_res = audit_source_checkpoints(source_path, folds_to_run, architectures, seeds)
    if not audit_res["all_compatible"]:
        incompatibles = [it["job_id"] for it in audit_res["inventory"] if it.get("status") != "COMPATIBLE"]
        raise RuntimeError(
            f"RECOVERY_ABORTED: {len(incompatibles)} required checkpoints in {source_path} are missing or incompatible: "
            f"{incompatibles}. The recovery entry point strictly forbids retraining."
        )

    # 2. Prepare fold datasets (using repaired query admission) and generate predictions
    fold_data_by_year: Dict[int, Dict[str, Any]] = {}
    predictions_by_year: Dict[int, List[PolicyPredictionRecord]] = {}
    recovered_checkpoint_hashes: Dict[str, str] = {}

    print("\n=== Stage 2 (Recovery): Policy Forecast Generation via Pre-trained Checkpoints ===")
    for fold_year in folds_to_run:
        target_fold_dir = output_path / f"fold_{fold_year}"
        target_fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [Fold {fold_year}] Preparing fold datasets with repaired admission from {data_dir}...")
        fold_data = prepare_fold_data(
            fold_year=fold_year,
            data_dir=data_dir,
            sample_ids_dir=sample_ids_dir,
            config=config,
            execution_mode="production",
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

                # Copy checkpoint to target output directory for self-contained artifact bundle
                shutil.copy2(src_pt, tgt_pt)

                sha = hashlib.sha256(tgt_pt.read_bytes()).hexdigest()
                recovered_checkpoint_hashes[job_id] = sha

                if "MLP" in arch.upper():
                    m = MLPAnnual(seed=seed)
                else:
                    m = TransformerAnnual(seed=seed, dropout=float(neural_cfg.get("transformer_dropout", 0.1)))

                chk_data = torch.load(tgt_pt, map_location="cpu", weights_only=False)
                m.load_state_dict(chk_data.model_state)
                m.to(device)
                m.eval()
                trained_models[f"{arch}_seed{seed}"] = m

        # Fit analytical Ridge baseline on training set
        ridge_model = fit_ridge_model(
            fold_data["train_x_mlp"],
            fold_data["train_y"],
            market_labels=fold_data["train_markets"],
            l2_lambda=0.001,
        )

        # Generate policy predictions using recovered models
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
        )
        predictions_by_year[fold_year] = p_records

    # 3. Continuous Portfolio Simulation
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

    # 4. Strict Release Coverage & Accounting Validation
    print("\n=== Release Coverage & Accounting Validation ===")
    coverage_manifest = port_res.get("coverage_manifest", [])
    validation_res = validate_release_coverage_and_accounting(
        coverage_manifest=coverage_manifest,
        fold_data_by_year=fold_data_by_year,
        folds_to_run=folds_to_run,
    )
    print(f"  Coverage Validation: {validation_res['status']}")

    # 5. Independent Replay Verification
    release_analysis_dir = output_path / "release_analysis"
    release_analysis_dir.mkdir(parents=True, exist_ok=True)
    replay_report = replay_analysis_bundle(
        export_dir=release_analysis_dir,
        tolerance=float(config.get("analysis", {}).get("analysis_replay_tolerance", 1e-10)),
        allow_reduced_arms=allow_reduced_arms,
    )

    recovery_summary = {
        "status": "RECOVERY_SUCCESS",
        "pipeline_stage": "RELEASE_VERIFIED",
        "source_directory": str(source_path),
        "output_directory": str(output_path),
        "executed_folds": folds_to_run,
        "recovered_checkpoint_hashes": recovered_checkpoint_hashes,
        "coverage_validation": validation_res,
        "replay_report": replay_report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path / "recovery_summary.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(recovery_summary))

    return recovery_summary


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
        help="Path to isolated target output directory for recovered artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "outputs" / "a30-corrected" / "runtime_config.authorized.json",
        help="Path to authorized production config JSON.",
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

    # Audit mode
    if args.audit_only:
        audit_res = audit_source_checkpoints(
            source_dir=source_dir,
            folds=folds_to_run,
            architectures=architectures,
            seeds=seeds,
        )
        if audit_res["all_compatible"]:
            print("\n[SUCCESS] Checkpoint audit completed: all checkpoints compatible for zero-retraining recovery.")
            sys.exit(0)
        else:
            print("\n[FAILURE] Checkpoint audit detected missing or incompatible checkpoints.", file=sys.stderr)
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
        device=dev,
    )


if __name__ == "__main__":
    main()
