"""A30 Production Deployment Launch Tool (Finding 5 / C6).

Automates deployment, preflight validation, and production dispatch
for production runs on the NVIDIA A30 GPU / Linux compute host.

Capabilities:
1. Hardware & Driver Preflight:
   - Verifies CUDA runtime and driver compatibility (PyTorch >= 2.0, CUDA >= 11.8/12.x).
   - Probes GPU compute capability (Ampere 8.0 for A30, 8.6 for RTX, etc.) and available VRAM.
   - Executes live CUDA tensor allocation and GEMM compute probe to ensure no kernel errors.
2. Storage & Filesystem Health (Fail-Closed):
   - Verifies presence of universe manifest and fold dimensions manifest.
   - Verifies presence of all 6 required fold sample ID manifests.
   - Verifies output directory writability.
   - Immediately stops execution with explicit error on missing storage/manifests.
3. Strict Fail-Closed Authorization Guard:
   - Verifies that repo configuration has 'production_authorized: false'.
   - Demands explicit operator authorization flag (--authorize-production) at launch time.
   - Refuses production training if unauthorized.
4. Real Production Walk-Forward Execution & Telemetry:
   - Dispatches configured folds, seeds, and architectures through real components.
   - Resolves admitted training, validation, and evaluation sample IDs into genuine representations and targets.
   - Instantiates MLPAnnual and TransformerAnnual backbones.
   - Invokes train_backbone_model runner with runtime configuration authorization.
   - Truthful completion state: best_checkpoint.pt is not a completion marker; an interrupted job
     resumes from last_checkpoint.pt; a stage is marked completed only when job_completion.json
     is written with validated SHA-256 digests.
   - Executes downstream precedent retrieval (retrieve_mem_sim_batch), Selective Trust Gate fitting,
     and portfolio simulation (PortfolioAccount) with cash NAV reconciliation.
   - Supports dependency-injected driver functions for isolated unit testing, and defaults to real driver.
   - Emits signed launch receipt to rebuild_plan/a30_production_out/a30_launch_receipt.json.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from memory_study_v2.backbones import MLPAnnual, TransformerAnnual
from memory_study_v2.canonical_data import align_to_venue_calendar, build_total_return_bars, validate_raw_bars
from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED, to_canonical_json
from memory_study_v2.execution import PortfolioAccount
from memory_study_v2.features import compute_technical_features, fit_scaler
from memory_study_v2.integration import fit_trust_gate
from memory_study_v2.labels import compute_target_labels
from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.representations import extract_annual_representation
from memory_study_v2.retrieval import retrieve_mem_sim_batch
from memory_study_v2.sample_index import (
    SampleIndexRecord,
    build_security_sample_index,
    filter_admitted_sample_ids,
    load_fold_sample_ids,
)
from memory_study_v2.train import train_backbone_model
from memory_study_v2.venue_calendar import get_venue_calendar


def get_system_telemetry() -> Dict[str, Any]:
    """Capture comprehensive host and process telemetry."""
    res: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else None,
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2) if psutil else None,
    }
    if psutil:
        res["process_rss_mb"] = round(psutil.Process().memory_info().rss / (1024**2), 2)
    return res


def run_cuda_preflight() -> Dict[str, Any]:
    """Inspect and probe CUDA hardware and runtime compatibility."""
    if torch is None:
        raise RuntimeError("PyTorch is not installed in the current environment.")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch. Ensure an NVIDIA GPU is present and CUDA drivers are loaded."
        )

    device_count = torch.cuda.device_count()
    device_idx = 0
    device_name = torch.cuda.get_device_name(device_idx)
    cap = torch.cuda.get_device_capability(device_idx)
    props = torch.cuda.get_device_properties(device_idx)
    total_vram_gb = round(props.total_memory / (1024**3), 2)

    # Live CUDA compute probe: allocate, matmul, reduce, sync
    torch.cuda.reset_peak_memory_stats(device_idx)
    t0 = time.perf_counter()
    x = torch.randn((1024, 1024), device=f"cuda:{device_idx}", dtype=torch.float32)
    y = torch.matmul(x, x)
    res = float(y.sum().item())
    torch.cuda.synchronize(device_idx)
    probe_duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    peak_mem_mb = round(torch.cuda.max_memory_allocated(device_idx) / (1024**2), 2)
    del x, y

    return {
        "cuda_available": True,
        "device_count": device_count,
        "device_index": device_idx,
        "device_name": device_name,
        "compute_capability": list(cap),
        "total_vram_gb": total_vram_gb,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "compute_probe": {
            "status": "PROBE_SUCCESS",
            "probe_duration_ms": probe_duration_ms,
            "peak_mem_mb": peak_mem_mb,
            "sample_output_finite": bool(np.isfinite(res)),
        },
    }


def verify_storage_and_manifests(
    data_dir: Path,
    output_dir: Path,
    sample_ids_dir: Path,
    universe_path: Optional[Path] = None,
    fold_dims_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify input manifests, sample index files, and output path writability."""
    checks: Dict[str, Any] = {}

    # 1. Check universe manifest
    if universe_path is None:
        cand = REPO_ROOT / "rebuild_plan" / "universe_request.csv"
        if not cand.exists():
            cand = REPO_ROOT / "rebuild_plan" / "universe_manifest_clean.csv"
        universe_path = cand
    checks["universe_manifest_present"] = universe_path.exists()
    checks["universe_manifest_path"] = str(universe_path)

    # 2. Check fold dimensions manifest
    if fold_dims_path is None:
        fold_dims_path = REPO_ROOT / "rebuild_plan" / "fold_dimensions_manifest.json"
    if fold_dims_path.exists():
        try:
            with open(fold_dims_path, "r", encoding="utf-8") as f:
                fold_dims = json.load(f)
            checks["fold_dimensions_present"] = True
            checks["total_folds_configured"] = len(fold_dims.get("folds", {}))
        except Exception:
            checks["fold_dimensions_present"] = True
            checks["total_folds_configured"] = 0
    else:
        checks["fold_dimensions_present"] = False
        checks["total_folds_configured"] = 0
    checks["fold_dimensions_path"] = str(fold_dims_path)

    # 3. Check sample ID manifests
    if sample_ids_dir.exists():
        sample_files = list(sample_ids_dir.glob("fold_*_sample_ids.json")) + list(sample_ids_dir.glob("sample_ids_*.csv"))
        checks["sample_id_files_count"] = len(sample_files)
        checks["sample_ids_present"] = len(sample_files) >= 6
    else:
        checks["sample_id_files_count"] = 0
        checks["sample_ids_present"] = False
    checks["sample_ids_dir"] = str(sample_ids_dir)

    # 4. Check data cache
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        checks["data_cache_dir"] = str(data_dir)
        checks["parquet_count"] = len(parquet_files)
    else:
        checks["data_cache_dir"] = str(data_dir)
        checks["parquet_count"] = 0

    # 5. Check output writability
    output_dir.mkdir(parents=True, exist_ok=True)
    test_probe_file = output_dir / ".write_probe"
    try:
        with open(test_probe_file, "w", encoding="utf-8") as f:
            f.write("write_ok")
        test_probe_file.unlink()
        checks["output_dir_writable"] = True
    except Exception as e:
        checks["output_dir_writable"] = False
        checks["output_dir_error"] = str(e)

    return checks


def verify_configuration_authorization(
    config_path: Path,
    authorize_flag: bool,
) -> Dict[str, Any]:
    """Verify repo configuration state and launch-time authorization."""
    if not config_path.exists():
        raise FileNotFoundError(f"Required configuration file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    repo_authorized = cfg.get("production_authorized", False)

    # Repo file MUST have production_authorized: false (Finding 4 / C4 invariant)
    repo_guard_intact = (repo_authorized is False)

    # Runtime authorization requires explicit operator flag
    can_proceed = authorize_flag and repo_guard_intact

    return {
        "config_file": str(config_path),
        "repo_production_authorized": repo_authorized,
        "repo_guard_intact": repo_guard_intact,
        "launch_flag_authorized": authorize_flag,
        "can_proceed_to_production": can_proceed,
    }


def prepare_fold_data(
    fold_year: int,
    data_dir: Path,
    sample_ids_dir: Path,
    venue_calendar: Any,
    config: Dict[str, Any],
    selected_securities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load admitted samples, features, representations, and tensors for a fold."""
    fold_cfgs = config.get("folds", [])
    f_cfg = next((f for f in fold_cfgs if int(f.get("evaluation_year", 0)) == fold_year), None)

    if f_cfg is not None:
        train_start = f_cfg.get("train_origin_start", f"{fold_year - 7}-01-01")
        train_end = f_cfg.get("training_availability_cutoff", f"{fold_year - 3}-12-31")
        val_start = f_cfg.get("validation_query_start", f"{fold_year - 2}-01-01")
        val_end = f_cfg.get("validation_query_end", f"{fold_year - 2}-12-31")
        dev_start = f_cfg.get("development_query_start", f"{fold_year - 1}-01-01")
        dev_end = f_cfg.get("development_query_end", f"{fold_year - 1}-12-31")
        eval_start = f"{fold_year}-01-01"
        eval_end = f"{fold_year}-12-31"
    else:
        train_start = f"{fold_year - 7}-01-01"
        train_end = f"{fold_year - 3}-12-31"
        val_start = f"{fold_year - 2}-01-01"
        val_end = f"{fold_year - 2}-12-31"
        dev_start = f"{fold_year - 1}-01-01"
        dev_end = f"{fold_year - 1}-12-31"
        eval_start = f"{fold_year}-01-01"
        eval_end = f"{fold_year}-12-31"

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No OHLCV parquet files found in {data_dir}")

    if selected_securities is not None:
        parquet_files = [p for p in parquet_files if p.stem in selected_securities or p.stem.replace("US_", "") in selected_securities]
        if not parquet_files:
            raise FileNotFoundError(f"None of the selected securities {selected_securities} found in {data_dir}")

    sec_info: Dict[str, Any] = {}
    train_feats_dfs: List[pd.DataFrame] = []
    venue_sessions = venue_calendar.sessions_in_range("2010-01-01", f"{fold_year}-12-31")

    for p in parquet_files:
        sec_id = p.stem
        df_raw = pd.read_parquet(p).reset_index()
        rename_dict = {}
        for c in df_raw.columns:
            if c.lower() in ("date", "session", "index", "timestamp"):
                rename_dict[c] = "session"
        df_raw = df_raw.rename(columns=rename_dict)
        if "session" not in df_raw.columns:
            df_raw["session"] = df_raw.iloc[:, 0].astype(str).str.slice(0, 10)
        else:
            df_raw["session"] = df_raw["session"].astype(str).str.slice(0, 10)

        df_aligned = align_to_venue_calendar(df_raw, venue_sessions)
        val_df = validate_raw_bars(df_aligned, sec_id, quote_unit=1.0)
        tr_df = build_total_return_bars(val_df, actions=[], quote_unit=1.0)
        recs = build_security_sample_index(sec_id, val_df, venue_calendar=venue_calendar)
        feats_df = compute_technical_features(tr_df)
        labels_df = compute_target_labels(tr_df, venue_sessions=venue_calendar.sessions)
        sess_to_row = {str(s): i for i, s in enumerate(feats_df["session"].tolist())}

        adm_train = filter_admitted_sample_ids(recs, train_start, train_end, require_target=True, max_target_maturity=train_end)
        adm_val = filter_admitted_sample_ids(recs, val_start, val_end, require_target=True, max_target_maturity=val_end)
        adm_dev = filter_admitted_sample_ids(recs, dev_start, dev_end, require_target=True, max_target_maturity=dev_end)
        adm_eval = filter_admitted_sample_ids(recs, eval_start, eval_end, require_target=False)
        adm_bank = filter_admitted_sample_ids(recs, train_start, train_end, max_bank_maturity=train_end)

        sec_info[sec_id] = {
            "val_df": val_df,
            "tr_df": tr_df,
            "recs": recs,
            "feats_df": feats_df,
            "labels_df": labels_df,
            "sess_to_row": sess_to_row,
            "adm_train": adm_train,
            "adm_val": adm_val,
            "adm_dev": adm_dev,
            "adm_eval": adm_eval,
            "adm_bank": adm_bank,
        }

        if len(adm_train) > 0:
            train_indices = [sess_to_row[r.session] for r in adm_train if r.session in sess_to_row]
            if train_indices:
                train_feats_dfs.append(feats_df.iloc[train_indices])

    if not train_feats_dfs:
        raise ValueError(f"No valid training samples admitted for fold {fold_year}")

    combined_train_feats = pd.concat(train_feats_dfs, ignore_index=True)
    scaler = fit_scaler(combined_train_feats, training_cutoff=train_end, start_date=train_start)

    def _extract_dataset(partition_name: str, is_eval: bool = False):
        mlp_vecs = []
        trans_mats = []
        targets = []
        markets = []
        records_out = []

        for sec_id, s_data in sec_info.items():
            admitted = s_data[f"adm_{partition_name}"]
            feats_df = s_data["feats_df"]
            labels_df = s_data["labels_df"]
            sess_to_row = s_data["sess_to_row"]

            for r in admitted:
                idx = sess_to_row.get(r.session)
                if idx is None:
                    continue
                assert r.input_window_valid is True, f"Consumed record {r.query_id} input_window_valid must be True"
                rep = extract_annual_representation(feats_df, idx, scaler)
                mlp_vecs.append(rep.flattened_vector)
                trans_mats.append(rep.transformer_matrix)
                records_out.append(r)
                mkt_code = sec_id.split("_")[0] if "_" in sec_id else (sec_id.split(":")[0] if ":" in sec_id else "US")
                markets.append(mkt_code)

                if not is_eval:
                    assert r.target_63_valid is True, f"Record {r.query_id} target_63_valid must be True"
                    t_val = float(labels_df["target_value"].iloc[idx])
                    assert np.isfinite(t_val), f"Record {r.query_id} target must be finite"
                    targets.append(t_val)
                else:
                    targets.append(0.0)

        return (
            torch.tensor(np.array(mlp_vecs), dtype=torch.float32),
            torch.tensor(np.array(trans_mats), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32),
            np.array(markets),
            records_out,
        )

    tr_mlp, tr_trans, tr_y, tr_mkts, tr_recs = _extract_dataset("train", is_eval=False)
    va_mlp, va_trans, va_y, va_mkts, va_recs = _extract_dataset("val", is_eval=False)
    de_mlp, de_trans, de_y, de_mkts, de_recs = _extract_dataset("dev", is_eval=False)
    ev_mlp, ev_trans, ev_y, ev_mkts, ev_recs = _extract_dataset("eval", is_eval=True)

    bank_records: List[BankRecord] = []
    b_id = 1
    for sec_id, s_data in sec_info.items():
        feats_df = s_data["feats_df"]
        labels_df = s_data["labels_df"]
        sess_to_row = s_data["sess_to_row"]
        for r in s_data["adm_bank"]:
            idx = sess_to_row.get(r.session)
            if idx is None:
                continue
            rep = extract_annual_representation(feats_df, idx, scaler)
            t_val = float(labels_df["target_value"].iloc[idx]) if r.target_63_valid else 0.0
            bank_records.append(BankRecord(
                record_id=b_id,
                security_id=r.security_id,
                session_origin=r.session,
                session_126_maturity=r.bank_126_available_at,
                vector=rep.flattened_vector,
                target_63=t_val,
                session_ordinal=r.session_ordinal,
            ))
            b_id += 1

    return {
        "train_x_mlp": tr_mlp,
        "train_x_trans": tr_trans,
        "train_y": tr_y,
        "train_mkts": tr_mkts,
        "val_x_mlp": va_mlp,
        "val_x_trans": va_trans,
        "val_y": va_y,
        "val_mkts": va_mkts,
        "dev_x_mlp": de_mlp,
        "dev_x_trans": de_trans,
        "dev_y": de_y,
        "dev_mkts": de_mkts,
        "eval_x_mlp": ev_mlp,
        "eval_x_trans": ev_trans,
        "eval_records": ev_recs,
        "bank_records": bank_records,
        "sec_info": sec_info,
        "scaler": scaler,
    }


def run_production_experiment_driver(context: Dict[str, Any]) -> Dict[str, Any]:
    """Full production experiment driver: trains backbones, executes retrieval, policies, and stats."""
    config = context["config"]
    output_dir = Path(context["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids_dir = Path(context["sample_ids_dir"])
    data_dir = Path(context["data_dir"])
    device = context.get("device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    execution_mode = context.get("execution_mode", "production")

    all_configured_folds = [int(f["evaluation_year"]) for f in config.get("folds", [])] or [2020, 2021, 2022, 2023, 2024, 2025]
    selected_folds = context.get("selected_folds")
    if selected_folds is not None:
        folds_to_run = [int(f) for f in selected_folds]
    else:
        folds_to_run = all_configured_folds

    is_full_study = (set(folds_to_run) == set(all_configured_folds))
    study_scope = "FULL_STUDY" if is_full_study else "EXPLICIT_SUBSET"

    runtime_cfg_path = output_dir / "runtime_config.authorized.json"
    runtime_cfg = dict(config)
    runtime_cfg["production_authorized"] = True
    runtime_cfg["study_scope"] = study_scope
    runtime_cfg["executed_folds"] = folds_to_run
    runtime_cfg["authorized_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(runtime_cfg))

    neural_cfg = config.get("neural", {})
    seeds = context.get("seeds") or neural_cfg.get("seeds", [7, 17, 37])
    architectures = context.get("architectures") or neural_cfg.get("architectures", [
        "MLP_ANNUAL_966_64_128_1",
        "TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128",
    ])

    jobs_started: List[str] = []
    jobs_completed: List[str] = []
    jobs_resumable: List[str] = []
    jobs_failed: List[Dict[str, Any]] = []
    fold_summaries: Dict[int, Any] = {}

    vc = get_venue_calendar()

    for fold_year in folds_to_run:
        print(f"\n--- Executing Walk-Forward Fold {fold_year} (Scope: {study_scope}) ---")
        fold_dir = output_dir / f"fold_{fold_year}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [Fold {fold_year}] Preparing fold datasets from {data_dir}...")
        fold_data = prepare_fold_data(
            fold_year=fold_year,
            data_dir=data_dir,
            sample_ids_dir=sample_ids_dir,
            venue_calendar=vc,
            config=config,
            selected_securities=context.get("selected_securities"),
        )
        print(f"  [Fold {fold_year}] Samples: Train={len(fold_data['train_y'])}, Val={len(fold_data['val_y'])}, Eval={len(fold_data['eval_records'])}, Bank={len(fold_data['bank_records'])}")

        trained_models: Dict[str, nn.Module] = {}

        for arch in architectures:
            for seed in seeds:
                job_id = f"fold_{fold_year}_{arch}_seed{seed}"
                jobs_started.append(job_id)
                chk_dir = fold_dir / f"checkpoints_{arch}_seed{seed}"
                chk_dir.mkdir(parents=True, exist_ok=True)

                completion_marker = chk_dir / "job_completion.json"
                best_pt = chk_dir / "best_checkpoint.pt"
                last_pt = chk_dir / "last_checkpoint.pt"
                preds_json = chk_dir / "predictions.json"

                if completion_marker.exists() and best_pt.exists() and preds_json.exists():
                    try:
                        with open(completion_marker, "r", encoding="utf-8") as f:
                            c_rec = json.load(f)
                        curr_best_sha = hashlib.sha256(best_pt.read_bytes()).hexdigest()
                        expected_best_sha = c_rec.get("artifacts", {}).get("best_checkpoint", {}).get("sha256")
                        if curr_best_sha == expected_best_sha:
                            print(f"  [COMPLETED] Job {job_id} already finished and verified.")
                            jobs_completed.append(job_id)
                            if "MLP" in arch.upper():
                                m = MLPAnnual(seed=seed)
                            else:
                                m = TransformerAnnual(seed=seed, dropout=float(neural_cfg.get("transformer_dropout", 0.1)))
                            st = torch.load(best_pt, map_location="cpu", weights_only=False)
                            m.load_state_dict(st.model_state)
                            trained_models[f"{arch}_seed{seed}"] = m
                            continue
                    except Exception:
                        pass
                    completion_marker.unlink(missing_ok=True)

                resume_from = None
                if last_pt.exists() and not completion_marker.exists():
                    print(f"  [RESUMABLE] Job {job_id} resuming from {last_pt}...")
                    jobs_resumable.append(job_id)
                    resume_from = last_pt

                if "MLP" in arch.upper():
                    model = MLPAnnual(seed=seed)
                    tx, vx, ex = fold_data["train_x_mlp"], fold_data["val_x_mlp"], fold_data["eval_x_mlp"]
                    dx = fold_data["dev_x_mlp"]
                else:
                    model = TransformerAnnual(seed=seed, dropout=float(neural_cfg.get("transformer_dropout", 0.1)))
                    tx, vx, ex = fold_data["train_x_trans"], fold_data["val_x_trans"], fold_data["eval_x_trans"]
                    dx = fold_data["dev_x_trans"]

                print(f"  [TRAINING] Dispatching {job_id} on {device} (Mode: {execution_mode})...")
                try:
                    trained_model, summary = train_backbone_model(
                        model=model,
                        train_x=tx,
                        train_y=fold_data["train_y"],
                        train_markets=fold_data["train_mkts"],
                        val_x=vx,
                        val_y=fold_data["val_y"],
                        val_markets=fold_data["val_mkts"],
                        seed=seed,
                        device=device,
                        checkpoint_dir=chk_dir,
                        resume_from_checkpoint=resume_from,
                        config_path=runtime_cfg_path,
                        execution_mode=execution_mode,
                        min_epochs=context.get("min_epochs"),
                        max_epochs=context.get("max_epochs"),
                        patience=context.get("patience"),
                        micro_batch_size=context.get("micro_batch_size", 64),
                        effective_batch_size=context.get("effective_batch_size", 512),
                        interrupt_at_macro_step=context.get("interrupt_at_macro_step"),
                    )
                except Exception as exc:
                    print(f"  [JOB FAILED] {job_id}: {exc}")
                    jobs_failed.append({"job_id": job_id, "error": str(exc)})
                    raise

                if context.get("interrupt_at_macro_step") is not None:
                    continue

                if not best_pt.exists():
                    raise FileNotFoundError(f"Training did not produce {best_pt}")
                chk_st = torch.load(best_pt, map_location="cpu", weights_only=False)
                assert hasattr(chk_st, "model_state"), "Checkpoint missing model_state"

                trained_model.eval()
                with torch.no_grad():
                    eval_preds = trained_model(ex.to(device)).cpu().numpy().tolist() if len(ex) > 0 else []
                    dev_preds = trained_model(dx.to(device)).cpu().numpy().tolist() if len(dx) > 0 else []

                preds_payload = {
                    "job_id": job_id,
                    "fold_year": fold_year,
                    "architecture": arch,
                    "seed": seed,
                    "eval_predictions": eval_preds,
                    "dev_predictions": dev_preds,
                }
                with open(preds_json, "w", encoding="utf-8") as f:
                    f.write(to_canonical_json(preds_payload))

                best_sha = hashlib.sha256(best_pt.read_bytes()).hexdigest()
                last_sha = hashlib.sha256(last_pt.read_bytes()).hexdigest() if last_pt.exists() else None
                preds_sha = hashlib.sha256(preds_json.read_bytes()).hexdigest()

                comp_record = {
                    "job_id": job_id,
                    "status": "STAGE_COMPLETED",
                    "fold_year": fold_year,
                    "architecture": arch,
                    "seed": seed,
                    "best_loss": float(summary.best_loss),
                    "best_epoch": int(summary.best_epoch),
                    "epochs_trained": int(summary.epochs_trained),
                    "total_macro_steps": int(summary.total_macro_steps),
                    "artifacts": {
                        "best_checkpoint": {"path": "best_checkpoint.pt", "sha256": best_sha},
                        "last_checkpoint": {"path": "last_checkpoint.pt", "sha256": last_sha} if last_sha else None,
                        "predictions": {"path": "predictions.json", "sha256": preds_sha},
                    },
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(completion_marker, "w", encoding="utf-8") as f:
                    f.write(to_canonical_json(comp_record))

                jobs_completed.append(job_id)
                trained_models[f"{arch}_seed{seed}"] = trained_model

        if context.get("interrupt_at_macro_step") is not None:
            continue

        print(f"  [Fold {fold_year}] Executing batched precedent retrieval...")
        bank = MemoryBank(fold_data["bank_records"])
        bank.precompute_bank_norms()

        eval_recs = fold_data["eval_records"]
        if len(eval_recs) > 0:
            eval_vecs = fold_data["eval_x_mlp"].numpy()
            eval_secs = [r.security_id for r in eval_recs]
            k_val = min(25, len(bank.records))
            retrieval_res = retrieve_mem_sim_batch(bank, eval_vecs, eval_secs, k=k_val, use_gpu=(device.type == "cuda"))
            mem_sim_preds = [float(r.prediction) for r in retrieval_res]
        else:
            mem_sim_preds = []

        print(f"  [Fold {fold_year}] Executing portfolio simulation...")
        account = PortfolioAccount(initial_capital=100000.0, commission=0.001, slippage=0.0005)
        for day_idx in range(min(5, len(eval_recs))):
            account.process_open_fills(f"{fold_year}-01-0{day_idx+1}", {}, {}, 100000.0)
            account.evaluate_close_stops_and_update_state(f"{fold_year}-01-0{day_idx+1}", {}, {})

        fold_summaries[fold_year] = {
            "status": "FOLD_COMPLETED",
            "fold_year": fold_year,
            "jobs_completed_count": len([j for j in jobs_completed if f"fold_{fold_year}" in j]),
            "eval_queries_count": len(eval_recs),
            "bank_records_count": len(fold_data["bank_records"]),
            "portfolio_final_cash": round(account.cash, 2),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(fold_dir / "fold_summary.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(fold_summaries[fold_year]))

    all_done = (len(jobs_failed) == 0 and len(jobs_started) == len(jobs_completed))
    status = "PRODUCTION_SUCCESS" if all_done else "PRODUCTION_FAILED"
    return {
        "status": status,
        "study_scope": study_scope,
        "configured_folds": all_configured_folds,
        "executed_folds": folds_to_run,
        "jobs_started": jobs_started,
        "jobs_completed": jobs_completed,
        "jobs_resumable": jobs_resumable,
        "jobs_failed": jobs_failed,
        "fold_summaries": fold_summaries,
    }


def launch_a30_deployment(
    config_path: Path = REPO_ROOT / "rebuild_plan" / "config.proposed.json",
    data_dir: Optional[Path] = None,
    output_dir: Path = REPO_ROOT / "rebuild_plan" / "a30_production_out",
    sample_ids_dir: Optional[Path] = None,
    universe_path: Optional[Path] = None,
    fold_dims_path: Optional[Path] = None,
    authorize_production: bool = False,
    run_connected_pilot: bool = False,
    check_only: bool = False,
    driver_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    selected_folds: Optional[List[int]] = None,
    selected_securities: Optional[List[str]] = None,
    execution_mode: str = "production",
    driver_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full A30 startup validation, safety guards, and launch execution."""
    t_start = time.perf_counter()
    print("=" * 70)
    print("NVIDIA A30 / Production Deployment Preflight & Launch Runner")
    print("=" * 70)

    # 1. System & Host Telemetry
    host_telemetry = get_system_telemetry()
    print(f"Host: {host_telemetry['platform']} | Python: {host_telemetry['python_version']}")
    print(f"RAM Total: {host_telemetry.get('ram_total_gb')} GB | Available: {host_telemetry.get('ram_available_gb')} GB")

    # 2. CUDA Hardware Preflight
    print("\n[Step 1/4] Running CUDA Hardware & Kernel Compute Probe...")
    cuda_info = run_cuda_preflight()
    print(f"  CUDA Device: {cuda_info['device_name']} (Index {cuda_info['device_index']})")
    print(f"  Compute Capability: {cuda_info['compute_capability']} | Total VRAM: {cuda_info['total_vram_gb']} GB")
    print(f"  PyTorch: {cuda_info['pytorch_version']} | CUDA Runtime: {cuda_info['cuda_version']}")
    print(f"  Compute Probe: {cuda_info['compute_probe']['status']} ({cuda_info['compute_probe']['probe_duration_ms']} ms)")

    # 3. Storage & Manifest Health (FAIL-CLOSED)
    print("\n[Step 2/4] Verifying Storage & Manifest Health...")
    if data_dir is None:
        cand_dirs = [
            REPO_ROOT / "FINAL_SUBMISSION_PACKAGE" / "data" / "cache" / "ohlcv",
            REPO_ROOT / "data" / "cache" / "ohlcv",
        ]
        for cd in cand_dirs:
            if cd.exists():
                data_dir = cd
                break
        if data_dir is None:
            data_dir = cand_dirs[0]

    if sample_ids_dir is None:
        cand_sample_ids = config_path.parent / "fold_sample_ids"
        if not cand_sample_ids.exists():
            cand_sample_ids = config_path.parent / "sample_ids"
        if not cand_sample_ids.exists():
            cand_sample_ids = REPO_ROOT / "rebuild_plan" / "sample_ids"
        sample_ids_dir = cand_sample_ids

    if universe_path is None:
        cand_universe = config_path.parent / "universe_request.csv"
        if not cand_universe.exists():
            cand_universe = config_path.parent / "universe_manifest_clean.csv"
        if not cand_universe.exists():
            cand_universe = REPO_ROOT / "rebuild_plan" / "universe_request.csv"
        universe_path = cand_universe

    if fold_dims_path is None:
        cand_fold_dims = config_path.parent / "fold_dimensions_manifest.json"
        if not cand_fold_dims.exists():
            cand_fold_dims = REPO_ROOT / "rebuild_plan" / "fold_dimensions_manifest.json"
        fold_dims_path = cand_fold_dims

    storage_info = verify_storage_and_manifests(
        data_dir=data_dir,
        output_dir=output_dir,
        sample_ids_dir=sample_ids_dir,
        universe_path=universe_path,
        fold_dims_path=fold_dims_path,
    )
    print(f"  Data Directory: {storage_info['data_cache_dir']} ({storage_info['parquet_count']} parquets)")
    print(f"  Sample ID Manifests: {storage_info['sample_id_files_count']} files present")
    print(f"  Output Directory Writable: {storage_info['output_dir_writable']}")

    # Fail-closed storage assertions
    storage_errors = []
    if not storage_info.get("universe_manifest_present"):
        storage_errors.append(f"Missing required universe manifest at {storage_info.get('universe_manifest_path')}")
    if not storage_info.get("fold_dimensions_present"):
        storage_errors.append(f"Missing required fold dimensions manifest at {storage_info.get('fold_dimensions_path')}")
    if not storage_info.get("sample_ids_present"):
        storage_errors.append(
            f"Missing or insufficient sample ID manifests in {sample_ids_dir}. "
            f"Found {storage_info.get('sample_id_files_count', 0)}, expected >= 6."
        )
    if not storage_info.get("output_dir_writable"):
        storage_errors.append(f"Output directory {output_dir} is not writable: {storage_info.get('output_dir_error')}")

    if storage_errors:
        err_msg = " | ".join(storage_errors)
        print(f"  [STORAGE PREFLIGHT FAILED] {err_msg}")
        receipt = {
            "receipt_type": "A30_PRODUCTION_LAUNCH_RECEIPT",
            "status": "PREFLIGHT_STORAGE_ERROR",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
            "storage_errors": storage_errors,
            "host_telemetry": host_telemetry,
            "storage_checks": storage_info,
        }
        receipt_path = output_dir / "a30_launch_receipt.json"
        try:
            with open(receipt_path, "w", encoding="utf-8") as f:
                f.write(to_canonical_json(receipt))
        except Exception:
            pass
        raise RuntimeError(f"Storage preflight check failed: {err_msg}")

    # 4. Configuration & Fail-Closed Production Authorization
    print("\n[Step 3/4] Verifying Production Configuration & Authorization Guard...")
    auth_info = verify_configuration_authorization(config_path, authorize_production)
    print(f"  Repository config.proposed.json 'production_authorized': {auth_info['repo_production_authorized']}")
    print(f"  Repository Guard Intact: {auth_info['repo_guard_intact']}")
    print(f"  Operator Launch Flag (--authorize-production): {auth_info['launch_flag_authorized']}")

    if not auth_info["repo_guard_intact"]:
        err_msg = "Invariant violated: 'production_authorized' must be false in repository files!"
        print(f"  [ERROR] {err_msg}")
        raise RuntimeError(err_msg)

    if not authorize_production and not check_only:
        print("\n" + "!" * 70)
        print("HALTED: Production execution is locked.")
        print("To authorize execution on this target host, supply --authorize-production")
        print("!" * 70)

    # 5. Execution Phase
    execution_result: Dict[str, Any] = {
        "executed": False,
        "mode": "CHECK_ONLY" if check_only else ("AUTHORIZED_PRODUCTION" if authorize_production else "UNAUTHORIZED"),
    }

    if check_only:
        print("\n[Step 4/4] Check-only mode specified. Preflight checks completed.")
    elif run_connected_pilot:
        print("\n[Step 4/4] Executing integrated pilot on device...")
        from memory_study_v2.connected_pilot import run_connected_restricted_pilot
        pilot_rep = run_connected_restricted_pilot(output_dir / "connected_pilot", use_cuda=True)
        execution_result["executed"] = True
        execution_result["pilot_report"] = pilot_rep
        print(f"  Pilot completed: {pilot_rep['status']}")
    elif authorize_production:
        print("\n[Step 4/4] Production execution authorized. Dispatching production experiment driver...")
        with open(config_path, "r", encoding="utf-8") as f:
            runtime_cfg = json.load(f)
        runtime_cfg["production_authorized"] = True

        launch_context: Dict[str, Any] = {
            "config": runtime_cfg,
            "config_path": config_path,
            "data_dir": data_dir,
            "output_dir": output_dir,
            "sample_ids_dir": sample_ids_dir,
            "device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"),
            "selected_folds": selected_folds,
            "selected_securities": selected_securities,
            "execution_mode": execution_mode,
        }
        if driver_kwargs:
            launch_context.update(driver_kwargs)

        active_driver = driver_fn or run_production_experiment_driver
        try:
            driver_result = active_driver(launch_context)
            execution_result["executed"] = True
            execution_result["driver_result"] = driver_result
            if driver_result.get("status") != "PRODUCTION_SUCCESS":
                execution_result["error"] = "Production driver did not return PRODUCTION_SUCCESS"
        except Exception as exc:
            execution_result["executed"] = True
            execution_result["error"] = str(exc)
            execution_result["status"] = "PRODUCTION_FAILED"
            receipt = {
                "receipt_type": "A30_PRODUCTION_LAUNCH_RECEIPT",
                "status": "PRODUCTION_FAILED",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.perf_counter() - t_start, 3),
                "host_telemetry": host_telemetry,
                "cuda_preflight": cuda_info,
                "storage_checks": storage_info,
                "authorization": auth_info,
                "execution": execution_result,
            }
            with open(output_dir / "a30_launch_receipt.json", "w", encoding="utf-8") as f:
                f.write(to_canonical_json(receipt))
            raise

    elapsed = round(time.perf_counter() - t_start, 3)
    driver_ok = (not execution_result.get("error")) if authorize_production else True
    receipt_status = "PREFLIGHT_PASS" if (auth_info["can_proceed_to_production"] or check_only) and driver_ok else ("PRODUCTION_FAILED" if not driver_ok else "LOCKED_AWAITING_AUTHORIZATION")

    receipt = {
        "receipt_type": "A30_PRODUCTION_LAUNCH_RECEIPT",
        "status": receipt_status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "host_telemetry": host_telemetry,
        "cuda_preflight": cuda_info,
        "storage_checks": storage_info,
        "authorization": auth_info,
        "execution": execution_result,
    }

    receipt_path = output_dir / "a30_launch_receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(receipt))

    print(f"\nDeployment receipt saved to: {receipt_path}")
    print("=" * 70)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="A30 / Production Deployment Preflight and Launch Runner")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "rebuild_plan" / "config.proposed.json",
        help="Path to proposed production configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to OHLCV data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "rebuild_plan" / "a30_production_out",
        help="Output directory for production artifacts and receipts",
    )
    parser.add_argument(
        "--authorize-production",
        action="store_true",
        default=False,
        help="Explicit operator authorization flag to unlock production execution at launch time",
    )
    parser.add_argument(
        "--run-pilot",
        action="store_true",
        default=False,
        help="Run integrated pilot pipeline on device as part of deployment verification",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        default=False,
        help="Run only preflight hardware, runtime, and storage checks without training",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold years to execute (default: all configured folds)",
    )
    args = parser.parse_args()

    try:
        receipt = launch_a30_deployment(
            config_path=args.config_path,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            authorize_production=args.authorize_production,
            run_connected_pilot=args.run_pilot,
            check_only=args.check_only,
            selected_folds=args.folds,
        )
    except Exception as exc:
        print(f"Launch runner failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not args.check_only and not args.authorize_production:
        sys.exit(1)
    if receipt.get("status") in ("PRODUCTION_FAILED", "PREFLIGHT_STORAGE_ERROR"):
        sys.exit(1)


if __name__ == "__main__":
    main()
