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
import collections
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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
from memory_study_v2.contracts import EXPECTED_16_ARMS, EXPECTED_FEATURES_ORDERED, to_canonical_json
import subprocess
from memory_study_v2.execution import DailyLedgerState, PassiveEqualWeightAccount, PortfolioAccount, TradeRecord
from memory_study_v2.features import compute_technical_features, fit_scaler
from memory_study_v2.inference import evaluate_primary_contrasts, export_analysis_bundle, replay_analysis_bundle
from memory_study_v2.integration import fit_trust_gate, select_mixture_mse, select_mixture_sr
from memory_study_v2.labels import compute_target_labels
from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.representations import extract_annual_representation
from memory_study_v2.retrieval import (
    retrieve_hist_prior,
    retrieve_knn_plain,
    retrieve_mem_random,
    retrieve_mem_sim_batch,
)
from memory_study_v2.ridge import fit_ridge_model, RidgeModel
from memory_study_v2.sample_index import (
    SampleIndexRecord,
    build_security_sample_index,
    filter_admitted_sample_ids,
    load_fold_sample_ids,
)
from memory_study_v2.train import train_backbone_model
from memory_study_v2.venue_calendar import get_market_venue_calendar, get_venue_calendar


@dataclass(frozen=True)
class PolicyPredictionRecord:
    """Canonical prediction record joining forecasts to admitted queries."""
    query_id: str
    security_id: str
    market: str
    decision_session: str
    fold: int
    policy_id: str
    realization_id: Optional[int]
    prediction: float
    source_artifact_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "security_id": self.security_id,
            "market": self.market,
            "decision_session": self.decision_session,
            "fold": self.fold,
            "policy_id": self.policy_id,
            "realization_id": self.realization_id,
            "prediction": float(self.prediction),
            "source_artifact_id": self.source_artifact_id,
        }


@dataclass
class DecisionRecord:
    """Traceability record linking decision to prediction, ranking, and fill."""
    session: str
    query_id: str
    security_id: str
    market: str
    policy_id: str
    realization_id: Optional[int]
    prediction: Optional[float]
    volatility_21: float
    score: Optional[float]
    eligible: bool
    rank: int
    selected_for_entry: bool
    action: str
    decision_state: str = "ELIGIBLE_SCORE"
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": str(self.session),
            "query_id": str(self.query_id),
            "security_id": str(self.security_id),
            "market": str(self.market),
            "policy_id": str(self.policy_id),
            "realization_id": int(self.realization_id) if self.realization_id is not None else None,
            "prediction": float(self.prediction) if (self.prediction is not None and math.isfinite(self.prediction)) else None,
            "volatility_21": float(self.volatility_21),
            "score": float(self.score) if (self.score is not None and math.isfinite(self.score)) else None,
            "eligible": bool(self.eligible),
            "rank": int(self.rank),
            "selected_for_entry": bool(self.selected_for_entry),
            "action": str(self.action),
            "decision_state": str(self.decision_state),
            "exclusion_reason": str(self.exclusion_reason) if self.exclusion_reason is not None else None,
        }


def get_system_telemetry() -> Dict[str, Any]:
    """Capture comprehensive host and process telemetry."""
    res: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if psutil is not None:
        vm = psutil.virtual_memory()
        res["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
        res["ram_available_gb"] = round(vm.available / (1024 ** 3), 2)
        res["cpu_count_logical"] = psutil.cpu_count(logical=True)
        res["cpu_count_physical"] = psutil.cpu_count(logical=False)
    return res


def run_cuda_preflight() -> Dict[str, Any]:
    """Verify CUDA runtime and driver compatibility and run compute probe."""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "compute_capability": None,
        "total_vram_gb": 0.0,
        "pytorch_version": torch.__version__ if torch is not None else None,
        "cuda_version": torch.version.cuda if torch is not None else None,
        "compute_probe": "NOT_RUN",
    }
    if torch is None or not torch.cuda.is_available():
        info["status"] = "CUDA_UNAVAILABLE"
        return info

    info["cuda_available"] = True
    info["device_count"] = torch.cuda.device_count()
    info["device_name"] = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    info["compute_capability"] = [cap[0], cap[1]]
    total_mem = torch.cuda.get_device_properties(0).total_memory
    info["total_vram_gb"] = round(total_mem / (1024 ** 3), 2)

    try:
        t0 = time.perf_counter()
        dev = torch.device("cuda:0")
        a = torch.randn((1024, 1024), device=dev, dtype=torch.float32)
        b = torch.randn((1024, 1024), device=dev, dtype=torch.float32)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(dev)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        val = float(c[0, 0].item())
        del a, b, c
        torch.cuda.empty_cache()
        info["compute_probe"] = "PROBE_SUCCESS"
        info["compute_probe_elapsed_ms"] = round(elapsed_ms, 2)
        info["probe_checksum"] = round(val, 6)
        info["status"] = "PREFLIGHT_PASS"
    except Exception as e:
        info["compute_probe"] = f"PROBE_FAILED: {str(e)}"
        info["status"] = "PREFLIGHT_CUDA_ERROR"

    return info


def verify_storage_and_manifests(
    data_dir: Path,
    output_dir: Path,
    sample_ids_dir: Path,
    universe_path: Optional[Path] = None,
    fold_dims_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify input manifests, sample index files, and output path writability."""
    checks: Dict[str, Any] = {}

    if universe_path is None:
        cand = REPO_ROOT / "rebuild_plan" / "universe_request.csv"
        if not cand.exists():
            cand = REPO_ROOT / "rebuild_plan" / "universe_manifest_clean.csv"
        universe_path = cand
    checks["universe_manifest_present"] = universe_path.exists()
    checks["universe_manifest_path"] = str(universe_path)

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

    if sample_ids_dir.exists():
        sample_files = list(sample_ids_dir.glob("fold_*_sample_ids.json")) + list(sample_ids_dir.glob("sample_ids_*.csv"))
        checks["sample_id_files_count"] = len(sample_files)
        checks["sample_ids_present"] = len(sample_files) >= 6
    else:
        checks["sample_id_files_count"] = 0
        checks["sample_ids_present"] = False
    checks["sample_ids_dir"] = str(sample_ids_dir)

    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        checks["data_cache_dir"] = str(data_dir)
        checks["parquet_count"] = len(parquet_files)
    else:
        checks["data_cache_dir"] = str(data_dir)
        checks["parquet_count"] = 0

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
    repo_guard_intact = (repo_authorized is False)
    can_proceed = authorize_flag and repo_guard_intact

    return {
        "config_file": str(config_path),
        "repo_production_authorized": repo_authorized,
        "repo_guard_intact": repo_guard_intact,
        "launch_flag_authorized": authorize_flag,
        "can_proceed_to_production": can_proceed,
    }



def compute_scientific_run_identity(
    config: Dict[str, Any],
    folds_to_run: List[int],
    context: Dict[str, Any],
    checkpoint_hashes: Optional[Dict[str, str]] = None,
    calendar_hashes: Optional[Dict[str, str]] = None,
    code_revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute stable scientific run identity excluding volatile timestamps (finding 3)."""
    if code_revision is None:
        try:
            git_rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            git_rev = "UNTRACKED_OR_DEV"
    else:
        git_rev = str(code_revision).strip()

    sci_config: Dict[str, Any] = {}
    for k in ("folds", "universe", "execution", "neural", "memory", "mixture", "gate", "analysis", "cost_stress"):
        if k in config:
            sci_config[k] = config[k]
    sci_cfg_sha = hashlib.sha256(to_canonical_json(sci_config).encode("utf-8")).hexdigest()

    sample_ids_dir = Path(context.get("sample_ids_dir", REPO_ROOT / "rebuild_plan" / "sample_ids"))
    fold_sample_shas: Dict[str, str] = {}
    for f in sorted(folds_to_run):
        f_path = sample_ids_dir / f"fold_{f}_sample_ids.json"
        if f_path.exists():
            fold_sample_shas[str(f)] = hashlib.sha256(f_path.read_bytes()).hexdigest()

    scope = {
        "folds": sorted(list(folds_to_run)),
        "securities": sorted(list(context["selected_securities"])) if context.get("selected_securities") is not None else "ALL",
        "policies": sorted(list(context["selected_policies"])) if context.get("selected_policies") is not None else "ALL",
        "seeds": sorted(list(context["seeds"])) if context.get("seeds") is not None else "ALL",
        "architectures": sorted(list(context["architectures"])) if context.get("architectures") is not None else "ALL",
    }

    identity = {
        "code_revision": git_rev,
        "scientific_configuration_sha256": sci_cfg_sha,
        "sample_manifest_sha256": fold_sample_shas,
        "calendar_hashes": calendar_hashes or {},
        "requested_scope": scope,
        "checkpoint_hashes": checkpoint_hashes or {},
    }
    identity["identity_sha256"] = hashlib.sha256(to_canonical_json(identity).encode("utf-8")).hexdigest()
    return identity


def compute_training_job_identity(
    fold_year: int,
    arch: str,
    seed: int,
    neural_cfg: Dict[str, Any],
    train_record_ids: List[str],
    train_markets: List[str],
    scaler_cutoff: str,
    scaler_start: str,
    calendar_hashes: Dict[str, str],
    code_revision: str,
) -> Dict[str, Any]:
    """Compute stable, immutable identity for a specific neural training job."""
    sorted_qids = sorted(train_record_ids)
    train_ids_sha = hashlib.sha256("\n".join(sorted_qids).encode("utf-8")).hexdigest()

    relevant_neural = {
        k: neural_cfg.get(k)
        for k in (
            "learning_rate", "weight_decay", "betas", "epsilon", "effective_batch", "effective_batch_size",
            "micro_batch_size", "max_epochs", "min_epochs", "early_stop_patience", "patience",
            "minimum_improvement", "gradient_norm_clip", "optimizer", "transformer_dropout",
        )
        if k in neural_cfg
    }
    neural_sha = hashlib.sha256(to_canonical_json(relevant_neural).encode("utf-8")).hexdigest()

    job_ident = {
        "fold_year": fold_year,
        "architecture": arch,
        "seed": seed,
        "code_revision": code_revision,
        "neural_configuration_sha256": neural_sha,
        "train_sample_ids_sha256": train_ids_sha,
        "train_markets": sorted(list(set(train_markets))),
        "calendar_hashes": {m: calendar_hashes[m] for m in set(train_markets) if m in calendar_hashes},
        "preprocessing": {
            "scaler_start": str(scaler_start),
            "scaler_cutoff": str(scaler_cutoff),
        },
    }
    job_ident["job_identity_sha256"] = hashlib.sha256(to_canonical_json(job_ident).encode("utf-8")).hexdigest()
    return job_ident

def prepare_fold_data(
    fold_year: int,
    data_dir: Path,
    sample_ids_dir: Path,
    venue_calendar: Any = None,
    config: Dict[str, Any] = None,
    selected_securities: Optional[List[str]] = None,
    custom_calendars: Optional[Dict[str, Any]] = None,
    execution_mode: str = "production",
) -> Dict[str, Any]:
    """Load admitted samples, features, representations, and tensors for a fold across native market calendars."""
    if config is None:
        config = {}
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
    market_calendars: Dict[str, Any] = {}

    for p in parquet_files:
        sec_id = p.stem
        mkt_code = sec_id.split("_")[0] if "_" in sec_id else (sec_id.split(":")[0] if ":" in sec_id else "US")
        sec_cal = get_market_venue_calendar(
            mkt_code,
            custom_calendars=custom_calendars,
            data_cache_dir=data_dir,
            execution_mode=execution_mode,
        )
        market_calendars[mkt_code] = sec_cal
        venue_sessions = sec_cal.sessions_in_range("2010-01-01", f"{fold_year}-12-31")

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
        val_df = validate_raw_bars(df_aligned, sec_id, quote_unit=1.0, reject_material=True)
        tr_df = build_total_return_bars(val_df, actions=[], quote_unit=1.0)
        s_min = val_df["session"].min()
        s_max = val_df["session"].max()
        sched_val_df = sec_cal.reindex_to_schedule(val_df, (s_min, s_max))
        recs = build_security_sample_index(sec_id, sched_val_df, venue_calendar=sec_cal, evaluation_year=fold_year)
        feats_df = compute_technical_features(tr_df)
        labels_df = compute_target_labels(tr_df, venue_sessions=sec_cal.sessions)
        sess_to_row = {str(s): i for i, s in enumerate(feats_df["session"].tolist())}

        adm_train = filter_admitted_sample_ids(recs, train_start, train_end, require_target=True, max_target_maturity=train_end)
        adm_val = filter_admitted_sample_ids(recs, val_start, val_end, require_target=True, max_target_maturity=val_end)
        adm_dev = filter_admitted_sample_ids(recs, dev_start, dev_end, require_target=True, max_target_maturity=dev_end)
        adm_eval = filter_admitted_sample_ids(recs, eval_start, eval_end, require_target=False)
        adm_bank = filter_admitted_sample_ids(recs, train_start, train_end, max_bank_maturity=train_end)

        sec_info[sec_id] = {
            "market": mkt_code,
            "calendar": sec_cal,
            "val_df": sched_val_df,
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
                mkt_code = s_data["market"]
                markets.append(mkt_code)

                if not is_eval:
                    assert r.target_63_valid is True, f"Record {r.query_id} target_63_valid must be True"
                    t_val = float(labels_df["target_value"].iloc[idx])
                    assert np.isfinite(t_val), f"Record {r.query_id} target must be finite"
                    targets.append(t_val)
                else:
                    targets.append(0.0)

        return (
            np.array(mlp_vecs, dtype=np.float32) if mlp_vecs else np.zeros((0, 966), dtype=np.float32),
            np.array(trans_mats, dtype=np.float32) if trans_mats else np.zeros((0, 42, 23), dtype=np.float32),
            np.array(targets, dtype=np.float32) if targets else np.zeros((0,), dtype=np.float32),
            np.array(markets) if markets else np.zeros((0,), dtype=object),
            records_out,
        )

    train_x_mlp, train_x_trans, train_y, train_markets, train_recs = _extract_dataset("train", is_eval=False)
    val_x_mlp, val_x_trans, val_y, val_markets, val_recs = _extract_dataset("val", is_eval=False)
    dev_x_mlp, dev_x_trans, dev_y, dev_markets, dev_recs = _extract_dataset("dev", is_eval=False)
    eval_x_mlp, eval_x_trans, eval_y, eval_markets, eval_recs = _extract_dataset("eval", is_eval=True)

    bank_records = []
    for sec_id, s_data in sec_info.items():
        feats_df = s_data["feats_df"]
        labels_df = s_data["labels_df"]
        sess_to_row = s_data["sess_to_row"]
        for r in s_data["adm_bank"]:
            idx = sess_to_row.get(r.session)
            if idx is None:
                continue
            rep = extract_annual_representation(feats_df, idx, scaler)
            t_val = float(labels_df["target_value"].iloc[idx])
            bank_records.append(BankRecord(
                record_id=len(bank_records) + 1,
                security_id=r.security_id,
                session_origin=r.session,
                session_126_maturity=r.bank_126_available_at,
                vector=rep.flattened_vector,
                target_63=t_val,
                session_ordinal=r.session_ordinal,
            ))

    bank = MemoryBank(bank_records)

    return {
        "train_x_mlp": train_x_mlp,
        "train_x_trans": train_x_trans,
        "train_y": train_y,
        "train_markets": train_markets,
        "train_records": train_recs,
        "val_x_mlp": val_x_mlp,
        "val_x_trans": val_x_trans,
        "val_y": val_y,
        "val_markets": val_markets,
        "val_records": val_recs,
        "dev_x_mlp": dev_x_mlp,
        "dev_x_trans": dev_x_trans,
        "dev_y": dev_y,
        "dev_markets": dev_markets,
        "dev_records": dev_recs,
        "eval_x_mlp": eval_x_mlp,
        "eval_x_trans": eval_x_trans,
        "eval_y": eval_y,
        "eval_markets": eval_markets,
        "eval_records": eval_recs,
        "bank_records": bank_records,
        "bank": bank,
        "scaler": scaler,
        "scaler_start": train_start,
        "scaler_cutoff": train_end,
        "sec_info": sec_info,
        "market_calendars": market_calendars,
    }

def compute_prediction_identity(
    fold_year: int,
    producing_checkpoints: Dict[str, str],
    eval_records: List[Any],
    dev_records: List[Any],
    configured_policies: Optional[List[str]],
    seeds: List[int],
    code_revision: str,
    prediction_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute stable, immutable identity for sealed policy predictions artifact."""
    eval_qids = sorted(r.query_id for r in eval_records)
    dev_qids = sorted(r.query_id for r in dev_records)
    eval_qids_sha = hashlib.sha256("\n".join(eval_qids).encode("utf-8")).hexdigest()
    dev_qids_sha = hashlib.sha256("\n".join(dev_qids).encode("utf-8")).hexdigest()

    pred_cfg_clean = prediction_config or {}
    pred_cfg_sha = hashlib.sha256(to_canonical_json(pred_cfg_clean).encode("utf-8")).hexdigest()

    ident = {
        "fold_year": fold_year,
        "producing_checkpoints": dict(sorted(producing_checkpoints.items())),
        "admitted_eval_queries_sha256": eval_qids_sha,
        "admitted_dev_queries_sha256": dev_qids_sha,
        "eval_queries_count": len(eval_qids),
        "dev_queries_count": len(dev_qids),
        "selected_policies": sorted(list(configured_policies)) if configured_policies is not None else "ALL_CONFIGURED",
        "seeds": sorted(list(seeds)),
        "code_revision": str(code_revision).strip(),
        "prediction_config_sha256": pred_cfg_sha,
    }
    ident["prediction_identity_sha256"] = hashlib.sha256(to_canonical_json(ident).encode("utf-8")).hexdigest()
    return ident


def generate_and_seal_policy_predictions(
    fold_year: int,
    fold_dir: Path,
    fold_data: Dict[str, Any],
    trained_models: Dict[str, nn.Module],
    ridge_model: RidgeModel,
    device: torch.device,
    config: Optional[Dict[str, Any]] = None,
    configured_policies: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    lambda_grid: Optional[List[float]] = None,
    custom_predictions: Optional[Dict[str, float]] = None,
    fail_on_corrupted_predictions: bool = False,
    checkpoint_hashes: Optional[Dict[str, str]] = None,
    code_revision: Optional[str] = None,
) -> List[PolicyPredictionRecord]:
    """Generate, validate, and atomically seal common prediction records for all configured arms."""
    pred_manifest_path = fold_dir / "predictions_manifest.json"
    pred_json_path = fold_dir / "policy_predictions.json"

    eval_recs: List[SampleIndexRecord] = fold_data["eval_records"]
    dev_recs: List[SampleIndexRecord] = fold_data.get("dev_records", [])
    admitted_query_ids = {r.query_id for r in eval_recs}

    seeds = seeds or [7, 17, 37]
    lambda_grid = lambda_grid or [0.0, 0.10, 0.25, 0.50, 1.00]

    # Collect producing checkpoints on disk and from completed checkpoint hashes
    producing_checkpoints: Dict[str, str] = {}
    for chk_dir in sorted(fold_dir.glob("checkpoints_*")):
        best_pt = chk_dir / "best_checkpoint.pt"
        if best_pt.exists():
            producing_checkpoints[chk_dir.name] = hashlib.sha256(best_pt.read_bytes()).hexdigest()
    if checkpoint_hashes:
        for k, v in checkpoint_hashes.items():
            if f"fold_{fold_year}" in k or any(chk_name in k for chk_name in producing_checkpoints):
                producing_checkpoints[k] = v

    actual_code_rev = str(code_revision).strip() if code_revision else "UNKNOWN_REVISION"
    pred_config_payload = {
        "lambda_grid": lambda_grid,
        "ridge_l2_lambda": 0.001,
        "trust_gate_steps": 50,
    }

    current_prediction_identity = compute_prediction_identity(
        fold_year=fold_year,
        producing_checkpoints=producing_checkpoints,
        eval_records=eval_recs,
        dev_records=dev_recs,
        configured_policies=configured_policies,
        seeds=seeds,
        code_revision=actual_code_rev,
        prediction_config=pred_config_payload,
    )

    # 1. Check existing verified predictions on disk with prediction_identity match & content validation
    if pred_manifest_path.exists() and pred_json_path.exists() and custom_predictions is None:
        try:
            with open(pred_manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            # Check exact prediction identity match
            stored_identity = manifest_data.get("prediction_identity")
            if not stored_identity:
                raise ValueError("Missing prediction_identity in predictions manifest")
            if stored_identity.get("prediction_identity_sha256") != current_prediction_identity["prediction_identity_sha256"]:
                mismatches = []
                for field in ("producing_checkpoints", "admitted_eval_queries_sha256", "admitted_dev_queries_sha256", "selected_policies", "seeds", "code_revision", "prediction_config_sha256"):
                    if stored_identity.get(field) != current_prediction_identity.get(field):
                        mismatches.append(f"{field} mismatch")
                raise ValueError(f"Prediction identity mismatch: {'; '.join(mismatches) if mismatches else 'sha256 diff'}")

            expected_sha = manifest_data.get("sha256")
            actual_sha = hashlib.sha256(pred_json_path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"Predictions digest mismatch: {actual_sha} != {expected_sha}")

            with open(pred_json_path, "r", encoding="utf-8") as f:
                cached_records = json.load(f)
            records = [
                PolicyPredictionRecord(**r) for r in cached_records
            ]
            # Validate key uniqueness, finite values, and admitted queries match
            seen = set()
            cached_queries = set()
            for r in records:
                k = (r.query_id, r.policy_id, r.realization_id)
                if k in seen:
                    raise ValueError(f"Duplicate prediction key in cached artifact: {k}")
                if not math.isfinite(r.prediction):
                    raise ValueError(f"Non-finite prediction in cached artifact: {k}")
                seen.add(k)
                cached_queries.add(r.query_id)
            if cached_queries != admitted_query_ids:
                raise ValueError(f"Cached queries do not match current admitted queries: {len(cached_queries)} != {len(admitted_query_ids)}")
            print(f"  [Fold {fold_year}] Verified {len(records)} existing predictions on disk (prediction_identity match).")
            return records
        except Exception as e:
            if fail_on_corrupted_predictions:
                raise RuntimeError(f"Corrupted predictions artifact rejected on fold {fold_year}: {e}")
            print(f"  [Fold {fold_year}] Cached predictions validation failed ({e}); regenerating...")
            pred_manifest_path.unlink(missing_ok=True)
            pred_json_path.unlink(missing_ok=True)

    seeds = seeds or [7, 17, 37]
    lambda_grid = lambda_grid or [0.0, 0.10, 0.25, 0.50, 1.00]
    exec_cfg = (config or {}).get("execution", {})

    eval_x_mlp = fold_data["eval_x_mlp"]
    eval_x_trans = fold_data["eval_x_trans"]
    dev_x_mlp = fold_data["dev_x_mlp"]
    dev_x_trans = fold_data["dev_x_trans"]
    dev_y = fold_data["dev_y"]
    dev_mkts = fold_data["dev_markets"]
    bank = fold_data["bank"]
    sec_info = fold_data["sec_info"]

    # Precompute base model evaluation predictions
    eval_mlp_by_seed: Dict[int, np.ndarray] = {}
    eval_trans_by_seed: Dict[int, np.ndarray] = {}
    dev_mlp_by_seed: Dict[int, np.ndarray] = {}
    dev_trans_by_seed: Dict[int, np.ndarray] = {}

    eval_secs = [r.security_id for r in eval_recs]
    dev_secs = [r.security_id for r in dev_recs]
    eval_markets = fold_data["eval_markets"]

    eval_x_mlp_t = torch.tensor(eval_x_mlp, dtype=torch.float32, device=device)
    eval_x_trans_t = torch.tensor(eval_x_trans, dtype=torch.float32, device=device)
    dev_x_mlp_t = torch.tensor(dev_x_mlp, dtype=torch.float32, device=device)
    dev_x_trans_t = torch.tensor(dev_x_trans, dtype=torch.float32, device=device)

    for s in seeds:
        m_mlp_key = f"MLP_ANNUAL_966_64_128_1_seed{s}"
        if m_mlp_key in trained_models:
            m = trained_models[m_mlp_key].to(device)
            m.eval()
            with torch.no_grad():
                eval_mlp_by_seed[s] = m(eval_x_mlp_t).squeeze(-1).cpu().numpy().astype(np.float64)
                if len(dev_x_mlp) > 0:
                    dev_mlp_by_seed[s] = m(dev_x_mlp_t).squeeze(-1).cpu().numpy().astype(np.float64)

        m_trans_key = f"TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128_seed{s}"
        if m_trans_key in trained_models:
            m = trained_models[m_trans_key].to(device)
            m.eval()
            with torch.no_grad():
                eval_trans_by_seed[s] = m(eval_x_trans_t).squeeze(-1).cpu().numpy().astype(np.float64)
                if len(dev_x_trans) > 0:
                    dev_trans_by_seed[s] = m(dev_x_trans_t).squeeze(-1).cpu().numpy().astype(np.float64)

    # Precompute memory retrievals
    k_val = int((config or {}).get("memory", {}).get("k", 25))
    bank_hash = hashlib.sha256(bank.vectors.tobytes()).hexdigest()

    eval_mem_res = retrieve_mem_sim_batch(bank, eval_x_mlp, eval_secs, k=k_val, use_gpu=(device.type == "cuda"))
    eval_mem_preds = np.array([res.prediction for res in eval_mem_res], dtype=np.float64)
    if len(dev_recs) > 0:
        dev_mem_res = retrieve_mem_sim_batch(bank, dev_x_mlp, dev_secs, k=k_val, use_gpu=(device.type == "cuda"))
        dev_mem_preds = np.array([res.prediction for res in dev_mem_res], dtype=np.float64)
    else:
        dev_mem_preds = np.zeros((0,), dtype=np.float64)

    knn_preds = np.array([retrieve_knn_plain(bank, eval_x_mlp[i], eval_secs[i], k=k_val).prediction for i in range(len(eval_recs))], dtype=np.float64)

    rand_preds_by_seed: Dict[int, np.ndarray] = {}
    for s in seeds:
        rand_preds_by_seed[s] = np.array([
            retrieve_mem_random(bank, query_security_id=eval_secs[i], bank_hash=bank_hash, fold_year=fold_year, query_id=eval_recs[i].query_id, master_seed=s, k=k_val).prediction
            for i in range(len(eval_recs))
        ], dtype=np.float64)

    # Precompute HIST_PRIOR and RIDGE_ANNUAL
    hist_val = float(bank.unconditional_mean)
    ridge_eval_preds = ridge_model.predict(eval_x_mlp)

    # Precompute Gates and Mixtures with actual development portfolio Sharpe scoring (A21 / R07)
    gate_mlp_by_seed: Dict[int, np.ndarray] = {}
    gate_trans_by_seed: Dict[int, np.ndarray] = {}
    mix_mse_mlp_by_seed: Dict[int, np.ndarray] = {}
    mix_mse_trans_by_seed: Dict[int, np.ndarray] = {}
    mix_sr_mlp_by_seed: Dict[int, np.ndarray] = {}
    mix_sr_trans_by_seed: Dict[int, np.ndarray] = {}

    def _build_dev_eval_fn(base_by_seed: Dict[int, np.ndarray]):
        initial_cap = float(exec_cfg.get("initial_capital_account_units", 100000.0))
        comm = float(exec_cfg.get("commission_per_side", 0.001))
        slip = float(exec_cfg.get("slippage_per_side", 0.0005))
        cash_frac = float(exec_cfg.get("cash_budget_fraction", 0.95))
        atr_mult = float(exec_cfg.get("atr_stop_multiplier", 2.5))
        min_stop = float(exec_cfg.get("minimum_stop_fraction", 0.10))
        max_hold = int(exec_cfg.get("maximum_holding_sessions", 63))
        max_pos = int(exec_cfg.get("max_positions", 3))

        def dev_eval_fn(lam: float, seed: int, market: str) -> float:
            m_indices = [i for i, r in enumerate(dev_recs) if str(dev_mkts[i]) == market]
            if not m_indices:
                return 0.0

            mix_preds = (1.0 - lam) * base_by_seed[seed][m_indices] + lam * dev_mem_preds[m_indices]
            pred_by_sec_sess: Dict[Tuple[str, str], float] = {}
            for sub_idx, orig_idx in enumerate(m_indices):
                r = dev_recs[orig_idx]
                pred_by_sec_sess[(r.security_id, r.session)] = float(mix_preds[sub_idx])

            dev_sessions = sorted(list(set(dev_recs[i].session for i in m_indices)))
            if not dev_sessions:
                return 0.0

            dev_acct = PortfolioAccount(
                initial_capital=initial_cap,
                commission=comm,
                slippage=slip,
                max_positions=max_pos,
                cash_budget_fraction=cash_frac,
                atr_multiplier=atr_mult,
                min_stop_fraction=min_stop,
                max_holding_sessions=max_hold,
            )

            market_secs = [s for s, s_data in sec_info.items() if s_data.get("market", "US") == market]

            for s_idx, t in enumerate(dev_sessions):
                is_terminal = (s_idx == len(dev_sessions) - 1)
                open_prices = {}
                close_prices = {}
                tradable_flags = {}
                atr_ratios = {}
                vol_map = {}

                for s in market_secs:
                    s_data = sec_info[s]
                    row_idx = s_data["sess_to_row"].get(t)
                    if row_idx is not None:
                        v_rows = s_data["val_df"].loc[s_data["val_df"]["session"] == t]
                        is_valid_bar = (len(v_rows) > 0 and (str(v_rows["bar_status"].iloc[0]) == "VALID" if "bar_status" in v_rows.columns else True))
                        tr_row = s_data["tr_df"].iloc[row_idx]
                        raw_o = float(tr_row["raw_open"])
                        raw_c = float(tr_row["raw_close"])
                        open_prices[s] = raw_o
                        close_prices[s] = raw_c
                        tradable_flags[s] = (is_valid_bar and raw_o > 0 and np.isfinite(raw_o))
                        f_row = s_data["feats_df"].iloc[row_idx]
                        atr_ratios[s] = float(f_row["atr_ratio_14"])
                        vol_map[s] = float(f_row["volatility_21"])

                dev_acct.handle_corporate_actions_before_open(t, {})
                dev_acct.process_open_fills(t, open_prices, tradable_flags)
                dev_acct.evaluate_close_stops_and_update_state(t, close_prices, atr_ratios)

                if not is_terminal:
                    cand_scores = {}
                    for s in market_secs:
                        if s in dev_acct.positions:
                            continue
                        p_val = pred_by_sec_sess.get((s, t))
                        if p_val is not None and p_val > 0.0 and np.isfinite(p_val):
                            v = vol_map.get(s, 0.0)
                            cand_scores[s] = p_val / (v + 1e-4)

                    sorted_cands = sorted(cand_scores.keys(), key=lambda sec: (-cand_scores[sec], sec))
                    dev_acct.plan_entries_at_close(sorted_cands)
                else:
                    dev_term_close_prices = dict(close_prices)
                    for sec_id, pos in dev_acct.positions.items():
                        if sec_id not in dev_term_close_prices or not np.isfinite(dev_term_close_prices[sec_id]) or dev_term_close_prices[sec_id] <= 0.0:
                            fallback_p = pos.last_valid_price if (pos.last_valid_price > 0.0 and np.isfinite(pos.last_valid_price)) else pos.cost_basis
                            dev_term_close_prices[sec_id] = float(fallback_p)
                    dev_acct.execute_terminal_liquidation(t, dev_term_close_prices)

            rets = [st.daily_return for st in dev_acct.daily_history]
            if not rets:
                return 0.0
            r_mean = float(np.mean(rets))
            r_std = float(np.std(rets, ddof=0))
            if r_std > 1e-8:
                return float((r_mean / r_std) * math.sqrt(252.0))
            return 0.0

        return dev_eval_fn

    if dev_mlp_by_seed:
        for s, d_mlp in dev_mlp_by_seed.items():
            fg_mlp = fit_trust_gate(d_mlp, dev_mem_preds, dev_y, dev_mkts, steps=50)
            p_mlp, _ = fg_mlp.predict(eval_mlp_by_seed[s], eval_mem_preds)
            gate_mlp_by_seed[s] = p_mlp

        sel_mse_mlp = select_mixture_mse(dev_mlp_by_seed, dev_mem_preds, dev_y, dev_mkts, lambda_grid=lambda_grid)
        dev_eval_fn_mlp = _build_dev_eval_fn(dev_mlp_by_seed)
        sel_sr_mlp = select_mixture_sr(dev_mlp_by_seed, dev_mem_preds, dev_y, dev_mkts, lambda_grid=lambda_grid, dev_eval_fn=dev_eval_fn_mlp)
        for s in dev_mlp_by_seed:
            mix_mse_mlp_by_seed[s] = (1.0 - sel_mse_mlp.selected_lambda) * eval_mlp_by_seed[s] + sel_mse_mlp.selected_lambda * eval_mem_preds
            mix_sr_mlp_by_seed[s] = (1.0 - sel_sr_mlp.selected_lambda) * eval_mlp_by_seed[s] + sel_sr_mlp.selected_lambda * eval_mem_preds

    if dev_trans_by_seed:
        for s, d_trans in dev_trans_by_seed.items():
            fg_trans = fit_trust_gate(d_trans, dev_mem_preds, dev_y, dev_mkts, steps=50)
            p_trans, _ = fg_trans.predict(eval_trans_by_seed[s], eval_mem_preds)
            gate_trans_by_seed[s] = p_trans

        sel_mse_trans = select_mixture_mse(dev_trans_by_seed, dev_mem_preds, dev_y, dev_mkts, lambda_grid=lambda_grid)
        dev_eval_fn_trans = _build_dev_eval_fn(dev_trans_by_seed)
        sel_sr_trans = select_mixture_sr(dev_trans_by_seed, dev_mem_preds, dev_y, dev_mkts, lambda_grid=lambda_grid, dev_eval_fn=dev_eval_fn_trans)
        for s in dev_trans_by_seed:
            mix_mse_trans_by_seed[s] = (1.0 - sel_mse_trans.selected_lambda) * eval_trans_by_seed[s] + sel_mse_trans.selected_lambda * eval_mem_preds
            mix_sr_trans_by_seed[s] = (1.0 - sel_sr_trans.selected_lambda) * eval_trans_by_seed[s] + sel_sr_trans.selected_lambda * eval_mem_preds

    # Generate records
    records: List[PolicyPredictionRecord] = []
    seen_keys: Set[Tuple[str, str, Optional[int]]] = set()

    def _add_record(qid: str, sec: str, mkt: str, sess: str, pol: str, real: Optional[int], pred: float, art: str):
        k = (qid, pol, real)
        if k in seen_keys:
            raise ValueError(f"Duplicate prediction key generated: {k}")
        if qid not in admitted_query_ids:
            raise ValueError(f"Prediction query {qid} does not belong to admitted queries")
        if not math.isfinite(pred):
            raise ValueError(f"Non-finite prediction encountered for {k}: {pred}")
        if custom_predictions is not None:
            if qid in custom_predictions:
                pred = float(custom_predictions[qid])
            elif sec in custom_predictions:
                pred = float(custom_predictions[sec])
            elif "default" in custom_predictions:
                pred = float(custom_predictions["default"])
        rec = PolicyPredictionRecord(
            query_id=qid,
            security_id=sec,
            market=mkt,
            decision_session=sess,
            fold=fold_year,
            policy_id=pol,
            realization_id=real,
            prediction=float(pred),
            source_artifact_id=art,
        )
        seen_keys.add(k)
        records.append(rec)

    for i, r in enumerate(eval_recs):
        qid = r.query_id
        sec = r.security_id
        mkt = str(eval_markets[i])
        sess = r.session

        # 1. MEM_SIM
        if configured_policies is None or "MEM_SIM" in configured_policies:
            _add_record(qid, sec, mkt, sess, "MEM_SIM", None, eval_mem_preds[i], f"bank_{fold_year}")

        # 2. KNN_PLAIN
        if configured_policies is None or "KNN_PLAIN" in configured_policies:
            _add_record(qid, sec, mkt, sess, "KNN_PLAIN", None, knn_preds[i], f"bank_{fold_year}")

        # 3. MEM_RANDOM
        if configured_policies is None or "MEM_RANDOM" in configured_policies:
            for s in seeds:
                _add_record(qid, sec, mkt, sess, "MEM_RANDOM", s, rand_preds_by_seed[s][i], f"bank_{fold_year}")

        # 4. HIST_PRIOR
        if configured_policies is None or "HIST_PRIOR" in configured_policies:
            _add_record(qid, sec, mkt, sess, "HIST_PRIOR", None, hist_val, f"bank_{fold_year}")

        # 5. RIDGE_ANNUAL
        if configured_policies is None or "RIDGE_ANNUAL" in configured_policies:
            _add_record(qid, sec, mkt, sess, "RIDGE_ANNUAL", None, ridge_eval_preds[i], f"ridge_{fold_year}")

        # 6. MLP_BASE
        if configured_policies is None or "MLP_BASE" in configured_policies:
            for s in seeds:
                if s in eval_mlp_by_seed:
                    _add_record(qid, sec, mkt, sess, "MLP_BASE", s, eval_mlp_by_seed[s][i], f"mlp_{fold_year}_seed{s}")

        # 7. TRANS_BASE
        if configured_policies is None or "TRANS_BASE" in configured_policies:
            for s in seeds:
                if s in eval_trans_by_seed:
                    _add_record(qid, sec, mkt, sess, "TRANS_BASE", s, eval_trans_by_seed[s][i], f"trans_{fold_year}_seed{s}")

        # 8. MLP_MIX_MSE
        if configured_policies is None or "MLP_MIX_MSE" in configured_policies:
            for s in seeds:
                if s in mix_mse_mlp_by_seed:
                    _add_record(qid, sec, mkt, sess, "MLP_MIX_MSE", s, mix_mse_mlp_by_seed[s][i], f"mlp_mix_mse_{fold_year}_seed{s}")

        # 9. TRANS_MIX_MSE
        if configured_policies is None or "TRANS_MIX_MSE" in configured_policies:
            for s in seeds:
                if s in mix_mse_trans_by_seed:
                    _add_record(qid, sec, mkt, sess, "TRANS_MIX_MSE", s, mix_mse_trans_by_seed[s][i], f"trans_mix_mse_{fold_year}_seed{s}")

        # 10. MLP_MIX_SR
        if configured_policies is None or "MLP_MIX_SR" in configured_policies:
            for s in seeds:
                if s in mix_sr_mlp_by_seed:
                    _add_record(qid, sec, mkt, sess, "MLP_MIX_SR", s, mix_sr_mlp_by_seed[s][i], f"mlp_mix_sr_{fold_year}_seed{s}")

        # 11. TRANS_MIX_SR
        if configured_policies is None or "TRANS_MIX_SR" in configured_policies:
            for s in seeds:
                if s in mix_sr_trans_by_seed:
                    _add_record(qid, sec, mkt, sess, "TRANS_MIX_SR", s, mix_sr_trans_by_seed[s][i], f"trans_mix_sr_{fold_year}_seed{s}")

        # 12. MLP_GATE
        if configured_policies is None or "MLP_GATE" in configured_policies:
            for s in seeds:
                if s in gate_mlp_by_seed:
                    _add_record(qid, sec, mkt, sess, "MLP_GATE", s, gate_mlp_by_seed[s][i], f"mlp_gate_{fold_year}_seed{s}")

        # 13. TRANS_GATE
        if configured_policies is None or "TRANS_GATE" in configured_policies:
            for s in seeds:
                if s in gate_trans_by_seed:
                    _add_record(qid, sec, mkt, sess, "TRANS_GATE", s, gate_trans_by_seed[s][i], f"trans_gate_{fold_year}_seed{s}")

        # 14. MOMENTUM_21
        if configured_policies is None or "MOMENTUM_21" in configured_policies:
            _add_record(qid, sec, mkt, sess, "MOMENTUM_21", None, 1.0, f"mom_{fold_year}")

        # 15. VOL_MOMENTUM_21
        if configured_policies is None or "VOL_MOMENTUM_21" in configured_policies:
            _add_record(qid, sec, mkt, sess, "VOL_MOMENTUM_21", None, 1.0, f"vol_mom_{fold_year}")

        # 16. PASSIVE_EQUAL_WEIGHT
        if configured_policies is None or "PASSIVE_EQUAL_WEIGHT" in configured_policies:
            _add_record(qid, sec, mkt, sess, "PASSIVE_EQUAL_WEIGHT", None, 1.0, f"passive_{fold_year}")

    # Atomically seal predictions artifact
    payload = [asdict(r) for r in records]
    with open(pred_json_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(payload))

    actual_sha = hashlib.sha256(pred_json_path.read_bytes()).hexdigest()
    manifest_data = {
        "fold_year": fold_year,
        "count": len(records),
        "sha256": actual_sha,
        "prediction_identity": current_prediction_identity,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(pred_manifest_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(manifest_data))

    print(f"  [Fold {fold_year}] Sealed {len(records)} prediction records (SHA-256: {actual_sha[:12]}...).")
    return records


def run_continuous_portfolio_simulation(
    folds_to_run: List[int],
    fold_data_by_year: Dict[int, Dict[str, Any]],
    predictions_by_year: Dict[int, List[PolicyPredictionRecord]],
    output_dir: Path,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    volatility_guard: float = 1e-4,
    max_positions: int = 3,
) -> Dict[str, Any]:
    """Execute continuous multi-year portfolio simulation preserving account state across native market sessions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_accounts: Dict[Tuple[str, str, Optional[int]], PortfolioAccount] = {}
    all_decisions: Dict[Tuple[str, str, Optional[int]], List[DecisionRecord]] = collections.defaultdict(list)

    # Determine unique account keys from predictions
    all_account_keys: Set[Tuple[str, str, Optional[int]]] = set()
    for y in folds_to_run:
        for r in predictions_by_year.get(y, []):
            p_id = getattr(r, "policy_id", None) or (r.get("policy_id") if isinstance(r, dict) else None)
            mkt = getattr(r, "market", None) or (r.get("market") if isinstance(r, dict) else None)
            real_id = getattr(r, "realization_id", None) if hasattr(r, "realization_id") else (r.get("realization_id") if isinstance(r, dict) else None)
            all_account_keys.add((p_id, mkt, real_id))

    sorted_account_keys = sorted(all_account_keys, key=lambda k: (k[0], k[1], k[2] if k[2] is not None else -1))
    first_fold_data = fold_data_by_year[folds_to_run[0]]

    # Initialize continuous accounts (Active PortfolioAccount vs PassiveEqualWeightAccount)
    for k in sorted_account_keys:
        pol_id, mkt, real_id = k
        if pol_id == "PASSIVE_EQUAL_WEIGHT":
            market_secs = [s for s, s_data in first_fold_data["sec_info"].items() if s_data.get("market", "US") == mkt]
            policy_accounts[k] = PassiveEqualWeightAccount(
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                cash_budget_fraction=0.95,
                universe_size=max(len(market_secs), 1),
            )
        else:
            policy_accounts[k] = PortfolioAccount(
                initial_capital=initial_capital,
                commission=commission,
                slippage=slippage,
                max_positions=max_positions,
            )

    unique_markets = sorted(list(set(k[1] for k in sorted_account_keys)))

    cov_counts: Dict[Tuple[int, str, str, Optional[int]], Dict[str, Any]] = collections.defaultdict(lambda: {
        "candidate_queries": 0,
        "admitted_queries": 0,
        "admitted_qids": set(),
        "exclusion_reasons": collections.defaultdict(int),
        "expected_forecasts": 0,
        "sealed_forecasts": 0,
        "valid_scores": 0,
        "eligible_scores": 0,
        "decisions": 0,
        "orders": 0,
        "fills": 0,
        "exposure_sessions": 0,
    })

    for fold_idx, fold_year in enumerate(folds_to_run):
        is_last_fold = (fold_idx == len(folds_to_run) - 1)
        fold_data = fold_data_by_year[fold_year]
        sec_info = fold_data["sec_info"]
        market_calendars = fold_data.get("market_calendars", {})

        fold_eval_records = fold_data.get("eval_records")
        if fold_eval_records is None:
            raise ValueError(f"MISSING_EVAL_RECORDS: fold_data for fold {fold_year} is missing 'eval_records' field")
        fold_eval_qids: Set[str] = {r.query_id for r in fold_eval_records}

        # Build session record map for fast lookup of input_window_valid and exclusion_reason
        sec_recs_map: Dict[str, Dict[str, Any]] = {
            s: {r.session: r for r in s_data.get("recs", [])}
            for s, s_data in sec_info.items()
        }

        # Pre-index price, feature, and prediction lookups for O(1) step access
        open_map: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        close_map: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        vol_map: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        atr_map: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        mom_map: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
        valid_map: Dict[str, Dict[str, bool]] = collections.defaultdict(dict)

        for s, s_data in sec_info.items():
            for _, row in s_data["tr_df"].iterrows():
                sess_str = str(row["session"])
                open_map[s][sess_str] = float(row["raw_open"])
                close_map[s][sess_str] = float(row["raw_close"])
            for _, row in s_data["feats_df"].iterrows():
                sess_str = str(row["session"])
                vol_map[s][sess_str] = float(row["volatility_21"])
                atr_map[s][sess_str] = float(row["atr_ratio_14"])
                mom_map[s][sess_str] = float(row["momentum_21"])
            for _, row in s_data["val_df"].iterrows():
                sess_str = str(row["session"])
                # Preserve bar validity: only VALID bars are tradable!
                valid_map[s][sess_str] = (str(row.get("bar_status", "")) == "VALID")

        pred_index: Dict[Tuple[str, Optional[int], str, str], float] = {}
        for r in predictions_by_year.get(fold_year, []):
            p_id = getattr(r, "policy_id", None) or (r.get("policy_id") if isinstance(r, dict) else None)
            real_id = getattr(r, "realization_id", None) if hasattr(r, "realization_id") else (r.get("realization_id") if isinstance(r, dict) else None)
            s_id = getattr(r, "security_id", None) or (r.get("security_id") if isinstance(r, dict) else None)
            d_sess = getattr(r, "decision_session", None) or (r.get("decision_session") if isinstance(r, dict) else None)
            pred_val = getattr(r, "prediction", None) if hasattr(r, "prediction") else (r.get("prediction") if isinstance(r, dict) else None)
            qid_r = getattr(r, "query_id", None) or (r.get("query_id") if isinstance(r, dict) else None)

            pred_key = (p_id, real_id, s_id, d_sess)
            if pred_key in pred_index:
                raise ValueError(f"DUPLICATE_PREDICTION_KEY: Duplicate prediction for key {pred_key} in fold {fold_year}")
            expected_qid = f"{fold_year}_{s_id}_{d_sess}"
            if qid_r is not None and qid_r != expected_qid:
                raise RuntimeError(
                    f"QUERY_IDENTITY_MISMATCH: Prediction query_id '{qid_r}' does not match expected '{expected_qid}' for fold {fold_year}"
                )
            pred_index[pred_key] = pred_val

        # Drive accounts strictly by their native market scheduled sessions
        for mkt in unique_markets:
            mkt_accounts = [k for k in sorted_account_keys if k[1] == mkt]
            if not mkt_accounts:
                continue

            mkt_cal = market_calendars.get(mkt) or get_market_venue_calendar(mkt)
            sched_sessions = mkt_cal.sessions_in_range(f"{fold_year}-01-01", f"{fold_year}-12-31")
            if not sched_sessions:
                sched_sessions = sorted(list(set(
                    r.session for r in fold_data.get("eval_records", []) if r.market == mkt
                )))
            if not sched_sessions:
                continue

            mkt_securities = sorted([s for s, s_data in sec_info.items() if s_data.get("market", "US") == mkt])

            for s_idx, t in enumerate(sched_sessions):
                is_final_session = (is_last_fold and (s_idx == len(sched_sessions) - 1))

                open_prices = {s: open_map[s][t] for s in mkt_securities if t in open_map[s]}
                close_prices = {s: close_map[s][t] for s in mkt_securities if t in close_map[s]}
                tradable_flags = {
                    s: (valid_map[s].get(t, False) and s in open_prices and open_prices[s] > 0 and np.isfinite(open_prices[s]))
                    for s in mkt_securities
                }
                atr_ratios = {s: atr_map[s].get(t, 0.0) for s in mkt_securities}

                for acct_key in mkt_accounts:
                    pol_id, _, real_id = acct_key
                    acct = policy_accounts[acct_key]
                    cov_key = (fold_year, mkt, pol_id, real_id)
                    cov = cov_counts[cov_key]

                    # 1. Passive benchmark initial purchase at fold 0 session 0
                    if isinstance(acct, PassiveEqualWeightAccount) and not acct.initial_entry_completed:
                        acct.initialize_passive_entries(mkt_securities)

                    # 2. Pre-open corporate actions
                    acct.handle_corporate_actions_before_open(t, {})

                    # 3. Open fills
                    acct.process_open_fills(t, open_prices, tradable_flags, yesterday_equity=acct.prev_equity)

                    # 4. Close stops & NAV update
                    acct.evaluate_close_stops_and_update_state(t, close_prices, atr_ratios)

                    # Track holding exposure
                    if len(acct.positions) > 0:
                        cov["exposure_sessions"] += 1

                    # 5. Entry planning at close (unless final study session)
                    if not is_final_session:
                        if isinstance(acct, PassiveEqualWeightAccount):
                            acct.plan_entries_at_close([])
                            for s in mkt_securities:
                                cov["candidate_queries"] += 1
                                cov["decisions"] += 1
                                qid = f"{fold_year}_{s}_{t}"
                                rec = sec_recs_map.get(s, {}).get(t)
                                if rec is not None and rec.input_window_valid and fold_eval_qids is not None and qid not in fold_eval_qids:
                                    raise RuntimeError(
                                        f"QUERY_IDENTITY_MISMATCH: Query '{qid}' has valid input window for session {t} "
                                        f"but was not admitted in fold_eval_qids ({len(fold_eval_qids)} ids registered)"
                                    )
                                is_adm = (rec is not None and rec.input_window_valid and (fold_eval_qids is None or qid in fold_eval_qids))
                                if is_adm:
                                    cov["admitted_queries"] += 1
                                    cov["admitted_qids"].add(qid)
                                else:
                                    excl = rec.exclusion_reason if (rec and rec.exclusion_reason) else ("MISSING_SESSION_BAR" if rec is None else "INVALID_BAR_IN_INPUT_WINDOW")
                                    cov["exclusion_reasons"][excl] += 1
                        else:
                            scores: Dict[str, float] = {}
                            cand_states: Dict[str, Tuple[str, Optional[str], Optional[float], Optional[float], bool]] = {}
                            # s -> (decision_state, exclusion_reason, raw_pred, score, is_pos)

                            for s in mkt_securities:
                                cov["candidate_queries"] += 1
                                cov["decisions"] += 1
                                qid = f"{fold_year}_{s}_{t}"
                                rec = sec_recs_map.get(s, {}).get(t)
                                if rec is not None and rec.input_window_valid and fold_eval_qids is not None and qid not in fold_eval_qids:
                                    raise RuntimeError(
                                        f"QUERY_IDENTITY_MISMATCH: Query '{qid}' has valid input window for session {t} "
                                        f"but was not admitted in fold_eval_qids ({len(fold_eval_qids)} ids registered)"
                                    )
                                is_admitted = (rec is not None and rec.input_window_valid and (fold_eval_qids is None or qid in fold_eval_qids))

                                if not is_admitted:
                                    excl_reason = rec.exclusion_reason if (rec and rec.exclusion_reason) else ("MISSING_SESSION_BAR" if rec is None else "INVALID_BAR_IN_INPUT_WINDOW")
                                    cov["exclusion_reasons"][excl_reason] += 1
                                    cand_states[s] = ("NO_ADMISSIBLE_INPUT", excl_reason, None, None, False)
                                    scores[s] = -math.inf
                                else:
                                    cov["admitted_queries"] += 1
                                    cov["admitted_qids"].add(qid)
                                    vol = vol_map[s].get(t, 0.0)
                                    mom = mom_map[s].get(t, 0.0)

                                    if pol_id == "MOMENTUM_21":
                                        if mom is None or not math.isfinite(mom):
                                            raise ValueError(f"INVALID_PREDICTION: Non-finite momentum for query '{qid}'")
                                        raw_pred = float(mom)
                                        sc = raw_pred
                                    elif pol_id == "VOL_MOMENTUM_21":
                                        if mom is None or vol is None or not math.isfinite(mom) or not math.isfinite(vol):
                                            raise ValueError(f"INVALID_PREDICTION: Non-finite momentum/vol for query '{qid}'")
                                        raw_pred = float(mom)
                                        sc = raw_pred / (float(vol) + volatility_guard)
                                    else:
                                        cov["expected_forecasts"] += 1
                                        p_key = (pol_id, real_id, s, t)
                                        if p_key not in pred_index:
                                            raise RuntimeError(
                                                f"MISSING_REQUIRED_PREDICTION: Admitted query '{qid}' lacks required "
                                                f"sealed prediction for policy '{pol_id}' (realization {real_id}) in fold {fold_year}!"
                                            )
                                        pred_val = pred_index[p_key]
                                        if pred_val is None or not math.isfinite(pred_val):
                                            raise ValueError(
                                                f"INVALID_PREDICTION: Prediction for admitted query '{qid}' is non-finite: {pred_val}!"
                                            )
                                        cov["sealed_forecasts"] += 1
                                        raw_pred = float(pred_val)
                                        sc = raw_pred / (float(vol) + volatility_guard)

                                    if not math.isfinite(sc):
                                        raise ValueError(f"INVALID_PREDICTION: Score for query '{qid}' is non-finite: {sc}!")

                                    cov["valid_scores"] += 1
                                    if sc <= 0.0:
                                        cand_states[s] = ("NONPOSITIVE_SCORE", None, raw_pred, sc, False)
                                    else:
                                        cov["eligible_scores"] += 1
                                        cand_states[s] = ("ELIGIBLE_SCORE", None, raw_pred, sc, True)

                                    scores[s] = sc

                            sorted_cands = sorted(scores.keys(), key=lambda s: (-scores[s], s))

                            eligible_cands = []
                            for rank_num, s in enumerate(sorted_cands, start=1):
                                d_state, excl_r, raw_pred, sc, is_pos = cand_states[s]
                                is_held = (s in acct.positions)
                                is_trad = tradable_flags.get(s, False)
                                is_elig = (d_state == "ELIGIBLE_SCORE" and not is_held and is_trad and is_pos)
                                if is_elig:
                                    eligible_cands.append(s)

                                if d_state == "NO_ADMISSIBLE_INPUT":
                                    action = f"EXCLUDED_{excl_r}"
                                elif is_held:
                                    action = "HELD_ALREADY"
                                elif not is_trad:
                                    action = "UNTRADABLE"
                                elif not is_pos:
                                    action = "NONPOSITIVE_SCORE"
                                elif is_elig:
                                    action = "ELIGIBLE_CANDIDATE"
                                else:
                                    action = "INELIGIBLE"

                                all_decisions[acct_key].append(DecisionRecord(
                                    session=t,
                                    query_id=f"{fold_year}_{s}_{t}",
                                    security_id=s,
                                    market=mkt,
                                    policy_id=pol_id,
                                    realization_id=real_id,
                                    prediction=raw_pred,
                                    volatility_21=vol_map[s].get(t, 0.0),
                                    score=sc,
                                    eligible=is_elig,
                                    rank=rank_num,
                                    selected_for_entry=False,
                                    action=action,
                                    decision_state=d_state,
                                    exclusion_reason=excl_r,
                                ))

                            acct.plan_entries_at_close(eligible_cands)
                            cov["orders"] += len(acct.pending_entries)

                            for sec_planned in acct.pending_entries:
                                for dec in reversed(all_decisions[acct_key]):
                                    if dec.session == t and dec.security_id == sec_planned:
                                        dec.selected_for_entry = True
                                        dec.action = "ENTRY_QUEUED"
                                        break
                    else:
                        # Terminal liquidation at final session
                        term_close_prices = dict(close_prices)
                        for sec_id, pos in acct.positions.items():
                            if sec_id not in term_close_prices or not np.isfinite(term_close_prices[sec_id]) or term_close_prices[sec_id] <= 0.0:
                                fallback_p = pos.last_valid_price if (pos.last_valid_price > 0.0 and np.isfinite(pos.last_valid_price)) else pos.cost_basis
                                print(f"  [AUDIT NOTICE] Held position '{sec_id}' missing terminal quote at {t}; using last valid price {fallback_p:.4f} for terminal liquidation.")
                                term_close_prices[sec_id] = float(fallback_p)
                        acct.execute_terminal_liquidation(t, term_close_prices)
                        assert abs(acct.cash - acct.daily_history[-1].total_nav) < 1e-6, (
                            f"Terminal cash reconciliation error for {acct_key}: cash={acct.cash}, nav={acct.daily_history[-1].total_nav}"
                        )
                        rets = [st.daily_return for st in acct.daily_history]
                        comp_nav = acct.initial_capital * float(np.prod([1.0 + r for r in rets]))
                        assert abs(comp_nav - acct.cash) < 1e-4, (
                            f"Terminal compound return reconciliation error for {acct_key}: comp={comp_nav}, cash={acct.cash}"
                        )

        # Export fold coverage manifest
        fold_cov_list = []
        for acct_key in sorted_account_keys:
            pol_id, mkt, real_id = acct_key
            cov_key = (fold_year, mkt, pol_id, real_id)
            cov = cov_counts[cov_key]
            buy_fills = len([
                tr for tr in policy_accounts[acct_key].trades
                if tr.session.startswith(str(fold_year)) and tr.side == "BUY"
            ])
            cov["fills"] = buy_fills
            fold_cov_list.append({
                "fold_year": fold_year,
                "market": mkt,
                "policy_id": pol_id,
                "realization_id": real_id,
                "candidate_queries": cov["candidate_queries"],
                "admitted_queries": cov["admitted_queries"],
                "admitted_qids": sorted(list(cov["admitted_qids"])),
                "exclusion_reasons": dict(cov["exclusion_reasons"]),
                "expected_forecasts": cov["expected_forecasts"],
                "sealed_forecasts": cov["sealed_forecasts"],
                "valid_scores": cov["valid_scores"],
                "eligible_scores": cov["eligible_scores"],
                "decisions": cov["decisions"],
                "orders": cov["orders"],
                "fills": cov["fills"],
                "exposure_sessions": cov["exposure_sessions"],
            })
        fold_cov_path = output_dir / f"fold_{fold_year}" / "coverage_manifest.json"
        if (output_dir / f"fold_{fold_year}").exists():
            with open(fold_cov_path, "w", encoding="utf-8") as f:
                f.write(to_canonical_json(fold_cov_list))

    # Full coverage manifest across all folds
    total_coverage_list = []
    for fold_year in folds_to_run:
        for acct_key in sorted_account_keys:
            pol_id, mkt, real_id = acct_key
            cov_key = (fold_year, mkt, pol_id, real_id)
            cov = cov_counts[cov_key]
            total_coverage_list.append({
                "fold_year": fold_year,
                "market": mkt,
                "policy_id": pol_id,
                "realization_id": real_id,
                "candidate_queries": cov["candidate_queries"],
                "admitted_queries": cov["admitted_queries"],
                "admitted_qids": sorted(list(cov["admitted_qids"])),
                "exclusion_reasons": dict(cov["exclusion_reasons"]),
                "expected_forecasts": cov["expected_forecasts"],
                "sealed_forecasts": cov["sealed_forecasts"],
                "valid_scores": cov["valid_scores"],
                "eligible_scores": cov["eligible_scores"],
                "decisions": cov["decisions"],
                "orders": cov["orders"],
                "fills": cov["fills"],
                "exposure_sessions": cov["exposure_sessions"],
            })
    cov_manifest_path = output_dir / "coverage_manifest.json"
    with open(cov_manifest_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(total_coverage_list))

    # Construct union calendar and per-series market-open mask for statistical analysis
    union_sessions = sorted(list(set(
        st.session
        for acct in policy_accounts.values()
        for st in acct.daily_history
    )))
    K = len(sorted_account_keys)
    T = len(union_sessions)
    aligned_returns: Dict[Tuple[str, str, Optional[int]], np.ndarray] = {}
    market_open_mask = np.zeros((K, T), dtype=bool)

    for k_idx, acct_key in enumerate(sorted_account_keys):
        acct = policy_accounts[acct_key]
        sess_to_ret = {st.session: st.daily_return for st in acct.daily_history}
        rets_arr = np.zeros(T, dtype=np.float64)
        for t_idx, sess in enumerate(union_sessions):
            if sess in sess_to_ret:
                market_open_mask[k_idx, t_idx] = True
                rets_arr[t_idx] = sess_to_ret[sess]
            else:
                market_open_mask[k_idx, t_idx] = False
                rets_arr[t_idx] = 0.0
        aligned_returns[acct_key] = rets_arr

    # Export account evidence and validate invariants
    accounts_dir = output_dir / "accounts"
    accounts_dir.mkdir(parents=True, exist_ok=True)
    manifest_artifacts: Dict[str, Any] = {}

    for acct_key in sorted_account_keys:
        pol_id, mkt, real_id = acct_key
        acct = policy_accounts[acct_key]
        tag = f"{pol_id}__{mkt}__{real_id}" if real_id is not None else f"{pol_id}__{mkt}__None"
        acct_dir = accounts_dir / tag
        acct_dir.mkdir(parents=True, exist_ok=True)

        for st in acct.daily_history:
            assert abs(st.total_nav - (st.cash + st.holdings_value)) < 1e-6, (
                f"NAV identity violated on {st.session} for {tag}: {st.total_nav} != {st.cash} + {st.holdings_value}"
            )

        dec_path = acct_dir / "decisions.json"
        tr_path = acct_dir / "trades.json"
        nav_path = acct_dir / "daily_nav.json"

        with open(dec_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json([d.to_dict() for d in all_decisions[acct_key]]))
        with open(tr_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json([asdict(tr) for tr in acct.trades]))
        with open(nav_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json([asdict(st) for st in acct.daily_history]))

        manifest_artifacts[tag] = {
            "decisions_sha": hashlib.sha256(dec_path.read_bytes()).hexdigest(),
            "trades_sha": hashlib.sha256(tr_path.read_bytes()).hexdigest(),
            "daily_nav_sha": hashlib.sha256(nav_path.read_bytes()).hexdigest(),
            "total_trades": len(acct.trades),
            "final_nav": acct.daily_history[-1].total_nav if acct.daily_history else acct.initial_capital,
        }

    port_manifest_path = output_dir / "portfolio_manifest.json"
    with open(port_manifest_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json({
            "accounts_count": len(sorted_account_keys),
            "union_sessions_count": len(union_sessions),
            "artifacts": manifest_artifacts,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }))

    return {
        "policy_accounts": policy_accounts,
        "all_decisions": all_decisions,
        "sorted_account_keys": sorted_account_keys,
        "union_sessions": union_sessions,
        "aligned_returns": aligned_returns,
        "market_open_mask": market_open_mask,
        "coverage_manifest": total_coverage_list,
    }


def build_expected_account_keys(
    config: Dict[str, Any],
    folds: List[int],
    markets: List[str],
    allow_reduced_arms: bool = False,
) -> Set[Tuple[int, str, str, Optional[int]]]:
    """Constructs the exact set of expected account keys (fold_year, market, policy_id, realization_id).

    Constructed strictly from approved configuration and market universe, NEVER inferred from emitted predictions.
    """
    seeds = [int(s) for s in config.get("neural", {}).get("seeds", [7, 17, 37])]
    neural_archs = [str(a).upper() for a in config.get("neural", {}).get("architectures", [])]
    has_mlp = any("MLP" in a for a in neural_archs)
    has_trans = any("TRANSFORMER" in a for a in neural_archs)

    configured_policies = config.get("configured_policies") or config.get("policies")

    if configured_policies is not None:
        active_policies = list(configured_policies)
    else:
        # Standard policies derived from configured architectures
        active_policies = [
            "MEM_SIM", "KNN_PLAIN", "HIST_PRIOR", "RIDGE_ANNUAL",
            "MOMENTUM_21", "VOL_MOMENTUM_21", "PASSIVE_EQUAL_WEIGHT",
        ]
        if has_mlp or has_trans:
            active_policies.append("MEM_RANDOM")
        if has_mlp:
            active_policies.extend(["MLP_BASE", "MLP_MIX_MSE", "MLP_MIX_SR", "MLP_GATE"])
        if has_trans:
            active_policies.extend(["TRANS_BASE", "TRANS_MIX_MSE", "TRANS_MIX_SR", "TRANS_GATE"])

    seeded_policies = {
        "MEM_RANDOM", "MLP_BASE", "TRANS_BASE",
        "MLP_MIX_MSE", "TRANS_MIX_MSE", "MLP_MIX_SR", "TRANS_MIX_SR",
        "MLP_GATE", "TRANS_GATE",
    }

    expected: Set[Tuple[int, str, str, Optional[int]]] = set()
    for f_yr in folds:
        for mkt in markets:
            for pol in active_policies:
                if pol in seeded_policies:
                    for s in seeds:
                        expected.add((int(f_yr), str(mkt), str(pol), int(s)))
                else:
                    expected.add((int(f_yr), str(mkt), str(pol), None))

    return expected


def validate_release_coverage_and_accounting(
    coverage_manifest: List[Dict[str, Any]],
    fold_data_by_year: Dict[int, Dict[str, Any]],
    folds_to_run: List[int],
    expected_accounts: Optional[Set[Tuple[int, str, str, Optional[int]]]] = None,
) -> Dict[str, Any]:
    """Strictly validates coverage accounting and query set equality against independent evaluation populations.

    Enforces:
    1. Zero missing sealed forecasts across all simulated accounts (both global sum and exact per-account reconciliation).
    2. Exact coverage accounting: candidate_queries == decisions == admitted_queries + sum(exclusion_reasons).
    3. Zero QUERY_IDENTITY_MISMATCH exclusion errors and permitted exclusion reasons only.
    4. Exact query-key set equality (or count equality if qids not tracked) against independent evaluation records.
    5. Detection and rejection of partial admission drops (e.g. 1 admitted + 999 excluded vs 1000 expected).
    6. Distinction between intentionally empty populations (expected count 0 -> admitted count 0) and missing metadata.
    7. Model policies expected_forecasts == admitted_queries == sealed_forecasts.
    8. Duplicate coverage account rejection and exact expected-versus-observed account key equality.
    9. Verification that admitted query IDs agree with fold year and market.
    """
    PERMITTED_EXCLUSIONS: Set[str] = {
        "NOT_IN_INDEX",
        "MISSING_SESSION_BAR",
        "INVALID_INPUT_WINDOW",
        "INVALID_BAR_IN_INPUT_WINDOW",
        "INSUFFICIENT_INPUT_HISTORY",
        "GAP_IN_INPUT_WINDOW",
        "MARKET_CLOSED",
        "MAX_POSITIONS_REACHED",
        "INSUFFICIENT_CASH",
        "NEGATIVE_SCORE",
        "NONPOSITIVE_SCORE",
        "NO_ELIGIBLE_SELECTION",
        "HALTED",
        "SESSION_NOT_SCHEDULED",
    }

    total_candidate_queries = sum(c.get("candidate_queries", 0) for c in coverage_manifest)
    total_admitted_queries = sum(c.get("admitted_queries", 0) for c in coverage_manifest)
    total_expected_forecasts = sum(c.get("expected_forecasts", 0) for c in coverage_manifest)
    total_sealed_forecasts = sum(c.get("sealed_forecasts", 0) for c in coverage_manifest)
    total_decisions = sum(c.get("decisions", 0) for c in coverage_manifest)

    missing_forecasts = total_expected_forecasts - total_sealed_forecasts
    if missing_forecasts > 0:
        raise RuntimeError(
            f"RELEASE_VERIFICATION_FAILURE: {missing_forecasts} expected forecasts are missing across simulated accounts!"
        )

    # 1. Reconcile internal coverage accounting for each account entry and check account duplicates
    seen_account_keys: Set[Tuple[int, str, str, Optional[int]]] = set()
    for c in coverage_manifest:
        f_yr = c.get("fold_year")
        mkt = c.get("market")
        pol = c.get("policy_id", "")
        real_id = c.get("realization_id")
        acct_key = (f_yr, mkt, pol, real_id)
        if acct_key in seen_account_keys:
            raise RuntimeError(
                f"RELEASE_VERIFICATION_FAILURE: Duplicate coverage manifest entry for account key: {acct_key}!"
            )
        seen_account_keys.add(acct_key)

        cand = c.get("candidate_queries", 0)
        dec = c.get("decisions", 0)
        adm = c.get("admitted_queries", 0)
        excls = c.get("exclusion_reasons", {})
        total_excl = sum(excls.values())

        if cand != dec:
            raise RuntimeError(
                f"RELEASE_VERIFICATION_FAILURE: Decisions count {dec} does not equal candidate queries {cand} "
                f"for ({f_yr}, {mkt}, {pol}, {real_id})!"
            )
        if cand != adm + total_excl:
            raise RuntimeError(
                f"RELEASE_VERIFICATION_FAILURE: Candidate queries {cand} does not reconcile with "
                f"admitted ({adm}) + exclusions ({total_excl}) for ({f_yr}, {mkt}, {pol}, {real_id})!"
            )
        if "QUERY_IDENTITY_MISMATCH" in excls and excls["QUERY_IDENTITY_MISMATCH"] > 0:
            raise RuntimeError(
                f"RELEASE_VERIFICATION_FAILURE: QUERY_IDENTITY_MISMATCH recorded in exclusions for "
                f"({f_yr}, {mkt}, {pol}, {real_id})!"
            )
        for reason in excls:
            if reason not in PERMITTED_EXCLUSIONS:
                raise RuntimeError(
                    f"RELEASE_VERIFICATION_FAILURE: Non-permitted exclusion reason '{reason}' recorded for "
                    f"({f_yr}, {mkt}, {pol}, {real_id})!"
                )
        if pol not in ("PASSIVE_EQUAL_WEIGHT", "MOMENTUM_21", "VOL_MOMENTUM_21"):
            exp_fc = c.get("expected_forecasts", 0)
            seal_fc = c.get("sealed_forecasts", 0)
            if exp_fc != adm:
                raise RuntimeError(
                    f"RELEASE_VERIFICATION_FAILURE: Model policy {pol} expected forecasts ({exp_fc}) "
                    f"does not match admitted queries ({adm}) for ({f_yr}, {mkt}, {real_id})!"
                )
            if seal_fc != exp_fc:
                raise RuntimeError(
                    f"RELEASE_VERIFICATION_FAILURE: Model policy {pol} sealed forecasts ({seal_fc}) "
                    f"does not match expected forecasts ({exp_fc}) for ({f_yr}, {mkt}, {real_id})! "
                    f"Missing forecasts: {exp_fc - seal_fc}."
                )

        adm_qids = c.get("admitted_qids")
        if adm > 0 and (adm_qids is None or len(adm_qids) == 0):
            raise RuntimeError(
                f"RELEASE_VERIFICATION_FAILURE: Account ({f_yr}, {mkt}, {pol}, {real_id}) "
                f"has {adm} admitted queries but admitted_qids is missing or empty!"
            )
        if adm_qids:
            for qid in adm_qids:
                parts = qid.split("_")
                if len(parts) >= 3:
                    q_fold = parts[0]
                    q_mkt = parts[1]
                    if q_fold != str(f_yr):
                        raise RuntimeError(
                            f"RELEASE_VERIFICATION_FAILURE: Query ID '{qid}' fold prefix '{q_fold}' "
                            f"does not match account fold year '{f_yr}'!"
                        )
                    if q_mkt != mkt and not str(mkt).startswith(q_mkt):
                        raise RuntimeError(
                            f"RELEASE_VERIFICATION_FAILURE: Query ID '{qid}' market prefix '{q_mkt}' "
                            f"does not match account market '{mkt}'!"
                        )

    # 1b. Check exact expected-versus-observed account key equality if expected_accounts provided
    if expected_accounts is not None:
        missing_accounts = expected_accounts - seen_account_keys
        unexpected_accounts = seen_account_keys - expected_accounts
        if missing_accounts or unexpected_accounts:
            raise RuntimeError(
                f"RELEASE_VERIFICATION_FAILURE: Exact account set mismatch! "
                f"Missing accounts: {len(missing_accounts)}, Unexpected accounts: {len(unexpected_accounts)}: "
                f"missing={sorted(list(missing_accounts))[:5]}, unexpected={sorted(list(unexpected_accounts))[:5]}!"
            )

    # 2. Reconcile against independent evaluation dataset population
    for f_yr in folds_to_run:
        if f_yr not in fold_data_by_year:
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Missing fold data metadata for fold {f_yr}!")
        f_data = fold_data_by_year[f_yr]
        if not isinstance(f_data, dict):
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Invalid fold data structure for fold {f_yr}!")
        if "eval_records" not in f_data:
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Missing required 'eval_records' metadata for fold {f_yr}!")
        if "sec_info" not in f_data:
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Missing required 'sec_info' metadata for fold {f_yr}!")

        f_eval_recs = f_data["eval_records"]
        if f_eval_recs is None:
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: 'eval_records' is None for fold {f_yr}!")
        f_sec_info = f_data["sec_info"]
        if f_sec_info is None:
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: 'sec_info' is None for fold {f_yr}!")

        # Determine terminal study session if this is the final fold in folds_to_run
        # The final session of the entire study performs terminal liquidation at close and plans no next-day entries
        terminal_sessions_by_mkt: Dict[str, Set[str]] = collections.defaultdict(set)
        if f_yr == folds_to_run[-1]:
            market_cals = f_data.get("market_calendars", {})
            for s_meta in f_sec_info.values():
                mkt_cand = s_meta.get("market") if isinstance(s_meta, dict) else getattr(s_meta, "market", None)
                if mkt_cand and mkt_cand not in terminal_sessions_by_mkt:
                    mkt_cal = market_cals.get(mkt_cand) or get_market_venue_calendar(mkt_cand)
                    if mkt_cal is not None:
                        sched = mkt_cal.sessions_in_range(f"{f_yr}-01-01", f"{f_yr}-12-31")
                        if sched and len(sched) > 1:
                            terminal_sessions_by_mkt[mkt_cand].add(sched[-1])

        # Group expected query IDs by market
        expected_qids_by_mkt: Dict[str, Set[str]] = collections.defaultdict(set)
        for r in f_eval_recs:
            qid = getattr(r, "query_id", None) or (r.get("query_id") if isinstance(r, dict) else None)
            if not qid:
                raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Missing query_id in eval_records for fold {f_yr}!")
            s_id = getattr(r, "security_id", None) or (r.get("security_id") if isinstance(r, dict) else None)
            mkt = getattr(r, "market", None) or (r.get("market") if isinstance(r, dict) else None)
            if not mkt and s_id and s_id in f_sec_info:
                sec_meta = f_sec_info[s_id]
                mkt = sec_meta.get("market") if isinstance(sec_meta, dict) else getattr(sec_meta, "market", None)
            if not mkt:
                raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Missing market metadata for query '{qid}' in fold {f_yr}!")

            sess = getattr(r, "session", None) or (r.get("session") if isinstance(r, dict) else None)
            if f_yr == folds_to_run[-1] and sess and sess in terminal_sessions_by_mkt.get(mkt, set()):
                continue

            expected_qids_by_mkt[mkt].add(qid)

        cov_entries = [c for c in coverage_manifest if c.get("fold_year") == f_yr]
        if not cov_entries and (f_eval_recs or any(len(q) > 0 for q in expected_qids_by_mkt.values())):
            raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Independent population has eval queries for fold {f_yr}, but no simulation accounts were found!")

        cov_by_mkt: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for c in cov_entries:
            mkt = c.get("market")
            if not mkt:
                raise RuntimeError(f"RELEASE_VERIFICATION_FAILURE: Coverage entry missing market in fold {f_yr}: {c}")
            cov_by_mkt[mkt].append(c)

        # Check markets present in expected_qids
        for mkt, exp_qids in expected_qids_by_mkt.items():
            mkt_cov = cov_by_mkt.get(mkt, [])
            if len(exp_qids) > 0 and not mkt_cov:
                raise RuntimeError(
                    f"RELEASE_VERIFICATION_FAILURE: Independent population has {len(exp_qids)} eval queries "
                    f"for market {mkt} in fold {f_yr}, but no simulation accounts were found!"
                )

        # Validate accounts in cov_by_mkt against expected query population
        for mkt, mkt_cov in cov_by_mkt.items():
            exp_qids = expected_qids_by_mkt.get(mkt, set())
            exp_count = len(exp_qids)

            for entry in mkt_cov:
                pol_id = entry.get("policy_id")
                real_id = entry.get("realization_id")
                adm_count = entry.get("admitted_queries", 0)
                cand_count = entry.get("candidate_queries", 0)
                excls = entry.get("exclusion_reasons", {})

                # Intentionally empty population:
                if exp_count == 0 and adm_count != 0:
                    raise RuntimeError(
                        f"RELEASE_VERIFICATION_FAILURE: Market {mkt} in fold {f_yr} has empty expected evaluation population, "
                        f"but account ({pol_id}, {real_id}) admitted {adm_count} queries!"
                    )

                if exp_count > 0:
                    if adm_count != exp_count:
                        raise RuntimeError(
                            f"RELEASE_VERIFICATION_FAILURE: Partial or mismatched query admission for market {mkt} in fold {f_yr}, "
                            f"policy {pol_id}, realization {real_id}: expected {exp_count} queries, but admitted {adm_count} "
                            f"(candidate={cand_count}, exclusions={excls})!"
                        )

                    admitted_qids = entry.get("admitted_qids")
                    if admitted_qids is not None:
                        adm_set = set(admitted_qids)
                        if adm_set != exp_qids:
                            diff_missing = exp_qids - adm_set
                            diff_extra = adm_set - exp_qids
                            raise RuntimeError(
                                f"RELEASE_VERIFICATION_FAILURE: Admitted query set mismatch for market {mkt} in fold {f_yr}, "
                                f"policy {pol_id}, realization {real_id}: missing={len(diff_missing)}, unexpected={len(diff_extra)}!"
                            )

    return {
        "status": "VALIDATED",
        "total_candidate_queries": total_candidate_queries,
        "total_admitted_queries": total_admitted_queries,
        "total_expected_forecasts": total_expected_forecasts,
        "total_sealed_forecasts": total_sealed_forecasts,
        "total_decisions": total_decisions,
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

    is_full_study = (
        set(folds_to_run) == set(all_configured_folds)
        and context.get("selected_policies") is None
        and context.get("selected_securities") is None
        and context.get("selected_seeds") is None
        and execution_mode == "production"
    )
    study_scope = "FULL_PRODUCTION" if is_full_study else "EXPLICIT_SUBSET"
    allow_reduced_arms = not is_full_study

    # Upfront Market Calendar Validation
    requested_markets: Set[str] = set()
    selected_secs = context.get("selected_securities")
    if selected_secs:
        for s in selected_secs:
            m = s.split("_")[0] if "_" in s else (s.split(":")[0] if ":" in s else "US")
            requested_markets.add(m.upper())
    else:
        for p in data_dir.glob("*.parquet"):
            stem = p.stem
            m = stem.split("_")[0] if "_" in stem else (stem.split(":")[0] if ":" in stem else "US")
            requested_markets.add(m.upper())
    if not requested_markets:
        requested_markets.add("US")

    resolved_market_calendars: Dict[str, VenueCalendar] = {}
    permissive_calendar_fallbacks: Dict[str, str] = {}
    for mkt in sorted(requested_markets):
        cal = get_market_venue_calendar(
            market=mkt,
            custom_calendars=context.get("custom_calendars"),
            data_cache_dir=data_dir,
            execution_mode=execution_mode,
        )
        resolved_market_calendars[mkt] = cal
        if getattr(cal, "is_inferred", False):
            permissive_calendar_fallbacks[mkt] = getattr(cal, "fallback_source", "inferred")

    calendar_hashes = {m: cal.schedule_hash for m, cal in resolved_market_calendars.items()}

    current_run_identity = compute_scientific_run_identity(
        config=config,
        folds_to_run=folds_to_run,
        context=context,
        calendar_hashes=calendar_hashes,
        code_revision=context.get("code_revision"),
    )

    # 1. Check existing verified completion on restart, bound to stable run identity
    pipeline_completion_file = output_dir / "pipeline_summary.json"
    legacy_completion_file = output_dir / "pipeline_completion.json"
    release_manifest_file = output_dir / "release_manifest.json"
    release_analysis_dir = output_dir / "analysis"
    if not release_analysis_dir.exists():
        release_analysis_dir = output_dir / "release_analysis"

    neural_cfg = config.get("neural", {})
    seeds = context.get("seeds") or neural_cfg.get("seeds", [7, 17, 37])
    architectures = context.get("architectures") or neural_cfg.get("architectures", [
        "MLP_ANNUAL_966_64_128_1",
        "TRANSFORMER_42x23_WIDTH64_HEADS4_LAYERS2_FF128_LATENT128",
    ])

    target_completion_file = pipeline_completion_file if pipeline_completion_file.exists() else legacy_completion_file
    if target_completion_file.exists() and release_manifest_file.exists() and release_analysis_dir.exists() and not context.get("force_rerun", False):
        try:
            with open(target_completion_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
            stored_identity = c_data.get("run_identity", {})

            # Compute current actual checkpoint hashes on disk right now
            actual_chk_hashes: Dict[str, str] = {}
            chk_missing = False
            for f_year in folds_to_run:
                for arch in architectures:
                    for s_seed in seeds:
                        jid = f"fold_{f_year}_{arch}_seed{s_seed}"
                        c_pt = output_dir / f"fold_{f_year}" / f"checkpoints_{arch}_seed{s_seed}" / "best_checkpoint.pt"
                        if c_pt.exists():
                            actual_chk_hashes[jid] = hashlib.sha256(c_pt.read_bytes()).hexdigest()
                        else:
                            chk_missing = True
                            actual_chk_hashes[jid] = "MISSING"

            mismatches = []
            if stored_identity.get("code_revision") != current_run_identity["code_revision"]:
                mismatches.append(f"code_revision mismatch: stored={stored_identity.get('code_revision')} vs current={current_run_identity['code_revision']}")
            if stored_identity.get("scientific_configuration_sha256") != current_run_identity["scientific_configuration_sha256"]:
                mismatches.append("scientific configuration mismatch")
            if stored_identity.get("sample_manifest_sha256") != current_run_identity["sample_manifest_sha256"]:
                mismatches.append("sample manifest mismatch")
            if stored_identity.get("calendar_hashes") != current_run_identity["calendar_hashes"]:
                mismatches.append("calendar hashes mismatch")
            if stored_identity.get("requested_scope") != current_run_identity["requested_scope"]:
                mismatches.append(f"scope mismatch: stored={stored_identity.get('requested_scope')} vs current={current_run_identity['requested_scope']}")
            if chk_missing or stored_identity.get("checkpoint_hashes") != actual_chk_hashes:
                mismatches.append("checkpoint hashes mismatch (stored vs current on disk)")

            if mismatches:
                print(f"  [PIPELINE_SCOPE_MISMATCH] Cached completion invalidated: {'; '.join(mismatches)}")
                pipeline_completion_file.unlink(missing_ok=True)
                legacy_completion_file.unlink(missing_ok=True)
                release_manifest_file.unlink(missing_ok=True)
            else:
                replay_res = replay_analysis_bundle(release_analysis_dir, tolerance=1e-10, allow_reduced_arms=allow_reduced_arms)
                print(f"  [PIPELINE_COMPLETE] Experiment already verified for identical run identity (Replay: {replay_res.get('status')}).")
                c_data["replayed"] = True
                c_data["reused_cached_completion"] = True
                return c_data
        except Exception as e:
            print(f"  [PIPELINE_INVALID] Existing completion invalid ({e}); continuing with stage execution...")
            pipeline_completion_file.unlink(missing_ok=True)
            legacy_completion_file.unlink(missing_ok=True)
            release_manifest_file.unlink(missing_ok=True)

    runtime_cfg_path = output_dir / "runtime_config.authorized.json"
    runtime_cfg = dict(config)
    runtime_cfg["production_authorized"] = True
    runtime_cfg["study_scope"] = study_scope
    runtime_cfg["executed_folds"] = folds_to_run
    runtime_cfg["authorized_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(runtime_cfg_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(runtime_cfg))

    jobs_started: List[str] = []
    jobs_completed: List[str] = []
    jobs_resumable: List[str] = []
    jobs_failed: List[Dict[str, Any]] = []

    fold_data_by_year: Dict[int, Dict[str, Any]] = {}
    trained_models_by_year: Dict[int, Dict[str, nn.Module]] = {}
    ridge_models_by_year: Dict[int, RidgeModel] = {}
    predictions_by_year: Dict[int, List[PolicyPredictionRecord]] = {}
    completed_checkpoint_hashes: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Stage 1: Backbone Training & Validation
    # -------------------------------------------------------------------------
    print(f"\n=== Stage 1: Backbone Training & Validation (Scope: {study_scope}) ===")
    for fold_year in folds_to_run:
        fold_dir = output_dir / f"fold_{fold_year}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [Fold {fold_year}] Preparing fold datasets from {data_dir}...")
        fold_data = prepare_fold_data(
            fold_year=fold_year,
            data_dir=data_dir,
            sample_ids_dir=sample_ids_dir,
            config=config,
            selected_securities=context.get("selected_securities"),
            custom_calendars=context.get("custom_calendars"),
            execution_mode=execution_mode,
        )
        fold_data_by_year[fold_year] = fold_data
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
                training_ident_file = chk_dir / "training_identity.json"

                job_train_records = fold_data.get("train_records", [])
                job_train_qids = [r.query_id for r in job_train_records]
                effective_neural_cfg = dict(neural_cfg)
                for k in ("min_epochs", "max_epochs", "micro_batch_size", "effective_batch_size", "effective_batch", "patience", "early_stop_patience", "learning_rate", "weight_decay"):
                    if context.get(k) is not None:
                        effective_neural_cfg[k] = context[k]
                if "effective_batch_size" in effective_neural_cfg and "effective_batch" not in effective_neural_cfg:
                    effective_neural_cfg["effective_batch"] = effective_neural_cfg["effective_batch_size"]
                if "effective_batch" in effective_neural_cfg and "effective_batch_size" not in effective_neural_cfg:
                    effective_neural_cfg["effective_batch_size"] = effective_neural_cfg["effective_batch"]

                current_job_training_identity = compute_training_job_identity(
                    fold_year=fold_year,
                    arch=arch,
                    seed=seed,
                    neural_cfg=effective_neural_cfg,
                    train_record_ids=job_train_qids,
                    train_markets=list(fold_data.get("train_markets", [])),
                    scaler_cutoff=str(fold_data.get("scaler_cutoff", "")),
                    scaler_start=str(fold_data.get("scaler_start", "")),
                    calendar_hashes=calendar_hashes,
                    code_revision=current_run_identity["code_revision"],
                )

                if completion_marker.exists() and best_pt.exists() and preds_json.exists():
                    try:
                        with open(completion_marker, "r", encoding="utf-8") as f:
                            c_rec = json.load(f)
                        stored_job_ident = c_rec.get("training_identity", {})
                        curr_best_sha = hashlib.sha256(best_pt.read_bytes()).hexdigest()
                        expected_best_sha = c_rec.get("artifacts", {}).get("best_checkpoint", {}).get("sha256")
                        if (
                            stored_job_ident.get("job_identity_sha256") == current_job_training_identity["job_identity_sha256"]
                            and curr_best_sha == expected_best_sha
                        ):
                            print(f"  [COMPLETED] Job {job_id} already finished and verified for identical training identity.")
                            jobs_completed.append(job_id)
                            completed_checkpoint_hashes[job_id] = curr_best_sha
                            if "MLP" in arch.upper():
                                m = MLPAnnual(seed=seed)
                            else:
                                m = TransformerAnnual(seed=seed, dropout=float(neural_cfg.get("transformer_dropout", 0.1)))
                            st = torch.load(best_pt, map_location="cpu", weights_only=False)
                            m.load_state_dict(st.model_state)
                            m.to(device)
                            trained_models[f"{arch}_seed{seed}"] = m
                            continue
                        else:
                            print(f"  [INCOMPATIBLE_TRAINING] Job {job_id} training identity or checkpoint digest mismatch; retraining...")
                    except Exception:
                        pass
                    completion_marker.unlink(missing_ok=True)
                    best_pt.unlink(missing_ok=True)
                    last_pt.unlink(missing_ok=True)
                    preds_json.unlink(missing_ok=True)
                    training_ident_file.unlink(missing_ok=True)

                resume_from = None
                if last_pt.exists() and not completion_marker.exists():
                    is_resume_compatible = False
                    if training_ident_file.exists():
                        try:
                            with open(training_ident_file, "r", encoding="utf-8") as f:
                                saved_ident = json.load(f)
                            if saved_ident.get("job_identity_sha256") == current_job_training_identity["job_identity_sha256"]:
                                is_resume_compatible = True
                        except Exception:
                            is_resume_compatible = False
                    if is_resume_compatible:
                        print(f"  [RESUMABLE] Job {job_id} resuming from {last_pt}...")
                        jobs_resumable.append(job_id)
                        resume_from = last_pt
                    else:
                        print(f"  [INCOMPATIBLE_RESUME] Job {job_id} partial checkpoint incompatible with new training inputs/config; retraining fresh...")
                        last_pt.unlink(missing_ok=True)
                        best_pt.unlink(missing_ok=True)

                if "MLP" in arch.upper():
                    model = MLPAnnual(seed=seed)
                    tx, vx, ex = fold_data["train_x_mlp"], fold_data["val_x_mlp"], fold_data["eval_x_mlp"]
                    dx = fold_data["dev_x_mlp"]
                else:
                    model = TransformerAnnual(seed=seed, dropout=float(neural_cfg.get("transformer_dropout", 0.1)))
                    tx, vx, ex = fold_data["train_x_trans"], fold_data["val_x_trans"], fold_data["eval_x_trans"]
                    dx = fold_data["dev_x_trans"]

                print(f"  [TRAINING] Dispatching {job_id} on {device} (Mode: {execution_mode})...")
                # Record training identity before running
                with open(training_ident_file, "w", encoding="utf-8") as f:
                    f.write(to_canonical_json(current_job_training_identity))
                try:
                    trained_model, summary = train_backbone_model(
                        model=model,
                        train_x=tx,
                        train_y=fold_data["train_y"],
                        train_markets=fold_data["train_markets"],
                        val_x=vx,
                        val_y=fold_data["val_y"],
                        val_markets=fold_data["val_markets"],
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
                    eval_preds = trained_model(torch.tensor(ex, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy().tolist() if len(ex) > 0 else []
                    dev_preds = trained_model(torch.tensor(dx, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy().tolist() if len(dx) > 0 else []

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

                c_data = {
                    "job_id": job_id,
                    "fold_year": fold_year,
                    "architecture": arch,
                    "seed": seed,
                    "status": "STAGE_COMPLETED",
                    "training_identity": current_job_training_identity,
                    "artifacts": {
                        "best_checkpoint": {"path": str(best_pt), "sha256": best_sha},
                        "last_checkpoint": {"path": str(last_pt), "sha256": last_sha},
                        "predictions": {"path": str(preds_json), "sha256": preds_sha},
                    },
                    "summary": asdict(summary) if hasattr(summary, "__dataclass_fields__") else summary,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                with open(completion_marker, "w", encoding="utf-8") as f:
                    f.write(to_canonical_json(c_data))

                jobs_completed.append(job_id)
                completed_checkpoint_hashes[job_id] = best_sha
                trained_models[f"{arch}_seed{seed}"] = trained_model

        trained_models_by_year[fold_year] = trained_models

        # Fit Ridge baseline on training set
        print(f"  [Fold {fold_year}] Fitting RIDGE_ANNUAL baseline...")
        ridge_model = fit_ridge_model(
            fold_data["train_x_mlp"],
            fold_data["train_y"],
            market_labels=fold_data["train_markets"],
            l2_lambda=0.001,
        )
        ridge_models_by_year[fold_year] = ridge_model

    if context.get("interrupt_at_macro_step") is not None:
        return {
            "status": "INTERRUPTED",
            "jobs_started": jobs_started,
            "jobs_completed": jobs_completed,
            "jobs_resumable": jobs_resumable,
        }

    # -------------------------------------------------------------------------
    # Stage 2: Policy Forecast Generation & Sealing
    # -------------------------------------------------------------------------
    print("\n=== Stage 2: Policy Forecast Generation & Sealing ===")
    for fold_year in folds_to_run:
        fold_dir = output_dir / f"fold_{fold_year}"
        fold_data = fold_data_by_year[fold_year]
        t_models = trained_models_by_year[fold_year]
        r_model = ridge_models_by_year[fold_year]

        p_records = generate_and_seal_policy_predictions(
            fold_year=fold_year,
            fold_dir=fold_dir,
            fold_data=fold_data,
            trained_models=t_models,
            ridge_model=r_model,
            device=device,
            config=config,
            configured_policies=context.get("selected_policies"),
            seeds=seeds,
            custom_predictions=context.get("custom_predictions"),
            fail_on_corrupted_predictions=bool(context.get("fail_on_corrupted_predictions", False)),
            checkpoint_hashes=completed_checkpoint_hashes,
            code_revision=current_run_identity["code_revision"],
        )
        predictions_by_year[fold_year] = p_records

        fold_summary_path = fold_dir / "fold_summary.json"
        fold_summary_data = {
            "fold_year": fold_year,
            "status": "FOLD_COMPLETED",
            "eval_queries_count": len(fold_data.get("eval_records", [])),
            "bank_records_count": len(fold_data.get("bank_records", [])),
            "predictions_count": len(p_records),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(fold_summary_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json(fold_summary_data))

    # -------------------------------------------------------------------------
    # Stage 3: Continuous Portfolio Simulation & Ledger Execution
    # -------------------------------------------------------------------------
    print("\n=== Stage 3: Continuous Portfolio Simulation & Ledger Execution ===")
    if context.get("induce_downstream_error"):
        raise RuntimeError("Induced downstream portfolio error during simulation")

    port_res = run_continuous_portfolio_simulation(
        folds_to_run=folds_to_run,
        fold_data_by_year=fold_data_by_year,
        predictions_by_year=predictions_by_year,
        output_dir=output_dir,
        initial_capital=float(config.get("execution", {}).get("initial_capital_account_units", 100000.0)),
        commission=float(config.get("execution", {}).get("commission_per_side", 0.001)),
        slippage=float(config.get("execution", {}).get("slippage_per_side", 0.0005)),
        max_positions=int(config.get("execution", {}).get("max_positions", 3)),
    )

    policy_accounts = port_res["policy_accounts"]
    union_sessions = port_res["union_sessions"]
    aligned_returns = port_res["aligned_returns"]
    market_open_mask = port_res["market_open_mask"]

    # -------------------------------------------------------------------------
    # Stage 4: Statistical Inference & Primary Contrasts
    # -------------------------------------------------------------------------
    print("\n=== Stage 4: Statistical Inference & Primary Contrasts ===")
    analysis_cfg = config.get("analysis", {})
    unique_markets = sorted(list(set(k[1] for k in policy_accounts.keys())))

    # In full production, disable reduced-arm mode and require complete coverage
    if not allow_reduced_arms:
        expected_markets = {"Brazil", "China", "France", "India", "UK", "US"}
        actual_markets = set(unique_markets)
        missing_m = expected_markets - actual_markets
        if missing_m:
            raise ValueError(f"Full production requires complete market coverage. Missing: {missing_m}")
        if len(policy_accounts) < 204:
            raise ValueError(f"Full production requires 204 continuous market paths. Found: {len(policy_accounts)}")

    contrast_results, draw_matrix, sampled_weeks = evaluate_primary_contrasts(
        returns_by_arm_market_realization=aligned_returns,
        markets=unique_markets,
        session_dates=union_sessions,
        valid_mask=market_open_mask,
        allow_reduced_arms=allow_reduced_arms,
        num_draws=int(analysis_cfg.get("bootstrap_draws", 100)),
        seed=42,
    )

    export_analysis_bundle(
        returns_by_key={f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in aligned_returns.items()},
        contrast_results=contrast_results,
        draw_matrix=draw_matrix,
        draw_week_indices=sampled_weeks,
        export_dir=release_analysis_dir,
        session_dates=union_sessions,
        valid_mask=market_open_mask,
        markets=unique_markets,
        num_draws=int(analysis_cfg.get("bootstrap_draws", 100)),
        seed=42,
        allow_reduced_arms=allow_reduced_arms,
    )

    # -------------------------------------------------------------------------
    # Stage 5: Independent Replay Verification
    # -------------------------------------------------------------------------
    print("\n=== Stage 5: Independent Replay Verification ===")
    replay_report = replay_analysis_bundle(
        export_dir=release_analysis_dir,
        tolerance=float(analysis_cfg.get("analysis_replay_tolerance", 1e-10)),
        allow_reduced_arms=allow_reduced_arms,
    )
    print(f"  Replay Verification: {replay_report.get('status')}")
    with open(release_analysis_dir / "replay_report.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(replay_report))

    # Write overall pipeline completion marker atomically
    all_done = (len(jobs_failed) == 0 and len(jobs_started) == len(jobs_completed))
    status = "PRODUCTION_SUCCESS" if all_done else "PRODUCTION_FAILED"

    final_run_identity = compute_scientific_run_identity(
        config=config,
        folds_to_run=folds_to_run,
        context=context,
        checkpoint_hashes=completed_checkpoint_hashes,
        calendar_hashes=calendar_hashes,
        code_revision=current_run_identity["code_revision"],
    )

    # Coverage verification check for release validation (Section 7)
    coverage_manifest = port_res.get("coverage_manifest", [])
    cov_val_res = validate_release_coverage_and_accounting(
        coverage_manifest=coverage_manifest,
        fold_data_by_year=fold_data_by_year,
        folds_to_run=folds_to_run,
    )
    total_admitted_queries = cov_val_res["total_admitted_queries"]
    total_expected_forecasts = cov_val_res["total_expected_forecasts"]
    total_sealed_forecasts = cov_val_res["total_sealed_forecasts"]
    missing_forecasts = total_expected_forecasts - total_sealed_forecasts

    pipeline_summary = {
        "status": status,
        "pipeline_stage": "RELEASE_VERIFIED",
        "study_scope": study_scope,
        "run_identity": final_run_identity,
        "permissive_calendar_fallbacks": permissive_calendar_fallbacks,
        "configured_folds": all_configured_folds,
        "executed_folds": folds_to_run,
        "jobs_started": jobs_started,
        "jobs_completed": jobs_completed,
        "jobs_resumable": jobs_resumable,
        "jobs_failed": jobs_failed,
        "accounts_simulated": len(policy_accounts),
        "union_sessions": len(union_sessions),
        "coverage_summary": {
            "total_admitted_queries": total_admitted_queries,
            "total_expected_forecasts": total_expected_forecasts,
            "total_sealed_forecasts": total_sealed_forecasts,
            "missing_forecasts": missing_forecasts,
        },
        "primary_contrasts_evaluated": len(contrast_results),
        "replay_report": replay_report,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(pipeline_completion_file, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(pipeline_summary))
    with open(output_dir / "pipeline_completion.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(pipeline_summary))

    cov_path = output_dir / "coverage_manifest.json"
    rel_artifacts = {
        "pipeline_summary.json": hashlib.sha256((output_dir / "pipeline_summary.json").read_bytes()).hexdigest(),
        "runtime_config.authorized.json": hashlib.sha256(runtime_cfg_path.read_bytes()).hexdigest(),
    }
    if cov_path.exists():
        rel_artifacts["coverage_manifest.json"] = hashlib.sha256(cov_path.read_bytes()).hexdigest()

    with open(output_dir / "release_manifest.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json({
            "status": "RELEASE_VERIFIED",
            "study_scope": study_scope,
            "artifacts": rel_artifacts,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }))

    return pipeline_summary

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
    if cuda_info["cuda_available"]:
        print(f"  CUDA Device: {cuda_info['device_name']} (Index 0)")
        print(f"  Compute Capability: {cuda_info['compute_capability']} | Total VRAM: {cuda_info['total_vram_gb']} GB")
        print(f"  PyTorch: {cuda_info['pytorch_version']} | CUDA Runtime: {cuda_info['cuda_version']}")
        print(f"  Compute Probe: {cuda_info['compute_probe']} ({cuda_info.get('compute_probe_elapsed_ms')} ms)")
    else:
        print("  [WARN] CUDA is not available on this host. Running CPU fallback.")

    # 3. Storage & Filesystem Health
    print("\n[Step 2/4] Verifying Storage & Manifest Health...")
    if data_dir is None:
        cand_data = REPO_ROOT / "FINAL_SUBMISSION_PACKAGE" / "data" / "cache" / "ohlcv"
        if not cand_data.exists():
            cand_data = REPO_ROOT / "data" / "cache" / "ohlcv"
        data_dir = cand_data
    if sample_ids_dir is None:
        sample_ids_dir = REPO_ROOT / "rebuild_plan" / "sample_ids"

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

    storage_errors = []
    if not storage_info["universe_manifest_present"]:
        storage_errors.append(f"Missing universe manifest at {storage_info['universe_manifest_path']}")
    if not storage_info["sample_ids_present"]:
        storage_errors.append(f"Insufficient sample ID manifests in {storage_info['sample_ids_dir']} (found {storage_info['sample_id_files_count']}, required >= 6)")
    if not storage_info["output_dir_writable"]:
        storage_errors.append(f"Output directory not writable: {storage_info.get('output_dir_error')}")

    receipt_path = output_dir / "a30_launch_receipt.json"

    if storage_errors:
        print("  [ERROR] Storage verification failed:")
        for err in storage_errors:
            print(f"    - {err}")
        err_msg = "; ".join(storage_errors)
        receipt = {
            "receipt_type": "A30_PRODUCTION_LAUNCH_RECEIPT",
            "status": "PREFLIGHT_STORAGE_ERROR",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
            "host_telemetry": host_telemetry,
            "cuda_preflight": cuda_info,
            "storage_checks": storage_info,
            "storage_errors": storage_errors,
        }
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
            execution_result["status"] = driver_result.get("status", "PRODUCTION_SUCCESS")
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

    # 6. Final Telemetry & Receipt
    elapsed_total = round(time.perf_counter() - t_start, 3)
    receipt_status = "PREFLIGHT_PASS" if (check_only or authorize_production) else "LOCKED_AWAITING_AUTHORIZATION"
    if execution_result.get("status") == "PRODUCTION_FAILED":
        receipt_status = "PRODUCTION_FAILED"

    receipt = {
        "receipt_type": "A30_PRODUCTION_LAUNCH_RECEIPT",
        "status": receipt_status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_total,
        "host_telemetry": host_telemetry,
        "cuda_preflight": cuda_info,
        "storage_checks": storage_info,
        "authorization": auth_info,
        "execution": execution_result,
    }

    try:
        with open(receipt_path, "w", encoding="utf-8") as f:
            f.write(to_canonical_json(receipt))
        print(f"\nDeployment receipt saved to: {receipt_path}")
    except Exception as e:
        print(f"\n[WARN] Failed to write receipt: {e}")

    print("=" * 70)
    return receipt


def main():
    parser = argparse.ArgumentParser(description="NVIDIA A30 / Production Launch Preflight and Dispatch Tool")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "rebuild_plan" / "config.proposed.json", help="Path to config.proposed.json")
    parser.add_argument("--data-dir", type=Path, default=None, help="Path to raw OHLCV parquet cache")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "rebuild_plan" / "a30_production_out", help="Output directory for production artifacts")
    parser.add_argument("--sample-ids-dir", type=Path, default=None, help="Directory containing fold_*_sample_ids.json")
    parser.add_argument("--authorize-production", action="store_true", help="Explicit operator flag authorizing production training")
    parser.add_argument("--run-pilot", action="store_true", help="Execute integrated connected pilot instead of full production")
    parser.add_argument("--check-only", action="store_true", help="Run preflight hardware and storage checks only without dispatching training")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="Optional subset of fold evaluation years to run")
    parser.add_argument("--securities", type=str, nargs="+", default=None, help="Optional subset of securities to run")
    parser.add_argument("--mode", type=str, default="production", choices=["production", "pilot"], help="Execution mode (production enforces all strict contracts)")

    args = parser.parse_args()

    try:
        receipt = launch_a30_deployment(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            sample_ids_dir=args.sample_ids_dir,
            authorize_production=args.authorize_production,
            run_connected_pilot=args.run_pilot,
            check_only=args.check_only,
            selected_folds=args.folds,
            selected_securities=args.securities,
            execution_mode=args.mode,
        )
        if receipt["status"] not in ("PREFLIGHT_PASS", "LOCKED_AWAITING_AUTHORIZATION"):
            sys.exit(1)
        if not args.authorize_production and not args.check_only and not args.run_pilot:
            sys.exit(1)
    except Exception as exc:
        print(f"Launch aborted due to fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
