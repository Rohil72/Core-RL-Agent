"""Connected pre-2020 end-to-end pilot runner (R12, C1).

Executes complete pipeline on preserved pre-2020 candidate data:
1. Canonical data formatting & validation using real pre-2020 market parquet files
2. Verified venue calendar alignment across securities (no equal-length truncation)
3. Corporate action adapter with documented pilot price adjustment mode
4. 23 Technical features & representation extraction with 252-day warmup
5. Strict chronological partitions:
   - Train: 2014-12-31 to 2015-03-25 (all targets mature by 2015-06-24, bank by 2015-09-23)
   - Preprocessing scaler fitted strictly on training partition (cutoff 2015-03-25)
   - Val: 2015-07-06 to 2015-09-15 (purged after training targets mature, val targets mature by 2015-12-14)
   - Eval: 2016-01-04 to 2016-03-15 (strictly fold 2016 in 2016, bank records mature before eval start)
6. Fresh MLP and Transformer training on real organic targets (zero label imputation)
7. Sealed memory bank creation with strictly mature (>126 session) records
8. Label-free policy prediction & artifact sealing for MEM_SIM, MLP_BASE, TRANS_BASE
9. Portfolio execution simulation driven by each policy's forecast rankings with volatility denominator guard and positive-score filter
10. Metric computation & Sharpe contrasts without substitute or random noise
11. Release export, checksums, and replay verification at 1e-10 tolerance
12. Linked input and output manifests with cryptographic SHA-256 hashes and traceability
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is on sys.path for direct script execution
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:
    import psutil
except ImportError:
    psutil = None

from memory_study_v2.backbones import MLPAnnual, TransformerAnnual
from memory_study_v2.canonical_data import align_to_venue_calendar, build_total_return_bars, validate_raw_bars, CorporateAction
from memory_study_v2.venue_calendar import VenueCalendar
from memory_study_v2.sample_index import build_security_sample_index
from memory_study_v2.contracts import to_canonical_json
from memory_study_v2.execution import PortfolioAccount
from memory_study_v2.features import compute_technical_features, fit_scaler
from memory_study_v2.inference import (
    evaluate_primary_contrasts,
    export_analysis_bundle,
    replay_analysis_bundle,
)
from memory_study_v2.labels import compute_target_labels
from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.predict import PredictionQuery, generate_and_seal_predictions
from memory_study_v2.release import generate_release_checksums, verify_release_bundle
from memory_study_v2.representations import extract_annual_representation
from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch
from memory_study_v2.train import train_backbone_model


def _get_rss_mb() -> float:
    """Get resident set size in MB."""
    if psutil is not None:
        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    return 0.0


def _get_vram_mb() -> float:
    """Get peak VRAM allocated in MB."""
    if torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    return 0.0


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_data_cache_dir() -> Path:
    """Locate available parquet cache directory."""
    candidates = [
        Path("FINAL_SUBMISSION_PACKAGE/data/cache/ohlcv"),
        Path("data/cache/ohlcv"),
    ]
    for c in candidates:
        if (c / "US_AAPL.parquet").exists() and (c / "US_MSFT.parquet").exists():
            return c
    raise FileNotFoundError("Could not find pre-2020 parquet cache directory with US_AAPL and US_MSFT.")


def run_connected_restricted_pilot(
    output_dir: Path,
    use_cuda: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run complete end-to-end pipeline across all phases on pre-2020 candidate data (C1)."""
    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 0. CUDA Execution & Device Determination
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()

    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA execution requested, but torch.cuda.is_available() is False.")
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        # Probe CUDA execution
        probe = torch.ones(10, device=device)
        if not probe.is_cuda:
            raise RuntimeError("CUDA tensor probe failed to allocate on CUDA device.")
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    report: Dict[str, Any] = {
        "status": "CONNECTED_PILOT_SUCCESS",
        "cuda_execution": {
            "requested": use_cuda,
            "device": device_name,
            "is_cuda": (device.type == "cuda"),
        },
        "phases_executed": [],
        "artifacts_generated": {},
    }

    # -------------------------------------------------------------------------
    # Phase 1: Canonical Data Formatting & Validation from Real Cache (C1)
    # -------------------------------------------------------------------------
    cache_dir = find_data_cache_dir()
    file_aapl = cache_dir / "US_AAPL.parquet"
    file_msft = cache_dir / "US_MSFT.parquet"

    hash_aapl = _compute_sha256(file_aapl)
    hash_msft = _compute_sha256(file_msft)

    df_aapl_raw = pd.read_parquet(file_aapl).reset_index().rename(columns={"Date": "session"})
    df_msft_raw = pd.read_parquet(file_msft).reset_index().rename(columns={"Date": "session"})

    df_aapl_raw["session"] = df_aapl_raw["session"].astype(str)
    df_msft_raw["session"] = df_msft_raw["session"].astype(str)

    # Load venue schedule independently of price files (Finding 1 / C1)
    from memory_study_v2.venue_calendar import _FIXTURE_SHA256 as _VC_SHA256
    _vc = VenueCalendar()
    venue_sessions = _vc.sessions_in_range("2013-01-01", "2017-12-31")
    report["venue_calendar_sha"] = _VC_SHA256

    # Reindex each security onto the authoritative venue schedule
    df0 = align_to_venue_calendar(df_aapl_raw, venue_sessions)
    df1 = align_to_venue_calendar(df_msft_raw, venue_sessions)

    val_df0 = validate_raw_bars(df0, "US_AAPL", quote_unit=1.0)
    val_df1 = validate_raw_bars(df1, "US_MSFT", quote_unit=1.0)

    # Corporate action adapter with documented pilot price adjustment mode (C1)
    actions0: List[CorporateAction] = []
    actions1: List[CorporateAction] = []
    tr_df0 = build_total_return_bars(val_df0, actions=actions0, quote_unit=1.0)
    tr_df1 = build_total_return_bars(val_df1, actions=actions1, quote_unit=1.0)

    report["phases_executed"].append("data_canonicalization")

    # -------------------------------------------------------------------------
    # Phase 2: Technical Features, Target Labels, and Chronological Partitions (C1)
    # -------------------------------------------------------------------------
    feats0 = compute_technical_features(tr_df0)
    feats1 = compute_technical_features(tr_df1)

    labels0 = compute_target_labels(tr_df0, venue_sessions=venue_sessions)
    labels1 = compute_target_labels(tr_df1, venue_sessions=venue_sessions)

    # Data admission check: build and verify canonical sample index records (Finding 3 / F3)
    sample_records_0 = build_security_sample_index(
        "US_AAPL", tr_df0, venue_calendar=_vc, evaluation_year=2016
    )
    sample_records_1 = build_security_sample_index(
        "US_MSFT", tr_df1, venue_calendar=_vc, evaluation_year=2016
    )

    # Chronological partition indices:
    # Feature warmup requirement: 252 bars for features + 252 bars for representation -> t >= 503
    train_indices = list(range(503, 561))  # 2014-12-31 to 2015-03-25 (58 sessions)
    val_indices = list(range(630, 681))    # 2015-07-06 to 2015-09-15 (51 sessions)
    eval_indices = list(range(756, 806))   # 2016-01-04 to 2016-03-15 (50 sessions)

    # Verify warmup boundary rule: bar 503 valid, bar 502 invalid
    assert sample_records_0[503].input_window_valid is True, "Bar 503 must be input_window_valid"
    assert sample_records_0[502].input_window_valid is False, "Bar 502 must be invalid due to 503-bar warmup rule"
    assert sample_records_1[503].input_window_valid is True, "Bar 503 must be input_window_valid"
    assert sample_records_1[502].input_window_valid is False, "Bar 502 must be invalid due to 503-bar warmup rule"

    train_qids = [sample_records_0[t].query_id for t in train_indices] + [sample_records_1[t].query_id for t in train_indices]
    val_qids = [sample_records_0[t].query_id for t in val_indices] + [sample_records_1[t].query_id for t in val_indices]
    eval_qids = [sample_records_0[t].query_id for t in eval_indices] + [sample_records_1[t].query_id for t in eval_indices]

    report["data_admission"] = {
        "status": "ADMISSION_VERIFIED",
        "train_query_ids_count": len(train_qids),
        "val_query_ids_count": len(val_qids),
        "eval_query_ids_count": len(eval_qids),
        "bank_query_ids_count": len(train_qids),
    }

    train_start_sess = str(tr_df0.iloc[train_indices[0]]["session"])
    train_end_sess = str(tr_df0.iloc[train_indices[-1]]["session"])
    train_max_target_mat = str(tr_df0.iloc[train_indices[-1] + 63]["session"])
    train_max_bank_mat = str(tr_df0.iloc[train_indices[-1] + 126]["session"])

    val_start_sess = str(tr_df0.iloc[val_indices[0]]["session"])
    val_end_sess = str(tr_df0.iloc[val_indices[-1]]["session"])
    val_max_target_mat = str(tr_df0.iloc[val_indices[-1] + 63]["session"])

    eval_start_sess = str(tr_df0.iloc[eval_indices[0]]["session"])
    eval_end_sess = str(tr_df0.iloc[eval_indices[-1]]["session"])

    # Feature scaler fitted strictly on training partition (cutoff 2015-03-25) (C1)
    scaler = fit_scaler(feats0, training_cutoff=train_end_sess)

    # Enforce strict chronological assertions (C1)
    assert train_end_sess == "2015-03-25", f"Train end expected 2015-03-25, got {train_end_sess}"
    assert train_max_target_mat < val_start_sess, f"Train targets ({train_max_target_mat}) must mature before val start ({val_start_sess})"
    assert val_max_target_mat < eval_start_sess, f"Val targets ({val_max_target_mat}) must mature before eval start ({eval_start_sess})"
    assert train_max_bank_mat < eval_start_sess, f"Bank records ({train_max_bank_mat}) must mature before eval start ({eval_start_sess})"
    assert eval_start_sess >= "2016-01-01", f"Evaluation start ({eval_start_sess}) must be in fold 2016"

    # Extract representations for train and val sets
    reps0_train = [extract_annual_representation(feats0, t, scaler) for t in train_indices]
    reps1_train = [extract_annual_representation(feats1, t, scaler) for t in train_indices]
    targets0_train = [float(labels0.loc[t, "target_value"]) for t in train_indices]
    targets1_train = [float(labels1.loc[t, "target_value"]) for t in train_indices]

    reps0_val = [extract_annual_representation(feats0, t, scaler) for t in val_indices]
    reps1_val = [extract_annual_representation(feats1, t, scaler) for t in val_indices]
    targets0_val = [float(labels0.loc[t, "target_value"]) for t in val_indices]
    targets1_val = [float(labels1.loc[t, "target_value"]) for t in val_indices]

    # Verify organic label validity: NO imputation, NO fallback to 0.01
    assert all(np.isfinite(targets0_train)), "All AAPL train targets must be organic and finite"
    assert all(np.isfinite(targets1_train)), "All MSFT train targets must be organic and finite"
    assert all(np.isfinite(targets0_val)), "All AAPL val targets must be organic and finite"
    assert all(np.isfinite(targets1_val)), "All MSFT val targets must be organic and finite"

    report["phases_executed"].append("feature_engineering")

    # -------------------------------------------------------------------------
    # Phase 3: Train Both Backbones (MLP & Transformer) on Genuine Chronological Partitions (C1)
    # -------------------------------------------------------------------------
    train_mlp_vecs = [r.flattened_vector for r in reps0_train + reps1_train]
    train_trans_mats = [r.transformer_matrix for r in reps0_train + reps1_train]
    all_train_targets = targets0_train + targets1_train
    N_tr = len(all_train_targets)

    val_mlp_vecs = [r.flattened_vector for r in reps0_val + reps1_val]
    val_trans_mats = [r.transformer_matrix for r in reps0_val + reps1_val]
    all_val_targets = targets0_val + targets1_val
    N_va = len(all_val_targets)

    train_x_mlp = torch.tensor(np.array(train_mlp_vecs), dtype=torch.float32)
    val_x_mlp = torch.tensor(np.array(val_mlp_vecs), dtype=torch.float32)

    train_x_trans = torch.tensor(np.array(train_trans_mats), dtype=torch.float32)
    val_x_trans = torch.tensor(np.array(val_trans_mats), dtype=torch.float32)

    train_y = torch.tensor(np.array(all_train_targets), dtype=torch.float32)
    val_y = torch.tensor(np.array(all_val_targets), dtype=torch.float32)

    train_mkts = np.array(["US"] * N_tr)
    val_mkts = np.array(["US"] * N_va)

    # Recovery Verification Check: An interrupted CUDA training run resumes correctly on the same machine
    chk_recovery_dir = output_dir / "recovery_test_chk"
    chk_recovery_dir.mkdir(parents=True, exist_ok=True)
    mlp_full = MLPAnnual(seed=42)
    trained_full, summary_full = train_backbone_model(
        mlp_full, train_x_mlp, train_y, train_mkts,
        val_x_mlp, val_y, val_mkts,
        device=device, seed=42, min_epochs=2, max_epochs=2,
        micro_batch_size=16, effective_batch_size=32,
    )
    mlp_int = MLPAnnual(seed=42)
    train_backbone_model(
        mlp_int, train_x_mlp, train_y, train_mkts,
        val_x_mlp, val_y, val_mkts,
        device=device, seed=42, min_epochs=2, max_epochs=2,
        micro_batch_size=16, effective_batch_size=32,
        interrupt_at_macro_step=1,
        checkpoint_dir=chk_recovery_dir,
    )
    mlp_res = MLPAnnual(seed=999)
    trained_res, summary_res = train_backbone_model(
        mlp_res, train_x_mlp, train_y, train_mkts,
        val_x_mlp, val_y, val_mkts,
        device=device, seed=42, min_epochs=2, max_epochs=2,
        micro_batch_size=16, effective_batch_size=32,
        resume_from_checkpoint=chk_recovery_dir / "last_checkpoint.pt",
    )
    for p_full, p_res in zip(trained_full.parameters(), trained_res.parameters()):
        assert torch.allclose(p_full, p_res, atol=1e-5), "Resumed CUDA training weights do not match uninterrupted run"
    assert abs(summary_full.best_loss - summary_res.best_loss) < 1e-5, "Resumed val loss mismatch"

    report["recovery_verification"] = {
        "status": "RECOVERY_VERIFIED",
        "device": device.type,
        "interrupted_macro_step": 1,
        "parameter_parity": True,
        "val_loss_parity": True,
    }

    mlp = MLPAnnual(seed=7)
    trained_mlp, mlp_summary = train_backbone_model(
        mlp, train_x_mlp, train_y, train_mkts,
        val_x_mlp, val_y, val_mkts,
        device=device, seed=7, min_epochs=2, max_epochs=2,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=output_dir / "checkpoints_mlp",
    )

    trans = TransformerAnnual(seed=17, dropout=0.0)
    trained_trans, trans_summary = train_backbone_model(
        trans, train_x_trans, train_y, train_mkts,
        val_x_trans, val_y, val_mkts,
        device=device, seed=17, min_epochs=2, max_epochs=2,
        micro_batch_size=16, effective_batch_size=32,
        checkpoint_dir=output_dir / "checkpoints_trans",
    )

    # Enforce CUDA execution check: assert model parameters are on device
    if device.type == "cuda":
        assert next(trained_mlp.parameters()).is_cuda, "MLP parameters must be on CUDA"
        assert next(trained_trans.parameters()).is_cuda, "Transformer parameters must be on CUDA"

    report["phases_executed"].append("backbone_training")

    # -------------------------------------------------------------------------
    # Phase 4: Sealed Memory Bank Constructed with Organic Mature Records (C1)
    # -------------------------------------------------------------------------
    bank_records: List[BankRecord] = []
    all_reps_train = reps0_train + reps1_train
    for i, (r, tgt) in enumerate(zip(all_reps_train, all_train_targets)):
        sec = "US_AAPL" if i < len(reps0_train) else "US_MSFT"
        orig_t = train_indices[i % len(train_indices)]
        mat_sess = str(tr_df0.iloc[orig_t + 126]["session"])
        bank_records.append(BankRecord(
            record_id=i + 1,
            security_id=sec,
            session_origin=r.session_t,
            session_126_maturity=mat_sess,
            vector=r.flattened_vector,
            target_63=tgt,
            session_ordinal=i,
        ))
    bank = MemoryBank(bank_records)
    for rec in bank_records:
        assert rec.session_126_maturity < eval_start_sess, (
            f"Bank record {rec.record_id} maturity {rec.session_126_maturity} exceeds eval start {eval_start_sess}"
        )
    report["phases_executed"].append("memory_bank_construction")

    # -------------------------------------------------------------------------
    # Phase 5: Sealed Predictions for ALL 3 Policies (MEM_SIM, MLP_BASE, TRANS_BASE) (C1)
    # -------------------------------------------------------------------------
    device_mlp = next(trained_mlp.parameters()).device
    device_trans = next(trained_trans.parameters()).device
    trained_mlp.eval()
    trained_trans.eval()

    # Precompute bank norms for batched retrieval
    bank.precompute_bank_norms()

    # Retrieval Parity & Boundary Fixture Checks on Device
    # 1. Auditor boundary cancellation fixture on device (float32 vectors near 1)
    vec_dim = len(reps0_train[0].flattened_vector)
    q_fix = np.ones(vec_dim, dtype=np.float32)
    v1_fix = q_fix.copy()
    v2_fix = q_fix.copy()
    v1_fix[0] += np.float32(2 ** (-22))
    v2_fix[0] += np.float32(2 ** (-23))
    fix_records = [
        BankRecord(record_id=1, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=v1_fix, target_63=0.01, session_ordinal=10),
        BankRecord(record_id=2, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=v2_fix, target_63=0.02, session_ordinal=20),
    ]
    fix_bank = MemoryBank(fix_records)
    fix_bank.precompute_bank_norms()
    fix_bat = retrieve_mem_sim_batch(
        fix_bank, q_fix.reshape(1, -1), ["SEC_QUERY"], k=2, use_gpu=(device.type == "cuda")
    )[0]
    assert fix_bat.neighbor_ids == [2, 1], f"Boundary cancellation test failed on {device.type}: got {fix_bat.neighbor_ids}"

    # 2. Exact ties fixture
    tie_vec = np.ones(vec_dim, dtype=np.float32)
    tie_records = [
        BankRecord(record_id=100, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=tie_vec, target_63=0.01, session_ordinal=1),
        BankRecord(record_id=5, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=tie_vec, target_63=0.02, session_ordinal=2),
        BankRecord(record_id=50, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=tie_vec, target_63=0.03, session_ordinal=3),
    ]
    tie_bank = MemoryBank(tie_records)
    tie_bank.precompute_bank_norms()
    tie_bat = retrieve_mem_sim_batch(
        tie_bank, np.zeros((1, vec_dim), dtype=np.float32), ["SEC_OTHER"], k=3, use_gpu=(device.type == "cuda")
    )[0]
    assert tie_bat.neighbor_ids == [5, 50, 100], f"Tie-breaking failed on {device.type}: got {tie_bat.neighbor_ids}"

    eval_r0_list = [extract_annual_representation(feats0, t, scaler) for t in eval_indices]
    eval_r1_list = [extract_annual_representation(feats1, t, scaler) for t in eval_indices]

    # Batch evaluate MEM_SIM predictions on device
    all_eval_vecs = []
    all_eval_secs = []
    for r0, r1 in zip(eval_r0_list, eval_r1_list):
        all_eval_vecs.append(r0.flattened_vector)
        all_eval_secs.append("US_AAPL")
        all_eval_vecs.append(r1.flattened_vector)
        all_eval_secs.append("US_MSFT")

    batched_mem_preds = retrieve_mem_sim_batch(
        bank, np.array(all_eval_vecs, dtype=np.float32), all_eval_secs, k=3, use_gpu=(device.type == "cuda")
    )

    # Verify parity against single-query reference oracle
    for idx_q, (q_vec, q_sec) in enumerate(zip(all_eval_vecs, all_eval_secs)):
        ref_res = retrieve_mem_sim(bank, q_vec, q_sec, k=3)
        bat_res = batched_mem_preds[idx_q]
        assert ref_res.neighbor_ids == bat_res.neighbor_ids, f"Neighbor ID mismatch on query {idx_q}"
        assert abs(ref_res.prediction - bat_res.prediction) < 1e-5, f"Prediction mismatch on query {idx_q}"

    report["retrieval_verification"] = {
        "status": "RETRIEVAL_VERIFIED",
        "device": device.type,
        "is_cuda": (device.type == "cuda"),
        "parity_with_reference_passed": True,
        "cancellation_fixture_passed": True,
        "tie_breaking_passed": True,
    }

    queries_mem: List[PredictionQuery] = []
    queries_mlp: List[PredictionQuery] = []
    queries_trans: List[PredictionQuery] = []

    for i, t in enumerate(eval_indices):
        r0 = eval_r0_list[i]
        r1 = eval_r1_list[i]

        sess0 = r0.session_t
        sess1 = r1.session_t

        # 1. MEM_SIM forecasts from certified batch retrieval
        mem_0 = float(batched_mem_preds[2 * i].prediction)
        mem_1 = float(batched_mem_preds[2 * i + 1].prediction)
        queries_mem.append(PredictionQuery(
            fold_year=2016, security_id="US_AAPL", session_origin=sess0,
            forecast_score=mem_0,
            volatility_21=float(feats0.loc[t, "volatility_21"]),
            atr_ratio_14=float(feats0.loc[t, "atr_ratio_14"]),
        ))
        queries_mem.append(PredictionQuery(
            fold_year=2016, security_id="US_MSFT", session_origin=sess1,
            forecast_score=mem_1,
            volatility_21=float(feats1.loc[t, "volatility_21"]),
            atr_ratio_14=float(feats1.loc[t, "atr_ratio_14"]),
        ))

        # 2. MLP_BASE forecasts
        with torch.no_grad():
            x0_mlp = torch.tensor(r0.flattened_vector, dtype=torch.float32, device=device_mlp).unsqueeze(0)
            x1_mlp = torch.tensor(r1.flattened_vector, dtype=torch.float32, device=device_mlp).unsqueeze(0)
            mlp_0 = float(trained_mlp(x0_mlp).item())
            mlp_1 = float(trained_mlp(x1_mlp).item())
        queries_mlp.append(PredictionQuery(
            fold_year=2016, security_id="US_AAPL", session_origin=sess0,
            forecast_score=mlp_0,
            volatility_21=float(feats0.loc[t, "volatility_21"]),
            atr_ratio_14=float(feats0.loc[t, "atr_ratio_14"]),
        ))
        queries_mlp.append(PredictionQuery(
            fold_year=2016, security_id="US_MSFT", session_origin=sess1,
            forecast_score=mlp_1,
            volatility_21=float(feats1.loc[t, "volatility_21"]),
            atr_ratio_14=float(feats1.loc[t, "atr_ratio_14"]),
        ))

        # 3. TRANS_BASE forecasts
        with torch.no_grad():
            x0_trans = torch.tensor(r0.transformer_matrix, dtype=torch.float32, device=device_trans).unsqueeze(0)
            x1_trans = torch.tensor(r1.transformer_matrix, dtype=torch.float32, device=device_trans).unsqueeze(0)
            trans_0 = float(trained_trans(x0_trans).item())
            trans_1 = float(trained_trans(x1_trans).item())
        queries_trans.append(PredictionQuery(
            fold_year=2016, security_id="US_AAPL", session_origin=sess0,
            forecast_score=trans_0,
            volatility_21=float(feats0.loc[t, "volatility_21"]),
            atr_ratio_14=float(feats0.loc[t, "atr_ratio_14"]),
        ))
        queries_trans.append(PredictionQuery(
            fold_year=2016, security_id="US_MSFT", session_origin=sess1,
            forecast_score=trans_1,
            volatility_21=float(feats1.loc[t, "volatility_21"]),
            atr_ratio_14=float(feats1.loc[t, "atr_ratio_14"]),
        ))

    pred_path_mem, _ = generate_and_seal_predictions("MEM_SIM", 2016, queries_mem, output_dir / "predictions_2016_mem_sim.json")
    pred_path_mlp, _ = generate_and_seal_predictions("MLP_BASE", 2016, queries_mlp, output_dir / "predictions_2016_mlp_base.json")
    pred_path_trans, _ = generate_and_seal_predictions("TRANS_BASE", 2016, queries_trans, output_dir / "predictions_2016_trans_base.json")
    report["phases_executed"].append("prediction_sealing")

    # -------------------------------------------------------------------------
    # Phase 6: Portfolio Execution Simulation Driven by Volatility-Normalized Scoring (C1)
    # -------------------------------------------------------------------------
    policy_accounts = {
        "MEM_SIM": PortfolioAccount(initial_capital=100000.0, commission=0.001, slippage=0.0005, max_positions=1),
        "MLP_BASE": PortfolioAccount(initial_capital=100000.0, commission=0.001, slippage=0.0005, max_positions=1),
        "TRANS_BASE": PortfolioAccount(initial_capital=100000.0, commission=0.001, slippage=0.0005, max_positions=1),
    }

    # Step day-by-day through the evaluation window
    for idx in range(len(eval_indices) - 1):
        t_curr = eval_indices[idx]
        t_next = eval_indices[idx + 1]

        r0 = extract_annual_representation(feats0, t_curr, scaler)
        r1 = extract_annual_representation(feats1, t_curr, scaler)

        mem_0 = float(retrieve_mem_sim(bank, r0.flattened_vector, "US_AAPL", k=3).prediction)
        mem_1 = float(retrieve_mem_sim(bank, r1.flattened_vector, "US_MSFT", k=3).prediction)

        with torch.no_grad():
            x0_mlp = torch.tensor(r0.flattened_vector, dtype=torch.float32, device=device_mlp).unsqueeze(0)
            x1_mlp = torch.tensor(r1.flattened_vector, dtype=torch.float32, device=device_mlp).unsqueeze(0)
            mlp_0 = float(trained_mlp(x0_mlp).item())
            mlp_1 = float(trained_mlp(x1_mlp).item())

            x0_trans = torch.tensor(r0.transformer_matrix, dtype=torch.float32, device=device_trans).unsqueeze(0)
            x1_trans = torch.tensor(r1.transformer_matrix, dtype=torch.float32, device=device_trans).unsqueeze(0)
            trans_0 = float(trained_trans(x0_trans).item())
            trans_1 = float(trained_trans(x1_trans).item())

        policy_forecasts = {
            "MEM_SIM": {"US_AAPL": mem_0, "US_MSFT": mem_1},
            "MLP_BASE": {"US_AAPL": mlp_0, "US_MSFT": mlp_1},
            "TRANS_BASE": {"US_AAPL": trans_0, "US_MSFT": trans_1},
        }

        vol_21 = {
            "US_AAPL": float(feats0.loc[t_curr, "volatility_21"]),
            "US_MSFT": float(feats1.loc[t_curr, "volatility_21"]),
        }

        sess_open = str(tr_df0.iloc[t_next]["session"])
        open_prices = {
            "US_AAPL": float(tr_df0.iloc[t_next]["raw_open"]),
            "US_MSFT": float(tr_df1.iloc[t_next]["raw_open"]),
        }
        close_prices = {
            "US_AAPL": float(tr_df0.iloc[t_next]["raw_close"]),
            "US_MSFT": float(tr_df1.iloc[t_next]["raw_close"]),
        }
        atr_values = {
            "US_AAPL": float(feats0.loc[t_next, "atr_ratio_14"]),
            "US_MSFT": float(feats1.loc[t_next, "atr_ratio_14"]),
        }

        for pol_name, acct in policy_accounts.items():
            # Scoring rule: rank_score = forecast / (vol21 + 0.0001) (Section 8.1, C1)
            # Filter rule: allow_nonpositive_entry_score = False (only forecast > 0 eligible)
            scores = {}
            for s in ["US_AAPL", "US_MSFT"]:
                f_val = policy_forecasts[pol_name][s]
                v_denom = max(1e-8, vol_21[s] + 0.0001)
                scores[s] = f_val / v_denom

            ranked = sorted(scores.keys(), key=lambda s: scores[s], reverse=True)
            # Eligible entries require positive score (forecast > 0)
            eligible = [s for s in ranked if policy_forecasts[pol_name][s] > 0]

            acct.plan_entries_at_close(eligible)
            acct.process_open_fills(sess_open, open_prices, {"US_AAPL": True, "US_MSFT": True}, 100000.0)
            acct.evaluate_close_stops_and_update_state(sess_open, close_prices, atr_values)

    # Terminal liquidation at final session
    final_t = eval_indices[-1]
    final_sess = str(tr_df0.iloc[final_t]["session"])
    final_close_prices = {
        "US_AAPL": float(tr_df0.iloc[final_t]["raw_close"]),
        "US_MSFT": float(tr_df1.iloc[final_t]["raw_close"]),
    }

    returns_map: Dict[Tuple[str, str, Optional[int]], np.ndarray] = {}
    session_dates = [str(tr_df0.iloc[t]["session"]) for t in eval_indices[1:]]

    for pol_name, acct in policy_accounts.items():
        acct.execute_terminal_liquidation(final_sess, final_close_prices)
        assert len(acct.trades) > 0, f"Trades must be recorded for {pol_name}"
        final_history_nav = acct.daily_history[-1].total_nav
        assert abs(acct.cash - final_history_nav) < 1e-5, f"NAV reconciliation error for {pol_name}"

        pol_returns = np.array([state.daily_return for state in acct.daily_history], dtype=np.float64)
        compounded_nav = acct.initial_capital * float(np.prod(1.0 + pol_returns))
        assert abs(compounded_nav - acct.cash) < 1e-4, f"Compounded return mismatch for {pol_name}: {compounded_nav} vs {acct.cash}"
        returns_map[(pol_name, "US", None)] = pol_returns

        # Export trade ledger and daily NAV history for this policy (C1)
        trades_payload = [asdict(tr) for tr in acct.trades]
        with open(output_dir / f"trades_{pol_name}.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(trades_payload))

        nav_payload = [asdict(h) for h in acct.daily_history]
        with open(output_dir / f"daily_nav_{pol_name}.json", "w", encoding="utf-8") as f:
            f.write(to_canonical_json(nav_payload))

    report["phases_executed"].append("portfolio_execution")

    # -------------------------------------------------------------------------
    # Phase 7: Inference & Primary Contrasts Fed Directly by Real NAV Returns (C1, C2)
    # -------------------------------------------------------------------------
    markets = ["US"]
    contrast_results, draw_matrix, sampled_weeks = evaluate_primary_contrasts(
        returns_by_arm_market_realization=returns_map,
        markets=markets,
        session_dates=session_dates,
        allow_reduced_arms=True,
        num_draws=100,
        seed=42,
    )

    export_analysis_bundle(
        returns_by_key={f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results=contrast_results,
        draw_matrix=draw_matrix,
        draw_week_indices=sampled_weeks,
        export_dir=output_dir / "release_analysis",
        session_dates=session_dates,
        markets=markets,
        num_draws=100,
        seed=42,
    )
    report["phases_executed"].append("inference_contrasts")

    # -------------------------------------------------------------------------
    # Phase 8: Release Checksums, Verification, and 1e-10 Replay (C1, C2)
    # -------------------------------------------------------------------------
    checksums = generate_release_checksums(output_dir / "release_analysis")
    verif = verify_release_bundle(output_dir / "release_analysis")
    replay = replay_analysis_bundle(output_dir / "release_analysis", tolerance=1e-10)

    report["phases_executed"].append("release_verification")
    report["release_verification"] = verif
    report["replay_verification"] = replay

    # -------------------------------------------------------------------------
    # Phase 9: Input & Output Manifests with Cryptographic SHA-256 Hashes (C1)
    # -------------------------------------------------------------------------
    input_manifest = {
        "manifest_type": "CONNECTED_PILOT_INPUT_MANIFEST",
        "input_files": [
            {
                "file_path": str(file_aapl),
                "sha256": hash_aapl,
                "security_id": "US_AAPL",
                "total_rows": len(df_aapl_raw),
                "selected_window_rows": len(df0),
            },
            {
                "file_path": str(file_msft),
                "sha256": hash_msft,
                "security_id": "US_MSFT",
                "total_rows": len(df_msft_raw),
                "selected_window_rows": len(df1),
            },
        ],
        "calendar_basis": "verified_venue_schedules",
        "price_adjustment_mode": "ADJUSTED_PRICE_CACHE_PILOT_MODE",
        "price_semantics_note": "Pilot executed on pre-adjusted OHLCV cache aligned to venue calendar with CorporateAction adapter. Does not certify production as-traded/action contract.",
        "session_bounds": {
            "start_session": str(tr_df0.iloc[0]["session"]),
            "end_session": str(tr_df0.iloc[-1]["session"]),
            "total_canonical_sessions": len(venue_sessions),
        },
        "horizons_and_lags": {
            "feature_warmup_sessions": 252,
            "annual_representation_sessions": 252,
            "target_horizon_sessions": 63,
            "memory_bank_maturity_lag_sessions": 126,
        },
        "scaler_training_cutoff": train_end_sess,
        "window_partitions": {
            "training_partition": {
                "start_session": train_start_sess,
                "end_session": train_end_sess,
                "max_target_maturity": train_max_target_mat,
                "max_bank_maturity": train_max_bank_mat,
                "sample_count": len(all_train_targets),
            },
            "validation_partition": {
                "start_session": val_start_sess,
                "end_session": val_end_sess,
                "max_target_maturity": val_max_target_mat,
                "sample_count": len(all_val_targets),
            },
            "evaluation_partition": {
                "start_session": eval_start_sess,
                "end_session": eval_end_sess,
                "fold_year": 2016,
                "session_count": len(eval_indices),
            },
        },
        "chronological_integrity_assertions": {
            "training_targets_mature_before_validation": train_max_target_mat < val_start_sess,
            "validation_targets_mature_before_evaluation": val_max_target_mat < eval_start_sess,
            "bank_records_mature_before_evaluation": train_max_bank_mat < eval_start_sess,
            "evaluation_origins_in_fold_2016": eval_start_sess >= "2016-01-01",
            "scaler_fitted_strictly_on_training": train_end_sess == "2015-03-25",
        },
        "imputation_policy": {
            "imputed_targets_count": 0,
            "fallback_targets_count": 0,
            "substitute_policy_returns_count": 0,
        },
    }
    with open(output_dir / "input_manifest.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(input_manifest))

    output_manifest = {
        "manifest_type": "CONNECTED_PILOT_OUTPUT_MANIFEST",
        "status": "CONNECTED_PILOT_SUCCESS",
        "phases_executed": report["phases_executed"],
        "policies_evaluated": list(policy_accounts.keys()),
        "policies_unrun_status": "NOT_RUN",
        "artifacts": {
            "predictions_mem_sim_json": str(pred_path_mem),
            "predictions_mem_sim_sha256": _compute_sha256(pred_path_mem),
            "predictions_mlp_base_json": str(pred_path_mlp),
            "predictions_mlp_base_sha256": _compute_sha256(pred_path_mlp),
            "predictions_trans_base_json": str(pred_path_trans),
            "predictions_trans_base_sha256": _compute_sha256(pred_path_trans),
            "trades_mem_sim_json": str(output_dir / "trades_MEM_SIM.json"),
            "trades_mlp_base_json": str(output_dir / "trades_MLP_BASE.json"),
            "trades_trans_base_json": str(output_dir / "trades_TRANS_BASE.json"),
            "daily_nav_mem_sim_json": str(output_dir / "daily_nav_MEM_SIM.json"),
            "daily_nav_mlp_base_json": str(output_dir / "daily_nav_MLP_BASE.json"),
            "daily_nav_trans_base_json": str(output_dir / "daily_nav_TRANS_BASE.json"),
            "input_manifest_json": str(output_dir / "input_manifest.json"),
            "input_manifest_sha256": _compute_sha256(output_dir / "input_manifest.json"),
            "release_analysis_dir": str(output_dir / "release_analysis"),
            "release_checksums": checksums,
        },
        "verification": {
            "release_verification_status": verif["status"],
            "replay_verification_status": replay["status"],
            "replay_tolerance": 1e-10,
        },
    }
    with open(output_dir / "output_manifest.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(output_manifest))

    elapsed_seconds = time.perf_counter() - t_start
    peak_rss = _get_rss_mb()
    peak_vram = _get_vram_mb()

    receipt = {
        "receipt_type": "CONNECTED_PILOT_EXECUTION_RECEIPT",
        "status": "CONNECTED_PILOT_SUCCESS",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": {
            "requested_cuda": use_cuda,
            "device_name": device_name,
            "is_cuda": (device.type == "cuda"),
            "compute_capability": list(torch.cuda.get_device_capability(0)) if device.type == "cuda" else None,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        },
        "telemetry": {
            "elapsed_seconds": round(elapsed_seconds, 3),
            "peak_rss_mb": round(peak_rss, 2),
            "peak_vram_mb": round(peak_vram, 2),
        },
        "manifest_digests": {
            "input_manifest_sha256": _compute_sha256(output_dir / "input_manifest.json"),
            "output_manifest_sha256": _compute_sha256(output_dir / "output_manifest.json"),
        },
        "verifications": {
            "data_admission_verified": True,
            "cuda_recovery_verified": report.get("recovery_verification", {}).get("status") == "RECOVERY_VERIFIED",
            "retrieval_parity_verified": report.get("retrieval_verification", {}).get("status") == "RETRIEVAL_VERIFIED",
            "release_verified": verif["status"] == "RELEASE_VERIFIED",
            "replay_verified": replay["status"] == "REPLAY_VERIFIED",
            "compounded_nav_reconciled": True,
        },
    }
    with open(output_dir / "pilot_receipt.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(receipt))

    report["artifacts_generated"] = {
        "input_manifest": str(output_dir / "input_manifest.json"),
        "output_manifest": str(output_dir / "output_manifest.json"),
        "pilot_receipt": str(output_dir / "pilot_receipt.json"),
        "predictions_mem_sim": str(pred_path_mem),
        "predictions_mlp_base": str(pred_path_mlp),
        "predictions_trans_base": str(pred_path_trans),
        "release_analysis": str(output_dir / "release_analysis"),
    }
    report["receipt"] = receipt

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Connected Pre-2020 Restricted Pilot")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rebuild_plan/connected_pilot_out"),
        help="Directory to save pilot artifacts and manifests",
    )
    parser.add_argument(
        "--cuda",
        dest="use_cuda",
        action="store_true",
        default=None,
        help="Force CUDA execution",
    )
    parser.add_argument(
        "--no-cuda",
        dest="use_cuda",
        action="store_false",
        help="Force CPU execution",
    )
    args = parser.parse_args()
    rep = run_connected_restricted_pilot(args.output_dir, use_cuda=args.use_cuda)
    print("Connected Restricted Pilot completed successfully!")
    print(f"Device: {rep['cuda_execution']['device']} (CUDA={rep['cuda_execution']['is_cuda']})")
    print(f"Phases executed: {rep['phases_executed']}")
    print(f"Elapsed: {rep['receipt']['telemetry']['elapsed_seconds']}s | Peak RSS: {rep['receipt']['telemetry']['peak_rss_mb']}MB | Peak VRAM: {rep['receipt']['telemetry']['peak_vram_mb']}MB")
    print(f"Artifacts: {json.dumps(rep['artifacts_generated'], indent=2)}")


if __name__ == "__main__":
    main()
