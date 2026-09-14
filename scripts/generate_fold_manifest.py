"""Generate hashed admissible-row manifest and walk-forward fold dimensions (C6, Finding 3).

Calls actual fold/loader/bank assembly functions (partition_fold from memory_study_v2.folds)
to determine true admissible sample counts across all 103 canonical market securities
for walk-forward evaluation years 2020..2025.

Reports the 6 per-fold population counts:
1. train_query_samples (and train_samples): feature warmup + 63-session label maturity satisfied.
2. val_query_samples (and validation_samples): validation partition queries.
3. dev_query_samples (and development_samples): development partition queries.
4. eval_query_samples (and evaluation_queries): information available at prediction time.
5. scored_eval_query_samples: subset of eval queries with mature forward target.
6. bank_record_samples (and bank_samples): records satisfying 126-session maturity + bank cutoff.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from memory_study_v2.folds import FoldPartitions, get_fold_boundaries
from memory_study_v2.sample_index import (
    SampleIndexRecord,
    build_security_sample_index,
    partition_sample_index,
    save_fold_sample_ids,
)
from memory_study_v2.venue_calendar import get_venue_calendar


def generate_manifest(
    cache_dir: str = "data/cache/ohlcv",
    config_path: str = "rebuild_plan/config.proposed.json",
    output_path: str = "rebuild_plan/fold_dimensions_manifest.json",
    sample_ids_output_dir: Optional[str] = "rebuild_plan/sample_ids",
) -> Dict[str, Any]:
    parquet_files = sorted(glob.glob(f"{cache_dir}/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {cache_dir}")

    # Config hash verification (Finding 3 / C6)
    cfg_p = Path(config_path)
    config_sha256 = hashlib.sha256(cfg_p.read_bytes()).hexdigest() if cfg_p.exists() else "CONFIG_NOT_FOUND"

    vc = get_venue_calendar()
    file_manifest: List[Dict[str, Any]] = []
    annual_totals: Dict[int, int] = {y: 0 for y in range(2013, 2026)}
    all_records: List[SampleIndexRecord] = []

    for fpath in parquet_files:
        p = Path(fpath)
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        sec_id = p.stem

        df = pd.read_parquet(p)
        recs = build_security_sample_index(sec_id, df, venue_calendar=vc)
        all_records.extend(recs)

        file_annual: Dict[int, int] = {}
        for r in recs:
            if r.input_window_valid:
                yr = int(r.session[:4])
                if 2013 <= yr <= 2025:
                    annual_totals[yr] += 1
                    file_annual[yr] = file_annual.get(yr, 0) + 1

        file_manifest.append({
            "filename": p.name,
            "relative_path": str(p).replace("\\", "/"),
            "security_id": sec_id,
            "sha256": sha,
            "file_size_bytes": len(data),
            "total_rows": len(df),
            "annual_admissible_rows": {str(k): v for k, v in sorted(file_annual.items())},
        })

    fold_dimensions: List[Dict[str, Any]] = []
    effective_batch = 512
    epochs_cap = 50

    sample_ids_dir = Path(sample_ids_output_dir) if sample_ids_output_dir else None
    if sample_ids_dir:
        sample_ids_dir.mkdir(parents=True, exist_ok=True)

    for y in range(2020, 2026):
        bound = get_fold_boundaries(y)
        part: FoldPartitions = partition_sample_index(all_records, bound)

        # Scored evaluation queries: subset of eval queries with mature forward target
        eval_records = [all_records[i] for i in part.evaluation_indices]
        scored_eval_count = sum(1 for r in eval_records if r.target_63_valid)

        train_s = len(part.train_indices)
        val_s = len(part.validation_indices)
        dev_s = len(part.development_indices)
        eval_q = len(part.evaluation_indices)
        bank_s = len(part.bank_indices)

        macro_per_ep = math.ceil(train_s / effective_batch)
        macro_per_fit = macro_per_ep * epochs_cap

        # Persist sample IDs per fold if directory provided
        if sample_ids_dir:
            save_fold_sample_ids(sample_ids_dir, y, part, all_records)

        fold_dimensions.append({
            "evaluation_year": y,
            "train_samples": train_s,
            "train_query_samples": train_s,
            "validation_samples": val_s,
            "val_query_samples": val_s,
            "development_samples": dev_s,
            "dev_query_samples": dev_s,
            "evaluation_queries": eval_q,
            "eval_query_samples": eval_q,
            "scored_eval_query_samples": scored_eval_count,
            "bank_samples": bank_s,
            "bank_record_samples": bank_s,
            "macro_steps_per_epoch": macro_per_ep,
            "macro_steps_per_fit": macro_per_fit,
            "boundaries": {
                "train_start": bound.train_start,
                "train_end": bound.train_end,
                "val_start": bound.val_start,
                "val_end": bound.val_end,
                "dev_start": bound.dev_start,
                "dev_end": bound.dev_end,
                "eval_start": bound.eval_start,
                "eval_end": bound.eval_end,
                "bank_cutoff": bound.bank_cutoff,
            },
            "sample_ids_sample": {
                "first_train_id": part.train_query_ids[0] if part.train_query_ids else None,
                "last_train_id": part.train_query_ids[-1] if part.train_query_ids else None,
                "first_eval_id": part.eval_query_ids[0] if part.eval_query_ids else None,
                "last_eval_id": part.eval_query_ids[-1] if part.eval_query_ids else None,
                "first_bank_id": part.bank_query_ids[0] if part.bank_query_ids else None,
                "last_bank_id": part.bank_query_ids[-1] if part.bank_query_ids else None,
            },
        })

    total_training_samples = sum(fd["train_samples"] for fd in fold_dimensions)
    avg_train_samples = round(total_training_samples / len(fold_dimensions))
    total_eval_queries = sum(fd["evaluation_queries"] for fd in fold_dimensions)
    total_macro_steps_all_fits = sum(fd["macro_steps_per_fit"] * 6 for fd in fold_dimensions)

    manifest_data = {
        "manifest_version": "2.1.0",
        "description": "Authoritative admissible sample manifest from partition_fold across 103 canonical market securities (C6, Finding 3).",
        "cache_directory": cache_dir.replace("\\", "/"),
        "config_sha256": config_sha256,
        "security_count": len(file_manifest),
        "annual_totals": {str(k): v for k, v in sorted(annual_totals.items())},
        "fold_dimensions": fold_dimensions,
        "summary": {
            "total_samples_timeline": sum(annual_totals.values()),
            "average_training_samples_per_fold": avg_train_samples,
            "total_evaluation_queries": total_eval_queries,
            "total_macro_steps_all_fits": total_macro_steps_all_fits,
            "fits_count": 36,
            "epochs_cap": epochs_cap,
            "effective_batch_size": effective_batch,
        },
        "file_manifest": file_manifest,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "rebuild_plan/fold_dimensions_manifest.json"
    manifest = generate_manifest(output_path=out_path)
    print(f"Generated fold dimensions manifest with {manifest['security_count']} securities at {out_path}.")
    print("Fold dimensions summary:")
    for fd in manifest["fold_dimensions"]:
        print(f"  Year {fd['evaluation_year']}: train={fd['train_samples']}, val={fd['validation_samples']}, dev={fd['development_samples']}, eval={fd['evaluation_queries']}, scored_eval={fd['scored_eval_query_samples']}, bank={fd['bank_samples']}")
