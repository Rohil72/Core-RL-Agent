"""Generate hashed admissible-row manifest and walk-forward fold dimensions (C6).

Parses all canonical parquet files in data/cache/ohlcv/, computes SHA-256 digests,
counts admissible rows across the historical timeline (2013..2025), and produces
the authoritative fold dimensions manifest for walk-forward evaluation years 2020..2025.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def generate_manifest(
    cache_dir: str = "data/cache/ohlcv",
    output_path: str = "rebuild_plan/fold_dimensions_manifest.json",
) -> Dict[str, Any]:
    parquet_files = sorted(glob.glob(f"{cache_dir}/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {cache_dir}")

    file_manifest: List[Dict[str, Any]] = []
    annual_totals: Dict[int, int] = {y: 0 for y in range(2013, 2026)}

    for fpath in parquet_files:
        p = Path(fpath)
        data = p.read_bytes()
        sha = hashlib.sha256(data).hexdigest()

        df = pd.read_parquet(p)
        if "date" in df.columns:
            dcol = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        elif "timestamp" in df.columns:
            dcol = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
        elif isinstance(df.index, pd.DatetimeIndex):
            dcol = df.index
        else:
            dcol = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))

        file_annual: Dict[int, int] = {}
        for yr in dcol.year:
            if 2013 <= yr <= 2025:
                annual_totals[yr] += 1
                file_annual[yr] = file_annual.get(yr, 0) + 1

        file_manifest.append({
            "filename": p.name,
            "relative_path": str(p).replace("\\", "/"),
            "sha256": sha,
            "file_size_bytes": len(data),
            "total_rows": len(df),
            "annual_admissible_rows": {str(k): v for k, v in sorted(file_annual.items())},
        })

    fold_dimensions: List[Dict[str, Any]] = []
    effective_batch = 512
    epochs_cap = 50

    for y in range(2020, 2026):
        train_s = sum(annual_totals[yr] for yr in range(2013, y - 2))
        val_s = annual_totals[y - 2]
        dev_s = annual_totals[y - 1]
        eval_q = annual_totals[y]
        macro_per_ep = math.ceil(train_s / effective_batch)
        macro_per_fit = macro_per_ep * epochs_cap

        fold_dimensions.append({
            "evaluation_year": y,
            "train_samples": train_s,
            "validation_samples": val_s,
            "development_samples": dev_s,
            "evaluation_queries": eval_q,
            "bank_samples": train_s,
            "macro_steps_per_epoch": macro_per_ep,
            "macro_steps_per_fit": macro_per_fit,
            "boundaries": {
                "train_start": "2013-01-01",
                "train_end": f"{y - 3}-12-31",
                "val_start": f"{y - 2}-01-01",
                "val_end": f"{y - 2}-12-31",
                "dev_start": f"{y - 1}-01-01",
                "dev_end": f"{y - 1}-12-31",
                "eval_start": f"{y}-01-01",
                "eval_end": f"{y}-12-31",
            },
        })

    total_training_samples = sum(fd["train_samples"] for fd in fold_dimensions)
    avg_train_samples = round(total_training_samples / len(fold_dimensions))
    total_eval_queries = sum(fd["evaluation_queries"] for fd in fold_dimensions)
    total_macro_steps_all_fits = sum(fd["macro_steps_per_fit"] * 6 for fd in fold_dimensions)

    manifest_data = {
        "manifest_version": "2.0.0",
        "description": "Authoritative hashed admissible-row manifest across 103 canonical market securities (C6).",
        "cache_directory": cache_dir.replace("\\", "/"),
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
        print(f"  Year {fd['evaluation_year']}: train={fd['train_samples']}, val={fd['validation_samples']}, dev={fd['development_samples']}, eval={fd['evaluation_queries']}")
