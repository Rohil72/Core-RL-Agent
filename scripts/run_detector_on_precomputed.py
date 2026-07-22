"""
Re-run cycle detection and oracle labeling on existing precomputed parquet files.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cycle.cycle_detector import detect_cycles
from src.cycle.oracle import annotate_cycle_targets
from src.data.features import ensure_sequence_model_features
from src.data.io_utils import read_dataframe, write_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/cycle_model.yaml"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def refresh_detector_outputs(
    config_path: str = DEFAULT_CONFIG_PATH,
    input_pattern: str | None = None,
    output_dir: str | None = None,
) -> dict:
    config = load_config(config_path)
    oracle_cfg = config["oracle"]
    data_pattern = input_pattern or config["data"]["precomputed_dir"]
    files = sorted(glob.glob(data_pattern))
    if not files:
        raise FileNotFoundError(f"No precomputed parquet files found for pattern: {data_pattern}")

    target_dir = Path(output_dir) if output_dir else None
    if target_dir is not None:
        target_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config_path": config_path,
        "input_pattern": data_pattern,
        "output_dir": str(target_dir) if target_dir is not None else "in_place",
        "files_processed": 0,
        "cycles_detected": 0,
        "per_ticker": [],
    }

    for path in tqdm(files, desc="Refreshing detector outputs"):
        frame = read_dataframe(path)
        if frame.empty:
            continue

        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        frame = ensure_sequence_model_features(frame)

        ticker = (
            str(frame["ticker"].iloc[0])
            if "ticker" in frame.columns and not frame.empty
            else Path(path).stem
        )
        frame["ticker"] = ticker

        # support optional learned scorer
        if hasattr(oracle_cfg, "get") and oracle_cfg.get("scorer_path"):
            from src.cycle.learned_detector import detect_cycles_learned

            cycles = detect_cycles_learned(
                frame["close"],
                feature_frame=frame,
                model_path=str(oracle_cfg.get("scorer_path")),
                device=oracle_cfg.get("scorer_device", "cpu"),
                min_duration_days=int(oracle_cfg["min_duration_days"]),
                max_duration_days=int(oracle_cfg["max_duration_days"]),
                min_return=float(oracle_cfg["min_return"]),
                soft_pullback_limit=float(oracle_cfg.get("soft_pullback_limit", 0.05)),
                hard_pullback_limit=float(oracle_cfg.get("hard_pullback_limit", 0.12)),
                volatility_window=int(oracle_cfg.get("volatility_window", 21)),
                volatility_multiplier=float(oracle_cfg.get("volatility_multiplier", 2.0)),
            )
        else:
            cycles = detect_cycles(
                frame["close"],
                min_duration_days=int(oracle_cfg["min_duration_days"]),
                max_duration_days=int(oracle_cfg["max_duration_days"]),
                min_return=float(oracle_cfg["min_return"]),
                feature_frame=frame,
                soft_pullback_limit=float(oracle_cfg.get("soft_pullback_limit", 0.05)),
                hard_pullback_limit=float(oracle_cfg.get("hard_pullback_limit", 0.12)),
                volatility_window=int(oracle_cfg.get("volatility_window", 21)),
                volatility_multiplier=float(oracle_cfg.get("volatility_multiplier", 2.0)),
                min_cycle_score=float(oracle_cfg.get("min_cycle_score", 0.58)),
                min_quality_score=float(oracle_cfg.get("min_quality_score", 0.20)),
            )
        annotated = annotate_cycle_targets(
            frame,
            cycles,
            catastrophic_return=float(oracle_cfg["catastrophic_return"]),
            positive_return_threshold=max(float(oracle_cfg["min_return"]), 0.10),
        )
        annotated["in_cycle"] = (annotated["oracle_cycle_id"] >= 0).astype("int8")
        annotated["ticker"] = ticker

        destination = (
            target_dir / Path(path).name if target_dir is not None else Path(path)
        )
        write_dataframe(annotated, destination)

        summary["files_processed"] += 1
        summary["cycles_detected"] += len(cycles)
        summary["per_ticker"].append(
            {
                "ticker": ticker,
                "rows": int(len(annotated)),
                "cycle_count": int(len(cycles)),
                "positive_rate": float(annotated["in_cycle"].mean()),
            }
        )

    if target_dir is not None:
        manifest_path = target_dir / "detector_refresh_manifest.json"
    else:
        manifest_path = Path("reports") / "experiments" / "detector_refresh_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Processed %s files and detected %s cycles.", summary["files_processed"], summary["cycles_detected"])
    logger.info("Wrote manifest to %s", manifest_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-pattern", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    refresh_detector_outputs(
        config_path=args.config_path,
        input_pattern=args.input_pattern,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
