"""
Render detector-only plots from precomputed parquet data.
"""

from __future__ import annotations

import argparse
import glob
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
from src.data.features import ensure_sequence_model_features
from src.data.io_utils import read_dataframe
from src.visualization.price_cycle_plot import plot_price_and_cycles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/cycle_model.yaml"
DEFAULT_OUTPUT_DIR = "data/detector_plots"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_detector_outputs(
    config_path: str = DEFAULT_CONFIG_PATH,
    input_pattern: str | None = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    plot_limit: int | None = None,
) -> list[str]:
    config = load_config(config_path)
    oracle_cfg = config["oracle"]
    data_pattern = input_pattern or config["data"]["precomputed_dir"]
    files = sorted(glob.glob(data_pattern))
    if not files:
        raise FileNotFoundError(f"No precomputed parquet files found for pattern: {data_pattern}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    selected_files = files[:plot_limit] if plot_limit is not None else files
    for path in tqdm(selected_files, desc="Rendering detector plots"):
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

        save_path = out_dir / f"{ticker}_detector.png"
        plot_price_and_cycles(
            dates=frame.index,
            close_prices=frame["close"].to_numpy(),
            cycles=cycles,
            title=f"{ticker} | Detector Cycles",
            save_path=str(save_path),
            show=False,
        )
        written.append(str(save_path))

    logger.info("Wrote %s detector plots to %s", len(written), out_dir)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--input-pattern", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plot-limit", type=int, default=None)
    args = parser.parse_args()

    plot_detector_outputs(
        config_path=args.config_path,
        input_pattern=args.input_pattern,
        output_dir=args.output_dir,
        plot_limit=args.plot_limit,
    )


if __name__ == "__main__":
    main()
