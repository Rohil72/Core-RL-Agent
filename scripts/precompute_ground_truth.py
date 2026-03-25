"""
Precompute price, trend, fundamental, and oracle action targets for cycle modeling.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cycle.cycle_detector import detect_cycles
from src.cycle.oracle import annotate_cycle_targets
from src.data.features import (
    compute_fundamental_features_aligned,
    compute_technical_features,
    ensure_sequence_model_features,
)
from src.data.io_utils import write_dataframe
from src.data.loader import load_market_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "data/precomputed"
CONFIG_PATH = "config/market_universe.yaml"
START = "2018-01-01"
END = "2024-12-31"


def precompute(
    config_path: str = CONFIG_PATH,
    output_dir: str = OUTPUT_DIR,
    start: str = START,
    end: str = END,
    ticker_limit: int | None = None,
    min_duration_days: int = 21,
    max_duration_days: int = 252,
    min_return: float = 0.30,
    catastrophic_return: float = -0.10,
    soft_pullback_limit: float = 0.05,
    hard_pullback_limit: float = 0.12,
    volatility_window: int = 21,
    volatility_multiplier: float = 2.0,
    min_cycle_score: float = 0.58,
    min_quality_score: float = 0.20,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    data = load_market_data(config_path, start, end)
    ticker_items = list(data.items())
    if ticker_limit is not None:
        ticker_items = ticker_items[:ticker_limit]

    run_manifest = {
        "config_path": config_path,
        "date_range": {"start": start, "end": end},
        "ticker_limit": ticker_limit,
        "tickers_requested": len(data),
        "tickers_selected": len(ticker_items),
        "tickers_processed": 0,
        "rows_written": 0,
        "per_ticker": [],
    }

    for ticker, dfs in tqdm(ticker_items, desc="Processing tickers"):
        price_df = dfs["price"]
        earnings_df = dfs["earnings"]
        if price_df.empty:
            continue

        price_df = compute_technical_features(price_df)
        price_df = compute_fundamental_features_aligned(price_df, earnings_df)
        price_df = ensure_sequence_model_features(price_df)

        cycles = detect_cycles(
            price_df["close"],
            min_duration_days=min_duration_days,
            max_duration_days=max_duration_days,
            min_return=min_return,
            feature_frame=price_df,
            soft_pullback_limit=soft_pullback_limit,
            hard_pullback_limit=hard_pullback_limit,
            volatility_window=volatility_window,
            volatility_multiplier=volatility_multiplier,
            min_cycle_score=min_cycle_score,
            min_quality_score=min_quality_score,
        )
        final_df = annotate_cycle_targets(
            price_df,
            cycles,
            catastrophic_return=catastrophic_return,
            positive_return_threshold=min_return,
        )
        final_df["in_cycle"] = (final_df["oracle_cycle_id"] >= 0).astype("int8")
        final_df["ticker"] = ticker

        write_dataframe(final_df, os.path.join(output_dir, f"{ticker}.parquet"))
        run_manifest["tickers_processed"] += 1
        run_manifest["rows_written"] += len(final_df)

        action_dist = (
            final_df["oracle_action"]
            .value_counts(normalize=True)
            .reindex([0, 1, 2, 3], fill_value=0.0)
            .to_dict()
        )
        run_manifest["per_ticker"].append(
            {
                "ticker": ticker,
                "rows": int(len(final_df)),
                "cycle_count": int(len(cycles)),
                "positive_rate": float(final_df["in_cycle"].mean()),
                "candidate_gate_rate": float(final_df["tech_minervini_gate"].mean())
                if "tech_minervini_gate" in final_df.columns
                else 0.0,
                "report_coverage": float(final_df["fund_report_available"].mean())
                if "fund_report_available" in final_df.columns
                else 0.0,
                "hard_negative_rate": float(final_df["oracle_hard_negative"].mean())
                if "oracle_hard_negative" in final_df.columns
                else 0.0,
                "action_distribution": {
                    str(k): float(v) for k, v in action_dist.items()
                },
            }
        )

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    logger.info("Precomputation complete. Saved to %s", output_dir)
    logger.info("Wrote run manifest to %s", manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=CONFIG_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--ticker-limit", type=int, default=None)
    parser.add_argument("--min-duration-days", type=int, default=21)
    parser.add_argument("--max-duration-days", type=int, default=252)
    parser.add_argument("--min-return", type=float, default=0.30)
    parser.add_argument("--catastrophic-return", type=float, default=-0.10)
    parser.add_argument("--soft-pullback-limit", type=float, default=0.05)
    parser.add_argument("--hard-pullback-limit", type=float, default=0.12)
    parser.add_argument("--volatility-window", type=int, default=21)
    parser.add_argument("--volatility-multiplier", type=float, default=2.0)
    parser.add_argument("--min-cycle-score", type=float, default=0.58)
    parser.add_argument("--min-quality-score", type=float, default=0.20)
    args = parser.parse_args()

    precompute(
        config_path=args.config_path,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        ticker_limit=args.ticker_limit,
        min_duration_days=args.min_duration_days,
        max_duration_days=args.max_duration_days,
        min_return=args.min_return,
        catastrophic_return=args.catastrophic_return,
        soft_pullback_limit=args.soft_pullback_limit,
        hard_pullback_limit=args.hard_pullback_limit,
        volatility_window=args.volatility_window,
        volatility_multiplier=args.volatility_multiplier,
        min_cycle_score=args.min_cycle_score,
        min_quality_score=args.min_quality_score,
    )
