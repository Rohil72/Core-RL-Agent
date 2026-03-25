"""
Evaluate the cycle reasoning model and render price-vs-cycle comparison plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cycle.oracle import decode_action_spans, extract_oracle_spans
from src.trainers.train_cycle_model import (
    collect_predictions,
    evaluate_split,
    load_checkpoint,
    load_config,
    load_precomputed_frame,
    make_datasets,
    make_loader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "models/cycle_reasoner/final_model.pt"
OUTPUT_DIR = "data/evaluation_plots"


def shade_cycles(ax, spans, color: str, alpha: float) -> None:
    for span in spans:
        ax.axvspan(span.start_date, span.end_date, color=color, alpha=alpha)


def plot_comparison(ticker, frame, oracle_cycles, predicted_cycles) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    prices = frame["close"]

    ax1.plot(prices.index, prices.values, color="black", alpha=0.7)
    shade_cycles(ax1, oracle_cycles, color="green", alpha=0.30)
    ax1.set_title(f"{ticker} - Oracle Cycles")

    ax2.plot(prices.index, prices.values, color="black", alpha=0.7)
    shade_cycles(ax2, predicted_cycles, color="blue", alpha=0.30)
    ax2.set_title(f"{ticker} - Model Cycles")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ticker}_comparison.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs/cycle_model.yaml")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument(
        "--split", choices=["val", "test", "holdout", "all"], default="test"
    )
    parser.add_argument("--plot-limit", type=int, default=28)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    config = load_config(args.config_path)
    frame = load_precomputed_frame(config)
    raw_frames, datasets, _, _ = make_datasets(frame, config)

    model, payload = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device

    if args.split == "all":
        split_names = [name for name in ["val", "test", "holdout"] if name in datasets]
    else:
        if args.split not in datasets:
            raise ValueError(f"Requested split '{args.split}' is not available.")
        split_names = [args.split]
    summary = {}

    for split_name in split_names:
        metrics = evaluate_split(
            model=model,
            dataset=datasets[split_name],
            raw_frame=raw_frames[split_name],
            device=device,
            config=config,
        )
        summary[split_name] = metrics
        logger.info(
            "%s precision=%.4f recall=%.4f profitable=%.4f catastrophic=%.4f",
            split_name,
            metrics["cycle_precision"],
            metrics["cycle_recall"],
            metrics["profitable_cycle_rate"],
            metrics["catastrophic_cycle_rate"],
        )

        loader = make_loader(
            datasets[split_name],
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
        )
        pred_df, _ = collect_predictions(model, loader, device)
        plotted = 0
        for ticker, group in raw_frames[split_name].groupby("ticker"):
            if plotted >= args.plot_limit:
                break
            eval_frame = group.sort_index().copy()
            ticker_pred = (
                pred_df[pred_df["ticker"] == ticker].set_index("timestamp").sort_index()
            )
            eval_frame = eval_frame.join(ticker_pred[["pred_action"]], how="inner")
            if eval_frame.empty:
                continue
            oracle_cycles = extract_oracle_spans(eval_frame)
            predicted_cycles = decode_action_spans(
                eval_frame,
                eval_frame["pred_action"].astype(int).tolist(),
                cooldown_days=int(config["evaluation"]["cooldown_days"]),
            )
            plot_comparison(ticker, eval_frame, oracle_cycles, predicted_cycles)
            plotted += 1

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
