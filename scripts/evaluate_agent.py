"""
Evaluate the cycle reasoning model and render price-vs-cycle comparison plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

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
METRICS_DIR = "reports/evaluations"


def shade_cycles(ax, spans, color: str, alpha: float) -> None:
    for span in spans:
        ax.axvspan(span.start_date, span.end_date, color=color, alpha=alpha)


def plot_comparison(ticker, frame, oracle_cycles, predicted_cycles) -> None:
    import re
    from pathlib import Path

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    prices = frame["close"]

    ax1.plot(prices.index, prices.values, color="black", alpha=0.7)
    shade_cycles(ax1, oracle_cycles, color="green", alpha=0.30)
    ax1.set_title(f"{ticker} - Oracle Cycles")

    ax2.plot(prices.index, prices.values, color="black", alpha=0.7)
    shade_cycles(ax2, predicted_cycles, color="blue", alpha=0.30)
    ax2.set_title(f"{ticker} - Model Cycles")

    plt.tight_layout()
    # ensure output directory exists
    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # sanitize ticker to produce a safe filename on Windows
    safe_ticker = re.sub(r'[^A-Za-z0-9_.-]', '_', str(ticker)).strip(' .')
    # avoid reserved device names like CON, PRN, AUX, NUL, COM1..COM9, LPT1..LPT9
    reserved = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1,10)} | {f"LPT{i}" for i in range(1,10)}
    if safe_ticker.upper() in reserved or len(safe_ticker) == 0:
        safe_ticker = f"ticker_{abs(hash(str(ticker))) & 0xffffffff:08x}"

    out_path = out_dir_path / f"{safe_ticker}_comparison.png"
    abs_path = out_path.resolve(strict=False)
    try:
        logger.info("Saving plot to %s", abs_path)
        logger.debug("Path repr: %r", str(abs_path))
        plt.savefig(str(abs_path))
    except Exception as e:
        logger.exception("Failed to save plot to %s: %s", abs_path, e)
        # try fallback filename using hash
        alt_name = f"{safe_ticker}_{abs(str(hash(str(ticker)))) & 0xffffffff:08x}_comparison.png"
        alt_path = out_dir_path / alt_name
        try:
            logger.info("Retrying save to fallback path %s", alt_path)
            plt.savefig(str(alt_path.resolve(strict=False)))
        except Exception:
            # In extreme case, write the figure to a bytes buffer and write to temp dir
            import io
            import tempfile
            buf = io.BytesIO()
            try:
                fig.savefig(buf, format="png")
                buf.seek(0)
                tmp = Path(tempfile.gettempdir()) / f"figure_{abs(hash(str(ticker))) & 0xffffffff:08x}.png"
                with open(tmp, "wb") as f:
                    f.write(buf.read())
                logger.info("Wrote fallback figure bytes to %s", tmp)
            except Exception:
                logger.exception("Failed writing fallback image to bytes")
            finally:
                buf.close()
            # re-raise original exception to surface error
            raise
    finally:
        plt.close()


def _metric_line(metrics: dict, key: str, label: str) -> str:
    return f"- {label}: {float(metrics.get(key, 0.0)):.4f}"


def write_evaluation_metrics(
    summary: dict,
    metrics_dir: str,
    config_path: str,
    checkpoint_path: str,
) -> tuple[str, str]:
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("evaluation_%Y%m%d_%H%M%S")
    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "splits": summary,
    }
    json_path = out_dir / f"{run_id}.json"
    md_path = out_dir / f"{run_id}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Evaluation Metrics",
        "",
        f"- Run ID: {run_id}",
        f"- Config: `{config_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        "",
    ]
    for split_name, metrics in summary.items():
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "### Action Classification",
                "",
                _metric_line(metrics, "action_accuracy", "Accuracy"),
                _metric_line(metrics, "action_balanced_accuracy", "Balanced accuracy"),
                _metric_line(metrics, "action_macro_f1", "Macro F1"),
                _metric_line(metrics, "action_weighted_f1", "Weighted F1"),
                "",
                "### Future/Event Regression",
                "",
                _metric_line(metrics, "future_target_mae", "MAE"),
                _metric_line(metrics, "future_target_rmse", "RMSE"),
                _metric_line(metrics, "future_target_r2", "Mean R2"),
                _metric_line(metrics, "future_target_pearson", "Mean Pearson"),
                "",
                "### Cycle Span Quality",
                "",
                _metric_line(metrics, "cycle_precision", "Precision"),
                _metric_line(metrics, "cycle_recall", "Recall"),
                _metric_line(metrics, "cycle_f1", "F1"),
                _metric_line(metrics, "profitable_cycle_rate", "Profitable cycle rate"),
                _metric_line(metrics, "average_cycle_return", "Average cycle return"),
                _metric_line(metrics, "median_cycle_return", "Median cycle return"),
                _metric_line(metrics, "catastrophic_cycle_rate", "Catastrophic cycle rate"),
                _metric_line(metrics, "bounded_exit_error", "Bounded exit error"),
                "",
            ]
        )
        per_target = metrics.get("future_target_metrics", {})
        if per_target:
            lines.extend(["### Per-Target Future/Event Metrics", ""])
            lines.append("| Target | MAE | RMSE | R2 | Pearson | Valid Count |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for target, target_metrics in per_target.items():
                lines.append(
                    "| "
                    f"{target} | "
                    f"{target_metrics.get('mae', 0.0):.4f} | "
                    f"{target_metrics.get('rmse', 0.0):.4f} | "
                    f"{target_metrics.get('r2', 0.0):.4f} | "
                    f"{target_metrics.get('pearson', 0.0):.4f} | "
                    f"{target_metrics.get('valid_count', 0.0):.0f} |"
                )
            lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(json_path), str(md_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs/cycle_model.yaml")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument(
        "--split", choices=["val", "test", "holdout", "all"], default="test"
    )
    parser.add_argument("--plot-limit", type=int, default=28)
    parser.add_argument("--metrics-dir", default=METRICS_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model, payload = load_checkpoint(args.checkpoint)
    device = next(model.parameters()).device
    config = load_config(args.config_path)
    config.setdefault("features", {})
    if "feature_cols" in payload:
        config["features"]["sequence"] = list(payload["feature_cols"])
    if "future_target_cols" in payload:
        config["features"]["future_targets"] = list(payload["future_target_cols"])
    frame = load_precomputed_frame(config)
    raw_frames, datasets, _, _ = make_datasets(frame, config)

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
            "%s action_f1=%.4f future_mae=%.4f cycle_f1=%.4f profitable=%.4f catastrophic=%.4f",
            split_name,
            metrics["action_macro_f1"],
            metrics["future_target_mae"],
            metrics["cycle_f1"],
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
            )
            plot_comparison(ticker, eval_frame, oracle_cycles, predicted_cycles)
            plotted += 1

    metric_paths = write_evaluation_metrics(
        summary=summary,
        metrics_dir=args.metrics_dir,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
    )
    logger.info("Saved evaluation metrics to %s and %s", *metric_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
