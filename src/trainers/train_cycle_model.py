from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cycle.cycle_detector import detect_cycles
from src.cycle.oracle import (
    annotate_cycle_targets,
    decode_action_spans,
    ensure_event_outcome_targets,
    ensure_oracle_cycle_metadata,
    extract_oracle_spans,
)
from src.data.features import ensure_sequence_model_features
from src.data.io_utils import read_dataframe
from src.data.sequence_dataset import (
    CycleSequenceDataset,
    FeatureStandardizer,
    build_buffered_period_frame,
    build_walk_forward_splits,
    filter_frame_by_period,
    select_holdout_tickers,
)
from src.eval.cycle_prediction_metrics import (
    compute_action_metrics,
    compute_cycle_metrics,
    compute_future_target_metrics,
    compute_cycle_moving_average_return,
)
from src.models import build_cycle_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "configs/cycle_model.yaml"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def _infer_feature_columns(df: pd.DataFrame, configured: list[str] | None) -> list[str]:
    if configured:
        available = [col for col in configured if col in df.columns]
        missing = sorted(set(configured) - set(available))
        if missing:
            logger.warning(
                "Skipping unavailable feature columns: %s", ", ".join(missing)
            )
        return available

    preferred = [
        "tech_return_1",
        "tech_intraday_range",
        "tech_momentum_3",
        "tech_momentum_10",
        "tech_momentum_21",
        "tech_vol_21",
        "tech_volume_ratio",
        "tech_volume_change_1",
        "tech_drawdown",
        "tech_trend_slope",
        "tech_close_vs_sma_50",
        "tech_close_vs_sma_150",
        "tech_close_vs_sma_200",
        "tech_sma_200_trend_20",
        "tech_pct_above_52w_low",
        "tech_pct_from_52w_high",
        "tech_up_down_volume_ratio_50",
        "tech_minervini_template_score",
        "tech_minervini_gate",
        "fund_eps_surprise",
        "fund_eps_growth_yoy",
        "fund_eps_rolling2_yoy",
        "fund_eps_accel",
        "fund_revenue_growth_yoy",
        "fund_revenue_rolling2_yoy",
        "fund_minervini_score",
        "fund_report_available",
        "fund_days_since_report",
    ]
    return [col for col in preferred if col in df.columns]


def _infer_future_target_columns(
    df: pd.DataFrame, configured: list[str] | None
) -> list[str]:
    if configured:
        available = [col for col in configured if col in df.columns]
        missing = sorted(set(configured) - set(available))
        if missing:
            logger.warning(
                "Skipping unavailable future target columns: %s", ", ".join(missing)
            )
        return available

    preferred = [
        "future_return_21",
        "future_return_63",
        "future_return_126",
        "future_return_252",
        "future_max_return_63",
        "future_min_return_63",
        "event_peak_offset_63",
        "event_drawdown_offset_63",
        "event_upside_before_drawdown_126",
        "event_upside_hit_126",
        "event_drawdown_hit_126",
    ]
    return [col for col in preferred if col in df.columns]


def _has_precomputed_oracle_targets(df: pd.DataFrame) -> bool:
    required = {"oracle_action", "oracle_cycle_id"}
    return required.issubset(df.columns)


def load_precomputed_frame(config: dict[str, Any]) -> pd.DataFrame:
    pattern = config["data"]["precomputed_dir"]
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No precomputed parquet files found for pattern: {pattern}"
        )

    prepared = []
    oracle_cfg = config["oracle"]

    for path in files:
        df = read_dataframe(path)
        if df.empty:
            continue
        df = _ensure_datetime_index(df)
        ticker = (
            str(df["ticker"].iloc[0]) if "ticker" in df.columns else Path(path).stem
        )
        df["ticker"] = ticker
        df = ensure_sequence_model_features(df)
        if _has_precomputed_oracle_targets(df):
            df = ensure_oracle_cycle_metadata(df)
        else:
            cycles = detect_cycles(
                df["close"],
                min_duration_days=int(oracle_cfg["min_duration_days"]),
                max_duration_days=int(oracle_cfg["max_duration_days"]),
                min_return=float(oracle_cfg["min_return"]),
                feature_frame=df,
                soft_pullback_limit=float(oracle_cfg.get("soft_pullback_limit", 0.05)),
                hard_pullback_limit=float(oracle_cfg.get("hard_pullback_limit", 0.12)),
                volatility_window=int(oracle_cfg.get("volatility_window", 21)),
                volatility_multiplier=float(
                    oracle_cfg.get("volatility_multiplier", 2.0)
                ),
                min_cycle_score=float(oracle_cfg.get("min_cycle_score", 0.58)),
                min_quality_score=float(oracle_cfg.get("min_quality_score", 0.20)),
            )
            df = annotate_cycle_targets(
                df,
                cycles,
                catastrophic_return=float(oracle_cfg["catastrophic_return"]),
                positive_return_threshold=float(oracle_cfg["min_return"]),
            )
        df = ensure_event_outcome_targets(
            df,
            drawdown_threshold=float(oracle_cfg["catastrophic_return"]),
            upside_threshold=float(oracle_cfg["min_return"]),
        )
        df["in_cycle"] = (df["oracle_cycle_id"] >= 0).astype(np.int8)
        prepared.append(df)

    if not prepared:
        raise ValueError("No usable data loaded from precomputed parquet files.")

    frame = pd.concat(prepared).sort_index()
    feature_cols = _infer_feature_columns(
        frame, config.get("features", {}).get("sequence")
    )
    future_target_cols = _infer_future_target_columns(
        frame, config.get("features", {}).get("future_targets")
    )

    required = feature_cols + ["oracle_action", "ticker", "close"]
    clean = frame.dropna(subset=required).copy()
    clean.attrs["feature_cols"] = feature_cols
    clean.attrs["future_target_cols"] = future_target_cols
    return clean


def make_datasets(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, CycleSequenceDataset],
    FeatureStandardizer,
    dict[str, Any],
]:
    feature_cols = frame.attrs["feature_cols"]
    future_target_cols = frame.attrs["future_target_cols"]
    split_cfg = config["split"]

    splits = build_walk_forward_splits(
        frame.index,
        train_years=int(split_cfg["train_years"]),
        val_years=int(split_cfg["val_years"]),
        test_years=int(split_cfg["test_years"]),
        step_years=int(split_cfg["step_years"]),
    )
    if not splits:
        raise ValueError("No walk-forward splits could be built from the dataset.")

    selected_fold = int(split_cfg.get("selected_fold", -1))
    split = splits[selected_fold]

    holdout_tickers = select_holdout_tickers(
        tickers=frame["ticker"].unique().tolist(),
        fraction=float(split_cfg.get("ticker_holdout_fraction", 0.0)),
        seed=int(split_cfg.get("seed", 7)),
    )

    periods = {
        "train": filter_frame_by_period(
            frame, split["train_start"], split["train_end"]
        ),
        "val": filter_frame_by_period(frame, split["val_start"], split["val_end"]),
        "test": filter_frame_by_period(frame, split["test_start"], split["test_end"]),
    }
    history_rows = int(config["model"]["window_size"]) - 1
    buffered = {
        "train": build_buffered_period_frame(
            frame,
            split["train_start"],
            split["train_end"],
            history_rows=history_rows,
        ),
        "val": build_buffered_period_frame(
            frame,
            split["val_start"],
            split["val_end"],
            history_rows=history_rows,
        ),
        "test": build_buffered_period_frame(
            frame,
            split["test_start"],
            split["test_end"],
            history_rows=history_rows,
        ),
    }
    buffered["holdout"] = buffered["test"][
        buffered["test"]["ticker"].isin(holdout_tickers)
    ].copy()

    train_frame = buffered["train"][
        ~buffered["train"]["ticker"].isin(holdout_tickers)
    ].copy()
    val_frame = buffered["val"][~buffered["val"]["ticker"].isin(holdout_tickers)].copy()
    test_frame = buffered["test"][
        ~buffered["test"]["ticker"].isin(holdout_tickers)
    ].copy()

    if train_frame.empty or val_frame.empty or test_frame.empty:
        raise ValueError(
            "One of the train/val/test splits is empty after applying holdouts."
        )

    standardizer = FeatureStandardizer.from_frame(train_frame, feature_cols)
    scaled_frames = {
        "train": standardizer.transform(train_frame, feature_cols),
        "val": standardizer.transform(val_frame, feature_cols),
        "test": standardizer.transform(test_frame, feature_cols),
    }
    if not buffered["holdout"].empty:
        scaled_frames["holdout"] = standardizer.transform(
            buffered["holdout"], feature_cols
        )

    window_size = int(config["model"]["window_size"])
    datasets = {
        name: CycleSequenceDataset(
            frame=scaled_frames[name],
            feature_cols=feature_cols,
            future_target_cols=future_target_cols,
            window_size=window_size,
            target_start=(
                split["train_start"]
                if name == "train"
                else split["val_start"]
                if name == "val"
                else split["test_start"]
            ),
            target_end=(
                split["train_end"]
                if name == "train"
                else split["val_end"]
                if name == "val"
                else split["test_end"]
            ),
        )
        for name in scaled_frames
    }

    raw_frames = {
        "train": train_frame,
        "val": val_frame,
        "test": test_frame,
    }
    if not buffered["holdout"].empty:
        raw_frames["holdout"] = buffered["holdout"]

    split_meta = {
        "selected_fold": selected_fold,
        "fold": {key: value.isoformat() for key, value in split.items()},
        "holdout_tickers": holdout_tickers,
        "feature_cols": feature_cols,
        "future_target_cols": future_target_cols,
    }
    return raw_frames, datasets, standardizer, split_meta


def build_model(
    config: dict[str, Any], input_dim: int, future_target_dim: int
) -> nn.Module:
    model_cfg = config["model"]
    return build_cycle_model(
        model_cfg=model_cfg,
        input_dim=input_dim,
        future_target_dim=future_target_dim,
        action_dim=4,
    )


def compute_class_weights(actions: pd.Series) -> torch.Tensor:
    counts = actions.value_counts().reindex([0, 1, 2, 3], fill_value=0).astype(float)
    weights = counts.sum() / counts.replace(0, np.nan)
    weights = weights.fillna(weights.max())
    weights = weights / weights.mean()
    return torch.tensor(weights.values, dtype=torch.float32)


def make_loader(
    dataset: CycleSequenceDataset, batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _forward_model(
    model: nn.Module,
    sequence: torch.Tensor,
    return_reconstruction: bool = False,
) -> dict[str, torch.Tensor]:
    if return_reconstruction:
        if not getattr(model, "supports_reconstruction", False):
            raise ValueError(
                "Self-supervised reconstruction requires a model with "
                "supports_reconstruction=True."
            )
        return model(sequence, return_reconstruction=True)
    return model(sequence)


def _apply_self_supervised_mask(
    sequence: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    ssl_cfg = config.get("self_supervised", {})
    mask_probability = float(ssl_cfg.get("mask_probability", 0.0))
    span_probability = float(ssl_cfg.get("mask_span_probability", 0.0))
    span_length = max(int(ssl_cfg.get("mask_span_length", 0)), 0)
    mask_value = float(ssl_cfg.get("mask_value", 0.0))

    mask = torch.zeros_like(sequence, dtype=torch.bool)
    if mask_probability > 0:
        mask |= torch.rand_like(sequence) < mask_probability

    if span_probability > 0 and span_length > 0:
        batch_size, _, time_steps = sequence.shape
        starts = torch.rand(
            batch_size,
            time_steps,
            device=sequence.device,
        ) < span_probability
        time_mask = torch.zeros(
            batch_size,
            time_steps,
            dtype=torch.bool,
            device=sequence.device,
        )
        for offset in range(span_length):
            if offset >= time_steps:
                break
            time_mask[:, offset:] |= starts[:, : time_steps - offset]
        mask |= time_mask.unsqueeze(1)

    return sequence.masked_fill(mask, mask_value), mask


def _compute_future_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    valid_mask = ~torch.isnan(target)
    if not valid_mask.any():
        return prediction.new_tensor(0.0)
    return F.smooth_l1_loss(
        prediction[valid_mask], target[valid_mask], reduction="mean"
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    action_loss_weight = float(config["training"].get("action_loss_weight", 1.0))
    future_loss_weight = float(config["training"]["future_loss_weight"])
    hard_negative_weight = float(config["training"]["hard_negative_weight"])
    grad_clip = float(config["training"]["grad_clip"])
    reconstruction_loss_weight = float(
        config.get("self_supervised", {}).get("reconstruction_loss_weight", 0.0)
    )

    for batch in loader:
        sequence = batch["sequence"].to(device)
        action = batch["action"].to(device)
        future_target = batch["future_target"].to(device)
        hard_negative = batch["hard_negative"].to(device)

        model_input = sequence
        reconstruction_mask = None
        if reconstruction_loss_weight > 0:
            model_input, reconstruction_mask = _apply_self_supervised_mask(
                sequence,
                config,
            )

        outputs = _forward_model(
            model,
            model_input,
            return_reconstruction=reconstruction_loss_weight > 0,
        )
        ce = F.cross_entropy(
            outputs["action_logits"],
            action,
            weight=class_weights,
            reduction="none",
        )
        row_weights = 1.0 + hard_negative * hard_negative_weight
        action_loss = (ce * row_weights).mean()
        future_loss = _compute_future_loss(outputs["future_pred"], future_target)
        loss = action_loss_weight * action_loss + future_loss_weight * future_loss
        if reconstruction_loss_weight > 0 and reconstruction_mask is not None:
            if reconstruction_mask.any():
                reconstruction_loss = F.smooth_l1_loss(
                    outputs["reconstruction"][reconstruction_mask],
                    sequence[reconstruction_mask],
                    reduction="mean",
                )
                loss = loss + reconstruction_loss_weight * reconstruction_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(sequence.size(0))
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    records = []
    future_true_batches = []
    future_pred_batches = []

    for batch in loader:
        sequence = batch["sequence"].to(device)
        action = batch["action"].cpu().numpy()
        future_target = batch["future_target"].to(device)

        outputs = _forward_model(model, sequence)
        logits = outputs["action_logits"]
        pred_action = logits.argmax(dim=1).cpu().numpy()

        future_true_batches.append(future_target.cpu().numpy())
        future_pred_batches.append(outputs["future_pred"].cpu().numpy())

        for ticker, timestamp, true_action, predicted_action in zip(
            batch["ticker"],
            batch["timestamp"],
            action,
            pred_action,
        ):
            records.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "true_action": int(true_action),
                    "pred_action": int(predicted_action),
                }
            )

    pred_df = pd.DataFrame(records)
    target_names = list(getattr(loader.dataset, "future_target_cols", []))
    if pred_df.empty:
        return pred_df, {
            "action_accuracy": 0.0,
            "action_balanced_accuracy": 0.0,
            "action_macro_precision": 0.0,
            "action_macro_recall": 0.0,
            "action_macro_f1": 0.0,
            "action_weighted_precision": 0.0,
            "action_weighted_recall": 0.0,
            "action_weighted_f1": 0.0,
            "future_target_mae": 0.0,
            "future_target_rmse": 0.0,
            "future_target_r2": 0.0,
            "future_target_pearson": 0.0,
            "future_target_valid_count": 0.0,
            "future_target_metrics": {},
        }
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], utc=True)
    pred_df = pred_df.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    action_metrics = compute_action_metrics(
        pred_df["true_action"].to_numpy(),
        pred_df["pred_action"].to_numpy(),
    )
    future_metrics = compute_future_target_metrics(
        np.concatenate(future_true_batches, axis=0),
        np.concatenate(future_pred_batches, axis=0),
        target_names=target_names,
    )
    return pred_df, {**action_metrics, **future_metrics}


def evaluate_split(
    model: nn.Module,
    dataset: CycleSequenceDataset,
    raw_frame: pd.DataFrame,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    if len(dataset) == 0 or raw_frame.empty:
        empty_cycle = compute_cycle_metrics([], [])
        return {
            "action_accuracy": 0.0,
            "action_macro_precision": 0.0,
            "action_macro_recall": 0.0,
            "action_macro_f1": 0.0,
            "future_target_mae": 0.0,
            **empty_cycle,
            "per_ticker": [],
        }

    loader = make_loader(
        dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False
    )
    pred_df, action_metrics = collect_predictions(model, loader, device)

    catastrophic_return = float(config["evaluation"]["catastrophic_return"])
    late_penalty_factor = float(config["evaluation"]["late_exit_penalty_factor"])

    per_ticker = []
    all_predicted = []
    all_oracle = []

    for ticker, group in raw_frame.groupby("ticker"):
        ticker_frame = group.sort_index().copy()
        eval_frame = ticker_frame.copy()
        ticker_pred = pred_df[pred_df["ticker"] == ticker].copy()
        if ticker_pred.empty:
            continue
        ticker_pred = ticker_pred.set_index("timestamp").sort_index()
        eval_frame = eval_frame.join(
            ticker_pred[["pred_action", "true_action"]], how="inner"
        )
        if eval_frame.empty:
            continue
        oracle_cycles = extract_oracle_spans(eval_frame)
        predicted_cycles = decode_action_spans(
            eval_frame,
            eval_frame["pred_action"].astype(int).tolist(),
        )
        cycle_metrics = compute_cycle_metrics(
            predicted_cycles,
            oracle_cycles,
            catastrophic_return=catastrophic_return,
            late_penalty_factor=late_penalty_factor,
        )
        cycle_metrics["ticker"] = ticker
        per_ticker.append(cycle_metrics)
        all_predicted.extend(predicted_cycles)
        all_oracle.extend(oracle_cycles)

    aggregate_cycle = compute_cycle_metrics(
        all_predicted,
        all_oracle,
        catastrophic_return=catastrophic_return,
        late_penalty_factor=late_penalty_factor,
    )

    # Independent profit metric: moving-average of oracle cycle returns
    moving_window = int(config.get("evaluation", {}).get("moving_avg_window", 50))
    moving_summary = compute_cycle_moving_average_return(all_oracle, window=moving_window)

    return {
        **action_metrics,
        **aggregate_cycle,
        "moving_avg_cycle_return": float(moving_summary.get("moving_avg_cycle_return", 0.0)),
        "moving_avg_window": int(moving_summary.get("moving_avg_window", moving_window)),
        "moving_avg_count": float(moving_summary.get("moving_avg_count", 0.0)),
        "per_ticker": per_ticker,
    }


def write_report(
    report: dict[str, Any], config: dict[str, Any], split_meta: dict[str, Any]
) -> tuple[str, str]:
    reports_dir = Path(config["evaluation"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("cycle_model_%Y%m%d_%H%M%S")

    payload = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_meta": split_meta,
        "report": report,
    }
    json_path = reports_dir / f"{run_id}.json"
    md_path = reports_dir / f"{run_id}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_lines = [
        "# Cycle Model Report",
        "",
        f"- Run ID: {run_id}",
        f"- Selected fold: {split_meta['selected_fold']}",
        f"- Holdout tickers: {', '.join(split_meta['holdout_tickers']) or 'none'}",
        "",
    ]
    for split_name, metrics in report.items():
        summary_lines.extend(
            [
                f"## {split_name.title()}",
                "",
                f"- Action accuracy: {metrics.get('action_accuracy', 0.0):.4f}",
                f"- Action balanced accuracy: {metrics.get('action_balanced_accuracy', 0.0):.4f}",
                f"- Action macro F1: {metrics.get('action_macro_f1', 0.0):.4f}",
                f"- Action weighted F1: {metrics.get('action_weighted_f1', 0.0):.4f}",
                f"- Future target MAE: {metrics.get('future_target_mae', 0.0):.4f}",
                f"- Future target RMSE: {metrics.get('future_target_rmse', 0.0):.4f}",
                f"- Future target R2: {metrics.get('future_target_r2', 0.0):.4f}",
                f"- Future target Pearson: {metrics.get('future_target_pearson', 0.0):.4f}",
                f"- Cycle precision: {metrics.get('cycle_precision', 0.0):.4f}",
                f"- Cycle recall: {metrics.get('cycle_recall', 0.0):.4f}",
                f"- Cycle F1: {metrics.get('cycle_f1', 0.0):.4f}",
                f"- Profitable cycle rate: {metrics.get('profitable_cycle_rate', 0.0):.4f}",
                f"- Average cycle return: {metrics.get('average_cycle_return', 0.0):.4f}",
                f"- Median cycle return: {metrics.get('median_cycle_return', 0.0):.4f}",
                f"- Moving avg cycle return: {metrics.get('moving_avg_cycle_return', 0.0):.4f}",
                f"- Catastrophic cycle rate: {metrics.get('catastrophic_cycle_rate', 0.0):.4f}",
                f"- Bounded exit error: {metrics.get('bounded_exit_error', 1.0):.4f}",
                "",
            ]
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    return str(json_path), str(md_path)


def save_checkpoint(
    model: nn.Module,
    standardizer: FeatureStandardizer,
    config: dict[str, Any],
    split_meta: dict[str, Any],
) -> str:
    model_dir = Path(config["training"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "final_model.pt"
    payload = {
        "model_state": model.state_dict(),
        "standardizer": standardizer.to_dict(),
        "feature_cols": split_meta["feature_cols"],
        "future_target_cols": split_meta["future_target_cols"],
        "model_config": deepcopy(config["model"]),
        "evaluation_config": deepcopy(config["evaluation"]),
        "split_meta": split_meta,
    }
    torch.save(payload, checkpoint_path)
    return str(checkpoint_path)


def load_checkpoint(
    checkpoint_path: str, device: torch.device | None = None
) -> tuple[nn.Module, dict[str, Any]]:
    map_location = device if device is not None else "cpu"
    payload = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    feature_cols = payload["feature_cols"]
    future_target_cols = payload["future_target_cols"]
    model_cfg = payload["model_config"]
    model = build_cycle_model(
        model_cfg=model_cfg,
        input_dim=len(feature_cols),
        future_target_dim=len(future_target_cols),
        action_dim=4,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def train(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    training_cfg = config["training"]
    device_name = training_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    set_seed(int(training_cfg["seed"]))

    frame = load_precomputed_frame(config)
    raw_frames, datasets, standardizer, split_meta = make_datasets(frame, config)
    model = build_model(
        config=config,
        input_dim=len(split_meta["feature_cols"]),
        future_target_dim=len(split_meta["future_target_cols"]),
    ).to(device)

    class_weights = compute_class_weights(raw_frames["train"]["oracle_action"]).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    train_loader = make_loader(
        datasets["train"],
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
    )
    best_state = None
    best_val_score = float("-inf")

    for epoch in range(int(training_cfg["epochs"])):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            class_weights=class_weights,
            device=device,
            config=config,
        )
        val_metrics = evaluate_split(
            model=model,
            dataset=datasets["val"],
            raw_frame=raw_frames["val"],
            device=device,
            config=config,
        )
        score = (
            val_metrics["cycle_precision"]
            + val_metrics["cycle_recall"]
            + val_metrics["profitable_cycle_rate"]
            - val_metrics["catastrophic_cycle_rate"]
            - val_metrics["bounded_exit_error"]
        )
        logger.info(
            "Epoch %s train_loss=%.4f val_precision=%.4f val_recall=%.4f profitable=%.4f catastrophic=%.4f",
            epoch + 1,
            train_loss,
            val_metrics["cycle_precision"],
            val_metrics["cycle_recall"],
            val_metrics["profitable_cycle_rate"],
            val_metrics["catastrophic_cycle_rate"],
        )
        if score > best_val_score:
            best_val_score = score
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)

    report = {
        "val": evaluate_split(
            model=model,
            dataset=datasets["val"],
            raw_frame=raw_frames["val"],
            device=device,
            config=config,
        ),
        "test": evaluate_split(
            model=model,
            dataset=datasets["test"],
            raw_frame=raw_frames["test"],
            device=device,
            config=config,
        ),
    }
    if "holdout" in datasets:
        report["holdout"] = evaluate_split(
            model=model,
            dataset=datasets["holdout"],
            raw_frame=raw_frames["holdout"],
            device=device,
            config=config,
        )

    checkpoint_path = save_checkpoint(model, standardizer, config, split_meta)
    report_paths = write_report(report, config, split_meta)

    summary = {
        "checkpoint_path": checkpoint_path,
        "report_json": report_paths[0],
        "report_md": report_paths[1],
        "split_meta": split_meta,
        "metrics": report,
    }
    logger.info("Saved checkpoint to %s", checkpoint_path)
    logger.info("Saved reports to %s and %s", *report_paths)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    train(config_path=args.config_path)
