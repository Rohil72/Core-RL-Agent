from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from torch.amp import autocast
except ImportError:  # pragma: no cover - older torch fallback
    from torch.cuda.amp import autocast

try:
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover - older torch fallback
    from torch.cuda.amp import GradScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cycle.cycle_detector import detect_cycles
from src.cycle.oracle import (
    annotate_cycle_targets,
    decode_action_spans,
    decode_return_spans,
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
from src.losses.outcome_geometry import (
    OutcomeGeometryLossConfig,
    OutcomeTargetNormalizer,
    LossBreakdown,
    compute_market_memory_loss,
)
from src.memory.outcomes import RelativeOutcomeConfig, attach_relative_outcomes
from src.models import build_cycle_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = "configs/cycle_model.yaml"


class TrainingInterrupted(RuntimeError):
    """Raised after a requested shutdown has been checkpointed safely."""


_STOP_REQUESTED = False


def _request_training_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    logger.warning("Received signal %s; checkpointing after the active optimizer step.", signum)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable digest of the immutable training contract."""
    stable = deepcopy({
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    })
    for operational_key in (
        "auto_resume",
        "checkpoint_interval_steps",
        "allow_hardware_mismatch_resume",
    ):
        stable.get("training", {}).pop(operational_key, None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runtime_fingerprint(device: torch.device) -> dict[str, Any]:
    """Capture software and accelerator properties required for strict resume."""
    payload: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "device_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "total_memory": int(properties.total_memory),
            }
        )
    return payload


def _rng_state(loader: DataLoader | None = None) -> dict[str, Any]:
    """Capture all RNG streams used by training and data ordering."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if loader is not None and loader.generator is not None:
        state["loader"] = loader.generator.get_state()
    return state


def _restore_rng_state(state: dict[str, Any], loader: DataLoader | None = None) -> None:
    """Restore RNG streams from a resumable training checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if loader is not None and loader.generator is not None and "loader" in state:
        loader.generator.set_state(state["loader"])


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a Torch payload atomically so preemption cannot leave a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    """Write JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


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
    data_cfg = config["data"]
    source_specs = data_cfg.get("precomputed_sources")
    if source_specs is None:
        source_specs = [{"market": data_cfg.get("market", "unknown"), "glob": data_cfg["precomputed_dir"]}]
    matched: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source in source_specs:
        pattern = str(source["glob"])
        market = str(source.get("market", "unknown"))
        for path in sorted(glob.glob(pattern, recursive=True)):
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                matched.append((path, market))
                seen.add(resolved)
    if not matched:
        raise FileNotFoundError(
            f"No precomputed parquet files found for configured sources: {source_specs}"
        )

    prepared = []
    oracle_cfg = config["oracle"]

    for path, market in matched:
        df = read_dataframe(path)
        if df.empty:
            continue
        df = _ensure_datetime_index(df)
        ticker = (
            str(df["ticker"].iloc[0]) if "ticker" in df.columns else Path(path).stem
        )
        df["ticker"] = ticker
        df["market"] = market
        df = ensure_sequence_model_features(df)
        use_detector_targets = bool(config.get("data", {}).get("use_detector_targets", True))
        if not use_detector_targets:
            df["oracle_action"] = 0
            df["oracle_cycle_id"] = -1
            df["oracle_hard_negative"] = 0.0
        elif _has_precomputed_oracle_targets(df):
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

    relative_cfg = data_cfg.get("relative_outcomes", {})
    if bool(relative_cfg.get("enabled", False)):
        with_timestamp = frame.copy()
        with_timestamp["timestamp"] = with_timestamp.index
        frame = attach_relative_outcomes(
            with_timestamp,
            RelativeOutcomeConfig(
                return_target=str(relative_cfg.get("return_target", "future_return_63")),
                sector_column=str(relative_cfg.get("sector_column", "sector")),
                minimum_sector_observations=int(
                    relative_cfg.get("minimum_sector_observations", 3)
                ),
                universe_alpha_column=str(
                    relative_cfg.get("universe_alpha_column", "future_universe_alpha_63")
                ),
                blended_alpha_column=str(
                    relative_cfg.get("blended_alpha_column", "future_blended_alpha_63")
                ),
                sector_weight=float(relative_cfg.get("sector_weight", 0.50)),
                group_column=(
                    str(relative_cfg["group_column"])
                    if relative_cfg.get("group_column") is not None
                    else None
                ),
            ),
        ).drop(columns=["timestamp"])

    relative_features = data_cfg.get("cross_sectional_relative_features", {})
    if relative_features:
        if not isinstance(relative_features, dict):
            raise ValueError("data.cross_sectional_relative_features must be a mapping.")
        timestamps = pd.Series(frame.index, index=frame.index)
        markets = frame["market"].astype(str)
        for source, destination in relative_features.items():
            if source not in frame:
                raise ValueError(f"Relative feature source {source!r} is unavailable.")
            values = pd.to_numeric(frame[source], errors="coerce")
            benchmark = values.groupby([markets, timestamps]).transform("median")
            frame[str(destination)] = values - benchmark

    # Optionally exclude particular calendar years (e.g., 2020) to avoid contamination
    exclude_years = config.get("data", {}).get("exclude_years", [])
    if exclude_years:
        mask = ~np.isin(frame.index.year, exclude_years)
        frame = frame.loc[mask]

    feature_cols = _infer_feature_columns(
        frame, config.get("features", {}).get("sequence")
    )
    future_target_cols = _infer_future_target_columns(
        frame, config.get("features", {}).get("future_targets")
    )

    # Safety check to prevent accidental leakage: ensure no feature column is also a future target
    _overlap = set(feature_cols) & set(future_target_cols)
    if _overlap:
        raise ValueError(
            f"Feature/target overlap detected. This may leak future information into features: {sorted(_overlap)}"
        )

    required = feature_cols + ["oracle_action", "ticker", "close"]
    clean = frame.dropna(subset=required).copy()
    clean.attrs["feature_cols"] = feature_cols
    clean.attrs["future_target_cols"] = future_target_cols
    return clean


def make_datasets(
    frame: pd.DataFrame,
    config: dict[str, Any],
    standardizer_override: FeatureStandardizer | None = None,
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

    configured_holdout_tickers = split_cfg.get("holdout_tickers")
    if configured_holdout_tickers is not None:
        if not isinstance(configured_holdout_tickers, list):
            raise ValueError("split.holdout_tickers must be a list when provided.")
        available_tickers = set(frame["ticker"].astype(str).unique().tolist())
        holdout_tickers = [str(ticker) for ticker in configured_holdout_tickers]
        unknown_tickers = sorted(set(holdout_tickers) - available_tickers)
        if unknown_tickers:
            raise ValueError(
                "Configured holdout tickers are absent from the dataset: "
                + ", ".join(unknown_tickers)
            )
        if len(set(holdout_tickers)) != len(holdout_tickers):
            raise ValueError("split.holdout_tickers must not contain duplicates.")
    else:
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

    evaluation_periods = split_cfg.get("evaluation_periods", {})
    if evaluation_periods and not isinstance(evaluation_periods, dict):
        raise ValueError("split.evaluation_periods must be a mapping.")
    for name, period in evaluation_periods.items():
        if name in {"train", "val", "test", "holdout"}:
            raise ValueError(f"Reserved evaluation period name: {name!r}.")
        start = pd.Timestamp(str(period["start"]), tz="UTC")
        end = pd.Timestamp(str(period["end"]), tz="UTC")
        if end < start:
            raise ValueError(f"Evaluation period {name!r} ends before it starts.")
        buffered[name] = build_buffered_period_frame(
            frame,
            start,
            end,
            history_rows=history_rows,
        )

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

    standardizer = standardizer_override or FeatureStandardizer.from_frame(train_frame, feature_cols)
    scaled_frames = {
        "train": standardizer.transform(train_frame, feature_cols),
        "val": standardizer.transform(val_frame, feature_cols),
        "test": standardizer.transform(test_frame, feature_cols),
    }
    if not buffered["holdout"].empty:
        scaled_frames["holdout"] = standardizer.transform(
            buffered["holdout"], feature_cols
        )
    for name in evaluation_periods:
        if buffered[name].empty:
            raise ValueError(f"Evaluation period {name!r} contains no rows.")
        scaled_frames[name] = standardizer.transform(buffered[name], feature_cols)

    window_size = int(config["model"]["window_size"])
    datasets = {
        name: CycleSequenceDataset(
            frame=scaled_frames[name],
            feature_cols=feature_cols,
            future_target_cols=future_target_cols,
            window_size=window_size,
            target_start=(
                pd.Timestamp(str(evaluation_periods[name]["start"]), tz="UTC")
                if name in evaluation_periods
                else split["train_start"]
                if name == "train"
                else split["val_start"]
                if name == "val"
                else split["test_start"]
            ),
            target_end=(
                pd.Timestamp(str(evaluation_periods[name]["end"]), tz="UTC")
                if name in evaluation_periods
                else split["train_end"]
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
    for name in evaluation_periods:
        raw_frames[name] = buffered[name]

    split_meta = {
        "selected_fold": selected_fold,
        "fold": {key: value.isoformat() for key, value in split.items()},
        "holdout_tickers": holdout_tickers,
        "feature_cols": feature_cols,
        "future_target_cols": future_target_cols,
        "evaluation_periods": {
            name: {"start": str(period["start"]), "end": str(period["end"])}
            for name, period in evaluation_periods.items()
        },
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
    dataset: CycleSequenceDataset,
    batch_size: int,
    shuffle: bool,
    market_balanced: bool = False,
) -> DataLoader:
    # Use a seeded torch.Generator tied to the global PyTorch RNG seed for deterministic shuffling
    gen = torch.Generator()
    try:
        seed_val = int(torch.initial_seed() & 0x7FFFFFFF)
    except Exception:
        seed_val = 7
    gen.manual_seed(seed_val)
    sampler = None
    if market_balanced:
        markets = np.asarray(dataset.sample_markets, dtype=str)
        labels, counts = np.unique(markets, return_counts=True)
        inverse = {label: 1.0 / count for label, count in zip(labels, counts)}
        weights = torch.as_tensor([inverse[label] for label in markets], dtype=torch.double)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=gen,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
        generator=gen,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


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
    valid_mask = torch.isfinite(target)
    if not valid_mask.any():
        return prediction.new_tensor(0.0)
    return F.smooth_l1_loss(
        prediction[valid_mask], target[valid_mask], reduction="mean"
    )


def _batch_debug_payload(batch: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            payload[key] = value.detach().cpu()
        else:
            payload[key] = value
    return payload


def _dump_training_failure(
    *,
    reason: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> None:
    dump_path = Path("debug")
    dump_path.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    payload = {
        "reason": reason,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "batch": _batch_debug_payload(batch) if batch is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, dump_path / f"emergency_checkpoint_{reason}_{ts}.pt")


def _first_nonfinite_parameter(model: nn.Module) -> str | None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param.detach()).all():
            return name
    return None


def _first_nonfinite_buffer(model: nn.Module) -> str | None:
    for name, value in model.named_buffers():
        if torch.is_tensor(value) and not torch.isfinite(value.detach()).all():
            return name
    return None


def _first_nonfinite_gradient(model: nn.Module) -> str | None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad.detach()).all():
            return name
    return None


def _global_norm(tensors: list[torch.Tensor]) -> float:
    if not tensors:
        return 0.0
    total = sum(t.detach().to(torch.float64).norm().pow(2) for t in tensors)
    return float(torch.sqrt(total).item())


def _recover_amp_overflow(
    scaler: GradScaler,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Skip an overflowed AMP update and let GradScaler reduce its scale."""
    previous_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return previous_scale, float(scaler.get_scale())


def _resolve_amp_dtype(training_cfg: dict[str, Any], device: torch.device) -> torch.dtype:
    bf16_supported = bool(
        device.type == "cuda"
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    requested = str(training_cfg.get("amp_dtype", "auto")).lower()
    if requested in {"float16", "fp16", "half"}:
        return torch.float16
    if requested in {"bfloat16", "bf16"}:
        if device.type == "cuda" and not bf16_supported:
            raise RuntimeError("training.amp_dtype=bf16 requires BF16-capable CUDA hardware.")
        return torch.bfloat16
    if requested != "auto":
        raise ValueError("training.amp_dtype must be auto, float16, or bfloat16.")
    if bf16_supported:
        return torch.bfloat16
    return torch.float16


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
    target_normalizer: OutcomeTargetNormalizer | None = None,
    loss_config: OutcomeGeometryLossConfig | None = None,
    scaler: GradScaler | None = None,
    start_batch: int = 0,
    initial_total_loss: float = 0.0,
    initial_total_count: int = 0,
    progress_callback: Callable[[int, float, int], None] | None = None,
) -> float:
    """Train for one epoch with stability guards and masked Huber future-outcome loss."""
    model.train()
    total_loss = float(initial_total_loss)
    total_count = int(initial_total_count)
    epoch_transport_triplets = 0
    epoch_analogue_anchors = 0

    # training hyperparams & new knobs
    training_cfg = config["training"]
    action_loss_weight = float(training_cfg.get("action_loss_weight", 1.0))
    future_loss_weight = float(training_cfg["future_loss_weight"])
    hard_negative_weight = float(training_cfg["hard_negative_weight"])
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    reconstruction_loss_weight = float(
        config.get("self_supervised", {}).get("reconstruction_loss_weight", 0.0)
    )

    profit_weight_lambda = float(training_cfg.get("profit_weight_lambda", 4.0))
    profit_weight_Rmax = float(training_cfg.get("profit_weight_Rmax", 0.5))
    label_smoothing = float(training_cfg.get("label_smoothing", 0.05))

    # precompute target index for ROI signal
    target_names = list(getattr(loader.dataset, "future_target_cols", []))
    ret_idx = None
    if "future_max_return_63" in target_names:
        ret_idx = target_names.index("future_max_return_63")
    elif "future_return_63" in target_names:
        ret_idx = target_names.index("future_return_63")

    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(training_cfg, device)
    amp_forward_fallback = bool(
        training_cfg.get("amp_forward_fallback_to_fp32", False)
    )
    amp_forward_fallback_count = 0
    non_blocking = device.type == "cuda"
    consecutive_amp_overflows = 0
    maximum_consecutive_amp_overflows = int(
        training_cfg.get("maximum_consecutive_amp_overflows", 8)
    )
    if maximum_consecutive_amp_overflows < 1:
        raise ValueError("training.maximum_consecutive_amp_overflows must be at least 1.")

    for batch_idx, batch in enumerate(loader):
        if batch_idx < start_batch:
            continue
        sequence = batch["sequence"].to(device, non_blocking=non_blocking)
        action = batch["action"].to(device, non_blocking=non_blocking)
        future_target = batch["future_target"].to(device, non_blocking=non_blocking)
        hard_negative = batch["hard_negative"].to(device, non_blocking=non_blocking)

        # Missing future targets are represented as NaN and masked in the Huber loss.
        # Inputs, labels, hard-negative flags, and infinite targets must still be clean.
        has_bad_target_inf = torch.isinf(future_target).any()
        has_bad_action = bool(((action < 0) | (action > 3)).any().item())
        if (
            not torch.isfinite(sequence).all()
            or not torch.isfinite(action.float()).all()
            or not torch.isfinite(hard_negative).all()
            or has_bad_target_inf
            or has_bad_action
        ):
            _dump_training_failure(
                reason="nonfinite_input",
                model=model,
                optimizer=optimizer,
                batch=batch,
                extra={
                    "future_target_has_inf": bool(has_bad_target_inf.item()),
                    "bad_action_label": has_bad_action,
                },
            )
            raise RuntimeError("Non-finite values found in input batch; dumped debug artifacts.")

        bad_param = _first_nonfinite_parameter(model)
        if bad_param is not None:
            _dump_training_failure(
                reason="nonfinite_parameter_before_forward",
                model=model,
                optimizer=optimizer,
                batch=batch,
                extra={"parameter": bad_param},
            )
            raise RuntimeError(f"Non-finite parameter before forward: {bad_param}")
        bad_buffer = _first_nonfinite_buffer(model)
        if bad_buffer is not None:
            _dump_training_failure(
                reason="nonfinite_buffer_before_forward",
                model=model,
                optimizer=optimizer,
                batch=batch,
                extra={"buffer": bad_buffer},
            )
            raise RuntimeError(f"Non-finite model buffer before forward: {bad_buffer}")

        model_input = sequence
        reconstruction_mask = None
        if reconstruction_loss_weight > 0:
            model_input, reconstruction_mask = _apply_self_supervised_mask(
                sequence,
                config,
            )

        # Forward + loss under AMP if enabled
        with autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            outputs = _forward_model(
                model,
                model_input,
                return_reconstruction=reconstruction_loss_weight > 0,
            )

            logits = outputs["action_logits"]
            pred = outputs["future_pred"]
            if (
                (not torch.isfinite(logits).all() or not torch.isfinite(pred).all())
                and use_amp
                and amp_forward_fallback
                and _first_nonfinite_buffer(model) is None
            ):
                logger.warning(
                    "Non-finite %s AMP output at batch=%d; retrying forward in FP32.",
                    str(amp_dtype).removeprefix("torch."),
                    batch_idx,
                )
                with autocast(device_type=device.type, enabled=False):
                    outputs = _forward_model(
                        model,
                        model_input.float(),
                        return_reconstruction=reconstruction_loss_weight > 0,
                    )
                logits = outputs["action_logits"]
                pred = outputs["future_pred"]
                if torch.isfinite(logits).all() and torch.isfinite(pred).all():
                    amp_forward_fallback_count += 1
            if not torch.isfinite(logits).all() or not torch.isfinite(pred).all():
                bad_buffer = _first_nonfinite_buffer(model)
                _dump_training_failure(
                    reason="nonfinite_model_output",
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    extra={
                        "logits_finite": bool(torch.isfinite(logits).all().item()),
                        "future_pred_finite": bool(torch.isfinite(pred).all().item()),
                        "amp_dtype": str(amp_dtype),
                        "nonfinite_buffer": bad_buffer,
                    },
                )
                raise FloatingPointError("Non-finite model output.")

            action_loss = pred.new_tensor(0.0)
            if action_loss_weight > 0:
                ce = F.cross_entropy(
                    logits,
                    action,
                    weight=class_weights,
                    reduction="none",
                    label_smoothing=label_smoothing,
                )
                row_weights = 1.0 + hard_negative * hard_negative_weight
                batch_size = int(sequence.size(0))
                if ret_idx is not None and profit_weight_lambda > 0:
                    ret_vals = future_target[:, ret_idx]
                    finite_mask = torch.isfinite(ret_vals)
                    base_ret = torch.where(finite_mask, ret_vals, torch.zeros_like(ret_vals))
                    pos_ret = torch.clamp(base_ret, min=0.0)
                    normalized = torch.clamp(pos_ret, max=profit_weight_Rmax) / float(profit_weight_Rmax)
                    profit_weights = 1.0 + profit_weight_lambda * normalized
                else:
                    profit_weights = torch.ones(batch_size, device=device, dtype=ce.dtype)

                instance_weights = row_weights.to(dtype=ce.dtype) * profit_weights.to(dtype=ce.dtype)
                weighted_sum = (ce * instance_weights).sum()
                denom = instance_weights.sum().clamp_min(1e-9)
                action_loss = weighted_sum / denom

            if loss_config is not None and target_normalizer is not None:
                # New Phase 4 loss logic
                loss_breakdown = compute_market_memory_loss(
                    latent=outputs["latent"],
                    future_prediction=pred,
                    future_target=future_target,
                    ticker=batch["ticker"],
                    market=batch.get("market"),
                    timestamp=batch.get("timestamp"),
                    target_names=target_names,
                    target_normalizer=target_normalizer,
                    config=loss_config,
                )
                epoch_transport_triplets += loss_breakdown.transport_triplet_count
                epoch_analogue_anchors += loss_breakdown.eligible_anchor_count
                
                # We need to compute gradients occasionally for diagnostics
                if batch_idx % 100 == 0:
                    try:
                        # Extract components that require grad
                        L_reg = loss_breakdown.regression * loss_config.lambda_reg
                        L_ana = loss_breakdown.analogue * loss_config.lambda_analogue
                        L_rnk = loss_breakdown.ranking * loss_config.lambda_rank
                        
                        shared_latent = outputs["latent"]
                        # Just a quick check to see if we can autograd
                        if shared_latent.requires_grad:
                            if L_reg.requires_grad and float(L_reg) > 0:
                                g_reg = torch.autograd.grad(L_reg, shared_latent, retain_graph=True)[0]
                            if L_ana.requires_grad and float(L_ana) > 0:
                                g_ana = torch.autograd.grad(L_ana, shared_latent, retain_graph=True)[0]
                            if L_rnk.requires_grad and float(L_rnk) > 0:
                                g_rnk = torch.autograd.grad(L_rnk, shared_latent, retain_graph=True)[0]
                    except Exception as e:
                        logger.debug(f"Gradient diagnostic failed: {e}")
                
                future_loss = loss_breakdown.total
                loss = action_loss_weight * action_loss + future_loss
            else:
                # Future regression loss (Huber / smooth L1) across configured future targets
                weights = [1.0] * (pred.size(1) if pred.dim() > 1 else 1)
                try:
                    idx = target_names.index("future_max_return_63")
                    weights[idx] = float(training_cfg.get("max_return_loss_weight", 2.0))
                except ValueError:
                    pass
                try:
                    idx = target_names.index("future_min_return_63")
                    weights[idx] = float(training_cfg.get("min_return_loss_weight", 2.0))
                except ValueError:
                    pass
    
                future_loss_total = pred.new_tensor(0.0)
                total_w = 0.0
                for k, w in enumerate(weights):
                    future_loss_total = future_loss_total + float(w) * _compute_future_loss(
                        pred[:, k], future_target[:, k]
                    )
                    total_w += float(w)
                future_loss = future_loss_total / max(total_w, 1e-9)
    
                loss = action_loss_weight * action_loss + future_loss_weight * future_loss

            if reconstruction_loss_weight > 0 and reconstruction_mask is not None:
                if reconstruction_mask.any():
                    reconstruction_loss = F.smooth_l1_loss(
                        outputs["reconstruction"][reconstruction_mask],
                        sequence[reconstruction_mask],
                        reduction="mean",
                    )
                    loss = loss + reconstruction_loss_weight * reconstruction_loss

        # Sanity check on loss
        if not torch.isfinite(loss):
            min_latent_norm = float(outputs["latent"].norm(p=2, dim=1).min().item()) if "latent" in outputs else None
            extra_payload = {
                "loss": float(loss.detach().cpu()) if loss.numel() == 1 else None,
                "min_latent_norm": min_latent_norm
            }
            if loss_config is not None and 'loss_breakdown' in locals():
                extra_payload.update({
                    "regression_loss": float(loss_breakdown.regression.detach().cpu()),
                    "analogue_loss": float(loss_breakdown.analogue.detach().cpu()),
                    "ranking_loss": float(loss_breakdown.ranking.detach().cpu()),
                    "variance_loss": float(loss_breakdown.variance.detach().cpu()),
                    "covariance_loss": float(loss_breakdown.covariance.detach().cpu()),
                    "transport_loss": float(loss_breakdown.transport.detach().cpu()),
                })
            _dump_training_failure(
                reason="nonfinite_loss",
                model=model,
                optimizer=optimizer,
                batch=batch,
                extra=extra_payload,
            )
            logger.error(f"Non-finite loss encountered! extra payload: {extra_payload}")
            raise RuntimeError("Non-finite loss encountered; dumped debug artifacts.")

        # Backward + optimizer step with optional AMP scaling and gradient clipping
        optimizer.zero_grad()
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            # unscale before clipping
            scaler.unscale_(optimizer)
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            bad_grad = _first_nonfinite_gradient(model)
            if bad_grad is not None:
                consecutive_amp_overflows += 1
                previous_scale, updated_scale = _recover_amp_overflow(
                    scaler,
                    optimizer,
                )
                logger.warning(
                    "AMP overflow at batch=%d parameter=%s scale=%.6g->%.6g "
                    "consecutive=%d/%d; optimizer update skipped",
                    batch_idx,
                    bad_grad,
                    previous_scale,
                    updated_scale,
                    consecutive_amp_overflows,
                    maximum_consecutive_amp_overflows,
                )
                if consecutive_amp_overflows >= maximum_consecutive_amp_overflows:
                    _dump_training_failure(
                        reason="repeated_amp_overflow",
                        model=model,
                        optimizer=optimizer,
                        batch=batch,
                        extra={
                            "parameter": bad_grad,
                            "previous_scale": previous_scale,
                            "updated_scale": updated_scale,
                            "consecutive_overflows": consecutive_amp_overflows,
                        },
                    )
                    raise RuntimeError(
                        "Repeated AMP gradient overflow: "
                        f"{bad_grad} ({consecutive_amp_overflows} consecutive batches)"
                    )
                if progress_callback is not None:
                    progress_callback(batch_idx + 1, total_loss, total_count)
                continue
            consecutive_amp_overflows = 0
            grad_norm = _global_norm(grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            bad_grad = _first_nonfinite_gradient(model)
            if bad_grad is not None:
                _dump_training_failure(
                    reason="nonfinite_gradient",
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    extra={"parameter": bad_grad},
                )
                raise RuntimeError(f"Non-finite gradient: {bad_grad}")
            grad_norm = _global_norm(grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # parameter norm for diagnostics
        params = [p.data for p in model.parameters() if p.data is not None]
        bad_param = _first_nonfinite_parameter(model)
        if bad_param is not None:
            _dump_training_failure(
                reason="nonfinite_parameter_after_step",
                model=model,
                optimizer=optimizer,
                batch=batch,
                extra={"parameter": bad_param},
            )
            raise RuntimeError(f"Non-finite parameter after optimizer step: {bad_param}")
        param_norm = _global_norm(params)

        # log diagnostics
        current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
        logger.info(
            "batch=%d loss=%.5f lr=%.6g param_norm=%.4f grad_norm=%.4f",
            batch_idx,
            float(loss.item()),
            current_lr,
            param_norm,
            grad_norm,
        )

        batch_size = int(sequence.size(0))
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        if progress_callback is not None:
            progress_callback(batch_idx + 1, total_loss, total_count)

    if loss_config is not None:
        logger.info(
            "Outcome geometry coverage: analogue_anchors=%d transport_triplets=%d "
            "fp32_forward_fallbacks=%d",
            epoch_analogue_anchors,
            epoch_transport_triplets,
            amp_forward_fallback_count,
        )
        if (
            loss_config.lambda_transport > 0
            and start_batch == 0
            and epoch_transport_triplets == 0
        ):
            raise RuntimeError(
                "Temporal transport loss produced zero eligible triplets for the epoch."
            )
    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str | None = None,
    export_latents: bool = False,
    latent_output_dir: str | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    records = []
    future_true_batches = []
    future_pred_batches = []

    target_names = list(getattr(loader.dataset, "future_target_cols", []))
    non_blocking = device.type == "cuda"
    for batch in loader:
        sequence = batch["sequence"].to(device, non_blocking=non_blocking)
        action = batch["action"].cpu().numpy()
        future_target = batch["future_target"].to(device, non_blocking=non_blocking)
 
        outputs = _forward_model(model, sequence)
        logits = outputs["action_logits"]
        pred_action = logits.argmax(dim=1).cpu().numpy()
 
        future_true_batches.append(future_target.cpu().numpy())
        future_pred_batches.append(outputs["future_pred"].cpu().numpy())
 
        future_preds = outputs["future_pred"].cpu().numpy()
        latent_vals = outputs.get("latent")
        if latent_vals is not None:
            latent_vals = latent_vals.cpu().numpy()
        for i, (ticker, timestamp, true_action, predicted_action) in enumerate(zip(
            batch["ticker"],
            batch["timestamp"],
            action,
            pred_action,
        )):
            rec = {
                "ticker": ticker,
                "market": batch.get("market", ["unknown"] * len(action))[i],
                "timestamp": timestamp,
                "true_action": int(true_action),
                "pred_action": int(predicted_action),
            }
            # include latent vector (as list) if available
            if latent_vals is not None:
                try:
                    rec["latent"] = latent_vals[i].astype(float).tolist()
                except Exception:
                    rec["latent"] = None
            # attach per-target future predictions so they can be joined later
            if future_preds is not None and len(target_names) == future_preds.shape[1]:
                for tname, val in zip(target_names, future_preds[i]):
                    try:
                        rec[f"pred_{tname}"] = float(val) if np.isfinite(val) else float("nan")
                    except Exception:
                        rec[f"pred_{tname}"] = float("nan")
            # also include true future target values for export convenience
            try:
                true_vals = future_target.cpu().numpy()[i]
                if len(true_vals) == len(target_names):
                    for tname, val in zip(target_names, true_vals):
                        try:
                            rec[f"true_{tname}"] = float(val) if np.isfinite(val) else float("nan")
                        except Exception:
                            rec[f"true_{tname}"] = float("nan")
            except Exception:
                pass
            records.append(rec)

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
    if export_latents:
        try:
            latent_path = _export_latents_for_analysis(pred_df, split_name or "unknown", latent_output_dir)
            future_metrics["latent_export_path"] = latent_path
        except Exception:
            logger.exception("Failed to export latent analysis data")
    return pred_df, {**action_metrics, **future_metrics}


def _export_latents_for_analysis(pred_df: pd.DataFrame, split_name: str, output_dir: str | None = None) -> str:
    if output_dir:
        latent_dir = Path(output_dir)
        latent_name = f"{split_name}_latents.parquet"
    else:
        latent_dir = Path("reports/latent_analysis")
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        latent_name = f"{split_name}_latents_{ts}.parquet"
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent_path = latent_dir / latent_name

    rows = []
    for rec in pred_df.to_dict("records"):
        latent = rec.get("latent")
        if not isinstance(latent, list):
            continue
        row = {
            "split": split_name,
            "ticker": rec.get("ticker"),
            "market": rec.get("market", "unknown"),
            "timestamp": rec.get("timestamp"),
        }
        for idx, value in enumerate(latent):
            row[f"latent_{idx}"] = float(value)
        for col in pred_df.columns:
            is_target_col = col.startswith("true_future_") or col.startswith("pred_future_")
            is_event_col = col.startswith("true_event_") or col.startswith("pred_event_")
            if is_target_col or is_event_col:
                export_col = col.replace("true_", "")
                row[export_col] = rec.get(col)
        rows.append(row)

    export_df = pd.DataFrame(rows)
    if export_df.empty:
        raise ValueError("No latent vectors were available to export.")
    export_df.to_parquet(latent_path, index=False)
    logger.info("Exported %s latent vectors to %s", len(export_df), str(latent_path))
    return str(latent_path)


def evaluate_split(
    model: nn.Module,
    dataset: CycleSequenceDataset,
    raw_frame: pd.DataFrame,
    device: torch.device,
    config: dict[str, Any],
    split_name: str | None = None,
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
    pred_df, action_metrics = collect_predictions(
        model,
        loader,
        device,
        split_name=split_name or config.get("_active_split_name"),
        export_latents=_should_export_latents(config, split_name or config.get("_active_split_name")),
        latent_output_dir=config.get("latent_export", {}).get("output_dir"),
    )

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
        cols_to_join = ["pred_action", "true_action"] + [c for c in ticker_pred.columns if c.startswith("pred_future_")]
        eval_frame = eval_frame.join(
            ticker_pred[cols_to_join], how="inner"
        )
        if eval_frame.empty:
            continue
        oracle_cycles = extract_oracle_spans(eval_frame)
        eval_method = config.get("evaluation", {}).get("cycle_extraction_method", "return_threshold")
        if eval_method == "return_threshold":
            enter = float(config.get("evaluation", {}).get("return_enter_threshold", 0.2))
            exit_thr = float(config.get("evaluation", {}).get("return_exit_threshold", 0.1))
            risk_thr = float(config.get("evaluation", {}).get("return_risk_threshold", 0.1))
            hysteresis_days = int(config.get("evaluation", {}).get("return_hysteresis_days", 5))
            cooldown_days = int(config.get("evaluation", {}).get("return_cooldown_days", 0))
            predicted_cycles = decode_return_spans(
                eval_frame,
                pred_max_col="pred_future_max_return_63",
                pred_min_col="pred_future_min_return_63",
                enter_threshold=enter,
                risk_threshold=risk_thr,
                exit_threshold=exit_thr,
                hysteresis_days=hysteresis_days,
                cooldown_days=cooldown_days,
            )
        else:
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


def _should_export_latents(config: dict[str, Any], split_name: str | None) -> bool:
    if bool(config.get("_export_latents", False)):
        return True
    if not split_name:
        return False
    export_cfg = config.get("latent_export", {})
    splits = export_cfg.get("splits", [])
    if isinstance(splits, str):
        splits = [s.strip() for s in splits.split(",") if s.strip()]
    return split_name in set(splits)


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
    _atomic_torch_save(payload, checkpoint_path)
    return str(checkpoint_path)


def _save_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config_fingerprint: str,
    runtime_fingerprint: dict[str, Any],
    loader: DataLoader,
    next_epoch: int,
    next_batch: int,
    epoch_total_loss: float,
    epoch_total_count: int,
    epoch_loader_state: torch.Tensor | None,
    best_val_score: float,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    """Persist a complete optimizer-level continuation point."""
    payload = {
        "format_version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint,
        "runtime_fingerprint": runtime_fingerprint,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": _rng_state(loader),
        "next_epoch": int(next_epoch),
        "next_batch": int(next_batch),
        "epoch_total_loss": float(epoch_total_loss),
        "epoch_total_count": int(epoch_total_count),
        "epoch_loader_state": epoch_loader_state,
        "best_val_score": float(best_val_score),
        "best_state": best_state,
    }
    _atomic_torch_save(payload, path)


def _load_training_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config_fingerprint: str,
    runtime_fingerprint: dict[str, Any],
    loader: DataLoader,
    allow_hardware_mismatch: bool,
) -> dict[str, Any]:
    """Load and validate an optimizer-level continuation point."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_fingerprint") != config_fingerprint:
        raise RuntimeError(
            "Refusing to resume because the training configuration changed. "
            "Start a new run directory for the new experiment."
        )
    previous_runtime = payload.get("runtime_fingerprint", {})
    if previous_runtime != runtime_fingerprint and not allow_hardware_mismatch:
        raise RuntimeError(
            "Refusing to resume on a different software or accelerator testbed. "
            "Set training.allow_hardware_mismatch_resume=true only for a run that "
            "will be explicitly marked non-comparable."
        )
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(next(model.parameters()).device)
    scaler.load_state_dict(payload.get("scaler_state", {}))
    _restore_rng_state(payload["rng_state"], loader)
    return payload


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


def train(
    config_path: str = DEFAULT_CONFIG_PATH,
    resume: bool | None = None,
) -> dict[str, Any]:
    """Train an encoder, optionally resuming an exact optimizer-level checkpoint."""
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    config = load_config(config_path)
    training_cfg = config["training"]
    resume_enabled = bool(training_cfg.get("auto_resume", False)) if resume is None else resume
    checkpoint_interval = max(int(training_cfg.get("checkpoint_interval_steps", 250)), 1)
    device_name = training_cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but this PyTorch installation cannot access a CUDA GPU.")

    if device.type == "cuda":
        vram_fraction = float(
            os.environ.get(
                "CORE_RL_VRAM_FRACTION",
                config.get("hardware", {}).get("vram_fraction", 0.80),
            )
        )
        if not 0.0 < vram_fraction <= 1.0:
            raise ValueError("VRAM fraction must be in the interval (0, 1].")
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(vram_fraction, device_index)
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        logger.info("Training on CUDA device: %s", torch.cuda.get_device_name(device))
        logger.info("CUDA allocator capped at %.1f%% of physical VRAM", 100.0 * vram_fraction)

    model_dir = Path(training_cfg["model_dir"])
    training_state_path = model_dir / "latest_training_state.pt"
    completion_path = model_dir / "training_complete.json"
    config_fingerprint = _config_fingerprint(config)
    runtime_fingerprint = _runtime_fingerprint(device)
    if resume_enabled and completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("config_fingerprint") != config_fingerprint:
            raise RuntimeError(
                "A completion marker exists for a different training configuration. "
                "Use a new model directory."
            )
        required_outputs = [
            completion.get("summary", {}).get("checkpoint_path"),
            completion.get("summary", {}).get("report_json"),
            completion.get("summary", {}).get("report_md"),
        ]
        if not all(value and Path(value).is_file() for value in required_outputs):
            raise RuntimeError(
                "Training completion marker exists, but a final artifact is missing. "
                "Restore the run directory or start a new run ID."
            )
        logger.info("Training is already complete; reusing %s", completion_path)
        return completion["summary"]

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
        market_balanced=bool(config.get("data", {}).get("market_balanced_sampling", False)),
    )

    # AMP scaler and option. Avoid positional args here because older torch
    # GradScaler variants interpret the first positional argument as init_scale.
    use_amp = bool(training_cfg.get("use_amp", True))
    amp_dtype = _resolve_amp_dtype(training_cfg, device)
    scaler = GradScaler(
        enabled=(use_amp and device.type == "cuda" and amp_dtype == torch.float16)
    )
    logger.info(
        "AMP enabled=%s dtype=%s grad_scaler=%s",
        use_amp and device.type == "cuda",
        str(amp_dtype).removeprefix("torch."),
        scaler.is_enabled(),
    )
    
    loss_cfg = config["training"].get("loss")
    loss_config = None
    target_normalizer = None
    if loss_cfg:
        loss_config = OutcomeGeometryLossConfig(
            lambda_reg=float(loss_cfg.get("lambda_reg", 1.0)),
            lambda_analogue=float(loss_cfg.get("lambda_analogue", 0.0)),
            lambda_rank=float(loss_cfg.get("lambda_rank", 0.0)),
            lambda_var=float(loss_cfg.get("lambda_var", 0.0)),
            is_supcon_control=bool(loss_cfg.get("is_supcon_control", False)),
            is_triplet_control=bool(loss_cfg.get("is_triplet_control", False)),
            triplet_margin=float(loss_cfg.get("triplet_margin", 0.20)),
            future_loss_weight=float(training_cfg.get("future_loss_weight", 1.0)),
            max_return_loss_weight=float(training_cfg.get("max_return_loss_weight", 1.0)),
            min_return_loss_weight=float(training_cfg.get("min_return_loss_weight", 1.0)),
            analogue_target_names=tuple(
                str(value)
                for value in loss_cfg.get(
                    "analogue_target_names",
                    (
                        "future_max_return_63",
                        "future_min_return_63",
                        "event_upside_before_drawdown_126",
                    ),
                )
            ),
            analogue_target_weights=tuple(
                float(value)
                for value in loss_cfg.get("analogue_target_weights", (1.0, 1.0, 0.5))
            ),
            analogue_candidate_mode=str(
                loss_cfg.get("analogue_candidate_mode", "cross_ticker")
            ),
            minimum_year_gap=int(loss_cfg.get("minimum_year_gap", 0)),
            balance_candidate_domains=bool(
                loss_cfg.get("balance_candidate_domains", False)
            ),
            lambda_transport=float(loss_cfg.get("lambda_transport", 0.0)),
            transport_positive_quantile=float(
                loss_cfg.get("transport_positive_quantile", 0.25)
            ),
            transport_negative_quantile=float(
                loss_cfg.get("transport_negative_quantile", 0.75)
            ),
            transport_margin=float(loss_cfg.get("transport_margin", 0.25)),
            variance_target=float(loss_cfg.get("variance_target", 0.20)),
            enable_variance_regularizer=bool(
                loss_cfg.get("enable_variance_regularizer", False)
            ),
        )
        target_normalizer = OutcomeTargetNormalizer()
        target_names = list(getattr(datasets["train"], "future_target_cols", []))
        train_targets = torch.tensor(datasets["train"].frame[target_names].to_numpy(), dtype=torch.float32, device=device)
        target_normalizer.fit(train_targets, target_names)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_score = float("-inf")
    start_epoch = 0
    start_batch = 0
    initial_total_loss = 0.0
    initial_total_count = 0
    resumed_epoch_loader_state = None
    if resume_enabled and training_state_path.exists():
        resumed = _load_training_state(
            training_state_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config_fingerprint=config_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            loader=train_loader,
            allow_hardware_mismatch=bool(
                training_cfg.get("allow_hardware_mismatch_resume", False)
            ),
        )
        start_epoch = int(resumed["next_epoch"])
        start_batch = int(resumed["next_batch"])
        initial_total_loss = float(resumed.get("epoch_total_loss", 0.0))
        initial_total_count = int(resumed.get("epoch_total_count", 0))
        resumed_epoch_loader_state = resumed.get("epoch_loader_state")
        best_val_score = float(resumed.get("best_val_score", float("-inf")))
        best_state = resumed.get("best_state")
        logger.info(
            "Resuming epoch %d at batch %d from %s",
            start_epoch + 1,
            start_batch,
            training_state_path,
        )

    previous_handlers: dict[int, Any] = {}
    termination_signals = {
        signal.SIGINT,
        getattr(signal, "SIGTERM", signal.SIGINT),
        getattr(signal, "SIGBREAK", signal.SIGINT),
    }
    if resume_enabled:
        for signum in termination_signals:
            try:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _request_training_stop)
            except (OSError, ValueError):
                pass

    try:
        for epoch in range(start_epoch, int(training_cfg["epochs"])):
            if start_batch > 0:
                if resumed_epoch_loader_state is None:
                    raise RuntimeError("Mid-epoch checkpoint is missing its loader state.")
                train_loader.generator.set_state(resumed_epoch_loader_state)
                epoch_loader_state = resumed_epoch_loader_state
            else:
                epoch_loader_state = train_loader.generator.get_state()

            def save_progress(next_batch: int, running_loss: float, running_count: int) -> None:
                should_save = resume_enabled and (
                    next_batch % checkpoint_interval == 0 or _STOP_REQUESTED
                )
                if should_save:
                    _save_training_state(
                        training_state_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config_fingerprint=config_fingerprint,
                        runtime_fingerprint=runtime_fingerprint,
                        loader=train_loader,
                        next_epoch=epoch,
                        next_batch=next_batch,
                        epoch_total_loss=running_loss,
                        epoch_total_count=running_count,
                        epoch_loader_state=epoch_loader_state,
                        best_val_score=best_val_score,
                        best_state=best_state,
                    )
                    logger.info("Saved resumable state at epoch %d batch %d", epoch + 1, next_batch)
                if _STOP_REQUESTED:
                    raise TrainingInterrupted(
                        f"Shutdown requested; continuation saved to {training_state_path}"
                    )

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                class_weights=class_weights,
                device=device,
                config=config,
                target_normalizer=target_normalizer,
                loss_config=loss_config,
                scaler=scaler,
                start_batch=start_batch,
                initial_total_loss=initial_total_loss,
                initial_total_count=initial_total_count,
                progress_callback=save_progress,
            )
            config["_active_split_name"] = "val"
            config["_export_latents"] = False
            val_metrics = evaluate_split(
                model=model,
                dataset=datasets["val"],
                raw_frame=raw_frames["val"],
                device=device,
                config=config,
            )
            selection_metric = str(training_cfg.get("validation_selection_metric", "future_target_mae"))
            selection_mode = str(training_cfg.get("validation_selection_mode", "min")).lower()
            metric_value = float(val_metrics.get(selection_metric, 0.0))
            score = -metric_value if selection_mode == "min" else metric_value
            logger.info(
                "Epoch %s train_loss=%.4f val_%s=%.6f val_future_mae=%.6f",
                epoch + 1,
                train_loss,
                selection_metric,
                metric_value,
                val_metrics["future_target_mae"],
            )
            if score > best_val_score:
                best_val_score = score
                best_state = deepcopy(model.state_dict())

            if resume_enabled:
                _save_training_state(
                    training_state_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config_fingerprint=config_fingerprint,
                    runtime_fingerprint=runtime_fingerprint,
                    loader=train_loader,
                    next_epoch=epoch + 1,
                    next_batch=0,
                    epoch_total_loss=0.0,
                    epoch_total_count=0,
                    epoch_loader_state=None,
                    best_val_score=best_val_score,
                    best_state=best_state,
                )
            if _STOP_REQUESTED:
                raise TrainingInterrupted(
                    f"Shutdown requested; continuation saved to {training_state_path}"
                )
            start_batch = 0
            initial_total_loss = 0.0
            initial_total_count = 0
            resumed_epoch_loader_state = None
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if best_state is None:
        raise RuntimeError("Training finished without a best checkpoint.")

    model.load_state_dict(best_state)

    report = {}
    report_splits = ["train", "val", "test"]
    report_splits.extend(
        name for name in datasets if name not in {"train", "val", "test", "holdout"}
    )
    for split_name in report_splits:
        config["_active_split_name"] = split_name
        config["_export_latents"] = False
        report[split_name] = evaluate_split(
            model=model,
            dataset=datasets[split_name],
            raw_frame=raw_frames[split_name],
            device=device,
            config=config,
            split_name=split_name,
        )
    if "holdout" in datasets:
        config["_active_split_name"] = "holdout"
        config["_export_latents"] = False
        report["holdout"] = evaluate_split(
            model=model,
            dataset=datasets["holdout"],
            raw_frame=raw_frames["holdout"],
            device=device,
            config=config,
            split_name="holdout",
        )

    checkpoint_path = save_checkpoint(model, standardizer, config, split_meta)
    report_paths = write_report(report, config, split_meta)

    summary = {
        "checkpoint_path": checkpoint_path,
        "report_json": report_paths[0],
        "report_md": report_paths[1],
        "split_meta": split_meta,
        "metrics": report,
        "latent_exports": {
            split: metrics.get("latent_export_path")
            for split, metrics in report.items()
            if isinstance(metrics, dict) and metrics.get("latent_export_path")
        },
        "runtime": {
            **runtime_fingerprint,
            "vram_fraction": vram_fraction if device.type == "cuda" else None,
            "peak_memory_allocated": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
            "peak_memory_reserved": (
                int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
            ),
        },
    }
    _atomic_json_write(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "config_fingerprint": config_fingerprint,
            "runtime_fingerprint": runtime_fingerprint,
            "summary": summary,
        },
        completion_path,
    )
    logger.info("Saved checkpoint to %s", checkpoint_path)
    logger.info("Saved reports to %s and %s", *report_paths)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Resume from model_dir/latest_training_state.pt when present.",
    )
    args = parser.parse_args()
    train(config_path=args.config_path, resume=args.resume)
