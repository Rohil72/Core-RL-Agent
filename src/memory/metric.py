from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.memory.experience import latent_columns, resolve_column


@dataclass(frozen=True)
class MetricLearningConfig:
    """Bounded training budget for a frozen-latent retrieval metric."""

    kind: str = "diagonal"
    output_dim: int = 32
    fit_end_year: int = 2021
    validation_year: int = 2022
    steps: int = 2000
    batch_size: int = 256
    learning_rate: float = 1e-3
    validation_interval: int = 100
    patience: int = 10
    gradient_clip: float = 1.0
    rank_weight: float = 0.10
    preserve_weight: float = 0.10
    weight_decay: float = 1e-4
    temperature: float = 1.0
    seed: int = 7
    device: str = "auto"
    alpha_target: str = "future_universe_alpha_63"
    downside_target: str = "future_min_return_63"
    path_target: str = "event_upside_before_drawdown_126"


class RetrievalMetric(Protocol):
    """Transforms frozen latents into the space used for nearest-neighbour search."""

    kind: str

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Transform a latent matrix without changing fitted state."""

    def save(self, path: str | Path) -> Path:
        """Persist the metric and its normalization statistics."""


@dataclass
class FittedRetrievalMetric:
    """Serializable identity, diagonal, or low-rank retrieval metric."""

    kind: str
    mean: np.ndarray
    scale: np.ndarray
    parameter: np.ndarray | None = None
    config: dict | None = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply fitted normalization and metric projection."""
        x = np.asarray(values, dtype=np.float32)
        z = (x - self.mean) / self.scale
        if self.kind == "diagonal":
            if self.parameter is None:
                raise ValueError("Diagonal metric is missing weights.")
            z = z * np.sqrt(np.maximum(self.parameter, 1e-8))
        elif self.kind == "low_rank":
            if self.parameter is None:
                raise ValueError("Low-rank metric is missing its projection.")
            z = z @ self.parameter
            norm = np.linalg.norm(z, axis=1, keepdims=True)
            z = z / np.maximum(norm, 1e-4)
        elif self.kind != "identity":
            raise ValueError(f"Unsupported retrieval metric: {self.kind}")
        if not np.isfinite(z).all():
            raise FloatingPointError("Retrieval metric produced non-finite values.")
        return z.astype(np.float32, copy=False)

    def save(self, path: str | Path) -> Path:
        """Persist the metric using NumPy's non-executable archive format."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            kind=np.asarray(self.kind),
            mean=self.mean,
            scale=self.scale,
            parameter=np.asarray([]) if self.parameter is None else self.parameter,
            config_json=np.asarray(json.dumps(self.config or {}, sort_keys=True)),
        )
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FittedRetrievalMetric":
        """Load a metric saved by :meth:`save`."""
        with np.load(path, allow_pickle=False) as data:
            parameter = data["parameter"]
            raw_config = json.loads(str(data["config_json"].item())) if "config_json" in data else {}
            return cls(
                kind=str(data["kind"].item()),
                mean=data["mean"].astype(np.float32),
                scale=data["scale"].astype(np.float32),
                parameter=None if parameter.size == 0 else parameter.astype(np.float32),
                config=dict(raw_config),
            )


def fit_retrieval_metric(
    frame: pd.DataFrame,
    config: MetricLearningConfig | None = None,
) -> tuple[FittedRetrievalMetric, dict[str, float | int | str]]:
    """Fit a small outcome-aware metric while leaving encoder weights untouched."""
    cfg = config or MetricLearningConfig()
    if cfg.kind not in {"identity", "diagonal", "low_rank"}:
        raise ValueError("kind must be identity, diagonal, or low_rank.")
    cols = latent_columns(frame)
    target_cols = [
        resolve_column(frame, cfg.alpha_target),
        resolve_column(frame, cfg.downside_target),
        resolve_column(frame, cfg.path_target),
    ]
    if any(col is None for col in target_cols):
        raise ValueError("Metric learning frame is missing one or more outcome targets.")

    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    work["ticker"] = work["ticker"].astype(str)
    years = work["timestamp"].dt.year
    fit_mask = years <= cfg.fit_end_year
    valid_mask = years == cfg.validation_year
    finite = np.isfinite(work[cols].to_numpy(dtype=float)).all(axis=1)
    finite &= np.isfinite(work[target_cols].to_numpy(dtype=float)).all(axis=1)
    fit = work.loc[fit_mask & finite].reset_index(drop=True)
    validation = work.loc[valid_mask & finite].reset_index(drop=True)
    if len(fit) < 32:
        raise ValueError("Metric learning requires at least 32 causal fitting rows.")

    fit_x = fit[cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(fit_x, axis=0).astype(np.float32)
    scale = np.nanstd(fit_x, axis=0).astype(np.float32)
    scale[(scale <= 1e-6) | ~np.isfinite(scale)] = 1.0
    if cfg.kind == "identity":
        metric = FittedRetrievalMetric("identity", mean, scale, config=asdict(cfg))
        score = _validation_kernel_error(metric, fit, validation, cols, target_cols)
        return metric, {"kind": cfg.kind, "fit_rows": len(fit), "validation_rows": len(validation), "validation_loss": score}

    device = _resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x = torch.as_tensor((fit_x - mean) / scale, device=device)
    y_raw = fit[target_cols].to_numpy(dtype=np.float32)
    y_median = np.nanmedian(y_raw, axis=0).astype(np.float32)
    y_scale = np.nanmedian(np.abs(y_raw - y_median), axis=0).astype(np.float32)
    y_scale[(y_scale <= 1e-6) | ~np.isfinite(y_scale)] = 1.0
    y = torch.as_tensor((y_raw - y_median) / y_scale, device=device)
    timestamps = torch.as_tensor(fit["timestamp"].astype("int64").to_numpy(), device=device)
    ticker_codes = torch.as_tensor(pd.factorize(fit["ticker"], sort=True)[0], device=device)
    available = _available_timestamp_tensor(fit, device)

    if cfg.kind == "diagonal":
        raw_parameter = torch.nn.Parameter(torch.zeros(x.shape[1], device=device))
    else:
        projection = _pca_initial_projection(x, cfg.output_dim)
        raw_parameter = torch.nn.Parameter(projection)
    optimizer = torch.optim.Adam([raw_parameter], lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    best_parameter = raw_parameter.detach().clone()
    best_loss = float("inf")
    stale = 0
    completed_steps = 0

    for step in range(1, cfg.steps + 1):
        batch_size = min(cfg.batch_size, len(fit))
        indices = torch.randint(len(fit), (batch_size,), generator=generator, device=device)
        batch_x = x[indices]
        batch_y = y[indices]
        if cfg.kind == "diagonal":
            weights = F.softplus(raw_parameter) + 1e-4
            weights = weights / weights.mean()
            student = batch_x * torch.sqrt(weights)
        else:
            student = F.normalize(batch_x @ raw_parameter, p=2, dim=1, eps=1e-4)

        student_dist = torch.cdist(student.float(), student.float())
        teacher_dist = torch.cdist(batch_x.float(), batch_x.float())
        outcome_dist = _outcome_distance(batch_y)
        legal = available[indices][None, :] <= timestamps[indices][:, None]
        legal &= ticker_codes[indices][None, :] != ticker_codes[indices][:, None]
        legal.fill_diagonal_(False)
        eligible = legal.any(dim=1)
        if not bool(eligible.any()):
            continue

        student_prob = _masked_softmax(-student_dist / cfg.temperature, legal)
        teacher_prob = _masked_softmax(-teacher_dist / cfg.temperature, legal)
        target_prob = _masked_softmax(-outcome_dist / cfg.temperature, legal)
        predicted = student_prob @ batch_y
        kernel_loss = F.smooth_l1_loss(predicted[eligible], batch_y[eligible])
        preserve = _masked_kl(teacher_prob[eligible], student_prob[eligible])
        rank = _masked_kl(target_prob[eligible], student_prob[eligible])
        loss = kernel_loss + cfg.preserve_weight * preserve
        if cfg.kind == "low_rank":
            loss = loss + cfg.rank_weight * rank
        if not torch.isfinite(loss):
            raise FloatingPointError("Retrieval metric loss became non-finite.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_parameter], cfg.gradient_clip)
        optimizer.step()
        completed_steps = step

        if step % cfg.validation_interval == 0 or step == cfg.steps:
            parameter = _materialize_parameter(cfg.kind, raw_parameter)
            candidate = FittedRetrievalMetric(cfg.kind, mean, scale, parameter, asdict(cfg))
            val_loss = _validation_kernel_error(candidate, fit, validation, cols, target_cols)
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_parameter = raw_parameter.detach().clone()
                stale = 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break

    parameter = _materialize_parameter(cfg.kind, best_parameter)
    metric = FittedRetrievalMetric(cfg.kind, mean, scale, parameter, asdict(cfg))
    return metric, {
        "kind": cfg.kind,
        "fit_rows": int(len(fit)),
        "validation_rows": int(len(validation)),
        "steps": int(completed_steps),
        "validation_loss": float(best_loss),
        "device": str(device),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA retrieval metric fitting was requested but is unavailable.")
    return device


def _available_timestamp_tensor(frame: pd.DataFrame, device: torch.device) -> torch.Tensor:
    if "outcome_available_timestamp" not in frame:
        return torch.as_tensor(frame["timestamp"].astype("int64").to_numpy(), device=device)
    available = pd.to_datetime(frame["outcome_available_timestamp"], utc=True, errors="coerce")
    values = available.astype("int64").to_numpy()
    values[available.isna().to_numpy()] = np.iinfo(np.int64).max
    return torch.as_tensor(values, device=device)


def _pca_initial_projection(x: torch.Tensor, output_dim: int) -> torch.Tensor:
    rank = max(1, min(output_dim, x.shape[1], x.shape[0] - 1))
    _, _, vectors = torch.pca_lowrank(x[: min(len(x), 4096)].float(), q=rank, center=False)
    return vectors[:, :rank].to(x.device)


def _materialize_parameter(kind: str, raw: torch.Tensor) -> np.ndarray:
    if kind == "diagonal":
        weights = F.softplus(raw.float()) + 1e-4
        weights = weights / weights.mean()
        return weights.detach().cpu().numpy().astype(np.float32)
    return raw.detach().cpu().numpy().astype(np.float32)


def _outcome_distance(y: torch.Tensor) -> torch.Tensor:
    delta = torch.abs(y[:, None, :] - y[None, :, :])
    weights = torch.as_tensor([1.0, 0.5, 0.25], device=y.device)
    return torch.sum(delta * weights, dim=-1)


def _masked_softmax(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(~legal, -torch.inf)
    probabilities = torch.softmax(masked, dim=1)
    return torch.where(torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities))


def _masked_kl(target: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    target = target.clamp_min(1e-8)
    predicted = predicted.clamp_min(1e-8)
    return torch.sum(target * (target.log() - predicted.log()), dim=1).mean()


def _validation_kernel_error(
    metric: FittedRetrievalMetric,
    memory: pd.DataFrame,
    queries: pd.DataFrame,
    latent_cols: list[str],
    target_cols: list[str],
) -> float:
    if queries.empty:
        return float("inf")
    memory_x = metric.transform(memory[latent_cols].to_numpy(dtype=np.float32))
    query_x = metric.transform(queries[latent_cols].to_numpy(dtype=np.float32))
    memory_y = memory[target_cols].to_numpy(dtype=np.float32)
    query_y = queries[target_cols].to_numpy(dtype=np.float32)
    errors: list[float] = []
    memory_tickers = memory["ticker"].astype(str).to_numpy()
    available = pd.to_datetime(
        memory.get("outcome_available_timestamp", memory["timestamp"]),
        utc=True,
        errors="coerce",
    )
    available_ns = available.astype("int64").to_numpy()
    available_ns[available.isna().to_numpy()] = np.iinfo(np.int64).max
    for start in range(0, len(queries), 256):
        stop = min(start + 256, len(queries))
        batch = query_x[start:stop]
        squared = (
            np.sum(batch * batch, axis=1, keepdims=True)
            + np.sum(memory_x * memory_x, axis=1)[None, :]
            - 2.0 * (batch @ memory_x.T)
        )
        distances = np.sqrt(np.maximum(squared, 0.0), dtype=np.float32)
        for local, row in enumerate(queries.iloc[start:stop].itertuples(index=False)):
            query_ns = pd.Timestamp(getattr(row, "timestamp")).value
            legal = available_ns <= query_ns
            legal &= memory_tickers != str(getattr(row, "ticker"))
            idx = np.flatnonzero(legal)
            if idx.size == 0:
                continue
            idx = idx[np.argsort(distances[local, idx])[:25]]
            bandwidth = max(float(np.median(distances[local, idx])), 1e-6)
            weights = np.exp(-0.5 * (distances[local, idx] / bandwidth) ** 2)
            weights /= max(float(weights.sum()), 1e-12)
            prediction = np.average(memory_y[idx], axis=0, weights=weights)
            errors.append(float(np.mean(np.abs(prediction - query_y[start + local]))))
    return float(np.mean(errors)) if errors else float("inf")
