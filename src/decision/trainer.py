from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Sampler, TensorDataset

from src.decision.adapter import DecisionAdapter, DecisionAdapterConfig
from src.decision.losses import (
    DecisionLossConfig,
    decision_alignment_loss,
    opportunity_allocation_losses,
    temporal_event_probabilities,
)
from src.memory.experience import latent_columns


@dataclass(frozen=True)
class DecisionTrainingConfig:
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 5
    gradient_clip: float = 1.0
    seed: int = 7
    device: str = "auto"
    neighbors: int = 25
    selection_metric: str = "retrieval_utility"
    selection_interval: int = 2
    minimum_epochs: int = 10
    retrieval_spearman_weight: float = 0.25
    learning_rate_patience: int = 3
    learning_rate_decay: float = 0.50
    minimum_learning_rate: float = 1e-5


class DateGroupedBatchSampler(Sampler[list[int]]):
    """Pack complete same-date cross sections so ranking always has alternatives."""

    def __init__(
        self,
        date_ids: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        values = date_ids.detach().cpu().numpy()
        self.groups = [np.flatnonzero(values == value).tolist() for value in np.unique(values)]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        order = np.arange(len(self.groups))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        batch: list[int] = []
        for group_index in order:
            group = self.groups[int(group_index)]
            if batch and len(batch) + len(group) > self.batch_size:
                yield batch
                batch = []
            if len(group) > self.batch_size:
                for start in range(0, len(group), self.batch_size):
                    if batch:
                        yield batch
                        batch = []
                    yield group[start : start + self.batch_size]
            else:
                batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        return int(np.ceil(sum(len(group) for group in self.groups) / self.batch_size))


OUTCOME_COLUMNS = (
    "decision_return_5",
    "decision_return_10",
    "decision_return_21",
    "decision_return_42",
    "decision_return_63",
    "decision_mae",
    "decision_mfe",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def _encode_ids(frame: pd.DataFrame, column: str) -> torch.Tensor:
    return torch.as_tensor(pd.factorize(frame[column], sort=True)[0], dtype=torch.long)


def _environment_ids(frame: pd.DataFrame) -> torch.Tensor:
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    market_column = next((column for column in ("market", "country", "region") if column in frame), None)
    market = frame[market_column].astype(str) if market_column else pd.Series("single_market", index=frame.index)
    if "decision_event_upside_barrier" in frame and frame["decision_event_upside_barrier"].nunique() >= 3:
        volatility_regime = pd.qcut(
            frame["decision_event_upside_barrier"].rank(method="first"),
            q=3,
            labels=["low_vol", "mid_vol", "high_vol"],
        ).astype(str)
    else:
        volatility_regime = pd.Series("all_vol", index=frame.index)
    environment = market + "|" + timestamp.dt.year.astype(str) + "|" + volatility_regime
    return torch.as_tensor(pd.factorize(environment, sort=True)[0], dtype=torch.long)


def _dataset(frame: pd.DataFrame, temporal_enabled: bool = False) -> TensorDataset:
    cols = latent_columns(frame)
    required = ["decision_utility", *OUTCOME_COLUMNS]
    if temporal_enabled:
        required.append("decision_event_class")
    if missing := [column for column in required if column not in frame]:
        raise ValueError(f"Decision frame lacks training outcomes: {missing}")
    finite = np.isfinite(frame[required + cols].to_numpy(dtype=float)).all(axis=1)
    clean = frame.loc[finite].reset_index(drop=True)
    if clean.empty:
        raise ValueError("No finite mature rows are available for adapter training.")
    return TensorDataset(
        torch.as_tensor(clean[cols].to_numpy(dtype=np.float32)),
        torch.as_tensor(clean["decision_utility"].to_numpy(dtype=np.float32)),
        torch.as_tensor(clean[list(OUTCOME_COLUMNS)].to_numpy(dtype=np.float32)),
        _encode_ids(clean, "timestamp"),
        _encode_ids(clean, "ticker"),
        torch.as_tensor(
            clean.get("decision_event_class", pd.Series(0, index=clean.index)).to_numpy(dtype=np.int64),
            dtype=torch.long,
        ),
        _environment_ids(clean),
    )


def _epoch(
    model: DecisionAdapter,
    loader: DataLoader,
    device: torch.device,
    loss_cfg: DecisionLossConfig,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
) -> dict[str, float]:
    model.train(optimizer is not None)
    totals: dict[str, float] = {"total": 0.0}
    rows = 0
    for latent, utility, outcomes, dates, tickers, temporal_target, environments in loader:
        latent, utility, outcomes = latent.to(device), utility.to(device), outcomes.to(device)
        dates, tickers = dates.to(device), tickers.to(device)
        temporal_target, environments = temporal_target.to(device), environments.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(latent)
            loss, parts = decision_alignment_loss(
                outputs,
                utility,
                outcomes,
                dates,
                tickers,
                model.config.quantiles,
                loss_cfg,
                temporal_target=temporal_target if model.config.temporal_horizons else None,
                environment_ids=environments,
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        count = len(latent)
        totals["total"] += float(loss.detach()) * count
        for key, value in parts.items():
            totals.setdefault(key, 0.0)
            totals[key] += float(value.detach()) * count
        rows += count
    return {key: value / max(rows, 1) for key, value in totals.items()}


def transform_decision_frame(model: DecisionAdapter, frame: pd.DataFrame, device: torch.device, batch_size: int = 2048) -> pd.DataFrame:
    """Append decision embeddings and utility quantiles to a frozen latent table."""
    model.eval()
    latent = torch.as_tensor(frame[latent_columns(frame)].to_numpy(dtype=np.float32))
    decisions: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    event_probabilities: list[np.ndarray] = []
    cumulative_event_risks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(latent), batch_size):
            output = model(latent[start : start + batch_size].to(device))
            decisions.append(output["decision"].cpu().numpy())
            quantiles.append(output["utility_quantiles"].cpu().numpy())
            if "event_logits" in output:
                probability, cumulative = temporal_event_probabilities(
                    output["event_logits"], len(model.config.temporal_horizons)
                )
                event_probabilities.append(probability.cpu().numpy())
                cumulative_event_risks.append(cumulative.cpu().numpy())
    out = frame.copy()
    decision_matrix = np.concatenate(decisions)
    quantile_matrix = np.concatenate(quantiles)
    for i in range(decision_matrix.shape[1]):
        out[f"decision_{i}"] = decision_matrix[:, i]
    for i, quantile in enumerate(model.config.quantiles):
        out[f"pred_utility_q{int(quantile * 100):02d}"] = quantile_matrix[:, i]
    if event_probabilities:
        probability_matrix = np.concatenate(event_probabilities)
        cumulative_matrix = np.concatenate(cumulative_event_risks)
        out["pred_event_class"] = probability_matrix.argmax(axis=1)
        out["pred_event_none_probability"] = probability_matrix[:, 0]
        for index, horizon in enumerate(model.config.temporal_horizons):
            out[f"pred_upside_by_{horizon}"] = cumulative_matrix[:, index, 0]
            out[f"pred_drawdown_by_{horizon}"] = cumulative_matrix[:, index, 1]
    return out


def retrieval_diagnostics(train: pd.DataFrame, query: pd.DataFrame, k: int = 25) -> dict[str, float]:
    """Measure whether decision-space neighbours transfer realized utility."""
    cols = sorted([c for c in train if c.startswith("decision_") and c.removeprefix("decision_").isdigit()], key=lambda c: int(c.split("_")[1]))
    if not cols:
        raise ValueError("No decision_* embedding columns were found.")
    memory = train.dropna(subset=[*cols, "decision_utility"]).copy()
    queries = query.dropna(subset=[*cols, "decision_utility"]).copy()
    memory_available = pd.to_datetime(memory["decision_outcome_available_timestamp"], utc=True, errors="coerce")
    memory_x = memory[cols].to_numpy(dtype=np.float32)
    predictions = np.full(len(queries), np.nan)
    search_count = min(len(memory), max(k * 8, 256))
    search = NearestNeighbors(n_neighbors=search_count, metric="euclidean", n_jobs=-1)
    search.fit(memory_x)
    candidate_distances, candidate_indices = search.kneighbors(
        queries[cols].to_numpy(dtype=np.float32), return_distance=True
    )
    memory_tickers = memory["ticker"].astype(str).to_numpy()
    for index, row in enumerate(queries.itertuples(index=False)):
        legal = (memory_available <= pd.Timestamp(row.timestamp)).to_numpy() & (memory_tickers != str(row.ticker))
        ordered = candidate_indices[index]
        selected = ordered[legal[ordered]][:k]
        if selected.size < k:
            candidates = np.flatnonzero(legal)
            if not candidates.size:
                continue
            exact = np.linalg.norm(memory_x[candidates] - queries.iloc[index][cols].to_numpy(dtype=np.float32), axis=1)
            take_count = min(k, len(candidates))
            selected = candidates[np.argpartition(exact, take_count - 1)[:take_count]]
        if not selected.size:
            continue
        vector = np.asarray([getattr(row, c) for c in cols], dtype=np.float32)
        local_distance = np.linalg.norm(memory_x[selected] - vector, axis=1)
        weights = np.exp(-np.square(local_distance / max(float(np.median(local_distance)), 1e-6)))
        predictions[index] = float(np.average(memory.iloc[selected]["decision_utility"], weights=weights))
    actual = queries["decision_utility"].to_numpy(dtype=float)
    finite = np.isfinite(predictions) & np.isfinite(actual)
    correlation = float(spearmanr(predictions[finite], actual[finite]).statistic) if finite.sum() > 2 else np.nan
    ranked = pd.DataFrame({"timestamp": queries["timestamp"], "prediction": predictions, "actual": actual}).dropna()
    selected = ranked.sort_values(["timestamp", "prediction"], ascending=[True, False]).groupby("timestamp").head(3)
    return {
        "query_rows": int(len(queries)),
        "eligible_rows": int(finite.sum()),
        "neighbor_utility_spearman": correlation,
        "top3_mean_utility": float(selected["actual"].mean()) if not selected.empty else np.nan,
        "top3_positive_fraction": float((selected["actual"] > 0).mean()) if not selected.empty else np.nan,
    }


def temporal_diagnostics(frame: pd.DataFrame, horizons: tuple[int, ...]) -> dict[str, float]:
    """Measure event identity, timing, calibration, and monotonicity."""

    if not horizons or "pred_event_class" not in frame or "decision_event_class" not in frame:
        return {}
    mature = frame["decision_event_class"].notna()
    if "decision_is_mature" in frame:
        mature &= frame["decision_is_mature"].fillna(False)
    evaluation = frame.loc[mature]
    if evaluation.empty:
        return {}
    actual = evaluation["decision_event_class"].to_numpy(dtype=int)
    predicted = evaluation["pred_event_class"].to_numpy(dtype=int)
    diagnostics: dict[str, float] = {
        "event_class_accuracy": float(np.mean(actual == predicted)),
        "event_none_rate": float(np.mean(actual == 0)),
    }
    brier: list[float] = []
    upside_predictions: list[np.ndarray] = []
    downside_predictions: list[np.ndarray] = []
    horizon_count = len(horizons)
    for index, horizon in enumerate(horizons):
        predicted_upside = evaluation[f"pred_upside_by_{horizon}"].to_numpy(dtype=float)
        predicted_downside = evaluation[f"pred_drawdown_by_{horizon}"].to_numpy(dtype=float)
        actual_upside = ((actual >= 1) & (actual <= index + 1)).astype(float)
        actual_downside = (
            (actual >= horizon_count + 1) & (actual <= horizon_count + index + 1)
        ).astype(float)
        brier.extend([(predicted_upside - actual_upside) ** 2, (predicted_downside - actual_downside) ** 2])
        upside_predictions.append(predicted_upside)
        downside_predictions.append(predicted_downside)
    upside_matrix = np.column_stack(upside_predictions)
    downside_matrix = np.column_stack(downside_predictions)
    diagnostics["event_brier"] = float(np.mean(np.concatenate(brier)))
    diagnostics["temporal_monotonicity_violations"] = float(
        np.mean(np.diff(upside_matrix, axis=1) < -1e-7)
        + np.mean(np.diff(downside_matrix, axis=1) < -1e-7)
    )
    actual_event = actual != 0
    if actual_event.any():
        actual_bucket = np.where(actual <= horizon_count, actual - 1, actual - 1 - horizon_count)
        predicted_bucket = np.where(
            predicted == 0,
            horizon_count,
            np.where(predicted <= horizon_count, predicted - 1, predicted - 1 - horizon_count),
        )
        diagnostics["event_bucket_mae"] = float(
            np.mean(np.abs(actual_bucket[actual_event] - predicted_bucket[actual_event]))
        )
    return diagnostics


def opportunity_diagnostics(
    model: DecisionAdapter,
    transformed: pd.DataFrame,
    loss_config: DecisionLossConfig,
) -> dict[str, float]:
    """Evaluate missed-opportunity regret under the learned cash threshold."""

    if model.cash_logit is None:
        return {}
    median_column = f"pred_utility_q{int(model.config.quantiles[1] * 100):02d}"
    scores = torch.as_tensor(transformed[median_column].to_numpy(dtype=np.float32))
    utility = torch.as_tensor(transformed["decision_utility"].to_numpy(dtype=np.float32))
    dates = _encode_ids(transformed, "timestamp")
    allocation, coverage, regret = opportunity_allocation_losses(
        scores,
        utility,
        dates,
        model.cash_logit.detach().cpu(),
        top_k=loss_config.opportunity_top_k,
        temperature=loss_config.opportunity_temperature,
        minimum_positive_utility=loss_config.minimum_positive_utility,
    )
    return {
        "opportunity_allocation_loss": float(allocation),
        "opportunity_coverage_error": float(coverage),
        "opportunity_regret": float(regret),
    }


def train_decision_adapter(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: str | Path,
    adapter_config: DecisionAdapterConfig | None = None,
    training_config: DecisionTrainingConfig | None = None,
    loss_config: DecisionLossConfig | None = None,
) -> dict[str, Any]:
    """Fit an adapter with early stopping and save auditable transformed tables."""
    train_cfg = training_config or DecisionTrainingConfig()
    loss_cfg = loss_config or DecisionLossConfig()
    _set_seed(train_cfg.seed)
    device = _device(train_cfg.device)
    model = DecisionAdapter(adapter_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    temporal_enabled = bool(model.config.temporal_horizons)
    train_data = _dataset(train_frame, temporal_enabled)
    val_data = _dataset(validation_frame, temporal_enabled)
    if temporal_enabled:
        maximum_class = 2 * len(model.config.temporal_horizons)
        if int(train_data.tensors[5].max()) > maximum_class or int(val_data.tensors[5].max()) > maximum_class:
            raise ValueError("Temporal event targets are incompatible with the adapter horizon count.")
    train_loader = DataLoader(
        train_data,
        batch_sampler=DateGroupedBatchSampler(
            train_data.tensors[3],
            train_cfg.batch_size,
            loss_cfg.temporal_coherence_weight <= 0,
            train_cfg.seed,
        ),
    )
    val_loader = DataLoader(
        val_data,
        batch_sampler=DateGroupedBatchSampler(
            val_data.tensors[3], train_cfg.batch_size, False, train_cfg.seed
        ),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_cfg.learning_rate_decay,
        patience=train_cfg.learning_rate_patience,
        min_lr=train_cfg.minimum_learning_rate,
    )
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_selection_score = float("-inf")
    stale = 0
    for epoch in range(train_cfg.epochs):
        train_metrics = _epoch(model, train_loader, device, loss_cfg, optimizer, train_cfg.gradient_clip)
        val_metrics = _epoch(model, val_loader, device, loss_cfg, None, train_cfg.gradient_clip)
        scheduler.step(val_metrics["total"])
        record: dict[str, Any] = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "validation": val_metrics,
        }
        evaluate_retrieval = (
            epoch + 1 >= train_cfg.minimum_epochs
            and ((epoch + 1 - train_cfg.minimum_epochs) % train_cfg.selection_interval == 0 or epoch + 1 == train_cfg.epochs)
        )
        if train_cfg.selection_metric == "retrieval_utility" and evaluate_retrieval:
            epoch_train = transform_decision_frame(model, train_frame, device)
            epoch_val = transform_decision_frame(model, validation_frame, device)
            epoch_diagnostics = retrieval_diagnostics(epoch_train, epoch_val, train_cfg.neighbors)
            correlation = float(epoch_diagnostics["neighbor_utility_spearman"])
            if not np.isfinite(correlation):
                correlation = -1.0
            selection_score = float(epoch_diagnostics["top3_mean_utility"]) + train_cfg.retrieval_spearman_weight * correlation
            record["retrieval"] = epoch_diagnostics
            record["selection_score"] = selection_score
            if selection_score > best_selection_score:
                best_selection_score, best_loss, stale = selection_score, val_metrics["total"], 0
                best_state = deepcopy(model.state_dict())
            else:
                stale += 1
        elif train_cfg.selection_metric != "retrieval_utility":
            if val_metrics["total"] < best_loss:
                best_loss, stale = val_metrics["total"], 0
                best_state = deepcopy(model.state_dict())
            else:
                stale += 1
        history.append(record)
        if best_state is not None and stale >= train_cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("Adapter training produced no checkpoint.")
    model.load_state_dict(best_state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_transformed = transform_decision_frame(model, train_frame, device)
    val_transformed = transform_decision_frame(model, validation_frame, device)
    train_transformed.to_parquet(output / "train_decisions.parquet", index=False)
    val_transformed.to_parquet(output / "val_decisions.parquet", index=False)
    diagnostics = retrieval_diagnostics(train_transformed, val_transformed, train_cfg.neighbors)
    temporal_metrics = temporal_diagnostics(val_transformed, model.config.temporal_horizons)
    opportunity_metrics = opportunity_diagnostics(model, val_transformed, loss_cfg)
    def raw_retrieval_frame(frame: pd.DataFrame) -> pd.DataFrame:
        raw = frame.copy()
        return raw.rename(columns={column: f"decision_{index}" for index, column in enumerate(latent_columns(raw))})
    baseline_diagnostics = retrieval_diagnostics(
        raw_retrieval_frame(train_frame), raw_retrieval_frame(validation_frame), train_cfg.neighbors
    )
    torch.save(model.checkpoint(), output / "decision_adapter.pt")
    summary = {
        "adapter_config": asdict(model.config),
        "training_config": asdict(train_cfg),
        "loss_config": asdict(loss_cfg),
        "best_validation_loss": best_loss,
        "best_retrieval_selection_score": (
            best_selection_score if np.isfinite(best_selection_score) else None
        ),
        "epochs_completed": len(history),
        "diagnostics": diagnostics,
        "temporal_diagnostics": temporal_metrics,
        "opportunity_diagnostics": opportunity_metrics,
        "raw_latent_diagnostics": baseline_diagnostics,
        "device": str(device),
    }
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
