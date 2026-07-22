from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.backtest.exposure_controller import ExposureDecision


@dataclass(frozen=True)
class OfflinePolicyDatasetConfig:
    """Fixed observation/action/reward contract shared by all policy learners."""

    top_k: int = 3
    exposures: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0)
    behavior_policies: tuple[str, ...] = (
        "cash",
        "full",
        "score_scaled",
        "volatility_target",
        "exploratory",
    )
    transaction_cost_bps: float = 10.0
    score_column: str = "pred_utility_q50"
    reward_column: str = "decision_return_1"
    decision_prefix: str = "decision_"
    target_daily_volatility: float = 0.01
    volatility_lookback: int = 21
    exploratory_probability: float = 0.20
    seed: int = 7


EVIDENCE_COLUMNS = (
    "pred_utility_q10", "pred_utility_q50", "pred_utility_q90",
    "retrieval_expected_alpha", "retrieval_expected_downside",
    "retrieval_confidence", "retrieval_agreement_score",
    "retrieval_disagreement_score", "retrieval_effective_sample_size",
    "retrieval_median_distance", "opportunity_score",
)


def policy_feature_columns(frame: pd.DataFrame, prefix: str = "decision_") -> list[str]:
    decision = sorted(
        [c for c in frame if c.startswith(prefix) and c.removeprefix(prefix).isdigit()],
        key=lambda c: int(c.removeprefix(prefix)),
    )
    return decision + [column for column in EVIDENCE_COLUMNS if column in frame]


def build_daily_context(frame: pd.DataFrame, config: OfflinePolicyDatasetConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Convert security rows into fixed-width top-k decision observations."""
    cfg = config or OfflinePolicyDatasetConfig()
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    features = policy_feature_columns(data, cfg.decision_prefix)
    if not features:
        raise ValueError("Policy observations require decision embeddings or evidence features.")
    for column in features:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    score = cfg.score_column if cfg.score_column in data else ("opportunity_score" if "opportunity_score" in data else features[0])
    rows: list[dict[str, Any]] = []
    output_features = [f"slot_{slot}_{column}" for slot in range(cfg.top_k) for column in features]
    output_features += ["market_score_mean", "market_score_std", "market_positive_breadth"]
    for timestamp, group in data.groupby("timestamp", sort=True):
        selected = group.sort_values(score, ascending=False).head(cfg.top_k)
        record: dict[str, Any] = {"timestamp": timestamp}
        values = np.zeros((cfg.top_k, len(features)), dtype=np.float32)
        selected_values = selected[features].to_numpy(dtype=np.float32)
        values[: len(selected_values)] = selected_values
        for slot in range(cfg.top_k):
            for feature_index, column in enumerate(features):
                record[f"slot_{slot}_{column}"] = float(values[slot, feature_index])
        scores = group[score].to_numpy(dtype=float)
        record["market_score_mean"] = float(np.nanmean(scores))
        record["market_score_std"] = float(np.nanstd(scores))
        record["market_positive_breadth"] = float(np.nanmean(scores > 0.0))
        reward_source = selected[cfg.reward_column] if cfg.reward_column in selected else selected.get("decision_return_1")
        market_source = group[cfg.reward_column] if cfg.reward_column in group else group.get("decision_return_1")
        basket_return = float(pd.to_numeric(reward_source, errors="coerce").mean()) if reward_source is not None else 0.0
        market_return = float(pd.to_numeric(market_source, errors="coerce").median()) if market_source is not None else 0.0
        record["basket_reward"] = basket_return - market_return
        record["selected_tickers"] = ",".join(selected["ticker"].astype(str))
        rows.append(record)
    return pd.DataFrame(rows), output_features


def build_offline_policy_dataset(frame: pd.DataFrame, config: OfflinePolicyDatasetConfig | None = None) -> dict[str, np.ndarray]:
    """Create chronological changing-exposure trajectories for offline policy comparison."""
    cfg = config or OfflinePolicyDatasetConfig()
    daily, features = build_daily_context(frame, cfg)
    base = daily[features].to_numpy(dtype=np.float32)
    rewards = daily["basket_reward"].fillna(0.0).to_numpy(dtype=np.float32)
    observations, actions, all_rewards, terminals, episode_ids, timestamps = [], [], [], [], [], []
    cost = cfg.transaction_cost_bps / 10_000.0
    rng = np.random.default_rng(cfg.seed)
    rolling_volatility = (
        pd.Series(rewards)
        .rolling(cfg.volatility_lookback, min_periods=5)
        .std(ddof=1)
        .fillna(cfg.target_daily_volatility)
        .clip(lower=1e-4)
        .to_numpy(dtype=float)
    )

    def behavior_action(name: str, index: int) -> float:
        if name == "cash":
            return 0.0
        if name == "full":
            return 1.0
        score_mean = float(daily.iloc[index]["market_score_mean"])
        breadth = float(daily.iloc[index]["market_positive_breadth"])
        score_scaled = float(np.clip(0.5 * breadth + 0.5 * (score_mean > 0.0), 0.0, 1.0))
        if name == "score_scaled":
            return min(cfg.exposures, key=lambda value: abs(value - score_scaled))
        if name == "volatility_target":
            target = score_scaled * min(1.0, cfg.target_daily_volatility / rolling_volatility[index])
            return min(cfg.exposures, key=lambda value: abs(value - target))
        if name == "exploratory":
            if rng.random() < cfg.exploratory_probability:
                return float(rng.choice(cfg.exposures))
            return float(cfg.exposures[index % len(cfg.exposures)])
        raise ValueError(f"Unknown behavior policy: {name}")

    for episode_id, behavior in enumerate(cfg.behavior_policies):
        previous = 0.0
        equity = 1.0
        peak = 1.0
        for index, context in enumerate(base):
            drawdown = equity / peak - 1.0
            observation = np.concatenate(
                [context, np.asarray([previous, drawdown], dtype=np.float32)]
            )
            exposure = behavior_action(behavior, index)
            reward = exposure * rewards[index] - cost * abs(exposure - previous)
            observations.append(observation)
            actions.append([exposure])
            all_rewards.append(reward)
            terminals.append(index == len(base) - 1)
            episode_ids.append(episode_id)
            timestamps.append(str(daily.iloc[index]["timestamp"]))
            equity *= max(1.0 + reward, 1e-6)
            peak = max(peak, equity)
            previous = exposure
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(all_rewards, dtype=np.float32),
        "terminals": np.asarray(terminals, dtype=np.float32),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "feature_names": np.asarray([*features, "previous_exposure", "current_drawdown"], dtype=str),
        "timestamps": np.asarray(timestamps, dtype=str),
        "behavior_names": np.asarray(cfg.behavior_policies, dtype=str),
    }


class ContextualBandit(nn.Module):
    """Small reward model used as the transparent policy baseline."""

    def __init__(self, observation_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([observation, action], dim=-1)).squeeze(-1)


def train_contextual_bandit(
    dataset: dict[str, np.ndarray], seed: int = 7, steps: int = 10_000,
    batch_size: int = 256, learning_rate: float = 3e-4, device: str = "auto",
) -> ContextualBandit:
    """Fit the baseline on the exact same offline tuples used by CQL/IQL/TD3+BC."""
    torch.manual_seed(seed)
    target_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    observations = torch.as_tensor(dataset["observations"], device=target_device)
    actions = torch.as_tensor(dataset["actions"], device=target_device)
    rewards = torch.as_tensor(dataset["rewards"], device=target_device)
    model = ContextualBandit(observations.size(1)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    for _ in range(steps):
        indices = torch.randint(len(observations), (min(batch_size, len(observations)),), generator=generator, device=target_device)
        loss = torch.nn.functional.smooth_l1_loss(model(observations[indices], actions[indices]), rewards[indices])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model


def bandit_action(model: ContextualBandit, observation: np.ndarray, exposures: tuple[float, ...]) -> float:
    """Select the fixed-grid exposure with maximum predicted net reward."""
    device = next(model.parameters()).device
    obs = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).repeat(len(exposures), 1)
    actions = torch.as_tensor(exposures, dtype=torch.float32, device=device).unsqueeze(1)
    with torch.no_grad():
        scores = model(obs, actions)
    return float(exposures[int(scores.argmax())])


class LearnedExposureController:
    """Duck-typed controller that lets learned allocation reuse the A2 backtester."""

    def __init__(
        self,
        action_function: Callable[[np.ndarray], float],
        feature_names: list[str],
        top_k: int = 3,
        rebalance_threshold: float = 0.05,
    ) -> None:
        self.action_function = action_function
        self.feature_names = feature_names
        self.top_k = top_k
        self.previous_exposure = 0.0
        self.config = type("ControllerConfig", (), {"rebalance_threshold": rebalance_threshold})()

    def decide(self, signal_rows: pd.DataFrame, equity: pd.DataFrame) -> ExposureDecision:
        frame = signal_rows.reset_index().rename(columns={signal_rows.index.name or "index": "ticker"})
        daily, features = build_daily_context(frame, OfflinePolicyDatasetConfig(top_k=self.top_k))
        portfolio_features = {"previous_exposure", "current_drawdown"}
        expected = [name for name in self.feature_names if name not in portfolio_features]
        values = (
            daily.reindex(columns=expected, fill_value=0.0).iloc[0].to_numpy(dtype=np.float32)
            if not daily.empty else np.zeros(len(expected), dtype=np.float32)
        )
        current_drawdown = 0.0
        if not equity.empty and "equity" in equity:
            equity_values = pd.to_numeric(equity["equity"], errors="coerce").dropna()
            if not equity_values.empty:
                current_drawdown = float(equity_values.iloc[-1] / equity_values.cummax().iloc[-1] - 1.0)
        observation = np.concatenate(
            [values, np.asarray([self.previous_exposure, current_drawdown], dtype=np.float32)]
        )
        target = float(np.clip(self.action_function(observation), 0.0, 1.0))
        self.previous_exposure = target
        return ExposureDecision(target, target, 1.0, 1.0, None, 0.0, None, None, None, None, "learned_policy")


def save_offline_dataset(path: str | Path, dataset: dict[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **dataset)
