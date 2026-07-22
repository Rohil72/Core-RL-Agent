"""Evaluate one learned allocation policy through the unchanged A2 hold/exit backtester."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.market_memory_backtester import PolicyConfig, compute_backtest_metrics, run_long_only_backtest  # noqa: E402
from src.policy.offline_policy import ContextualBandit, LearnedExposureController, bandit_action  # noqa: E402


def _load_action_function(algorithm: str, model_path: Path, exposures: tuple[float, ...]):
    if algorithm == "bandit":
        payload = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        model = ContextualBandit(int(payload["observation_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        return lambda observation: bandit_action(model, observation, exposures)
    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError("Evaluate d3rlpy policies from the isolated .venv-phase5-rl environment.") from exc
    learner = d3rlpy.load_learnable(str(model_path), device="cuda:0" if torch.cuda.is_available() else "cpu")
    return lambda observation: float(np.clip(learner.predict(observation[None, :])[0, 0], 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=("bandit", "cql", "iql", "td3bc"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--signals", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase4e-config", default="configs/phase4e_cross_market_sharpe.yaml")
    args = parser.parse_args()
    data = dict(np.load(args.dataset, allow_pickle=False))
    feature_names = data["feature_names"].astype(str).tolist()
    exposures = (0.0, 0.25, 0.50, 0.75, 1.0)
    action_function = _load_action_function(args.algorithm, Path(args.model), exposures)
    decisions = pd.read_parquet(args.decisions)
    signals = pd.read_parquet(args.signals)
    decision_cols = [c for c in decisions if c.startswith("decision_") or c.startswith("pred_utility_")]
    missing_decision_cols = [column for column in decision_cols if column not in signals]
    if missing_decision_cols:
        signals = signals.merge(
            decisions[["ticker", "timestamp", *missing_decision_cols]],
            on=["ticker", "timestamp"],
            how="left",
        )
    phase4e = yaml.safe_load(Path(args.phase4e_config).read_text(encoding="utf-8"))
    policy = PolicyConfig(**phase4e["policy"])
    controller = LearnedExposureController(action_function, feature_names, top_k=policy.top_k)
    exposure_log: list[dict] = []
    trades, equity = run_long_only_backtest(
        signals, policy, score_col="opportunity_score", exit_score_col="opportunity_score",
        exposure_controller=controller, exposure_log=exposure_log,
    )
    metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
    exposure_frame = pd.DataFrame(exposure_log)
    if "target_exposure" in exposure_frame:
        target_exposure = pd.to_numeric(exposure_frame["target_exposure"], errors="coerce").dropna()
        metrics["target_exposure_mean"] = float(target_exposure.mean()) if len(target_exposure) else None
        metrics["target_exposure_std"] = float(target_exposure.std(ddof=0)) if len(target_exposure) else None
        metrics["target_exposure_unique"] = int(target_exposure.round(6).nunique())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    trades.to_csv(output / "trades.csv", index=False)
    equity.to_csv(output / "equity_curve.csv", index=False)
    exposure_frame.to_csv(output / "exposure_decisions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
