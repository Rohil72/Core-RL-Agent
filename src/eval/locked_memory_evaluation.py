from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.consensus_policy import (
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import PolicyConfig
from src.eval.memory_reliability import (
    ReliabilityConfig,
    build_consensus_frame,
    fit_reliability_model,
    reliability_metrics,
)
from src.eval.policy_baselines import (
    BaselineSuiteConfig,
    equal_weight_buy_hold,
    evaluate_standard_baselines,
    run_scored_policy,
    save_policy_result,
    write_json,
)

# Retained for callers that imported the old private helper.
_equal_weight_buy_hold = equal_weight_buy_hold


@dataclass(frozen=True)
class LockedBaselineConfig:
    """Predeclared baseline suite for a locked market-memory candidate."""

    nominal_coverage: float = 0.25
    momentum_window: int = 21
    random_trials: int = 20
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.nominal_coverage not in {0.25, 0.50, 0.75, 1.0}:
            raise ValueError("nominal_coverage must match a fitted reliability threshold.")
        if self.momentum_window <= 0:
            raise ValueError("momentum_window must be positive.")
        if self.random_trials <= 0:
            raise ValueError("random_trials must be positive.")


def evaluate_locked_market(
    *,
    development_signals: dict[int, pd.DataFrame],
    development_neighbors: dict[int, pd.DataFrame],
    query_signals: dict[int, pd.DataFrame],
    query_neighbors: dict[int, pd.DataFrame],
    policy: PolicyConfig,
    consensus_config: ConsensusSignalConfig,
    reliability_config: ReliabilityConfig,
    useful_alpha_after_costs: float,
    baseline_config: LockedBaselineConfig,
    output: Path,
) -> dict[str, Any]:
    """Evaluate one frozen market with reliability and predeclared baselines."""
    development = build_consensus_frame(
        development_signals,
        development_neighbors,
        useful_alpha_after_costs=useful_alpha_after_costs,
    )
    query = build_consensus_frame(
        query_signals,
        query_neighbors,
        useful_alpha_after_costs=useful_alpha_after_costs,
    )
    model = fit_reliability_model(development, reliability_config)
    predictions = model.predict(query)
    threshold = float(model.calibration_thresholds[baseline_config.nominal_coverage])

    consensus = build_consensus_signals(query_signals, policy, consensus_config)
    reliability_columns = predictions[
        [
            "ticker",
            "timestamp",
            "reliability_probability",
            "predicted_absolute_error",
            "state_shift_score",
        ]
    ]
    consensus = consensus.merge(
        reliability_columns,
        on=["ticker", "timestamp"],
        how="left",
        validate="one_to_one",
    )
    consensus["reliability_pass"] = consensus["reliability_probability"] >= threshold

    locked = run_scored_policy(
        consensus,
        policy,
        "consensus_entry_rank",
        exit_score_col="consensus_exit_score",
        row_filter=lambda row, cfg, _score: (
            bool(row.get("reliability_pass", False))
            and consensus_row_filter(row, cfg, consensus_config.minimum_votes)
        ),
    )
    raw_memory = run_scored_policy(
        consensus,
        policy,
        "consensus_entry_rank",
        exit_score_col="consensus_exit_score",
        row_filter=lambda row, cfg, _score: consensus_row_filter(
            row,
            cfg,
            consensus_config.minimum_votes,
        ),
    )

    standard, random_table, random_summary = evaluate_standard_baselines(
        consensus,
        policy,
        BaselineSuiteConfig(
            momentum_window=baseline_config.momentum_window,
            random_trials=baseline_config.random_trials,
            random_seed=baseline_config.random_seed,
        ),
        reference_sharpe=float(locked["metrics"]["sharpe"]),
    )

    output.mkdir(parents=True, exist_ok=True)
    consensus.to_parquet(output / "consensus_signals.parquet", index=False)
    predictions.to_parquet(output / "confirmation_reliability.parquet", index=False)
    write_json(output / "reliability_model_audit.json", model.to_dict())
    reliability_audit = reliability_metrics(predictions, model)
    write_json(output / "reliability_metrics.json", reliability_audit)

    results = {
        "locked_memory": locked,
        "raw_memory": raw_memory,
        **standard,
    }
    for name, result in results.items():
        save_policy_result(output, name, result)
    random_table.to_csv(output / "baselines" / "random_trials.csv", index=False)
    random_summary["probability_beating_locked_sharpe"] = random_summary.pop(
        "probability_beating_reference_sharpe"
    )
    write_json(output / "baselines" / "random_summary.json", random_summary)

    payload = {
        "status": "completed",
        "nominal_coverage": baseline_config.nominal_coverage,
        "reliability_threshold": threshold,
        "realized_coverage": float(consensus["reliability_pass"].mean()),
        "useful_alpha_after_costs": useful_alpha_after_costs,
        "policy": asdict(policy),
        "consensus": asdict(consensus_config),
        "reliability": reliability_audit,
        "baselines": {name: result["metrics"] for name, result in results.items()},
        "random": random_summary,
    }
    write_json(output / "metrics.json", payload)
    return payload
