from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.consensus_policy import (
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import PolicyConfig
from src.eval.policy_baselines import (
    BaselineSuiteConfig,
    evaluate_standard_baselines,
    run_scored_policy,
    save_policy_result,
    write_json,
)


DIAGNOSTIC_COLUMNS = (
    "consensus_economic_score",
    "retrieval_expected_alpha",
    "retrieval_agreement_score",
    "retrieval_confidence",
    "retrieval_scale_score_std",
    "retrieval_scale_sign_agreement",
    "seed_vote_count",
    "seed_rank_std",
)


def _spearman(left: pd.Series, right: pd.Series) -> float | None:
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 3 or pair["left"].nunique() < 2 or pair["right"].nunique() < 2:
        return None
    value = pair["left"].rank(method="average").corr(
        pair["right"].rank(method="average")
    )
    return None if value is None or not np.isfinite(value) else float(value)


def _diagnostics(consensus: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rows": int(len(consensus)),
        "dates": int(consensus["timestamp"].nunique()),
        "tickers": int(consensus["ticker"].nunique()),
        "mean_seed_score_std": float(consensus["seed_score_std"].mean()),
        "mean_seed_rank_std": float(consensus["seed_rank_std"].mean()),
        "mean_seed_vote_fraction": float(consensus["seed_vote_fraction"].mean()),
    }
    for column in (
        "retrieval_confidence",
        "retrieval_agreement_score",
        "retrieval_historical_diversity",
        "retrieval_effective_sample_size",
        "retrieval_cross_ticker_rate",
        "retrieval_scale_score_std",
        "retrieval_scale_sign_agreement",
    ):
        if column in consensus:
            values = pd.to_numeric(consensus[column], errors="coerce").dropna()
            result[f"mean_{column}"] = (
                float(values.mean()) if len(values) else None
            )
    target = next(
        (
            column
            for column in (
                "decision_net_alpha",
                "decision_utility",
                "decision_fixed_horizon_return",
            )
            if column in consensus
        ),
        None,
    )
    if target is not None:
        result["realized_target"] = target
        result["realized_spearman"] = {
            column: _spearman(consensus[column], consensus[target])
            for column in DIAGNOSTIC_COLUMNS
            if column in consensus
        }
    return result


def evaluate_local_rank_market(
    *,
    seed_signals: dict[int, pd.DataFrame],
    policy: PolicyConfig,
    consensus_config: ConsensusSignalConfig,
    baseline_config: BaselineSuiteConfig,
    output: Path,
) -> dict[str, Any]:
    """Evaluate an ungated continuous-rank ensemble and its declared baselines."""

    if consensus_config.minimum_votes != 0:
        raise ValueError("The local-rank reconstruction requires minimum_votes=0.")
    if len(seed_signals) < 3:
        raise ValueError("The local-rank reconstruction requires at least three seeds.")

    consensus = build_consensus_signals(seed_signals, policy, consensus_config)
    primary = run_scored_policy(
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
    seed_results = {
        f"seed_{seed}_memory": run_scored_policy(frame, policy, "opportunity_score")
        for seed, frame in sorted(seed_signals.items())
    }
    standard, random_table, random_summary = evaluate_standard_baselines(
        consensus,
        policy,
        baseline_config,
        reference_sharpe=float(primary["metrics"]["sharpe"]),
    )

    output.mkdir(parents=True, exist_ok=True)
    consensus.to_parquet(output / "consensus_signals.parquet", index=False)
    save_policy_result(output, "local_rank_memory", primary)
    for name, result in seed_results.items():
        save_policy_result(output, name, result)
    for name, result in standard.items():
        save_policy_result(output, name, result)
    random_table.to_csv(output / "baselines" / "random_trials.csv", index=False)
    write_json(output / "baselines" / "random_summary.json", random_summary)

    seed_sharpes = [
        float(result["metrics"]["sharpe"]) for result in seed_results.values()
    ]
    payload = {
        "status": "completed",
        "protocol": f"market_local_continuous_{consensus_config.rank_aggregation}_rank",
        "promotion_allowed": False,
        "policy": asdict(policy),
        "consensus": asdict(consensus_config),
        "diagnostics": _diagnostics(consensus),
        "ensemble": primary["metrics"],
        "seed_memory": {
            name: result["metrics"] for name, result in seed_results.items()
        },
        "ensemble_sharpe_lift_vs_mean_seed": float(primary["metrics"]["sharpe"])
        - float(np.mean(seed_sharpes)),
        "baselines": {
            name: result["metrics"] for name, result in standard.items()
        },
        "random": random_summary,
    }
    write_json(output / "metrics.json", payload)
    return payload
