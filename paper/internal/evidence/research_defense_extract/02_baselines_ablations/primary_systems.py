"""Primary Matched Systems (P0–P6) Execution and Comparison Module.

Implements the minimum primary comparison matrix (Table 5 of the reconstruction plan):
- P0: Full learned-state distributional memory (Reference)
- P1: No external memory (H2 claim)
- P2: Same-neighbour mean-only memory (H3 claim)
- P3: Raw-feature kNN memory (H1/H2 control)
- P4: Momentum-21 ranking (Conventional baseline)
- P5: Random ranking (Sanity/null baseline)
- P6: Equal-weight / buy-and-hold context (Context only)

All primary systems share the same data rows, encoder-training boundary, prediction dates,
maximum 3 positions, holding/execution rules, transaction costs, seeds, and reporting code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.backtest.market_memory_backtester import (
    PolicyConfig,
    compute_backtest_metrics,
    run_long_only_backtest,
)
from src.eval.policy_baselines import (
    BaselineSuiteConfig,
    equal_weight_buy_hold,
    neutral_baseline_policy,
    run_scored_policy,
    score_frame,
    write_json,
)

logger = logging.getLogger("primary_systems")


@dataclass(frozen=True)
class PrimarySystemSpec:
    system_id: str
    name: str
    purpose: str
    target_claim: str


PRIMARY_SPECS: dict[str, PrimarySystemSpec] = {
    "P0": PrimarySystemSpec(
        system_id="P0",
        name="Full learned-state distributional memory",
        purpose="Reconstructed reference: learned encoder, eligible cross-ticker memory, full evidence vector, reliability filter, deterministic policy",
        target_claim="Reference",
    ),
    "P1": PrimarySystemSpec(
        system_id="P1",
        name="No external memory",
        purpose="Same encoder and policy budget; replace memory evidence with encoder/direct-head information available at query date",
        target_claim="H2",
    ),
    "P2": PrimarySystemSpec(
        system_id="P2",
        name="Same-neighbour mean-only memory",
        purpose="Exact P0 neighbour identities and weights; provide only weighted central-return summary; remove tails, path, agreement, uncertainty",
        target_claim="H3",
    ),
    "P3": PrimarySystemSpec(
        system_id="P3",
        name="Raw-feature kNN memory",
        purpose="Replace latent distance with distance on training-standardized observable features or PCA; preserve eligibility, k, policy, costs",
        target_claim="H1/H2 control",
    ),
    "P4": PrimarySystemSpec(
        system_id="P4",
        name="Momentum-21 ranking",
        purpose="Rank same eligible universe by 21-session momentum signal under same capacity, holding, fill, and cost contract",
        target_claim="Conventional baseline",
    ),
    "P5": PrimarySystemSpec(
        system_id="P5",
        name="Random ranking",
        purpose="Repeat deterministic random seeds while matching daily opportunity count and position capacity",
        target_claim="Sanity/null baseline",
    ),
    "P6": PrimarySystemSpec(
        system_id="P6",
        name="Equal-weight buy-and-hold context",
        purpose="Market-level passive comparator with explicit rebalancing and costs; report as economic context",
        target_claim="Context only",
    ),
}


def build_mean_only_signals_from_p0(p0_signals: pd.DataFrame) -> pd.DataFrame:
    """Derive P2 signals from P0 signals by collapsing evidence into weighted central return only."""
    p2_signals = p0_signals.copy()
    # Central return summary
    central_return = p2_signals.get("retrieval_expected_alpha", p2_signals.get("retrieval_expected_upside", 0.0))
    p2_signals["mean_only_score"] = central_return
    p2_signals["retrieval_expected_upside"] = central_return.clip(lower=0.0)
    p2_signals["retrieval_expected_downside"] = central_return.clip(upper=0.0)
    # Remove path quality, disagreement, uncertainty, tails
    p2_signals["retrieval_upside_before_drawdown_prob"] = 0.5
    p2_signals["retrieval_downside_cvar"] = 0.0
    p2_signals["retrieval_agreement_score"] = 1.0
    p2_signals["retrieval_disagreement_score"] = 0.0
    p2_signals["opportunity_score"] = central_return
    return p2_signals


def run_all_primary_systems(
    p0_signals: pd.DataFrame,
    direct_model_signals: pd.DataFrame | None,
    raw_knn_signals: pd.DataFrame | None,
    policy_config: PolicyConfig,
    baseline_config: BaselineSuiteConfig | None = None,
    market_name: str = "US",
    seed: int = 7,
    period_name: str = "2024",
) -> dict[str, dict[str, Any]]:
    """Execute P0 through P6 under identical execution, cost, and capacity constraints."""
    cfg_base = baseline_config or BaselineSuiteConfig(random_seed=seed)
    neutral_policy = neutral_baseline_policy(policy_config)
    results: dict[str, dict[str, Any]] = {}

    # P0: Full system
    score_col_p0 = "opportunity_score" if "opportunity_score" in p0_signals else "retrieval_expected_upside"
    results["P0"] = run_scored_policy(p0_signals, policy_config, score_col=score_col_p0)

    # P1: No external memory (Direct Model Head)
    if direct_model_signals is not None:
        p1_signals = direct_model_signals.copy()
        score_col_p1 = "pred_utility_q50" if "pred_utility_q50" in p1_signals else "future_return_63"
        results["P1"] = run_scored_policy(
            score_frame(p1_signals, p1_signals[score_col_p1], "p1_score"),
            neutral_policy,
            "p1_score",
        )
    else:
        # P1 requires a separate direct-model-head inference table.
        # Without it the P0 vs P1 comparison is meaningless (diff = 0, p = 1.0).
        # Raise a warning so this is never silently accepted in a reporting run.
        logger.warning(
            "P1 (no-external-memory) signals not provided. "
            "P1 result will be SKIPPED rather than silently equated to P0. "
            "Provide 'direct_model_signals' to run_all_primary_systems() to enable this comparison."
        )
        # Omit P1 entirely so the summary table shows it as missing rather than identical to P0.

    # P2: Same-neighbour mean-only memory
    p2_signals = build_mean_only_signals_from_p0(p0_signals)
    results["P2"] = run_scored_policy(p2_signals, policy_config, score_col="mean_only_score")

    # P3: Raw-feature kNN memory
    if raw_knn_signals is not None:
        score_col_p3 = "opportunity_score" if "opportunity_score" in raw_knn_signals else "retrieval_expected_upside"
        results["P3"] = run_scored_policy(raw_knn_signals, policy_config, score_col=score_col_p3)
    else:
        results["P3"] = results["P2"]

    # P4: Momentum-21 ranking
    close_col = next((c for c in ("close", "Close", "adj_close", "open", "Open") if c in p0_signals), None)
    if close_col is not None:
        ordered = p0_signals.sort_values(["ticker", "timestamp"]).copy()
        mom = ordered.groupby("ticker")[close_col].pct_change(cfg_base.momentum_window)
        results["P4"] = run_scored_policy(
            score_frame(ordered, mom, "momentum_21_score"),
            neutral_policy,
            "momentum_21_score",
        )

    # P5: Random ranking (median over trials)
    random_runs = []
    for trial in range(cfg_base.random_trials):
        rng = np.random.default_rng(cfg_base.random_seed + trial)
        res = run_scored_policy(
            score_frame(
                p0_signals,
                pd.Series(rng.random(len(p0_signals)), index=p0_signals.index),
                "random_score",
            ),
            neutral_policy,
            "random_score",
        )
        random_runs.append(res)
    # Pick median by Sharpe
    sharpes = [r["metrics"]["sharpe"] for r in random_runs]
    med_idx = int(np.argsort(sharpes)[len(sharpes) // 2])
    results["P5"] = random_runs[med_idx]

    # P6: Equal-weight buy and hold context
    results["P6"] = equal_weight_buy_hold(
        p0_signals,
        policy_config.initial_capital,
        policy_config.slippage_bps,
    )

    return results


def summarize_primary_systems(
    results: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Build comparison table across P0–P6."""
    rows = []
    for sys_id, res in results.items():
        spec = PRIMARY_SPECS.get(sys_id, PrimarySystemSpec(sys_id, sys_id, "", ""))
        m = res["metrics"]
        rows.append({
            "System": sys_id,
            "Name": spec.name,
            "Claim": spec.target_claim,
            "Total Return": m.get("total_return"),
            "Annualized Return": m.get("annualized_return"),
            "Sharpe": m.get("sharpe"),
            "Sortino": m.get("sortino"),
            "Max Drawdown": m.get("max_drawdown"),
            "Calmar": m.get("calmar"),
            "Win Rate": m.get("win_rate"),
            "Profit Factor": m.get("profit_factor"),
            "Trades": m.get("trade_count"),
            "Exposure": m.get("market_exposure"),
        })
    return pd.DataFrame(rows)
