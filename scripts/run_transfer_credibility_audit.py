"""Audit transfer robustness and mechanism from immutable experiment artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.policy_baselines import write_json  # noqa: E402
from src.eval.statistical_promotion import (  # noqa: E402
    deflated_sharpe_probability,
    probability_of_backtest_overfitting,
)
from src.eval.transfer_credibility import (  # noqa: E402
    annualized_sharpe,
    country_jackknife,
    equal_market_returns,
    equity_returns,
    market_cluster_bootstrap,
    neighbor_age_profile,
    neighbor_market_profile,
    paired_block_bootstrap,
    reliability_deciles,
)


def _key(config: dict[str, Any], market: str, seed: int) -> str:
    prefix = "global" if config["experiment"].get("source_layout") == "global_shared" else "regional"
    return f"{prefix}_{market}_seed_{seed}"


def _curve(path: Path) -> pd.Series:
    return equity_returns(pd.read_csv(path))


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _markdown(summary: dict[str, Any], strategy: pd.DataFrame) -> str:
    lines = [
        "# Transfer Credibility Audit",
        "",
        f"Period: `{summary['period']}`",
        "",
        "This report is diagnostic evidence. It does not promote a policy or redefine a sealed holdout.",
        "",
        "## Strategy Results",
        "",
        "```text\n" + strategy.to_string(index=False) + "\n```"
        if len(strategy)
        else "No complete strategies were found.",
        "",
        "## Interpretation Contract",
        "",
        "- Block-bootstrap intervals preserve local return dependence at 5, 21, and 63 sessions.",
        "- Country jackknife results show whether one market carries the pooled result.",
        "- DSR and PBO account for the declared candidate count; neither "
        "converts diagnostic periods into confirmation.",
        "- Neighbor-age and source-market tables identify what memory actually transported.",
        "- Reliability deciles test whether confidence orders realized alpha "
        "rather than merely distance.",
    ]
    return "\n".join(lines) + "\n"


def run(config_path: str, run_id: str, period: str) -> dict[str, Any]:
    """Generate one period's complete statistical and mechanism audit."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["experiment"]["output_root"] / run_id
    destination = output / "credibility" / period
    destination.mkdir(parents=True, exist_ok=True)
    audit_cfg = config.get("credibility", {})
    samples = int(audit_cfg.get("bootstrap_samples", 2000))
    seed = int(audit_cfg.get("bootstrap_seed", 7))
    block_lengths = [int(value) for value in audit_cfg.get("block_lengths", [5, 21, 63])]
    baseline_names = list(
        audit_cfg.get("paired_baselines", ["equal_weight_buy_hold", "momentum_21"])
    )
    markets = list(config["experiment"]["markets"])
    strategy_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    jackknife_rows: list[dict[str, Any]] = []
    reliability_rows: list[pd.DataFrame] = []
    age_rows: list[pd.DataFrame] = []
    source_rows: list[pd.DataFrame] = []
    pooled_by_variant: dict[str, pd.Series] = {}
    variant_summaries: dict[str, Any] = {}

    for variant in config["variants"]:
        variant_id = str(variant["id"])
        candidate_curves: dict[str, pd.Series] = {}
        baseline_curves: dict[str, dict[str, pd.Series]] = {
            name: {} for name in baseline_names
        }
        candidate_sharpes: dict[str, float] = {}
        baseline_sharpes: dict[str, dict[str, float]] = {
            name: {} for name in baseline_names
        }
        for market in markets:
            root = output / "evaluation" / variant_id / period / market
            primary = root / "baselines" / "local_rank_memory"
            if not (primary / "equity_curve.csv").exists():
                continue
            candidate = _curve(primary / "equity_curve.csv")
            candidate_curves[market] = candidate
            candidate_sharpes[market] = annualized_sharpe(candidate)
            consensus_path = root / "consensus_signals.parquet"
            if consensus_path.exists():
                table = reliability_deciles(pd.read_parquet(consensus_path))
                if len(table):
                    table.insert(0, "market", market)
                    table.insert(0, "variant", variant_id)
                    reliability_rows.append(table)
            for baseline_name in baseline_names:
                baseline_path = root / "baselines" / baseline_name / "equity_curve.csv"
                if baseline_path.exists():
                    baseline = _curve(baseline_path)
                    baseline_curves[baseline_name][market] = baseline
                    baseline_sharpes[baseline_name][market] = annualized_sharpe(baseline)
                    for block_length in block_lengths:
                        for result in paired_block_bootstrap(
                            candidate,
                            baseline,
                            block_length=block_length,
                            samples=samples,
                            seed=seed + block_length,
                        ):
                            bootstrap_rows.append(
                                {
                                    "variant": variant_id,
                                    "scope": market,
                                    "baseline": baseline_name,
                                    **result.to_dict(),
                                }
                            )
            for seed_value in config["experiment"]["seeds"]:
                neighbor_path = (
                    output
                    / "retrieval"
                    / variant_id
                    / _key(config, market, int(seed_value))
                    / period
                    / "eval"
                    / "neighbors.parquet"
                )
                if not neighbor_path.exists():
                    raise FileNotFoundError(
                        f"Missing required neighbor artifact: {neighbor_path}"
                    )
                neighbors = pd.read_parquet(neighbor_path)
                age = neighbor_age_profile(neighbors)
                if len(age):
                    age.insert(0, "seed", int(seed_value))
                    age.insert(0, "market", market)
                    age.insert(0, "variant", variant_id)
                    age_rows.append(age)
                sources = neighbor_market_profile(neighbors)
                if len(sources):
                    sources.insert(0, "seed", int(seed_value))
                    sources.insert(0, "query_market", market)
                    sources.insert(0, "variant", variant_id)
                    source_rows.append(sources)

        missing_markets = sorted(set(markets).difference(candidate_curves))
        if missing_markets:
            raise FileNotFoundError(
                f"{variant_id} lacks market equity curves: {missing_markets}"
            )
        for baseline_name, curves in baseline_curves.items():
            missing_baselines = sorted(set(markets).difference(curves))
            if missing_baselines:
                raise FileNotFoundError(
                    f"{variant_id}/{baseline_name} lacks markets: {missing_baselines}"
                )

        pooled = equal_market_returns(candidate_curves)
        pooled_by_variant[variant_id] = pooled
        pooled_sharpe = annualized_sharpe(pooled)
        strategy_rows.append(
            {
                "strategy": variant_id,
                "kind": "memory",
                "markets": len(candidate_curves),
                "pooled_sharpe": pooled_sharpe,
                "annualized_return": float(pooled.mean() * 252.0) if len(pooled) else np.nan,
                "dsr_probability": deflated_sharpe_probability(
                    pooled.to_numpy(dtype=float), len(config["variants"])
                ),
            }
        )
        jackknife = country_jackknife(candidate_curves)
        if len(jackknife):
            jackknife.insert(0, "variant", variant_id)
            jackknife_rows.extend(jackknife.to_dict("records"))
        comparisons: dict[str, Any] = {}
        for baseline_name, curves in baseline_curves.items():
            baseline_pooled = equal_market_returns(curves)
            for block_length in block_lengths:
                for result in paired_block_bootstrap(
                    pooled,
                    baseline_pooled,
                    block_length=block_length,
                    samples=samples,
                    seed=seed + block_length + 100,
                ):
                    bootstrap_rows.append(
                        {
                            "variant": variant_id,
                            "scope": "equal_market_pool",
                            "baseline": baseline_name,
                            **result.to_dict(),
                        }
                    )
            comparisons[baseline_name] = market_cluster_bootstrap(
                candidate_sharpes,
                baseline_sharpes[baseline_name],
                samples=samples,
                seed=seed,
            )
        variant_summaries[variant_id] = {
            "markets": len(candidate_curves),
            "pooled_sharpe": pooled_sharpe,
            "paired_market_cluster": comparisons,
        }

    tabular_root = output / "tabular_baselines" / period
    for model in ("elastic_net", "hist_gradient_boosting", "pca_knn"):
        curves = {}
        for market in markets:
            path = tabular_root / market / "baselines" / model / "equity_curve.csv"
            if path.exists():
                curves[market] = _curve(path)
        pooled = equal_market_returns(curves)
        missing_markets = sorted(set(markets).difference(curves))
        if missing_markets:
            raise FileNotFoundError(
                f"Tabular baseline {model} lacks markets: {missing_markets}"
            )
        strategy_rows.append(
            {
                "strategy": model,
                "kind": "frozen_state_decoder",
                "markets": len(curves),
                "pooled_sharpe": annualized_sharpe(pooled),
                "annualized_return": float(pooled.mean() * 252.0),
                "dsr_probability": deflated_sharpe_probability(
                    pooled.to_numpy(dtype=float), 3
                ),
            }
        )

    performance = pd.concat(
        [series.rename(name) for name, series in pooled_by_variant.items()],
        axis=1,
        join="inner",
    ).dropna()
    pbo = (
        probability_of_backtest_overfitting(
            performance.to_numpy(dtype=float),
            int(audit_cfg.get("pbo_blocks", 8)),
        )
        if performance.shape[1] >= 2
        else np.nan
    )
    strategy = pd.DataFrame(strategy_rows).sort_values(
        "pooled_sharpe", ascending=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(destination / "block_bootstrap.csv", index=False)
    pd.DataFrame(jackknife_rows).to_csv(destination / "country_jackknife.csv", index=False)
    strategy.to_csv(destination / "strategy_summary.csv", index=False)
    _concat_frames(reliability_rows).to_csv(
        destination / "reliability_deciles.csv", index=False
    )
    _concat_frames(age_rows).to_csv(
        destination / "neighbor_age_profile.csv", index=False
    )
    _concat_frames(source_rows).to_csv(
        destination / "neighbor_market_profile.csv", index=False
    )
    summary = {
        "status": "completed",
        "period": period,
        "evidence_status": config["periods"][period]["evidence_status"],
        "promotion_allowed": False,
        "pbo": pbo,
        "variant_count": len(config["variants"]),
        "variant_summaries": variant_summaries,
    }
    write_json(destination / "summary.json", summary)
    (destination / "audit_report.md").write_text(
        _markdown(summary, strategy), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--period", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.period), indent=2))


if __name__ == "__main__":
    main()
