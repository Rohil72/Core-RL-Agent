from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapDelta:
    """Dependence-aware uncertainty for a paired strategy-minus-baseline metric."""

    metric: str
    block_length: int
    observed_delta: float
    ci_low: float
    ci_high: float
    probability_positive: float
    samples: int
    observations: int

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a JSON-safe representation."""

        return asdict(self)


def annualized_sharpe(values: np.ndarray | pd.Series) -> float:
    """Compute zero-risk-free daily Sharpe using 252 trading sessions."""

    returns = np.asarray(values, dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size < 2:
        return float("nan")
    deviation = float(np.std(returns, ddof=1))
    if deviation <= 1e-12:
        return 0.0
    return float(np.mean(returns) / deviation * np.sqrt(252.0))


def equity_returns(frame: pd.DataFrame) -> pd.Series:
    """Convert an equity artifact to a UTC-indexed daily return series."""

    required = {"timestamp", "equity"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Equity frame is missing columns: {sorted(missing)}")
    indexed = frame.copy()
    indexed["timestamp"] = pd.to_datetime(indexed["timestamp"], utc=True)
    values = indexed.sort_values("timestamp").set_index("timestamp")["equity"]
    return pd.to_numeric(values, errors="coerce").pct_change().fillna(0.0)


def paired_block_bootstrap(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    block_length: int,
    samples: int,
    seed: int,
) -> list[BootstrapDelta]:
    """Bootstrap paired mean-return and Sharpe deltas with circular blocks."""

    if block_length <= 0 or samples <= 0:
        raise ValueError("block_length and samples must be positive.")
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 3:
        return [
            BootstrapDelta(metric, block_length, np.nan, np.nan, np.nan, np.nan, samples, len(aligned))
            for metric in ("annualized_return", "sharpe")
        ]
    left = aligned["candidate"].to_numpy(dtype=float)
    right = aligned["baseline"].to_numpy(dtype=float)
    observed = {
        "annualized_return": float((left.mean() - right.mean()) * 252.0),
        "sharpe": annualized_sharpe(left) - annualized_sharpe(right),
    }
    rng = np.random.default_rng(seed)
    estimates = {name: np.empty(samples, dtype=float) for name in observed}
    n = len(left)
    block = min(int(block_length), n)
    blocks_needed = int(np.ceil(n / block))
    offsets = np.arange(block, dtype=int)
    for draw in range(samples):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sampled_left = left[indices]
        sampled_right = right[indices]
        estimates["annualized_return"][draw] = (
            sampled_left.mean() - sampled_right.mean()
        ) * 252.0
        estimates["sharpe"][draw] = annualized_sharpe(
            sampled_left
        ) - annualized_sharpe(sampled_right)
    return [
        BootstrapDelta(
            metric=name,
            block_length=block_length,
            observed_delta=value,
            ci_low=float(np.quantile(estimates[name], 0.025)),
            ci_high=float(np.quantile(estimates[name], 0.975)),
            probability_positive=float(np.mean(estimates[name] > 0.0)),
            samples=samples,
            observations=n,
        )
        for name, value in observed.items()
    ]


def equal_market_returns(curves: Mapping[str, pd.Series]) -> pd.Series:
    """Pool market return streams with equal weight on every available date."""

    if not curves:
        return pd.Series(dtype=float)
    panel = pd.concat(
        [series.rename(str(market)) for market, series in sorted(curves.items())],
        axis=1,
    ).sort_index()
    return panel.mean(axis=1, skipna=True).dropna()


def country_jackknife(curves: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Recompute pooled performance while omitting each market in turn."""

    rows: list[dict[str, float | int | str]] = []
    for excluded in sorted(curves):
        retained = {key: value for key, value in curves.items() if key != excluded}
        pooled = equal_market_returns(retained)
        rows.append(
            {
                "excluded_market": excluded,
                "retained_markets": len(retained),
                "observations": len(pooled),
                "pooled_sharpe": annualized_sharpe(pooled),
                "annualized_return": float(pooled.mean() * 252.0) if len(pooled) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def market_cluster_bootstrap(
    candidate_sharpes: Mapping[str, float],
    baseline_sharpes: Mapping[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap markets as independent clusters for cross-country robustness."""

    markets = sorted(set(candidate_sharpes).intersection(baseline_sharpes))
    deltas = np.asarray(
        [candidate_sharpes[market] - baseline_sharpes[market] for market in markets],
        dtype=float,
    )
    deltas = deltas[np.isfinite(deltas)]
    if not len(deltas):
        return {"market_count": 0, "observed_median_delta": np.nan}
    rng = np.random.default_rng(seed)
    estimates = np.asarray(
        [np.median(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(samples)],
        dtype=float,
    )
    return {
        "market_count": int(len(deltas)),
        "observed_median_delta": float(np.median(deltas)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "probability_positive": float(np.mean(estimates > 0.0)),
        "samples": int(samples),
    }


def reliability_deciles(
    signals: pd.DataFrame,
    *,
    reliability_column: str = "retrieval_confidence",
    outcome_column: str = "future_blended_alpha_63",
) -> pd.DataFrame:
    """Measure realized outcomes from low to high retrieval reliability."""

    if reliability_column not in signals or outcome_column not in signals:
        return pd.DataFrame()
    frame = signals[[reliability_column, outcome_column]].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()
    if len(frame) < 10 or frame[reliability_column].nunique() < 2:
        return pd.DataFrame()
    bins = min(10, int(frame[reliability_column].nunique()))
    frame["reliability_decile"] = pd.qcut(
        frame[reliability_column].rank(method="first"),
        bins,
        labels=False,
    ) + 1
    grouped = frame.groupby("reliability_decile", as_index=False)
    result = grouped.agg(
        rows=(outcome_column, "size"),
        mean_reliability=(reliability_column, "mean"),
        mean_realized_alpha=(outcome_column, "mean"),
        median_realized_alpha=(outcome_column, "median"),
        positive_alpha_rate=(outcome_column, lambda values: float((values > 0.0).mean())),
    )
    return result


def neighbor_age_profile(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Summarize which historical ages and source markets supply evidence."""

    if "neighbor_age_days" not in neighbors:
        return pd.DataFrame()
    frame = neighbors.copy()
    frame["neighbor_age_days"] = pd.to_numeric(
        frame["neighbor_age_days"], errors="coerce"
    )
    edges = [-np.inf, 365, 730, 1461, 2520, np.inf]
    labels = ["0-1y", "1-2y", "2-4y", "4-7y", "7y+"]
    frame["age_bucket"] = pd.cut(
        frame["neighbor_age_days"], edges, labels=labels, right=False
    )
    all_evidence_weight = pd.to_numeric(
        frame.get("neighbor_evidence_weight", pd.Series(1.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    total_evidence_weight = float(all_evidence_weight.sum())
    outcome = next(
        (
            column
            for column in ("future_blended_alpha_63", "future_universe_alpha_63")
            if column in frame
        ),
        None,
    )
    rows = []
    for bucket, group in frame.groupby("age_bucket", observed=False):
        evidence_weight = pd.to_numeric(
            group.get("neighbor_evidence_weight", pd.Series(1.0, index=group.index)),
            errors="coerce",
        ).fillna(0.0)
        row: dict[str, float | int | str] = {
            "age_bucket": str(bucket),
            "neighbors": int(len(group)),
            "fraction": float(len(group) / len(frame)) if len(frame) else 0.0,
            "evidence_weight_fraction": (
                float(evidence_weight.sum() / total_evidence_weight)
                if total_evidence_weight > 0.0
                else 0.0
            ),
            "median_distance": float(pd.to_numeric(group["distance"], errors="coerce").median()),
            "unique_tickers": int(group["neighbor_ticker"].nunique()),
        }
        if outcome is not None:
            values = pd.to_numeric(group[outcome], errors="coerce")
            row["mean_neighbor_alpha"] = float(values.mean())
            row["positive_neighbor_alpha_rate"] = float((values > 0.0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def neighbor_market_profile(neighbors: pd.DataFrame) -> pd.DataFrame:
    """Measure source-market concentration in a global memory query."""

    if "neighbor_market" not in neighbors:
        return pd.DataFrame()
    values = neighbors["neighbor_market"].fillna("unknown").astype(str)
    counts = values.value_counts(dropna=False)
    total = max(int(counts.sum()), 1)
    result = pd.DataFrame(
        {
            "neighbor_market": counts.index,
            "neighbors": counts.to_numpy(dtype=int),
            "fraction": counts.to_numpy(dtype=float) / total,
        }
    )
    if "neighbor_evidence_weight" in neighbors:
        weighted = (
            pd.DataFrame(
                {
                    "neighbor_market": values,
                    "weight": pd.to_numeric(
                        neighbors["neighbor_evidence_weight"], errors="coerce"
                    ).fillna(0.0),
                }
            )
            .groupby("neighbor_market")["weight"]
            .sum()
        )
        denominator = float(weighted.sum())
        result["evidence_weight_fraction"] = result["neighbor_market"].map(
            lambda market: (
                float(weighted.get(market, 0.0) / denominator)
                if denominator > 0.0
                else 0.0
            )
        )
    return result
