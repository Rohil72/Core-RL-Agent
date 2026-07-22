"""Select one algorithm/representation pair from the six-market three-seed matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.statistical_promotion import probability_of_backtest_overfitting  # noqa: E402


def _returns(path: str) -> pd.Series:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")["equity"].astype(float).pct_change().rename(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--pilot-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-market-wins", type=int, default=5)
    parser.add_argument("--maximum-pbo", type=float, default=0.50)
    parser.add_argument("--maximum-drawdown", type=float, default=0.20)
    args = parser.parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    pilot = json.loads(Path(args.pilot_selection).read_text(encoding="utf-8"))
    selected_algorithms = set(pilot.get("selected_algorithms", []))

    rows = []
    candidate_returns: dict[str, list[pd.Series]] = {}
    for candidate, records in matrix["candidates"].items():
        algorithm = candidate.split("__", 1)[0]
        if algorithm not in selected_algorithms:
            continue
        for record in records:
            metrics = json.loads(Path(record["metrics"]).read_text(encoding="utf-8"))
            baseline = json.loads(Path(record["baseline"]).read_text(encoding="utf-8"))
            rows.append(
                {
                    "candidate": candidate,
                    "market": record["market"],
                    "seed": int(record["seed"]),
                    "sharpe": float(metrics.get("sharpe", 0.0)),
                    "baseline_sharpe": float(baseline.get("sharpe", 0.0)),
                    "max_drawdown": abs(float(metrics.get("max_drawdown", 0.0))),
                    "target_exposure_std": float(metrics.get("target_exposure_std") or 0.0),
                }
            )
            candidate_returns.setdefault(candidate, []).append(_returns(record["equity"]))
    detail = pd.DataFrame(rows)
    market = detail.groupby(["candidate", "market"], as_index=False).agg(
        sharpe=("sharpe", "mean"),
        baseline_sharpe=("baseline_sharpe", "mean"),
        max_drawdown=("max_drawdown", "max"),
        target_exposure_std=("target_exposure_std", "mean"),
    )
    market["win"] = market["sharpe"] > market["baseline_sharpe"]
    summaries = market.groupby("candidate", as_index=False).agg(
        market_wins=("win", "sum"),
        median_sharpe=("sharpe", "median"),
        median_baseline_sharpe=("baseline_sharpe", "median"),
        worst_drawdown=("max_drawdown", "max"),
        nondegenerate_markets=("target_exposure_std", lambda values: int((values > 0.01).sum())),
    )
    summaries["median_excess_sharpe"] = summaries["median_sharpe"] - summaries["median_baseline_sharpe"]

    pooled = []
    names = sorted(candidate_returns)
    for candidate in names:
        pooled.append(pd.concat(candidate_returns[candidate], axis=1).mean(axis=1).rename(candidate))
    return_matrix = pd.concat(pooled, axis=1).dropna(how="all").fillna(0.0)
    pbo = probability_of_backtest_overfitting(return_matrix[names].to_numpy(dtype=float))
    summaries["passes"] = (
        (summaries["market_wins"] >= args.minimum_market_wins)
        & (summaries["nondegenerate_markets"] >= args.minimum_market_wins)
        & (summaries["worst_drawdown"] <= args.maximum_drawdown)
        & (summaries["median_excess_sharpe"] > 0.0)
        & bool(np.isfinite(pbo) and pbo <= args.maximum_pbo)
    )
    summaries = summaries.sort_values(
        ["passes", "median_excess_sharpe", "median_sharpe", "worst_drawdown"],
        ascending=[False, False, False, True],
    )
    selected = summaries.loc[summaries["passes"], "candidate"].head(1).tolist()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output.with_name("final_policy_cell_metrics.csv"), index=False)
    market.to_csv(output.with_name("final_policy_market_metrics.csv"), index=False)
    summaries.to_csv(output.with_name("final_policy_leaderboard.csv"), index=False)
    payload = {
        "status": "selected" if selected else "rejected",
        "selected_candidate": selected[0] if selected else None,
        "pbo": float(pbo) if np.isfinite(pbo) else None,
        "maximum_pbo": args.maximum_pbo,
        "minimum_market_wins": args.minimum_market_wins,
        "confirmation_unlocked": False,
        "development_only": True,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not selected:
        raise RuntimeError("Final development gate rejected every policy/representation candidate.")


if __name__ == "__main__":
    main()
