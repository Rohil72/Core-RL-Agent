"""Select the two offline policies that transfer across pilot markets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=2)
    parser.add_argument("--minimum-market-wins", type=int, default=3)
    parser.add_argument("--maximum-drawdown-degradation", type=float, default=0.05)
    args = parser.parse_args()

    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    rows = []
    for algorithm, markets in matrix["candidates"].items():
        for market, paths in markets.items():
            candidate = json.loads(Path(paths["candidate"]).read_text(encoding="utf-8"))
            baseline = json.loads(Path(paths["baseline"]).read_text(encoding="utf-8"))
            candidate_sharpe = float(candidate.get("sharpe", 0.0))
            baseline_sharpe = float(baseline.get("sharpe", 0.0))
            candidate_drawdown = abs(float(candidate.get("max_drawdown", 0.0)))
            baseline_drawdown = abs(float(baseline.get("max_drawdown", 0.0)))
            action_std = float(candidate.get("target_exposure_std") or 0.0)
            rows.append(
                {
                    "algorithm": algorithm,
                    "market": market,
                    "sharpe": candidate_sharpe,
                    "baseline_sharpe": baseline_sharpe,
                    "excess_sharpe": candidate_sharpe - baseline_sharpe,
                    "max_drawdown": candidate_drawdown,
                    "baseline_drawdown": baseline_drawdown,
                    "market_win": candidate_sharpe > baseline_sharpe,
                    "drawdown_pass": candidate_drawdown <= baseline_drawdown + args.maximum_drawdown_degradation,
                    "nondegenerate": action_std > 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    for algorithm, group in frame.groupby("algorithm"):
        summaries.append(
            {
                "algorithm": algorithm,
                "market_wins": int(group["market_win"].sum()),
                "drawdown_passes": int(group["drawdown_pass"].sum()),
                "nondegenerate_markets": int(group["nondegenerate"].sum()),
                "median_sharpe": float(group["sharpe"].median()),
                "median_excess_sharpe": float(group["excess_sharpe"].median()),
                "worst_drawdown": float(group["max_drawdown"].max()),
            }
        )
    leaderboard = pd.DataFrame(summaries)
    leaderboard["passes"] = (
        (leaderboard["market_wins"] >= args.minimum_market_wins)
        & (leaderboard["drawdown_passes"] >= args.minimum_market_wins)
        & (leaderboard["nondegenerate_markets"] >= args.minimum_market_wins)
        & (leaderboard["median_excess_sharpe"] > 0.0)
    )
    leaderboard = leaderboard.sort_values(
        ["passes", "median_excess_sharpe", "median_sharpe", "worst_drawdown"],
        ascending=[False, False, False, True],
    )
    selected = leaderboard.loc[leaderboard["passes"], "algorithm"].head(args.top_n).tolist()
    status = "selected" if len(selected) == args.top_n else "rejected"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(output.with_suffix(".csv"), index=False)
    payload = {
        "status": status,
        "selected_algorithms": selected,
        "required_count": args.top_n,
        "minimum_market_wins": args.minimum_market_wins,
        "development_only": True,
        "rows": int(len(frame)),
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "selected":
        raise RuntimeError("Pilot policy gate rejected: two transferable algorithms were not found.")


if __name__ == "__main__":
    main()
