"""Aggregate the locked six-market confirmation without tuning any threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rows = []
    returns = []
    for record in matrix["records"]:
        metrics = json.loads(Path(record["metrics"]).read_text(encoding="utf-8"))
        equity = pd.read_csv(record["equity"])
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        returns.append(equity.set_index("timestamp")["equity"].astype(float).pct_change().rename(f"{record['market']}_{record['seed']}"))
        rows.append({"market": record["market"], "seed": record["seed"], **metrics})
    detail = pd.DataFrame(rows)
    market = detail.groupby("market", as_index=False).agg(
        mean_return=("total_return", "mean"),
        mean_sharpe=("sharpe", "mean"),
        worst_drawdown=("max_drawdown", "min"),
    )
    pooled = pd.concat(returns, axis=1).mean(axis=1, skipna=True).dropna()
    pooled_sharpe = float(pooled.mean() / pooled.std(ddof=1) * np.sqrt(252.0)) if len(pooled) > 2 and pooled.std(ddof=1) > 1e-12 else 0.0
    cumulative = (1.0 + pooled).cumprod()
    pooled_return = float(cumulative.iloc[-1] - 1.0) if len(cumulative) else 0.0
    pooled_drawdown = float((cumulative / cumulative.cummax() - 1.0).min()) if len(cumulative) else 0.0
    positive_markets = int((market["mean_return"] > 0.0).sum())
    positive_profit = market["mean_return"].clip(lower=0.0)
    concentration = float(positive_profit.max() / positive_profit.sum()) if positive_profit.sum() > 0 else 1.0
    gates = config["promotion_gates"]
    passes = bool(
        pooled_sharpe >= float(gates["target_pooled_sharpe"])
        and abs(pooled_drawdown) <= float(gates["maximum_drawdown"])
        and positive_markets >= int(gates["minimum_positive_markets"])
        and concentration <= float(gates["maximum_profit_concentration"])
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output.with_name("confirmation_cell_metrics.csv"), index=False)
    market.to_csv(output.with_name("confirmation_market_metrics.csv"), index=False)
    payload = {
        "status": "confirmed" if passes else "rejected",
        "pooled_total_return": pooled_return,
        "pooled_sharpe": pooled_sharpe,
        "pooled_max_drawdown": pooled_drawdown,
        "positive_markets": positive_markets,
        "profit_concentration": concentration,
        "development_parameters_changed": False,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
