from __future__ import annotations

import glob
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.market_memory_backtester import (
    PolicyConfig,
    compute_backtest_metrics,
    equal_weight_baseline,
    run_long_only_backtest,
)
from src.memory import MarketMemoryConfig, load_latent_frame, score_market_memory
from src.memory import FittedRetrievalMetric, RelativeOutcomeConfig, attach_relative_outcomes
from src.eval.representation_metrics import compute_memory_metrics

try:
    import yaml
except ImportError:  # pragma: no cover - environment dependent
    yaml = None


LOGGER = logging.getLogger(__name__)


def load_evaluation_config(path: str | Path) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    return _simple_yaml_load(text)


def run_market_memory_evaluation(
    config: dict,
    root: Path,
    *,
    dry_run: bool = False,
    run_id: str | None = None,
) -> Path:
    run_name = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if dry_run and not run_name.startswith("dry_run_"):
        run_name = f"dry_run_{run_name}"
    out_dir = root / config["data"].get("output_dir", "reports/market_memory_backtests") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    memory = load_latent_frame(root / config["data"]["train_latents"])
    query = load_latent_frame(root / config["data"]["test_latents"])
    query = join_prices(query, config["data"].get("precomputed_glob", "data/precomputed_v2/*.parquet"), root)
    relative_cfg = config.get("relative_outcomes", {})
    if relative_cfg.get("enabled", False):
        outcome_args = {key: value for key, value in relative_cfg.items() if key != "enabled"}
        outcome_cfg = RelativeOutcomeConfig(**outcome_args)
        memory = attach_relative_outcomes(memory, outcome_cfg)
        query = attach_relative_outcomes(query, outcome_cfg)
    if dry_run:
        query = query.head(int(config.get("run", {}).get("dry_run_rows", 500)))

    memory_cfg = MarketMemoryConfig(**config.get("memory", {}))
    policy_cfg = PolicyConfig(**config.get("policy", {}))
    metric = None
    metric_path = config.get("retrieval_metric", {}).get("path")
    if metric_path:
        resolved_metric_path = Path(metric_path)
        if not resolved_metric_path.is_absolute():
            resolved_metric_path = root / resolved_metric_path
        metric = FittedRetrievalMetric.load(resolved_metric_path)
    keep_cols = _query_columns_to_keep(query)
    signals, neighbors = score_market_memory(
        memory,
        query,
        memory_cfg,
        keep_query_columns=keep_cols,
        retrieval_metric=metric,
    )

    runs = run_policy_suite(signals, policy_cfg, config.get("evaluation", {}))
    primary = runs["retrieval"]
    extra = equal_weight_baseline(signals, policy_cfg.initial_capital)
    extra.update(signal_quality_metrics(signals))
    metric_target = config.get("evaluation", {}).get("memory_metric_target")
    target_names = [metric_target] if metric_target else [memory_cfg.target_alpha or memory_cfg.target_upside]
    extra.update(compute_memory_metrics(signals, neighbors, target_names=target_names))
    extra.update({f"{name}_policy_return": data["metrics"].get("total_return") for name, data in runs.items() if name != "retrieval"})
    if extra.get("model_head_policy_return") is not None:
        extra["retrieval_vs_model_delta"] = primary["metrics"]["total_return"] - extra["model_head_policy_return"]
    else:
        extra["retrieval_vs_model_delta"] = None

    metrics = dict(primary["metrics"])
    metrics.update(extra)
    metrics["policy_suite"] = {name: data["metrics"] for name, data in runs.items()}
    metrics["config"] = {
        "memory": asdict(memory_cfg),
        "policy": asdict(policy_cfg),
        "dry_run": dry_run,
    }

    write_run_artifacts(
        out_dir,
        config,
        signals,
        neighbors,
        primary["trades"],
        primary["equity"],
        metrics,
        primary.get("decisions"),
    )
    if config.get("run", {}).get("write_memory_reports", True):
        write_memory_reports(root / "reports" / "memory", signals, neighbors, primary["trades"], metrics)
    return out_dir


def run_policy_suite(signals: pd.DataFrame, policy_cfg: PolicyConfig, evaluation_cfg: dict | None = None) -> dict[str, dict]:
    evaluation_cfg = evaluation_cfg or {}
    baseline_names = _baseline_names(evaluation_cfg)
    suite: dict[str, dict] = {}
    suite["retrieval"] = _run_named_policy(signals, policy_cfg, "opportunity_score")

    if "model_head" in baseline_names:
        model_frame = model_head_score_frame(signals)
        if model_frame is not None:
            suite["model_head"] = _run_named_policy(model_frame, policy_cfg, "model_head_score")

    if "momentum" in baseline_names:
        momentum_frame = momentum_score_frame(signals, window=int(evaluation_cfg.get("momentum_window", 21)))
        if momentum_frame is not None:
            suite["momentum"] = _run_named_policy(momentum_frame, policy_cfg, "momentum_score")

    if "random" in baseline_names:
        random_frame = random_score_frame(signals, seed=int(evaluation_cfg.get("random_seed", 7)))
        suite["random"] = _run_named_policy(random_frame, policy_cfg, "random_score")

    return suite


def _run_named_policy(signals: pd.DataFrame, policy_cfg: PolicyConfig, score_col: str) -> dict:
    decisions: list[dict] = []
    trades, equity = run_long_only_backtest(
        signals,
        policy_cfg,
        score_col=score_col,
        decision_log=decisions,
    )
    metrics = compute_backtest_metrics(trades, equity, policy_cfg.initial_capital)
    return {"trades": trades, "equity": equity, "metrics": metrics, "decisions": pd.DataFrame(decisions)}


def join_prices(latents: pd.DataFrame, parquet_glob: str, root: Path) -> pd.DataFrame:
    frame = latents.copy()
    if any(c in frame.columns for c in ("close", "Close", "adj_close")) and any(
        c in frame.columns for c in ("open", "Open", "adj_open")
    ):
        return frame

    pattern = str(root / parquet_glob) if not Path(parquet_glob).is_absolute() else parquet_glob
    paths = [Path(p) for p in sorted(glob.glob(pattern))]
    if not paths:
        LOGGER.warning("No precomputed price files matched %s", parquet_glob)
        return frame

    wanted = set(frame["ticker"].astype(str).unique())
    parts = []
    for path in paths:
        if path.stem not in wanted:
            continue
        pq = pd.read_parquet(path)
        if "timestamp" not in pq.columns:
            pq = pq.reset_index().rename(columns={"Date": "timestamp", "index": "timestamp"})
        pq["timestamp"] = pd.to_datetime(pq["timestamp"], utc=True)
        pq["ticker"] = str(pq["ticker"].iloc[0]) if "ticker" in pq.columns else path.stem
        keep = ["ticker", "timestamp"] + [c for c in ("open", "close", "Open", "Close", "adj_open", "adj_close") if c in pq]
        parts.append(pq[keep])

    if not parts:
        return frame
    prices = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "timestamp"])
    out = frame.merge(prices, on=["ticker", "timestamp"], how="left", suffixes=("", "_price"))
    for col in ("open", "close", "Open", "Close", "adj_open", "adj_close"):
        price_col = f"{col}_price"
        if price_col in out:
            out[col] = out[col].combine_first(out[price_col]) if col in out else out[price_col]
            out = out.drop(columns=[price_col])
    return out


def model_head_score_frame(signals: pd.DataFrame) -> pd.DataFrame | None:
    up_col = _first_existing(signals, ["pred_future_max_return_63", "future_max_return_63_pred"])
    down_col = _first_existing(signals, ["pred_future_min_return_63", "future_min_return_63_pred"])
    if up_col is None:
        return None
    out = signals.copy()
    downside = out[down_col].abs() if down_col else 0.0
    out["model_head_score"] = out[up_col] - 0.80 * downside
    out["retrieval_expected_upside"] = out[up_col]
    out["retrieval_expected_downside"] = out[down_col] if down_col else 0.0
    out["retrieval_confidence"] = 1.0
    return out


def momentum_score_frame(signals: pd.DataFrame, window: int = 21) -> pd.DataFrame | None:
    price_col = _first_existing(signals, ["close", "Close", "adj_close"])
    if price_col is None:
        return None
    out = signals.sort_values(["ticker", "timestamp"]).copy()
    out["momentum_score"] = out.groupby("ticker")[price_col].pct_change(window)
    out["retrieval_expected_upside"] = out["momentum_score"].clip(lower=0.0)
    out["retrieval_expected_downside"] = out["momentum_score"].clip(upper=0.0)
    out["retrieval_confidence"] = out["momentum_score"].notna().astype(float)
    return out


def random_score_frame(signals: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    out = signals.copy()
    rng = np.random.default_rng(seed)
    out["random_score"] = rng.random(len(out))
    out["retrieval_expected_upside"] = 1.0
    out["retrieval_expected_downside"] = 0.0
    out["retrieval_confidence"] = 1.0
    return out


def signal_quality_metrics(signals: pd.DataFrame) -> dict:
    metrics: dict[str, Any] = {}
    if "opportunity_score" in signals and "future_max_return_63" in signals:
        valid = signals[["opportunity_score", "future_max_return_63"]].dropna()
        if len(valid) > 2:
            left = valid["opportunity_score"].rank(method="average")
            right = valid["future_max_return_63"].rank(method="average")
            metrics["score_realized_return_spearman"] = float(left.corr(right))
    for col in ("retrieval_same_ticker_rate", "retrieval_cross_ticker_rate", "retrieval_neighbor_count"):
        if col in signals:
            metrics[f"{col}_mean"] = float(signals[col].dropna().mean()) if signals[col].notna().any() else None
    if "retrieval_cross_ticker_rate" in signals:
        p = signals["retrieval_cross_ticker_rate"].dropna().clip(1e-12, 1.0)
        metrics["neighbor_entropy_proxy"] = float((-p * np.log2(p)).mean()) if len(p) else None
    return metrics


def write_run_artifacts(
    out_dir: Path,
    config: dict,
    signals: pd.DataFrame,
    neighbors: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: dict,
    decisions: pd.DataFrame | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_snapshot.yaml").write_text(dump_config(config), encoding="utf-8")
    write_frame(signals, out_dir / "signals.parquet")
    if config.get("run", {}).get("write_neighbors", True):
        write_frame(neighbors, out_dir / "neighbors.parquet")
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
    if decisions is not None:
        decisions.to_csv(out_dir / "decisions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(_native(metrics), indent=2), encoding="utf-8")
    write_report(out_dir / "report.md", _native(metrics), trades)


def write_frame(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:  # pragma: no cover - depends on optional parquet engines
        fallback = path.with_suffix(".csv")
        LOGGER.warning("Failed to write parquet %s (%s); writing %s", path, exc, fallback)
        df.to_csv(fallback, index=False)
        return fallback


def write_report(path: Path, metrics: dict, trades: pd.DataFrame) -> None:
    lines = ["# Market Memory Trading Backtest", "", "## Core Metrics", ""]
    for key in [
        "total_return",
        "annualized_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "win_rate",
        "trade_count",
        "average_trade_return",
        "average_holding_days",
        "equal_weight_baseline_return",
        "model_head_policy_return",
        "momentum_policy_return",
        "random_policy_return",
        "retrieval_vs_model_delta",
    ]:
        lines.append(f"- {key}: {metrics.get(key)}")
    lines.extend(["", "## Top Trades", ""])
    if trades.empty:
        lines.append("No trades were taken.")
    else:
        cols = ["ticker", "entry_date", "exit_date", "net_return", "pnl", "exit_reason"]
        lines.append(_markdown_table(trades[cols].head(20)))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_memory_reports(report_dir: Path, signals: pd.DataFrame, neighbors: pd.DataFrame, trades: pd.DataFrame, metrics: dict) -> None:
    """Write interpretable market-memory reasoning reports."""
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_memory_statistics(report_dir / "memory_statistics.md", signals, neighbors, metrics)
    _write_confidence_analysis(report_dir / "confidence_analysis.md", signals)
    _write_retrieval_analysis(report_dir / "retrieval_analysis.md", signals, neighbors)
    _write_experience_examples(report_dir / "experience_examples.md", neighbors)
    _write_policy_reasoning(report_dir / "policy_reasoning.md", signals, neighbors, trades)


def _write_memory_statistics(path: Path, signals: pd.DataFrame, neighbors: pd.DataFrame, metrics: dict) -> None:
    lines = ["# Memory Statistics", ""]
    lines.append(f"- Signals scored: {len(signals)}")
    lines.append(f"- Neighbor evidence rows: {len(neighbors)}")
    lines.append(f"- Mean neighbor count: {_fmt(signals.get('retrieval_neighbor_count'))}")
    lines.append(f"- Mean effective sample size: {_fmt(signals.get('retrieval_effective_sample_size'))}")
    lines.append(f"- Mean confidence: {_fmt(signals.get('retrieval_confidence'))}")
    lines.append(f"- Mean agreement: {_fmt(signals.get('retrieval_agreement_score'))}")
    lines.append(f"- Retrieval policy return: {metrics.get('total_return')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_confidence_analysis(path: Path, signals: pd.DataFrame) -> None:
    cols = [
        "retrieval_confidence",
        "retrieval_agreement_score",
        "retrieval_disagreement_score",
        "retrieval_effective_sample_size",
        "retrieval_entropy",
        "retrieval_historical_diversity",
    ]
    lines = ["# Confidence Analysis", "", "| metric | mean | p25 | p75 |", "| --- | --- | --- | --- |"]
    for col in cols:
        if col in signals:
            s = signals[col].dropna()
            lines.append(f"| {col} | {_num(s.mean())} | {_num(s.quantile(0.25))} | {_num(s.quantile(0.75))} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_retrieval_analysis(path: Path, signals: pd.DataFrame, neighbors: pd.DataFrame) -> None:
    lines = ["# Retrieval Analysis", ""]
    lines.append(f"- Cross-ticker rate: {_fmt(signals.get('retrieval_cross_ticker_rate'))}")
    lines.append(f"- Same-ticker rate: {_fmt(signals.get('retrieval_same_ticker_rate'))}")
    if "distance" in neighbors:
        s = neighbors["distance"].dropna()
        lines.append(f"- Distance median: {_num(s.median())}")
        lines.append(f"- Distance p90: {_num(s.quantile(0.90))}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_experience_examples(path: Path, neighbors: pd.DataFrame) -> None:
    lines = ["# Experience Examples", ""]
    if neighbors.empty:
        lines.append("No neighbor evidence was produced.")
    else:
        cols = [c for c in ["query_ticker", "query_timestamp", "rank", "neighbor_ticker", "neighbor_timestamp", "distance", "evidence_confidence"] if c in neighbors]
        lines.append(_markdown_table(neighbors[cols].head(30)))
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_policy_reasoning(path: Path, signals: pd.DataFrame, neighbors: pd.DataFrame, trades: pd.DataFrame) -> None:
    lines = ["# Policy Reasoning", ""]
    if trades.empty:
        lines.append("No trades were taken.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    sig = signals.copy()
    sig["timestamp"] = pd.to_datetime(sig["timestamp"], utc=True)
    for trade in trades.head(20).to_dict("records"):
        entry_date = pd.Timestamp(trade["entry_date"])
        ticker = str(trade["ticker"])
        candidates = sig[(sig["ticker"].astype(str) == ticker) & (sig["timestamp"] <= entry_date)].sort_values("timestamp")
        evidence = candidates.iloc[-1].to_dict() if not candidates.empty else {}
        lines.append(f"## {ticker} {entry_date.date()}")
        lines.append(f"- Exit reason: {trade.get('exit_reason')}")
        lines.append(f"- Net return: {trade.get('net_return')}")
        lines.append(f"- Opportunity score: {evidence.get('opportunity_score')}")
        lines.append(f"- Confidence: {evidence.get('retrieval_confidence')}")
        lines.append(f"- Agreement: {evidence.get('retrieval_agreement_score')}")
        lines.append(f"- Expected upside: {evidence.get('retrieval_expected_upside')}")
        lines.append(f"- Expected downside: {evidence.get('retrieval_expected_downside')}")
        qid = evidence.get("query_id")
        if qid is not None and not neighbors.empty:
            ncols = [c for c in ["rank", "neighbor_ticker", "neighbor_timestamp", "distance", "evidence_confidence"] if c in neighbors]
            lines.append(_markdown_table(neighbors[neighbors["query_id"] == qid][ncols].head(5)))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def dump_config(config: dict) -> str:
    if yaml is not None:
        return yaml.safe_dump(config, sort_keys=False)
    return json.dumps(config, indent=2)


def _query_columns_to_keep(query: pd.DataFrame) -> list[str]:
    keep = {
        "market",
        "sector",
        "industry",
        "open",
        "close",
        "Open",
        "Close",
        "adj_open",
        "adj_close",
        "future_max_return_63",
        "future_return_63",
        "future_universe_alpha_63",
        "future_blended_alpha_63",
        "future_min_return_63",
        "event_upside_before_drawdown_126",
        "pred_future_max_return_63",
        "pred_future_min_return_63",
    }
    prefixes = (
        "decision_",
        "pred_utility_",
        "pred_event_",
        "pred_upside_",
        "pred_drawdown_",
        "pred_cash_",
        "pred_target_",
        "pred_stock_",
        "pred_action_",
    )
    return [c for c in query.columns if c in keep or c.startswith(prefixes)]


def _baseline_names(evaluation_cfg: dict) -> set[str]:
    raw = evaluation_cfg.get("baselines", "model_head,momentum,random")
    if isinstance(raw, str):
        return {name.strip() for name in raw.split(",") if name.strip()}
    return {str(name) for name in raw}


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(_native(row[c])) for c in cols) + " |")
    return "\n".join(rows)


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def _fmt(series: pd.Series | None) -> str:
    if series is None:
        return "n/a"
    s = series.dropna()
    return _num(s.mean()) if len(s) else "n/a"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.4f}" if np.isfinite(float(value)) else "n/a"
    except Exception:
        return "n/a"


def _simple_yaml_load(text: str) -> dict:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
