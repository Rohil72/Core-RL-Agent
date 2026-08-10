from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd

from src.backtest.exposure_controller import CausalExposureController, ExposureDecision
from src.backtest.risk_ledger import RiskLedgerConfig, TickerRiskLedger


@dataclass(frozen=True)
class PolicyConfig:
    top_k: int = 3
    min_score: float = 0.03
    min_expected_upside: float = 0.04
    max_expected_downside: float = 0.12
    min_confidence: float = 0.20
    min_hold_days: int = 5
    max_hold_days: int = 63
    exit_score_fraction: float = 0.50
    exit_min_score: float = 0.01
    stop_loss: float = 0.10
    slippage_bps: float = 10
    initial_capital: float = 100000.0
    min_alpha_lcb: float | None = None
    min_absolute_return_lcb: float | None = None
    use_calibrated_downside: bool = False
    holding_mode: str = "current"
    breadth_exposure_enabled: bool = False
    breadth_exposure_map: dict[int, float] | None = None
    entry_cutoff_date: str | pd.Timestamp | None = None
    accounting_cutoff_date: str | pd.Timestamp | None = None
    accounting_protocol: str = "calendar_portfolio"
    max_downside_cvar: float | None = None
    min_neighbor_count: int | None = None
    require_ood_pass: bool = False
    risk_guard_enabled: bool = False
    cooldown_sessions: int = 63
    rolling_loss_window_sessions: int = 126
    quarantine_sessions: int = 126
    max_stop_losses: int = 2
    cumulative_loss_limit: float = -0.15


def _price_col(df: pd.DataFrame) -> str:
    for col in ("open", "Open", "adj_open", "close", "Close", "adj_close"):
        if col in df.columns:
            return col
    raise ValueError("Signals must contain an open/close price column.")


def _close_col(df: pd.DataFrame) -> str:
    for col in ("close", "Close", "adj_close", "open", "Open", "adj_open"):
        if col in df.columns:
            return col
    raise ValueError("Signals must contain a close/open price column.")


def _tradable(row: pd.Series, cfg: PolicyConfig, score_col: str) -> bool:
    return tradability_reason(row, cfg, score_col) is None


def tradability_reason(row: pd.Series, cfg: PolicyConfig, score_col: str) -> str | None:
    """Return the first deterministic evidence gate that rejects a signal."""
    score = row.get(score_col)
    upside = row.get("retrieval_expected_upside", row.get(score_col))
    downside_key = "retrieval_calibrated_downside" if cfg.use_calibrated_downside else "retrieval_expected_downside"
    downside = row.get(downside_key, row.get("retrieval_expected_downside", 0.0))
    confidence = row.get("retrieval_confidence", 1.0)
    
    try:
        score_f = float(score) if score is not None and pd.notna(score) else None
        upside_f = float(upside) if upside is not None and pd.notna(upside) else None
    except (ValueError, TypeError):
        return "invalid_score"

    if score_f is None or upside_f is None or not np.isfinite(score_f) or not np.isfinite(upside_f):
        return "invalid_score"
    if score_f < cfg.min_score or upside_f < cfg.min_expected_upside:
        return "low_score"

    try:
        downside_f = float(downside) if downside is not None and pd.notna(downside) else None
    except (ValueError, TypeError):
        downside_f = None

    if downside_f is not None and np.isfinite(downside_f) and abs(downside_f) > cfg.max_expected_downside:
        return "downside"

    try:
        confidence_f = float(confidence) if confidence is not None and pd.notna(confidence) else None
    except (ValueError, TypeError):
        confidence_f = None

    if confidence_f is not None and np.isfinite(confidence_f) and confidence_f < cfg.min_confidence:
        return "low_confidence"

    if cfg.min_alpha_lcb is not None:
        alpha_lcb = row.get("retrieval_alpha_lcb", row.get("retrieval_alpha_ci_low"))
        if alpha_lcb is None or pd.isna(alpha_lcb) or not np.isfinite(float(alpha_lcb)):
            return "missing_alpha_lcb"
        if float(alpha_lcb) < cfg.min_alpha_lcb:
            return "low_alpha_lcb"

    if cfg.min_absolute_return_lcb is not None:
        abs_lcb = row.get("retrieval_absolute_return_lcb", row.get("retrieval_absolute_return_ci_low"))
        if abs_lcb is None or pd.isna(abs_lcb) or not np.isfinite(float(abs_lcb)):
            return "missing_absolute_return_lcb"
        if float(abs_lcb) < cfg.min_absolute_return_lcb:
            return "low_absolute_return_lcb"

    if cfg.max_downside_cvar is not None:
        cvar = row.get("retrieval_downside_cvar")
        if cvar is None or pd.isna(cvar) or not np.isfinite(float(cvar)):
            return "missing_downside_cvar"
        if abs(float(cvar)) > cfg.max_downside_cvar:
            return "downside_cvar"

    if cfg.min_neighbor_count is not None:
        count = row.get("retrieval_neighbor_count", 0)
        if count is None or pd.isna(count) or int(count) < cfg.min_neighbor_count:
            return "insufficient_neighbors"

    if cfg.require_ood_pass and not bool(row.get("retrieval_ood_pass", False)):
        return "ood"

    return None


# Backward-compatible private alias for older experiment scripts.
_tradability_reason = tradability_reason


def _slipped(price: float, bps: float, side: str) -> float:
    mult = 1.0 + bps / 10000.0 if side == "buy" else 1.0 - bps / 10000.0
    return float(price) * mult


def run_long_only_backtest(
    signals: pd.DataFrame,
    config: PolicyConfig | None = None,
    score_col: str = "opportunity_score",
    exit_score_col: str | None = None,
    row_filter: Callable[[pd.Series, PolicyConfig, str], bool] | None = None,
    decision_log: list[dict] | None = None,
    exposure_controller: CausalExposureController | None = None,
    exposure_log: list[dict] | None = None,
    entry_allocation_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a next-bar long-only policy with optional per-opportunity entry sizing."""
    cfg = config or PolicyConfig()
    if cfg.top_k <= 0:
        raise ValueError("top_k must be positive.")
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    exit_score_col = exit_score_col or score_col

    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["ticker"] = frame["ticker"].astype(str)
    exec_price_col = _price_col(frame)
    mark_price_col = _close_col(frame)
    frame = frame.sort_values(["timestamp", "ticker"]).reset_index(drop=True)
    dates = sorted(frame["timestamp"].unique())
    by_date = {d: g.set_index("ticker") for d, g in frame.groupby("timestamp", sort=True)}
    filter_fn = row_filter or _tradable
    risk_ledger = TickerRiskLedger(
        RiskLedgerConfig(
            cooldown_sessions=cfg.cooldown_sessions,
            rolling_window_sessions=cfg.rolling_loss_window_sessions,
            quarantine_sessions=cfg.quarantine_sessions,
            max_stop_losses=cfg.max_stop_losses,
            cumulative_loss_limit=cfg.cumulative_loss_limit,
        )
    )

    cash = float(cfg.initial_capital)
    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    next_trade_id = 0

    for i, current_date in enumerate(dates):
        current = by_date[current_date]
        if i == 0:
            equity_rows.append(
                {
                    "timestamp": current_date,
                    "equity": cash,
                    "cash": cash,
                    "positions": 0,
                    "exposure": 0.0,
                    "target_exposure": 1.0,
                }
            )
            continue

        signal_date = dates[i - 1]
        signal_rows = by_date[signal_date]
        exits: list[tuple[str, str]] = []

        for ticker, pos in list(positions.items()):
            if ticker not in current.index:
                continue
            exec_price = float(current.loc[ticker, exec_price_col])
            if not np.isfinite(exec_price) or exec_price <= 0:
                continue
            hold_days = i - pos["entry_i"]
            raw_return = exec_price / pos["entry_price"] - 1.0
            sig = signal_rows.loc[ticker] if ticker in signal_rows.index else None
            score = float(sig.get(exit_score_col, np.nan)) if sig is not None else np.nan
            downside_key = "retrieval_calibrated_downside" if cfg.use_calibrated_downside else "retrieval_expected_downside"
            downside = float(sig.get(downside_key, sig.get("retrieval_expected_downside", np.nan))) if sig is not None else np.nan

            score_decay_blocked = (
                (cfg.holding_mode == "minimum_hold_21" and hold_days < 21)
                or (cfg.holding_mode == "fixed_63_diagnostic")
            )

            if raw_return <= -cfg.stop_loss:
                exits.append((ticker, "stop_loss"))
            elif hold_days >= cfg.max_hold_days:
                exits.append((ticker, "max_hold"))
            else:
                if not score_decay_blocked and hold_days >= cfg.min_hold_days:
                    if np.isfinite(score) and score < max(cfg.exit_min_score, pos["entry_exit_score"] * cfg.exit_score_fraction):
                        exits.append((ticker, "score_decay"))
                if hold_days >= cfg.min_hold_days:
                    if np.isfinite(downside) and abs(downside) > cfg.max_expected_downside:
                        exits.append((ticker, "risk_rise"))

        for ticker, reason in exits:
            if ticker not in positions or ticker not in current.index:
                continue
            pos = positions.pop(ticker)
            exit_px = _slipped(float(current.loc[ticker, exec_price_col]), cfg.slippage_bps, "sell")
            proceeds = pos["shares"] * exit_px
            cash += proceeds
            ret = exit_px / pos["entry_fill_price"] - 1.0
            trades.append(
                {
                    "ticker": ticker,
                    "trade_id": pos["trade_id"],
                    "entry_date": pos["entry_date"],
                    "exit_date": current_date,
                    "entry_price": pos["entry_fill_price"],
                    "exit_price": exit_px,
                    "shares": pos["shares"],
                    "cost_basis": pos["cost_basis"],
                    "entry_score": pos["entry_score"],
                    "entry_allocation_fraction": pos.get("entry_allocation_fraction", 1.0),
                    "exit_reason": reason,
                    "holding_days": i - pos["entry_i"],
                    "gross_return": float(current.loc[ticker, exec_price_col]) / pos["entry_price"] - 1.0,
                    "net_return": ret,
                    "pnl": proceeds - pos["cost_basis"],
                    "is_rebalance": False,
                }
            )
            if cfg.risk_guard_enabled:
                risk_ledger.register_exit(ticker, i, ret, reason)

        exposure_decision = (
            exposure_controller.decide(signal_rows, pd.DataFrame(equity_rows))
            if exposure_controller is not None
            else ExposureDecision(1.0, 1.0, 1.0, 1.0, None, 0.0, None, None, None, None, "disabled")
        )
        if exposure_log is not None:
            exposure_log.append({"timestamp": current_date, **exposure_decision.to_dict()})
        if exposure_controller is not None:
            cash = _trim_to_target_exposure(
                positions,
                current,
                exec_price_col,
                cash,
                exposure_decision,
                cfg,
                i,
                current_date,
                trades,
                exposure_controller.config.rebalance_threshold,
            )

        available_slots = cfg.top_k - len(positions)
        candidates = signal_rows.copy()
        if row_filter is None:
            evidence_reasons = candidates.apply(lambda r: tradability_reason(r, cfg, score_col), axis=1)
            if decision_log is not None:
                for ticker, reason in evidence_reasons.items():
                    if reason is not None:
                        sig_r = candidates.loc[ticker] if ticker in candidates.index else None
                        decision_log.append(
                            {
                                "timestamp": current_date,
                                "ticker": ticker,
                                "decision": "reject",
                                "reason": reason,
                                "relative_alpha_lcb": sig_r.get("retrieval_alpha_lcb", sig_r.get("retrieval_alpha_ci_low")) if sig_r is not None else None,
                                "absolute_return_lcb": sig_r.get("retrieval_absolute_return_lcb", sig_r.get("retrieval_absolute_return_ci_low")) if sig_r is not None else None,
                                "calibrated_downside": sig_r.get("retrieval_calibrated_downside") if cfg.use_calibrated_downside and sig_r is not None else (sig_r.get("retrieval_expected_downside") if sig_r is not None else None),
                                "confidence": sig_r.get("retrieval_confidence") if sig_r is not None else None,
                            }
                        )
            candidates = candidates[evidence_reasons.isna()]
        else:
            candidates = candidates[candidates.apply(lambda r: filter_fn(r, cfg, score_col), axis=1)]
        if cfg.risk_guard_enabled and not candidates.empty:
            blocked_reasons = pd.Series(
                {ticker: risk_ledger.blocked_reason(str(ticker), i) for ticker in candidates.index}
            )
            if decision_log is not None:
                for ticker, reason in blocked_reasons.items():
                    if reason is not None:
                        decision_log.append(
                            {"timestamp": current_date, "ticker": ticker, "decision": "reject", "reason": reason}
                        )
            candidates = candidates[blocked_reasons.isna()]

        if cfg.breadth_exposure_enabled:
            b_map = cfg.breadth_exposure_map or {0: 0.0, 1: 0.33, 2: 0.67, 3: 1.00}
            eligible_count = len(candidates)
            target_exp = float(b_map.get(eligible_count, b_map.get(3, 1.00) if eligible_count >= 3 else 0.0))
        else:
            target_exp = exposure_decision.target_exposure

        cutoff_passed = True
        if cfg.entry_cutoff_date is not None:
            cutoff_ts = pd.to_datetime(cfg.entry_cutoff_date, utc=True)
            if current_date > cutoff_ts:
                cutoff_passed = False

        candidates = candidates.sort_values(score_col, ascending=False)
        if entry_allocation_col is None:
            candidates = candidates.head(cfg.top_k)

        if cutoff_passed:
            for ticker, sig in candidates.iterrows():
                if available_slots <= 0:
                    break
                if ticker in positions or ticker not in current.index:
                    continue
                price = float(current.loc[ticker, exec_price_col])
                if not np.isfinite(price) or price <= 0:
                    continue
                allocation_fraction = _entry_allocation_fraction(sig, entry_allocation_col)
                if allocation_fraction <= 0.0:
                    if decision_log is not None:
                        decision_log.append(
                            {
                                "timestamp": current_date,
                                "ticker": ticker,
                                "decision": "reject",
                                "reason": "allocator_abstain",
                                "entry_allocation_fraction": allocation_fraction,
                            }
                        )
                    continue
                equity = cash + _positions_value(positions, current, exec_price_col)
                if exposure_controller is None:
                    allocation = min(cash, equity * target_exp / cfg.top_k * allocation_fraction)
                else:
                    current_gross = _positions_value(positions, current, exec_price_col)
                    target_gross = equity * target_exp
                    target_position = target_gross / cfg.top_k * allocation_fraction
                    allocation = min(cash, target_position, max(0.0, target_gross - current_gross))
                if allocation <= 0:
                    continue
                fill = _slipped(price, cfg.slippage_bps, "buy")
                shares = allocation / fill
                cash -= allocation
                positions[ticker] = {
                    "trade_id": next_trade_id,
                    "entry_i": i,
                    "entry_date": current_date,
                    "entry_price": price,
                    "entry_fill_price": fill,
                    "shares": shares,
                    "cost_basis": allocation,
                    "entry_score": float(sig[score_col]),
                    "entry_exit_score": float(sig.get(exit_score_col, sig[score_col])),
                    "entry_target_exposure": target_exp,
                    "entry_allocation_fraction": allocation_fraction,
                }
                next_trade_id += 1
                available_slots -= 1
                if decision_log is not None:
                    decision_log.append(
                        {
                            "timestamp": current_date,
                            "ticker": ticker,
                            "decision": "enter",
                            "reason": "accepted",
                            "target_exposure": target_exp,
                            "entry_allocation_fraction": allocation_fraction,
                            "relative_alpha_lcb": sig.get("retrieval_alpha_lcb", sig.get("retrieval_alpha_ci_low")),
                            "absolute_return_lcb": sig.get("retrieval_absolute_return_lcb", sig.get("retrieval_absolute_return_ci_low")),
                            "calibrated_downside": sig.get("retrieval_calibrated_downside") if cfg.use_calibrated_downside else sig.get("retrieval_expected_downside"),
                            "confidence": sig.get("retrieval_confidence"),
                        }
                    )
        if exposure_controller is not None:
            cash, capital_added = _increase_to_target_exposure(
                positions,
                current,
                exec_price_col,
                cash,
                exposure_decision,
                cfg,
                exposure_controller.config.rebalance_threshold,
            )
            if exposure_log is not None and exposure_log:
                exposure_log[-1]["capital_added"] = capital_added

        mark_value = _positions_value(positions, current, mark_price_col)
        mark_equity = cash + mark_value
        equity_rows.append(
            {
                "timestamp": current_date,
                "equity": mark_equity,
                "cash": cash,
                "positions": len(positions),
                "exposure": mark_value / mark_equity if mark_equity else 0.0,
                "target_exposure": target_exp if cfg.breadth_exposure_enabled else exposure_decision.target_exposure,
            }
        )

    if dates and positions:
        final_date = dates[-1]
        last_rows = frame.sort_values("timestamp").drop_duplicates("ticker", keep="last").set_index("ticker")
        for ticker, pos in list(positions.items()):
            if ticker not in last_rows.index:
                continue
            final_row = last_rows.loc[ticker]
            final_date = final_row["timestamp"]
            exit_px = _slipped(float(final_row[exec_price_col]), cfg.slippage_bps, "sell")
            proceeds = pos["shares"] * exit_px
            cash += proceeds
            trades.append(
                {
                    "ticker": ticker,
                    "trade_id": pos["trade_id"],
                    "entry_date": pos["entry_date"],
                    "exit_date": final_date,
                    "entry_price": pos["entry_fill_price"],
                    "exit_price": exit_px,
                    "shares": pos["shares"],
                    "cost_basis": pos["cost_basis"],
                    "entry_score": pos["entry_score"],
                    "entry_allocation_fraction": pos.get("entry_allocation_fraction", 1.0),
                    "exit_reason": "end_of_test",
                    "holding_days": len(dates) - 1 - pos["entry_i"],
                    "gross_return": float(final_row[exec_price_col]) / pos["entry_price"] - 1.0,
                    "net_return": exit_px / pos["entry_fill_price"] - 1.0,
                    "pnl": proceeds - pos["cost_basis"],
                    "is_rebalance": False,
                }
            )
            positions.pop(ticker)
        final_date = dates[-1]
        equity_rows.append(
            {
                "timestamp": final_date,
                "equity": cash,
                "cash": cash,
                "positions": 0,
                "exposure": 0.0,
                "target_exposure": 0.0,
            }
        )

    equity = pd.DataFrame(equity_rows).drop_duplicates("timestamp", keep="last")
    if not equity.empty:
        equity["return"] = equity["equity"].pct_change().fillna(0.0)
        peak = equity["equity"].cummax()
        equity["drawdown"] = equity["equity"] / peak - 1.0
    return pd.DataFrame(trades), equity.reset_index(drop=True)


def _positions_value(positions: dict[str, dict], rows: pd.DataFrame, price_col: str) -> float:
    total = 0.0
    for ticker, pos in positions.items():
        if ticker in rows.index:
            px = float(rows.loc[ticker, price_col])
            if np.isfinite(px):
                total += pos["shares"] * px
    return float(total)


def _entry_allocation_fraction(row: pd.Series, column: str | None) -> float:
    """Resolve a bounded entry fraction, failing closed when an allocator output is invalid."""
    if column is None:
        return 1.0
    value = row.get(column)
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(fraction):
        return 0.0
    return float(np.clip(fraction, 0.0, 1.0))


def _trim_to_target_exposure(
    positions: dict[str, dict],
    rows: pd.DataFrame,
    price_col: str,
    cash: float,
    decision: ExposureDecision,
    cfg: PolicyConfig,
    session_index: int,
    timestamp: pd.Timestamp,
    trades: list[dict],
    rebalance_threshold: float,
) -> float:
    """Proportionally trim existing positions when exposure exceeds the causal target."""
    if not positions:
        return cash
    gross = _positions_value(positions, rows, price_col)
    equity = cash + gross
    if equity <= 0 or gross <= 0:
        return cash
    occupied_fraction = min(len(positions), cfg.top_k) / cfg.top_k
    target_gross = equity * decision.target_exposure * occupied_fraction
    excess_fraction = max(0.0, gross - target_gross) / equity
    if excess_fraction < rebalance_threshold:
        return cash
    retained_fraction = float(np.clip(target_gross / gross, 0.0, 1.0))
    for ticker, pos in list(positions.items()):
        if ticker not in rows.index:
            continue
        price = float(rows.loc[ticker, price_col])
        if not np.isfinite(price) or price <= 0:
            continue
        sold_fraction = 1.0 - retained_fraction
        shares_sold = pos["shares"] * sold_fraction
        if shares_sold <= 1e-12:
            continue
        exit_px = _slipped(price, cfg.slippage_bps, "sell")
        proceeds = shares_sold * exit_px
        released_cost = pos["cost_basis"] * sold_fraction
        cash += proceeds
        trades.append(
            {
                "ticker": ticker,
                "trade_id": pos["trade_id"],
                "entry_date": pos["entry_date"],
                "exit_date": timestamp,
                "entry_price": pos["entry_fill_price"],
                "exit_price": exit_px,
                "shares": shares_sold,
                "cost_basis": released_cost,
                "entry_score": pos["entry_score"],
                "exit_reason": "exposure_rebalance",
                "holding_days": session_index - pos["entry_i"],
                "gross_return": price / pos["entry_price"] - 1.0,
                "net_return": exit_px / pos["entry_fill_price"] - 1.0,
                "pnl": proceeds - released_cost,
                "is_rebalance": True,
                "target_exposure": decision.target_exposure,
                "controller_reason": decision.reason,
            }
        )
        pos["shares"] *= retained_fraction
        pos["cost_basis"] *= retained_fraction
        if pos["shares"] <= 1e-12 or pos["cost_basis"] <= 1e-8:
            positions.pop(ticker, None)
    return cash


def _increase_to_target_exposure(
    positions: dict[str, dict],
    rows: pd.DataFrame,
    price_col: str,
    cash: float,
    decision: ExposureDecision,
    cfg: PolicyConfig,
    rebalance_threshold: float,
) -> tuple[float, float]:
    """Increase existing positions proportionally when exposure is below target."""
    if not positions or cash <= 0:
        return cash, 0.0
    gross = _positions_value(positions, rows, price_col)
    equity = cash + gross
    if equity <= 0 or gross <= 0:
        return cash, 0.0
    occupied_fraction = min(len(positions), cfg.top_k) / cfg.top_k
    target_gross = equity * decision.target_exposure * occupied_fraction
    shortfall = max(0.0, target_gross - gross)
    if shortfall / equity < rebalance_threshold:
        return cash, 0.0
    budget = min(cash, shortfall)
    eligible: list[tuple[str, float, float]] = []
    for ticker, pos in positions.items():
        if ticker not in rows.index:
            continue
        price = float(rows.loc[ticker, price_col])
        if np.isfinite(price) and price > 0:
            eligible.append((ticker, price, float(pos["shares"]) * price))
    eligible_gross = sum(value for _, _, value in eligible)
    if eligible_gross <= 0:
        return cash, 0.0
    spent = 0.0
    for ticker, price, market_value in eligible:
        allocation = min(cash, budget * market_value / eligible_gross)
        if allocation <= 0:
            continue
        pos = positions[ticker]
        fill = _slipped(price, cfg.slippage_bps, "buy")
        added_shares = allocation / fill
        old_shares = float(pos["shares"])
        new_shares = old_shares + added_shares
        raw_notional = old_shares * float(pos["entry_price"]) + added_shares * price
        pos["shares"] = new_shares
        pos["cost_basis"] = float(pos["cost_basis"]) + allocation
        pos["entry_fill_price"] = pos["cost_basis"] / new_shares
        pos["entry_price"] = raw_notional / new_shares
        cash -= allocation
        spent += allocation
    return cash, float(spent)


def compute_backtest_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
    extra: dict | None = None,
) -> dict:
    metrics = dict(extra or {})
    if equity_curve.empty:
        metrics.update({"total_return": 0.0, "trade_count": 0})
        return metrics

    eq = equity_curve.copy()
    total_return = float(eq["equity"].iloc[-1] / initial_capital - 1.0)
    periods = max(len(eq) - 1, 1)
    annualized = float((1.0 + total_return) ** (252.0 / periods) - 1.0)
    daily = eq["return"].to_numpy(dtype=float) if "return" in eq else np.array([])
    sharpe = _ratio(daily.mean(), daily.std(ddof=1), sqrt(252)) if daily.size > 1 else 0.0
    downside = daily[daily < 0]
    sortino = _ratio(daily.mean(), downside.std(ddof=1), sqrt(252)) if downside.size > 1 else 0.0
    max_dd = float(eq["drawdown"].min()) if "drawdown" in eq else 0.0

    metrics.update(
        {
            "total_return": total_return,
            "annualized_return": annualized,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": annualized / abs(max_dd) if max_dd < 0 else None,
            "exposure": float(eq["exposure"].mean()) if "exposure" in eq else 0.0,
        }
    )

    if trades.empty:
        metrics.update(
            {
                "trade_count": 0,
                "win_rate": 0.0,
                "average_trade_return": 0.0,
                "median_trade_return": 0.0,
                "profit_factor": None,
                "average_holding_days": 0.0,
                "turnover": 0.0,
                "capital_efficiency": 0.0,
            }
        )
        return metrics

    returns = trades["net_return"].to_numpy(dtype=float)
    pnl = trades["pnl"].to_numpy(dtype=float)
    wins = pnl[pnl > 0].sum()
    losses = pnl[pnl < 0].sum()
    mean_exposure = float(eq["exposure"].mean()) if "exposure" in eq else 0.0
    metrics.update(
        {
            "trade_count": int(trades["trade_id"].nunique()) if "trade_id" in trades else int(len(trades)),
            "execution_count": int(len(trades)),
            "rebalance_count": int(trades.get("is_rebalance", pd.Series(False, index=trades.index)).fillna(False).sum()),
            "win_rate": float(np.mean(returns > 0)),
            "average_trade_return": float(np.mean(returns)),
            "median_trade_return": float(np.median(returns)),
            "profit_factor": float(wins / abs(losses)) if losses < 0 else None,
            "average_holding_days": float(trades["holding_days"].mean()),
            "turnover": float(trades["cost_basis"].sum() / initial_capital) if "cost_basis" in trades else None,
            "capital_efficiency": total_return / mean_exposure if mean_exposure > 1e-6 else None,
        }
    )
    return metrics


def equal_weight_baseline(signals: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    if signals.empty:
        return {"equal_weight_baseline_return": 0.0}
    frame = signals.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    price_col = _close_col(frame)
    pivot = frame.pivot_table(index="timestamp", columns="ticker", values=price_col, aggfunc="last").sort_index()
    daily = pivot.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).mean(axis=1).fillna(0.0)
    curve = initial_capital * (1.0 + daily).cumprod()
    total = float(curve.iloc[-1] / initial_capital - 1.0) if not curve.empty else 0.0
    return {"equal_weight_baseline_return": total}


def _ratio(mean: float, std: float, scale: float) -> float:
    if not np.isfinite(std) or std <= 1e-12:
        return 0.0
    return float(mean / std * scale)
