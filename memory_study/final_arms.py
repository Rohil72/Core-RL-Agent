"""
Missing Arms and Signal Generators for the Final Comparative Study.

Implements:
1. MEM_RANDOM: Random-priority constrained retrieval across 3 seeds ([1001, 1002, 1003])
   executed as 3 independent policy realizations (not forecast-averaged).
2. HIST_PRIOR: Constant historical prior mean (c = 0.066782) ranked by inverse volatility c / (v + 1e-4).
3. BENCH_MOMENTUM_21: 21-day price momentum signal ranked by volatility-scaled score mom_21 / (v + 1e-4).
4. BENCH_EQUAL_WEIGHT: Passive equal-weight buy-and-hold portfolio across universe tickers with institutional fees.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def retrieve_mem_random_seeds(
    query_tickers: np.ndarray,
    mem_tickers: np.ndarray,
    mem_sessions: np.ndarray,
    mem_returns: np.ndarray,
    retrieval_seeds: List[int] = [1001, 1002, 1003],
    k: int = 25,
    max_per_ticker: int = 3,
    min_separation: int = 21,
) -> Dict[int, np.ndarray]:
    """
    Computes separate random-priority eligible retrieval predictions for each seed.
    DO NOT average across seeds: each seed is executed as its own independent policy realization.
    """
    n_queries = len(query_tickers)
    n_mem = len(mem_tickers)
    seed_preds: Dict[int, np.ndarray] = {}

    for s in retrieval_seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(n_mem)
        preds = np.zeros(n_queries, dtype=np.float32)

        for q_i, q_tkr in enumerate(query_tickers):
            accepted = []
            ticker_counts: Dict[str, int] = {}
            ticker_sessions: Dict[str, List[int]] = {}

            for idx in perm:
                t = mem_tickers[idx]
                if t == q_tkr:
                    continue
                if ticker_counts.get(t, 0) >= max_per_ticker:
                    continue
                sess = mem_sessions[idx]
                past_s = ticker_sessions.get(t, [])
                if any(abs(sess - s_prev) < min_separation for s_prev in past_s):
                    continue

                accepted.append(idx)
                ticker_counts[t] = ticker_counts.get(t, 0) + 1
                if t not in ticker_sessions:
                    ticker_sessions[t] = []
                ticker_sessions[t].append(sess)

                if len(accepted) == k:
                    break

            if len(accepted) == k:
                preds[q_i] = float(np.mean(mem_returns[accepted]))
            else:
                preds[q_i] = float(np.mean(mem_returns)) if len(mem_returns) > 0 else 0.0

        seed_preds[s] = preds

    return seed_preds


def compute_hist_prior_predictions(
    n_queries: int,
    mem_returns: np.ndarray,
) -> np.ndarray:
    """
    HIST_PRIOR: Constant unweighted mean return over the pre-2021 historical pool.
    When scored as c / (v + 1e-4), this ranks tickers strictly by inverse volatility.
    """
    c = float(np.mean(mem_returns))
    return np.full(n_queries, c, dtype=np.float32)


def compute_momentum_21_matrix(
    market_dfs: Dict[str, pd.DataFrame],
    tickers: List[str],
    calendar_dates: List[str],
) -> np.ndarray:
    """
    Computes 21-session momentum for each ticker on each calendar date:
    mom_21[t, i] = (P_close[t] - P_close[t-21]) / P_close[t-21].
    Uses full historical series from market_dfs to ensure no warm-up truncation.
    """
    n_sessions = len(calendar_dates)
    n_tickers = len(tickers)
    mom_matrix = np.zeros((n_sessions, n_tickers), dtype=np.float32)

    for t_i, tkr in enumerate(tickers):
        df_t = market_dfs[tkr]
        close_s = df_t["close"]
        mom_s = close_s.pct_change(21).fillna(0.0)

        for d_i, dt_str in enumerate(calendar_dates):
            dt = pd.to_datetime(dt_str)
            if dt in mom_s.index:
                mom_matrix[d_i, t_i] = float(mom_s.loc[dt])
            else:
                # Forward fill
                sub = mom_s.loc[mom_s.index <= dt]
                mom_matrix[d_i, t_i] = float(sub.iloc[-1]) if len(sub) > 0 else 0.0

    return mom_matrix


def simulate_equal_weight_benchmark(
    market: str,
    tickers: List[str],
    calendar_dates: List[str],
    price_open: np.ndarray,
    price_close: np.ndarray,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0010,
    cash_buffer: float = 0.95,
) -> Dict[str, Any]:
    """
    Simulates a passive equal-weight buy-and-hold portfolio across universe tickers.
    Enforces exact institutional fee accounting and penny reconciliation identity.
    """
    n_sessions = len(calendar_dates)
    n_tickers = len(tickers)
    budget_per_ticker = (initial_capital * cash_buffer) / n_tickers

    # Day 0: Open positions
    positions = []
    total_entry_fees = 0.0
    cash = initial_capital

    for t_i, tkr in enumerate(tickers):
        p_open_0 = float(price_open[0, t_i])
        shares = int(np.floor(budget_per_ticker / p_open_0)) if p_open_0 > 0 else 0
        gross_cost = shares * p_open_0
        entry_fee = gross_cost * fee_rate
        total_entry_fees += entry_fee
        cash -= (gross_cost + entry_fee)

        positions.append({
            "ticker_idx": t_i,
            "ticker": tkr,
            "shares": shares,
            "entry_price": p_open_0,
            "entry_fee": entry_fee,
        })

    portfolio_equity = np.zeros(n_sessions, dtype=np.float64)
    daily_returns = np.zeros(n_sessions, dtype=np.float64)

    for d_i in range(n_sessions):
        pos_val = sum(p["shares"] * float(price_close[d_i, p["ticker_idx"]]) for p in positions)
        eq = cash + pos_val
        portfolio_equity[d_i] = eq
        if d_i == 0:
            daily_returns[d_i] = (eq - initial_capital) / initial_capital
        else:
            daily_returns[d_i] = (eq - portfolio_equity[d_i - 1]) / portfolio_equity[d_i - 1]

    # Close out on final day for accounting ledger
    final_day_idx = n_sessions - 1
    terminal_exit_fees = 0.0
    cell_trades = []

    for p in positions:
        c_p = float(price_close[final_day_idx, p["ticker_idx"]])
        exit_fee = p["shares"] * c_p * fee_rate
        terminal_exit_fees += exit_fee
        gross_pnl = (c_p - p["entry_price"]) * p["shares"]
        net_pnl = gross_pnl - (p["entry_fee"] + exit_fee)

        tot_cost = p["entry_price"] * p["shares"] + p["entry_fee"]
        ret_pct = round(net_pnl / tot_cost, 4) if tot_cost > 0 else 0.0

        cell_trades.append({
            "trade_id": f"EW_{market}_{p['ticker']}",
            "market": market,
            "ticker": p["ticker"],
            "signal_date": calendar_dates[0],
            "entry_date": calendar_dates[0],
            "exit_date": calendar_dates[final_day_idx],
            "holding_days": n_sessions,
            "exit_reason": "calendar_end",
            "entry_price": round(p["entry_price"], 2),
            "exit_price": round(c_p, 2),
            "shares": p["shares"],
            "entry_fee": round(p["entry_fee"], 2),
            "exit_fee": round(exit_fee, 2),
            "realized_pnl": round(net_pnl, 2),
            "return_pct": ret_pct,
            "win": int(net_pnl > 0),
        })

    final_equity = float(portfolio_equity[-1])
    total_ledger_pnl = sum(t["realized_pnl"] for t in cell_trades)
    equity_delta = final_equity - initial_capital
    ledger_delta = total_ledger_pnl + terminal_exit_fees
    accounting_diff = abs(equity_delta - ledger_delta)
    penny_pass = accounting_diff <= 0.10

    tot_ret = (final_equity - initial_capital) / initial_capital
    ann_factor = 252.0 / float(n_sessions)
    ann_ret = (1.0 + tot_ret) ** ann_factor - 1.0

    daily_std = float(np.std(daily_returns))
    sharpe = float(np.mean(daily_returns) / (daily_std + 1e-9) * np.sqrt(252.0)) if daily_std > 1e-6 else 0.0

    cum_eq = portfolio_equity
    peaks = np.maximum.accumulate(cum_eq)
    drawdowns = (cum_eq - peaks) / peaks
    max_dd = float(np.min(drawdowns))

    wins = [t for t in cell_trades if t["win"] == 1]
    win_rate = len(wins) / len(cell_trades) if len(cell_trades) > 0 else 0.0

    return {
        "market": market,
        "final_equity": final_equity,
        "annualized_return": ann_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "total_trades": len(cell_trades),
        "turnover": 0.0,
        "avg_exposure": 1.0,
        "penny_reconciled": penny_pass,
        "accounting_discrepancy": round(accounting_diff, 4),
        "daily_equity": portfolio_equity,
        "daily_returns": daily_returns,
        "trade_ledger": cell_trades,
    }
