"""
Execution Engine Adapter for Memory-Centric Equity Selection.

Encapsulates the verified literal 9-step execution state machine:
- Open: Pending exits processed first (cash += proceeds * 0.999)
- Open: Pending entries processed second (shares = floor(0.95 * budget / P_open))
- Close: Mark-to-market equity & peak tracking
- Close: Chandelier stop (dd < -max(0.10, 2.5 * ATR_ratio)) and 63-session horizon
- Close: Scoring and queueing top available unheld tickers for next open
- Final day: Mark-to-market ledger closeout and strict penny accounting audit
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


class SimulationEngine:
    """
    Simulates portfolio execution for a single sovereign market cell over a given period.
    Enforces exact penny accounting closure and institutional trade accounting.
    """
    def __init__(
        self,
        market: str,
        tickers: List[str],
        calendar_dates: List[str],
        price_open: np.ndarray,      # shape (n_sessions, n_tickers)
        price_close: np.ndarray,     # shape (n_sessions, n_tickers)
        atr_ratio_14: np.ndarray,    # shape (n_sessions, n_tickers)
        daily_vol: np.ndarray,       # shape (n_sessions, n_tickers)
        initial_capital: float = 100000.0,
        max_slots: int = 3,
        fee_rate: float = 0.0010,
        cash_buffer: float = 0.95,
        chandelier_mult: float = 2.5,
        chandelier_min: float = 0.10,
        max_holding_days: int = 63,
    ):
        self.market = market
        self.tickers = tickers
        self.dates = calendar_dates
        self.n_sessions = len(calendar_dates)
        self.n_tickers = len(tickers)

        self.price_open = price_open
        self.price_close = price_close
        self.atr_ratio = atr_ratio_14
        self.daily_vol = daily_vol

        self.initial_capital = initial_capital
        self.max_slots = max_slots
        self.fee_rate = fee_rate
        self.cash_buffer = cash_buffer
        self.chandelier_mult = chandelier_mult
        self.chandelier_min = chandelier_min
        self.max_holding_days = max_holding_days

    def run_simulation(
        self,
        score_matrix: np.ndarray,  # shape (n_sessions, n_tickers)
        run_tag: str = "SIM",
    ) -> Dict[str, Any]:
        """
        Executes simulation using the precomputed score matrix.
        Returns cell metrics, daily equity curve, trade ledger, and accounting verification.
        """
        cash = self.initial_capital
        portfolio_equity = np.zeros(self.n_sessions, dtype=np.float64)
        daily_returns = np.zeros(self.n_sessions, dtype=np.float64)
        market_exposure = np.zeros(self.n_sessions, dtype=np.float64)

        open_positions: List[Dict[str, Any]] = []
        cell_trades: List[Dict[str, Any]] = []
        pending_exits: List[Dict[str, Any]] = []
        pending_entries: List[Dict[str, Any]] = []
        trade_seq = 0
        entry_seq = 0

        for day_idx in range(self.n_sessions):
            cur_date_str = self.dates[day_idx]

            # Step 1 & 2: EXITS AT OPEN
            for ex in pending_exits:
                pos = ex["pos"]
                t_i = pos["ticker_idx"]
                exit_price = float(self.price_open[day_idx, t_i])
                gross_proceeds = pos["shares"] * exit_price
                exit_fee = gross_proceeds * self.fee_rate
                net_proceeds = gross_proceeds - exit_fee
                cash += net_proceeds

                tot_cost = pos["entry_price"] * pos["shares"] + pos["entry_fee"]
                realized_pnl = net_proceeds - tot_cost
                ret_pct = realized_pnl / tot_cost
                held_days = day_idx - pos["entry_idx"]

                trade_seq += 1
                cell_trades.append({
                    "trade_id": pos["trade_id"],
                    "market": self.market,
                    "ticker": pos["ticker"],
                    "signal_date": ex["signal_date"],
                    "entry_date": self.dates[pos["entry_idx"]],
                    "exit_date": cur_date_str,
                    "holding_days": int(held_days),
                    "exit_reason": ex["exit_reason"],
                    "entry_price": round(float(pos["entry_price"]), 2),
                    "exit_price": round(float(exit_price), 2),
                    "shares": int(pos["shares"]),
                    "entry_fee": round(float(pos["entry_fee"]), 2),
                    "exit_fee": round(float(exit_fee), 2),
                    "realized_pnl": round(float(realized_pnl), 2),
                    "return_pct": round(float(ret_pct), 4),
                    "win": int(realized_pnl > 0),
                })
                open_positions.remove(pos)
            pending_exits = []

            # Step 3 & 4: ENTRIES AT OPEN
            for ent in pending_entries:
                t_i = ent["ticker_idx"]
                open_p = float(self.price_open[day_idx, t_i])
                if open_p > 0:
                    target_cap = ent["allocated_capital"]
                    shares = int((target_cap * self.cash_buffer) / open_p)
                    if shares > 0:
                        cost = shares * open_p
                        fee = cost * self.fee_rate
                        if cash >= (cost + fee):
                            cash -= (cost + fee)
                            entry_seq += 1
                            open_positions.append({
                                "trade_id": f"TRD_{run_tag}_{self.market}_{entry_seq:04d}",
                                "ticker": self.tickers[t_i],
                                "ticker_idx": t_i,
                                "entry_idx": day_idx,
                                "entry_price": open_p,
                                "peak_price": open_p,
                                "shares": shares,
                                "entry_fee": fee,
                            })
            pending_entries = []

            # Step 5: MARK TO MARKET AT CLOSE
            pos_value = sum(p["shares"] * float(self.price_close[day_idx, p["ticker_idx"]]) for p in open_positions)
            for p in open_positions:
                p["peak_price"] = max(p["peak_price"], float(self.price_close[day_idx, p["ticker_idx"]]))

            tot_equity = cash + pos_value
            portfolio_equity[day_idx] = tot_equity
            daily_returns[day_idx] = (
                (tot_equity - self.initial_capital) / self.initial_capital
                if day_idx == 0
                else (tot_equity - portfolio_equity[day_idx - 1]) / portfolio_equity[day_idx - 1]
            )
            market_exposure[day_idx] = pos_value / tot_equity if tot_equity > 0 else 0.0

            # Step 6: CLOSE-BASED EXIT EVALUATION
            if day_idx < self.n_sessions - 1:
                for p in open_positions:
                    held_days = day_idx - p["entry_idx"] + 1
                    cur_p = float(self.price_close[day_idx, p["ticker_idx"]])
                    dd_peak = (cur_p - p["peak_price"]) / p["peak_price"]
                    atr_rat = float(self.atr_ratio[day_idx, p["ticker_idx"]])
                    stop_dist = max(self.chandelier_min, self.chandelier_mult * atr_rat)

                    is_stop = (dd_peak < -stop_dist) and (held_days >= 1)
                    is_max_horizon = held_days >= self.max_holding_days
                    if is_stop or is_max_horizon:
                        pending_exits.append({
                            "pos": p,
                            "signal_date": cur_date_str,
                            "exit_reason": "atr_chandelier_stop" if is_stop else "max_horizon",
                        })

                # Step 7 & 8: CANDIDATE SCORING & ORDER QUEUEING
                exiting_ids = {id(item["pos"]) for item in pending_exits}
                remaining_count = len([p for p in open_positions if id(p) not in exiting_ids])
                slots_open = self.max_slots - remaining_count

                if slots_open > 0 and day_idx < self.n_sessions - 5:
                    cap_per_slot = tot_equity / float(self.max_slots)
                    held_tickers = {p["ticker"] for p in open_positions if id(p) not in exiting_ids}
                    available_indices = [i for i in range(self.n_tickers) if self.tickers[i] not in held_tickers]

                    if available_indices:
                        cand_scores = score_matrix[day_idx, available_indices]
                        top_order = np.argsort(-cand_scores)
                        for chosen_c in top_order[:slots_open]:
                            real_ti = available_indices[chosen_c]
                            pending_entries.append({
                                "ticker_idx": real_ti,
                                "signal_date": cur_date_str,
                                "allocated_capital": cap_per_slot,
                            })

        # Terminal Day Accounting: Close out remaining positions in ledger
        final_day_idx = self.n_sessions - 1
        terminal_exit_fees = 0.0
        for p in open_positions:
            c_p = float(self.price_close[final_day_idx, p["ticker_idx"]])
            f_exit = p["shares"] * c_p * self.fee_rate
            terminal_exit_fees += f_exit
            gross_pnl = (c_p - p["entry_price"]) * p["shares"]
            pnl = gross_pnl - (p["entry_fee"] + f_exit)
            held_days = final_day_idx - p["entry_idx"]
            trade_seq += 1
            cell_trades.append({
                "trade_id": p["trade_id"],
                "market": self.market,
                "ticker": p["ticker"],
                "signal_date": self.dates[final_day_idx],
                "entry_date": self.dates[p["entry_idx"]],
                "exit_date": self.dates[final_day_idx],
                "holding_days": int(held_days),
                "exit_reason": "calendar_end",
                "entry_price": round(float(p["entry_price"]), 2),
                "exit_price": round(float(c_p), 2),
                "shares": int(p["shares"]),
                "entry_fee": round(float(p["entry_fee"]), 2),
                "exit_fee": round(float(f_exit), 2),
                "realized_pnl": round(float(pnl), 2),
                "return_pct": round(float(pnl / (p["entry_price"] * p["shares"] + p["entry_fee"])), 4),
                "win": int(pnl > 0),
            })

        # Audit Penny Reconciliation
        final_equity = float(portfolio_equity[-1])
        total_ledger_pnl = sum(t["realized_pnl"] for t in cell_trades)
        equity_delta = final_equity - self.initial_capital
        ledger_delta = total_ledger_pnl + terminal_exit_fees
        accounting_diff = abs(equity_delta - ledger_delta)
        penny_pass = accounting_diff <= 0.10

        # Performance Metrics
        tot_ret = (final_equity - self.initial_capital) / self.initial_capital
        ann_factor = 252.0 / float(self.n_sessions)
        ann_ret = (1.0 + tot_ret) ** ann_factor - 1.0

        daily_std = float(np.std(daily_returns))
        sharpe = float(np.mean(daily_returns) / (daily_std + 1e-9) * np.sqrt(252.0)) if daily_std > 1e-6 else 0.0

        cum_equity = portfolio_equity
        peaks = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - peaks) / peaks
        max_dd = float(np.min(drawdowns))

        wins = [t for t in cell_trades if t["win"] == 1]
        win_rate = len(wins) / len(cell_trades) if len(cell_trades) > 0 else 0.0

        # Turnover
        tot_traded_val = sum(t["shares"] * (t["entry_price"] + t["exit_price"]) for t in cell_trades)
        avg_equity = float(np.mean(portfolio_equity))
        turnover = (tot_traded_val / (2.0 * avg_equity)) * ann_factor if avg_equity > 0 else 0.0

        avg_exposure = float(np.mean(market_exposure))

        return {
            "market": self.market,
            "final_equity": final_equity,
            "annualized_return": ann_ret,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(cell_trades),
            "turnover": turnover,
            "avg_exposure": avg_exposure,
            "penny_reconciled": penny_pass,
            "accounting_discrepancy": round(accounting_diff, 4),
            "daily_equity": portfolio_equity,
            "daily_returns": daily_returns,
            "trade_ledger": cell_trades,
        }
