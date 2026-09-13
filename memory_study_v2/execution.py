"""Execution engine, portfolio accounting, order event lifecycle, and valuation (v2).

Acceptance criteria addressed:
- A23: Exits processed before entries; integer share sizing; cash not exceeded.
- A24: Max holding age 63 sessions; strict inequality for Chandelier stop.
- A25: Dividends credited before open; peak price adjusted; cash/NAV conserved.
- A26: Annual boundary preserves state; terminal liquidation closes all positions.
- A27: Reconstructed cash/NAV matches within 1e-6 tolerance.
- R05: Action-adjusted last valid mark for missing prices (not peak) + stale session counter;
       Immediate return for zero free slots in entry planning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class CorporateAction:
    split_ratio: float = 1.0      # S: old shares become old * S
    cash_dividend: float = 0.0    # D: cash dividend per pre-split share


@dataclass
class Position:
    security_id: str
    quantity: float
    cost_basis: float = 0.0
    peak_price: float = 0.0
    last_valid_price: float = 0.0
    entry_price: float = 0.0
    stale_sessions: int = 0
    age_sessions: int = 0

    def __post_init__(self):
        if self.entry_price > 0.0 and self.cost_basis == 0.0:
            self.cost_basis = self.entry_price
        elif self.cost_basis > 0.0 and self.entry_price == 0.0:
            self.entry_price = self.cost_basis
        if self.last_valid_price == 0.0:
            self.last_valid_price = self.cost_basis or self.peak_price


@dataclass
class TradeRecord:
    session: str
    security_id: str
    side: str                     # "BUY", "SELL_STOP", "SELL_AGE", "TERMINAL_SELL"
    quantity: float
    raw_price: float
    fill_price: float
    gross_notional: float
    commission: float
    slippage: float
    reason: str


@dataclass
class DailyLedgerState:
    session: str
    cash: float
    holdings_value: float
    total_nav: float
    daily_return: float
    turnover_notional: float
    num_positions: int


class PortfolioAccount:
    """Continuous local-currency trading account for a market realization."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_positions: int = 3,
        commission: float = 0.001,
        slippage: float = 0.0005,
        cash_budget_fraction: float = 0.95,
        atr_multiplier: float = 2.5,
        min_stop_fraction: float = 0.10,
        max_holding_sessions: int = 63,
    ):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.max_positions = max_positions
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.cash_budget_fraction = float(cash_budget_fraction)
        self.atr_multiplier = float(atr_multiplier)
        self.min_stop_fraction = float(min_stop_fraction)
        self.max_holding_sessions = max_holding_sessions

        self.positions: Dict[str, Position] = {}
        self.pending_exits: Dict[str, str] = {}    # sec_id -> reason
        self.pending_entries: List[str] = []       # ranked list of sec_ids

        self.trades: List[TradeRecord] = []
        self.daily_history: List[DailyLedgerState] = []
        self.prev_equity = float(initial_capital)

    def current_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate current total equity (NAV) = cash + sum(quantity * price).

        Fallback for missing prices is action-adjusted last_valid_price, NOT peak_price (R05).
        """
        holdings = 0.0
        for sec_id, pos in self.positions.items():
            price = current_prices.get(sec_id, pos.last_valid_price)
            holdings += pos.quantity * price
        return self.cash + holdings

    def handle_corporate_actions_before_open(
        self,
        session: str,
        actions: Dict[str, CorporateAction],
    ) -> None:
        """Apply corporate actions before open (Section 8.3):

        For existing holdings:
        - Credit old_quantity * D to cash.
        - Multiply quantity by S.
        - Divide per-share cost basis by S.
        - Adjust peak: max(1e-9, (old_peak - D) / S).
        - Adjust last_valid_price: max(1e-9, (old_last_valid - D) / S) (R05).
        """
        for sec_id, act in actions.items():
            if hasattr(act, "dividend_cash") and getattr(act, "dividend_cash", 0.0) != 0.0 and getattr(act, "cash_dividend", 0.0) == 0.0:
                raise ValueError(
                    f"Conflicting/deprecated field 'dividend_cash' found on action for {sec_id}. Use canonical 'cash_dividend'."
                )
            if sec_id in self.positions:
                pos = self.positions[sec_id]
                old_qty = pos.quantity
                S = act.split_ratio
                D = getattr(act, "cash_dividend", 0.0)

                # 1. Credit dividend cash
                if D > 0.0:
                    self.cash += old_qty * D

                # 2. Split adjustment (retain action-created fractional shares)
                if S > 0.0 and S != 1.0:
                    pos.quantity = float(old_qty * S)
                    pos.cost_basis = pos.cost_basis / S

                # 3. Peak adjustment
                adj_peak = (pos.peak_price - D) / S
                pos.peak_price = max(1e-9, adj_peak)

                # 4. Last valid price adjustment (R05)
                adj_last_valid = (pos.last_valid_price - D) / S
                pos.last_valid_price = max(1e-9, adj_last_valid)

    def process_open_fills(
        self,
        session: str,
        open_prices: Dict[str, float],
        tradable_flags: Dict[str, bool],
        yesterday_equity: Optional[float] = None,
        close_decision_equity: Optional[float] = None,
    ) -> None:
        eq = yesterday_equity if yesterday_equity is not None else (close_decision_equity if close_decision_equity is not None else self.cash)
        """Execute orders at session open (Section 8.2):

        Order of operations:
        1. Process pending exits first (freed cash is available for entries today).
        2. Process pending entries in rank order up to remaining budget.
        """
        # Step 1: Process pending exits
        for sec_id, reason in list(self.pending_exits.items()):
            if not tradable_flags.get(sec_id, False) or sec_id not in open_prices:
                # Untradable open: exit remains pending until next tradable session
                continue

            raw_open = open_prices[sec_id]
            pos = self.positions[sec_id]

            # Sell fill = open * (1 - slippage)
            sell_fill = raw_open * (1.0 - self.slippage)
            gross = pos.quantity * sell_fill
            comm = self.commission * gross
            net_cash = gross - comm

            self.cash += net_cash
            self.trades.append(TradeRecord(
                session=session,
                security_id=sec_id,
                side=f"SELL_{reason}",
                quantity=pos.quantity,
                raw_price=raw_open,
                fill_price=sell_fill,
                gross_notional=gross,
                commission=comm,
                slippage=raw_open * self.slippage * pos.quantity,
                reason=reason,
            ))
            del self.positions[sec_id]
            del self.pending_exits[sec_id]

        # Step 2: Process pending entries
        per_slot_budget = eq / float(self.max_positions)

        for sec_id in list(self.pending_entries):
            # Check slot availability
            if len(self.positions) >= self.max_positions:
                break

            if not tradable_flags.get(sec_id, False) or sec_id not in open_prices:
                # Untradable open: entry instructions expire immediately
                continue

            raw_open = open_prices[sec_id]
            if raw_open <= 0.0 or not np.isfinite(raw_open):
                continue

            buy_fill = raw_open * (1.0 + self.slippage)

            # Integer quantity sizing (Section 8.2):
            # Q = floor(0.95 * budget / [fill * (1 + f)])
            # bounded by available cash equivalent
            max_by_budget = int(math.floor(
                (self.cash_budget_fraction * per_slot_budget) / (buy_fill * (1.0 + self.commission))
            ))
            max_by_cash = int(math.floor(
                (self.cash_budget_fraction * self.cash) / (buy_fill * (1.0 + self.commission))
            ))
            quantity = min(max_by_budget, max_by_cash)

            if quantity < 1:
                # Cannot afford minimum 1 share
                continue

            gross = quantity * buy_fill
            comm = self.commission * gross
            total_cost = gross + comm

            if total_cost > self.cash:
                continue

            self.cash -= total_cost
            self.positions[sec_id] = Position(
                security_id=sec_id,
                quantity=quantity,
                cost_basis=buy_fill,
                peak_price=buy_fill,
                last_valid_price=buy_fill,
                stale_sessions=0,
                age_sessions=0,
            )

            slip = raw_open * self.slippage * quantity
            self.trades.append(TradeRecord(
                session=session,
                security_id=sec_id,
                side="BUY",
                quantity=quantity,
                raw_price=raw_open,
                fill_price=buy_fill,
                gross_notional=gross,
                commission=comm,
                slippage=slip,
                reason="ENTRY",
            ))

        # Clear processed entry instructions
        self.pending_entries = []

    def evaluate_close_stops_and_update_state(
        self,
        session: str,
        close_prices: Dict[str, float],
        atr_ratios: Dict[str, float],
    ) -> float:
        """Evaluate Chandelier stop and max holding age at close, and record ledger state (R05).

        Rules (Section 8.3):
        - If valid close present: peak = max(peak, valid close), last_valid = close, stale = 0.
        - If close missing: use action-adjusted last_valid_price (NOT peak) and increment stale_sessions.
        - Stop rule: close / peak - 1 < -max(0.10, 2.5 * ATR_ratio14) queues exit.
        - Age rule: age >= 63 sessions queues exit.
        """
        holdings_value = 0.0

        for sec_id, pos in list(self.positions.items()):
            has_valid_close = (
                sec_id in close_prices
                and np.isfinite(close_prices[sec_id])
                and close_prices[sec_id] > 0.0
            )

            if has_valid_close:
                c = close_prices[sec_id]
                pos.last_valid_price = c
                pos.stale_sessions = 0
                pos.peak_price = max(pos.peak_price, c)
                holdings_value += pos.quantity * c

                # Evaluate stop
                atr_ratio = atr_ratios.get(sec_id, 0.0)
                stop_thresh = max(self.min_stop_fraction, self.atr_multiplier * atr_ratio)
                drawdown_from_peak = (c / pos.peak_price) - 1.0

                if drawdown_from_peak < -stop_thresh:
                    if sec_id not in self.pending_exits:
                        self.pending_exits[sec_id] = "STOP"
            else:
                # Held security missing quote (R05):
                pos.stale_sessions += 1
                holdings_value += pos.quantity * pos.last_valid_price

            # Increment age (at close, position has completed another holding session)
            pos.age_sessions += 1

            # Check age rule: age >= 63 queues exit
            if pos.age_sessions >= self.max_holding_sessions:
                if sec_id not in self.pending_exits:
                    self.pending_exits[sec_id] = "MAX_AGE"

        total_nav = self.cash + holdings_value
        executed_notional = sum(
            t.gross_notional for t in self.trades if t.session == session
        )

        daily_ret = (total_nav / self.prev_equity) - 1.0 if self.prev_equity > 0.0 else 0.0
        self.prev_equity = total_nav

        state = DailyLedgerState(
            session=session,
            cash=self.cash,
            holdings_value=holdings_value,
            total_nav=total_nav,
            daily_return=daily_ret,
            turnover_notional=executed_notional,
            num_positions=len(self.positions),
        )
        self.daily_history.append(state)
        return total_nav

    def plan_entries_at_close(
        self,
        ranked_candidates: List[str],
    ) -> None:
        """Select top candidates for empty slots at close t (R05).

        Slot reservation rule (Section 8.1):
        An exit queued for tomorrow still occupies a slot today!
        If empty_slots <= 0, returns immediately without queuing entries.
        """
        currently_occupied = len(self.positions)
        empty_slots = self.max_positions - currently_occupied

        if empty_slots <= 0:
            self.pending_entries = []
            return

        eligible_entries = []
        for sec in ranked_candidates:
            if sec not in self.positions and sec not in eligible_entries:
                eligible_entries.append(sec)
            if len(eligible_entries) == empty_slots:
                break

        self.pending_entries = eligible_entries

    def execute_terminal_liquidation(
        self,
        session: str,
        close_prices: Dict[str, float],
    ) -> float:
        """Synthetic terminal liquidation at final valid session close (Section 8.3).

        Strictly requires valid close price for each held position.
        Deducts commission and slippage to determine final realized cash NAV.
        """
        for sec_id, pos in list(self.positions.items()):
            if sec_id not in close_prices or not np.isfinite(close_prices[sec_id]) or close_prices[sec_id] <= 0.0:
                raise ValueError(
                    f"Missing or non-positive terminal close price for held position '{sec_id}' at terminal session {session}."
                )
            c = float(close_prices[sec_id])
            sell_fill = c * (1.0 - self.slippage)
            gross = pos.quantity * sell_fill
            comm = self.commission * gross
            net_cash = gross - comm
            self.cash += net_cash

            self.trades.append(TradeRecord(
                session=session,
                security_id=sec_id,
                side="TERMINAL_SELL",
                quantity=pos.quantity,
                raw_price=c,
                fill_price=sell_fill,
                gross_notional=gross,
                commission=comm,
                slippage=c * self.slippage * pos.quantity,
                reason="TERMINAL",
            ))
            del self.positions[sec_id]

        final_nav = self.cash
        total_turnover = sum(t.gross_notional for t in self.trades if t.session == session)
        if self.daily_history and self.daily_history[-1].session == session:
            prev_eq = self.daily_history[-2].total_nav if len(self.daily_history) > 1 else self.initial_capital
            terminal_ret = (final_nav / prev_eq) - 1.0 if prev_eq > 0.0 else 0.0
            self.daily_history[-1] = DailyLedgerState(
                session=session,
                cash=final_nav,
                holdings_value=0.0,
                total_nav=final_nav,
                daily_return=terminal_ret,
                turnover_notional=total_turnover,
                num_positions=0,
            )
            self.prev_equity = final_nav
        else:
            prev_eq = self.daily_history[-1].total_nav if self.daily_history else self.initial_capital
            terminal_ret = (final_nav / prev_eq) - 1.0 if prev_eq > 0.0 else 0.0
            self.daily_history.append(DailyLedgerState(
                session=session,
                cash=final_nav,
                holdings_value=0.0,
                total_nav=final_nav,
                daily_return=terminal_ret,
                turnover_notional=total_turnover,
                num_positions=0,
            ))
            self.prev_equity = final_nav
        return final_nav
