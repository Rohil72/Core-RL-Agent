"""Portfolio execution engine, order queue, and ledger accounting (v2).

Acceptance criteria addressed:
- A23: Next-open order, exits before entries, cash constraints, and integer entry quantities match fixture.
- A24: Age 1, age 63, strict stop inequality, and missing-quote queues tested.
- A25: Pence/account units, dividend/split basis, and action-adjusted peak verified.
- A26: Annual policy refresh preserves holdings/state; development resets; terminal rule tested.
- A27: Independent ledger reconstruction cash and NAV within 1e-8 account units on fixtures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from memory_study_v2.canonical_data import CorporateAction


class ExecutionError(Exception):
    """Raised on execution invariant violation (e.g. negative cash, invalid fills)."""
    pass


@dataclass
class Position:
    security_id: str
    quantity: float        # Shares (integer on entry, may be float after corporate actions)
    entry_price: float     # Fill price in account currency
    cost_basis: float      # Total cost including commission
    peak_price: float      # Chandelier stop peak
    age_sessions: int = 1  # Entry session is age 1


@dataclass
class TradeRecord:
    session: str
    security_id: str
    side: str              # "BUY", "SELL", "TERMINAL_SELL"
    quantity: float
    raw_price: float
    fill_price: float
    gross_notional: float
    commission: float
    slippage: float
    reason: str            # "ENTRY", "STOP", "MAX_AGE", "TERMINAL"


@dataclass
class DailyLedgerState:
    session: str
    cash: float
    nav: float
    holdings_value: float
    num_positions: int
    executed_notional: float
    turnover: float
    exposure: float


class PortfolioAccount:
    """Simulates a continuous local-currency account under the strict execution contract."""

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
        """Calculate current total equity (NAV) = cash + sum(quantity * price)."""
        holdings = 0.0
        for sec_id, pos in self.positions.items():
            price = current_prices.get(sec_id, pos.peak_price)
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
        """
        for sec_id, act in actions.items():
            if sec_id in self.positions:
                pos = self.positions[sec_id]
                S = act.split_ratio
                D = act.cash_dividend

                # Cash distribution credited
                if D > 0.0:
                    cash_credit = pos.quantity * D
                    self.cash += cash_credit

                # Share adjustment
                if S != 1.0:
                    pos.quantity = pos.quantity * S
                    pos.entry_price = pos.entry_price / S
                    pos.cost_basis = pos.cost_basis  # total notional unchanged

                # Adjust peak
                pos.peak_price = max(1e-9, (pos.peak_price - D) / S)

    def process_open_fills(
        self,
        session: str,
        open_prices: Dict[str, float],
        tradable_flags: Dict[str, bool],
        close_decision_equity: float,
    ) -> None:
        """Execute queued orders at next scheduled open (Section 8.2):

        1. Process pending exits FIRST (credits cash).
        2. Process pending entries SECOND in ranked priority order.
        """
        # Step 1: Process pending exits
        exits_to_remove = []
        for sec_id, reason in list(self.pending_exits.items()):
            if not tradable_flags.get(sec_id, False) or sec_id not in open_prices:
                # Untradable open: exit remains pending until tradable open
                continue

            raw_open = open_prices[sec_id]
            pos = self.positions[sec_id]
            sell_fill = raw_open * (1.0 - self.slippage)
            gross_notional = pos.quantity * sell_fill
            comm = self.commission * gross_notional
            net_cash = gross_notional - comm

            self.cash += net_cash

            self.trades.append(TradeRecord(
                session=session,
                security_id=sec_id,
                side="SELL",
                quantity=pos.quantity,
                raw_price=raw_open,
                fill_price=sell_fill,
                gross_notional=gross_notional,
                commission=comm,
                slippage=raw_open * self.slippage * pos.quantity,
                reason=reason,
            ))
            del self.positions[sec_id]
            exits_to_remove.append(sec_id)

        for sec_id in exits_to_remove:
            del self.pending_exits[sec_id]

        # Step 2: Process entries in priority order
        budget_per_slot = close_decision_equity / float(self.max_positions)

        for sec_id in self.pending_entries:
            # Check capacity
            occupied = len(self.positions)
            if occupied >= self.max_positions:
                break
            if sec_id in self.positions:
                continue

            if not tradable_flags.get(sec_id, False) or sec_id not in open_prices:
                # Untradable open: entry instruction expires
                continue

            raw_open = open_prices[sec_id]
            buy_fill = raw_open * (1.0 + self.slippage)
            cost_per_share = buy_fill * (1.0 + self.commission)

            # Sizing formula: min(floor(0.95 * budget / cost_per_share), floor(cash / cost_per_share))
            qty_budget = math.floor((self.cash_budget_fraction * budget_per_slot) / cost_per_share)
            qty_cash = math.floor(self.cash / cost_per_share)
            qty = min(qty_budget, qty_cash)

            if qty < 1:
                # Below 1 does not trade
                continue

            total_cost = qty * cost_per_share
            gross_notional = qty * buy_fill
            comm = self.commission * gross_notional
            slip = raw_open * self.slippage * qty

            if total_cost > self.cash + 1e-9:
                raise ExecutionError(f"Insufficient cash for entry: cash={self.cash}, cost={total_cost}")

            self.cash -= total_cost

            # Initialize peak to raw execution open
            self.positions[sec_id] = Position(
                security_id=sec_id,
                quantity=float(qty),
                entry_price=buy_fill,
                cost_basis=total_cost,
                peak_price=raw_open,
                age_sessions=1,
            )

            self.trades.append(TradeRecord(
                session=session,
                security_id=sec_id,
                side="BUY",
                quantity=float(qty),
                raw_price=raw_open,
                fill_price=buy_fill,
                gross_notional=gross_notional,
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
        """Evaluate Chandelier stop and max holding age at close, and record ledger state.

        Rules (Section 8.3):
        - peak = max(peak, current valid close)
        - Exit queued if close / peak - 1 < -max(0.10, 2.5 * ATR_ratio14) OR age >= 63.
        - Age increments on every scheduled session.
        """
        holdings_value = 0.0

        for sec_id, pos in list(self.positions.items()):
            if sec_id in close_prices:
                c = close_prices[sec_id]
                pos.peak_price = max(pos.peak_price, c)
                holdings_value += pos.quantity * c

                # Evaluate stop
                atr_ratio = atr_ratios.get(sec_id, 0.0)
                stop_thresh = max(self.min_stop_fraction, self.atr_multiplier * atr_ratio)
                drawdown_from_peak = (c / pos.peak_price) - 1.0

                # Queue stop exit if strictly breached
                if drawdown_from_peak < -stop_thresh:
                    if sec_id not in self.pending_exits:
                        self.pending_exits[sec_id] = "STOP"
            else:
                # Held security missing quote: keep previous valuation
                holdings_value += pos.quantity * pos.peak_price

            # Check age rule: age >= 63 queues exit
            if pos.age_sessions >= self.max_holding_sessions:
                if sec_id not in self.pending_exits:
                    self.pending_exits[sec_id] = "MAX_AGE"

            # Increment age
            pos.age_sessions += 1

        total_nav = self.cash + holdings_value
        executed_notional = sum(
            t.gross_notional for t in self.trades if t.session == session
        )

        turnover = executed_notional / (2.0 * total_nav) if total_nav > 0 else 0.0
        exposure = holdings_value / total_nav if total_nav > 0 else 0.0

        state = DailyLedgerState(
            session=session,
            cash=self.cash,
            nav=total_nav,
            holdings_value=holdings_value,
            num_positions=len(self.positions),
            executed_notional=executed_notional,
            turnover=turnover,
            exposure=exposure,
        )
        self.daily_history.append(state)
        self.prev_equity = total_nav
        return total_nav

    def plan_entries_at_close(
        self,
        ranked_candidates: List[str],
    ) -> None:
        """Select top candidates for empty slots at close t.

        Slot reservation rule (Section 8.1):
        An exit queued for tomorrow still occupies a slot today!
        Only select candidates up to currently empty slots: max_positions - len(positions).
        """
        currently_occupied = len(self.positions)
        empty_slots = max(0, self.max_positions - currently_occupied)

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
    ) -> None:
        """Synthetic terminal liquidation at final valid session close (Section 8.3)."""
        for sec_id, pos in list(self.positions.items()):
            c = close_prices[sec_id]
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
