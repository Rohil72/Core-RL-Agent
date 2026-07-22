from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RiskLedgerConfig:
    """Monotonic ticker-level re-entry protection."""

    cooldown_sessions: int = 63
    rolling_window_sessions: int = 126
    quarantine_sessions: int = 126
    max_stop_losses: int = 2
    cumulative_loss_limit: float = -0.15


@dataclass
class TickerRiskState:
    """Execution-only state for one ticker."""

    cooldown_until: int = -1
    quarantine_until: int = -1
    exits: list[tuple[int, float, str]] = field(default_factory=list)


class TickerRiskLedger:
    """Tracks realized exits without feeding trade outcomes back into memory."""

    def __init__(self, config: RiskLedgerConfig | None = None) -> None:
        self.config = config or RiskLedgerConfig()
        self._states: dict[str, TickerRiskState] = {}

    def blocked_reason(self, ticker: str, session_index: int) -> str | None:
        """Return the active immutable entry block, if any."""
        state = self._states.get(str(ticker))
        if state is None:
            return None
        if session_index <= state.quarantine_until:
            return "quarantine"
        if session_index <= state.cooldown_until:
            return "cooldown"
        return None

    def register_exit(
        self,
        ticker: str,
        session_index: int,
        net_return: float,
        reason: str,
    ) -> None:
        """Register one realized exit and only move block expiries forward."""
        key = str(ticker)
        state = self._states.setdefault(key, TickerRiskState())
        state.exits.append((int(session_index), float(net_return), str(reason)))
        window_start = int(session_index) - self.config.rolling_window_sessions + 1
        state.exits = [record for record in state.exits if record[0] >= window_start]

        if reason == "stop_loss":
            state.cooldown_until = max(
                state.cooldown_until,
                int(session_index) + self.config.cooldown_sessions,
            )
        stops = sum(record[2] == "stop_loss" for record in state.exits)
        cumulative = sum(record[1] for record in state.exits)
        if stops >= self.config.max_stop_losses or cumulative <= self.config.cumulative_loss_limit:
            state.quarantine_until = max(
                state.quarantine_until,
                int(session_index) + self.config.quarantine_sessions,
            )

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Return a serializable state snapshot for reporting and tests."""
        return {
            ticker: {
                "cooldown_until": state.cooldown_until,
                "quarantine_until": state.quarantine_until,
                "exits": list(state.exits),
            }
            for ticker, state in self._states.items()
        }
