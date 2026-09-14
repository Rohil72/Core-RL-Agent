"""Independent venue calendar for NYSE/NASDAQ exchange sessions (Finding 1 / C1).

Provides an authoritative, version-pinned schedule of US equity trading sessions
(2010-2026) derived from the bundled fixture CSV, completely independent of any
price files.  Satisfies the auditor requirement that session ordinals, 3-state
bar status, and reindexed DataFrames are all derived from this schedule—not from
the intersection of whichever price files happen to be present.

Three-state bar classification per security per session:
    "CLOSED"  — exchange was not open on this date (weekend / holiday)
    "VALID"   — exchange was open and the security has a valid bar
    "MISSING" — exchange was open but the security's bar is absent or invalid

Session ordinals are 0-based indices into the schedule list.  They are stable
across securities, so deleting one security's bar never shifts another
security's ordinal.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pinned SHA-256 of the bundled fixture so any accidental edit is detected.
# ---------------------------------------------------------------------------
_FIXTURE_SHA256 = "8939423a476cb2a588e6bf559f02329d324d32a2b5d58a00f68f84679160a878"

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "us_trading_sessions.csv"


def _load_sessions_from_fixture(path: Path = _FIXTURE_PATH) -> List[str]:
    """Load the ordered list of NYSE/NASDAQ trading sessions from the pinned CSV.

    Verifies the SHA-256 digest of the file against the pinned value so that
    any accidental modification is caught immediately.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Venue calendar fixture not found at '{path}'. "
            "The fixture is part of the repository at data/fixtures/us_trading_sessions.csv."
        )
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != _FIXTURE_SHA256:
        raise ValueError(
            f"Venue calendar fixture SHA-256 mismatch!\n"
            f"  Expected : {_FIXTURE_SHA256}\n"
            f"  Got      : {actual_sha}\n"
            f"  Path     : {path}\n"
            "The fixture has been modified. Regenerate it from the deterministic "
            "algorithm in scripts/generate_venue_calendar.py and update the pinned hash."
        )
    sessions: List[str] = []
    text = raw.decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        s = str(row["session"]).strip()
        if s:
            sessions.append(s)
    if not sessions:
        raise ValueError("Venue calendar fixture is empty.")
    return sessions


def _is_valid_bar_row(row: Any) -> bool:
    """Validate price and volume attributes of an existing bar row.

    Checks that close (or tr_close) is finite and positive, high/low/open/close
    bounds are consistent if present, and volume is finite and non-negative.
    """
    close_val = None
    for c_col in ("close", "tr_close"):
        if c_col in row and pd.notna(row[c_col]):
            try:
                close_val = float(row[c_col])
            except (ValueError, TypeError):
                return False
            break
    if close_val is None or not math.isfinite(close_val) or close_val <= 0.0:
        return False

    has_o = "open" in row and pd.notna(row["open"])
    has_h = "high" in row and pd.notna(row["high"])
    has_l = "low" in row and pd.notna(row["low"])
    if has_o and has_h and has_l:
        try:
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
        except (ValueError, TypeError):
            return False
        if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l)):
            return False
        if o <= 0.0 or h <= 0.0 or l <= 0.0:
            return False
        if h < l or h < o or h < close_val or l > o or l > close_val:
            return False

    if "volume" in row and pd.notna(row["volume"]):
        try:
            v = float(row["volume"])
        except (ValueError, TypeError):
            return False
        if not math.isfinite(v) or v < 0.0:
            return False

    return True


class VenueCalendar:
    """Authoritative NYSE/NASDAQ exchange session schedule (2010-2026).

    Constructed once from the version-pinned fixture CSV.  All ordinals,
    status classifications, and reindexed DataFrames come from this object.
    """

    def __init__(
        self,
        fixture_path: Optional[Path] = None,
        sessions: Optional[List[str]] = None,
    ) -> None:
        if sessions is not None:
            self.sessions = sorted(list(set(str(s).strip() for s in sessions if str(s).strip())))
        else:
            path = fixture_path or _FIXTURE_PATH
            self.sessions = _load_sessions_from_fixture(path)
        self._session_set: Set[str] = set(self.sessions)
        # 0-based ordinal for each session date string
        self.ordinal: Dict[str, int] = {s: i for i, s in enumerate(self.sessions)}
        self.is_inferred: bool = False
        self.fallback_source: Optional[str] = None

    @property
    def schedule_hash(self) -> str:
        """Deterministic SHA-256 hash of the session schedule."""
        return hashlib.sha256("\n".join(self.sessions).encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_open(self, date_str: str) -> bool:
        """Return True if the exchange was open on *date_str* (YYYY-MM-DD)."""
        return date_str in self._session_set

    def get_ordinal(self, date_str: str) -> int:
        """Return 0-based session ordinal for an open session date.

        Raises KeyError if the date is not a valid exchange session.
        """
        try:
            return self.ordinal[date_str]
        except KeyError:
            raise KeyError(
                f"'{date_str}' is not a scheduled exchange session. "
                "Use is_open() to check before calling get_ordinal()."
            )

    def sessions_in_range(self, start_inclusive: str, end_inclusive: str) -> List[str]:
        """Return ordered list of open sessions in [start_inclusive, end_inclusive]."""
        return [s for s in self.sessions if start_inclusive <= s <= end_inclusive]

    def bar_status(self, date_str: str, has_valid_bar: bool) -> str:
        """Classify a single session for a given security.

        Args:
            date_str:      YYYY-MM-DD session date.
            has_valid_bar: True if the security has a finite, positive bar for
                           this date (after all validation checks).

        Returns:
            "CLOSED"  — exchange not open (weekend/holiday).
            "VALID"   — exchange open and security has a valid bar.
            "MISSING" — exchange open but security bar is absent or invalid.
        """
        if not self.is_open(date_str):
            return "CLOSED"
        return "VALID" if has_valid_bar else "MISSING"

    def reindex_to_schedule(
        self,
        security_df: pd.DataFrame,
        date_range_inclusive: Tuple[str, str],
        date_col: str = "session",
    ) -> pd.DataFrame:
        """Reindex *security_df* onto the scheduled sessions in the given range.

        For each scheduled session in [start, end]:
          - If the security has a row, it is retained as-is.
          - If the security has no row, a row of NaN is inserted for numeric
            price/volume columns; the 'session' column is filled with the date.

        Also adds two columns:
          - ``bar_status``: "VALID" | "MISSING" (CLOSED sessions are not included
            since by definition the schedule only contains open sessions).
          - ``session_ordinal``: 0-based index from the venue calendar.

        Args:
            security_df:           DataFrame with a *date_col* column (YYYY-MM-DD strings).
            date_range_inclusive:  (start_date, end_date) tuple of YYYY-MM-DD strings.
            date_col:              Name of the date column in *security_df*.

        Returns:
            New DataFrame with one row per scheduled session in range, sorted
            by session date ascending.
        """
        start, end = date_range_inclusive
        scheduled = self.sessions_in_range(start, end)
        if not scheduled:
            return pd.DataFrame(columns=list(security_df.columns) + ["bar_status", "session_ordinal"])

        sched_df = pd.DataFrame({date_col: scheduled})
        sched_df["session_ordinal"] = [self.ordinal[s] for s in scheduled]

        sec_df = security_df.copy()
        sec_df[date_col] = sec_df[date_col].astype(str)
        sec_df = sec_df.drop_duplicates(subset=[date_col], keep="first")

        result = pd.merge(sched_df, sec_df, on=date_col, how="left")

        # Vectorized bar validation
        is_valid = np.ones(len(result), dtype=bool)

        c_col = "close" if "close" in result.columns else ("tr_close" if "tr_close" in result.columns else None)
        if c_col is not None:
            c_vals = pd.to_numeric(result[c_col], errors="coerce").to_numpy(dtype=float)
            is_valid &= np.isfinite(c_vals) & (c_vals > 0.0)
        else:
            is_valid[:] = False

        if all(c in result.columns for c in ("open", "high", "low")):
            o_vals = pd.to_numeric(result["open"], errors="coerce").to_numpy(dtype=float)
            h_vals = pd.to_numeric(result["high"], errors="coerce").to_numpy(dtype=float)
            l_vals = pd.to_numeric(result["low"], errors="coerce").to_numpy(dtype=float)
            finite_ohl = np.isfinite(o_vals) & np.isfinite(h_vals) & np.isfinite(l_vals)
            pos_ohl = (o_vals > 0.0) & (h_vals > 0.0) & (l_vals > 0.0)
            bounds_ok = (h_vals >= l_vals) & (h_vals >= o_vals) & (h_vals >= c_vals) & (l_vals <= o_vals) & (l_vals <= c_vals)
            is_valid &= finite_ohl & pos_ohl & bounds_ok

        if "volume" in result.columns:
            v_vals = pd.to_numeric(result["volume"], errors="coerce").to_numpy(dtype=float)
            valid_vol = np.isfinite(v_vals) & (v_vals >= 0.0)
            is_valid &= valid_vol

        result["bar_status"] = np.where(is_valid, "VALID", "MISSING")
        return result

    def invalidate_windows_with_missing(
        self,
        session_dates: List[str],
        status_per_session: Dict[str, str],
        window_size: int,
    ) -> List[bool]:
        """Return a boolean validity mask for origin sessions in *session_dates*.

        A session's window is invalid if:
        1. It does not have at least *window_size* sessions of preceding history.
        2. *Any* session in the look-back window of size *window_size*
           (including the origin) has status "MISSING".

        Args:
            session_dates:      Ordered list of session dates (YYYY-MM-DD).
            status_per_session: Mapping from session date → bar_status string.
            window_size:        Number of sessions in the look-back window.

        Returns:
            List[bool] of length len(session_dates).  True = window is valid.
        """
        n = len(session_dates)
        valid = []
        for i in range(n):
            if i < window_size - 1:
                # Lookback history is incomplete (< window_size sessions)
                valid.append(False)
                continue
            window = session_dates[i - window_size + 1 : i + 1]
            has_missing = any(status_per_session.get(s, "MISSING") == "MISSING" for s in window)
            valid.append(not has_missing)
        return valid


# ---------------------------------------------------------------------------
# Module-level convenience singleton (lazy-loaded)
# ---------------------------------------------------------------------------
_GLOBAL_CALENDAR: Optional[VenueCalendar] = None


def get_venue_calendar() -> VenueCalendar:
    """Return the module-level VenueCalendar singleton (loaded once)."""
    global _GLOBAL_CALENDAR
    if _GLOBAL_CALENDAR is None:
        _GLOBAL_CALENDAR = VenueCalendar()
    return _GLOBAL_CALENDAR


def get_market_venue_calendar(
    market: str,
    custom_calendars: Optional[Dict[str, VenueCalendar]] = None,
    fixture_dir: Optional[Path] = None,
    data_cache_dir: Optional[Path] = None,
    execution_mode: str = "production",
) -> VenueCalendar:
    """Resolve authoritative venue calendar for a specific market.

    In production mode:
    Requires an explicitly supplied, validated calendar for every requested market.
    If fixture or custom calendar is missing, raises ValueError identifying the missing market.
    Automatic parquet-derived inference and US-substitute fallbacks are strictly disabled.

    In pilot mode:
    Permissive fallbacks are permitted but explicitly recorded on the returned calendar.
    """
    m_clean = str(market).strip().upper()
    if custom_calendars:
        for k, v in custom_calendars.items():
            if str(k).strip().upper() == m_clean:
                return v

    f_dir = fixture_dir or (_FIXTURE_PATH.parent)
    cand_csv = f_dir / f"{m_clean.lower()}_trading_sessions.csv"

    if cand_csv.exists():
        sessions: List[str] = []
        with open(cand_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = str(row.get("session", "")).strip()
                if s:
                    sessions.append(s)
        cal = VenueCalendar(sessions=sessions)
        return cal

    if m_clean == "US" and _FIXTURE_PATH.exists():
        return get_venue_calendar()

    # Missing authoritative calendar fixture
    if execution_mode == "production":
        raise ValueError(
            f"Missing authoritative venue calendar for market '{m_clean}' (expected fixture '{cand_csv}'). "
            "In production mode, every requested market requires an explicitly supplied, validated calendar fixture. "
            "Parquet date inference and US-substitute fallbacks are strictly disabled."
        )

    # Permissive pilot fallbacks (explicitly recorded)
    c_dir = data_cache_dir or (_FIXTURE_PATH.parents[1] / "data" / "cache" / "ohlcv")
    if c_dir.exists():
        all_dates: Set[str] = set()
        for pq_path in c_dir.glob(f"{m_clean}_*.parquet"):
            try:
                df = pd.read_parquet(pq_path, columns=["date"])
                all_dates.update(df["date"].astype(str).tolist())
            except Exception:
                continue
        if all_dates:
            cal = VenueCalendar(sessions=sorted(list(all_dates)))
            cal.is_inferred = True
            cal.fallback_source = f"parquet_date_inference:{m_clean}"
            return cal

    cal = VenueCalendar(sessions=get_venue_calendar().sessions)
    cal.is_inferred = True
    cal.fallback_source = f"us_calendar_substitute:{m_clean}"
    return cal
