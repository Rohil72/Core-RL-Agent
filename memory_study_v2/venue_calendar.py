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
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Pinned SHA-256 of the bundled fixture so any accidental edit is detected.
# ---------------------------------------------------------------------------
_FIXTURE_SHA256 = "18bd392bc68f71d9f4a9eccf5b2bfdfc9f1e8b34250785066057b7a23157b330"

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


class VenueCalendar:
    """Authoritative NYSE/NASDAQ exchange session schedule (2010-2026).

    Constructed once from the version-pinned fixture CSV.  All ordinals,
    status classifications, and reindexed DataFrames come from this object.
    """

    def __init__(self, fixture_path: Optional[Path] = None) -> None:
        path = fixture_path or _FIXTURE_PATH
        self.sessions: List[str] = _load_sessions_from_fixture(path)
        self._session_set: Set[str] = set(self.sessions)
        # 0-based ordinal for each session date string
        self.ordinal: Dict[str, int] = {s: i for i, s in enumerate(self.sessions)}

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

        # Index existing data by session date for O(1) lookup
        df_indexed = security_df.copy()
        df_indexed[date_col] = df_indexed[date_col].astype(str)
        df_indexed = df_indexed.set_index(date_col)

        rows = []
        for sess in scheduled:
            if sess in df_indexed.index:
                row = df_indexed.loc[sess].copy()
                row["bar_status"] = "VALID"
            else:
                # Missing bar on an open session
                row = pd.Series({col: float("nan") for col in df_indexed.columns})
                row["bar_status"] = "MISSING"
            row["session_ordinal"] = self.ordinal[sess]
            row.name = sess
            rows.append(row)

        result = pd.DataFrame(rows)
        result.index.name = date_col
        result = result.reset_index().rename(columns={"index": date_col})
        # Ensure the session column is correctly named
        if "level_0" in result.columns:
            result = result.drop(columns=["level_0"])
        if date_col not in result.columns and result.index.name == date_col:
            result = result.reset_index()

        return result

    def invalidate_windows_with_missing(
        self,
        session_dates: List[str],
        status_per_session: Dict[str, str],
        window_size: int,
    ) -> List[bool]:
        """Return a boolean validity mask for origin sessions in *session_dates*.

        A session's window is invalid if *any* session in the look-back window
        of size *window_size* (including the origin) has status "MISSING".

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
            window_start = max(0, i - window_size + 1)
            window = session_dates[window_start : i + 1]
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
