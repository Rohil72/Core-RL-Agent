"""Unified sample-index builder and partition filtering (Finding 3 / F3).

Guarantees:
1. One shared sample-index calculation used by both manifest generator and model loaders.
2. Evaluates input_window_valid based on actual feature warmup (252 bars) and representation
   window (252 bars, totaling 503 contiguous valid preceding bars) on the VenueCalendar.
3. Strict session t exclusion: input window strictly spans bars t-503 through t-1.
4. Boundaries and target maturity are applied AFTER input-window validity calculation.
5. Evaluates evaluation queries independently of future targets; scored-evaluation queries
   are tracked as the subset with mature forward returns.
6. Persisted sample IDs are consumed by loaders to ensure 100% population identity.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from memory_study_v2.folds import FoldBoundaries, FoldPartitions
from memory_study_v2.venue_calendar import VenueCalendar, get_venue_calendar


@dataclass
class SampleIndexRecord:
    """Canonical sample index record per security and decision session."""
    query_id: str
    security_id: str
    session: str
    session_ordinal: int
    input_window_valid: bool
    target_63_valid: bool
    target_63_available_at: str
    bank_126_valid: bool
    bank_126_available_at: str
    exclusion_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_security_sample_index(
    security_id: str,
    bars_df: pd.DataFrame,
    venue_calendar: Optional[VenueCalendar] = None,
    evaluation_year: Optional[int] = None,
    required_feature_warmup: int = 252,
    required_repr_window: int = 252,
) -> List[SampleIndexRecord]:
    """Build canonical sample index for a single security across its history.

    Args:
        security_id: Canonical ticker (e.g. "US_AAPL").
        bars_df: DataFrame with 'session' (or DatetimeIndex) and price columns ('close' or 'tr_close').
        venue_calendar: Authoritative VenueCalendar singleton.
        evaluation_year: Mandatory fold evaluation year for query_id formatting (e.g. 2020..2025).
        required_feature_warmup: Number of bars required for technical feature warmup (252).
        required_repr_window: Number of feature bars required for annual representation (252).

    Total preceding history bars required: (required_feature_warmup - 1) + required_repr_window = 503 bars.
    Strictly uses rows t-503 through t-1 (session t is the decision session and excluded from inputs).
    """
    if evaluation_year is None or not isinstance(evaluation_year, int) or evaluation_year < 1900 or evaluation_year > 2200:
        raise ValueError(
            f"Explicit valid integer evaluation_year is required for sample index construction, got: {evaluation_year}"
        )

    vc = venue_calendar or get_venue_calendar()

    # Reindex onto venue calendar
    first_sess = vc.sessions[0]
    last_sess = vc.sessions[-1]

    # Find date range of available bars
    df = bars_df.copy()
    if "session" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "session"})
        elif isinstance(df.index, pd.DatetimeIndex) or df.index.name in ("Date", "session"):
            df = df.reset_index()
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "session"})
            elif "index" in df.columns:
                df = df.rename(columns={"index": "session"})
            else:
                df = df.rename(columns={df.columns[0]: "session"})
        else:
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "session"})

    df["session"] = df["session"].astype(str).str[:10]

    # Reindex onto venue calendar schedule if not already aligned with bar_status
    if "bar_status" not in df.columns:
        s_min = df["session"].min()
        s_max = df["session"].max()
        s_min = max(s_min, vc.sessions[0])
        s_max = min(s_max, vc.sessions[-1])
        df = vc.reindex_to_schedule(df, (s_min, s_max))
    else:
        df = df.sort_values(by="session").reset_index(drop=True)

    N = len(df)
    if N == 0:
        return []

    sessions = df["session"].tolist()

    # Reject unknown session ordinals instead of substituting row positions!
    for sess in sessions:
        if sess not in vc.ordinal:
            raise ValueError(
                f"Session '{sess}' for security '{security_id}' is not a valid scheduled session on venue calendar."
            )

    close_col = "close" if "close" in df.columns else "tr_close"
    closes = df[close_col].to_numpy(dtype=float) if close_col in df.columns else np.zeros(N)

    # 503 bars of preceding history required: 252 warmup + 252 representation (excluding t)
    min_preceding_bars = (required_feature_warmup - 1) + required_repr_window  # 251 + 252 = 503

    # Validity mask: preserve complete validity mask from calendar alignment
    if "bar_status" in df.columns:
        valid_bar = (df["bar_status"] == "VALID").to_numpy(dtype=bool)
    else:
        valid_bar = np.ones(N, dtype=bool)

    if "is_valid_bar" in df.columns:
        valid_bar = valid_bar & df["is_valid_bar"].to_numpy(dtype=bool)

    # Check finite and positive close
    valid_bar = valid_bar & np.isfinite(closes) & (closes > 0.0)

    # If segments present, track them
    has_segments = "segment_id" in df.columns
    segments = df["segment_id"].to_numpy() if has_segments else np.zeros(N, dtype=int)

    # Compute target labels using canonical labels module with venue_sessions
    from memory_study_v2.labels import compute_target_labels
    tr_df = pd.DataFrame({
        "session": sessions,
        "tr_close": closes,
        "is_valid_bar": valid_bar,
        "security_id": security_id,
    })
    if has_segments:
        tr_df["segment_id"] = segments

    lbl_df = compute_target_labels(tr_df, venue_sessions=sessions)
    v63_arr = lbl_df["valid_63"].to_numpy(dtype=bool)
    v126_arr = lbl_df["valid_126"].to_numpy(dtype=bool)
    s63_arr = lbl_df["session_63"].astype(str).to_numpy()
    s126_arr = lbl_df["session_126"].astype(str).to_numpy()

    # Cumulative valid bars for O(1) interval checks
    cum_valid = np.zeros(N + 1, dtype=int)
    np.cumsum(valid_bar, out=cum_valid[1:])

    records: List[SampleIndexRecord] = []

    for t in range(N):
        sess_t = sessions[t]
        ord_t = vc.get_ordinal(sess_t)
        qid = f"{evaluation_year}_{security_id}_{sess_t}"

        # 1. Derive input_window_valid strictly from preceding 503 bars (t - 503 through t - 1)
        if t < min_preceding_bars:
            input_valid = False
            excl_reason = "INSUFFICIENT_INPUT_HISTORY"
        else:
            win_start = t - min_preceding_bars
            win_end = t  # slice [win_start:win_end] has length min_preceding_bars

            # Check all 503 bars are valid and finite
            valid_count = cum_valid[win_end] - cum_valid[win_start]
            if valid_count != min_preceding_bars:
                input_valid = False
                excl_reason = "INVALID_BAR_IN_INPUT_WINDOW"
            elif has_segments and (segments[win_start] != segments[win_end - 1]):
                input_valid = False
                excl_reason = "GAP_IN_INPUT_WINDOW"
            else:
                input_valid = True
                excl_reason = None

        # 2. Forward target and bank maturity from compute_target_labels
        target_63_valid = bool(v63_arr[t])
        target_63_avail = s63_arr[t] if target_63_valid else ""

        bank_126_valid = bool(v126_arr[t])
        bank_126_avail = s126_arr[t] if bank_126_valid else ""

        records.append(SampleIndexRecord(
            query_id=qid,
            security_id=security_id,
            session=sess_t,
            session_ordinal=ord_t,
            input_window_valid=input_valid,
            target_63_valid=target_63_valid,
            target_63_available_at=target_63_avail,
            bank_126_valid=bank_126_valid,
            bank_126_available_at=bank_126_avail,
            exclusion_reason=excl_reason,
        ))

    return records


def filter_admitted_sample_ids(
    records: List[SampleIndexRecord],
    start_session: str,
    end_session: str,
    require_target: bool = True,
    max_target_maturity: Optional[str] = None,
    max_bank_maturity: Optional[str] = None,
) -> List[SampleIndexRecord]:
    """Filter and return strictly admitted sample records for a date range.

    Enforces that:
    1. Session falls strictly within [start_session, end_session].
    2. input_window_valid is True (all preceding 503 bars are valid).
    3. If require_target: target_63_valid is True, and matures <= max_target_maturity.
    4. If max_bank_maturity: bank_126_valid is True, and matures <= max_bank_maturity.
    """
    admitted = []
    for r in records:
        if start_session <= r.session <= end_session:
            if not r.input_window_valid:
                continue
            if require_target:
                if not r.target_63_valid:
                    continue
                if max_target_maturity is not None and r.target_63_available_at > max_target_maturity:
                    continue
            if max_bank_maturity is not None:
                if not r.bank_126_valid or r.bank_126_available_at > max_bank_maturity:
                    continue
            admitted.append(r)
    return admitted


def partition_sample_index(
    records: List[SampleIndexRecord],
    boundaries: FoldBoundaries,
) -> FoldPartitions:
    """Partition sample index records into disjoint walk-forward fold roles.

    Enforces:
    1. input_window_valid is MANDATORY for all partitions (train, val, dev, eval, bank).
    2. Fold boundaries and forward maturities are applied AFTER input-window validation.
    3. Evaluation eligibility is independent of future target availability.
    4. Scored evaluation queries are tracked separately as eval queries with mature target.
    """
    train_idx: List[int] = []
    val_idx: List[int] = []
    dev_idx: List[int] = []
    eval_idx: List[int] = []
    bank_idx: List[int] = []

    train_qids: List[str] = []
    val_qids: List[str] = []
    dev_qids: List[str] = []
    eval_qids: List[str] = []
    bank_qids: List[str] = []

    for i, r in enumerate(records):
        # Mandatory input window validity for ALL roles
        if not r.input_window_valid:
            continue

        s = r.session
        qid = f"{boundaries.evaluation_year}_{r.security_id}_{s}"

        # 1. Training partition
        if boundaries.train_start <= s <= boundaries.train_end:
            if r.target_63_valid and r.target_63_available_at <= boundaries.train_end:
                train_idx.append(i)
                train_qids.append(qid)

        # 2. Validation partition
        if boundaries.val_start <= s <= boundaries.val_end:
            if r.target_63_valid and r.target_63_available_at <= boundaries.val_end:
                val_idx.append(i)
                val_qids.append(qid)

        # 3. Development partition
        if boundaries.dev_start <= s <= boundaries.dev_end:
            if r.target_63_valid and r.target_63_available_at <= boundaries.dev_end:
                dev_idx.append(i)
                dev_qids.append(qid)

        # 4. Evaluation partition: independent of forward target maturity!
        if boundaries.eval_start <= s <= boundaries.eval_end:
            eval_idx.append(i)
            eval_qids.append(qid)

        # 5. Memory bank admission: 126-session maturity within bank cutoff
        if boundaries.train_start <= s <= boundaries.bank_cutoff:
            if r.bank_126_valid and r.bank_126_available_at <= boundaries.bank_cutoff:
                bank_idx.append(i)
                bank_qids.append(qid)

    return FoldPartitions(
        evaluation_year=boundaries.evaluation_year,
        boundaries=boundaries,
        train_indices=train_idx,
        validation_indices=val_idx,
        development_indices=dev_idx,
        evaluation_indices=eval_idx,
        bank_indices=bank_idx,
        train_query_ids=train_qids,
        val_query_ids=val_qids,
        dev_query_ids=dev_qids,
        eval_query_ids=eval_qids,
        bank_query_ids=bank_qids,
    )


def save_fold_sample_ids(
    output_dir: Path,
    evaluation_year: int,
    partitions: FoldPartitions,
    records: Optional[List[SampleIndexRecord]] = None,
) -> Path:
    """Persist canonical sample IDs for a fold to disk for consumer loaders."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"fold_{evaluation_year}_sample_ids.json"

    scored_eval_ids = []
    if records is not None:
        eval_records = [records[i] for i in partitions.evaluation_indices]
        scored_eval_ids = [
            f"{evaluation_year}_{r.security_id}_{r.session}"
            for r in eval_records if r.target_63_valid
        ]

    payload = {
        "evaluation_year": evaluation_year,
        "train_query_ids": partitions.train_query_ids,
        "val_query_ids": partitions.val_query_ids,
        "dev_query_ids": partitions.dev_query_ids,
        "eval_query_ids": partitions.eval_query_ids,
        "scored_eval_query_ids": scored_eval_ids,
        "bank_query_ids": partitions.bank_query_ids,
        "counts": {
            "train": len(partitions.train_query_ids),
            "val": len(partitions.val_query_ids),
            "dev": len(partitions.dev_query_ids),
            "eval": len(partitions.eval_query_ids),
            "scored_eval": len(scored_eval_ids),
            "bank": len(partitions.bank_query_ids),
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_file


def load_fold_sample_ids(
    evaluation_year: int,
    sample_ids_dir: Union[str, Path] = "rebuild_plan/sample_ids",
) -> Dict[str, Any]:
    """Load persisted canonical sample IDs for a given fold year.

    Raises FileNotFoundError if the sample IDs file is missing.
    """
    p = Path(sample_ids_dir) / f"fold_{evaluation_year}_sample_ids.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Canonical sample IDs file not found for fold {evaluation_year} at '{p}'. "
            "Run scripts/generate_fold_manifest.py to generate sample IDs."
        )
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
