"""Shared, explicitly bounded OHLC roundoff policy (4-ULP guard band).

Permits at most four float64 representable steps (ULPs) for each violated OHLC
ordering comparison, constructed strictly via np.nextafter.
Does not use price-independent absolute tolerances, default np.isclose, or float32 conversion.

Classifications:
- ALREADY_VALID: All strict ordering inequalities hold.
- ROUNDOFF_ONLY: All violated relationships lie within the 4-step allowance.
                 Offending high/low values are normalized minimally so strict
                 inequalities hold.
- MATERIAL_DISCREPANCY: Any violated relationship exceeds 4 ULPs. Kept rejected.
- OTHER_INVALID: Non-finite, non-positive, or missing prices/volumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


MAX_ROUNDOFF_ULPS: int = 4


class OHLCClassification(str, Enum):
    ALREADY_VALID = "ALREADY_VALID"
    ROUNDOFF_ONLY = "ROUNDOFF_ONLY"
    MATERIAL_DISCREPANCY = "MATERIAL_DISCREPANCY"
    OTHER_INVALID = "OTHER_INVALID"


def step_down_float64(val: float, steps: int = MAX_ROUNDOFF_ULPS) -> float:
    """Step downward towards -inf by *steps* representable float64 increments."""
    res = float(val)
    for _ in range(steps):
        res = float(np.nextafter(res, -np.inf))
    return res


def step_up_float64(val: float, steps: int = MAX_ROUNDOFF_ULPS) -> float:
    """Step upward towards +inf by *steps* representable float64 increments."""
    res = float(val)
    for _ in range(steps):
        res = float(np.nextafter(res, np.inf))
    return res


def step_down_float64_vec(arr: np.ndarray, steps: int = MAX_ROUNDOFF_ULPS) -> np.ndarray:
    """Vectorized step downward by *steps* float64 increments."""
    res = np.asarray(arr, dtype=np.float64).copy()
    for _ in range(steps):
        res = np.nextafter(res, -np.inf)
    return res


def step_up_float64_vec(arr: np.ndarray, steps: int = MAX_ROUNDOFF_ULPS) -> np.ndarray:
    """Vectorized step upward by *steps* float64 increments."""
    res = np.asarray(arr, dtype=np.float64).copy()
    for _ in range(steps):
        res = np.nextafter(res, np.inf)
    return res


@dataclass(frozen=True)
class OHLCValidationResult:
    classification: OHLCClassification
    is_valid_or_roundoff: bool
    normalized_open: float
    normalized_high: float
    normalized_low: float
    normalized_close: float
    violations: List[str]
    max_abs_violation: float
    max_rel_violation: float


def classify_and_normalize_ohlc_row(
    open_val: float,
    high_val: float,
    low_val: float,
    close_val: float,
    max_ulps: int = MAX_ROUNDOFF_ULPS,
) -> OHLCValidationResult:
    """Classify and minimally normalize a single OHLC bar.

    Finite and positive prices are required. If all relationships hold,
    returns ALREADY_VALID with unmodified prices.
    If every violated ordering inequality is within *max_ulps* of the boundary,
    returns ROUNDOFF_ONLY with high/low normalized minimally.
    If any violated inequality exceeds *max_ulps*, returns MATERIAL_DISCREPANCY
    with original prices.
    """
    try:
        o = float(open_val)
        h = float(high_val)
        l = float(low_val)
        c = float(close_val)
    except (ValueError, TypeError):
        return OHLCValidationResult(
            classification=OHLCClassification.OTHER_INVALID,
            is_valid_or_roundoff=False,
            normalized_open=open_val,
            normalized_high=high_val,
            normalized_low=low_val,
            normalized_close=close_val,
            violations=["non_numeric_price"],
            max_abs_violation=float("nan"),
            max_rel_violation=float("nan"),
        )

    # Check finite and positive
    if not (math.isfinite(o) and math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
        return OHLCValidationResult(
            classification=OHLCClassification.OTHER_INVALID,
            is_valid_or_roundoff=False,
            normalized_open=o,
            normalized_high=h,
            normalized_low=l,
            normalized_close=c,
            violations=["non_finite_price"],
            max_abs_violation=float("nan"),
            max_rel_violation=float("nan"),
        )

    if o <= 0.0 or h <= 0.0 or l <= 0.0 or c <= 0.0:
        return OHLCValidationResult(
            classification=OHLCClassification.OTHER_INVALID,
            is_valid_or_roundoff=False,
            normalized_open=o,
            normalized_high=h,
            normalized_low=l,
            normalized_close=c,
            violations=["non_positive_price"],
            max_abs_violation=float("nan"),
            max_rel_violation=float("nan"),
        )

    # Check 5 core ordering relationships
    v_h_l = (h < l)
    v_h_o = (h < o)
    v_h_c = (h < c)
    v_l_o = (l > o)
    v_l_c = (l > c)

    violations = []
    if v_h_l:
        violations.append("high < low")
    if v_h_o:
        violations.append("high < open")
    if v_h_c:
        violations.append("high < close")
    if v_l_o:
        violations.append("low > open")
    if v_l_c:
        violations.append("low > close")

    if not violations:
        return OHLCValidationResult(
            classification=OHLCClassification.ALREADY_VALID,
            is_valid_or_roundoff=True,
            normalized_open=o,
            normalized_high=h,
            normalized_low=l,
            normalized_close=c,
            violations=[],
            max_abs_violation=0.0,
            max_rel_violation=0.0,
        )

    # Test whether EVERY violated relationship lies within the approved ULP allowance
    is_material = False
    abs_violations: List[float] = []

    if v_h_l:
        bound = step_down_float64(l, max_ulps)
        abs_v = l - h
        abs_violations.append(abs_v)
        if h < bound:
            is_material = True

    if v_h_o:
        bound = step_down_float64(o, max_ulps)
        abs_v = o - h
        abs_violations.append(abs_v)
        if h < bound:
            is_material = True

    if v_h_c:
        bound = step_down_float64(c, max_ulps)
        abs_v = c - h
        abs_violations.append(abs_v)
        if h < bound:
            is_material = True

    if v_l_o:
        bound = step_up_float64(o, max_ulps)
        abs_v = l - o
        abs_violations.append(abs_v)
        if l > bound:
            is_material = True

    if v_l_c:
        bound = step_up_float64(c, max_ulps)
        abs_v = l - c
        abs_violations.append(abs_v)
        if l > bound:
            is_material = True

    max_abs = max(abs_violations) if abs_violations else 0.0
    max_rel = max_abs / c if c > 0.0 else 0.0

    if is_material:
        return OHLCValidationResult(
            classification=OHLCClassification.MATERIAL_DISCREPANCY,
            is_valid_or_roundoff=False,
            normalized_open=o,
            normalized_high=h,
            normalized_low=l,
            normalized_close=c,
            violations=violations,
            max_abs_violation=max_abs,
            max_rel_violation=max_rel,
        )

    # Roundoff-only: normalize high and low minimally so strict inequalities hold
    norm_h = max(h, o, c)
    norm_l = min(l, o, c)
    if norm_h < norm_l:
        norm_h = max(norm_h, norm_l)

    return OHLCValidationResult(
        classification=OHLCClassification.ROUNDOFF_ONLY,
        is_valid_or_roundoff=True,
        normalized_open=o,
        normalized_high=norm_h,
        normalized_low=norm_l,
        normalized_close=c,
        violations=violations,
        max_abs_violation=max_abs,
        max_rel_violation=max_rel,
    )


def normalize_ohlc_dataframe(
    df: pd.DataFrame,
    security_id: str = "",
    audit_records: Optional[List[Dict[str, Any]]] = None,
    max_ulps: int = MAX_ROUNDOFF_ULPS,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, int]]:
    """Scan and normalize OHLC DataFrame in-place on a copy.

    Leaves valid rows completely untouched.
    Normalizes rows classified as ROUNDOFF_ONLY.
    Returns (normalized_df, audit_records, stats).
    """
    out_df = df.copy()
    if out_df.empty:
        return out_df, [] if audit_records is None else audit_records, {"valid": 0, "roundoff": 0, "material": 0}

    stats = {"valid": 0, "roundoff": 0, "material": 0, "other": 0}
    records = audit_records if audit_records is not None else []

    date_col = "session" if "session" in out_df.columns else ("Date" if "Date" in out_df.columns else None)
    dates = out_df[date_col].astype(str).tolist() if date_col else [str(i) for i in range(len(out_df))]

    o_arr = pd.to_numeric(out_df["open"], errors="coerce").to_numpy(dtype=np.float64)
    h_arr = pd.to_numeric(out_df["high"], errors="coerce").to_numpy(dtype=np.float64)
    l_arr = pd.to_numeric(out_df["low"], errors="coerce").to_numpy(dtype=np.float64)
    c_col = "close" if "close" in out_df.columns else ("tr_close" if "tr_close" in out_df.columns else "close")
    c_arr = pd.to_numeric(out_df[c_col], errors="coerce").to_numpy(dtype=np.float64)

    new_h = h_arr.copy()
    new_l = l_arr.copy()

    for i in range(len(out_df)):
        res = classify_and_normalize_ohlc_row(o_arr[i], h_arr[i], l_arr[i], c_arr[i], max_ulps=max_ulps)
        if res.classification == OHLCClassification.ALREADY_VALID:
            stats["valid"] += 1
        elif res.classification == OHLCClassification.ROUNDOFF_ONLY:
            stats["roundoff"] += 1
            new_h[i] = res.normalized_high
            new_l[i] = res.normalized_low
            records.append({
                "security": security_id,
                "session": dates[i],
                "classification": "ROUNDOFF_ONLY",
                "original_open": float(o_arr[i]),
                "original_high": float(h_arr[i]),
                "original_low": float(l_arr[i]),
                "original_close": float(c_arr[i]),
                "corrected_high": float(res.normalized_high),
                "corrected_low": float(res.normalized_low),
                "violations": "; ".join(res.violations),
                "max_abs_violation": float(res.max_abs_violation),
                "max_rel_violation": float(res.max_rel_violation),
                "reason": f"Bounded roundoff within {max_ulps} ULPs; normalized minimally",
            })
        elif res.classification == OHLCClassification.MATERIAL_DISCREPANCY:
            stats["material"] += 1
            records.append({
                "security": security_id,
                "session": dates[i],
                "classification": "MATERIAL_DISCREPANCY",
                "original_open": float(o_arr[i]),
                "original_high": float(h_arr[i]),
                "original_low": float(l_arr[i]),
                "original_close": float(c_arr[i]),
                "corrected_high": float(h_arr[i]),
                "corrected_low": float(l_arr[i]),
                "violations": "; ".join(res.violations),
                "max_abs_violation": float(res.max_abs_violation),
                "max_rel_violation": float(res.max_rel_violation),
                "reason": f"Material discrepancy exceeding {max_ulps} ULPs; rejected",
            })
        else:
            stats["other"] += 1

    out_df["high"] = new_h
    out_df["low"] = new_l
    return out_df, records, stats
