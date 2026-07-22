"""
Feature engineering for trend and report-aware fundamentals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_REPORT_LOOKBACK_DAYS = 9_999.0
_FUNDAMENTAL_FILL_COLUMNS = [
    "fund_eps_surprise",
    "fund_eps_growth_yoy",
    "fund_eps_rolling2_yoy",
    "fund_eps_accel",
    "fund_revenue_growth_yoy",
    "fund_revenue_rolling2_yoy",
    "fund_eps_rolling2_gt_15",
    "fund_eps_surprise_positive",
    "fund_revenue_growth_positive",
    "fund_minervini_score_raw",
    "fund_minervini_components",
    "fund_minervini_score",
    "fund_report_available",
]


def _safe_growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    denom = previous.abs().replace(0, np.nan)
    return (current - previous) / denom


def _canonicalize_report_frame(earnings_df: pd.DataFrame) -> pd.DataFrame:
    if earnings_df is None or earnings_df.empty:
        return pd.DataFrame(columns=["report_date"])

    edf = earnings_df.copy()
    if "report_date" not in edf.columns:
        if isinstance(edf.index, pd.DatetimeIndex):
            index_name = edf.index.name or "index"
            edf = edf.reset_index().rename(columns={index_name: "report_date"})
        else:
            for col in edf.columns:
                if "date" in str(col).lower():
                    edf = edf.rename(columns={col: "report_date"})
                    break

    if "report_date" not in edf.columns:
        edf["report_date"] = pd.to_datetime(edf.iloc[:, 0], utc=True, errors="coerce")
    else:
        edf["report_date"] = pd.to_datetime(
            edf["report_date"], utc=True, errors="coerce"
        )

    edf = edf.rename(
        columns={
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "reported_eps",
            "Surprise(%)": "surprise_pct",
            "Total Revenue": "total_revenue",
            "Operating Revenue": "total_revenue",
        }
    )

    for col in ["eps_estimate", "reported_eps", "surprise_pct", "total_revenue"]:
        if col not in edf.columns:
            edf[col] = np.nan

    edf["eps_estimate"] = pd.to_numeric(edf["eps_estimate"], errors="coerce")
    edf["reported_eps"] = pd.to_numeric(edf["reported_eps"], errors="coerce")
    edf["surprise_pct"] = pd.to_numeric(edf["surprise_pct"], errors="coerce")
    edf["total_revenue"] = pd.to_numeric(edf["total_revenue"], errors="coerce")

    return (
        edf[
            [
                "report_date",
                "eps_estimate",
                "reported_eps",
                "surprise_pct",
                "total_revenue",
            ]
        ]
        .dropna(subset=["report_date"])
        .sort_values("report_date")
        .drop_duplicates(subset=["report_date"], keep="last")
    )


def _apply_fundamental_defaults(price_df: pd.DataFrame) -> pd.DataFrame:
    out = price_df.copy()
    for col in _FUNDAMENTAL_FILL_COLUMNS:
        out[col] = 0.0
    out["fund_days_since_report"] = _DEFAULT_REPORT_LOOKBACK_DAYS
    return out


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily price/volume features with multi-scale trend context.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    df["tech_return_1"] = df["close"].pct_change()

    for lag in [3, 10, 21]:
        df[f"tech_momentum_{lag}"] = df["close"].pct_change(lag)

    df["tech_vol_21"] = df["close"].pct_change().rolling(window=21).std()
    df["tech_avg_volume_21"] = df["volume"].rolling(window=21).mean()
    df["tech_volume_ratio"] = df["volume"] / (df["tech_avg_volume_21"] + 1e-9)
    df["tech_volume_change_1"] = (
        df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    )
    df["tech_intraday_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)

    rolling_max = df["close"].rolling(window=252, min_periods=1).max()
    df["tech_drawdown"] = (df["close"] / rolling_max) - 1.0

    def calculate_slope(x: np.ndarray) -> float:
        if len(x) < 10 or np.any(np.isnan(x)):
            return 0.0
        return float(np.polyfit(np.arange(len(x)), x, 1)[0])

    df["tech_trend_slope"] = df["close"].rolling(window=10).apply(calculate_slope)

    for window in [50, 150, 200]:
        df[f"tech_sma_{window}"] = df["close"].rolling(window=window).mean()
        df[f"tech_close_vs_sma_{window}"] = (
            df["close"] / (df[f"tech_sma_{window}"] + 1e-9)
        ) - 1.0

    df["tech_sma_200_trend_20"] = (
        df["tech_sma_200"] / df["tech_sma_200"].shift(20)
    ) - 1.0
    df["tech_52w_low"] = df["close"].rolling(window=252).min()
    df["tech_52w_high"] = df["close"].rolling(window=252).max()
    df["tech_pct_above_52w_low"] = (df["close"] / df["tech_52w_low"]) - 1.0
    df["tech_pct_from_52w_high"] = (df["close"] / df["tech_52w_high"]) - 1.0

    up_volume = df["volume"].where(df["close"].diff() > 0, 0.0)
    down_volume = df["volume"].where(df["close"].diff() < 0, 0.0)
    df["tech_up_down_volume_ratio_50"] = up_volume.rolling(
        window=50, min_periods=20
    ).sum() / (down_volume.rolling(window=50, min_periods=20).sum() + 1e-9)

    template_checks = [
        (df["close"] > df["tech_sma_150"]) & (df["close"] > df["tech_sma_200"]),
        df["tech_sma_150"] > df["tech_sma_200"],
        df["tech_sma_200_trend_20"] > 0,
        (df["tech_sma_50"] > df["tech_sma_150"])
        & (df["tech_sma_50"] > df["tech_sma_200"]),
        df["close"] > df["tech_sma_50"],
        df["tech_pct_above_52w_low"] > 0.30,
        df["tech_pct_from_52w_high"] > -0.25,
    ]

    df["tech_minervini_template_score"] = sum(
        check.fillna(False).astype(np.float32) for check in template_checks
    )
    df["tech_minervini_gate"] = (df["tech_minervini_template_score"] >= 6.0).astype(
        np.float32
    )

    return df


def compute_fundamental_features_aligned(
    price_df: pd.DataFrame,
    earnings_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align report-level fundamental features to daily prices without lookahead.

    Enhancements added:
    - Two-year rolling averages (8-quarter) for EPS and revenue
    - Comparison of latest EPS/revenue vs 2-year average
    - Days-to-next-report and imminent/post-report flags
    """
    edf = _canonicalize_report_frame(earnings_df)
    if edf.empty:
        return _apply_fundamental_defaults(price_df)

    denom = edf["eps_estimate"].abs().replace(0, np.nan)
    edf["fund_eps_surprise"] = (edf["reported_eps"] - edf["eps_estimate"]) / denom
    edf["fund_eps_surprise"] = edf["fund_eps_surprise"].fillna(
        edf["surprise_pct"] / 100.0
    )

    # short rolling (2-quarter) used previously
    eps_roll2 = edf["reported_eps"].rolling(window=2, min_periods=2).mean()
    rev_roll2 = edf["total_revenue"].rolling(window=2, min_periods=2).mean()

    # 2-year rolling averages (8 quarters)
    eps_roll8 = edf["reported_eps"].rolling(window=8, min_periods=2).mean()
    rev_roll8 = edf["total_revenue"].rolling(window=8, min_periods=2).mean()
    edf["fund_eps_2y_avg"] = eps_roll8
    edf["fund_rev_2y_avg"] = rev_roll8

    # compare latest reported value to 2-year average
    edf["fund_eps_vs_2y_avg"] = np.where(
        edf["fund_eps_2y_avg"].notna(),
        (edf["reported_eps"] / (edf["fund_eps_2y_avg"] + 1e-9)) - 1.0,
        np.nan,
    )
    edf["fund_rev_vs_2y_avg"] = np.where(
        edf["fund_rev_2y_avg"].notna(),
        (edf["total_revenue"] / (edf["fund_rev_2y_avg"] + 1e-9)) - 1.0,
        np.nan,
    )

    # year-over-year and rolling-growth signals
    edf["fund_eps_growth_yoy"] = _safe_growth(
        edf["reported_eps"], edf["reported_eps"].shift(4)
    )
    edf["fund_eps_rolling2_yoy"] = _safe_growth(eps_roll2, eps_roll2.shift(4))
    edf["fund_eps_rolling8_yoy"] = _safe_growth(eps_roll8, eps_roll8.shift(4))
    edf["fund_eps_accel"] = edf["fund_eps_rolling2_yoy"].diff()
    edf["fund_revenue_growth_yoy"] = _safe_growth(
        edf["total_revenue"], edf["total_revenue"].shift(4)
    )
    edf["fund_revenue_rolling2_yoy"] = _safe_growth(rev_roll2, rev_roll2.shift(4))
    edf["fund_revenue_rolling8_yoy"] = _safe_growth(rev_roll8, rev_roll8.shift(4))

    composite = pd.DataFrame(
        {
            "eps_rolling2": np.where(
                edf["fund_eps_rolling2_yoy"].notna(),
                (edf["fund_eps_rolling2_yoy"] > 0.15).astype(np.float32),
                np.nan,
            ),
            "eps_surprise": np.where(
                edf["fund_eps_surprise"].notna(),
                (edf["fund_eps_surprise"] > 0.0).astype(np.float32),
                np.nan,
            ),
            "revenue_rolling2": np.where(
                edf["fund_revenue_rolling2_yoy"].notna(),
                (edf["fund_revenue_rolling2_yoy"] > 0.0).astype(np.float32),
                np.nan,
            ),
        }
    )

    edf["fund_eps_rolling2_gt_15"] = np.nan_to_num(composite["eps_rolling2"], nan=0.0)
    edf["fund_eps_surprise_positive"] = np.nan_to_num(
        composite["eps_surprise"], nan=0.0
    )
    edf["fund_revenue_growth_positive"] = np.nan_to_num(
        composite["revenue_rolling2"], nan=0.0
    )
    edf["fund_minervini_components"] = composite.notna().sum(axis=1).astype(np.float32)
    edf["fund_minervini_score_raw"] = (
        composite.fillna(0.0).sum(axis=1).astype(np.float32)
    )
    edf["fund_minervini_score"] = np.where(
        edf["fund_minervini_components"] > 0,
        edf["fund_minervini_score_raw"] / edf["fund_minervini_components"],
        0.0,
    )
    edf["fund_report_available"] = 1.0

    # Align to daily prices using the most recent past report
    price_sorted = price_df.sort_index()
    price_with_date = price_sorted.reset_index()
    date_col = price_with_date.columns[0]

    merged = pd.merge_asof(
        price_with_date,
        edf[
            [
                "report_date",
                *_FUNDAMENTAL_FILL_COLUMNS,
                "fund_eps_2y_avg",
                "fund_rev_2y_avg",
                "fund_eps_vs_2y_avg",
                "fund_rev_vs_2y_avg",
            ]
        ].sort_values("report_date"),
        left_on=date_col,
        right_on="report_date",
        direction="backward",
    )

    merged["fund_days_since_report"] = (
        merged[date_col] - merged["report_date"]
    ).dt.total_seconds() / 86_400.0
    merged["fund_days_since_report"] = (
        merged["fund_days_since_report"]
        .fillna(_DEFAULT_REPORT_LOOKBACK_DAYS)
        .clip(lower=0.0)
    )

    # Also align the next scheduled report (if available) to compute days-to-next-report
    next_merged = pd.merge_asof(
        price_with_date,
        edf[["report_date"]].sort_values("report_date"),
        left_on=date_col,
        right_on="report_date",
        direction="forward",
    )
    merged["next_report_date"] = next_merged["report_date"]
    merged["fund_days_to_next_report"] = (
        merged["next_report_date"] - merged[date_col]
    ).dt.total_seconds() / 86_400.0
    merged["fund_days_to_next_report"] = (
        merged["fund_days_to_next_report"]
        .fillna(_DEFAULT_REPORT_LOOKBACK_DAYS)
        .clip(lower=0.0)
    )

    # Flags for imminence / just-post-report
    REPORT_IMMINENT_DAYS = 14
    POST_REPORT_DAYS = 7
    merged["fund_report_imminent"] = (
        merged["fund_days_to_next_report"] <= REPORT_IMMINENT_DAYS
    ).astype(np.float32)
    merged["fund_just_reported"] = (
        merged["fund_days_since_report"] <= POST_REPORT_DAYS
    ).astype(np.float32)

    merged = merged.drop(columns=["report_date"])

    # Fill any remaining NaNs for consistency
    for col in _FUNDAMENTAL_FILL_COLUMNS + [
        "fund_eps_2y_avg",
        "fund_rev_2y_avg",
        "fund_eps_vs_2y_avg",
        "fund_rev_vs_2y_avg",
        "fund_days_to_next_report",
        "fund_report_imminent",
        "fund_just_reported",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    return merged.set_index(date_col)


def ensure_sequence_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill sequence-model inputs for older precomputed parquet runs.
    """
    out = df.copy()

    if "tech_return_1" not in out.columns:
        out["tech_return_1"] = out["close"].pct_change()
    if "tech_volume_change_1" not in out.columns:
        out["tech_volume_change_1"] = (
            out["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        )
    if "tech_intraday_range" not in out.columns:
        out["tech_intraday_range"] = (out["high"] - out["low"]) / (out["close"] + 1e-9)

    for window in [50, 150, 200]:
        sma_col = f"tech_sma_{window}"
        ratio_col = f"tech_close_vs_sma_{window}"
        if sma_col not in out.columns:
            out[sma_col] = out["close"].rolling(window=window).mean()
        if ratio_col not in out.columns:
            out[ratio_col] = (out["close"] / (out[sma_col] + 1e-9)) - 1.0

    return out
