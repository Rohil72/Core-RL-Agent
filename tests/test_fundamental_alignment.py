import numpy as np
import pandas as pd

from src.data.features import compute_fundamental_features_aligned


def test_fundamental_alignment_uses_report_index_and_builds_rolling_features():
    price_index = pd.date_range("2024-01-02", "2024-04-30", freq="B", tz="UTC")
    price_df = pd.DataFrame(
        {
            "open": np.linspace(100, 140, len(price_index)),
            "high": np.linspace(101, 141, len(price_index)),
            "low": np.linspace(99, 139, len(price_index)),
            "close": np.linspace(100, 140, len(price_index)),
            "volume": np.linspace(1_000_000, 1_500_000, len(price_index)),
        },
        index=price_index,
    )

    report_dates = pd.to_datetime(
        [
            "2022-11-15",
            "2023-02-15",
            "2023-05-15",
            "2023-08-15",
            "2023-11-15",
            "2024-02-15",
        ],
        utc=True,
    )
    earnings_df = pd.DataFrame(
        {
            "EPS Estimate": [0.90, 1.00, 1.10, 1.20, 1.50, 1.80],
            "Reported EPS": [1.00, 1.10, 1.25, 1.35, 1.90, 2.30],
            "Surprise(%)": [11.1, 10.0, 13.6, 12.5, 26.7, 27.8],
            "total_revenue": [100, 110, 120, 130, 150, 180],
        },
        index=report_dates,
    )
    earnings_df.index.name = "Earnings Date"

    aligned = compute_fundamental_features_aligned(price_df, earnings_df)

    assert len(aligned) == len(price_df)
    assert "report_date" not in aligned.columns
    assert "fund_days_since_report" in aligned.columns
    assert aligned["fund_report_available"].iloc[0] == 1.0
    assert aligned["fund_eps_surprise"].iloc[0] > 0
    assert aligned.loc["2024-04-30", "fund_eps_rolling2_yoy"] > 0.15
    assert aligned.loc["2024-04-30", "fund_revenue_rolling2_yoy"] > 0
    assert aligned.loc["2024-04-30", "fund_minervini_score"] > 0
    assert not aligned.filter(like="fund_").isna().any().any()


def test_fundamental_alignment_falls_back_to_neutral_defaults():
    price_index = pd.date_range("2024-01-02", periods=20, freq="B", tz="UTC")
    price_df = pd.DataFrame(
        {
            "open": np.linspace(100, 110, len(price_index)),
            "high": np.linspace(101, 111, len(price_index)),
            "low": np.linspace(99, 109, len(price_index)),
            "close": np.linspace(100, 110, len(price_index)),
            "volume": np.linspace(1_000_000, 1_200_000, len(price_index)),
        },
        index=price_index,
    )

    aligned = compute_fundamental_features_aligned(price_df, pd.DataFrame())

    assert len(aligned) == len(price_df)
    assert aligned["fund_report_available"].eq(0.0).all()
    assert aligned["fund_days_since_report"].eq(9999.0).all()
    assert aligned["fund_minervini_score"].eq(0.0).all()
