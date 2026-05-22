import numpy as np
import pandas as pd

from src.data.features import compute_fundamental_features_aligned


def test_compute_fundamental_features_2y_and_flags():
    # daily prices covering a window before and after a report
    dates = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    price = pd.DataFrame(index=dates)
    price["open"] = 100.0 + np.arange(len(dates))
    price["high"] = price["open"] + 1.0
    price["low"] = price["open"] - 1.0
    price["close"] = price["open"]
    price["volume"] = 1000

    # two synthetic quarterly reports so the 2-year rolling (8-quarter) has at least two points
    edf = pd.DataFrame(
        {
            "report_date": [pd.Timestamp("2019-01-05", tz="UTC"), pd.Timestamp("2020-01-05", tz="UTC")],
            "reported_eps": [0.5, 1.5],
            "eps_estimate": [0.45, 1.4],
            "surprise_pct": [np.nan, np.nan],
            "total_revenue": [900.0, 1200.0],
        }
    )

    out = compute_fundamental_features_aligned(price, edf)

    # columns exist
    assert "fund_eps_2y_avg" in out.columns
    assert "fund_eps_vs_2y_avg" in out.columns
    assert "fund_report_imminent" in out.columns
    assert "fund_just_reported" in out.columns

    # the day before the 2020-01-05 report should be flagged as imminent
    assert out.loc[pd.Timestamp("2020-01-04", tz="UTC")]["fund_report_imminent"] == 1.0

    # the report day should be flagged as just_reported
    assert out.loc[pd.Timestamp("2020-01-05", tz="UTC")]["fund_just_reported"] == 1.0

    # with two reports (0.5 and 1.5) the 2-year average on the latest report should be ~1.0
    assert abs(out.loc[pd.Timestamp("2020-01-05", tz="UTC")]["fund_eps_2y_avg"] - 1.0) < 1e-9

    # compare current reported eps to 2y avg -> (1.5 / 1.0) - 1 = 0.5
    assert abs(out.loc[pd.Timestamp("2020-01-05", tz="UTC")]["fund_eps_vs_2y_avg"] - 0.5) < 1e-6

    # at least one report was aligned
    assert out["fund_report_available"].sum() > 0
