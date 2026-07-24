from __future__ import annotations

import pandas as pd

from scripts.audit_staged_fundamentals import audit_frame


def _frame(report_dates: list[str]) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", "2021-12-31", freq="B", tz="UTC")
    reports = pd.to_datetime(report_dates, utc=True)
    aligned = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    for report in reports:
        aligned.loc[aligned.index >= report] = report
    return pd.DataFrame({"report_date": aligned}, index=index)


def test_audit_accepts_three_reports_in_each_full_year() -> None:
    frame = _frame(
        [
            "2020-02-01", "2020-05-01", "2020-08-01",
            "2021-02-01", "2021-05-01", "2021-08-01",
        ]
    )
    result = audit_frame(
        frame,
        market="test",
        ticker="GOOD",
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2021-12-31", tz="UTC"),
        minimum_sessions=500,
        minimum_reports_per_full_year=3,
        maximum_latest_report_age_days=180,
    )
    assert result.status == "ok"
    assert result.report_count == 6


def test_audit_rejects_late_and_sparse_history() -> None:
    frame = _frame(["2021-05-01"])
    result = audit_frame(
        frame,
        market="test",
        ticker="BAD",
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2021-12-31", tz="UTC"),
        minimum_sessions=500,
        minimum_reports_per_full_year=3,
        maximum_latest_report_age_days=180,
    )
    assert result.status == "failed"
    assert "missing_full_years" in result.reasons
    assert "sparse_full_years" in result.reasons
    assert "history_starts_late" in result.reasons
