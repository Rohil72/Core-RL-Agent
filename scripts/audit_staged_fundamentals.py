"""Audit point-in-time earnings coverage in staged international Parquet files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class FundamentalAudit:
    """Coverage result for one staged security."""

    market: str
    ticker: str
    status: str
    reasons: tuple[str, ...]
    sessions: int
    report_count: int
    first_session: str | None
    last_session: str | None
    first_report: str | None
    last_report: str | None
    missing_full_years: tuple[int, ...]
    sparse_full_years: tuple[int, ...]


def _full_years(start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    """Return calendar years fully contained in the requested interval."""
    first = start.year if (start.month, start.day) == (1, 1) else start.year + 1
    last = end.year if (end.month, end.day) == (12, 31) else end.year - 1
    return list(range(first, last + 1)) if first <= last else []


def audit_frame(
    frame: pd.DataFrame,
    *,
    market: str,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    minimum_sessions: int,
    minimum_reports_per_full_year: int,
    maximum_latest_report_age_days: int,
) -> FundamentalAudit:
    """Evaluate whether a staged frame has usable point-in-time report history."""
    reasons: list[str] = []
    sessions = len(frame)
    if sessions < minimum_sessions:
        reasons.append("insufficient_sessions")

    session_dates = pd.to_datetime(frame.index, utc=True, errors="coerce")
    valid_sessions = pd.Series(session_dates).dropna()
    first_session = valid_sessions.min() if not valid_sessions.empty else None
    last_session = valid_sessions.max() if not valid_sessions.empty else None

    if "report_date" not in frame:
        report_dates = pd.Series(dtype="datetime64[ns, UTC]")
        reasons.append("missing_report_date_column")
    else:
        report_dates = (
            pd.to_datetime(frame["report_date"], utc=True, errors="coerce")
            .dropna()
            .drop_duplicates()
            .sort_values()
        )
        report_dates = report_dates[(report_dates >= start) & (report_dates <= end)]
        if report_dates.empty:
            reasons.append("no_earnings_reports")

    counts = report_dates.dt.year.value_counts() if not report_dates.empty else pd.Series(dtype=int)
    full_years = _full_years(start, end)
    missing_years = tuple(year for year in full_years if int(counts.get(year, 0)) == 0)
    sparse_years = tuple(
        year
        for year in full_years
        if 0 < int(counts.get(year, 0)) < minimum_reports_per_full_year
    )
    if missing_years:
        reasons.append("missing_full_years")
    if sparse_years:
        reasons.append("sparse_full_years")

    first_report = report_dates.min() if not report_dates.empty else None
    last_report = report_dates.max() if not report_dates.empty else None
    if first_report is not None and full_years and first_report.year > full_years[0]:
        reasons.append("history_starts_late")
    if last_report is not None and last_session is not None:
        age_days = int((last_session - last_report).days)
        if age_days > maximum_latest_report_age_days:
            reasons.append("latest_report_stale")

    return FundamentalAudit(
        market=market,
        ticker=ticker,
        status="ok" if not reasons else "failed",
        reasons=tuple(reasons),
        sessions=sessions,
        report_count=len(report_dates),
        first_session=first_session.isoformat() if first_session is not None else None,
        last_session=last_session.isoformat() if last_session is not None else None,
        first_report=first_report.isoformat() if first_report is not None else None,
        last_report=last_report.isoformat() if last_report is not None else None,
        missing_full_years=missing_years,
        sparse_full_years=sparse_years,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Audit every configured ticker and write machine-readable reports."""
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    data_root = Path(args.data_root or experiment["data_root"])
    start = pd.Timestamp(args.start or experiment["start"], tz="UTC")
    end = pd.Timestamp(args.end or experiment["end"], tz="UTC")
    minimum_sessions = int(args.minimum_sessions or experiment["minimum_sessions"])
    rows: list[FundamentalAudit] = []

    for market, metadata in config["markets"].items():
        for ticker in metadata["tickers"]:
            path = data_root / market / f"{ticker}.parquet"
            if not path.exists():
                rows.append(
                    FundamentalAudit(
                        market=market,
                        ticker=str(ticker),
                        status="failed",
                        reasons=("missing_parquet",),
                        sessions=0,
                        report_count=0,
                        first_session=None,
                        last_session=None,
                        first_report=None,
                        last_report=None,
                        missing_full_years=tuple(_full_years(start, end)),
                        sparse_full_years=(),
                    )
                )
                continue
            frame = pd.read_parquet(path)
            rows.append(
                audit_frame(
                    frame,
                    market=market,
                    ticker=str(ticker),
                    start=start,
                    end=end,
                    minimum_sessions=minimum_sessions,
                    minimum_reports_per_full_year=args.minimum_reports_per_full_year,
                    maximum_latest_report_age_days=args.maximum_latest_report_age_days,
                )
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = [asdict(row) for row in rows]
    pd.DataFrame(records).to_csv(output / "fundamental_coverage.csv", index=False)
    failed = [row for row in records if row["status"] != "ok"]
    summary = {
        "status": "passed" if not failed else "failed",
        "ticker_count": len(records),
        "passed_count": len(records) - len(failed),
        "failed_count": len(failed),
        "failed_by_market": {
            market: sum(row["market"] == market for row in failed)
            for market in config["markets"]
        },
        "criteria": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "minimum_sessions": minimum_sessions,
            "minimum_reports_per_full_year": args.minimum_reports_per_full_year,
            "maximum_latest_report_age_days": args.maximum_latest_report_age_days,
        },
        "failed_tickers": failed,
    }
    (output / "fundamental_coverage.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_research_testbed.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--minimum-sessions", type=int)
    parser.add_argument("--minimum-reports-per-full-year", type=int, default=3)
    parser.add_argument("--maximum-latest-report-age-days", type=int, default=180)
    args = parser.parse_args()
    summary = run(args)
    raise SystemExit(0 if summary["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
