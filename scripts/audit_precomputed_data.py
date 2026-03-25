"""
Generate a compact research-oriented audit for precomputed daily datasets.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = "reports/data_quality"
DATA_PATTERN = "data/precomputed/*.parquet"


def build_audit(pattern: str = DATA_PATTERN) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        return {"rows": 0, "message": "No precomputed data found."}

    frames = [read_dataframe(path) for path in files]
    df = pd.concat(frames, ignore_index=False)
    if df.empty:
        return {"rows": 0, "message": "No precomputed data found."}

    per_ticker = []
    for ticker, ticker_df in df.groupby("ticker", sort=True):
        per_ticker.append(
            {
                "ticker": ticker,
                "rows": int(len(ticker_df)),
                "positive_rate": float(ticker_df["in_cycle"].mean())
                if "in_cycle" in ticker_df
                else 0.0,
                "candidate_gate_rate": float(ticker_df["tech_minervini_gate"].mean())
                if "tech_minervini_gate" in ticker_df
                else 0.0,
                "avg_template_score": float(
                    ticker_df["tech_minervini_template_score"].mean()
                )
                if "tech_minervini_template_score" in ticker_df
                else 0.0,
                "report_coverage": float(ticker_df["fund_report_available"].mean())
                if "fund_report_available" in ticker_df
                else 0.0,
            }
        )

    missing_share = df.isna().mean().sort_values(ascending=False).head(20).to_dict()

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "positive_rate": float(df["in_cycle"].mean())
        if "in_cycle" in df.columns
        else 0.0,
        "candidate_gate_rate": float(df["tech_minervini_gate"].mean())
        if "tech_minervini_gate" in df.columns
        else 0.0,
        "report_coverage": float(df["fund_report_available"].mean())
        if "fund_report_available" in df.columns
        else 0.0,
        "missing_share_top20": missing_share,
        "per_ticker": per_ticker,
    }


def write_audit(audit: dict) -> tuple[str, str]:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"precomputed_audit_{run_id}.json")
    md_path = os.path.join(OUTPUT_DIR, f"precomputed_audit_{run_id}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    top_tickers = sorted(
        audit.get("per_ticker", []), key=lambda row: row["positive_rate"], reverse=True
    )
    lines = [
        "# Precomputed Dataset Audit",
        "",
        f"- Rows: {audit.get('rows', 0)}",
        f"- Tickers: {audit.get('tickers', 0)}",
        f"- Positive rate: {audit.get('positive_rate', 0.0):.4f}",
        f"- Candidate gate rate: {audit.get('candidate_gate_rate', 0.0):.4f}",
        f"- Report coverage: {audit.get('report_coverage', 0.0):.4f}",
        "",
        "## Top Positive-Rate Tickers",
        "",
    ]

    for row in top_tickers[:10]:
        lines.append(
            f"- {row['ticker']}: rows={row['rows']}, positive_rate={row['positive_rate']:.4f}, "
            f"gate_rate={row['candidate_gate_rate']:.4f}, report_coverage={row['report_coverage']:.4f}"
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default=DATA_PATTERN)
    args = parser.parse_args()

    audit = build_audit(pattern=args.pattern)
    json_path, md_path = write_audit(audit)
    logger.info("Wrote dataset audit to %s and %s", json_path, md_path)


if __name__ == "__main__":
    main()
