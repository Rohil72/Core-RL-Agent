"""Universe Ledger and Data Manifest Module.

Tracks requested vs available securities across the 6 target markets:
- United States (18 requested, 18 available)
- India (18 requested, 18 available)
- China (18 requested, 18 available)
- Brazil (18 requested, 15 available; 3 exclusions)
- France (18 requested, 17 available; 1 exclusion)
- United Kingdom (18 requested, 17 available; 1 exclusion)
Total: 108 requested, 103 available, 5 excluded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

MARKET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "US": {
        "exchange": "NYSE_NASDAQ",
        "currency": "USD",
        "benchmark": "^GSPC",
        "execution_cost_bps": 10.0,
        "tickers": [
            "AAPL", "ADBE", "AMD", "AMGN", "AMZN", "AVGO", "CRM", "GOOGL",
            "INTU", "ISRG", "LMT", "META", "MSFT", "NOC", "NVDA", "ORCL",
            "REGN", "V",
        ],
        "excluded": {},
    },
    "India": {
        "exchange": "NSE",
        "currency": "INR",
        "benchmark": "^NSEI",
        "execution_cost_bps": 20.0,
        "tickers": [
            "ASIANPAINT.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "EICHERMOT.NS",
            "HCLTECH.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "LT.NS",
            "M&M.NS", "MARUTI.NS", "PIDILITIND.NS", "RELIANCE.NS", "SUNPHARMA.NS",
            "TCS.NS", "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS",
        ],
        "excluded": {},
    },
    "China": {
        "exchange": "SSE_SZSE",
        "currency": "CNY",
        "benchmark": "000300.SS",
        "execution_cost_bps": 20.0,
        "tickers": [
            "000333.SZ", "000725.SZ", "000858.SZ", "002230.SZ", "002241.SZ",
            "002415.SZ", "002475.SZ", "002594.SZ", "300059.SZ", "600036.SS",
            "600196.SS", "600276.SS", "600309.SS", "600519.SS", "600887.SS",
            "601012.SS", "601318.SS", "601888.SS",
        ],
        "excluded": {},
    },
    "Brazil": {
        "exchange": "B3",
        "currency": "BRL",
        "benchmark": "^BVSP",
        "execution_cost_bps": 15.0,
        "tickers": [
            "B3SA3.SA", "EQTL3.SA", "FLRY3.SA", "ITUB4.SA", "KLBN11.SA",
            "LREN3.SA", "MGLU3.SA", "PETR4.SA", "RADL3.SA", "RAIL3.SA",
            "RENT3.SA", "SUZB3.SA", "TOTS3.SA", "VALE3.SA", "WEGE3.SA",
        ],
        "excluded": {
            "CIEL3.SA": "historical reason unavailable",
            "EMBR3.SA": "historical reason unavailable",
            "JBSS3.SA": "historical reason unavailable",
        },
    },
    "France": {
        "exchange": "EURONEXT_PARIS",
        "currency": "EUR",
        "benchmark": "^FCHI",
        "execution_cost_bps": 20.0,
        "tickers": [
            "AI.PA", "AIR.PA", "CAP.PA", "DG.PA", "DIM.PA",
            "DSY.PA", "EL.PA", "LR.PA", "MC.PA", "ML.PA",
            "OR.PA", "RI.PA", "RMS.PA", "SAF.PA", "SU.PA",
            "TEP.PA", "WLN.PA",
        ],
        "excluded": {
            "STM.PA": "historical reason unavailable",
        },
    },
    "UK": {
        "exchange": "LSE",
        "currency": "GBP",
        "benchmark": "^FTSE",
        "execution_cost_bps": 25.0,
        "tickers": [
            "AUTO.L", "AZN.L", "BA.L", "CPG.L", "CRDA.L",
            "DGE.L", "EXPN.L", "HLMA.L", "JD.L", "LSEG.L",
            "OCDO.L", "PRU.L", "REL.L", "RMV.L", "RTO.L",
            "SGE.L", "SPX.L",
        ],
        "excluded": {
            "AHT.L": "historical reason unavailable",
        },
    },
}


@dataclass(frozen=True)
class UniverseSummary:
    market: str
    requested_count: int
    available_count: int
    excluded_count: int
    available_tickers: list[str]
    excluded_tickers: dict[str, str]


def get_universe_summary() -> dict[str, UniverseSummary]:
    """Return universe count and exclusions per market."""
    out: dict[str, UniverseSummary] = {}
    for market, info in MARKET_DEFINITIONS.items():
        avail = list(info["tickers"])
        excl = dict(info["excluded"])
        out[market] = UniverseSummary(
            market=market,
            requested_count=len(avail) + len(excl),
            available_count=len(avail),
            excluded_count=len(excl),
            available_tickers=avail,
            excluded_tickers=excl,
        )
    return out


def generate_universe_ledger_report(
    output_dir: str | Path = "reports/reconstruction_v1/manifests",
) -> dict[str, Any]:
    """Generate and write the symbol universe ledger."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary = get_universe_summary()
    total_requested = sum(s.requested_count for s in summary.values())
    total_available = sum(s.available_count for s in summary.values())
    total_excluded = sum(s.excluded_count for s in summary.values())

    report = {
        "totals": {
            "requested": total_requested,
            "available": total_available,
            "excluded": total_excluded,
        },
        "by_market": {m: asdict(s) for m, s in summary.items()},
    }

    out_file = out_path / "universe_ledger.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
