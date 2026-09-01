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
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
            "AVGO", "CRM", "ORCL", "AMD", "ADBE", "INTU",
            "ISRG", "AMGN", "REGN", "LMT", "NOC", "V",
        ],
        "excluded": {},
    },
    "India": {
        "exchange": "NSE",
        "currency": "INR",
        "benchmark": "^NSEI",
        "execution_cost_bps": 20.0,
        "tickers": [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "BHARTIARTL.NS", "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
            "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "TATAMOTORS.NS", "AXISBANK.NS", "NTPC.NS", "TITAN.NS",
        ],
        "excluded": {},
    },
    "China": {
        "exchange": "SSE",
        "currency": "CNY",
        "benchmark": "000300.SS",
        "execution_cost_bps": 20.0,
        "tickers": [
            "600519.SS", "601398.SS", "601288.SS", "601939.SS", "601857.SS",
            "600036.SS", "601988.SS", "600276.SS", "601318.SS", "601088.SS",
            "600900.SS", "600030.SS", "601668.SS", "600028.SS", "601899.SS",
            "601328.SS", "601998.SS", "600019.SS",
        ],
        "excluded": {},
    },
    "Brazil": {
        "exchange": "B3",
        "currency": "BRL",
        "benchmark": "^BVSP",
        "execution_cost_bps": 15.0,
        "tickers": [
            "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA",
            "B3SA3.SA", "BBAS3.SA", "WEGE3.SA", "RENT3.SA", "SUZB3.SA",
            "GGBR4.SA", "RDOR3.SA", "LREN3.SA", "EQTL3.SA", "PRIO3.SA",
        ],
        "excluded": {
            "CIEL3.SA": "historical reason unavailable",
            "JBSS3.SA": "historical reason unavailable",
            "EMBR3.SA": "historical reason unavailable",
        },
    },
    "France": {
        "exchange": "EURONEXT_PARIS",
        "currency": "EUR",
        "benchmark": "^FCHI",
        "execution_cost_bps": 20.0,
        "tickers": [
            "MC.PA", "OR.PA", "TTE.PA", "SAN.PA", "AIR.PA",
            "SU.PA", "AI.PA", "RMS.PA", "BNP.PA", "EL.PA",
            "CS.PA", "CDIB.PA", "SAF.PA", "DG.PA", "KER.PA",
            "BN.PA", "DSY.PA",
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
            "SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "BP.L",
            "REL.L", "DGE.L", "GSK.L", "RIO.L", "BATS.L",
            "LSEG.L", "CPG.L", "PRU.L", "VOD.L", "BA.L",
            "NG.L", "EXPN.L",
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
