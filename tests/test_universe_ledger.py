"""Tests for universe ledger and market definitions."""

from __future__ import annotations

from src.data.universe_ledger import (
    MARKET_DEFINITIONS,
    generate_universe_ledger_report,
    get_universe_summary,
)


def test_universe_counts_match_recovered_contract():
    summary = get_universe_summary()
    assert len(summary) == 6

    # US: 18 / 18
    assert summary["US"].requested_count == 18
    assert summary["US"].available_count == 18
    assert summary["US"].excluded_count == 0

    # India: 18 / 18
    assert summary["India"].requested_count == 18
    assert summary["India"].available_count == 18
    assert summary["India"].excluded_count == 0

    # China: 18 / 18
    assert summary["China"].requested_count == 18
    assert summary["China"].available_count == 18
    assert summary["China"].excluded_count == 0

    # Brazil: 18 requested, 15 available, 3 excluded
    assert summary["Brazil"].requested_count == 18
    assert summary["Brazil"].available_count == 15
    assert summary["Brazil"].excluded_count == 3
    assert set(summary["Brazil"].excluded_tickers.keys()) == {
        "CIEL3.SA",
        "JBSS3.SA",
        "EMBR3.SA",
    }

    # France: 18 requested, 17 available, 1 excluded
    assert summary["France"].requested_count == 18
    assert summary["France"].available_count == 17
    assert summary["France"].excluded_count == 1
    assert "STM.PA" in summary["France"].excluded_tickers

    # UK: 18 requested, 17 available, 1 excluded
    assert summary["UK"].requested_count == 18
    assert summary["UK"].available_count == 17
    assert summary["UK"].excluded_count == 1
    assert "AHT.L" in summary["UK"].excluded_tickers

    # Grand total: 108 requested, 103 available, 5 excluded
    tot_req = sum(s.requested_count for s in summary.values())
    tot_avail = sum(s.available_count for s in summary.values())
    tot_excl = sum(s.excluded_count for s in summary.values())
    assert tot_req == 108
    assert tot_avail == 103
    assert tot_excl == 5


def test_generate_universe_ledger_report(tmp_path):
    report = generate_universe_ledger_report(output_dir=tmp_path)
    assert report["totals"]["requested"] == 108
    assert report["totals"]["available"] == 103
    assert report["totals"]["excluded"] == 5
    assert (tmp_path / "universe_ledger.json").exists()
