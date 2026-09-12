#!/usr/bin/env python3
"""
C09 Temporal Separation Invariant Auditor & Discard Decomposition
================================================================
Verifies that every retrieved neighbor set satisfies:
|session_a - session_b| >= 21 trading sessions for ticker_a == ticker_b
with zero violations across all cells, trades, and systems.
Decomposes total candidate removals by system, market, cell, and trade.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
CANON_DIR = PROJECT_ROOT / "canonical_benchmark_outputs"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def verify_c09_invariants():
    print("=" * 85)
    print("C09 TEMPORAL SEPARATION & RETRIEVAL DEDUPLICATION AUDIT")
    print("=" * 85)

    nbr_path = CANON_DIR / "canonical_25_neighbor_ledger.csv"
    audit_path = CANON_DIR / "canonical_retrieval_deduplication_audit_ledger.csv"

    if not nbr_path.exists():
        print(f"[-] Missing: {nbr_path}")
        sys.exit(1)

    df_nbr = pd.read_csv(nbr_path)
    print(f"[+] Loaded {len(df_nbr)} neighbor rows across {df_nbr['trade_id'].nunique()} executed trades.")

    # 1. VERIFY TEMPORAL SEPARATION INVARIANT
    total_trades_checked = 0
    total_peer_pairs_checked = 0
    total_violations = 0
    violation_records = []

    for trade_id, group in df_nbr.groupby("trade_id"):
        total_trades_checked += 1
        sys_id = group["system"].iloc[0]
        mkt = group["market"].iloc[0]
        seed = group["seed"].iloc[0]

        # Check entity leakage first
        leaks = (group["query_ticker"] == group["neighbor_ticker"]).sum()
        if leaks > 0:
            total_violations += leaks
            violation_records.append({
                "trade_id": trade_id,
                "type": "entity_leakage",
                "count": int(leaks),
            })

        # Group by neighbor ticker
        by_ticker = {}
        for _, row in group.iterrows():
            t = str(row["neighbor_ticker"])
            sess = int(row["neighbor_session_index"])
            by_ticker.setdefault(t, []).append(sess)

        for t, sessions in by_ticker.items():
            if len(sessions) > 1:
                sorted_sess = sorted(sessions)
                for i in range(len(sorted_sess) - 1):
                    total_peer_pairs_checked += 1
                    gap = sorted_sess[i + 1] - sorted_sess[i]
                    if gap < 21:
                        total_violations += 1
                        violation_records.append({
                            "trade_id": trade_id,
                            "system": sys_id,
                            "market": mkt,
                            "seed": int(seed),
                            "ticker": t,
                            "session_1": sorted_sess[i],
                            "session_2": sorted_sess[i + 1],
                            "gap": gap,
                            "type": "temporal_separation_lt_21",
                        })

    print(f"\n[INVARIANT CHECK RESULT]")
    print(f"  Trades Evaluated:              {total_trades_checked}")
    print(f"  Same-Peer Pairs Evaluated:     {total_peer_pairs_checked}")
    print(f"  Total Invariant Violations:    {total_violations}")
    print(f"  Invariant Status:              {'PASS (100% Zero Violations)' if total_violations == 0 else 'FAIL'}")

    # 2. AUDIT CANDIDATE REMOVALS FROM AUDIT LEDGER
    removal_summary = {}
    if audit_path.exists():
        df_audit = pd.read_csv(audit_path)
        total_evaluated = len(df_audit)
        total_kept = int((df_audit["status"] == "KEPT").sum())
        total_discarded = int((df_audit["status"] == "DISCARDED").sum())
        discard_rate = total_discarded / max(1, total_evaluated) * 100

        print(f"\n[RETRIEVAL DEDUPLICATION METRICS]")
        print(f"  Total Candidates Evaluated:    {total_evaluated:,}")
        print(f"  Total Candidates Kept:         {total_kept:,}")
        print(f"  Total Candidates Discarded:    {total_discarded:,}")
        print(f"  Overall Discard Rate:          {discard_rate:.2f}%")

        # Breakdown by System
        by_sys = []
        for s_id, s_grp in df_audit.groupby("system"):
            n_eval = len(s_grp)
            n_disc = int((s_grp["status"] == "DISCARDED").sum())
            n_k = int((s_grp["status"] == "KEPT").sum())
            by_sys.append({
                "system": s_id,
                "evaluated": n_eval,
                "kept": n_k,
                "discarded": n_disc,
                "discard_rate_pct": round(n_disc / max(1, n_eval) * 100, 2),
            })
        df_by_sys = pd.DataFrame(by_sys)
        print("\n--- Candidate Removals by System ---")
        print(df_by_sys.to_string(index=False))

        # Breakdown by Market
        by_mkt = []
        for m_id, m_grp in df_audit.groupby("market"):
            n_eval = len(m_grp)
            n_disc = int((m_grp["status"] == "DISCARDED").sum())
            n_k = int((m_grp["status"] == "KEPT").sum())
            by_mkt.append({
                "market": m_id,
                "evaluated": n_eval,
                "kept": n_k,
                "discarded": n_disc,
                "discard_rate_pct": round(n_disc / max(1, n_eval) * 100, 2),
            })
        df_by_mkt = pd.DataFrame(by_mkt)
        print("\n--- Candidate Removals by Market ---")
        print(df_by_mkt.to_string(index=False))

        # Trade-level distribution of removals
        trade_discards = df_audit[df_audit["status"] == "DISCARDED"].groupby("trade_id").size()
        all_trade_ids = df_audit["trade_id"].unique()
        trade_discard_counts = [trade_discards.get(tid, 0) for tid in all_trade_ids]
        
        trade_dist = {
            "total_trades_logged": len(all_trade_ids),
            "min_discards_per_trade": int(np.min(trade_discard_counts)),
            "p25_discards_per_trade": float(np.percentile(trade_discard_counts, 25)),
            "median_discards_per_trade": float(np.median(trade_discard_counts)),
            "mean_discards_per_trade": round(float(np.mean(trade_discard_counts)), 2),
            "p75_discards_per_trade": float(np.percentile(trade_discard_counts, 75)),
            "max_discards_per_trade": int(np.max(trade_discard_counts)),
        }
        print("\n--- Trade-Level Discard Distribution ---")
        for k, v in trade_dist.items():
            print(f"  {k:30s}: {v}")

        # Cell breakdown
        by_cell = []
        for (s_id, m_id, seed), c_grp in df_audit.groupby(["system", "market", "seed"]):
            n_eval = len(c_grp)
            n_disc = int((c_grp["status"] == "DISCARDED").sum())
            n_k = int((c_grp["status"] == "KEPT").sum())
            by_cell.append({
                "system": s_id,
                "market": m_id,
                "seed": int(seed),
                "evaluated": n_eval,
                "kept": n_k,
                "discarded": n_disc,
                "discard_rate_pct": round(n_disc / max(1, n_eval) * 100, 2),
            })

        removal_summary = {
            "total_evaluated": total_evaluated,
            "total_kept": total_kept,
            "total_discarded": total_discarded,
            "discard_rate_pct": round(discard_rate, 2),
            "by_system": by_sys,
            "by_market": by_mkt,
            "trade_distribution": trade_dist,
            "by_cell": by_cell,
        }

    report = {
        "status": "PASS" if total_violations == 0 else "FAIL",
        "total_trades_checked": total_trades_checked,
        "total_peer_pairs_checked": total_peer_pairs_checked,
        "total_violations": total_violations,
        "violations": violation_records,
        "removal_summary": removal_summary,
    }

    report_file = REPORT_DIR / "c09_temporal_separation_audit_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Saved audit report to: {report_file}")

    if total_violations > 0:
        print("[-] AUDIT FAILED: Temporal separation violations detected!")
        sys.exit(1)
    else:
        print("[+] AUDIT PASSED: 100% C09 temporal separation compliance verified!")

if __name__ == "__main__":
    verify_c09_invariants()
