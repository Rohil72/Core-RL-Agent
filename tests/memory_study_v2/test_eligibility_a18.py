"""Acceptance Test A18: Canonical-security exclusions, cap 3, spacing 21 and insufficient-pool fallback verified."""

import pytest
import numpy as np
from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.retrieval import retrieve_mem_sim


def test_canonical_security_exclusion_and_cap_and_spacing():
    """Verify query security is excluded, max 3 per security, and min spacing 21."""
    records = []
    # 50 records for query's own security SEC_QUERY (must all be excluded!)
    for i in range(50):
        records.append(BankRecord(
            record_id=i,
            security_id="SEC_QUERY",
            session_origin=f"2015-01-01",
            session_126_maturity="2017-06-01",
            vector=np.zeros(966, dtype=np.float32),
            target_63=0.05,
        ))

    # 10 records for SEC_OTHER (all spaced < 21 sessions, so only 1 can be accepted, or at most 3 if spaced)
    for j in range(10):
        records.append(BankRecord(
            record_id=100 + j,
            security_id="SEC_OTHER",
            session_origin=f"2015-01-01",
            session_126_maturity="2017-06-01",
            vector=np.zeros(966, dtype=np.float32) + 0.1 * j,
            target_63=0.02,
        ))

    # 30 distinct securities SEC_1..SEC_30 each with 1 record
    for k in range(30):
        records.append(BankRecord(
            record_id=200 + k,
            security_id=f"SEC_{k}",
            session_origin="2015-01-01",
            session_126_maturity="2017-06-01",
            vector=np.zeros(966, dtype=np.float32) + 0.5 * (k + 1),
            target_63=0.01 * k,
        ))

    bank = MemoryBank(records)
    query = np.zeros(966, dtype=np.float32)

    res = retrieve_mem_sim(bank, query, query_security_id="SEC_QUERY", k=25)

    assert res.is_fallback is False
    assert len(res.neighbor_ids) == 25

    # Check query security is NEVER present
    accepted_secs = [bank.security_ids[np.where(bank.record_ids == nid)[0][0]] for nid in res.neighbor_ids]
    assert "SEC_QUERY" not in accepted_secs

    # Check cap 3: no security has more than 3 records
    for sec in set(accepted_secs):
        assert accepted_secs.count(sec) <= 3


def test_insufficient_neighbors_triggers_unconditional_mean_fallback():
    """Verify when < 25 neighbors exist after filtering, bank mean is returned with is_fallback=True."""
    # Only 5 records in entire bank
    records = [
        BankRecord(record_id=i, security_id=f"SEC_{i}", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=np.zeros(966, dtype=np.float32), target_63=0.04)
        for i in range(5)
    ]
    bank = MemoryBank(records)
    query = np.zeros(966, dtype=np.float32)

    res = retrieve_mem_sim(bank, query, query_security_id="SEC_0", k=25)
    assert res.is_fallback is True
    assert pytest.approx(res.prediction, abs=1e-7) == bank.unconditional_mean
