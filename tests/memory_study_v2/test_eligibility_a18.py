"""Acceptance & Regression Test for R03: Native session ordinal spacing invariant to bank order."""

import numpy as np
import pytest

from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.retrieval import retrieve_mem_random, retrieve_mem_sim


def test_spacing_uses_native_session_ordinals_interleaved():
    # Create bank with 2 securities interleaved:
    # Record 1: SEC_A, session_ordinal = 100
    # Record 2: SEC_B, session_ordinal = 50
    # Record 3: SEC_A, session_ordinal = 101  (Only 1 session after Record 1 for SEC_A!)
    # Record 4: SEC_A, session_ordinal = 130  (30 sessions after Record 1 for SEC_A)
    records = [
        BankRecord(
            record_id=1, security_id="SEC_A", session_origin="2015-01-01",
            session_126_maturity="2017-01-01", vector=np.zeros(966, dtype=np.float32),
            target_63=0.05, session_ordinal=100,
        ),
        BankRecord(
            record_id=2, security_id="SEC_B", session_origin="2015-01-02",
            session_126_maturity="2017-01-01", vector=np.ones(966, dtype=np.float32) * 0.1,
            target_63=0.02, session_ordinal=50,
        ),
        BankRecord(
            record_id=3, security_id="SEC_A", session_origin="2015-01-03",
            session_126_maturity="2017-01-01", vector=np.ones(966, dtype=np.float32) * 0.2,
            target_63=0.08, session_ordinal=101,  # Should be REJECTED by spacing rule (gap=1 < 21)!
        ),
        BankRecord(
            record_id=4, security_id="SEC_A", session_origin="2015-02-15",
            session_126_maturity="2017-01-01", vector=np.ones(966, dtype=np.float32) * 0.3,
            target_63=0.10, session_ordinal=130,  # Should be ACCEPTED (gap=30 >= 21)
        ),
    ]
    # Add dummy records to reach k=3
    for i in range(5, 10):
        records.append(BankRecord(
            record_id=i, security_id=f"SEC_DUMMY_{i}", session_origin="2015-01-01",
            session_126_maturity="2017-01-01", vector=np.ones(966, dtype=np.float32) * float(i),
            target_63=0.01, session_ordinal=1,
        ))

    bank = MemoryBank(records)
    query_vec = np.zeros(966, dtype=np.float32)
    # Query is SEC_QUERY (different security)
    res = retrieve_mem_sim(bank, query_vec, query_security_id="SEC_QUERY", k=3, min_spacing_sessions=21)

    assert 1 in res.neighbor_ids, "Record 1 should be accepted"
    assert 3 not in res.neighbor_ids, "Record 3 must be REJECTED by native session ordinal spacing (gap=1)!"
    assert 4 in res.neighbor_ids, "Record 4 should be accepted (gap=30 >= 21)"


def test_spacing_invariant_to_bank_physical_reordering():
    rng = np.random.default_rng(42)
    records = []
    for i in range(50):
        records.append(BankRecord(
            record_id=i + 1,
            security_id=f"SEC_{i % 5}",
            session_origin=f"2015-01-{(i % 28) + 1:02d}",
            session_126_maturity="2017-01-01",
            vector=rng.normal(0, 1, 966).astype(np.float32),
            target_63=0.03,
            session_ordinal=i * 5,
        ))

    bank_ordered = MemoryBank(records)
    # Permute the records before building second bank
    permuted_records = [records[idx] for idx in rng.permutation(len(records))]
    bank_permuted = MemoryBank(permuted_records)

    q = rng.normal(0, 1, 966).astype(np.float32)
    res1 = retrieve_mem_sim(bank_ordered, q, query_security_id="SEC_QUERY", k=10)
    res2 = retrieve_mem_sim(bank_permuted, q, query_security_id="SEC_QUERY", k=10)

    assert res1.neighbor_ids == res2.neighbor_ids, "Physical bank ordering must not change accepted neighbor IDs!"
