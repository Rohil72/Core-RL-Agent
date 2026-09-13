"""Acceptance Test A19: Plain kNN removes only cap/spacing; random expansion continues same permutation."""

import pytest
import numpy as np
from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.retrieval import retrieve_knn_plain, retrieve_mem_random


def test_knn_plain_removes_cap_and_spacing():
    """Verify KNN_PLAIN allows > 3 records from the same security."""
    records = []
    # 25 records from SEC_A with closest distance
    for i in range(25):
        records.append(BankRecord(
            record_id=i,
            security_id="SEC_A",
            session_origin="2015-01-01",
            session_126_maturity="2017-06-01",
            vector=np.zeros(966, dtype=np.float32),
            target_63=0.02,
        ))
    # 25 records from SEC_B with larger distance
    for j in range(25):
        records.append(BankRecord(
            record_id=50 + j,
            security_id="SEC_B",
            session_origin="2015-01-01",
            session_126_maturity="2017-06-01",
            vector=np.ones(966, dtype=np.float32),
            target_63=0.05,
        ))

    bank = MemoryBank(records)
    query = np.zeros(966, dtype=np.float32)

    res = retrieve_knn_plain(bank, query, query_security_id="QUERY", k=25)
    # KNN_PLAIN should take all 25 from SEC_A
    assert len(res.neighbor_ids) == 25
    accepted_secs = [bank.security_ids[np.where(bank.record_ids == nid)[0][0]] for nid in res.neighbor_ids]
    assert accepted_secs.count("SEC_A") == 25


def test_mem_random_deterministic_and_reproducible():
    """Verify MEM_RANDOM with master seed 1001 produces deterministic results."""
    records = [
        BankRecord(record_id=i, security_id=f"SEC_{i}", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=np.zeros(966, dtype=np.float32), target_63=float(i) * 0.01)
        for i in range(50)
    ]
    bank = MemoryBank(records)

    res1 = retrieve_mem_random(
        bank, query_security_id="QUERY", bank_hash="hash_bank_1",
        fold_year=2020, query_id="query_1", master_seed=1001, k=25
    )
    res2 = retrieve_mem_random(
        bank, query_security_id="QUERY", bank_hash="hash_bank_1",
        fold_year=2020, query_id="query_1", master_seed=1001, k=25
    )
    # Identical results
    assert res1.neighbor_ids == res2.neighbor_ids
    assert res1.prediction == res2.prediction
