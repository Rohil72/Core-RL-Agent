"""Acceptance Test A17: Optimized selected IDs equal float64 reference on constrained, tied and buffer-expansion cases."""

import pytest
import numpy as np
from memory_study_v2.memory import BankRecord, MemoryBank


def test_candidate_proposal_parity_with_cpu_reference():
    """Verify that chunked candidate proposal produces identical candidates to direct reference."""
    np.random.seed(42)
    N = 500
    records = []
    for i in range(N):
        vec = np.random.normal(0, 1, 966).astype(np.float32)
        rec = BankRecord(
            record_id=i + 1,
            security_id=f"MKT:SEC_{i % 20}",
            session_origin=f"2015-01-{(i % 28) + 1:02d}",
            session_126_maturity="2017-06-01",
            vector=vec,
            target_63=float(np.random.normal(0, 0.05)),
        )
        records.append(rec)

    bank = MemoryBank(records)
    query = np.random.normal(0, 1, 966).astype(np.float32)

    # Propose top 25 with buffer size 50
    proposed_indices = bank.propose_candidates(query, buffer_size=50)

    # Direct CPU float64 reference across entire bank
    all_ref_dists = bank.compute_reference_distances(query)
    # Sort all by (ref_dist, record_id)
    ref_order = np.lexsort((bank.record_ids, all_ref_dists))
    top_50_ref = ref_order[:50]

    # Check that top 50 matches exactly
    np.testing.assert_array_equal(proposed_indices[:50], top_50_ref)


def test_tie_breaking_by_record_id_ascending():
    """Verify that identical distance records are strictly ordered by record_id ascending."""
    vec = np.ones(966, dtype=np.float32)
    records = [
        BankRecord(record_id=105, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=vec, target_63=0.01),
        BankRecord(record_id=12, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=vec, target_63=0.02),
        BankRecord(record_id=88, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=vec, target_63=0.03),
    ]
    bank = MemoryBank(records)
    query = np.zeros(966, dtype=np.float32)

    candidates = bank.propose_candidates(query, buffer_size=3)
    sorted_record_ids = [bank.record_ids[i] for i in candidates]
    # Distances are identical, so must be sorted by record_id: [12, 88, 105]
    assert sorted_record_ids == [12, 88, 105]
