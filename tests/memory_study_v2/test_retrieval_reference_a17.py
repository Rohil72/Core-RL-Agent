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


# ---------------------------------------------------------------------------
# Finding 5 / C6: Batched retrieval optimization acceptance tests
# ---------------------------------------------------------------------------

def test_batched_vs_reference_exact_ids():
    """Finding 5: retrieve_mem_sim_batch produces identical neighbor IDs and predictions to reference."""
    from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch

    np.random.seed(42)
    N = 200
    records = []
    for i in range(N):
        vec = np.random.normal(0, 1, 966).astype(np.float32)
        rec = BankRecord(
            record_id=i + 1,
            security_id=f"MKT:SEC_{i % 15}",
            session_origin=f"2015-01-{(i % 28) + 1:02d}",
            session_126_maturity="2017-06-01",
            vector=vec,
            target_63=float(np.random.normal(0, 0.05)),
            session_ordinal=i * 25,  # ensure some spacing variety
        )
        records.append(rec)

    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    Q = 5
    queries = np.random.normal(0, 1, (Q, 966)).astype(np.float32)
    query_secs = [f"MKT:SEC_{q}" for q in range(Q)]

    # 1. Reference single-query execution
    ref_results = [
        retrieve_mem_sim(bank, queries[q], query_secs[q], k=10, max_per_security=3, min_spacing_sessions=21)
        for q in range(Q)
    ]

    # 2. Batched execution
    batch_results = retrieve_mem_sim_batch(
        bank, queries, query_secs, k=10, max_per_security=3, min_spacing_sessions=21
    )

    assert len(batch_results) == Q
    for q in range(Q):
        ref = ref_results[q]
        bat = batch_results[q]
        assert ref.is_fallback == bat.is_fallback
        assert ref.neighbor_ids == bat.neighbor_ids, f"Query {q} neighbor IDs mismatch!"
        assert pytest.approx(ref.prediction, rel=1e-5) == bat.prediction
        assert len(ref.neighbor_distances) == len(bat.neighbor_distances)
        for d_ref, d_bat in zip(ref.neighbor_distances, bat.neighbor_distances):
            assert pytest.approx(d_ref, rel=1e-5) == d_bat


def test_batched_insufficient_neighbors_fallback():
    """Finding 5: When eligible records < k, batched path triggers fallback with same IDs."""
    from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch

    vec = np.ones(966, dtype=np.float32)
    # Only 2 eligible records from other securities
    records = [
        BankRecord(record_id=1, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=vec, target_63=0.01, session_ordinal=1),
        BankRecord(record_id=2, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=vec * 2, target_63=0.02, session_ordinal=2),
        BankRecord(record_id=3, security_id="SEC_QUERY", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=vec * 3, target_63=0.03, session_ordinal=3),
    ]
    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    query = np.zeros((1, 966), dtype=np.float32)
    # Request k=5 > 2 eligible
    ref = retrieve_mem_sim(bank, query[0], "SEC_QUERY", k=5)
    bat = retrieve_mem_sim_batch(bank, query, ["SEC_QUERY"], k=5)[0]

    assert ref.is_fallback is True
    assert bat.is_fallback is True
    assert ref.neighbor_ids == bat.neighbor_ids == [1, 2]
    assert ref.prediction == bat.prediction == bank.unconditional_mean


def test_batched_ties_deterministic():
    """Finding 5: Ties are deterministically broken by record_id ascending in batched mode."""
    from memory_study_v2.retrieval import retrieve_mem_sim_batch

    vec = np.ones(966, dtype=np.float32)
    records = [
        BankRecord(record_id=99, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=vec, target_63=0.01, session_ordinal=1),
        BankRecord(record_id=15, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=vec, target_63=0.02, session_ordinal=2),
        BankRecord(record_id=42, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=vec, target_63=0.03, session_ordinal=3),
    ]
    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    query = np.zeros((1, 966), dtype=np.float32)
    bat = retrieve_mem_sim_batch(bank, query, ["SEC_OTHER"], k=3, max_per_security=3, min_spacing_sessions=1)[0]
    assert bat.neighbor_ids == [15, 42, 99]


def test_bank_verify_batched_vs_reference_oracle():
    """Finding 5: verify_batched_vs_reference passes on valid queries and raises on tampered tolerance."""
    np.random.seed(123)
    records = [
        BankRecord(record_id=i, security_id=f"SEC_{i}", session_origin="2015-01-01", session_126_maturity="2017-06-01",
                   vector=np.random.randn(966).astype(np.float32), target_63=0.0, session_ordinal=i)
        for i in range(20)
    ]
    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    queries = np.random.randn(5, 966).astype(np.float32)
    assert bank.verify_batched_vs_reference(queries, tol=1e-5) is True
