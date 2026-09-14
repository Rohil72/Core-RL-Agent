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


def test_f5_near_identical_vectors_order_certified_against_reference_oracle():
    """Finding 5 acceptance test:
    Verify that 966-d float32 vectors near 1 with distances near 1e-14 do NOT suffer from
    floating-point cancellation drift in candidate selection.
    
    Reproduces auditor fixture:
    - ID 1: direct dist 5.68434189e-14, expanded formula: 0
    - ID 2: direct dist 1.42108547e-14, expanded formula: 0
    
    Direct reference selects ID 2. Uncertified expanded formula creates a tie and selects ID 1.
    Certified retrieval with reference refinement must select ID 2, matching reference oracle.
    Also tests exact ties, per-security caps, session spacing, and buffer expansion.
    """
    from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch

    q = np.ones(966, dtype=np.float32)
    v1 = q.copy()
    v2 = q.copy()
    # 2^-22 in one float32 element -> squared diff in float64 is 4 * 2^-46 = 5.68434189e-14
    # 2^-23 in one float32 element -> squared diff in float64 is 2^-46 = 1.42108547e-14
    v1[0] += np.float32(2**(-22))
    v2[0] += np.float32(2**(-23))

    # Record 1 has lower record_id (1), but Record 2 is strictly closer
    records = [
        BankRecord(record_id=1, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=v1, target_63=0.01, session_ordinal=10),
        BankRecord(record_id=2, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=v2, target_63=0.02, session_ordinal=20),
        # Add exact ties at larger distance
        BankRecord(record_id=100, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=q * 2.0, target_63=0.03, session_ordinal=30),
        BankRecord(record_id=50, security_id="SEC_D", session_origin="2015-01-04", session_126_maturity="2017-06-01", vector=q * 2.0, target_63=0.04, session_ordinal=40),
        # Add record violating spacing (< 21 sessions from SEC_C)
        BankRecord(record_id=10, security_id="SEC_C", session_origin="2015-01-05", session_126_maturity="2017-06-01", vector=q * 2.0, target_63=0.05, session_ordinal=35),
    ]

    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    # 1. Single-query reference oracle
    ref_res = retrieve_mem_sim(bank, q, "SEC_QUERY", k=2, max_per_security=3, min_spacing_sessions=21)
    # 2. Batched retrieval with reference refinement
    bat_res = retrieve_mem_sim_batch(bank, q.reshape(1, -1), ["SEC_QUERY"], k=2, max_per_security=3, min_spacing_sessions=21)[0]

    # Verify ID 2 is chosen FIRST, not ID 1!
    assert ref_res.neighbor_ids == [2, 1], f"Reference oracle expected [2, 1], got {ref_res.neighbor_ids}"
    assert bat_res.neighbor_ids == [2, 1], f"Batched refined retrieval expected [2, 1], got {bat_res.neighbor_ids}"
    assert bat_res.is_fallback is False
    assert pytest.approx(ref_res.prediction, rel=1e-5) == bat_res.prediction

    # 3. Test k=4: tests exact ties deterministically broken by record_id ascending (10 before 50 before 100)
    # and spacing rule (record 100 rejected because ordinal 30 is within 21 of SEC_C record 10 at ordinal 35)
    ref_k4 = retrieve_mem_sim(bank, q, "SEC_QUERY", k=4, max_per_security=3, min_spacing_sessions=21)
    bat_k4 = retrieve_mem_sim_batch(bank, q.reshape(1, -1), ["SEC_QUERY"], k=4, max_per_security=3, min_spacing_sessions=21)[0]

    assert ref_k4.neighbor_ids == [2, 1, 10, 50], f"Expected [2, 1, 10, 50], got {ref_k4.neighbor_ids}"
    assert bat_k4.neighbor_ids == [2, 1, 10, 50], f"Expected [2, 1, 10, 50], got {bat_k4.neighbor_ids}"
    assert bat_k4.neighbor_ids == ref_k4.neighbor_ids
    assert bat_k4.is_fallback == ref_k4.is_fallback
    assert pytest.approx(bat_k4.prediction, rel=1e-5) == ref_k4.prediction


def test_batched_query_chunking_and_gpu():
    """Finding 5 / C6: Verify that query chunking and optional GPU acceleration produce identical results."""
    from memory_study_v2.retrieval import retrieve_mem_sim_batch

    np.random.seed(99)
    N = 100
    records = [
        BankRecord(
            record_id=i + 1,
            security_id=f"SEC_{i % 10}",
            session_origin="2015-01-01",
            session_126_maturity="2017-06-01",
            vector=np.random.randn(966).astype(np.float32),
            target_63=float(np.random.randn() * 0.05),
            session_ordinal=i * 5,
        )
        for i in range(N)
    ]
    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    queries = np.random.randn(7, 966).astype(np.float32)
    sec_ids = [f"SEC_{i}" for i in range(7)]

    # Run without chunking (chunk_size=256)
    res_unchunked = retrieve_mem_sim_batch(bank, queries, sec_ids, k=5, query_chunk_size=256)
    # Run with small chunk size = 2 (forces multiple chunk iterations)
    res_chunked = retrieve_mem_sim_batch(bank, queries, sec_ids, k=5, query_chunk_size=2)

    assert len(res_unchunked) == 7
    assert len(res_chunked) == 7
    for u, c in zip(res_unchunked, res_chunked):
        assert u.neighbor_ids == c.neighbor_ids
        assert pytest.approx(u.prediction, rel=1e-5) == c.prediction

    # If GPU available, verify GPU path matches CPU path exactly
    try:
        import torch
        if torch.cuda.is_available():
            res_gpu = retrieve_mem_sim_batch(bank, queries, sec_ids, k=5, use_gpu=True)
            for u, g in zip(res_unchunked, res_gpu):
                assert u.neighbor_ids == g.neighbor_ids
                assert pytest.approx(u.prediction, rel=1e-5) == g.prediction
    except ImportError:
        pass
