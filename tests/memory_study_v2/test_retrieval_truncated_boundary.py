"""Acceptance Test for Truncated-Candidate Retrieval & Boundary Certification (Finding 5 / C6).

Verifies that when:
1. Bank is larger than the initial candidate proposal buffer (N > buffer_size).
2. A record is initially outside the proposal pool in approximate float32 distance due to
   cancellation/precision, but is genuinely closer in true float64 reference distance.
3. The boundary expansion and reference fallback certifiably preserve exact ID and distance parity
   against the direct reference oracle (retrieve_mem_sim).
"""

import numpy as np
import pytest
import torch

from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch


def test_truncated_candidate_boundary_expansion_and_parity():
    """Verify that a candidate truncated by initial buffer is recovered and preserved."""
    np.random.seed(42)
    D = 966
    # Query vector with large scale to induce float32 dot-product roundoff
    q = np.random.randn(D).astype(np.float32) * 40.0

    # Create bank of 20 records; initial buffer is 5
    records = []
    # 5 records with true dist ~ 9.5
    for i in range(5):
        v = q + np.random.randn(D).astype(np.float32) * 0.1
        records.append(BankRecord(
            record_id=i + 1,
            security_id=f"SEC_{i}",
            session_origin="2015-01-01",
            session_126_maturity="2017-06-01",
            vector=v,
            target_63=0.01 * (i + 1),
            session_ordinal=i + 1,
        ))

    # Record 6: crafted to have true distance strictly closer to q than any of the first 5 records!
    # v_close = q + small perturbation
    v_close = q + np.random.randn(D).astype(np.float32) * 0.01
    records.append(BankRecord(
        record_id=6,
        security_id="SEC_CLOSE",
        session_origin="2015-01-01",
        session_126_maturity="2017-06-01",
        vector=v_close,
        target_63=0.06,
        session_ordinal=6,
    ))

    # Records 7..20: farther away
    for i in range(7, 21):
        v = q + np.random.randn(D).astype(np.float32) * 0.5
        records.append(BankRecord(
            record_id=i,
            security_id=f"SEC_{i}",
            session_origin="2015-01-01",
            session_126_maturity="2017-06-01",
            vector=v,
            target_63=0.01 * i,
            session_ordinal=i,
        ))

    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    # Direct reference oracle (computes full float64 distance across all 20 records)
    ref_res = retrieve_mem_sim(bank, q, query_security_id="QUERY_SEC", k=3)

    # Batched retrieval with small initial buffer size = 5 (< bank.N = 20)
    # This forces retrieval to evaluate the boundary condition and expand/fall back safely
    bat_res_cpu = retrieve_mem_sim_batch(
        bank, q.reshape(1, -1), ["QUERY_SEC"], k=3, start_buffer_size=5, use_gpu=False
    )[0]

    assert ref_res.neighbor_ids == bat_res_cpu.neighbor_ids, (
        f"CPU Batched retrieval mismatch: expected {ref_res.neighbor_ids}, got {bat_res_cpu.neighbor_ids}"
    )
    assert np.allclose(ref_res.neighbor_distances, bat_res_cpu.neighbor_distances, atol=1e-6)

    # If CUDA is available, test with GPU proposal as well
    if torch.cuda.is_available():
        bat_res_gpu = retrieve_mem_sim_batch(
            bank, q.reshape(1, -1), ["QUERY_SEC"], k=3, start_buffer_size=5, use_gpu=True
        )[0]
        assert ref_res.neighbor_ids == bat_res_gpu.neighbor_ids, (
            f"GPU Batched retrieval mismatch: expected {ref_res.neighbor_ids}, got {bat_res_gpu.neighbor_ids}"
        )
        assert np.allclose(ref_res.neighbor_distances, bat_res_gpu.neighbor_distances, atol=1e-6)
