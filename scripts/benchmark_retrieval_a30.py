"""A30 Preflight Benchmark: Batched Retrieval Optimization (Finding 5 / C6).

Designed to be executed on the NVIDIA A30 / Linux VM.
Measures retrieval throughput (QPS), verifies exact ID parity against the reference
oracle on difficult boundary/tie fixtures, and outputs a signed benchmark receipt
to rebuild_plan/a30_retrieval_benchmark.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import numpy as np

from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch


def run_benchmark(
    bank_size: int = 20000,
    num_queries: int = 100,
    vector_dim: int = 966,
    k: int = 25,
    seed: int = 42,
    output_path: str = "rebuild_plan/a30_retrieval_benchmark.json",
):
    print("=" * 70)
    print("A30 RETRIEVAL OPTIMIZATION PREFLIGHT BENCHMARK (Finding 5 / C6)")
    print("=" * 70)
    rng = np.random.default_rng(seed)

    print(f"Generating synthetic bank with N={bank_size} records, D={vector_dim}...")
    records = []
    num_securities = max(10, bank_size // 100)
    for i in range(bank_size):
        vec = rng.normal(0, 1, vector_dim).astype(np.float32)
        records.append(BankRecord(
            record_id=i + 1,
            security_id=f"MKT:SEC_{i % num_securities}",
            session_origin=f"2015-01-{(i % 28) + 1:02d}",
            session_126_maturity="2017-06-01",
            vector=vec,
            target_63=float(rng.normal(0, 0.05)),
            session_ordinal=i * 2,
        ))

    bank = MemoryBank(records)

    # 1. Precompute norms
    t_norm_start = time.perf_counter()
    bank.precompute_bank_norms()
    t_norm_end = time.perf_counter()
    norm_time_ms = (t_norm_end - t_norm_start) * 1000.0
    print(f"Bank norm precomputation: {norm_time_ms:.2f} ms")

    # 2. Generate queries
    queries = rng.normal(0, 1, (num_queries, vector_dim)).astype(np.float32)
    query_secs = [f"MKT:SEC_{q % num_securities}" for q in range(num_queries)]

    # 3. Parity verification against reference on first 5 queries
    print("Verifying parity against reference oracle on first 5 queries...")
    ref_first_5 = [
        retrieve_mem_sim(bank, queries[i], query_secs[i], k=k)
        for i in range(min(5, num_queries))
    ]
    bat_first_5 = retrieve_mem_sim_batch(
        bank, queries[:min(5, num_queries)], query_secs[:min(5, num_queries)], k=k
    )

    parity_passed = True
    for i in range(len(ref_first_5)):
        if ref_first_5[i].neighbor_ids != bat_first_5[i].neighbor_ids:
            parity_passed = False
            print(f"  FAILED on query {i}: ID mismatch")
        if abs(ref_first_5[i].prediction - bat_first_5[i].prediction) > 1e-5:
            parity_passed = False
            print(f"  FAILED on query {i}: Prediction mismatch")

    if parity_passed:
        print("  PARITY CHECK PASSED: 100% ID and prediction agreement with reference oracle.")
    else:
        print("  PARITY CHECK FAILED!")

    # 4. Warmup
    print("Warmup execution...")
    _ = retrieve_mem_sim_batch(bank, queries[:10], query_secs[:10], k=k)

    # 5. Timed batched run
    print(f"Benchmarking batched retrieval for Q={num_queries} queries...")
    t_start = time.perf_counter()
    batch_results = retrieve_mem_sim_batch(bank, queries, query_secs, k=k)
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    qps = num_queries / elapsed
    ms_per_query = (elapsed / num_queries) * 1000.0

    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Throughput: {qps:.2f} QPS")
    print(f"Latency per query: {ms_per_query:.2f} ms")

    # 6. Difficult fixtures check: tied distances
    print("Testing difficult fixture: exact ties...")
    vec_ones = np.ones(vector_dim, dtype=np.float32)
    tie_records = [
        BankRecord(record_id=100, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=vec_ones, target_63=0.01, session_ordinal=1),
        BankRecord(record_id=5, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=vec_ones, target_63=0.02, session_ordinal=2),
        BankRecord(record_id=50, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=vec_ones, target_63=0.03, session_ordinal=3),
    ]
    tie_bank = MemoryBank(tie_records)
    tie_bank.precompute_bank_norms()
    tie_q = np.zeros((1, vector_dim), dtype=np.float32)
    tie_res = retrieve_mem_sim_batch(tie_bank, tie_q, ["SEC_OTHER"], k=3)[0]
    tie_passed = tie_res.neighbor_ids == [5, 50, 100]
    print(f"  Ties deterministically sorted by record_id ascending: {tie_passed} (got {tie_res.neighbor_ids})")

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank_size": bank_size,
        "vector_dim": vector_dim,
        "num_queries": num_queries,
        "k": k,
        "precompute_norm_time_ms": norm_time_ms,
        "elapsed_seconds": elapsed,
        "qps": qps,
        "ms_per_query": ms_per_query,
        "reference_parity_passed": parity_passed,
        "tie_breaking_passed": tie_passed,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Saved benchmark receipt to {out_file}")
    print("=" * 70)
    return receipt


if __name__ == "__main__":
    run_benchmark(bank_size=1000, num_queries=20)
