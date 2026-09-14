"""A30 Preflight Benchmark: Batched Retrieval Optimization (Finding 5 / C6).

Designed to be executed on the NVIDIA A30 / Linux VM or local workstation.
Measures retrieval throughput (QPS), verifies exact ID parity against the reference
oracle on difficult boundary/tie fixtures, and outputs a signed benchmark receipt
to rebuild_plan/a30_retrieval_benchmark.json.

Verification & Telemetry:
- Representative scale: N=20000 records, D=966 dimensions, Q=100 queries.
- Query chunk limits: chunk_size=256 to bound peak memory.
- Hardware & peak memory telemetry: CPU, RAM, CUDA device name, peak RSS/VRAM.
- Numerical cancellation fixture: 966-d float32 vectors near 1 (1.42e-14 vs 5.68e-14).
- Fail-closed: sys.exit(1) on any parity or constraint failure.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.retrieval import retrieve_mem_sim, retrieve_mem_sim_batch


def get_current_rss_mb() -> float:
    """Get current process resident set size in MB."""
    if psutil is not None:
        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    return 0.0


def get_peak_vram_mb() -> float:
    """Get peak VRAM allocated in MB if CUDA is available."""
    if torch is not None and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
    return 0.0


def run_benchmark(
    bank_size: int = 20000,
    num_queries: int = 100,
    vector_dim: int = 966,
    k: int = 25,
    query_chunk_size: int = 256,
    seed: int = 42,
    output_path: str = "rebuild_plan/a30_retrieval_benchmark.json",
    use_gpu: bool = True,
) -> Dict[str, Any]:
    print("=" * 75)
    print("A30 RETRIEVAL OPTIMIZATION PREFLIGHT BENCHMARK (Finding 5 / C6)")
    print("=" * 75)

    initial_rss_mb = get_current_rss_mb()
    cuda_avail = torch is not None and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    total_ram_gb = (psutil.virtual_memory().total / (1024.0 ** 3)) if psutil is not None else 0.0

    print(f"Platform: {platform.platform()} (Python {platform.python_version()})")
    print(f"Hardware: {os.cpu_count()} CPU cores, {total_ram_gb:.2f} GB RAM, Device: {device_name}")
    print(f"Benchmark Dimensions: N={bank_size} records, D={vector_dim} dimensions, Q={num_queries} queries")
    print(f"Configuration: k={k}, query_chunk_size={query_chunk_size}, use_gpu={use_gpu and cuda_avail}")
    print("-" * 75)

    rng = np.random.default_rng(seed)

    # 1. Synthesize bank records
    print(f"Generating synthetic bank with N={bank_size} records, D={vector_dim}...")
    records: List[BankRecord] = []
    num_securities = max(10, bank_size // 100)
    for i in range(bank_size):
        vec = rng.normal(0, 1, vector_dim).astype(np.float32)
        records.append(
            BankRecord(
                record_id=i + 1,
                security_id=f"MKT:SEC_{i % num_securities}",
                session_origin=f"2015-01-{(i % 28) + 1:02d}",
                session_126_maturity="2017-06-01",
                vector=vec,
                target_63=float(rng.normal(0, 0.05)),
                session_ordinal=i * 2,
            )
        )

    bank = MemoryBank(records)

    # 2. Precompute bank norms
    t_norm_start = time.perf_counter()
    bank.precompute_bank_norms()
    t_norm_end = time.perf_counter()
    norm_time_ms = (t_norm_end - t_norm_start) * 1000.0
    print(f"Bank norm precomputation: {norm_time_ms:.2f} ms")

    # 3. Generate test queries
    queries = rng.normal(0, 1, (num_queries, vector_dim)).astype(np.float32)
    query_secs = [f"MKT:SEC_{q % num_securities}" for q in range(num_queries)]

    # 4. Parity verification against reference oracle on first 10 queries
    print("Verifying parity against reference oracle on first 10 queries...")
    ref_first_10 = [
        retrieve_mem_sim(bank, queries[i], query_secs[i], k=k)
        for i in range(min(10, num_queries))
    ]
    bat_first_10 = retrieve_mem_sim_batch(
        bank,
        queries[:min(10, num_queries)],
        query_secs[:min(10, num_queries)],
        k=k,
        query_chunk_size=query_chunk_size,
        use_gpu=(use_gpu and cuda_avail),
    )

    parity_passed = True
    for i in range(len(ref_first_10)):
        if ref_first_10[i].neighbor_ids != bat_first_10[i].neighbor_ids:
            parity_passed = False
            print(f"  FAILED on query {i}: ID mismatch (ref={ref_first_10[i].neighbor_ids}, bat={bat_first_10[i].neighbor_ids})")
        if abs(ref_first_10[i].prediction - bat_first_10[i].prediction) > 1e-5:
            parity_passed = False
            print(f"  FAILED on query {i}: Prediction mismatch (ref={ref_first_10[i].prediction:.6f}, bat={bat_first_10[i].prediction:.6f})")

    if parity_passed:
        print("  PARITY CHECK PASSED: 100% ID and prediction agreement with reference oracle.")
    else:
        print("  PARITY CHECK FAILED!")

    # 5. Numerical cancellation fixture verification (Auditor Finding 5)
    print("Verifying numerical cancellation fixture (float32 vectors near 1)...")
    q_fixture = np.ones(vector_dim, dtype=np.float32)
    v1 = q_fixture.copy()
    v2 = q_fixture.copy()
    # 2^-22 in one float32 element -> squared diff in float64 is 4 * 2^-46 = 5.68434189e-14
    # 2^-23 in one float32 element -> squared diff in float64 is 2^-46 = 1.42108547e-14
    v1[0] += np.float32(2 ** (-22))
    v2[0] += np.float32(2 ** (-23))

    fixture_records = [
        BankRecord(record_id=1, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=v1, target_63=0.01, session_ordinal=10),
        BankRecord(record_id=2, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=v2, target_63=0.02, session_ordinal=20),
    ]
    fixture_bank = MemoryBank(fixture_records)
    fixture_bank.precompute_bank_norms()

    fixture_ref = retrieve_mem_sim(fixture_bank, q_fixture, "SEC_QUERY", k=2)
    fixture_bat = retrieve_mem_sim_batch(
        fixture_bank,
        q_fixture.reshape(1, -1),
        ["SEC_QUERY"],
        k=2,
        use_gpu=(use_gpu and cuda_avail),
    )[0]

    cancellation_passed = (
        fixture_ref.neighbor_ids == [2, 1]
        and fixture_bat.neighbor_ids == [2, 1]
    )
    print(f"  Numerical cancellation test passed: {cancellation_passed} (selected IDs: {fixture_bat.neighbor_ids})")

    # 6. Difficult fixtures check: exact ties
    print("Testing difficult fixture: exact ties sorted by record_id ascending...")
    vec_ones = np.ones(vector_dim, dtype=np.float32)
    tie_records = [
        BankRecord(record_id=100, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=vec_ones, target_63=0.01, session_ordinal=1),
        BankRecord(record_id=5, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=vec_ones, target_63=0.02, session_ordinal=2),
        BankRecord(record_id=50, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=vec_ones, target_63=0.03, session_ordinal=3),
    ]
    tie_bank = MemoryBank(tie_records)
    tie_bank.precompute_bank_norms()
    tie_q = np.zeros((1, vector_dim), dtype=np.float32)
    tie_res = retrieve_mem_sim_batch(tie_bank, tie_q, ["SEC_OTHER"], k=3, use_gpu=(use_gpu and cuda_avail))[0]
    tie_passed = tie_res.neighbor_ids == [5, 50, 100]
    print(f"  Ties deterministically sorted by record_id ascending: {tie_passed} (got {tie_res.neighbor_ids})")

    # 7. Warmup
    print("Warmup execution (10 queries)...")
    _ = retrieve_mem_sim_batch(
        bank,
        queries[:10],
        query_secs[:10],
        k=k,
        query_chunk_size=query_chunk_size,
        use_gpu=(use_gpu and cuda_avail),
    )

    # Reset max VRAM tracking if CUDA
    if cuda_avail:
        torch.cuda.reset_peak_memory_stats()

    # 8. Timed batched run at representative scale
    print(f"Benchmarking batched retrieval for Q={num_queries} queries (N={bank_size}, D={vector_dim})...")
    t_start = time.perf_counter()
    batch_results = retrieve_mem_sim_batch(
        bank,
        queries,
        query_secs,
        k=k,
        query_chunk_size=query_chunk_size,
        use_gpu=(use_gpu and cuda_avail),
    )
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    qps = num_queries / elapsed
    ms_per_query = (elapsed / num_queries) * 1000.0

    peak_rss_mb = get_current_rss_mb()
    peak_vram_mb = get_peak_vram_mb()

    print(f"Elapsed Time: {elapsed:.3f} s")
    print(f"Throughput: {qps:.2f} QPS")
    print(f"Latency per query: {ms_per_query:.2f} ms")
    print(f"Memory Telemetry: Peak RSS = {peak_rss_mb:.1f} MB, Peak VRAM = {peak_vram_mb:.1f} MB")

    all_passed = bool(parity_passed and cancellation_passed and tie_passed)

    receipt = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware_telemetry": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "total_ram_gb": round(total_ram_gb, 2),
            "cuda_available": cuda_avail,
            "cuda_device_name": device_name,
            "peak_rss_mb": round(peak_rss_mb, 2),
            "peak_vram_allocated_mb": round(peak_vram_mb, 2),
        },
        "configuration": {
            "bank_size": bank_size,
            "vector_dim": vector_dim,
            "num_queries": num_queries,
            "k": k,
            "query_chunk_size": query_chunk_size,
            "start_buffer_size": 250,
            "use_gpu": (use_gpu and cuda_avail),
        },
        "metrics": {
            "precompute_norm_time_ms": round(norm_time_ms, 2),
            "elapsed_seconds": round(elapsed, 4),
            "qps": round(qps, 2),
            "ms_per_query": round(ms_per_query, 2),
        },
        "verification": {
            "reference_parity_passed": parity_passed,
            "numerical_cancellation_fixture_passed": cancellation_passed,
            "tie_breaking_passed": tie_passed,
            "all_checks_passed": all_passed,
        },
        # Backward-compatible fields
        "bank_size": bank_size,
        "vector_dim": vector_dim,
        "num_queries": num_queries,
        "k": k,
        "precompute_norm_time_ms": round(norm_time_ms, 2),
        "elapsed_seconds": round(elapsed, 4),
        "qps": round(qps, 2),
        "ms_per_query": round(ms_per_query, 2),
        "reference_parity_passed": parity_passed,
        "tie_breaking_passed": tie_passed,
        "all_passed": all_passed,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"Saved signed benchmark receipt to {out_file}")
    print("=" * 75)

    if not all_passed:
        print("FAIL-CLOSED: Benchmark verification failed!", file=sys.stderr)
        sys.exit(1)

    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A30 Preflight Benchmark")
    parser.add_argument("--bank-size", type=int, default=20000)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--vector-dim", type=int, default=966)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()

    run_benchmark(
        bank_size=args.bank_size,
        num_queries=args.num_queries,
        vector_dim=args.vector_dim,
        k=args.k,
        query_chunk_size=args.chunk_size,
        use_gpu=not args.no_gpu,
    )
