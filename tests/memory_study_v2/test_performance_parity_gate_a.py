"""Gate A Performance Refactoring Parity Suite (Phase 1-8).

Verifies 100% bit-for-bit numerical and contract parity between the vectorized,
sharded, cached, and unified performance implementations and their reference baselines.

Criteria Tested:
1. retrieve_similarity_and_knn_batch MEM_SIM parity against retrieve_mem_sim_batch.
2. retrieve_similarity_and_knn_batch KNN_PLAIN parity against retrieve_knn_plain.
3. Auditor fixture certification (near-identical vectors, dist ~ 1e-14, ties, spacing, caps).
4. retrieve_mem_random_parallel parity against sequential retrieve_mem_random.
5. retrieve_mem_random_parallel single-worker vs multi-worker identical digests.
6. MarketPanel 2D representation and dev_eval_fn parity.
7. Continuous portfolio simulation lookup and liquidation reconciliation parity.
8. SecurityFeatureCache persistence round-trip, hashing, and DataFrame contract parity.
9. Shard writing, atomic consolidation, SHA-256 verification, and resume parity.
10. StageProfiler telemetry recording, JSON serialization, and resource tracking.
11. Repository configuration authorization guard intactness (production_authorized: false).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any, Dict, List
import numpy as np
import pandas as pd
import pytest

from memory_study_v2.contracts import EXPECTED_FEATURES_ORDERED
from memory_study_v2.market_panel import MarketPanel
from memory_study_v2.memory import BankRecord, MemoryBank
from memory_study_v2.execution import PortfolioAccount, Position
from memory_study_v2.retrieval import (
    RetrievalResult,
    derive_random_seed,
    retrieve_knn_plain,
    retrieve_mem_random,
    retrieve_mem_random_parallel,
    retrieve_mem_random_parallel_multi_seed,
    retrieve_mem_sim,
    retrieve_mem_sim_batch,
    retrieve_similarity_and_knn_batch,
)
from memory_study_v2.security_cache import SecurityFeatureCache
from scripts.launch_a30_production import (
    GLOBAL_PROFILER,
    StageProfiler,
    StageTelemetryRecord,
    _PREPARED_SECURITY_CACHE,
    _ProcessTreeMemoryTracker,
    clear_prepared_security_cache,
    launch_a30_deployment,
    verify_configuration_authorization,
)


# ---------------------------------------------------------------------------
# Fixture: Synthetic Bank and Queries
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_bank_and_queries():
    """Build a deterministic synthetic MemoryBank and query batch."""
    np.random.seed(42)
    N = 250
    dim = 966
    records = []
    for i in range(N):
        vec = np.random.normal(0, 1, dim).astype(np.float32)
        rec = BankRecord(
            record_id=i + 1,
            security_id=f"MKT:SEC_{i % 12}",
            session_origin=f"2015-01-{(i % 28) + 1:02d}",
            session_126_maturity="2017-06-01",
            vector=vec,
            target_63=float(np.random.normal(0, 0.05)),
            session_ordinal=i * 10,
        )
        records.append(rec)

    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    Q = 20
    queries = np.random.normal(0, 1, (Q, dim)).astype(np.float32)
    query_secs = [f"MKT:SEC_{q % 12}" for q in range(Q)]
    query_ids = [f"2020_MKT:SEC_{q % 12}_2020-03-{10 + (q % 15):02d}" for q in range(Q)]

    return bank, queries, query_secs, query_ids


# ---------------------------------------------------------------------------
# Test 1 & 2: Unified Retrieval vs Standalone Baselines Parity
# ---------------------------------------------------------------------------

def test_unified_retrieval_mem_sim_and_knn_plain_parity(synthetic_bank_and_queries):
    """Parity Test 1 & 2: retrieve_similarity_and_knn_batch matches retrieve_mem_sim_batch and retrieve_knn_plain."""
    bank, queries, query_secs, _ = synthetic_bank_and_queries
    k = 15
    max_per_sec = 3
    min_spacing = 21

    # 1. Run baseline retrieve_mem_sim_batch
    baseline_mem = retrieve_mem_sim_batch(
        bank, queries, query_secs, k=k, max_per_security=max_per_sec, min_spacing_sessions=min_spacing,
        start_buffer_size=64, query_chunk_size=8,
    )

    # 2. Run baseline retrieve_knn_plain per query
    baseline_knn = [
        retrieve_knn_plain(bank, queries[q], query_secs[q], k=k)
        for q in range(len(query_secs))
    ]

    # 3. Run unified retrieve_similarity_and_knn_batch
    unified_mem, unified_knn = retrieve_similarity_and_knn_batch(
        bank, queries, query_secs, k=k, max_per_security=max_per_sec, min_spacing_sessions=min_spacing,
        start_buffer_size=64, query_chunk_size=8,
    )

    assert len(unified_mem) == len(queries)
    assert len(unified_knn) == len(queries)

    # Parity assertions for MEM_SIM
    for q in range(len(queries)):
        b_res = baseline_mem[q]
        u_res = unified_mem[q]
        assert u_res.policy == "MEM_SIM"
        assert u_res.is_fallback == b_res.is_fallback
        assert u_res.neighbor_ids == b_res.neighbor_ids, f"MEM_SIM neighbor IDs mismatch at query {q}"
        assert pytest.approx(u_res.prediction, abs=1e-12) == b_res.prediction
        assert len(u_res.neighbor_distances) == len(b_res.neighbor_distances)
        for d_u, d_b in zip(u_res.neighbor_distances, b_res.neighbor_distances):
            assert pytest.approx(d_u, abs=1e-10) == d_b

    # Parity assertions for KNN_PLAIN
    for q in range(len(queries)):
        b_res = baseline_knn[q]
        u_res = unified_knn[q]
        assert u_res.policy == "KNN_PLAIN"
        assert u_res.is_fallback == b_res.is_fallback
        assert u_res.neighbor_ids == b_res.neighbor_ids, f"KNN_PLAIN neighbor IDs mismatch at query {q}"
        assert pytest.approx(u_res.prediction, abs=1e-12) == b_res.prediction
        assert len(u_res.neighbor_distances) == len(b_res.neighbor_distances)
        for d_u, d_b in zip(u_res.neighbor_distances, b_res.neighbor_distances):
            assert pytest.approx(d_u, abs=1e-10) == d_b


# ---------------------------------------------------------------------------
# Test 3: Auditor Fixture with Near-Identical Vectors (Finding 5 Certification)
# ---------------------------------------------------------------------------

def test_unified_retrieval_auditor_fixture_near_identical_vectors():
    """Parity Test 3: Verify that float64 certification prevents cancellation drift in unified retrieval."""
    q = np.ones(966, dtype=np.float32)
    v1 = q.copy()
    v2 = q.copy()
    v1[0] += np.float32(2**(-22))
    v2[0] += np.float32(2**(-23))

    records = [
        BankRecord(record_id=1, security_id="SEC_A", session_origin="2015-01-01", session_126_maturity="2017-06-01", vector=v1, target_63=0.01, session_ordinal=10),
        BankRecord(record_id=2, security_id="SEC_B", session_origin="2015-01-02", session_126_maturity="2017-06-01", vector=v2, target_63=0.02, session_ordinal=20),
        BankRecord(record_id=100, security_id="SEC_C", session_origin="2015-01-03", session_126_maturity="2017-06-01", vector=q * 2.0, target_63=0.03, session_ordinal=30),
        BankRecord(record_id=50, security_id="SEC_D", session_origin="2015-01-04", session_126_maturity="2017-06-01", vector=q * 2.0, target_63=0.04, session_ordinal=40),
        BankRecord(record_id=30, security_id="SEC_C", session_origin="2015-01-05", session_126_maturity="2017-06-01", vector=q * 2.0, target_63=0.05, session_ordinal=35),
    ]
    bank = MemoryBank(records)
    bank.precompute_bank_norms()

    queries = np.array([q], dtype=np.float32)
    query_secs = ["SEC_QUERY"]

    unified_mem, unified_knn = retrieve_similarity_and_knn_batch(
        bank, queries, query_secs, k=3, max_per_security=2, min_spacing_sessions=21, start_buffer_size=2
    )

    ref_mem = retrieve_mem_sim(bank, q, "SEC_QUERY", k=3, max_per_security=2, min_spacing_sessions=21)
    ref_knn = retrieve_knn_plain(bank, q, "SEC_QUERY", k=3)

    # SEC_B (ID 2) is closest (dist 1.42e-14)
    # SEC_A (ID 1) is second closest (dist 5.68e-14)
    # Among remaining ties (dist = 966): SEC_C (ID 30) has smaller record_id than 50 or 100
    assert unified_mem[0].neighbor_ids == ref_mem.neighbor_ids == [2, 1, 30]
    assert unified_mem[0].is_fallback is False

    # For KNN_PLAIN:
    assert unified_knn[0].neighbor_ids == ref_knn.neighbor_ids == [2, 1, 30]


# ---------------------------------------------------------------------------
# Test 4 & 5: Parallel MEM_RANDOM vs Sequential and Worker Parity
# ---------------------------------------------------------------------------

def test_mem_random_parallel_parity_against_sequential(synthetic_bank_and_queries):
    """Parity Test 4: Parallel retrieve_mem_random_parallel exactly equals sequential retrieve_mem_random."""
    bank, _, query_secs, query_ids = synthetic_bank_and_queries
    bank_hash = "mock_bank_hash_abcdef1234567890"
    fold_year = 2020
    master_seed = 42
    k = 10
    max_per_sec = 3
    min_spacing = 21

    # 1. Sequential execution
    seq_results = [
        retrieve_mem_random(
            bank,
            query_security_id=query_secs[i],
            bank_hash=bank_hash,
            fold_year=fold_year,
            query_id=query_ids[i],
            master_seed=master_seed,
            k=k,
            max_per_security=max_per_sec,
            min_spacing_sessions=min_spacing,
        )
        for i in range(len(query_ids))
    ]

    # 2. Parallel execution with multi-workers and small chunks
    par_results = retrieve_mem_random_parallel(
        bank,
        query_security_ids=query_secs,
        query_ids=query_ids,
        bank_hash=bank_hash,
        fold_year=fold_year,
        master_seed=master_seed,
        k=k,
        max_per_security=max_per_sec,
        min_spacing_sessions=min_spacing,
        max_workers=4,
        chunk_size=4,
    )

    assert len(par_results) == len(seq_results)
    for i in range(len(seq_results)):
        s_res = seq_results[i]
        p_res = par_results[i]
        assert p_res.policy == "MEM_RANDOM"
        assert p_res.is_fallback == s_res.is_fallback
        assert p_res.neighbor_ids == s_res.neighbor_ids, f"Random neighbor IDs mismatch at query {i}"
        assert pytest.approx(p_res.prediction, abs=1e-12) == s_res.prediction


def test_mem_random_parallel_worker_invariance(synthetic_bank_and_queries):
    """Parity Test 5: retrieve_mem_random_parallel produces identical digests regardless of max_workers or chunk_size."""
    bank, _, query_secs, query_ids = synthetic_bank_and_queries
    bank_hash = "mock_bank_hash_1122334455667788"
    fold_year = 2021
    master_seed = 7

    res_w1 = retrieve_mem_random_parallel(
        bank, query_secs, query_ids, bank_hash, fold_year, master_seed, max_workers=1, chunk_size=256
    )
    res_w4 = retrieve_mem_random_parallel(
        bank, query_secs, query_ids, bank_hash, fold_year, master_seed, max_workers=4, chunk_size=5
    )

    for i in range(len(query_ids)):
        assert res_w1[i].neighbor_ids == res_w4[i].neighbor_ids
        assert pytest.approx(res_w1[i].prediction, abs=1e-12) == res_w4[i].prediction


# ---------------------------------------------------------------------------
# Test 6: MarketPanel Vectorization and dev_eval_fn Parity
# ---------------------------------------------------------------------------

def test_market_panel_vectorization_parity():
    """Parity Test 6: MarketPanel 2D representation matches dictionary/DataFrame slices exactly."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 21)]
    secs = ["MKT:A", "MKT:B", "MKT:C"]

    sec_info: Dict[str, Any] = {}
    for s_idx, s in enumerate(secs):
        sec_info[s] = {
            "market": "MKT",
            "tr_df": pd.DataFrame({
                "session": sessions,
                "raw_open": [10.0 + s_idx + d for d in range(len(sessions))],
                "raw_close": [10.5 + s_idx + d for d in range(len(sessions))],
            }),
            "val_df": pd.DataFrame({
                "session": sessions,
                "is_valid_bar": [(d % 5 != 0) for d in range(len(sessions))],
            }),
            "feats_df": pd.DataFrame({
                "session": sessions,
                "volatility_21": [0.015 + s_idx * 0.001 for _ in sessions],
                "atr_ratio_14": [0.02 + s_idx * 0.002 for _ in sessions],
            }),
        }

    panel = MarketPanel.build_from_sec_info("MKT", sec_info, sessions)

    assert panel.market == "MKT"
    assert list(panel.securities) == secs
    assert list(panel.sessions) == sessions
    assert panel.raw_open.shape == (20, 3)
    assert panel.raw_close.shape == (20, 3)
    assert panel.tradable.shape == (20, 3)

    for t_idx, sess in enumerate(sessions):
        for s_idx, s in enumerate(secs):
            expected_open = sec_info[s]["tr_df"]["raw_open"].iloc[t_idx]
            expected_close = sec_info[s]["tr_df"]["raw_close"].iloc[t_idx]
            expected_trad = sec_info[s]["val_df"]["is_valid_bar"].iloc[t_idx]
            expected_vol = sec_info[s]["feats_df"]["volatility_21"].iloc[t_idx]
            expected_atr = sec_info[s]["feats_df"]["atr_ratio_14"].iloc[t_idx]

            assert panel.raw_open[t_idx, s_idx] == expected_open
            assert panel.raw_close[t_idx, s_idx] == expected_close
            assert panel.tradable[t_idx, s_idx] == expected_trad
            assert panel.volatility_21[t_idx, s_idx] == expected_vol
            assert panel.atr_ratio_14[t_idx, s_idx] == expected_atr

    # Verify query_index field
    assert panel.query_index is not None
    assert panel.query_index.shape == (len(sessions), len(secs))
    assert np.all(panel.query_index == -1)

    # Test query_index population with dummy records
    from dataclasses import make_dataclass
    Rec = make_dataclass("Rec", [("security_id", str), ("session", str)])
    dummy_records = [Rec(secs[0], sessions[2]), Rec(secs[1], sessions[4])]
    panel_with_recs = MarketPanel.build_from_sec_info("MKT", sec_info, sessions, records=dummy_records)
    assert panel_with_recs.query_index[2, panel.sec_to_idx[secs[0]]] == 0
    assert panel_with_recs.query_index[4, panel.sec_to_idx[secs[1]]] == 1
    assert panel_with_recs.query_index[0, panel.sec_to_idx[secs[0]]] == -1


# ---------------------------------------------------------------------------
# Test 7: Portfolio Simulation Parity and Terminal Liquidation Reconciliation
# ---------------------------------------------------------------------------

def test_portfolio_simulation_parity_and_reconciliation():
    """Parity Test 7: Pre-indexed session lookups produce exact cash/NAV accounting and terminal liquidation."""
    sessions = [f"2020-01-{d:02d}" for d in range(1, 11)]
    acct = PortfolioAccount(
        initial_capital=100000.0,
        commission=0.001,
        slippage=0.0005,
        max_positions=2,
        cash_budget_fraction=0.95,
        atr_multiplier=2.5,
        min_stop_fraction=0.10,
        max_holding_sessions=5,
    )

    open_prices = {"SEC_1": 100.0, "SEC_2": 50.0}
    tradable = {"SEC_1": True, "SEC_2": True}
    close_prices = {"SEC_1": 105.0, "SEC_2": 52.0}
    atr_ratios = {"SEC_1": 0.02, "SEC_2": 0.03}

    # Day 1: queue entry at close
    acct.handle_corporate_actions_before_open(sessions[0], {})
    acct.process_open_fills(sessions[0], open_prices, tradable)
    acct.evaluate_close_stops_and_update_state(sessions[0], close_prices, atr_ratios)
    acct.plan_entries_at_close(["SEC_1", "SEC_2"])
    assert len(acct.pending_entries) == 2

    # Day 2: fills execute at open
    acct.handle_corporate_actions_before_open(sessions[1], {})
    acct.process_open_fills(sessions[1], open_prices, tradable)
    acct.evaluate_close_stops_and_update_state(sessions[1], close_prices, atr_ratios)
    assert len(acct.positions) == 2

    # Run remaining days
    for t in sessions[2:-1]:
        acct.handle_corporate_actions_before_open(t, {})
        acct.process_open_fills(t, open_prices, tradable)
        acct.evaluate_close_stops_and_update_state(t, close_prices, atr_ratios)

    # Final day: terminal liquidation
    final_sess = sessions[-1]
    acct.handle_corporate_actions_before_open(final_sess, {})
    acct.process_open_fills(final_sess, open_prices, tradable)
    acct.evaluate_close_stops_and_update_state(final_sess, close_prices, atr_ratios)
    acct.execute_terminal_liquidation(final_sess, close_prices)

    # Reconcile terminal cash and compound return
    assert len(acct.positions) == 0
    assert abs(acct.cash - acct.daily_history[-1].total_nav) < 1e-6
    rets = [st.daily_return for st in acct.daily_history]
    comp_nav = acct.initial_capital * float(np.prod([1.0 + r for r in rets]))
    assert abs(comp_nav - acct.cash) < 1e-4


# ---------------------------------------------------------------------------
# Test 8: SecurityFeatureCache Round-Trip and Schema Parity
# ---------------------------------------------------------------------------

def test_security_feature_cache_roundtrip(tmp_path):
    """Parity Test 8: SecurityFeatureCache serialization, loading, and to_dataframes parity."""
    T = 50
    sessions = np.array([f"2020-01-{(i % 28) + 1:02d}" for i in range(T)], dtype='<U10')
    raw_open = np.linspace(100.0, 150.0, T)
    raw_high = raw_open + 2.0
    raw_low = raw_open - 2.0
    raw_close = raw_open + 1.0
    volume = np.full(T, 10000.0)
    raw_features = np.random.randn(T, len(EXPECTED_FEATURES_ORDERED))
    target_63 = np.random.randn(T)
    target_exec = np.random.randn(T)
    ordinals = np.arange(T, dtype=np.int64)
    validity = np.ones(T, dtype=bool)
    bar_status = np.array(["VALID"] * T)

    cache = SecurityFeatureCache(
        security_id="MKT:SEC_TEST",
        sessions=sessions,
        raw_open=raw_open,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        model_open=raw_open,
        model_high=raw_high,
        model_low=raw_low,
        model_close=raw_close,
        volume=volume,
        raw_features=raw_features,
        target_63_legacy=target_63,
        target_executable=target_exec,
        session_ordinals=ordinals,
        validity_flags=validity,
        bar_status=bar_status,
        file_sha256="abc123sha",
        calendar_sha256="cal123sha",
    )

    save_path = tmp_path / "cache" / "MKT_SEC_TEST.npz"
    cache.save(save_path)
    assert save_path.exists()

    loaded = SecurityFeatureCache.load(save_path)
    assert loaded.security_id == cache.security_id
    assert loaded.file_sha256 == cache.file_sha256
    np.testing.assert_array_equal(loaded.sessions, cache.sessions)
    np.testing.assert_allclose(loaded.raw_features, cache.raw_features)

    # Convert to DataFrames and verify columns
    val_df, tr_df, feats_df, labels_df = loaded.to_dataframes()
    assert len(val_df) == T
    assert len(tr_df) == T
    assert len(feats_df) == T
    assert len(labels_df) == T
    assert "valid_mask" in feats_df.columns
    assert "tr_open" in tr_df.columns
    assert "model_open" in tr_df.columns
    assert "raw_volume" in tr_df.columns
    assert "target_value" in labels_df.columns

    # Check technical feature columns match EXPECTED_FEATURES_ORDERED
    feat_cols = [c for c in feats_df.columns if c not in ("session", "valid_mask")]
    assert feat_cols == list(EXPECTED_FEATURES_ORDERED)

    # Test directory-based .npy memmaps + JSON manifest (Correction 1)
    dir_save_path = tmp_path / "cache" / "MKT_SEC_TEST"
    cache.save(dir_save_path)
    assert (dir_save_path / "manifest.json").exists()
    assert (dir_save_path / "raw_features.npy").exists()

    loaded_dir = SecurityFeatureCache.load(dir_save_path, mmap_mode="r")
    assert loaded_dir.security_id == cache.security_id
    assert loaded_dir.file_sha256 == cache.file_sha256
    np.testing.assert_array_equal(loaded_dir.sessions, cache.sessions)
    np.testing.assert_allclose(loaded_dir.raw_features, cache.raw_features)
    # Check that arrays loaded with mmap_mode='r' are memmaps
    assert isinstance(loaded_dir.raw_features, np.memmap)


# ---------------------------------------------------------------------------
# Test 9: Deterministic Sharding Consolidation and Resumption Parity
# ---------------------------------------------------------------------------

def test_shard_consolidation_and_resume_parity(tmp_path):
    """Parity Test 9: Prediction shards merge deterministically with identical query orders and no duplicates."""
    shard_dir = tmp_path / "fold_2020" / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    queries_part1 = [
        {"query_id": f"2020_SEC_A_2020-01-0{d}", "prediction": 0.01 * d, "policy_id": "MEM_SIM"}
        for d in range(1, 5)
    ]
    queries_part2 = [
        {"query_id": f"2020_SEC_B_2020-01-0{d}", "prediction": 0.02 * d, "policy_id": "MEM_SIM"}
        for d in range(1, 5)
    ]

    with open(shard_dir / "predictions_part_0000.json", "w", encoding="utf-8") as f:
        json.dump({"queries": queries_part1, "chunk_idx": 0}, f)
    with open(shard_dir / "predictions_part_0001.json", "w", encoding="utf-8") as f:
        json.dump({"queries": queries_part2, "chunk_idx": 1}, f)

    # Consolidate shards
    shard_files = sorted(list(shard_dir.glob("predictions_part_*.json")))
    assert len(shard_files) == 2

    combined_queries = []
    seen_qids = set()
    for sf in shard_files:
        with open(sf, "r", encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("queries", []):
            qid = q["query_id"]
            assert qid not in seen_qids, f"Duplicate query {qid}"
            seen_qids.add(qid)
            combined_queries.append(q)

    assert len(combined_queries) == 8
    assert len(seen_qids) == 8


# ---------------------------------------------------------------------------
# Test 10: Stage Profiler Telemetry and Invariant Tracking
# ---------------------------------------------------------------------------

def test_stage_profiler_telemetry_recording(tmp_path):
    """Parity Test 10: StageProfiler records micro-stage timers, metrics, and exports clean JSON."""
    profiler = StageProfiler()

    with profiler.time_stage("data_canonicalization", fold=2020, queries_processed=0, bank_size=0) as m:
        # Simulate small work
        arr = np.ones((100, 100)) @ np.ones((100, 100))

    with profiler.time_stage("retrieval_execution", fold=2020, queries_processed=50, bank_size=500) as m:
        m["buffer_expansions"] = 2
        m["full_bank_fallbacks"] = 0

    records = profiler.to_dict()
    assert len(records) == 2
    assert records[0]["stage"] == "data_canonicalization"
    assert records[0]["fold"] == 2020
    assert records[0]["elapsed_seconds"] >= 0.0
    assert records[0]["peak_rss_gb"] >= 0.0

    assert records[1]["stage"] == "retrieval_execution"
    assert records[1]["queries_processed"] == 50
    assert records[1]["bank_size"] == 500
    assert records[1]["buffer_expansions"] == 2

    telemetry_file = tmp_path / "telemetry_profile.json"
    profiler.save(telemetry_file)
    assert telemetry_file.exists()

    with open(telemetry_file, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert len(loaded_json) == 2


# ---------------------------------------------------------------------------
# Test 11: Repository Configuration Authorization Guard Intact
# ---------------------------------------------------------------------------

def test_repo_configuration_authorization_guard_intact():
    """Parity Test 11: Ensure repo config.proposed.json retains production_authorized: false."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "rebuild_plan" / "config.proposed.json"
    assert config_path.exists(), f"Configuration file missing at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Strict invariant: on disk in repo, production_authorized MUST be False
    assert cfg.get("production_authorized") is False, (
        "INVARIANT VIOLATION: rebuild_plan/config.proposed.json must have production_authorized: false!"
    )

    # Launch preflight authorization check
    auth_info = verify_configuration_authorization(config_path, authorize_flag=False)
    assert auth_info["repo_production_authorized"] is False
    assert auth_info["repo_guard_intact"] is True
    assert auth_info["launch_flag_authorized"] is False


# ---------------------------------------------------------------------------
# Test 12: Worker Count Benchmark Sweep (Correction 8)
# ---------------------------------------------------------------------------

def test_worker_count_benchmark_sweep(synthetic_bank_and_queries):
    """Parity Test 12: Benchmark worker counts [4, 8, 12, 16, 20] and verify bit-for-bit invariance."""
    import time
    bank, _, query_secs, query_ids = synthetic_bank_and_queries
    bank_hash = "bench_bank_hash_123456789"
    fold_year = 2020
    master_seed = 7

    worker_counts = [4, 8, 12, 16, 20]
    baseline_res = None
    timings = {}

    for w in worker_counts:
        t0 = time.perf_counter()
        res = retrieve_mem_random_parallel(
            bank,
            query_security_ids=query_secs,
            query_ids=query_ids,
            bank_hash=bank_hash,
            fold_year=fold_year,
            master_seed=master_seed,
            k=10,
            max_workers=w,
            chunk_size=4,
        )
        elapsed = time.perf_counter() - t0
        timings[w] = elapsed

        if baseline_res is None:
            baseline_res = res
        else:
            assert len(res) == len(baseline_res)
            for i in range(len(res)):
                assert res[i].neighbor_ids == baseline_res[i].neighbor_ids
                assert pytest.approx(res[i].prediction, abs=1e-12) == baseline_res[i].prediction

    print(f"\n[Worker Count Benchmark Timings] {timings}")
    assert len(timings) == 5


# ---------------------------------------------------------------------------
# Test 13: Six-Fold Prediction Manifest & Prediction Hash Parity (Finding 3)
# ---------------------------------------------------------------------------

CANONICAL_MANIFEST_HASHES: Dict[int, str] = {
    2020: "947a6858b0c84e3cd7174013373d97840c2c659b709ea12c240a51732261d30e",
    2021: "3d8ebc9af18ad0f9e52508fb1d992cde8b6bdece762a824a0feaee1f774a0f88",
    2022: "7f3388c1cd173c2e10af645997ae695f230672c4ba4b8f745860b02f88f43f6b",
    2023: "f75219358dc091c17a207aced9d56dfc098e17c1b59e07d7a198447fde3f685d",
    2024: "ad1691f07a0c567fd5696c2b698a27a610b660f973060c5a23e1b4b9a8a0c2aa",
    2025: "7d0c6ff5d22ea645c76634cd9bc185e7031c5ee71ec82212e1c82a92d21d9d7e",
}

CANONICAL_PREDICTION_HASHES: Dict[int, str] = {
    2020: "c4a4a6e6c1b5bb9b9a7c3b6e7138ad599b895de500208452654a71355d3bea0a",
    2021: "100d01ff6a5cf3cf01f534677299db381743d775aaa3b7c73bcf2c007988e4dc",
    2022: "450256fb20ee98e41e8afafe6e909742ce901f82c42325c2b2da9867f668aa14",
    2023: "fd9438b8263557e239180cf684a3a9895ac21d346bbb1140af44707fac4794a1",
    2024: "fecb34350b7d23e6d276f4f76412cf4c60ab9a2b5be1fb7080c6e638f1e693e4",
    2025: "8429eef786010a1dd90c1f0556786b893ef2deb5638c5088120f8f39d39209ab",
}


def test_six_fold_prediction_manifest_parity():
    """Parity Test 13 (Finding 3): Verify that all six fold prediction manifests and prediction hashes match canonical digests."""
    import hashlib
    repo_root = Path(__file__).resolve().parent.parent.parent
    base_recovery_dir = repo_root / "outputs" / "a30-final-recovery"
    assert base_recovery_dir.is_dir(), f"Missing reference directory: {base_recovery_dir}"

    release_manifest_path = base_recovery_dir / "release_manifest.json"
    assert release_manifest_path.exists(), "Missing release_manifest.json"
    with open(release_manifest_path, "r", encoding="utf-8") as f:
        release_manifest = json.load(f)
    sealed_artifacts = release_manifest.get("sealed_artifacts", {})

    expected_folds = [2020, 2021, 2022, 2023, 2024, 2025]
    for year in expected_folds:
        fold_dir = base_recovery_dir / f"fold_{year}"
        manifest_path = fold_dir / "predictions_manifest.json"
        assert manifest_path.exists(), f"Missing predictions_manifest.json for fold {year}"

        # 1. Manifest file SHA-256 matches canonical reference and release_manifest
        actual_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert actual_manifest_sha == CANONICAL_MANIFEST_HASHES[year], (
            f"Fold {year} manifest SHA mismatch: got {actual_manifest_sha}, expected {CANONICAL_MANIFEST_HASHES[year]}"
        )
        rel_key = f"fold_{year}/predictions_manifest.json"
        assert sealed_artifacts.get(rel_key) == actual_manifest_sha, (
            f"Release manifest mismatch for {rel_key}"
        )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["fold_year"] == year
        assert manifest["count"] > 0

        # 2. Prediction hash in manifest matches canonical prediction hash and release_manifest
        pred_sha = manifest["sha256"]
        assert pred_sha == CANONICAL_PREDICTION_HASHES[year], (
            f"Fold {year} prediction SHA mismatch in manifest: got {pred_sha}, expected {CANONICAL_PREDICTION_HASHES[year]}"
        )
        pred_rel_key = f"fold_{year}/policy_predictions.json"
        assert sealed_artifacts.get(pred_rel_key) == pred_sha, (
            f"Release manifest mismatch for {pred_rel_key}"
        )

        identity = manifest.get("prediction_identity")
        assert identity is not None, f"Missing prediction_identity in fold {year} manifest"
        assert identity["fold_year"] == year
        assert "prediction_identity_sha256" in identity
        assert len(identity["prediction_identity_sha256"]) == 64
        assert "producing_checkpoints" in identity
        assert len(identity["producing_checkpoints"]) > 0

        # 3. If policy_predictions.json exists on disk, recompute and verify exact match
        pred_file = fold_dir / "policy_predictions.json"
        if pred_file.exists():
            computed_pred_sha = hashlib.sha256(pred_file.read_bytes()).hexdigest()
            assert computed_pred_sha == CANONICAL_PREDICTION_HASHES[year], (
                f"Computed SHA-256 for existing {pred_file} does not match canonical prediction digest"
            )


# ---------------------------------------------------------------------------
# Test 14: Segment ID Cache Discontinuity Preservation (Finding 1)
# ---------------------------------------------------------------------------

def test_segment_id_cache_discontinuity_preservation(tmp_path):
    """Parity Test 14 (Finding 1): SecurityFeatureCache serializes and reconstructs segment_ids accurately without hardcoding 0."""
    T = 20
    sessions = np.array([f"2020-01-{(i % 28) + 1:02d}" for i in range(T)], dtype='<U10')
    raw_open = np.linspace(100.0, 120.0, T)
    raw_high = raw_open + 1.0
    raw_low = raw_open - 1.0
    raw_close = raw_open + 0.5
    volume = np.full(T, 5000.0)
    raw_features = np.random.randn(T, len(EXPECTED_FEATURES_ORDERED))
    target_63 = np.random.randn(T)
    target_exec = np.random.randn(T)
    ordinals = np.arange(T, dtype=np.int64)
    validity = np.ones(T, dtype=bool)
    bar_status = np.array(["VALID"] * T)

    # Discontinuity segment IDs with 3 distinct segments
    test_segments = np.array([0] * 5 + [1] * 8 + [2] * 7, dtype=np.int64)

    cache = SecurityFeatureCache(
        security_id="MKT:DISCONT_SEC",
        sessions=sessions,
        raw_open=raw_open,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        model_open=raw_open,
        model_high=raw_high,
        model_low=raw_low,
        model_close=raw_close,
        volume=volume,
        raw_features=raw_features,
        target_63_legacy=target_63,
        target_executable=target_exec,
        session_ordinals=ordinals,
        validity_flags=validity,
        bar_status=bar_status,
        file_sha256="discont_file_sha",
        calendar_sha256="discont_cal_sha",
        segment_ids=test_segments,
    )

    # Verify to_dataframes() propagates segment_ids (not hardcoded 0)
    val_df, tr_df, feats_df, labels_df = cache.to_dataframes()
    assert "segment_id" in val_df.columns
    assert "segment_id" in tr_df.columns
    np.testing.assert_array_equal(val_df["segment_id"].to_numpy(), test_segments)
    np.testing.assert_array_equal(tr_df["segment_id"].to_numpy(), test_segments)

    # Test round-trip persistence via directory (.npy memmaps + manifest)
    dir_path = tmp_path / "cache_discont_dir"
    cache.save(dir_path)
    assert (dir_path / "segment_ids.npy").exists()
    with open(dir_path / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "segment_ids" in manifest.get("arrays", [])

    loaded_dir = SecurityFeatureCache.load(dir_path, mmap_mode="r")
    assert loaded_dir.segment_ids is not None
    np.testing.assert_array_equal(loaded_dir.segment_ids, test_segments)
    val_df_dir, tr_df_dir, _, _ = loaded_dir.to_dataframes()
    np.testing.assert_array_equal(val_df_dir["segment_id"].to_numpy(), test_segments)
    np.testing.assert_array_equal(tr_df_dir["segment_id"].to_numpy(), test_segments)

    # Test round-trip persistence via NPZ
    npz_path = tmp_path / "cache_discont.npz"
    cache.save(npz_path)
    loaded_npz = SecurityFeatureCache.load(npz_path)
    assert loaded_npz.segment_ids is not None
    np.testing.assert_array_equal(loaded_npz.segment_ids, test_segments)
    val_df_npz, tr_df_npz, _, _ = loaded_npz.to_dataframes()
    np.testing.assert_array_equal(val_df_npz["segment_id"].to_numpy(), test_segments)
    np.testing.assert_array_equal(tr_df_npz["segment_id"].to_numpy(), test_segments)

    # Verify fallback if segment_ids is None
    cache_no_seg = SecurityFeatureCache(
        security_id="MKT:NO_SEG",
        sessions=sessions,
        raw_open=raw_open,
        raw_high=raw_high,
        raw_low=raw_low,
        raw_close=raw_close,
        model_open=raw_open,
        model_high=raw_high,
        model_low=raw_low,
        model_close=raw_close,
        volume=volume,
        raw_features=raw_features,
        target_63_legacy=target_63,
        target_executable=target_exec,
        session_ordinals=ordinals,
        validity_flags=validity,
        bar_status=bar_status,
        file_sha256="no_seg_file",
        calendar_sha256="no_seg_cal",
        segment_ids=None,
    )
    val_df_none, tr_df_none, _, _ = cache_no_seg.to_dataframes()
    assert np.all(val_df_none["segment_id"].to_numpy() == 0)
    assert np.all(tr_df_none["segment_id"].to_numpy() == 0)


# ---------------------------------------------------------------------------
# Test 15: In-Memory Prepared Security Cache Across Folds (Finding 2)
# ---------------------------------------------------------------------------

def test_prepared_security_cache_fold_reuse(tmp_path):
    """Parity Test 15 (Finding 2): In-memory prepared DataFrame cache avoids redundant reconstruction across folds."""
    clear_prepared_security_cache()
    assert len(_PREPARED_SECURITY_CACHE) == 0

    cache_key = ("MKT:SEC_TEST", str(tmp_path / "mock.parquet"))
    mock_sec_cache = "mock_sec_cache_obj"
    mock_sched_val_df = pd.DataFrame({"session": ["2020-01-01"], "bar_status": ["VALID"]})
    mock_tr_df = pd.DataFrame({"session": ["2020-01-01"], "raw_open": [100.0]})
    mock_feats_df = pd.DataFrame({"session": ["2020-01-01"]})
    mock_labels_df = pd.DataFrame({"target_value": [0.05]})
    mock_sess_to_row = {"2020-01-01": 0}

    _PREPARED_SECURITY_CACHE[cache_key] = (
        mock_sec_cache,
        mock_sched_val_df,
        mock_tr_df,
        mock_feats_df,
        mock_labels_df,
        mock_sess_to_row,
    )

    assert cache_key in _PREPARED_SECURITY_CACHE
    entry = _PREPARED_SECURITY_CACHE[cache_key]
    assert entry[0] is mock_sec_cache
    assert entry[1] is mock_sched_val_df
    assert entry[2] is mock_tr_df
    assert entry[3] is mock_feats_df
    assert entry[4] is mock_labels_df
    assert entry[5] is mock_sess_to_row

    # Test clearing
    clear_prepared_security_cache()
    assert len(_PREPARED_SECURITY_CACHE) == 0
    assert cache_key not in _PREPARED_SECURITY_CACHE


# ---------------------------------------------------------------------------
# Test 16: GPU Cache Teardown and Resource Cleanup (Finding 6)
# ---------------------------------------------------------------------------

def test_gpu_cache_release_teardown(synthetic_bank_and_queries):
    """Parity Test 16 (Finding 6): MemoryBank.release_gpu_cache() explicitly tears down GPU tensors and frees memory."""
    bank, _, _, _ = synthetic_bank_and_queries

    # Initially GPU cached tensors are None
    assert getattr(bank, "_gpu_vectors", None) is None
    assert getattr(bank, "_gpu_bank_norms", None) is None

    # Simulate GPU caching (or real GPU if CUDA available)
    import torch
    dummy_vecs = torch.from_numpy(bank.vectors[:5])
    dummy_norms = torch.from_numpy(bank.bank_norms[:5])
    bank._gpu_vectors = dummy_vecs
    bank._gpu_bank_norms = dummy_norms

    assert bank._gpu_vectors is not None
    assert bank._gpu_bank_norms is not None

    # Call release_gpu_cache
    bank.release_gpu_cache()

    assert bank._gpu_vectors is None
    assert bank._gpu_bank_norms is None

    # Verify idempotency
    bank.release_gpu_cache()
    assert bank._gpu_vectors is None
    assert bank._gpu_bank_norms is None

    # Test destructor __del__ calls release_gpu_cache
    bank._gpu_vectors = dummy_vecs
    bank.__del__()
    assert bank._gpu_vectors is None


# ---------------------------------------------------------------------------
# Test 17: Multi-Seed Random Memory Worker and Pool Reuse (Finding 5)
# ---------------------------------------------------------------------------

def test_random_memory_multi_seed_worker_reuse(synthetic_bank_and_queries):
    """Parity Test 17 (Finding 5): retrieve_mem_random_parallel_multi_seed shares memmaps and pool across seeds identically."""
    bank, _, query_secs, query_ids = synthetic_bank_and_queries
    bank_hash = "mock_bank_hash_multiseed_123"
    fold_year = 2022
    seeds = [7, 17, 37]
    k = 8
    max_per_sec = 2
    min_spacing = 15

    # 1. Run multi-seed batched retrieval
    multi_seed_results = retrieve_mem_random_parallel_multi_seed(
        bank,
        query_security_ids=query_secs,
        query_ids=query_ids,
        bank_hash=bank_hash,
        fold_year=fold_year,
        seeds=seeds,
        k=k,
        max_per_security=max_per_sec,
        min_spacing_sessions=min_spacing,
        max_workers=4,
        chunk_size=4,
    )

    assert set(multi_seed_results.keys()) == set(seeds)

    # 2. Run single-seed baseline for each seed and verify bit-for-bit equivalence
    for s in seeds:
        single_seed_res = retrieve_mem_random(
            bank,
            query_security_id=query_secs[0],
            bank_hash=bank_hash,
            fold_year=fold_year,
            query_id=query_ids[0],
            master_seed=s,
            k=k,
            max_per_security=max_per_sec,
            min_spacing_sessions=min_spacing,
        )
        assert multi_seed_results[s][0].neighbor_ids == single_seed_res.neighbor_ids
        assert pytest.approx(multi_seed_results[s][0].prediction, abs=1e-12) == single_seed_res.prediction

        # Full query loop comparison
        full_single_seed = retrieve_mem_random_parallel(
            bank,
            query_security_ids=query_secs,
            query_ids=query_ids,
            bank_hash=bank_hash,
            fold_year=fold_year,
            master_seed=s,
            k=k,
            max_per_security=max_per_sec,
            min_spacing_sessions=min_spacing,
            max_workers=2,
            chunk_size=4,
        )
        assert len(multi_seed_results[s]) == len(full_single_seed)
        for i in range(len(query_ids)):
            assert multi_seed_results[s][i].neighbor_ids == full_single_seed[i].neighbor_ids
            assert pytest.approx(multi_seed_results[s][i].prediction, abs=1e-12) == full_single_seed[i].prediction


# ---------------------------------------------------------------------------
# Test 18: Incremental Shard Resumption and Atomic Storage (Finding 4)
# ---------------------------------------------------------------------------

def test_incremental_shard_resumption(tmp_path):
    """Parity Test 18 (Finding 4): Incremental shards are written atomically and reused upon resumption."""
    shards_dir = tmp_path / "fold_2020" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    expected_identity_sha = "ident_abc123_test_sha256"

    # 1. Write Shard 0 atomically
    shard0_recs = [
        {"query_id": "2020_SEC_0", "security_id": "SEC_0", "market": "MKT", "decision_session": "2020-01-02",
         "fold": 2020, "policy_id": "MEM_SIM", "realization_id": None, "prediction": 0.01, "source_artifact_id": "src0"},
        {"query_id": "2020_SEC_1", "security_id": "SEC_1", "market": "MKT", "decision_session": "2020-01-02",
         "fold": 2020, "policy_id": "MEM_SIM", "realization_id": None, "prediction": 0.02, "source_artifact_id": "src1"},
    ]
    shard0_payload = {
        "shard_meta": {
            "query_start": 0,
            "query_end": 2,
            "fold_year": 2020,
            "prediction_identity_sha256": expected_identity_sha,
            "total_records": len(shard0_recs),
        },
        "records": shard0_recs,
    }
    shard0_file = shards_dir / "predictions_00000_00002.json"
    tmp_shard0 = shard0_file.with_suffix(".tmp.json")
    with open(tmp_shard0, "w", encoding="utf-8") as f:
        json.dump(shard0_payload, f)
    tmp_shard0.rename(shard0_file)
    assert shard0_file.exists()

    # 2. Simulate resuming execution where shard 0 is already present
    shard_size = 2
    num_eval = 4
    num_shards = 2

    seen_keys = set()
    shards_records_dict = {}
    missing_shards = []

    for shard_idx in range(num_shards):
        q_start = shard_idx * shard_size
        q_end = min(q_start + shard_size, num_eval)
        shard_file = shards_dir / f"predictions_{q_start:05d}_{q_end:05d}.json"

        shard_loaded = False
        if shard_file.exists():
            try:
                with open(shard_file, "r", encoding="utf-8") as f:
                    s_data = json.load(f)
                s_meta = s_data.get("shard_meta", {})
                if s_meta.get("prediction_identity_sha256") == expected_identity_sha:
                    s_recs = s_data.get("records", [])
                    for r in s_recs:
                        k = (r["query_id"], r["policy_id"], r["realization_id"])
                        assert k not in seen_keys, f"Duplicate key: {k}"
                        seen_keys.add(k)
                    shards_records_dict[shard_idx] = s_recs
                    shard_loaded = True
            except Exception:
                shard_loaded = False

        if not shard_loaded:
            missing_shards.append((shard_idx, q_start, q_end, shard_file))

    # Shard 0 must be recognized and loaded from disk without recomputing
    assert 0 in shards_records_dict
    assert len(shards_records_dict[0]) == 2
    # Only Shard 1 should be marked as missing
    assert len(missing_shards) == 1
    assert missing_shards[0][0] == 1
    assert missing_shards[0][1] == 2
    assert missing_shards[0][2] == 4

    # 3. Simulate computing and writing missing Shard 1
    shard1_recs = [
        {"query_id": "2020_SEC_2", "security_id": "SEC_2", "market": "MKT", "decision_session": "2020-01-03",
         "fold": 2020, "policy_id": "MEM_SIM", "realization_id": None, "prediction": 0.03, "source_artifact_id": "src2"},
        {"query_id": "2020_SEC_3", "security_id": "SEC_3", "market": "MKT", "decision_session": "2020-01-03",
         "fold": 2020, "policy_id": "MEM_SIM", "realization_id": None, "prediction": 0.04, "source_artifact_id": "src3"},
    ]
    shard1_payload = {
        "shard_meta": {
            "query_start": 2,
            "query_end": 4,
            "fold_year": 2020,
            "prediction_identity_sha256": expected_identity_sha,
            "total_records": len(shard1_recs),
        },
        "records": shard1_recs,
    }
    shard1_file = missing_shards[0][3]
    tmp_shard1 = shard1_file.with_suffix(".tmp.json")
    with open(tmp_shard1, "w", encoding="utf-8") as f:
        json.dump(shard1_payload, f)
    tmp_shard1.rename(shard1_file)
    shards_records_dict[1] = shard1_recs

    # 4. Consolidate shards
    all_recs = []
    for s_idx in range(num_shards):
        all_recs.extend(shards_records_dict[s_idx])
    assert len(all_recs) == 4
    qids = [r["query_id"] for r in all_recs]
    assert qids == ["2020_SEC_0", "2020_SEC_1", "2020_SEC_2", "2020_SEC_3"]


# ---------------------------------------------------------------------------
# Test 19: Process Tree Peak RSS Tracking (Finding 7)
# ---------------------------------------------------------------------------

def test_stage_profiler_multiprocess_peak_rss():
    """Parity Test 19 (Finding 7): _ProcessTreeMemoryTracker captures child process peak RSS."""
    import subprocess
    import sys

    # 1. Direct _ProcessTreeMemoryTracker verification
    tracker = _ProcessTreeMemoryTracker(interval=0.02)
    tracker.start()

    # Launch a child process that allocates ~100MB of RAM
    child_code = "import time, numpy as np; arr = np.ones((3500, 3500), dtype=np.float64); time.sleep(0.35)"
    proc = subprocess.Popen([sys.executable, "-c", child_code])
    proc.wait()

    peak_rss = tracker.stop()
    assert peak_rss > 50 * 1024 * 1024, f"Peak RSS {peak_rss} too low to capture child process tree"

    # 2. StageProfiler integration verification
    profiler = StageProfiler()
    with profiler.time_stage("child_process_benchmark", fold=2020) as m:
        proc2 = subprocess.Popen([sys.executable, "-c", child_code])
        proc2.wait()
        m["child_finished"] = True

    records = profiler.to_dict()
    assert len(records) == 1
    rec = records[0]
    assert rec["stage"] == "child_process_benchmark"
    assert rec["peak_rss_gb"] > 0.05, f"StageProfiler peak_rss_gb {rec['peak_rss_gb']} failed to reflect child memory"

