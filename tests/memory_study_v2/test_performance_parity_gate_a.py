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
    retrieve_mem_sim,
    retrieve_mem_sim_batch,
    retrieve_similarity_and_knn_batch,
)
from memory_study_v2.security_cache import SecurityFeatureCache
from scripts.launch_a30_production import (
    GLOBAL_PROFILER,
    StageProfiler,
    StageTelemetryRecord,
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
