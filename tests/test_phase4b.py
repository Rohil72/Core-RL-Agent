import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.losses.outcome_geometry import compute_market_memory_loss, OutcomeGeometryLossConfig, OutcomeTargetNormalizer
from src.eval.causal_memory import attach_outcome_availability
from src.memory.retrieval import RetrievalConfig, retrieve_neighbors, deduplicate_close_neighbors

class TestPhase4B(unittest.TestCase):

    def test_per_anchor_analogue_kl_matches_hand_computed(self):
        # We construct a scenario with 3 distinct tickers so cross-ticker is satisfied
        latent = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0]
        ])
        
        # Norm targets for upside, downside, path quality
        future_target = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        target_names = ["future_max_return_63", "future_min_return_63", "event_upside_before_drawdown_126"]
        
        class MockNormalizer:
            def normalize(self, t, names):
                return t
                
        config = OutcomeGeometryLossConfig(
            lambda_reg=0.0, 
            lambda_analogue=1.0,
            future_loss_weight=1.0,
            max_return_loss_weight=1.0,
            min_return_loss_weight=1.0
        )
        
        # Expected computation in double precision:
        # Anchor 0:
        py0 = np.array([1/(1+np.exp(-4)), np.exp(-4)/(1+np.exp(-4))])
        pz0 = np.array([1/(1+np.exp(-10)), np.exp(-10)/(1+np.exp(-10))])
        kl0 = np.sum(py0 * (np.log(py0) - np.log(pz0)))
        
        # Anchor 1:
        py1 = np.array([1/(1+np.exp(-4)), np.exp(-4)/(1+np.exp(-4))])
        pz1 = np.array([0.5, 0.5])
        kl1 = np.sum(py1 * (np.log(py1) - np.log(pz1)))
        
        # Anchor 2:
        py2 = np.array([0.5, 0.5])
        pz2 = np.array([1/(1+np.exp(-10)), np.exp(-10)/(1+np.exp(-10))])
        kl2 = np.sum(py2 * (np.log(py2) - np.log(pz2)))
        
        expected_mean_kl = float((kl0 + kl1 + kl2) / 3.0)
        
        res = compute_market_memory_loss(
            latent=latent,
            future_prediction=future_target, # just dummy for reg loss
            future_target=future_target,
            ticker=["AAPL", "MSFT", "GOOG"],
            target_names=target_names,
            target_normalizer=MockNormalizer(),
            config=config,
        )
        
        self.assertAlmostEqual(float(res.analogue), expected_mean_kl, places=5)
        self.assertEqual(res.eligible_anchor_count, 3)
        self.assertEqual(res.valid_pair_count, 6)

    def test_same_ticker_and_missing_target_pairs_never_contribute(self):
        latent = torch.randn(3, 8)
        future_target = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [float('nan'), 1.0, 0.0]
        ])
        target_names = ["future_max_return_63", "future_min_return_63", "event_upside_before_drawdown_126"]
        
        class MockNormalizer:
            def normalize(self, t, names):
                return t
                
        config = OutcomeGeometryLossConfig(lambda_reg=0.0, lambda_analogue=1.0)
        
        # Tickers: 0 and 1 are same ticker. 2 has NaN target.
        # So for anchor 0: cand 1 is same ticker (invalid), cand 2 has NaN (invalid)
        res = compute_market_memory_loss(
            latent=latent,
            future_prediction=torch.zeros_like(future_target),
            future_target=future_target,
            ticker=["AAPL", "AAPL", "GOOG"],
            target_names=target_names,
            target_normalizer=MockNormalizer(),
            config=config,
        )
        
        # Should be exactly zero loss because 0 eligible anchors
        self.assertEqual(float(res.analogue), 0.0)
        self.assertEqual(res.eligible_anchor_count, 0)
        self.assertEqual(res.valid_pair_count, 0)

    def test_loss_remains_finite_under_cuda_amp_float32(self):
        # We test that autocast disabling works correctly.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        latent = torch.randn(5, 16, device=device, dtype=dtype)
        future_target = torch.randn(5, 3, device=device, dtype=dtype)
        target_names = ["future_max_return_63", "future_min_return_63", "event_upside_before_drawdown_126"]
        
        class MockNormalizer:
            def normalize(self, t, names):
                return t
                
        config = OutcomeGeometryLossConfig(lambda_reg=0.0, lambda_analogue=1.0)
        
        with torch.amp.autocast(device, enabled=True):
            res = compute_market_memory_loss(
                latent=latent,
                future_prediction=future_target,
                future_target=future_target,
                ticker=["AAPL", "MSFT", "GOOG", "AMZN", "META"],
                target_names=target_names,
                target_normalizer=MockNormalizer(),
                config=config,
            )
            
        self.assertTrue(torch.isfinite(res.analogue))
        self.assertIsInstance(float(res.analogue), float)
        
    def test_causal_exact_maturity_excludes_rows(self):
        # We mock precomputed parquet
        timestamps = pd.date_range("2020-01-01", periods=200, freq="D")
        precomputed = pd.DataFrame(index=timestamps)
        precomputed_file = "AAPL.parquet"
        precomputed.to_parquet(precomputed_file)
        
        # Query frame
        # row 1: ts = 2020-01-01 (has 126 sessions ahead: idx 0 -> 126 is valid)
        # row 2: ts = 2020-05-01 (idx 121, 121+126=247 > 200, invalid, no outcome!)
        q = pd.DataFrame({
            "ticker": ["AAPL", "AAPL"],
            "timestamp": [timestamps[0], timestamps[121]]
        })
        
        out = attach_outcome_availability(q, precomputed_file, horizon_sessions=126)
        
        self.assertFalse(pd.isna(out["outcome_available_timestamp"].iloc[0]))
        self.assertTrue(pd.isna(out["outcome_available_timestamp"].iloc[1]))
        
        os.remove(precomputed_file)

    def test_neighbor_deduplication_keeps_nearer_neighbor(self):
        # Suppose ordered indices by distance: 0 (dist=1.0, ticker=A, session=10), 1 (dist=1.5, ticker=A, session=20)
        # 10 and 20 are separated by 10 sessions. If min_separation = 21, the second should be discarded.
        ordered = np.array([0, 1])
        tickers = np.array(["AAPL", "AAPL"])
        sessions = np.array([10, 20])
        
        kept = deduplicate_close_neighbors(ordered, tickers, sessions, min_separation=21)
        self.assertEqual(kept.tolist(), [0])
        
        # If min_sep = 5, both kept
        kept_5 = deduplicate_close_neighbors(ordered, tickers, sessions, min_separation=5)
        self.assertEqual(kept_5.tolist(), [0, 1])
        
    def test_retrieval_mode_filters(self):
        memory = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "sector": ["Tech", "Tech", "Comm"],
            "industry": ["Hardware", "Software", "Internet"],
            "outcome_available_timestamp": [pd.Timestamp("2020-01-01")] * 3,
            "session_index": [0, 0, 0]
        })
        mem_x = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        
        query_x = np.array([1.0, 0.0])
        ts = pd.Timestamp("2021-01-01")
        
        # Test cross_ticker_only
        cfg = RetrievalConfig(same_ticker_mode="exclude", exclude_query_sector=False, k=10)
        indices, _ = retrieve_neighbors(memory, mem_x, query_x, ts, "AAPL", cfg, "Tech", "Hardware")
        # Should exclude AAPL
        self.assertNotIn(0, indices)
        self.assertIn(1, indices)
        self.assertIn(2, indices)
        
        # Test sector_excluded
        cfg_sec = RetrievalConfig(same_ticker_mode="exclude", exclude_query_sector=True, k=10)
        indices_sec, _ = retrieve_neighbors(memory, mem_x, query_x, ts, "AAPL", cfg_sec, "Tech", "Hardware")
        # Should exclude AAPL (same ticker) and MSFT (same sector)
        self.assertNotIn(0, indices_sec)
        self.assertNotIn(1, indices_sec)
        self.assertIn(2, indices_sec)

    def test_loss_gradient_remains_finite_under_collapsed_norms(self):
        latent = torch.tensor([
            [1.0, 0.0],
            [0.0, 0.0],
            [1e-8, 1e-8]
        ], requires_grad=True)
        
        future_target = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        target_names = ["future_max_return_63", "future_min_return_63", "event_upside_before_drawdown_126"]
        
        class MockNormalizer:
            def normalize(self, t, names):
                return t
                
        config = OutcomeGeometryLossConfig(
            lambda_reg=0.0, 
            lambda_analogue=1.0,
            future_loss_weight=1.0,
            max_return_loss_weight=1.0,
            min_return_loss_weight=1.0
        )
        
        res = compute_market_memory_loss(
            latent=latent,
            future_prediction=future_target,
            future_target=future_target,
            ticker=["AAPL", "MSFT", "GOOG"],
            target_names=target_names,
            target_normalizer=MockNormalizer(),
            config=config,
        )
        
        self.assertTrue(torch.isfinite(res.analogue))
        res.total.backward()
        self.assertTrue(torch.isfinite(latent.grad).all())

if __name__ == '__main__':
    unittest.main()
