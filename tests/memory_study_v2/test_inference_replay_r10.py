"""Acceptance & Regression Test for R10: Full return-to-contrast pipeline and disk replay."""

import numpy as np
import pytest

from memory_study_v2.inference import (
    ReplayVerificationError,
    evaluate_primary_contrasts,
    export_analysis_bundle,
    replay_analysis_bundle,
)


def test_inference_full_pipeline_and_replay(tmp_path):
    rng = np.random.default_rng(42)
    returns_map = {}
    arms = [
        "MEM_SIM", "KNN_PLAIN", "MEM_RANDOM", "HIST_PRIOR", "RIDGE_ANNUAL",
        "MLP_BASE", "TRANS_BASE", "MLP_MIX_SR", "TRANS_MIX_SR", "MLP_GATE", "TRANS_GATE"
    ]
    markets = ["US", "IN", "CN", "FR", "GB", "BR"]
    for arm in arms:
        for mkt in markets:
            for seed in ([7, 17, 37] if "MLP" in arm or "TRANS" in arm or "RANDOM" in arm else [None]):
                returns_map[(arm, mkt, seed)] = rng.normal(0.0004, 0.01, 252)

    # Evaluate 8 primary contrasts
    contrast_results, draw_matrix = evaluate_primary_contrasts(returns_map, markets, num_draws=200)
    assert len(contrast_results) == 8

    # Export to disk
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}_{k[1]}_{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results,
        draw_matrix,
        tmp_path / "bundle",
    )

    # Replay in fresh call
    rep = replay_analysis_bundle(bundle_dir, tolerance=1e-10)
    assert rep["status"] == "REPLAY_VERIFIED"


def test_replay_detects_deliberate_draw_corruption(tmp_path):
    rng = np.random.default_rng(42)
    returns_map = {}
    arms = [
        "MEM_SIM", "KNN_PLAIN", "MEM_RANDOM", "HIST_PRIOR", "RIDGE_ANNUAL",
        "MLP_BASE", "TRANS_BASE", "MLP_MIX_SR", "TRANS_MIX_SR", "MLP_GATE", "TRANS_GATE"
    ]
    markets = ["US", "IN", "CN", "FR", "GB", "BR"]
    for arm in arms:
        for mkt in markets:
            returns_map[(arm, mkt, None)] = rng.normal(0.0004, 0.01, 252)

    contrast_results, draw_matrix = evaluate_primary_contrasts(returns_map, markets, num_draws=100)
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}_{k[1]}_{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results,
        draw_matrix,
        tmp_path / "bundle",
    )

    # Corrupt one value in the draw matrix on disk
    draws_path = bundle_dir / "contrast_draws.npy"
    draws = np.load(draws_path)
    draws[0, 0] = draws[0, 0] + 10.0  # Large discrepancy
    np.save(draws_path, draws)

    with pytest.raises(ReplayVerificationError, match="Replay mismatch"):
        replay_analysis_bundle(bundle_dir)


def test_replay_detects_deliberate_returns_corruption(tmp_path):
    import json
    rng = np.random.default_rng(42)
    returns_map = {}
    arms = [
        "MEM_SIM", "KNN_PLAIN", "MEM_RANDOM", "HIST_PRIOR", "RIDGE_ANNUAL",
        "MLP_BASE", "TRANS_BASE", "MLP_MIX_SR", "TRANS_MIX_SR", "MLP_GATE", "TRANS_GATE"
    ]
    markets = ["US", "IN", "CN", "FR", "GB", "BR"]
    for arm in arms:
        for mkt in markets:
            returns_map[(arm, mkt, None)] = rng.normal(0.0004, 0.01, 252)

    contrast_results, draw_matrix = evaluate_primary_contrasts(returns_map, markets, num_draws=100)
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}_{k[1]}_{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results,
        draw_matrix,
        tmp_path / "bundle_ret_tamper",
    )

    # Corrupt a single return in daily_returns.json on disk
    ret_path = bundle_dir / "daily_returns.json"
    with open(ret_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    first_key = list(data.keys())[0]
    data[first_key][0] += 0.05  # Tamper with single return value
    with open(ret_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Replay must fail because recomputing from tampered returns diverges from stored draws/theta
    with pytest.raises(ReplayVerificationError, match="Replay mismatch"):
        replay_analysis_bundle(bundle_dir)

