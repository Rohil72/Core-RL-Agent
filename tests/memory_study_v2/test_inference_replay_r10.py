"""Acceptance & Regression Test for R10/C2: Full return-to-contrast pipeline, numerical oracle, and disk replay."""

import json
import numpy as np
import pytest

from memory_study_v2.inference import (
    PRIMARY_CONTRASTS,
    ReplayVerificationError,
    compute_sharpe_ratio,
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
    contrast_results, draw_matrix, draw_weeks = evaluate_primary_contrasts(
        returns_map, markets, num_draws=200, block_length_weeks=4
    )
    assert len(contrast_results) == 8
    assert draw_matrix.shape == (8, 200)

    # Verify contrast identities
    for r in contrast_results:
        assert np.isfinite(r.theta)
        assert np.isfinite(r.p_value)
        assert np.isfinite(r.p_value_holm)

    # Export to disk
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results,
        draw_matrix,
        draw_weeks,
        tmp_path / "bundle",
        markets=markets,
        num_draws=200,
        block_length_weeks=4,
    )

    # Replay in fresh call
    rep = replay_analysis_bundle(bundle_dir, tolerance=1e-10)
    assert rep["status"] == "REPLAY_VERIFIED"
    assert rep["contrasts_verified"] == 8


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

    contrast_results, draw_matrix, draw_weeks = evaluate_primary_contrasts(
        returns_map, markets, num_draws=100
    )
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results,
        draw_matrix,
        draw_weeks,
        tmp_path / "bundle",
        markets=markets,
    )

    # Corrupt one value in the draw matrix on disk
    draws_path = bundle_dir / "contrast_draws.npy"
    draws = np.load(draws_path)
    draws[0, 0] = draws[0, 0] + 10.0  # Large discrepancy
    np.save(draws_path, draws)

    with pytest.raises(ReplayVerificationError, match="Replay mismatch in bootstrap draw matrix"):
        replay_analysis_bundle(bundle_dir)


def test_replay_detects_deliberate_returns_corruption(tmp_path):
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

    contrast_results, draw_matrix, draw_weeks = evaluate_primary_contrasts(
        returns_map, markets, num_draws=100
    )
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results,
        draw_matrix,
        draw_weeks,
        tmp_path / "bundle_ret_tamper",
        markets=markets,
    )

    # Corrupt a single return in daily_returns.json on disk
    ret_path = bundle_dir / "daily_returns.json"
    with open(ret_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    first_key = list(payload["returns"].keys())[0]
    payload["returns"][first_key][0] += 0.05  # Tamper with single return value
    with open(ret_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    with pytest.raises(ReplayVerificationError, match="Replay mismatch"):
        replay_analysis_bundle(bundle_dir)


def test_numerical_oracle_with_fixed_blocks(tmp_path):
    """Verify small independent numerical oracle against exact mathematical expectations (C2)."""
    # 2 markets, 11 arms, 10 sessions (2 calendar weeks of 5 days each)
    markets = ["US", "IN"]
    arms = [
        "MEM_SIM", "KNN_PLAIN", "MEM_RANDOM", "HIST_PRIOR", "RIDGE_ANNUAL",
        "MLP_BASE", "TRANS_BASE", "MLP_MIX_SR", "TRANS_MIX_SR", "MLP_GATE", "TRANS_GATE"
    ]
    returns_map = {}
    for idx, arm in enumerate(arms):
        for mkt_idx, mkt in enumerate(markets):
            # Distinct non-scale-invariant return profiles across arms and weeks with within-week variance
            pattern = np.array([
                0.005 * (idx + 1) * (1 + 0.3 * (t % 5)) + 0.001 * (t // 5)
                for t in range(10)
            ], dtype=np.float64) + (0.0005 * mkt_idx)
            returns_map[(arm, mkt, None)] = pattern

    # Supply fixed block choices: e.g. draw 0 chooses [0, 0], draw 1 chooses [1, 1], draw 2 chooses [0, 1]
    supplied_blocks = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
    ], dtype=np.int64)

    contrast_results, draw_matrix, _ = evaluate_primary_contrasts(
        returns_map, markets, draw_week_indices=supplied_blocks, num_draws=3, allow_reduced_arms=False
    )
    assert draw_matrix.shape == (8, 3)

    # Calculate oracle for P1 (MEM_SIM vs MEM_RANDOM) on draw 2 (which uses original weeks [0, 1])
    sr_sim_us = compute_sharpe_ratio(returns_map[("MEM_SIM", "US", None)])
    sr_sim_in = compute_sharpe_ratio(returns_map[("MEM_SIM", "IN", None)])
    sr_sim_mean = (sr_sim_us + sr_sim_in) / 2.0

    sr_rnd_us = compute_sharpe_ratio(returns_map[("MEM_RANDOM", "US", None)])
    sr_rnd_in = compute_sharpe_ratio(returns_map[("MEM_RANDOM", "IN", None)])
    sr_rnd_mean = (sr_rnd_us + sr_rnd_in) / 2.0

    expected_p1_theta = sr_sim_mean - sr_rnd_mean
    assert pytest.approx(contrast_results[0].theta, rel=1e-12) == expected_p1_theta
    # On draw 2 (weeks [0, 1]), draw value must equal original theta
    assert pytest.approx(draw_matrix[0, 2], rel=1e-12) == expected_p1_theta

    # Now verify that changing supplied blocks changes the draws
    altered_blocks = np.array([
        [1, 1],
        [0, 0],
        [0, 0],
    ], dtype=np.int64)
    _, altered_draws, _ = evaluate_primary_contrasts(
        returns_map, markets, draw_week_indices=altered_blocks, num_draws=3
    )
    assert not np.allclose(draw_matrix, altered_draws)


def test_missing_required_policy_arm_raises_error():
    """Verify evaluator strictly rejects missing required policy arms instead of zero-imputing (C2)."""
    markets = ["US", "IN"]
    returns_map = {
        ("MEM_SIM", "US", None): np.array([0.01] * 10),
        ("MEM_SIM", "IN", None): np.array([0.01] * 10),
        # Missing all other arms (MLP_BASE, TRANS_BASE, etc.)
    }
    with pytest.raises(ValueError, match="Required policy arms missing from returns data"):
        evaluate_primary_contrasts(returns_map, markets, allow_reduced_arms=False)
