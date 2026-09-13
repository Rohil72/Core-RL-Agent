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


def test_calendar_week_identity_no_collision():
    """Verify 2 Jan 2024 and 30 Dec 2024 do NOT collide into the same week key (C2)."""
    from memory_study_v2.inference import build_calendar_weeks_mapping
    dates = ["2024-01-02", "2024-12-30"]
    g_weeks, years, unique_yw = build_calendar_weeks_mapping(dates, 2)
    assert len(unique_yw) == 2, f"Expected 2 unique weeks, got {unique_yw}"
    assert g_weeks[0] != g_weeks[1]
    assert unique_yw[0] != unique_yw[1]


def test_year_boundary_strata_separation():
    """Verify year boundary sessions remain strictly separated by calendar-year strata (C2)."""
    from memory_study_v2.inference import build_calendar_weeks_mapping
    dates = ["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]
    g_weeks, years, unique_yw = build_calendar_weeks_mapping(dates, 4)
    assert years == [2024, 2024, 2025, 2025]
    # Unique year-week tuples must preserve calendar year strata
    y_2024 = [yw for yw in unique_yw if yw[0] == 2024]
    y_2025 = [yw for yw in unique_yw if yw[0] == 2025]
    assert len(y_2024) >= 1
    assert len(y_2025) >= 1
    assert all(yw[0] == 2024 for yw in y_2024)
    assert all(yw[0] == 2025 for yw in y_2025)


def test_malformed_session_dates_rejected():
    """Verify malformed session dates are strictly rejected without fallback (C2)."""
    from memory_study_v2.inference import build_calendar_weeks_mapping
    bad_dates = ["2024-01-02", "2024/01/03"]
    with pytest.raises(ValueError, match="Malformed production session date"):
        build_calendar_weeks_mapping(bad_dates, 2)


def test_holiday_observed_by_only_one_market_sufficient_stats_oracle():
    """Verify per-series holiday mask produces exact numerical identity with direct daily calculations (C2)."""
    from memory_study_v2.inference import build_calendar_weeks_mapping, compute_weekly_sufficient_statistics
    # 10 trading sessions
    dates = [
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08",
        "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12", "2024-01-15",
    ]
    g_weeks, years, unique_yw = build_calendar_weeks_mapping(dates, 10)

    # Market 0 has complete 10 days; Market 1 has a holiday on day 2 and day 7 (NaN)
    r0 = np.array([0.01, 0.02, -0.01, 0.005, 0.015, 0.01, -0.005, 0.02, -0.01, 0.005], dtype=np.float64)
    r1 = np.array([0.005, 0.01, np.nan, 0.015, -0.005, 0.02, -0.01, np.nan, 0.015, -0.005], dtype=np.float64)
    ret_matrix = np.stack([r0, r1], axis=0)

    w_counts, w_sums, w_sumsq = compute_weekly_sufficient_statistics(ret_matrix, g_weeks, len(unique_yw))

    # Test draw selecting both weeks
    draw_counts = np.sum(w_counts, axis=0)
    draw_sums = np.sum(w_sums, axis=0)
    draw_sumsq = np.sum(w_sumsq, axis=0)

    # Direct daily calculation excluding NaNs
    valid_r1 = r1[np.isfinite(r1)]
    direct_sr1 = compute_sharpe_ratio(valid_r1)

    # From weekly sufficient stats
    N1 = draw_counts[1]
    S1 = draw_sums[1]
    S2 = draw_sumsq[1]
    mean1 = S1 / N1
    var1 = (S2 / N1) - (mean1 ** 2)
    stat_sr1 = (mean1 * 252.0) / (np.sqrt(var1) * np.sqrt(252.0))

    assert N1 == 8.0, f"Expected 8 valid sessions for Market 1, got {N1}"
    assert abs(direct_sr1 - stat_sr1) < 1e-12, "Sufficient stats Sharpe must match direct daily calculation (<1e-12)"


def test_missing_required_market_path_rejected():
    """Verify evaluator strictly rejects missing required market paths when allow_reduced_arms=False (C2)."""
    markets = ["US", "IN"]
    arms = [
        "MEM_SIM", "KNN_PLAIN", "MEM_RANDOM", "HIST_PRIOR", "RIDGE_ANNUAL",
        "MLP_BASE", "TRANS_BASE", "MLP_MIX_SR", "TRANS_MIX_SR", "MLP_GATE", "TRANS_GATE"
    ]
    returns_map = {}
    for arm in arms:
        for mkt in markets:
            # Deliberately omit market IN for MLP_BASE
            if arm == "MLP_BASE" and mkt == "IN":
                continue
            returns_map[(arm, mkt, None)] = np.array([0.01] * 10)

    with pytest.raises(ValueError, match="Required market path missing for arm 'MLP_BASE'"):
        evaluate_primary_contrasts(returns_map, markets, allow_reduced_arms=False)


def test_replay_detects_renamed_contrast_id(tmp_path):
    """Verify replay rejects tampered/renamed contrast IDs (C2)."""
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

    contrast_results, draw_matrix, draw_weeks = evaluate_primary_contrasts(returns_map, markets, num_draws=50)
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results, draw_matrix, draw_weeks, tmp_path / "bundle_rename", markets=markets, num_draws=50
    )

    contrasts_file = bundle_dir / "primary_contrasts.json"
    with open(contrasts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data[0]["contrast_id"] = "P_RENAMED"
    with open(contrasts_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ReplayVerificationError, match="Contrast ID mismatch"):
        replay_analysis_bundle(bundle_dir)


def test_replay_detects_truncated_summary_list(tmp_path):
    """Verify replay rejects truncated summary lists (C2)."""
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

    contrast_results, draw_matrix, draw_weeks = evaluate_primary_contrasts(returns_map, markets, num_draws=50)
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results, draw_matrix, draw_weeks, tmp_path / "bundle_trunc", markets=markets, num_draws=50
    )

    contrasts_file = bundle_dir / "primary_contrasts.json"
    with open(contrasts_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop()  # Drop last contrast
    with open(contrasts_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with pytest.raises(ReplayVerificationError, match="Stored contrast count"):
        replay_analysis_bundle(bundle_dir)


def test_replay_detects_nan_corrupted_stored_draws(tmp_path):
    """Verify replay rejects NaN-corrupted stored draws (C2)."""
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

    contrast_results, draw_matrix, draw_weeks = evaluate_primary_contrasts(returns_map, markets, num_draws=50)
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrast_results, draw_matrix, draw_weeks, tmp_path / "bundle_nan", markets=markets, num_draws=50
    )

    draws_file = bundle_dir / "contrast_draws.npy"
    draws = np.load(draws_file)
    draws[0, 5] = np.nan
    np.save(draws_file, draws)

    with pytest.raises(ReplayVerificationError, match="contains non-finite"):
        replay_analysis_bundle(bundle_dir)


# ---------------------------------------------------------------------------
# Finding 2 / C2: Canonical mask acceptance tests
# ---------------------------------------------------------------------------

def _make_minimal_returns_map(n_sessions=60, seed=7):
    """Helper: minimal returns map with MEM_SIM and MLP_BASE for US market only."""
    rng = np.random.default_rng(seed)
    arms = ["MEM_SIM", "MLP_BASE", "TRANS_BASE", "MEM_RANDOM", "HIST_PRIOR",
            "RIDGE_ANNUAL", "KNN_PLAIN", "MLP_GATE", "TRANS_GATE", "MLP_MIX_SR", "TRANS_MIX_SR"]
    markets = ["US", "IN", "CN", "FR", "GB", "BR"]
    m = {}
    for arm in arms:
        for mkt in markets:
            m[(arm, mkt, None)] = rng.normal(0.0004, 0.01, n_sessions)
    return m, markets


def test_sharpe_math_oracle_c2():
    """Finding 2: Verify Sharpe formula matches auditor's two-row illustration.

    Input:        [0.01, 0.02, 0.00, -0.01]  → annualized Sharpe ≈ 7.0993
    Masked (3rd): [0.01, 0.02, -0.01]        → annualized Sharpe ≈ 8.4853
    """
    from memory_study_v2.inference import compute_sharpe_ratio
    import math

    r4 = np.array([0.01, 0.02, 0.00, -0.01], dtype=np.float64)
    r3 = np.array([0.01, 0.02, -0.01], dtype=np.float64)

    sr4 = compute_sharpe_ratio(r4)
    sr3 = compute_sharpe_ratio(r3)

    assert abs(sr4 - 7.0993) < 1e-3, f"4-obs Sharpe expected 7.0993, got {sr4:.4f}"
    assert abs(sr3 - 8.4853) < 1e-3, f"3-obs Sharpe expected 8.4853, got {sr3:.4f}"
    assert sr3 > sr4, "Excluding the zero-return session must increase Sharpe"


def test_masked_out_session_change_does_not_affect_estimates_c2(tmp_path):
    """Finding 2 (a): Changing a return where valid_mask=False must NOT change point estimate or draws."""
    returns_map, markets = _make_minimal_returns_map(n_sessions=60, seed=11)
    n_sessions = 60

    # Create a mask that excludes session 5 (closed-market placeholder)
    valid_mask = np.ones(n_sessions, dtype=bool)
    valid_mask[5] = False

    # Baseline evaluation with original returns
    contrasts_orig, draws_orig, weeks_orig = evaluate_primary_contrasts(
        returns_map, markets, valid_mask=valid_mask, num_draws=100, allow_reduced_arms=True
    )
    theta_orig = {c.contrast_id: c.theta for c in contrasts_orig if c.status == "COMPLETED"}

    # Change a masked-out return (session 5) in every series → must be irrelevant
    returns_map_changed = {k: v.copy() for k, v in returns_map.items()}
    for k in returns_map_changed:
        returns_map_changed[k][5] = 999.0  # large change to masked session

    contrasts_changed, draws_changed, _ = evaluate_primary_contrasts(
        returns_map_changed, markets, valid_mask=valid_mask,
        draw_week_indices=weeks_orig, num_draws=100, allow_reduced_arms=True
    )

    # Point estimates must be identical
    for cid, th in theta_orig.items():
        ch_theta = next(c.theta for c in contrasts_changed if c.contrast_id == cid)
        assert abs(th - ch_theta) < 1e-12, (
            f"Contrast {cid}: changing a masked-out return changed theta {th:.6f} → {ch_theta:.6f}"
        )

    # Bootstrap draws must be identical
    np.testing.assert_array_equal(draws_orig, draws_changed,
        err_msg="Changing a masked-out return must not affect any bootstrap draw")


def test_valid_return_change_affects_estimates_c2():
    """Finding 2 (b): Changing a valid return (valid_mask=True) must affect point estimate and draws."""
    returns_map, markets = _make_minimal_returns_map(n_sessions=60, seed=13)
    n_sessions = 60
    valid_mask = np.ones(n_sessions, dtype=bool)

    contrasts_orig, draws_orig, weeks_orig = evaluate_primary_contrasts(
        returns_map, markets, valid_mask=valid_mask, num_draws=100, allow_reduced_arms=True
    )
    theta_orig = {c.contrast_id: c.theta for c in contrasts_orig if c.status == "COMPLETED"}

    # Change session 10 (an open session) to an extreme value in ALL series
    returns_map_changed = {k: v.copy() for k, v in returns_map.items()}
    for k in returns_map_changed:
        returns_map_changed[k][10] = 9.9  # extreme change

    contrasts_changed, draws_changed, _ = evaluate_primary_contrasts(
        returns_map_changed, markets, valid_mask=valid_mask,
        draw_week_indices=weeks_orig, num_draws=100, allow_reduced_arms=True
    )

    # At least one theta must have changed
    any_changed = any(
        abs(theta_orig.get(c.contrast_id, 0.0) - c.theta) > 1e-12
        for c in contrasts_changed if c.status == "COMPLETED"
    )
    assert any_changed, "Changing a valid return must affect at least one point estimate"


def test_mask_corruption_detected_in_replay_c2(tmp_path):
    """Finding 2 (c): Corrupting the stored mask must cause replay to raise ReplayVerificationError."""
    returns_map, markets = _make_minimal_returns_map(n_sessions=60, seed=17)
    n_sessions = 60
    valid_mask = np.ones(n_sessions, dtype=bool)
    valid_mask[3] = False

    contrasts, draws, weeks = evaluate_primary_contrasts(
        returns_map, markets, valid_mask=valid_mask, num_draws=50, allow_reduced_arms=True
    )
    bundle_dir = export_analysis_bundle(
        {f"{k[0]}__{k[1]}__{k[2]}": list(v) for k, v in returns_map.items()},
        contrasts, draws, weeks,
        tmp_path / "bundle_mask",
        valid_mask=valid_mask,
        markets=markets,
        num_draws=50,
        allow_reduced_arms=True,
    )

    # Corrupt the stored mask in daily_returns.json
    import json as _json
    with open(bundle_dir / "daily_returns.json", "r", encoding="utf-8") as f:
        payload = _json.load(f)
    # Replace with a mask of wrong type to trigger corruption detection
    payload["open_session_mask"] = "corrupted_value_not_a_list"
    with open(bundle_dir / "daily_returns.json", "w", encoding="utf-8") as f:
        _json.dump(payload, f)

    with pytest.raises(ReplayVerificationError):
        replay_analysis_bundle(bundle_dir)


def test_open_session_missing_return_raises_c2():
    """Finding 2 (d): valid_mask=True but return is NaN must raise ValueError."""
    returns_map, markets = _make_minimal_returns_map(n_sessions=60, seed=19)
    n_sessions = 60

    # Mark session 7 as open, but inject NaN into the first series
    valid_mask = np.ones(n_sessions, dtype=bool)
    first_key = list(returns_map.keys())[0]
    returns_map[first_key][7] = float("nan")  # NaN on open session

    with pytest.raises(ValueError, match="unexpected missing"):
        evaluate_primary_contrasts(
            returns_map, markets, valid_mask=valid_mask, num_draws=50, allow_reduced_arms=True
        )
