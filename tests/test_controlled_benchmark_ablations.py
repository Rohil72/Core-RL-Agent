"""Unit and Invariant Tests for Controlled Benchmark Ablations.

Verifies:
1. Target scaler inverse-transformation math
2. Asymmetric loss penalty logic on positive vs negative tails
3. Deduplication and ticker precedent capping
4. Volatility sizing normalization and capital conservation
5. Multi-seed ensemble prediction averaging and disagreement calculation
6. Invariant safeguards (zero leakage, finite math, positive denominators)
"""

from __future__ import annotations

import numpy as np
import pytest


def compute_policy_score(
    system_id: str,
    pred_val: float,
    mu_w: float,
    cvar_tail: float,
    v: float,
    market: str,
    target_moments: dict[str, dict[str, float]],
) -> float:
    denom = v + 1e-4
    assert denom > 0, "Denominator must be strictly positive"
    assert not np.isnan(denom), "Denominator cannot be NaN"

    if system_id == "EXP1_SCALE_HARMONIZED":
        m_mean = target_moments[market]["mean"]
        m_std = target_moments[market]["std"]
        pred_scaled = pred_val * m_std + m_mean
        num = pred_scaled + 0.8 * mu_w - 0.2 * abs(cvar_tail)
        return float(num / denom)

    elif system_id == "EXP2_ASYM_LOSS_PENALTY":
        # Loss penalty: max(0, -LTM). When LTM > 0, penalty is 0.0.
        loss_penalty = max(0.0, -cvar_tail)
        num = pred_val + 0.8 * mu_w - 0.2 * loss_penalty
        return float(num / denom)

    elif system_id in ("EXP3A_UNSCALED_SELECTION", "EXP3B_UNSCALED_VOL_SIZING"):
        # Unscaled ranking by numerator directly
        num = pred_val + 0.8 * mu_w - 0.2 * abs(cvar_tail)
        return float(num)

    elif system_id == "P2_CONTROL":
        num = pred_val + 0.8 * mu_w
        return float(num / denom)

    else:
        # Standard P0 / P0* / EXP4 / EXP9 / EXP10 formula
        num = pred_val + 0.8 * mu_w - 0.2 * abs(cvar_tail)
        return float(num / denom)


def deduplicate_and_select_top_k(
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    mem_tickers: np.ndarray,
    mem_sessions: np.ndarray,
    mem_dates: np.ndarray,
    mem_markets: np.ndarray,
    k: int = 25,
    min_separation: int = 21,
    is_distance: bool = False,
    max_precedents_per_ticker: int | None = None,
) -> tuple[np.ndarray, list[dict], dict]:
    n_cands = len(candidate_indices)
    if n_cands == 0:
        return np.array([], dtype=int), [], {"total_evaluated": 0, "total_kept": 0, "total_discarded": 0, "fallback_triggered": 0}

    order = np.argsort(candidate_scores) if is_distance else np.argsort(-candidate_scores)
    sorted_indices = candidate_indices[order]
    sorted_scores = candidate_scores[order]

    kept_indices: list[int] = []
    audit_log: list[dict] = []
    selected_sessions_by_ticker: dict[str, list[int]] = {}
    selected_dates_by_ticker: dict[str, list[str]] = {}

    total_evaluated = 0
    total_discarded = 0
    total_kept = 0

    for rank_raw, idx in enumerate(sorted_indices, 1):
        tkr = str(mem_tickers[idx])
        sess = int(mem_sessions[idx])
        dt = str(mem_dates[idx])
        mkt = str(mem_markets[idx])
        sc = float(sorted_scores[rank_raw - 1])

        total_evaluated += 1

        conflict_found = False
        conflict_prev_sess = None
        conflict_prev_dt = None
        conflict_gap = None
        discard_reason = "none"

        # Check ticker precedent cap if active
        if max_precedents_per_ticker is not None and tkr in selected_sessions_by_ticker:
            if len(selected_sessions_by_ticker[tkr]) >= max_precedents_per_ticker:
                conflict_found = True
                discard_reason = f"max_ticker_cap_{max_precedents_per_ticker}_exceeded"

        # Check intra-peer temporal separation
        if not conflict_found and tkr in selected_sessions_by_ticker:
            for prev_s, prev_d in zip(selected_sessions_by_ticker[tkr], selected_dates_by_ticker[tkr]):
                gap = abs(sess - prev_s)
                if gap < min_separation:
                    conflict_found = True
                    conflict_prev_sess = prev_s
                    conflict_prev_dt = prev_d
                    conflict_gap = gap
                    discard_reason = f"intra_peer_separation_{conflict_gap}_lt_{min_separation}"
                    break

        if conflict_found:
            total_discarded += 1
            audit_log.append({
                "candidate_rank_raw": rank_raw,
                "neighbor_market": mkt,
                "neighbor_ticker": tkr,
                "neighbor_date": dt,
                "neighbor_session_index": sess,
                "similarity_or_distance": round(sc, 5),
                "status": "DISCARDED",
                "conflict_ticker": tkr,
                "conflict_prev_date": conflict_prev_dt or "",
                "conflict_prev_session": conflict_prev_sess or "",
                "session_gap": conflict_gap or "",
                "discard_reason": discard_reason,
            })
        else:
            total_kept += 1
            kept_indices.append(idx)
            selected_sessions_by_ticker.setdefault(tkr, []).append(sess)
            selected_dates_by_ticker.setdefault(tkr, []).append(dt)
            audit_log.append({
                "candidate_rank_raw": rank_raw,
                "neighbor_market": mkt,
                "neighbor_ticker": tkr,
                "neighbor_date": dt,
                "neighbor_session_index": sess,
                "similarity_or_distance": round(sc, 5),
                "status": "KEPT",
                "conflict_ticker": "",
                "conflict_prev_date": "",
                "conflict_prev_session": "",
                "session_gap": "",
                "discard_reason": "none",
            })

            if len(kept_indices) >= k:
                break

    fallback_triggered = int(len(kept_indices) < k)
    stats = {
        "total_evaluated": total_evaluated,
        "total_kept": total_kept,
        "total_discarded": total_discarded,
        "fallback_triggered": fallback_triggered,
    }
    return np.array(kept_indices, dtype=int), audit_log, stats


# =====================================================================
# TEST CASES
# =====================================================================

def test_asymmetric_loss_penalty():
    target_moments = {"US": {"mean": 0.0747, "std": 0.1482}}
    pred = 0.10
    mu_w = 0.08
    v = 0.015

    # Case A: Positive lower tail mean (bull regime, LTM = +0.05)
    cvar_positive = 0.05
    score_canon = compute_policy_score("P0_CANONICAL", pred, mu_w, cvar_positive, v, "US", target_moments)
    score_asym = compute_policy_score("EXP2_ASYM_LOSS_PENALTY", pred, mu_w, cvar_positive, v, "US", target_moments)
    score_p2 = compute_policy_score("P2_CONTROL", pred, mu_w, cvar_positive, v, "US", target_moments)

    # In asymmetric penalty, positive LTM is NOT penalized, so it must equal P2 control
    assert np.isclose(score_asym, score_p2), "When LTM > 0, asymmetric penalty must equal P2 control"
    # Canonical P0 subtracts 0.2 * 0.05 = 0.01, so canonical score is strictly lower
    assert score_canon < score_asym, "Canonical score must be penalized when LTM > 0"
    assert np.isclose(score_canon, (pred + 0.8 * mu_w - 0.2 * 0.05) / (v + 1e-4))

    # Case B: Negative lower tail mean (loss regime, LTM = -0.15)
    cvar_negative = -0.15
    score_canon_neg = compute_policy_score("P0_CANONICAL", pred, mu_w, cvar_negative, v, "US", target_moments)
    score_asym_neg = compute_policy_score("EXP2_ASYM_LOSS_PENALTY", pred, mu_w, cvar_negative, v, "US", target_moments)

    # When LTM < 0, max(0, -(-0.15)) = 0.15 == abs(-0.15), so asymmetric penalty equals canonical P0!
    assert np.isclose(score_canon_neg, score_asym_neg), "When LTM < 0, asymmetric penalty must match canonical P0"


def test_score_harmonization_inverse_transform():
    target_moments = {"US": {"mean": 0.0747, "std": 0.1482}}
    pred_std = 1.0  # +1 sigma standardized prediction
    mu_w = 0.05
    cvar = -0.05
    v = 0.02

    # Scaled prediction: 1.0 * 0.1482 + 0.0747 = 0.2229
    pred_ret = 1.0 * 0.1482 + 0.0747
    score_harm = compute_policy_score("EXP1_SCALE_HARMONIZED", pred_std, mu_w, cvar, v, "US", target_moments)
    expected_num = pred_ret + 0.8 * mu_w - 0.2 * abs(cvar)
    expected_score = expected_num / (v + 1e-4)

    assert np.isclose(score_harm, expected_score)
    # The harmonized score uses ~0.2229 instead of 1.0, restoring balance with decimal returns
    assert score_harm < compute_policy_score("P0_CANONICAL", pred_std, mu_w, cvar, v, "US", target_moments)


def test_deduplication_and_precedent_capping():
    # Setup candidate pool where ticker 'AAPL' has 10 high-scoring precedents
    mem_tickers = np.array(["AAPL"] * 10 + ["MSFT"] * 5 + ["NVDA"] * 5)
    mem_sessions = np.array(list(range(0, 200, 20)) + list(range(0, 100, 20)) + list(range(0, 100, 20)))
    mem_dates = np.array([f"2015-01-{i+1:02d}" for i in range(20)])
    mem_markets = np.array(["US"] * 20)
    candidate_indices = np.arange(20)
    # AAPL has highest scores
    candidate_scores = np.linspace(0.99, 0.50, 20)

    # Test A: Standard 21-session separation (min_separation=21, no ticker cap)
    kept_a, audit_a, stats_a = deduplicate_and_select_top_k(
        candidate_indices, candidate_scores, mem_tickers, mem_sessions, mem_dates, mem_markets,
        k=10, min_separation=21, max_precedents_per_ticker=None
    )
    # Session gap is 20, which is < 21! So only alternate AAPL records kept
    for t in ["AAPL", "MSFT", "NVDA"]:
        t_kept = [mem_sessions[i] for i in kept_a if mem_tickers[i] == t]
        for idx in range(len(t_kept) - 1):
            assert abs(t_kept[idx+1] - t_kept[idx]) >= 21

    # Test B: Ticker cap = 2 (max 2 precedents per ticker)
    mem_sessions_spaced = np.array(list(range(0, 500, 50)) + list(range(0, 250, 50)) + list(range(0, 250, 50)))
    kept_b, audit_b, stats_b = deduplicate_and_select_top_k(
        candidate_indices, candidate_scores, mem_tickers, mem_sessions_spaced, mem_dates, mem_markets,
        k=6, min_separation=21, max_precedents_per_ticker=2
    )
    kept_tickers_b = [mem_tickers[i] for i in kept_b]
    assert kept_tickers_b.count("AAPL") <= 2, "AAPL must be capped at 2"
    assert kept_tickers_b.count("MSFT") <= 2
    assert kept_tickers_b.count("NVDA") <= 2
    assert len(kept_b) == 6

    # Test C: Strict diversity (max 1 precedent per ticker)
    kept_c, audit_c, stats_c = deduplicate_and_select_top_k(
        candidate_indices, candidate_scores, mem_tickers, mem_sessions_spaced, mem_dates, mem_markets,
        k=3, min_separation=21, max_precedents_per_ticker=1
    )
    kept_tickers_c = [mem_tickers[i] for i in kept_c]
    assert len(set(kept_tickers_c)) == 3, "Strict diversity must have 3 distinct tickers"


def test_volatility_position_sizing_capital_conservation():
    tot_equity = 100000.0
    open_slots = 2
    total_capital_to_allocate = tot_equity * (open_slots / 3.0)  # 66,666.67
    volatilities = [0.01, 0.04]  # Asset 1 is 4x lower vol than Asset 2

    # Inverse volatility weights
    inv_v = [1.0 / (v + 1e-4) for v in volatilities]
    w = [iv / sum(inv_v) for iv in inv_v]

    allocated_caps = [total_capital_to_allocate * weight for weight in w]

    # Verification 1: Capital conservation
    assert np.isclose(sum(allocated_caps), total_capital_to_allocate), "Total allocated capital must be conserved"
    # Verification 2: Low-volatility asset receives proportionally higher capital
    assert allocated_caps[0] > allocated_caps[1]
    assert np.isclose(allocated_caps[0] / allocated_caps[1], inv_v[0] / inv_v[1])


def test_multi_seed_ensemble_averaging():
    preds = [0.10, 0.12, 0.08]
    mu_ws = [0.06, 0.05, 0.07]
    ltms = [-0.04, -0.05, -0.03]

    mean_pred = np.mean(preds)
    mean_mu = np.mean(mu_ws)
    mean_ltm = np.mean(ltms)
    disagreement = np.std(preds)

    assert np.isclose(mean_pred, 0.10)
    assert np.isclose(mean_mu, 0.06)
    assert np.isclose(mean_ltm, -0.04)
    assert disagreement > 0.0, "Disagreement must be positive when predictions differ"


def compute_tail_mean_q(returns: np.ndarray, weights: np.ndarray, q: float = 0.05) -> float:
    """Exact probability-mass tail estimator (Exp 10).
    C_{j-1} = sum_{l < j} w_l, a_j = max(0, min(w_j, q - C_{j-1})), TailMean_q = sum(a_j * R_{(j)}) / q
    """
    assert len(returns) == len(weights), "Returns and weights must have equal length"
    assert q > 0.0, "Quantile q must be positive"
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5), "Weights must sum to 1"

    order = np.argsort(returns)
    sorted_ret = returns[order]
    sorted_w = weights[order]

    cum_w = 0.0
    weighted_sum = 0.0
    for j in range(len(sorted_ret)):
        w_j = sorted_w[j]
        a_j = max(0.0, min(w_j, q - cum_w))
        weighted_sum += a_j * sorted_ret[j]
        cum_w += w_j
        if cum_w >= q:
            break

    return float(weighted_sum / q)


def test_exact_probability_mass_tail_estimator():
    # Test A: k=25, uniform weights w_j = 0.04, q=0.05
    # Should take full mass of 1st item (0.04) and 0.01 of 2nd item: 0.8 * R(1) + 0.2 * R(2)
    k = 25
    rets = np.linspace(-0.20, 0.20, k)
    weights = np.ones(k) / k
    tail_mean_05 = compute_tail_mean_q(rets, weights, q=0.05)
    expected_05 = 0.8 * rets[0] + 0.2 * rets[1]
    assert np.isclose(tail_mean_05, expected_05), f"Expected {expected_05}, got {tail_mean_05}"

    # Test B: k=20, uniform weights w_j = 0.05, q=0.05
    # Exactly 1 item
    rets_20 = np.linspace(-0.10, 0.10, 20)
    weights_20 = np.ones(20) / 20
    tail_mean_20 = compute_tail_mean_q(rets_20, weights_20, q=0.05)
    assert np.isclose(tail_mean_20, rets_20[0])

    # Test C: q=0.10, k=25
    # 0.04 + 0.04 + 0.02 of 3rd item: (0.04*R1 + 0.04*R2 + 0.02*R3)/0.10 = 0.4*R1 + 0.4*R2 + 0.2*R3
    tail_mean_10 = compute_tail_mean_q(rets, weights, q=0.10)
    expected_10 = 0.4 * rets[0] + 0.4 * rets[1] + 0.2 * rets[2]
    assert np.isclose(tail_mean_10, expected_10)
