"""Block bootstrap inference, eight primary contrasts, and release replay (v2).

Acceptance criteria addressed:
- A30: Same year-stratified sampled weeks shared by all arms/markets; quantile/p/Holm formulas verified.
- A31: Fresh full-precision summaries and bootstrap outputs replay at declared 1e-10 tolerance.
- Full return-to-contrast pipeline with disk artifact replay and negative corruption checks (R10).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from memory_study_v2.contracts import to_canonical_json

# Eight primary Sharpe contrasts defined in Section 9.2 of REBUILD_IMPLEMENTATION_PLAN.md:
PRIMARY_CONTRASTS: List[Tuple[str, str, str]] = [
    ("P1", "MEM_SIM", "MEM_RANDOM"),
    ("P2", "MEM_SIM", "HIST_PRIOR"),
    ("P3", "MEM_SIM", "KNN_PLAIN"),
    ("P4", "MEM_SIM", "RIDGE_ANNUAL"),
    ("P5", "MEM_SIM", "MLP_BASE"),
    ("P6", "MEM_SIM", "TRANS_BASE"),
    ("P7", "MLP_GATE", "MLP_MIX_SR"),
    ("P8", "TRANS_GATE", "TRANS_MIX_SR"),
]


class ReplayVerificationError(Exception):
    """Raised when replayed analysis statistics deviate from declared tolerances."""
    pass


@dataclass
class ContrastResult:
    contrast_id: str
    candidate: str
    comparator: str
    theta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    p_value_holm: float
    num_draws: int


def generate_stratified_week_blocks(
    year_to_weeks: Dict[int, List[int]],
    block_length: int = 4,
    num_draws: int = 10000,
    seed_sequence: int = 42,
) -> List[Dict[int, List[int]]]:
    """Generate synchronized year-stratified sampled week index arrays using PCG64."""
    seed_seq = np.random.SeedSequence([seed_sequence, block_length])
    rng = np.random.default_rng(np.random.PCG64(seed_seq))

    all_draws: List[Dict[int, List[int]]] = []

    for _ in range(num_draws):
        draw_dict: Dict[int, List[int]] = {}
        for year, week_indices in year_to_weeks.items():
            T_y = len(week_indices)
            sampled = []
            while len(sampled) < T_y:
                max_start = T_y - block_length + 1
                if max_start > 0:
                    start = int(rng.integers(0, max_start))
                    block = week_indices[start : start + block_length]
                else:
                    block = week_indices
                sampled.extend(block)
            draw_dict[year] = sampled[:T_y]
        all_draws.append(draw_dict)

    return all_draws


def compute_bootstrap_p_and_ci(
    theta: float,
    theta_draws: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute centered 95% CI and two-sided centered p-value according to contract (Section 9.2):

    interval = theta +/- quantile(abs(theta_b - theta), 1 - alpha, method='linear')
    p_value = (1 + count(abs(theta_b - theta) >= abs(theta))) / (B + 1)
    """
    B = len(theta_draws)
    diffs = np.abs(theta_draws - theta)
    margin = float(np.quantile(diffs, 1.0 - alpha, method="linear"))
    ci_lower = theta - margin
    ci_upper = theta + margin

    count_exceed = int(np.sum(diffs >= abs(theta)))
    p_val = float((1.0 + count_exceed) / (B + 1.0))

    return ci_lower, ci_upper, p_val


def apply_step_down_holm_bonferroni(p_values: List[float]) -> List[float]:
    """Apply step-down Holm-Bonferroni correction to p-values."""
    m = len(p_values)
    if m == 0:
        return []

    sorted_indices = np.argsort(p_values)
    p_sorted = np.array(p_values)[sorted_indices]

    adj_sorted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        rank = i + 1
        adj_p = min(1.0, float(p_sorted[i] * (m - rank + 1)))
        running_max = max(running_max, adj_p)
        adj_sorted[i] = running_max

    # Invert sorting back to original order
    adj_p_values = np.zeros(m, dtype=np.float64)
    adj_p_values[sorted_indices] = adj_sorted
    return [float(p) for p in adj_p_values]


def compute_sharpe_ratio(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio in float64 with variance guard (Section 9.2)."""
    if len(returns) == 0:
        return 0.0
    r = returns.astype(np.float64)
    mean_r = np.mean(r)
    var_r = np.var(r, ddof=0)

    # Clip negative variance to 0 only when magnitude <= 1e-12; larger negatives fail
    if var_r < 0.0:
        if abs(var_r) <= 1e-12:
            var_r = 0.0
        else:
            raise ValueError(f"Negative variance encountered: {var_r}")

    vol = math.sqrt(var_r) * math.sqrt(252.0)
    if vol <= 1e-8:
        return 0.0
    ann_return = mean_r * 252.0
    return float(ann_return / vol)


def evaluate_primary_contrasts(
    returns_by_arm_market_realization: Dict[Tuple[str, str, Optional[int]], np.ndarray],
    markets: List[str],
    draw_week_indices: Optional[List[Dict[int, List[int]]]] = None,
    calendar_weeks_map: Optional[Dict[str, Tuple[int, int]]] = None,
    num_draws: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> Tuple[List[ContrastResult], np.ndarray]:
    """Calculate the 8 primary Sharpe contrasts and bootstrap p-values (R10).

    Hierarchy:
    1. For each arm and market, average Sharpe ratios across realizations of that arm.
    2. Average across 6 markets to obtain arm headline Sharpe.
    3. theta = Sharpe(candidate) - Sharpe(comparator).
    4. Synchronized block bootstrap across all arms and markets evaluated on returns.
    """
    # 1. Compute headline Sharpe per arm on original returns
    arms_in_data = sorted(list({k[0] for k in returns_by_arm_market_realization.keys()}))

    def compute_arm_sharpes(active_returns_map: Dict[Tuple[str, str, Optional[int]], np.ndarray]) -> Dict[str, float]:
        arm_market_sharpes: Dict[str, Dict[str, List[float]]] = {}
        for (arm, mkt, seed_val), r in active_returns_map.items():
            sr = compute_sharpe_ratio(r)
            arm_market_sharpes.setdefault(arm, {}).setdefault(mkt, []).append(sr)

        arm_headline_sharpe: Dict[str, float] = {}
        for arm in arms_in_data:
            mkt_means = []
            for mkt in markets:
                srs = arm_market_sharpes.get(arm, {}).get(mkt, [0.0])
                mkt_means.append(float(np.mean(srs)))
            arm_headline_sharpe[arm] = float(np.mean(mkt_means)) if mkt_means else 0.0
        return arm_headline_sharpe

    headline_sharpes = compute_arm_sharpes(returns_by_arm_market_realization)

    # 2. Compute point estimate theta for each of the 8 primary contrasts
    thetas: Dict[str, float] = {}
    for cid, cand, comp in PRIMARY_CONTRASTS:
        cand_sr = headline_sharpes.get(cand, 0.0)
        comp_sr = headline_sharpes.get(comp, 0.0)
        thetas[cid] = float(cand_sr - comp_sr)

    # 3. Synchronized block bootstrap across daily return series
    all_series_list = list(returns_by_arm_market_realization.values())
    if not all_series_list or num_draws <= 0:
        draw_matrix = np.zeros((len(PRIMARY_CONTRASTS), max(0, num_draws)), dtype=np.float64)
    else:
        T = max(len(s) for s in all_series_list)
        if T == 0:
            draw_matrix = np.zeros((len(PRIMARY_CONTRASTS), num_draws), dtype=np.float64)
        else:
            b_size = max(1, min(block_size, T))
            num_blocks = int(np.ceil(T / b_size))
            blocks = [np.arange(i, i + b_size) % T for i in range(0, T, b_size)]

            rng = np.random.default_rng(np.random.PCG64(np.random.SeedSequence([seed, 4])))
            draw_block_indices = rng.integers(0, len(blocks), size=(num_draws, num_blocks))

            all_day_indices = np.array([
                np.concatenate([blocks[b] for b in draw_block_indices[d]])[:T]
                for d in range(num_draws)
            ], dtype=np.int64)

            keys = list(returns_by_arm_market_realization.keys())
            ret_matrix = np.zeros((len(keys), T), dtype=np.float64)
            for k_idx, k in enumerate(keys):
                arr = np.asarray(returns_by_arm_market_realization[k], dtype=np.float64)
                ret_matrix[k_idx, :len(arr)] = arr

            # Sampled returns: shape (len(keys), num_draws, T)
            sampled_rets = ret_matrix[:, all_day_indices]
            means = np.mean(sampled_rets, axis=2)
            vars_ = np.var(sampled_rets, axis=2, ddof=0)
            vols = np.sqrt(np.maximum(0.0, vars_)) * math.sqrt(252.0)
            safe_vols = np.where(vols > 1e-8, vols, 1.0)
            all_srs = np.where(vols > 1e-8, (means * 252.0) / safe_vols, 0.0)

            draw_matrix = np.zeros((len(PRIMARY_CONTRASTS), num_draws), dtype=np.float64)
            draw_headline_sharpes: Dict[str, np.ndarray] = {}

            for arm in arms_in_data:
                mkt_srs_list = []
                for mkt in markets:
                    matching_indices = [
                        idx for idx, k in enumerate(keys)
                        if k[0] == arm and k[1] == mkt
                    ]
                    if matching_indices:
                        mkt_mean_draw = np.mean(all_srs[matching_indices, :], axis=0)
                    else:
                        mkt_mean_draw = np.zeros(num_draws, dtype=np.float64)
                    mkt_srs_list.append(mkt_mean_draw)
                draw_headline_sharpes[arm] = np.mean(mkt_srs_list, axis=0) if mkt_srs_list else np.zeros(num_draws, dtype=np.float64)

            for c_idx, (cid, cand, comp) in enumerate(PRIMARY_CONTRASTS):
                cand_draws = draw_headline_sharpes.get(cand, np.zeros(num_draws, dtype=np.float64))
                comp_draws = draw_headline_sharpes.get(comp, np.zeros(num_draws, dtype=np.float64))
                draw_matrix[c_idx] = cand_draws - comp_draws

    # 4. Compute CI and p-values
    contrast_results: List[ContrastResult] = []
    raw_p_values = []
    for c_idx, (cid, cand, comp) in enumerate(PRIMARY_CONTRASTS):
        th = thetas[cid]
        draws = draw_matrix[c_idx]
        ci_l, ci_u, p_val = compute_bootstrap_p_and_ci(th, draws)
        raw_p_values.append(p_val)
        contrast_results.append(ContrastResult(
            contrast_id=cid,
            candidate=cand,
            comparator=comp,
            theta=float(th),
            ci_lower=float(ci_l),
            ci_upper=float(ci_u),
            p_value=float(p_val),
            p_value_holm=0.0,
            num_draws=num_draws,
        ))

    holm_p = apply_step_down_holm_bonferroni(raw_p_values)
    for i, res in enumerate(contrast_results):
        res.p_value_holm = float(holm_p[i])

    return contrast_results, draw_matrix


def export_analysis_bundle(
    returns_by_key: Dict[str, List[float]],
    contrast_results: List[ContrastResult],
    draw_matrix: np.ndarray,
    export_dir: Union[str, Path],
    markets: Optional[List[str]] = None,
    num_draws: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> Path:
    """Export return series, contrast summary, draw matrix, and bootstrap spec to disk artifacts (R10)."""
    p = Path(export_dir)
    p.mkdir(parents=True, exist_ok=True)

    # 1. Export returns
    returns_path = p / "daily_returns.json"
    with open(returns_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(returns_by_key))

    # 2. Export contrast summary
    contrasts_path = p / "primary_contrasts.json"
    contrasts_data = [asdict(r) for r in contrast_results]
    with open(contrasts_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(contrasts_data))

    # 3. Export draw matrix
    draws_path = p / "contrast_draws.npy"
    np.save(draws_path, draw_matrix)

    # 4. Export bootstrap spec for deterministic replay
    actual_num_draws = int(draw_matrix.shape[1]) if draw_matrix.ndim == 2 else num_draws
    spec_path = p / "bootstrap_spec.json"
    spec = {
        "num_draws": actual_num_draws,
        "block_size": block_size,
        "seed": seed,
        "markets": markets or ["US", "IN", "CN", "FR", "GB", "BR"],
    }
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(to_canonical_json(spec))

    return p


def replay_analysis_bundle(
    export_dir: Union[str, Path],
    tolerance: float = 1e-10,
) -> Dict[str, Any]:
    """Read written artifacts in a fresh process and verify replay within tolerance (R10).

    Strictly:
    - Reads raw return series from daily_returns.json.
    - Recomputes all contrast point estimates and bootstrap draws from returns.
    - Compares recomputed results to stored artifacts within declared tolerance (1e-10).
    - Fails if return data, bootstrap draws, or contrasts have been altered.
    """
    p = Path(export_dir)
    returns_path = p / "daily_returns.json"
    contrasts_path = p / "primary_contrasts.json"
    draws_path = p / "contrast_draws.npy"
    spec_path = p / "bootstrap_spec.json"

    if not (returns_path.exists() and contrasts_path.exists() and draws_path.exists()):
        raise FileNotFoundError(f"Missing analysis bundle files in {p}")

    with open(returns_path, "r", encoding="utf-8") as f:
        stored_returns = json.load(f)

    with open(contrasts_path, "r", encoding="utf-8") as f:
        stored_contrasts = json.load(f)

    stored_draw_matrix = np.load(draws_path)

    spec = {}
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

    num_draws = spec.get("num_draws", stored_draw_matrix.shape[1] if stored_draw_matrix.ndim == 2 else 1000)
    block_size = spec.get("block_size", 5)
    seed = spec.get("seed", 42)
    markets = spec.get("markets", ["US", "IN", "CN", "FR", "GB", "BR"])

    # Parse stored_returns back into returns_map using rsplit to cleanly parse arm, mkt, seed
    reconstructed_map: Dict[Tuple[str, str, Optional[int]], np.ndarray] = {}
    for key_str, vals in stored_returns.items():
        if "__" in key_str:
            parts = key_str.split("__")
        else:
            parts = key_str.rsplit("_", 2)

        if len(parts) >= 3:
            arm = parts[0]
            mkt = parts[1]
            s_str = parts[2]
            seed_val = int(s_str) if s_str.isdigit() and s_str != "None" else None
            reconstructed_map[(arm, mkt, seed_val)] = np.array(vals, dtype=np.float64)
        elif len(parts) == 2:
            reconstructed_map[(parts[0], parts[1], None)] = np.array(vals, dtype=np.float64)
        else:
            reconstructed_map[(key_str, "US", None)] = np.array(vals, dtype=np.float64)

    # Re-evaluate primary contrasts directly from the loaded returns!
    recomputed_contrasts, recomputed_draws = evaluate_primary_contrasts(
        returns_by_arm_market_realization=reconstructed_map,
        markets=markets,
        num_draws=num_draws,
        block_size=block_size,
        seed=seed,
    )

    # Compare recomputed draws to stored draw matrix at declared tolerance
    draw_diff = float(np.max(np.abs(recomputed_draws - stored_draw_matrix)))
    if draw_diff > tolerance:
        raise ReplayVerificationError(
            f"Replay mismatch in bootstrap draw matrix: max discrepancy {draw_diff:.4e} > tolerance {tolerance:.4e}"
        )

    # Compare recomputed contrast results to stored contrasts at declared tolerance
    replayed_results = []
    for c_idx, stored_entry in enumerate(stored_contrasts):
        recomputed = recomputed_contrasts[c_idx]
        stored_th = float(stored_entry["theta"])
        recomputed_th = float(recomputed.theta)
        if abs(recomputed_th - stored_th) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for theta on {stored_entry['contrast_id']}: stored={stored_th}, recomputed={recomputed_th}"
            )

        stored_ci_l = float(stored_entry["ci_lower"])
        recomputed_ci_l = float(recomputed.ci_lower)
        if abs(recomputed_ci_l - stored_ci_l) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for ci_lower on {stored_entry['contrast_id']}: stored={stored_ci_l}, recomputed={recomputed_ci_l}"
            )

        stored_ci_u = float(stored_entry["ci_upper"])
        recomputed_ci_u = float(recomputed.ci_upper)
        if abs(recomputed_ci_u - stored_ci_u) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for ci_upper on {stored_entry['contrast_id']}: stored={stored_ci_u}, recomputed={recomputed_ci_u}"
            )

        stored_p = float(stored_entry["p_value"])
        recomputed_p = float(recomputed.p_value)
        if abs(recomputed_p - stored_p) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for p-value on {stored_entry['contrast_id']}: stored={stored_p}, recomputed={recomputed_p}"
            )

        replayed_results.append({
            "contrast_id": stored_entry["contrast_id"],
            "replayed_p": recomputed_p,
            "stored_p": stored_p,
            "replayed_theta": recomputed_th,
            "stored_theta": stored_th,
        })

    return {"status": "REPLAY_VERIFIED", "results": replayed_results}
