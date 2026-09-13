"""Statistical inference, primary contrasts, and replay verification (v2).

Acceptance criteria addressed:
- A28: Point-estimate metric parity against numerical reference fixtures.
- A30: Replay verification on stored daily returns with 1e-10 tolerance.
- A31: Full precision in release outputs, independent verification from raw returns.
- C2: Calendar-week block bootstrap (year-stratified, moving non-wrapping blocks, native-market masks).
- Bounded-memory execution via weekly sufficient statistics (O(weeks * series) instead of O(draws * days * series)).
- Reject absent required policy series (no zero-padding or imputing).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from memory_study_v2.contracts import to_canonical_json


class ReplayVerificationError(Exception):
    """Raised when recomputed analysis results fail to match saved artifacts within tolerance."""
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
    status: str = "COMPLETED"


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


def compute_bootstrap_p_and_ci(
    theta: float,
    theta_draws: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute centered linear quantile bootstrap CI and two-sided p-value (Section 9.2)."""
    B = len(theta_draws)
    if B == 0:
        return theta, theta, 1.0
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
    p_sorted = np.array(p_values, dtype=np.float64)[sorted_indices]

    adj_sorted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for i in range(m):
        rank = i + 1
        adj_p = min(1.0, float(p_sorted[i] * (m - rank + 1)))
        running_max = max(running_max, adj_p)
        adj_sorted[i] = running_max

    adj_p_values = np.zeros(m, dtype=np.float64)
    adj_p_values[sorted_indices] = adj_sorted
    return [float(p) for p in adj_p_values]


def compute_sharpe_ratio(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio in float64 with variance guard (Section 9.2)."""
    if len(returns) == 0:
        return 0.0
    r = np.asarray(returns, dtype=np.float64)
    # Mask out non-finite (holiday/unobserved) sessions
    if np.any(~np.isfinite(r)):
        r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return 0.0
    mean_r = np.mean(r)
    var_r = np.var(r, ddof=0)

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


def build_calendar_weeks_mapping(
    session_dates: Optional[List[str]],
    total_sessions: int,
) -> Tuple[List[int], List[int], List[Tuple[int, Any]]]:
    """Assign sessions to year and week indices, with Monday-based non-wrapping within-year partitions (C2).

    Uses Monday anchor date within each calendar year to avoid ISO week collisions (e.g. 2024-01-02 vs 2024-12-30).
    Rejects malformed production dates.
    """
    if session_dates is not None:
        if len(session_dates) != total_sessions:
            raise ValueError(f"session_dates length ({len(session_dates)}) does not match total_sessions ({total_sessions})")
        years = []
        monday_keys = []
        for s in session_dates:
            s_str = str(s)[:10]
            try:
                dt = datetime.strptime(s_str, "%Y-%m-%d")
            except Exception as e:
                raise ValueError(f"Malformed production session date '{s}': expected YYYY-MM-DD ({e})")
            y = dt.year
            mon = dt - timedelta(days=dt.weekday())
            years.append(y)
            # Monday key anchored within calendar year avoids collision
            monday_keys.append((y, mon.strftime("%Y-%m-%d")))
    else:
        # 5-day trading week chunks for synthetic tests
        years = [(t // 5) // 52 for t in range(total_sessions)]
        monday_keys = [(years[t], f"synthetic_w_{(t // 5) % 52:02d}") for t in range(total_sessions)]

    # Map unique (year, week_anchor) to global week index
    unique_yw = []
    seen = set()
    for ym in monday_keys:
        if ym not in seen:
            seen.add(ym)
            unique_yw.append(ym)
    unique_yw.sort()
    yw_to_global = {ym: idx for idx, ym in enumerate(unique_yw)}

    session_global_weeks = [yw_to_global[ym] for ym in monday_keys]
    return session_global_weeks, years, unique_yw


def sample_calendar_week_blocks(
    unique_yw: List[Tuple[int, int]],
    num_draws: int = 10000,
    block_length_weeks: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Sample year-stratified, moving non-wrapping calendar-week blocks (C2, Section 9.2)."""
    if not unique_yw or num_draws <= 0:
        return np.zeros((num_draws, 0), dtype=np.int64)

    # Group global week indices by year
    weeks_by_year: Dict[int, List[int]] = {}
    for g_idx, (y, w) in enumerate(unique_yw):
        weeks_by_year.setdefault(y, []).append(g_idx)

    rng = np.random.default_rng(np.random.PCG64(np.random.SeedSequence([seed, block_length_weeks])))
    total_weeks = len(unique_yw)
    draw_matrix = np.zeros((num_draws, total_weeks), dtype=np.int64)

    for d in range(num_draws):
        sampled_weeks_d = []
        for y, y_weeks in sorted(weeks_by_year.items()):
            n_y = len(y_weeks)
            if n_y <= block_length_weeks:
                # If year has fewer weeks than block size, sample with replacement from available weeks
                chosen_starts = rng.integers(0, n_y, size=n_y)
                sampled_weeks_d.extend([y_weeks[s] for s in chosen_starts])
            else:
                # Non-wrapping block starts: s in {0, ..., n_y - block_length_weeks}
                max_start = n_y - block_length_weeks
                n_blocks = int(math.ceil(n_y / block_length_weeks))
                starts = rng.integers(0, max_start + 1, size=n_blocks)
                y_sampled = []
                for s in starts:
                    y_sampled.extend(y_weeks[s : s + block_length_weeks])
                sampled_weeks_d.extend(y_sampled[:n_y])
        draw_matrix[d] = np.array(sampled_weeks_d, dtype=np.int64)

    return draw_matrix


def generate_stratified_week_blocks(
    year_to_weeks: Dict[int, List[int]],
    block_length: int = 4,
    num_draws: int = 10000,
    seed_sequence: int = 42,
) -> List[Dict[int, List[int]]]:
    """Generate synchronized year-stratified sampled week index arrays using PCG64 (A30 compatibility)."""
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


def compute_weekly_sufficient_statistics(
    returns_matrix: np.ndarray,
    session_global_weeks: List[int],
    total_weeks: int,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (N, Sum, SumSq) per week and per series for bounded-memory bootstrap (C2).

    Applies per-series holiday/open masks: non-observed holiday sessions are excluded
    and do not increment N or accumulate 0.0 into sums.
    """
    K = returns_matrix.shape[0]
    W = total_weeks

    week_counts = np.zeros((W, K), dtype=np.float64)
    week_sums = np.zeros((W, K), dtype=np.float64)
    week_sumsq = np.zeros((W, K), dtype=np.float64)

    if valid_mask is None:
        mask = np.isfinite(returns_matrix)
    else:
        mask = valid_mask & np.isfinite(returns_matrix)

    # Accumulate into weekly bins with holiday/open mask
    for t_idx, w_idx in enumerate(session_global_weeks):
        m_t = mask[:, t_idx]
        if np.any(m_t):
            r_t = np.where(m_t, returns_matrix[:, t_idx], 0.0)
            week_counts[w_idx, :] += m_t.astype(np.float64)
            week_sums[w_idx, :] += r_t
            week_sumsq[w_idx, :] += (r_t ** 2) * m_t

    return week_counts, week_sums, week_sumsq


def evaluate_primary_contrasts(
    returns_by_arm_market_realization: Dict[Tuple[str, str, Optional[int]], np.ndarray],
    markets: List[str],
    session_dates: Optional[List[str]] = None,
    draw_week_indices: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    num_draws: int = 1000,
    block_length_weeks: int = 4,
    seed: int = 42,
    allow_reduced_arms: bool = False,
) -> Tuple[List[ContrastResult], np.ndarray, np.ndarray]:
    """Calculate the 8 primary Sharpe contrasts using year-stratified calendar block bootstrap (C2, R10).

    Uses weekly sufficient statistics to bound memory allocation (O(weeks * series) instead of O(draws * days * series)).
    Full precision float64 arithmetic throughout.
    """
    keys = sorted(list(returns_by_arm_market_realization.keys()))
    if not keys:
        return [], np.zeros((len(PRIMARY_CONTRASTS), 0), dtype=np.float64), np.zeros((0, 0), dtype=np.int64)

    arms_in_data = sorted(list({k[0] for k in keys}))

    # Verify required policy series exist (no silent zero-imputation) (C2)
    missing_required = []
    for cid, cand, comp in PRIMARY_CONTRASTS:
        if cand not in arms_in_data:
            missing_required.append(f"{cid}: candidate '{cand}' missing")
        if comp not in arms_in_data:
            missing_required.append(f"{cid}: comparator '{comp}' missing")
    if missing_required and not allow_reduced_arms:
        raise ValueError(f"Required policy arms missing from returns data: {missing_required}")

    # When not allowing reduced arms, verify complete market coverage for all candidate and comparator arms
    if not allow_reduced_arms:
        for cid, cand, comp in PRIMARY_CONTRASTS:
            for arm in (cand, comp):
                arm_mkts = {k[1] for k in keys if k[0] == arm}
                missing_mkts = [m for m in markets if m not in arm_mkts]
                if missing_mkts:
                    raise ValueError(f"Required market path missing for arm '{arm}' in contrast {cid}: missing {missing_mkts}")

    # Build matrix of return series
    lengths = [len(returns_by_arm_market_realization[k]) for k in keys]
    T = max(lengths) if lengths else 0
    if min(lengths) != T:
        raise ValueError(f"Inconsistent return series lengths across keys: min={min(lengths)}, max={T}")

    ret_matrix = np.array([returns_by_arm_market_realization[k] for k in keys], dtype=np.float64)

    # 1. Point estimates: compute headline Sharpe per arm on original returns
    arm_market_sharpes: Dict[str, Dict[str, List[float]]] = {}
    for idx, (arm, mkt, seed_val) in enumerate(keys):
        sr = compute_sharpe_ratio(ret_matrix[idx])
        arm_market_sharpes.setdefault(arm, {}).setdefault(mkt, []).append(sr)

    headline_sharpes: Dict[str, float] = {}
    for arm in arms_in_data:
        mkt_means = []
        for mkt in markets:
            srs = arm_market_sharpes.get(arm, {}).get(mkt, [])
            if srs:
                mkt_means.append(float(np.mean(srs)))
        headline_sharpes[arm] = float(np.mean(mkt_means)) if mkt_means else 0.0

    thetas: Dict[str, float] = {}
    for cid, cand, comp in PRIMARY_CONTRASTS:
        cand_sr = headline_sharpes.get(cand, 0.0)
        comp_sr = headline_sharpes.get(comp, 0.0)
        thetas[cid] = float(cand_sr - comp_sr)

    # 2. Assign sessions to calendar weeks
    session_weeks, session_years, unique_yw = build_calendar_weeks_mapping(session_dates, T)
    total_weeks = len(unique_yw)

    # 3. Sample calendar-week blocks if not supplied
    if draw_week_indices is None:
        draw_week_indices = sample_calendar_week_blocks(
            unique_yw=unique_yw,
            num_draws=num_draws,
            block_length_weeks=block_length_weeks,
            seed=seed,
        )
    actual_num_draws = len(draw_week_indices)

    # 4. Weekly Sufficient Statistics (Bounded Memory) (C2)
    # week_counts, week_sums, week_sumsq shape: (W, K)
    w_counts, w_sums, w_sumsq = compute_weekly_sufficient_statistics(ret_matrix, session_weeks, total_weeks, valid_mask=valid_mask)

    # 5. Build draw frequency matrix F shape: (D, W)
    # F[d, w] is count of times week w was sampled in draw d
    F = np.zeros((actual_num_draws, total_weeks), dtype=np.float64)
    for d in range(actual_num_draws):
        np.add.at(F[d], draw_week_indices[d], 1.0)

    # Vectorized matrix product across all draws and series: (D, W) @ (W, K) -> (D, K)
    draw_counts = F @ w_counts
    draw_sums = F @ w_sums
    draw_sumsq = F @ w_sumsq

    safe_counts = np.maximum(1.0, draw_counts)
    draw_means = draw_sums / safe_counts
    draw_vars = np.maximum(0.0, (draw_sumsq / safe_counts) - (draw_means ** 2))
    draw_vols = np.sqrt(draw_vars) * math.sqrt(252.0)
    safe_vols = np.where(draw_vols > 1e-8, draw_vols, 1.0)
    draw_srs = np.where(draw_vols > 1e-8, (draw_means * 252.0) / safe_vols, 0.0)  # Shape: (D, K)

    # 6. Aggregate draw Sharpes to arm headlines
    arm_headline_draws: Dict[str, np.ndarray] = {}
    for arm in arms_in_data:
        mkt_draw_list = []
        for mkt in markets:
            matching_indices = [
                idx for idx, k in enumerate(keys)
                if k[0] == arm and k[1] == mkt
            ]
            if matching_indices:
                mkt_mean_draw = np.mean(draw_srs[:, matching_indices], axis=1)  # Shape: (D,)
                mkt_draw_list.append(mkt_mean_draw)
        arm_headline_draws[arm] = np.mean(mkt_draw_list, axis=0) if mkt_draw_list else np.zeros(actual_num_draws, dtype=np.float64)

    # 7. Compute primary contrast draws and results
    draw_matrix = np.zeros((len(PRIMARY_CONTRASTS), actual_num_draws), dtype=np.float64)
    contrast_results: List[ContrastResult] = []
    raw_p_values: List[float] = []

    for c_idx, (cid, cand, comp) in enumerate(PRIMARY_CONTRASTS):
        if cand in arms_in_data and comp in arms_in_data:
            cand_draws = arm_headline_draws[cand]
            comp_draws = arm_headline_draws[comp]
            c_draws = cand_draws - comp_draws
            draw_matrix[c_idx] = c_draws

            th = thetas[cid]
            ci_l, ci_u, p_val = compute_bootstrap_p_and_ci(th, c_draws)
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
                num_draws=actual_num_draws,
                status="COMPLETED",
            ))
        else:
            raw_p_values.append(1.0)
            contrast_results.append(ContrastResult(
                contrast_id=cid,
                candidate=cand,
                comparator=comp,
                theta=0.0,
                ci_lower=0.0,
                ci_upper=0.0,
                p_value=1.0,
                p_value_holm=1.0,
                num_draws=actual_num_draws,
                status="NOT_RUN",
            ))

    holm_p = apply_step_down_holm_bonferroni(raw_p_values)
    for i, res in enumerate(contrast_results):
        res.p_value_holm = float(holm_p[i])

    return contrast_results, draw_matrix, draw_week_indices


def export_analysis_bundle(
    returns_by_key: Dict[str, List[float]],
    contrast_results: List[ContrastResult],
    draw_matrix: np.ndarray,
    draw_week_indices: np.ndarray,
    export_dir: Union[str, Path],
    session_dates: Optional[List[str]] = None,
    markets: Optional[List[str]] = None,
    num_draws: int = 1000,
    block_length_weeks: int = 4,
    seed: int = 42,
    allow_reduced_arms: bool = False,
) -> Path:
    """Export return series, contrast summary, draw matrix, and sampled week blocks to disk artifacts (R10, C2)."""
    p = Path(export_dir)
    p.mkdir(parents=True, exist_ok=True)

    # 1. Export returns and session dates
    returns_payload = {
        "session_dates": session_dates if session_dates is not None else [],
        "returns": returns_by_key,
    }
    with open(p / "daily_returns.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(returns_payload))

    # 2. Export contrast summary
    contrasts_data = [asdict(r) for r in contrast_results]
    with open(p / "primary_contrasts.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(contrasts_data))

    # 3. Export draw matrix and sampled weeks
    np.save(p / "contrast_draws.npy", draw_matrix)
    np.save(p / "sampled_weeks.npy", draw_week_indices)

    # 4. Export bootstrap spec for deterministic replay
    actual_num_draws = int(draw_matrix.shape[1]) if draw_matrix.ndim == 2 else num_draws
    has_not_run = any(getattr(c, "status", "") == "NOT_RUN" for c in contrast_results)
    spec = {
        "num_draws": actual_num_draws,
        "block_length_weeks": block_length_weeks,
        "seed": seed,
        "markets": markets or ["US", "IN", "CN", "FR", "GB", "BR"],
        "allow_reduced_arms": bool(allow_reduced_arms or has_not_run),
    }
    with open(p / "bootstrap_spec.json", "w", encoding="utf-8") as f:
        f.write(to_canonical_json(spec))

    return p


def replay_analysis_bundle(
    export_dir: Union[str, Path],
    tolerance: float = 1e-10,
    allow_reduced_arms: bool = False,
) -> Dict[str, Any]:
    """Read written artifacts and independently verify replay from return data (R10, C2).

    Validates:
    1. Returns read from daily_returns.json.
    2. Persisted sampled calendar weeks read from sampled_weeks.npy.
    3. Recomputation of point estimates, draw distributions, CIs, and Holm p-values.
    4. Exact agreement of contrast identities (cand - comp == theta).
    5. Bitwise agreement within tolerance (1e-10) against contrast_draws.npy and primary_contrasts.json.
    6. Non-missing, finite values across all outputs.
    """
    p = Path(export_dir)
    returns_path = p / "daily_returns.json"
    contrasts_path = p / "primary_contrasts.json"
    draws_path = p / "contrast_draws.npy"
    weeks_path = p / "sampled_weeks.npy"
    spec_path = p / "bootstrap_spec.json"

    if not (returns_path.exists() and contrasts_path.exists() and draws_path.exists() and weeks_path.exists()):
        raise FileNotFoundError(f"Missing analysis bundle files in {p}")

    with open(returns_path, "r", encoding="utf-8") as f:
        returns_payload = json.load(f)

    if isinstance(returns_payload, dict) and "returns" in returns_payload:
        stored_returns = returns_payload["returns"]
        session_dates = returns_payload.get("session_dates", None)
    else:
        stored_returns = returns_payload
        session_dates = None

    with open(contrasts_path, "r", encoding="utf-8") as f:
        stored_contrasts = json.load(f)

    stored_draw_matrix = np.load(draws_path)
    stored_weeks_matrix = np.load(weeks_path)

    spec = {}
    if spec_path.exists():
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

    num_draws = spec.get("num_draws", stored_draw_matrix.shape[1] if stored_draw_matrix.ndim == 2 else 1000)
    block_length = spec.get("block_length_weeks", 4)
    seed = spec.get("seed", 42)
    markets = spec.get("markets", ["US", "IN", "CN", "FR", "GB", "BR"])
    allow_reduced_arms = spec.get("allow_reduced_arms", allow_reduced_arms)

    # Reconstruct returns map
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

    # Re-evaluate primary contrasts directly from returns and persisted sampled weeks
    recomputed_contrasts, recomputed_draws, _ = evaluate_primary_contrasts(
        returns_by_arm_market_realization=reconstructed_map,
        markets=markets,
        session_dates=session_dates if session_dates else None,
        draw_week_indices=stored_weeks_matrix,
        num_draws=num_draws,
        block_length_weeks=block_length,
        seed=seed,
        allow_reduced_arms=allow_reduced_arms,
    )

    # 1. Structural count and shape checks
    if len(stored_contrasts) != len(recomputed_contrasts):
        raise ReplayVerificationError(
            f"Stored contrast count ({len(stored_contrasts)}) != recomputed count ({len(recomputed_contrasts)})"
        )
    if stored_draw_matrix.shape != recomputed_draws.shape:
        raise ReplayVerificationError(
            f"Stored draw matrix shape {stored_draw_matrix.shape} != recomputed shape {recomputed_draws.shape}"
        )

    # 2. Strict finite checks on all stored and recomputed matrices (reject NaNs/Infs) (C2)
    if not np.all(np.isfinite(stored_draw_matrix)):
        raise ReplayVerificationError("Stored draw matrix contains non-finite (NaN or Inf) values")
    if not np.all(np.isfinite(recomputed_draws)):
        raise ReplayVerificationError("Recomputed draw matrix contains non-finite (NaN or Inf) values")
    if not np.all(np.isfinite(stored_weeks_matrix)):
        raise ReplayVerificationError("Stored sampled weeks matrix contains non-finite values")

    # 3. Compare recomputed draws to stored draw matrix within tolerance
    draw_diff = float(np.max(np.abs(recomputed_draws - stored_draw_matrix)))
    if draw_diff > tolerance:
        raise ReplayVerificationError(
            f"Replay mismatch in bootstrap draw matrix: max discrepancy {draw_diff:.4e} > tolerance {tolerance:.4e}"
        )

    # 4. Compare recomputed contrast results to stored contrasts
    replayed_results = []
    for c_idx, stored_entry in enumerate(stored_contrasts):
        recomputed = recomputed_contrasts[c_idx]

        # Explicit contrast ID, candidate, comparator, and status check (C2)
        stored_cid = stored_entry.get("contrast_id")
        if stored_cid != recomputed.contrast_id:
            raise ReplayVerificationError(
                f"Contrast ID mismatch at index {c_idx}: stored '{stored_cid}' != recomputed '{recomputed.contrast_id}'"
            )
        if stored_entry.get("candidate") != recomputed.candidate:
            raise ReplayVerificationError(
                f"Candidate mismatch on {stored_cid}: stored '{stored_entry.get('candidate')}' != recomputed '{recomputed.candidate}'"
            )
        if stored_entry.get("comparator") != recomputed.comparator:
            raise ReplayVerificationError(
                f"Comparator mismatch on {stored_cid}: stored '{stored_entry.get('comparator')}' != recomputed '{recomputed.comparator}'"
            )
        if stored_entry.get("status") != recomputed.status:
            raise ReplayVerificationError(
                f"Status mismatch on {stored_cid}: stored '{stored_entry.get('status')}' != recomputed '{recomputed.status}'"
            )

        stored_th = float(stored_entry["theta"])
        recomputed_th = float(recomputed.theta)
        stored_ci_l = float(stored_entry["ci_lower"])
        recomputed_ci_l = float(recomputed.ci_lower)
        stored_ci_u = float(stored_entry["ci_upper"])
        recomputed_ci_u = float(recomputed.ci_upper)
        stored_p = float(stored_entry["p_value"])
        recomputed_p = float(recomputed.p_value)
        stored_holm_p = float(stored_entry["p_value_holm"])
        recomputed_holm_p = float(recomputed.p_value_holm)

        # Finite checks on every single metric (stored and recomputed)
        for name, val in [
            ("theta", stored_th), ("ci_lower", stored_ci_l), ("ci_upper", stored_ci_u),
            ("p_value", stored_p), ("p_value_holm", stored_holm_p),
        ]:
            if not np.isfinite(val):
                raise ReplayVerificationError(f"Non-finite stored {name} on {stored_cid}: {val}")

        for name, val in [
            ("theta", recomputed_th), ("ci_lower", recomputed_ci_l), ("ci_upper", recomputed_ci_u),
            ("p_value", recomputed_p), ("p_value_holm", recomputed_holm_p),
        ]:
            if not np.isfinite(val):
                raise ReplayVerificationError(f"Non-finite recomputed {name} on {stored_cid}: {val}")

        # Strict tolerance checks
        if abs(recomputed_th - stored_th) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for theta on {stored_cid}: stored={stored_th}, recomputed={recomputed_th}"
            )
        if abs(recomputed_ci_l - stored_ci_l) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for ci_lower on {stored_cid}: stored={stored_ci_l}, recomputed={recomputed_ci_l}"
            )
        if abs(recomputed_ci_u - stored_ci_u) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for ci_upper on {stored_cid}: stored={stored_ci_u}, recomputed={recomputed_ci_u}"
            )
        if abs(recomputed_p - stored_p) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for p-value on {stored_cid}: stored={stored_p}, recomputed={recomputed_p}"
            )
        if abs(recomputed_holm_p - stored_holm_p) > tolerance:
            raise ReplayVerificationError(
                f"Replay mismatch for p_value_holm on {stored_cid}: stored={stored_holm_p}, recomputed={recomputed_holm_p}"
            )

        replayed_results.append({
            "contrast_id": stored_cid,
            "replayed_p": recomputed_p,
            "stored_p": stored_p,
            "replayed_p_holm": recomputed_holm_p,
            "stored_p_holm": stored_holm_p,
            "replayed_theta": recomputed_th,
            "stored_theta": stored_th,
        })

    return {
        "status": "REPLAY_VERIFIED",
        "tolerance": tolerance,
        "contrasts_verified": len(stored_contrasts),
        "draws_shape": list(stored_draw_matrix.shape),
        "results": replayed_results,
    }
