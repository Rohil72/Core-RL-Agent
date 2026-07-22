"""Analyze latent state transitions over time per-ticker.

Computes transition matrices for 1-step, 5-step, 20-step, state persistence, and opportunity dynamics.

Inputs: latent files with columns: 'timestamp', ticker or 'symbol', 'cluster_id' (state), and target columns.

Outputs JSON diagnostics in reports/final_direction/. Does not run experiments by default.

Fixes applied:
  Bug#1 - Added upfront dataset diagnostics (rows, tickers, clusters) to catch "wrong dataset" early
  Bug#2 - Deterministic target column resolution (exact → true_ → endswith, never candidates[0])
  Bug#3 - Removed dead split/train/test code; analyze_transitions operates on full dataset
  Bug#4 - Fixed transition probability normalization (zero-init + explicit mask, no garbage memory)
  Bug#5 - Step-size validation: only count transitions where timestamp gap ≈ step × median_gap
  Bug#6 - Opportunity target now measured at state B (the destination), not state A
  Bug#7 - Added state persistence duration (mean/median/max consecutive run length)
  Bug#8 - Added per-ticker row count diagnostics before analysis
  Bug#9 - Added cluster population diagnostics before analysis
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import argparse
import logging
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('reports/final_direction')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEP_SIZES = (1, 5, 20)
TARGET_COLS = ['future_max_return_63', 'future_min_return_63', 'event_upside_before_drawdown_126']

# Fraction of median gap a transition's actual elapsed time may deviate before being excluded.
# E.g. 0.5 means the gap must be within [0.5x, 1.5x] the expected gap for that step size.
STEP_TOLERANCE = 0.5


def decode_byte_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert byte array columns back to strings."""
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        if sample is not None and isinstance(sample, (list, np.ndarray)):
            try:
                if all(isinstance(x, (int, np.integer)) and 0 <= int(x) <= 255 for x in sample):
                    df[col] = df[col].apply(lambda x: bytes([int(v) for v in x]).decode('utf-8'))
                    logger.info("Decoded column '%s' from byte array to string", col)
            except (ValueError, UnicodeDecodeError, TypeError):
                pass
    return df


def _resolve_target_col(all_cols: List[str], base: str) -> Optional[str]:
    """Deterministically resolve a target column.

    Bug#2 fix: priority order: exact match → true_ prefix → endswith fallback.
    """
    if base in all_cols:
        return base
    true_name = f'true_{base}'
    if true_name in all_cols:
        return true_name
    endswith_matches = sorted(c for c in all_cols if c.endswith(base) and c != base)
    if endswith_matches:
        logger.warning("Target '%s' resolved via endswith fallback to '%s'", base, endswith_matches[0])
        return endswith_matches[0]
    return None


def _gather_time_series(full: pd.DataFrame, ticker_col: str, ts_col: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for ticker, group in full.groupby(ticker_col):
        g = group.sort_values(ts_col).reset_index(drop=True)
        out[str(ticker)] = g
    return out


def _compute_median_gap(series: Dict[str, pd.DataFrame], ts_col: str) -> float:
    """Compute the median consecutive-row timestamp gap in seconds across all tickers."""
    gaps = []
    for df in series.values():
        ts = pd.to_datetime(df[ts_col]).astype('int64') // 10 ** 9
        diffs = np.diff(ts.to_numpy())
        if len(diffs):
            gaps.extend(diffs.tolist())
    return float(np.median(gaps)) if gaps else 1.0


def _state_persistence_durations(states: List[int]) -> Dict[int, List[int]]:
    """Return lists of consecutive run lengths per state."""
    durations: Dict[int, List[int]] = defaultdict(list)
    if not states:
        return durations
    current = states[0]
    run = 1
    for s in states[1:]:
        if s == current:
            run += 1
        else:
            durations[current].append(run)
            current = s
            run = 1
    durations[current].append(run)
    return durations


def _build_transition_probability_matrix(
    counts: Dict[Tuple[int, int], int], all_states: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Build count matrix and probability matrix.

    Bug#4 fix: zero-initialize prob matrix and only fill rows with nonzero sums.
    np.divide with where= leaves uninitialized memory in skipped slots; avoid it.
    """
    idx = {s: i for i, s in enumerate(all_states)}
    n = len(all_states)
    mat = np.zeros((n, n), dtype=int)
    for (a, b), c in counts.items():
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = c
    row_sums = mat.sum(axis=1)  # shape (n,)
    prob = np.zeros((n, n), dtype=float)  # Bug#4: explicit zero init
    nonzero_rows = row_sums > 0
    prob[nonzero_rows] = mat[nonzero_rows] / row_sums[nonzero_rows, np.newaxis]
    return mat, prob


def analyze_transitions(
    full: pd.DataFrame,
    ticker_col: str = 'ticker',
    ts_col: str = 'timestamp',
    available_targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Core transition analysis. Operates on the full dataset (no train/test split needed).

    Bug#3 fix: split logic removed — transitions are a property of the full history.
    """
    if available_targets is None:
        available_targets = []

    series = _gather_time_series(full, ticker_col, ts_col)

    # Bug#8: per-ticker row count diagnostics
    ticker_row_counts = {t: len(df) for t, df in series.items()}
    row_counts_arr = np.array(list(ticker_row_counts.values()))
    logger.info(
        'Ticker diagnostics: n_tickers=%d | rows: min=%d median=%.0f max=%d',
        len(series),
        int(row_counts_arr.min()) if len(row_counts_arr) else 0,
        float(np.median(row_counts_arr)) if len(row_counts_arr) else 0,
        int(row_counts_arr.max()) if len(row_counts_arr) else 0,
    )
    sparse_tickers = [t for t, n in ticker_row_counts.items() if n < max(STEP_SIZES) + 1]
    if sparse_tickers:
        logger.warning(
            '%d tickers have fewer rows than max step size (%d) — step_%d transitions will be empty for them',
            len(sparse_tickers), max(STEP_SIZES), max(STEP_SIZES),
        )

    # Bug#5: compute median gap to validate step sizes
    median_gap_s = _compute_median_gap(series, ts_col)
    logger.info('Median consecutive timestamp gap: %.1f seconds (%.2f days)', median_gap_s, median_gap_s / 86400)

    all_states = sorted(full['cluster_id'].astype(int).unique())

    results: Dict[str, Any] = {
        'transition_matrices': {},
        'persistence': {},
        'persistence_duration': {},
        'opportunity_transitions': {},
    }

    # Bug#7: state persistence duration across all tickers
    all_durations: Dict[int, List[int]] = defaultdict(list)
    for df in series.values():
        states_seq = df['cluster_id'].astype(int).tolist()
        for state, runs in _state_persistence_durations(states_seq).items():
            all_durations[state].extend(runs)

    persistence_duration_summary = {}
    for s in all_states:
        runs = all_durations.get(s, [])
        if runs:
            persistence_duration_summary[str(s)] = {
                'mean_rows': float(np.mean(runs)),
                'median_rows': float(np.median(runs)),
                'max_rows': int(np.max(runs)),
                'count': len(runs),
            }
        else:
            persistence_duration_summary[str(s)] = {'mean_rows': float('nan'), 'median_rows': float('nan'), 'max_rows': 0, 'count': 0}
    results['persistence_duration'] = persistence_duration_summary

    for step in STEP_SIZES:
        expected_gap_s = step * median_gap_s
        gap_lo = expected_gap_s * (1 - STEP_TOLERANCE)
        gap_hi = expected_gap_s * (1 + STEP_TOLERANCE)

        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        skipped_gap_violations = 0
        opportunity_stats: Dict[Tuple[int, int, str], List[float]] = defaultdict(list)

        for ticker, df in series.items():
            states = df['cluster_id'].astype(int).tolist()
            ts_arr = pd.to_datetime(df[ts_col]).astype('int64').to_numpy() // 10 ** 9

            for i in range(len(states) - step):
                # Bug#5 fix: validate actual timestamp gap before counting as a step transition
                actual_gap = float(ts_arr[i + step] - ts_arr[i])
                if not (gap_lo <= actual_gap <= gap_hi):
                    skipped_gap_violations += 1
                    continue

                a = states[i]
                b = states[i + step]
                counts[(a, b)] += 1

                # Bug#6 fix: opportunity target measured at state B (destination), not state A
                for tgt in available_targets:
                    if tgt in df.columns:
                        val = df[tgt].iloc[i + step]  # ← at destination state
                        if pd.notna(val):
                            opportunity_stats[(a, b, tgt)].append(float(val))

        if skipped_gap_violations:
            logger.warning(
                'step_%d: skipped %d transitions due to timestamp gap violations '
                '(expected %.0f–%.0f s, tolerance=%.0f%%)',
                step, skipped_gap_violations, gap_lo, gap_hi, STEP_TOLERANCE * 100,
            )

        mat, prob = _build_transition_probability_matrix(counts, all_states)

        results['transition_matrices'][f'step_{step}'] = {
            'states': all_states,
            'counts': mat.tolist(),
            'probabilities': prob.tolist(),
            'total_transitions': int(mat.sum()),
        }

        # self-transition probability (persistence)
        idx_map = {s: i for i, s in enumerate(all_states)}
        self_prob = {str(s): float(prob[idx_map[s], idx_map[s]]) for s in all_states}
        results['persistence'][f'step_{step}'] = self_prob

        # opportunity transitions
        opp_summary: Dict[str, Dict[str, Any]] = {}
        for (a, b, tgt), vals in opportunity_stats.items():
            key = f'{a}->{b}'
            if key not in opp_summary:
                opp_summary[key] = {}
            opp_summary[key][tgt] = {
                'count': len(vals),
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
            }
        results['opportunity_transitions'][f'step_{step}'] = opp_summary

    return results


def _to_native(o: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(o, dict):
        return {_to_native(k): _to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_native(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return _to_native(o.tolist())
    return o


def main(paths: List[Path]) -> None:
    dfs = []
    for p in paths:
        try:
            df = read_dataframe(str(p))
            if df is not None and not df.empty:
                df = decode_byte_columns(df)
                latent_col = None
                for c in df.columns:
                    sample = df[c].iloc[0] if len(df) > 0 else None
                    if isinstance(sample, (list, np.ndarray)):
                        latent_col = c
                        break
                if latent_col is not None:
                    arr = np.vstack(df[latent_col].apply(lambda x: np.array(x, dtype=float)).values)
                    for i in range(arr.shape[1]):
                        df[f'latent_{i}'] = arr[:, i]
                dfs.append(df)
        except Exception as exc:
            logger.warning('Failed to read %s: %s', p, exc)

    if not dfs:
        logger.error('No latent files found')
        return

    full = pd.concat(dfs, ignore_index=True)

    # Required column checks
    if 'cluster_id' not in full.columns:
        raise RuntimeError('cluster_id column missing — ensure upstream clustering has been saved to these files')
    ticker_col = 'ticker' if 'ticker' in full.columns else ('symbol' if 'symbol' in full.columns else None)
    if ticker_col is None:
        raise RuntimeError('ticker or symbol column missing')
    ts_col = 'timestamp' if 'timestamp' in full.columns else None
    if ts_col is None:
        raise RuntimeError('timestamp column missing')

    # Bug#1 & Bug#9: upfront dataset diagnostics — catch "wrong dataset loaded" immediately
    n_rows = len(full)
    n_tickers = full[ticker_col].nunique()
    n_clusters = full['cluster_id'].nunique()
    logger.info('=== Dataset loaded: rows=%d  tickers=%d  clusters=%d ===', n_rows, n_tickers, n_clusters)
    logger.info('Cluster population:\n%s', full['cluster_id'].value_counts().sort_index().to_string())

    if n_clusters <= 3:
        logger.warning(
            'Only %d clusters detected — expected 8 from Phase 3. '
            'This strongly suggests the wrong latent export was loaded. '
            'Check --latent-glob / --paths arguments before proceeding.',
            n_clusters,
        )

    # Bug#2 fix: deterministic target resolution
    all_cols = list(full.columns)
    tgt_map: Dict[str, str] = {}
    for base in TARGET_COLS:
        resolved = _resolve_target_col(all_cols, base)
        if resolved is not None:
            tgt_map[base] = resolved
            if resolved != base:
                full[base] = full[resolved]
        else:
            logger.warning("Target '%s' not found in any column — will be skipped", base)

    available_targets = list(tgt_map.keys())
    logger.info('Resolved targets: %s', tgt_map)

    # Bug#3 fix: no split needed — analyze full dataset
    results = analyze_transitions(full, ticker_col=ticker_col, ts_col=ts_col, available_targets=available_targets)

    outp = OUTPUT_DIR / 'state_transitions.json'
    with outp.open('w', encoding='utf-8') as fh:
        json.dump(_to_native(results), fh, indent=2)
    logger.info('Wrote state transition analysis to %s', outp)


if __name__ == '__main__':
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument('--latent-glob', type=str, default='reports/latent_analysis/*.parquet')
    ap.add_argument('--paths', type=str, nargs='*')
    args = ap.parse_args()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        files = [Path(p) for p in glob.glob(args.latent_glob)]
    main(files)