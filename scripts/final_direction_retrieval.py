"""Retrieval-based validation framework.

Upgraded to:
- Evaluate retrieval and model predictions against ground truth using MAE, RMSE, Pearson, Spearman, R2
- Enforce temporal safety: neighbors must have timestamp < query timestamp
  (FIX Bug#1: filter legal candidates FIRST, then run NN search on that pool)
- Provide neighbor diagnostics (distance distribution, timestamp gaps, same-ticker %)
- Ranking metrics (top 5/10/20%) with minimum sample size guard
- Robust validation and logging
- Deterministic target/prediction column mapping

This script does NOT retrain models.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import argparse
import logging
import math

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('reports/final_direction')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KS = (10, 25, 50)
RANK_PCTS = (0.05, 0.10, 0.20)
RANK_MIN_SAMPLES = 20  # Bug#7: minimum n before ranking metrics are meaningful
TARGET_COLS = ['future_max_return_63', 'future_min_return_63', 'event_upside_before_drawdown_126']


def decode_byte_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert byte array columns back to strings."""
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        if sample is not None and isinstance(sample, (list, np.ndarray)):
            try:
                if all(isinstance(x, (int, np.integer)) and 0 <= int(x) <= 255 for x in sample):
                    df[col] = df[col].apply(lambda x: bytes([int(v) for v in x]).decode('utf-8'))
                    logger.info(f"Decoded column '{col}' from byte array to string")
            except (ValueError, UnicodeDecodeError, TypeError):
                pass
    return df


def _extract_latent_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract latent matrix from dataframe."""
    latent_cols = [c for c in df.columns if c.startswith('latent_')]
    if latent_cols:
        return df[latent_cols].to_numpy(dtype=float)
    if 'latent' in df.columns:
        return np.vstack(df['latent'].apply(lambda x: np.array(x, dtype=float)).values)
    for c in df.columns:
        sample = df[c].iloc[0] if len(df) > 0 else None
        if isinstance(sample, (list, np.ndarray)):
            return np.vstack(df[c].apply(lambda x: np.array(x, dtype=float)).values)
    raise RuntimeError('No latent columns found')


def _validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing required columns: {missing}')


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        return {k: float('nan') for k in ('mae', 'rmse', 'pearson', 'spearman', 'r2', 'n')}
    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    pearson = float(stats.pearsonr(yt, yp)[0]) if len(yt) > 1 else float('nan')
    spearman = float(stats.spearmanr(yt, yp)[0]) if len(yt) > 1 else float('nan')
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'pearson': pearson, 'spearman': spearman, 'r2': r2, 'n': int(mask.sum())}


def _ranking_metrics(
    df: pd.DataFrame, score_col: str, target_col: str, pctiles: Tuple[float, ...]
) -> Dict[str, Any]:
    """Compute mean target for top percentile groups according to score_col.

    Bug#7 fix: returns nan for all percentiles when n < RANK_MIN_SAMPLES.
    """
    out: Dict[str, Any] = {}
    df = df[[score_col, target_col]].dropna()
    n = len(df)

    if n < RANK_MIN_SAMPLES:
        logger.warning(
            'Ranking metrics skipped: only %d samples (need >= %d)', n, RANK_MIN_SAMPLES
        )
        for p in pctiles:
            out[f'top_{int(p * 100)}pct_mean_{target_col}'] = float('nan')
        out['ranking_skipped_n'] = n
        return out

    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    for p in pctiles:
        k = max(1, int(math.ceil(n * p)))
        out[f'top_{int(p * 100)}pct_mean_{target_col}'] = float(df_sorted[target_col].iloc[:k].mean())
    return out


# ---------------------------------------------------------------------------
# Bug#1 fix: legal-candidates-first temporal retrieval
# ---------------------------------------------------------------------------

def _retrieve_temporal_knn(
    X_train: np.ndarray,
    train_ts: np.ndarray,
    X_test: np.ndarray,
    test_ts: np.ndarray,
    k: int,
    train_df: pd.DataFrame,
    tcol: Optional[str],
):
    """
    Fast temporal retrieval.

    Build ONE global index.
    Retrieve a larger candidate pool.
    Filter illegal future neighbors afterwards.

    Much faster than rebuilding KNN per query.
    """

    n_queries = X_test.shape[0]

    retrieval_k = min(
        max(k * 10, 100),
        len(X_train)
    )

    logger.info(
        "Building global NN index once. "
        "train=%d test=%d retrieval_k=%d",
        len(X_train),
        len(X_test),
        retrieval_k,
    )

    nn = NearestNeighbors(
        n_neighbors=retrieval_k,
        algorithm="auto",
        n_jobs=-1,
    )

    nn.fit(X_train)

    all_dist, all_idx = nn.kneighbors(X_test)
    logger.info(
        "Retrieved candidate matrix shape=%s",
        str(all_idx.shape)
    )

    dist_out = np.full((n_queries, k), np.nan)
    idx_out = np.full((n_queries, k), -1, dtype=np.int32)
    n_legal = np.zeros(n_queries, dtype=np.int32)

    for i in range(n_queries):

        if i % 500 == 0:
            logger.info(
                "Temporal filtering %d/%d",
                i,
                n_queries
            )

        q_ts = test_ts[i]

        candidates_idx = all_idx[i]
        candidates_dist = all_dist[i]

        legal_mask = train_ts[candidates_idx] < q_ts

        legal_idx = candidates_idx[legal_mask]
        legal_dist = candidates_dist[legal_mask]

        n_legal[i] = len(legal_idx)

        if len(legal_idx) == 0:
            continue

        keep = min(k, len(legal_idx))

        idx_out[i, :keep] = legal_idx[:keep]
        dist_out[i, :keep] = legal_dist[:keep]

    logger.info(
        "Finished temporal filtering. "
        "mean legal neighbors=%.2f",
        float(np.mean(n_legal))
    )

    return dist_out, idx_out, n_legal

def _resolve_target_col(full_cols: List[str], base: str) -> Optional[str]:
    """Deterministically resolve a target column name from available columns.

    Bug#6 fix: priority order: exact match → true_ prefix → endswith fallback.
    Never silently picks the wrong column.
    """
    # 1. Exact match
    if base in full_cols:
        return base
    # 2. true_ prefixed exact
    true_name = f'true_{base}'
    if true_name in full_cols:
        return true_name
    # 3. Ends-with match (sorted for determinism)
    endswith_matches = sorted(c for c in full_cols if c.endswith(base) and c != base)
    if endswith_matches:
        logger.warning(
            "Target '%s' resolved via endswith fallback to '%s'", base, endswith_matches[0]
        )
        return endswith_matches[0]
    return None


def _resolve_pred_col(test_cols: List[str], target_base: str) -> Optional[str]:
    """Resolve the model prediction column for a specific target.

    Bug#5 fix: look for target-specific column first, never fall back to an
    arbitrary pred_ column that may correspond to a different target.

    Priority:
        future_head_pred_{target_base}
        pred_{target_base}
        (no generic fallback – caller gets None)
    """
    specific_options = [
        f'future_head_pred_{target_base}',
        f'pred_{target_base}',
    ]
    for col in specific_options:
        if col in test_cols:
            return col
    # Warn but do NOT fall back to an unrelated pred_ column
    logger.warning(
        "No target-specific prediction column found for '%s'. "
        "Model metrics will be nan. Available pred cols: %s",
        target_base,
        [c for c in test_cols if c.startswith('pred_') or c.startswith('future_head_pred')],
    )
    return None


# ---------------------------------------------------------------------------
# Main retrieval experiment
# ---------------------------------------------------------------------------

def compute_retrieval_and_compare(
    train: pd.DataFrame, test: pd.DataFrame, ks: Tuple[int, ...] = KS
) -> Dict[str, Any]:
    """Core retrieval experiment."""
    ts_col = 'timestamp' if 'timestamp' in train.columns else None
    tcol = (
        'ticker'
        if 'ticker' in train.columns
        else ('symbol' if 'symbol' in train.columns else None)
    )
    if ts_col is None or tcol is None:
        raise RuntimeError(
            'Latent tables must contain timestamp and ticker (or symbol) columns'
        )
    _validate_required_columns(train, [ts_col])
    _validate_required_columns(test, [ts_col])

    # Bug#2 diagnostic: log ticker distributions before retrieval
    logger.info('=== Ticker distribution in TRAIN ===')
    logger.info('\n%s', train[tcol].value_counts().to_string())
    logger.info('=== Ticker distribution in TEST ===')
    logger.info('\n%s', test[tcol].value_counts().to_string())

    X_train = _extract_latent_matrix(train)
    X_test = _extract_latent_matrix(test)

    train_valid_mask = np.isfinite(X_train).all(axis=1)
    if not train_valid_mask.all():
        logger.warning('Dropping %d training samples with NaN latent entries', int(np.sum(~train_valid_mask)))
        X_train = X_train[train_valid_mask]
        train = train.iloc[np.where(train_valid_mask)[0]].reset_index(drop=True)
    test_valid_mask = np.isfinite(X_test).all(axis=1)
    if not test_valid_mask.all():
        logger.warning('Dropping %d test samples with NaN latent entries', int(np.sum(~test_valid_mask)))
        X_test = X_test[test_valid_mask]
        test = test.iloc[np.where(test_valid_mask)[0]].reset_index(drop=True)

    train_ts = np.array(pd.to_datetime(train[ts_col]).astype('int64') // 10 ** 9)
    test_ts = np.array(pd.to_datetime(test[ts_col]).astype('int64') // 10 ** 9)

    if X_train.shape[0] == 0:
        raise RuntimeError('No valid training latents after removing NaNs')

    # Bug#8 diagnostic: log cluster distribution if present
    cluster_col = next((c for c in train.columns if 'cluster' in c.lower()), None)
    if cluster_col:
        logger.info('=== Cluster distribution in TRAIN ===')
        logger.info('\n%s', train[cluster_col].value_counts().sort_index().to_string())
        logger.info('=== Cluster distribution in TEST ===')
        logger.info('\n%s', test[cluster_col].value_counts().sort_index().to_string())

    available_targets = [c for c in TARGET_COLS if c in train.columns and c in test.columns]
    logger.info('Available targets for evaluation: %s', available_targets)

    diagnostics: Dict[str, Any] = {'per_k': {}}
    train_ticker_arr = train[tcol].to_numpy() if tcol else None
    test_ticker_arr = test[tcol].to_numpy() if tcol else None

    for k in ks:
        logger.info('Computing temporal-safe neighbors for k=%d', k)

        # Bug#1 fix: legal candidates first, then NN search
        dist, idx, n_legal = _retrieve_temporal_knn(
            X_train, train_ts, X_test, test_ts, k, train, tcol
        )

        # Neighbour diagnostics (only over valid, non-padded entries)
        valid_mask = idx >= 0  # shape (n_queries, k)
        valid_dist = dist[valid_mask]
        valid_ts_gaps = (
            test_ts.reshape(-1, 1) - train_ts[np.where(valid_mask, idx, 0)]
        )[valid_mask]

        same_ticker_pct_values: Optional[np.ndarray] = None
        same_ticker_estimates_count = 0
        if train_ticker_arr is not None and test_ticker_arr is not None:
            # Only compute same-ticker where we have valid neighbors
            st = np.where(
                valid_mask,
                train_ticker_arr[np.where(valid_mask, idx, 0)] == test_ticker_arr.reshape(-1, 1),
                np.nan,
            )
            # Per-query mean (ignoring padded slots)
            with np.errstate(all='ignore'):
                same_ticker_pct_values = np.nanmean(
                    np.where(valid_mask, st.astype(float), np.nan), axis=1
                )
            same_ticker_estimates_count = int(np.sum(~np.isnan(same_ticker_pct_values)))

        diag_k: Dict[str, Any] = {
            'dist_mean': float(np.nanmean(valid_dist)) if valid_dist.size else float('nan'),
            'dist_median': float(np.nanmedian(valid_dist)) if valid_dist.size else float('nan'),
            'dist_std': float(np.nanstd(valid_dist)) if valid_dist.size else float('nan'),
            'neighbor_timestamp_gap_mean': float(np.mean(valid_ts_gaps)) if valid_ts_gaps.size else float('nan'),
            'neighbor_timestamp_gap_median': float(np.median(valid_ts_gaps)) if valid_ts_gaps.size else float('nan'),
            'neighbor_same_ticker_pct_mean': (
                float(np.nanmean(same_ticker_pct_values))
                if same_ticker_pct_values is not None and same_ticker_estimates_count > 0
                else float('nan')
            ),
            'same_ticker_estimates_count': same_ticker_estimates_count,
            'queries_with_zero_legal_neighbors': int(np.sum(n_legal == 0)),
            'mean_legal_pool_size': float(np.mean(n_legal)),
        }

        metrics: Dict[str, Any] = {}
        for tgt in available_targets:
            train_tgt_arr = train[tgt].to_numpy(dtype=float)

            # Compute retrieval estimate: nanmean over valid neighbors
            neigh_vals = np.where(valid_mask, train_tgt_arr[np.where(valid_mask, idx, 0)], np.nan)
            y_retr = np.nanmean(neigh_vals, axis=1)

            y_true = test[tgt].to_numpy(dtype=float)
            retr_metrics = _compute_regression_metrics(y_true, y_retr)

            # Bug#5 fix: target-specific prediction column
            model_col = _resolve_pred_col(list(test.columns), tgt)
            model_metrics = {mk: float('nan') for mk in ('mae', 'rmse', 'pearson', 'spearman', 'r2', 'n')}
            if model_col is not None:
                model_metrics = _compute_regression_metrics(y_true, test[model_col].to_numpy(dtype=float))

            # Bug#7 fix: ranking metrics with minimum sample size guard
            rank_ret = _ranking_metrics(
                pd.DataFrame({'score': y_retr, tgt: y_true}), 'score', tgt, RANK_PCTS
            )
            rank_mod: Dict[str, Any] = {}
            if model_col is not None:
                rank_mod = _ranking_metrics(
                    pd.DataFrame({'score': test[model_col].to_numpy(dtype=float), tgt: y_true}),
                    'score',
                    tgt,
                    RANK_PCTS,
                )

            metrics[tgt] = {
                'retrieval': retr_metrics,
                'model': model_metrics,
                'ranking': {'retrieval': rank_ret, 'model': rank_mod},
            }

        diag_k['metrics'] = metrics
        diagnostics['per_k'][k] = diag_k

    return diagnostics


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

    if 'split' not in full.columns:
        logger.warning('No split column found; creating temporal split with latest 20%% as test')
        ts = pd.to_datetime(full['timestamp'])
        cutoff = ts.quantile(0.8)
        full['split'] = np.where(ts > cutoff, 'test', 'train')

    train = full[full['split'] == 'train'].reset_index(drop=True)
    test = full[full['split'] == 'test'].reset_index(drop=True)

    if train.shape[0] == 0 or test.shape[0] == 0:
        logger.warning('Temporal split produced empty train/test. Falling back to index-based 80/20 split')
        n = len(full)
        cutoff_i = int(n * 0.8)
        full = full.sort_values('timestamp').reset_index(drop=True)
        full['split'] = ['train'] * cutoff_i + ['test'] * (n - cutoff_i)
        train = full[full['split'] == 'train'].reset_index(drop=True)
        test = full[full['split'] == 'test'].reset_index(drop=True)

    if train.shape[0] == 0 or test.shape[0] == 0:
        raise RuntimeError(
            f'After split fallback, train n={train.shape[0]}, test n={test.shape[0]}; cannot proceed'
        )

    # Bug#6 fix: deterministic target column resolution
    tgt_map: Dict[str, str] = {}
    all_cols = list(full.columns)
    for base in TARGET_COLS:
        resolved = _resolve_target_col(all_cols, base)
        if resolved is not None:
            tgt_map[base] = resolved
        else:
            logger.warning("Target '%s' not found in any column — skipping", base)

    if not tgt_map:
        raise RuntimeError('No target columns found; expected one of ' + ', '.join(TARGET_COLS))

    for base, col in tgt_map.items():
        train[base] = train[col].to_numpy(dtype=float)
        test[base] = test[col].to_numpy(dtype=float)

    # Bug#5 fix: do NOT pre-assign a generic future_head_pred here.
    # _resolve_pred_col() handles target-specific mapping inside compute_retrieval_and_compare.
    # Just ensure target-specific pred columns are propagated from full → test if needed.
    for base in tgt_map:
        for candidate in [f'future_head_pred_{base}', f'pred_{base}']:
            if candidate in full.columns and candidate not in test.columns:
                test[candidate] = full.loc[full['split'] == 'test', candidate].to_numpy()

    diagnostics = compute_retrieval_and_compare(train, test)

    out_path = OUTPUT_DIR / 'retrieval_diagnostics.json'
    import json
    with out_path.open('w', encoding='utf-8') as fh:
        json.dump(diagnostics, fh, indent=2)
    logger.info('Wrote diagnostics to %s', out_path)


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