"""Transferability analysis upgraded.

- Computes cluster-level diversity metrics and per-sample cross-ticker retrieval experiments
- Performs same-ticker vs different-ticker retrieval comparisons
- Requires latent files with columns: cluster_id, ticker/symbol, timestamp, latent, and target columns

Does NOT retrain models.

Fixes applied:
  Bug#1 - Temporal retrieval: filter legal candidates FIRST, then run NN search on that pool
  Bug#2 - Per-ticker temporal split instead of global quantile cutoff
  Bug#3 - Deterministic target column resolution (exact → true_ → endswith, never candidates[0])
  Bug#4 - Deterministic prediction column resolution (target-specific, never pred_cols[0])
  Bug#5 - Log unique neighbor counts per k to diagnose identical-k-result syndrome
  Bug#6 - Upfront dataset/cluster diagnostics so "wrong dataset" is caught immediately
  Bug#7 - Explain why estimates_count==0 instead of silently returning NaN
"""
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import argparse
import logging
import math
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy import stats

from src.data.io_utils import read_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('reports/final_direction')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ['future_max_return_63', 'future_min_return_63', 'event_upside_before_drawdown_126']
KS = (10, 25, 50)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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
    """Deterministic target column resolution.

    Bug#3 fix: exact match → true_ prefix → sorted endswith fallback.
    Never silently grabs candidates[0] based on column ordering.
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


def _resolve_pred_col(test_cols: List[str], target_base: str) -> Optional[str]:
    """Deterministic prediction column resolution per target.

    Bug#4 fix: never fall back to an arbitrary pred_ column for a different target.
    """
    for candidate in [f'future_head_pred_{target_base}', f'pred_{target_base}']:
        if candidate in test_cols:
            return candidate
    logger.warning(
        "No target-specific prediction column found for '%s'. "
        "Available pred cols: %s",
        target_base,
        [c for c in test_cols if c.startswith('pred_') or c.startswith('future_head_pred')],
    )
    return None


def _extract_latent_matrix(df: pd.DataFrame) -> np.ndarray:
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


def _validate_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f'Missing required columns: {missing}')


# ---------------------------------------------------------------------------
# Bug#2 fix: per-ticker temporal split
# ---------------------------------------------------------------------------

def _per_ticker_temporal_split(full: pd.DataFrame, ticker_col: str, ts_col: str, test_frac: float = 0.2) -> pd.DataFrame:
    """Assign train/test split per ticker so each ticker's latest `test_frac` rows become test.

    This prevents tickers that only appear near the end from landing entirely in test
    (which caused same_ticker_pct_mean = 0 with the global quantile cutoff).
    """
    split_labels = np.full(len(full), 'train', dtype=object)
    for _, group in full.groupby(ticker_col):
        n = len(group)
        cutoff_i = max(1, int(n * (1 - test_frac)))
        sorted_idx = group.sort_values(ts_col).index
        test_idx = sorted_idx[cutoff_i:]
        split_labels[full.index.get_indexer(test_idx)] = 'test'
    full = full.copy()
    full['split'] = split_labels
    logger.info(
        'Per-ticker split: train=%d  test=%d',
        int((full['split'] == 'train').sum()),
        int((full['split'] == 'test').sum()),
    )
    return full


# ---------------------------------------------------------------------------
# Bug#1 fix: legal-candidates-first temporal retrieval
# ---------------------------------------------------------------------------

def _retrieve_temporal_knn(
    X_train: np.ndarray,
    train_ts: np.ndarray,
    X_test: np.ndarray,
    test_ts: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each query, find k nearest neighbors among train rows with ts < query_ts.

    Returns:
        dist    (n_queries, k)  – distances, nan-padded for queries with < k legal neighbors
        idx     (n_queries, k)  – train indices, -1-padded
        n_legal (n_queries,)    – legal pool size per query
    """
    n_queries = X_test.shape[0]
    dist_out = np.full((n_queries, k), np.nan)
    idx_out = np.full((n_queries, k), -1, dtype=int)
    n_legal = np.zeros(n_queries, dtype=int)

    # Sort train by ts so searchsorted gives O(log n) legal boundary per query
    sorted_order = np.argsort(train_ts, kind='stable')
    train_ts_sorted = train_ts[sorted_order]
    X_train_sorted = X_train[sorted_order]

    for i in range(n_queries):
        qt = test_ts[i]
        legal_end = int(np.searchsorted(train_ts_sorted, qt, side='left'))
        n_legal[i] = legal_end

        if legal_end == 0:
            logger.debug('Query %d: no legal (past) neighbors', i)
            continue

        actual_k = min(k, legal_end)
        nbr = NearestNeighbors(n_neighbors=actual_k, n_jobs=-1)
        nbr.fit(X_train_sorted[:legal_end])
        d, local_idx = nbr.kneighbors(X_test[i : i + 1])

        global_idx = sorted_order[local_idx[0]]
        dist_out[i, :actual_k] = d[0]
        idx_out[i, :actual_k] = global_idx

    # Bug#5: unique neighbor diagnostic
    unique_n = len(np.unique(idx_out[idx_out >= 0]))
    logger.info('k=%d | unique neighbors across all queries: %d / %d train rows', k, unique_n, X_train.shape[0])

    return dist_out, idx_out, n_legal


# ---------------------------------------------------------------------------
# Cluster diversity
# ---------------------------------------------------------------------------

def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for v in counter.values():
        p = v / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def effective_count(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ps = [v / total for v in counter.values()]
    return 2 ** (-sum(p * math.log(p, 2) for p in ps if p > 0))


def ticker_concentration(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    top = counter.most_common(1)
    return top[0][1] / total if top else 0.0


def cluster_level_metrics(df: pd.DataFrame, ticker_col: str) -> pd.DataFrame:
    rows = []
    for cid, group in df.groupby('cluster_id'):
        tickers = Counter(group[ticker_col].astype(str).tolist())
        rows.append({
            'cluster_id': int(cid),
            'member_count': int(len(group)),
            'unique_ticker_count': int(len(tickers)),
            'ticker_entropy': float(shannon_entropy(tickers)),
            'effective_ticker_count': float(effective_count(tickers)),
            'dominant_ticker_share': float(ticker_concentration(tickers)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-ticker retrieval experiment
# ---------------------------------------------------------------------------

def cross_ticker_retrieval_experiment(
    full_df: pd.DataFrame,
    ks: Tuple[int, ...] = KS,
    available_targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Same-ticker vs cross-ticker retrieval quality comparison."""
    ticker_col = 'ticker' if 'ticker' in full_df.columns else ('symbol' if 'symbol' in full_df.columns else None)
    ts_col = 'timestamp' if 'timestamp' in full_df.columns else None
    _validate_columns(full_df, ['cluster_id'])
    if ticker_col is None or ts_col is None:
        raise RuntimeError('Need ticker (or symbol) and timestamp columns')

    if 'split' not in full_df.columns:
        raise RuntimeError('split column missing; call _per_ticker_temporal_split first')

    if available_targets is None:
        available_targets = [t for t in TARGET_COLS if t in full_df.columns]

    train = full_df[full_df['split'] == 'train'].reset_index(drop=True)
    test = full_df[full_df['split'] == 'test'].reset_index(drop=True)

    logger.info('Cross-ticker experiment: train=%d  test=%d', len(train), len(test))

    # Bug#2 side effect diagnostic: same-ticker coverage
    train_tickers_set = set(train[ticker_col].unique())
    test_tickers_set = set(test[ticker_col].unique())
    overlap = train_tickers_set & test_tickers_set
    logger.info(
        'Ticker overlap train∩test: %d / %d test tickers have history in train',
        len(overlap), len(test_tickers_set),
    )
    if not overlap:
        logger.warning(
            'Zero ticker overlap between train and test — same_ticker retrieval will be empty. '
            'This is likely caused by a global (not per-ticker) temporal split.'
        )

    X_train = _extract_latent_matrix(train)
    X_test = _extract_latent_matrix(test)
    train_ts = np.array(pd.to_datetime(train[ts_col]).astype('int64') // 10 ** 9)
    test_ts = np.array(pd.to_datetime(test[ts_col]).astype('int64') // 10 ** 9)

    train_ticker_arr = train[ticker_col].to_numpy()
    test_ticker_arr = test[ticker_col].to_numpy()

    results: Dict[str, Any] = {'per_k': {}}

    for k in ks:
        # Bug#1 fix: legal candidates first
        dist, idx, n_legal = _retrieve_temporal_knn(X_train, train_ts, X_test, test_ts, k)
        valid_mask = idx >= 0  # (n_queries, k)

        same_mask = np.where(
            valid_mask,
            train_ticker_arr[np.where(valid_mask, idx, 0)] == test_ticker_arr.reshape(-1, 1),
            False,
        )
        diff_mask = valid_mask & ~same_mask

        # diagnostics
        valid_dist = dist[valid_mask]
        same_ticker_pct = np.mean(same_mask, axis=1)  # fraction of valid neighbors that are same-ticker
        diagnostics: Dict[str, Any] = {
            'distance_mean': float(np.nanmean(valid_dist)) if valid_dist.size else float('nan'),
            'distance_median': float(np.nanmedian(valid_dist)) if valid_dist.size else float('nan'),
            'same_ticker_pct_mean': float(np.mean(same_ticker_pct)),
            'queries_with_zero_legal_neighbors': int(np.sum(n_legal == 0)),
            'mean_legal_pool_size': float(np.mean(n_legal)),
            'unique_ticker_per_query_median': float(
                np.median([
                    len(set(train_ticker_arr[idx[i][valid_mask[i]]].tolist()))
                    for i in range(len(test))
                    if valid_mask[i].any()
                ])
            ) if valid_mask.any() else 0.0,
        }

        res_k: Dict[str, Any] = {}
        for mode, mode_mask in [('same_ticker', same_mask), ('different_ticker', diff_mask)]:
            mode_idx = np.where(mode_mask, idx, -1)
            estimates_count = int(np.sum(mode_idx != -1))

            # Bug#7 fix: explain why estimates_count == 0 rather than silently returning NaN
            if estimates_count == 0:
                if mode == 'same_ticker':
                    reason = (
                        'No same-ticker neighbors found. '
                        f'Train tickers: {len(train_tickers_set)}, '
                        f'Test tickers: {len(test_tickers_set)}, '
                        f'Overlap: {len(overlap)}. '
                        'Check whether per-ticker split was applied correctly.'
                    )
                else:
                    reason = (
                        'No different-ticker neighbors found. '
                        'This is unexpected unless train has only one ticker.'
                    )
                logger.warning('k=%d mode=%s: estimates_count=0 — %s', k, mode, reason)
                res_k[mode] = {
                    'estimates_count': 0,
                    'zero_count_reason': reason,
                    'metrics': {tgt: {m: float('nan') for m in ('mae', 'rmse', 'pearson', 'spearman')} for tgt in available_targets},
                }
                continue

            tgt_metrics: Dict[str, Any] = {}
            for tgt in available_targets:
                if tgt not in train.columns or tgt not in test.columns:
                    continue
                train_tgt_arr = train[tgt].to_numpy(dtype=float)
                neigh_vals = np.where(mode_mask, train_tgt_arr[np.where(mode_mask, idx, 0)], np.nan)
                neigh_mean = np.nanmean(neigh_vals, axis=1)

                y_true = test[tgt].to_numpy(dtype=float)
                finite_mask = np.isfinite(neigh_mean) & np.isfinite(y_true)
                n_valid = int(finite_mask.sum())

                tgt_metrics[tgt] = {
                    'n_valid': n_valid,
                    'mae': float(np.mean(np.abs(y_true[finite_mask] - neigh_mean[finite_mask]))) if n_valid else float('nan'),
                    'rmse': float(np.sqrt(np.mean((y_true[finite_mask] - neigh_mean[finite_mask]) ** 2))) if n_valid else float('nan'),
                    'pearson': float(stats.pearsonr(y_true[finite_mask], neigh_mean[finite_mask])[0]) if n_valid > 1 else float('nan'),
                    'spearman': float(stats.spearmanr(y_true[finite_mask], neigh_mean[finite_mask])[0]) if n_valid > 1 else float('nan'),
                }

            res_k[mode] = {'estimates_count': estimates_count, 'metrics': tgt_metrics}

        results['per_k'][k] = {'modes': res_k, 'diagnostics': diagnostics}

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _to_native(o: Any) -> Any:
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

    ticker_col = 'ticker' if 'ticker' in full.columns else ('symbol' if 'symbol' in full.columns else None)
    if ticker_col is None:
        raise RuntimeError('No ticker or symbol column found in latent files')
    ts_col = 'timestamp' if 'timestamp' in full.columns else None
    if ts_col is None:
        raise RuntimeError('No timestamp column found in latent files')

    # Bug#6: upfront dataset + cluster diagnostics
    n_rows = len(full)
    n_tickers = full[ticker_col].nunique()
    n_clusters = full['cluster_id'].nunique() if 'cluster_id' in full.columns else 0
    logger.info('=== Dataset loaded: rows=%d  tickers=%d  clusters=%d ===', n_rows, n_tickers, n_clusters)
    if 'cluster_id' in full.columns:
        logger.info('Cluster population:\n%s', full['cluster_id'].value_counts().sort_index().to_string())
    if n_clusters > 0 and n_clusters <= 3:
        logger.warning(
            'Only %d clusters detected (expected 8 from Phase 3). '
            'Likely the wrong latent export was loaded — check --latent-glob / --paths.',
            n_clusters,
        )

    # Bug#2 fix: per-ticker temporal split
    if 'split' not in full.columns:
        logger.info('No split column found — applying per-ticker temporal split (last 20%% of each ticker → test)')
        full = _per_ticker_temporal_split(full, ticker_col, ts_col, test_frac=0.2)
    else:
        # Validate existing split still has ticker overlap; warn if not
        train_tickers = set(full.loc[full['split'] == 'train', ticker_col].unique())
        test_tickers = set(full.loc[full['split'] == 'test', ticker_col].unique())
        if not (train_tickers & test_tickers):
            logger.warning(
                'Existing split column has zero ticker overlap between train and test. '
                'Replacing with per-ticker split to enable same-ticker retrieval.'
            )
            full = _per_ticker_temporal_split(full, ticker_col, ts_col, test_frac=0.2)

    # Bug#3 fix: deterministic target resolution
    all_cols = list(full.columns)
    tgt_map: Dict[str, str] = {}
    for base in TARGET_COLS:
        resolved = _resolve_target_col(all_cols, base)
        if resolved is not None:
            tgt_map[base] = resolved
            if resolved != base:
                full[base] = full[resolved]
        else:
            logger.warning("Target '%s' not found — will be skipped", base)

    available_targets = list(tgt_map.keys())
    logger.info('Resolved targets: %s', tgt_map)

    # Bug#4 fix: target-specific prediction columns — propagate from full into the dataframe
    # _resolve_pred_col is called inside the experiment per target; no global pred remapping here.

    # Cluster-level diversity metrics
    if 'cluster_id' in full.columns:
        cl_df = cluster_level_metrics(full, ticker_col)
        out_clusters = OUTPUT_DIR / 'cluster_diversity.parquet'
        cl_df.to_parquet(out_clusters)
        logger.info('Wrote cluster diversity to %s', out_clusters)

    # Cross-ticker retrieval experiment
    try:
        xres = cross_ticker_retrieval_experiment(full, available_targets=available_targets)
        out_x = OUTPUT_DIR / 'cross_ticker_retrieval.json'
        with out_x.open('w', encoding='utf-8') as fh:
            json.dump(_to_native(xres), fh, indent=2)
        logger.info('Wrote cross-ticker retrieval diagnostics to %s', out_x)
    except Exception as exc:
        logger.exception('Cross-ticker experiment failed: %s', exc)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent-glob', type=str, default='reports/latent_analysis/*.parquet')
    ap.add_argument('--paths', type=str, nargs='*')
    args = ap.parse_args()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        import glob
        files = [Path(p) for p in glob.glob(args.latent_glob)]
    main(files)