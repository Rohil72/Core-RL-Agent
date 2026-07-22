"""
Compute global and per-ticker statistics and plots for configured future targets.
Saves CSV/PNG outputs under reports/phase2/artifacts/.
"""
from pathlib import Path
import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'phase2' / 'artifacts'
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_PRECOMPUTED = str(ROOT / 'data' / 'precomputed' / '*.parquet')
TARGET_COLS = [
    'future_return_21', 'future_return_63', 'future_return_126',
    'future_max_return_63', 'future_min_return_63',
    'event_peak_offset_63', 'event_drawdown_offset_63',
    'event_upside_before_drawdown_126', 'event_upside_hit_126', 'event_drawdown_hit_126'
]


def load_frames(precomputed_glob):
    files = sorted(glob.glob(precomputed_glob))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df['__source_file'] = Path(f).name
            dfs.append(df)
        except Exception as e:
            print('Failed to read', f, e)
    if not dfs:
        raise SystemExit('No parquet files found for pattern: ' + precomputed_glob)
    return pd.concat(dfs, ignore_index=True)


def describe_targets(df):
    stats = {}
    for c in TARGET_COLS:
        if c not in df.columns:
            continue
        arr = df[c].dropna().values
        if arr.size == 0:
            continue
        stats[c] = {
            'count': int(arr.size),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'skew': float(stats_module.skew(arr)),
            'kurtosis': float(stats_module.kurtosis(arr)),
        }
        quantiles = np.percentile(arr, [1,5,10,25,50,75,90,95,99]).tolist()
        stats[c]['quantiles'] = quantiles
    return stats


# scipy.stats wrapper to avoid shadowing
from scipy import stats as stats_module


def per_ticker_stats(df):
    if 'symbol' not in df.columns:
        return {}
    out = {}
    for sym, g in df.groupby('symbol'):
        out[sym] = {}
        for c in TARGET_COLS:
            if c not in g.columns:
                continue
            arr = g[c].dropna().values
            if arr.size == 0:
                continue
            out[sym][c] = {
                'count': int(arr.size),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
    return out


def plot_distributions(df):
    for c in TARGET_COLS:
        if c not in df.columns:
            continue
        plt.figure(figsize=(6,4))
        sns.histplot(df[c].dropna(), bins=200, kde=True)
        plt.title(c)
        p = OUT / f'{c}_hist.png'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()

    # correlation heatmap
    present = [c for c in TARGET_COLS if c in df.columns]
    if len(present) >= 2:
        corr = df[present].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag')
        plt.title('target_correlation')
        p = OUT / 'target_correlation.png'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed-dir', default=DEFAULT_PRECOMPUTED)
    args = parser.parse_args()

    df = load_frames(args.precomputed_dir)
    print('Read rows:', len(df))

    stats = describe_targets(df)
    pd.Series({k: v['mean'] for k, v in stats.items()}).to_csv(OUT / 'target_global_means.csv')

    # save JSON summary
    import json
    with open(OUT / 'target_global_stats.json', 'w') as fh:
        json.dump(stats, fh, indent=2)

    per = per_ticker_stats(df)
    with open(OUT / 'target_per_ticker_stats.json', 'w') as fh:
        json.dump(per, fh, indent=2)

    plot_distributions(df)
    print('Artifacts written to', OUT)

