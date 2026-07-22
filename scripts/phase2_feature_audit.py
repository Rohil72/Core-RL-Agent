"""
Compute feature-target correlations, mutual information, and tree importances. Save artifacts under reports/phase2/artifacts/.
"""
from pathlib import Path
import argparse
import glob
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

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


def load_frame(precomputed_glob):
    files = sorted(glob.glob(precomputed_glob))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print('Failed to read', f, e)
    if not dfs:
        raise SystemExit('No parquet files found for pattern: ' + precomputed_glob)
    return pd.concat(dfs, ignore_index=True)


def select_features(df):
    exclude = set(TARGET_COLS + ['date', 'symbol', '__source_file'])
    num = df.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if c not in exclude]
    return df[cols].fillna(0), cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed-dir', default=DEFAULT_PRECOMPUTED)
    args = parser.parse_args()

    df = load_frame(args.precomputed_dir)
    X_df, feature_cols = select_features(df)

    correlations = {}
    mutual_info = {}
    tree_importances = {}

    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        y = df[target].fillna(0).values
        X = X_df.values
        # Pearson correlation per feature
        corrs = np.array([np.corrcoef(X[:,i], y)[0,1] if np.std(X[:,i])>0 else 0.0 for i in range(X.shape[1])])
        correlations[target] = dict(zip(feature_cols, np.nan_to_num(corrs).tolist()))

        # mutual info (may be slow)
        try:
            mi = mutual_info_regression(X, y, random_state=7)
            mutual_info[target] = dict(zip(feature_cols, mi.tolist()))
        except Exception as e:
            print('Mutual info failed for', target, e)
            mutual_info[target] = {}

        # tree importance
        try:
            rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=7)
            rf.fit(X, y)
            imp = rf.feature_importances_
            tree_importances[target] = dict(zip(feature_cols, imp.tolist()))
        except Exception as e:
            print('Tree importance failed for', target, e)
            tree_importances[target] = {}

    import json
    with open(OUT / 'feature_correlations.json', 'w') as fh:
        json.dump(correlations, fh, indent=2)
    with open(OUT / 'feature_mutual_info.json', 'w') as fh:
        json.dump(mutual_info, fh, indent=2)
    with open(OUT / 'feature_tree_importances.json', 'w') as fh:
        json.dump(tree_importances, fh, indent=2)

    print('Feature audit artifacts saved to', OUT)

