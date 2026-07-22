"""
Train simple baseline models per target using the same flat features (no sequence models).
Saves CSV performance tables under reports/phase2/artifacts/.
"""
from pathlib import Path
import argparse
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

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

MODEL_REGISTRY = {
    'Ridge': Ridge(),
    'RandomForest': RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=7),
}


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
    # Heuristic: use numeric columns excluding target cols and obvious identifiers
    exclude = set(TARGET_COLS + ['date', 'symbol', '__source_file'])
    num = df.select_dtypes(include=[np.number])
    cols = [c for c in num.columns if c not in exclude]
    return df[cols].fillna(0), cols


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed-dir', default=DEFAULT_PRECOMPUTED)
    args = parser.parse_args()

    df = load_frame(args.precomputed_dir)
    X_all, feature_cols = select_features(df)
    results = []

    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        y = df[target].fillna(0).values
        X = X_all.values
        # simple split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

        # baselines
        mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
        median_pred = np.full_like(y_test, np.median(y_train), dtype=float)
        mae_mean = mean_absolute_error(y_test, mean_pred)
        mae_median = mean_absolute_error(y_test, median_pred)

        for name, model in MODEL_REGISTRY.items():
            m = model
            try:
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
            except Exception as e:
                print('Model failed', name, 'for target', target, e)
                continue
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            try:
                pearson = pearsonr(y_test, y_pred)[0]
            except Exception:
                pearson = float('nan')
            results.append({'target': target, 'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2, 'pearson': pearson})

        # add baseline rows
        results.append({'target': target, 'model': 'MeanBaseline', 'mae': mae_mean, 'rmse': float('nan'), 'r2': float('nan'), 'pearson': float('nan')})
        results.append({'target': target, 'model': 'MedianBaseline', 'mae': mae_median, 'rmse': float('nan'), 'r2': float('nan'), 'pearson': float('nan')})

    pd.DataFrame(results).to_csv(OUT / 'baseline_model_performance.csv', index=False)
    print('Saved baseline performance to', OUT)

