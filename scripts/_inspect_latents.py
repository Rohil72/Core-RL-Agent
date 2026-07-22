from pathlib import Path
import pandas as pd
p=Path('reports/latent_analysis')
if not p.exists():
    print('NO_DIR')
else:
    for f in sorted(p.glob('*')):
        try:
            try:
                df = pd.read_parquet(f)
                reader='parquet'
            except Exception:
                df = pd.read_pickle(f)
                reader='pickle'
            cols = df.columns.tolist()
            n = len(df)
            sample = {c: type(df[c].iloc[0]).__name__ if n>0 else None for c in df.columns}
            print(f.name, reader, 'n=', n, 'cols=', cols, 'sample_types=', sample)
        except Exception as e:
            print(f.name, 'ERROR', e)

print('INSPECTION_DONE')
