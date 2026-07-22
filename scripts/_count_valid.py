import pandas as pd
import numpy as np
from pathlib import Path
p=Path('reports/latent_analysis')
files=sorted(p.glob('*'))
total_valid=0
for f in files:
    try:
        df=pd.read_parquet(f)
    except Exception:
        df=pd.read_pickle(f)
    # detect latent col
    latent_col=None
    for c in df.columns:
        sample = df[c].iloc[0] if len(df)>0 else None
        if isinstance(sample, (list, np.ndarray)):
            latent_col=c
            break
    if latent_col is None:
        print('no latent col for', f.name)
        continue
    arrs = np.vstack(df[latent_col].apply(lambda x: np.array(x)).values)
    valid = np.isfinite(arrs).all(axis=1)
    total_valid += int(valid.sum())
print('files:', len(files), 'total_valid_latents=', total_valid)
