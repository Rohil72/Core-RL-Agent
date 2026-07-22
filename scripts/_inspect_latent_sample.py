import pandas as pd
from pathlib import Path
import numpy as np
p=Path('reports/latent_analysis')
files=sorted(p.glob('*'))
if not files:
    print('NO_FILES')
else:
    f=files[0]
    df=pd.read_parquet(f)
    print(f.name, len(df))
    print('columns:', df.columns.tolist())
    print('latent type sample:', type(df['latent'].iloc[0]))
    arrs = [np.array(x) for x in df['latent'].iloc[:20]]
    print('shapes (first 20):', [a.shape for a in arrs])
    print('any nan in first 200?:', any(np.isnan(np.array(x)).any() for x in df['latent'].iloc[:200]))
    try:
        s = np.vstack(arrs)
        print('stack dtype:', s.dtype, 'shape:', s.shape)
    except Exception as e:
        print('vstack failed', e)
