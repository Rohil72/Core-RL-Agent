from pathlib import Path
import pandas as pd
import numpy as np
Path('reports/latent_analysis').mkdir(parents=True, exist_ok=True)
n=20
latents=[list(np.random.randn(8)) for _ in range(n)]
df=pd.DataFrame({
 'latent':latents,
 'split':['train']*10+['test']*10,
 'timestamp':pd.date_range('2020-01-01', periods=20, freq='D'),
 'ticker':['A']*5+['B']*5+['C']*5+['D']*5,
 'cluster_id':[0,1,0,1,2]*4,
 'future_max_return_63':np.random.randn(20),
 'future_min_return_63':np.random.randn(20),
 'event_upside_before_drawdown_126':np.random.randn(20),
 'future_head_pred':np.random.randn(20),
})
df.to_pickle('reports/latent_analysis/test_small.parquet')
print('created')
