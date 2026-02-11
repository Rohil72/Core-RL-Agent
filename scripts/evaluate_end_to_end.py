import argparse
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import logging
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.labeler import create_cycle_example
from src.data.feature_engineering import assemble_features
from src.models.timesnet_encoder import TimesNetEncoder
from src.eval import metrics as eval_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def load_or_generate_cycles(quick=False):
    """Load existing cycles or generate synthetic ones."""
    cycles_dir = "data/cycles"
    
    if os.path.exists(cycles_dir) and glob.glob(os.path.join(cycles_dir, "*.jsonl")):
        logger.info(f"Loading existing cycles from {cycles_dir}")
        all_cycles = []
        for jsonl_file in glob.glob(os.path.join(cycles_dir, "*.jsonl")):
            df = pd.read_json(jsonl_file, lines=True)
            all_cycles.append(df)
        if all_cycles:
            return pd.concat(all_cycles, ignore_index=True)
    
    logger.warning("No cycles found. Generating synthetic data.")
    num_cycles = 50 if quick else 200
    base_time = pd.Timestamp("2023-01-01", tz='UTC')
    
    cycles = []
    for i in range(num_cycles):
        cycle = {
            'ticker': f'STOCK{i % 10}',
            'cycle_start': (base_time + pd.Timedelta(days=i*7)).isoformat(),
            'cycle_end': (base_time + pd.Timedelta(days=i*7 + 30)).isoformat(),
            'context_window': [],
            'derived_features': {
                'duration_days': 30,
                'net_return': np.random.uniform(0.1, 0.5),
                'peak_date': (base_time + pd.Timedelta(days=i*7 + 15)).isoformat()
            },
            'fundamental_snapshot': {
                'Reported EPS': np.random.uniform(1.0, 3.0),
                'EPS Estimate': np.random.uniform(1.0, 3.0)
            },
            'future_returns': {'ret_21': np.random.uniform(-0.1, 0.3)},
            'fundamental_confirmations': {
                'fundamental_confirmed': bool(np.random.random() > 0.6),
                'fundamental_confirmed_int': 1 if np.random.random() > 0.6 else 0,
                'next_report_date': (base_time + pd.Timedelta(days=i*7 + 60)).isoformat()
            }
        }
        cycles.append(cycle)
    
    return pd.DataFrame(cycles)

def build_feature_tables(cycles_df, quick=False):
    """Step 1: Build feature tables from cycles."""
    logger.info("Step 1: Building feature tables")
    
    os.makedirs("data/feature_tables", exist_ok=True)
    
    features_list = []
    for idx, row in cycles_df.iterrows():
        try:
            feats = assemble_features(row.to_dict())
            feats['confirmation_label'] = row.get('fundamental_confirmations', {}).get('fundamental_confirmed', False)
            features_list.append(feats)
        except Exception as e:
            logger.warning(f"Failed to process cycle {idx}: {e}")
            continue
    
    if not features_list:
        raise ValueError("No features generated")
    
    features_df = pd.DataFrame(features_list)
    output_path = "data/feature_tables/features.parquet"
    features_df.to_parquet(output_path)
    logger.info(f"Saved {len(features_df)} feature rows to {output_path}")
    
    return features_df

def pretrain_encoder(features_df, config):
    """Step 2: Pretrain encoder with simple reconstruction task."""
    logger.info("Step 2: Pretraining encoder")
    
    # Extract numeric features
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'confirmation_label' in numeric_cols:
        numeric_cols.remove('confirmation_label')
    
    X = features_df[numeric_cols].fillna(0).values.astype(np.float32)
    
    # Normalize
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - mean) / std
    
    # Reshape to (batch, seq_len, features)
    # For simplicity, treat each feature vector as a sequence of length 1
    # Or pad/truncate to fixed length
    in_dim = min(X_norm.shape[1], 10)  # Use first 10 features
    X_input = X_norm[:, :in_dim]
    X_input = X_input.reshape(len(X_input), 1, in_dim)  # (N, 1, in_dim)
    
    encoder = TimesNetEncoder(
        in_dim=in_dim, 
        embed_dim=config['embedding_dim'], 
        num_layers=2
    )
    
    # Simple reconstruction head
    decoder = nn.Linear(config['embedding_dim'], in_dim)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    decoder.to(device)
    
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), 
        lr=1e-3
    )
    criterion = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_input))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    epochs = config['encoder_epochs']
    for epoch in range(epochs):
        total_loss = 0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            embeddings = encoder(batch)  # (B, 1, embed_dim)
            reconstructed = decoder(embeddings)  # (B, 1, in_dim)
            
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % max(1, epochs // 5) == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
    
    # Save encoder
    os.makedirs("models/encoder", exist_ok=True)
    encoder_path = f"models/encoder/{config['run_id']}_encoder.pt"
    torch.save({
        'state_dict': encoder.state_dict(),
        'config': {'in_dim': in_dim, 'embed_dim': config['embedding_dim']},
        'normalization': {'mean': mean.tolist(), 'std': std.tolist()}
    }, encoder_path)
    
    logger.info(f"Encoder saved to {encoder_path}")
    return encoder_path, encoder, (mean, std, in_dim)

def generate_embeddings(features_df, encoder, norm_params, config):
    """Step 3: Generate embeddings for all cycles."""
    logger.info("Step 3: Generating embeddings")
    
    mean, std, in_dim = norm_params
    
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'confirmation_label' in numeric_cols:
        numeric_cols.remove('confirmation_label')
    
    X = features_df[numeric_cols].fillna(0).values.astype(np.float32)
    X_norm = (X - mean) / std
    X_input = X_norm[:, :in_dim].reshape(len(X_norm), 1, in_dim)
    
    device = next(encoder.parameters()).device
    encoder.eval()
    
    embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(X_input), 64):
            batch = torch.FloatTensor(X_input[i:i+64]).to(device)
            emb = encoder(batch)
            # Take mean over sequence dimension
            emb_mean = emb.mean(dim=1).cpu().numpy()
            embeddings_list.append(emb_mean)
    
    embeddings = np.vstack(embeddings_list)
    
    # Build embedding dataframe
    emb_df = pd.DataFrame({
        'cycle_id': [f"cycle_{i}" for i in range(len(embeddings))],
        'ticker': features_df['ticker'].values if 'ticker' in features_df.columns else ['UNKNOWN'] * len(embeddings),
        'timestamp': features_df['cycle_end'].values if 'cycle_end' in features_df.columns else pd.date_range('2023-01-01', periods=len(embeddings)),
        'embedding': [emb.astype(np.float32) for emb in embeddings],
        'price_features': [np.random.randn(5).astype(np.float32) for _ in range(len(embeddings))],
        'confirmation_label': features_df['confirmation_label'].values if 'confirmation_label' in features_df.columns else np.zeros(len(embeddings)),
        'confirmation_time': features_df['cycle_end'].values if 'cycle_end' in features_df.columns else pd.date_range('2023-01-01', periods=len(embeddings))
    })
    
    os.makedirs("data/embeddings", exist_ok=True)
    emb_path = f"data/embeddings/{config['run_id']}_embeddings.parquet"
    emb_df.to_parquet(emb_path)
    logger.info(f"Saved {len(emb_df)} embeddings to {emb_path}")
    
    return emb_path, emb_df

def train_rl_agent(embeddings_path, config):
    """Step 4: Train RL agent."""
    logger.info("Step 4: Training RL agent")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from src.envs.cycle_trade_env import CycleTradeEnv
    except ImportError as e:
        logger.error(f"RL dependencies missing: {e}")
        return None, None
    
    # Load embeddings
    emb_df = pd.read_parquet(embeddings_path)
    
    # Split train/test
    split_idx = int(len(emb_df) * 0.8)
    train_df = emb_df.iloc[:split_idx]
    test_df = emb_df.iloc[split_idx:]
    
    # Create dataset iterator
    class EmbeddingDataset:
        def __init__(self, df):
            self.data = df.to_dict('records')
            self.idx = 0
        
        def __iter__(self):
            self.idx = 0
            return self
        
        def __next__(self):
            if self.idx >= len(self.data):
                raise StopIteration
            item = self.data[self.idx]
            self.idx += 1
            return item
        
        def reset(self):
            self.idx = 0
    
    # Environment config
    env_config = {
        'embedding_size': config['embedding_dim'],
        'action_space_size': 3,
        'reward_params': {
            'c_flag': -0.01,
            'R_confirm': 1.0,
            'R_miss': -0.5,
            'R_miss_ignore': 0.0
        }
    }
    
    # Create environment
    def make_env():
        env = CycleTradeEnv(env_config, data_source=EmbeddingDataset(train_df))
        return env
    
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Train PPO
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        verbose=0
    )
    
    model.learn(total_timesteps=config['rl_steps'])
    
    # Save model
    os.makedirs(f"models/rl/{config['run_id']}", exist_ok=True)
    model_path = f"models/rl/{config['run_id']}/final_model"
    model.save(model_path)
    vec_env.save(f"models/rl/{config['run_id']}/vec_normalize.pkl")
    
    logger.info(f"RL model saved to {model_path}")
    
    return model_path, test_df

def evaluate_agent(model_path, test_df, config):
    """Step 5: Evaluate agent."""
    logger.info("Step 5: Evaluating agent")
    
    try:
        from stable_baselines3 import PPO
    except ImportError:
        logger.error("Cannot evaluate without stable-baselines3")
        return {}
    
    model = PPO.load(model_path)
    
    y_true = []
    y_pred_probs = []
    y_pred_actions = []
    
    for idx, row in test_df.iterrows():
        obs = {
            'embedding': row['embedding'],
            'price_features': row['price_features']
        }
        
        action, _ = model.predict(obs, deterministic=True)
        y_pred_actions.append(action)
        
        # Get probability
        with torch.no_grad():
            obs_tensor = model.policy.obs_to_tensor(obs)[0]
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]
            y_pred_probs.append(probs[1] + probs[2])  # Flag + Proactive
        
        y_true.append(int(row['confirmation_label']))
    
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_flags = (np.array(y_pred_actions) > 0).astype(int)
    
    # Compute metrics
    class_metrics = eval_metrics.compute_classification_metrics(y_true, y_pred_probs)
    calib_metrics = eval_metrics.compute_calibration_metrics(y_true, y_pred_probs)
    coverage = eval_metrics.compute_coverage(y_pred_flags)
    
    # Stability metric
    embeddings = np.stack(test_df['embedding'].values)
    tickers = test_df['ticker'].values.tolist()
    stability = eval_metrics.compute_stability_metric(embeddings, tickers)
    
    metrics_dict = {
        **class_metrics,
        'brier_score': calib_metrics['brier_score'],
        'coverage': coverage,
        'stability_drift': stability
    }
    
    # Generate plots
    eval_metrics.generate_all_plots(
        calib_metrics,
        metrics_dict,
        "reports/figures",
        config['run_id']
    )
    
    logger.info(f"Metrics: {metrics_dict}")
    return metrics_dict, calib_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick run for testing')
    args = parser.parse_args()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = {
        'run_id': run_id,
        'quick_run': args.quick,
        'encoder_epochs': 2 if args.quick else 10,
        'rl_steps': 2000 if args.quick else 50000,
        'embedding_dim': 128
    }
    
    report = {
        'run_id': run_id,
        'config': config,
        'steps': {},
        'artifacts': {},
        'metrics': {}
    }
    
    try:
        # Step 1: Load/Generate cycles
        cycles_df = load_or_generate_cycles(quick=args.quick)
        report['steps']['load_cycles'] = 'SUCCESS'
        
        # Step 2: Build features
        features_df = build_feature_tables(cycles_df, quick=args.quick)
        report['steps']['feature_engineering'] = 'SUCCESS'
        
        # Step 3: Pretrain encoder
        encoder_path, encoder, norm_params = pretrain_encoder(features_df, config)
        report['steps']['encoder_pretraining'] = 'SUCCESS'
        report['artifacts']['encoder_path'] = encoder_path
        
        # Step 4: Generate embeddings
        emb_path, emb_df = generate_embeddings(features_df, encoder, norm_params, config)
        report['steps']['embedding_generation'] = 'SUCCESS'
        report['artifacts']['embeddings_path'] = emb_path
        
        # Step 5: Train RL
        model_path, test_df = train_rl_agent(emb_path, config)
        if model_path:
            report['steps']['rl_training'] = 'SUCCESS'
            report['artifacts']['rl_model_path'] = model_path
            
            # Step 6: Evaluate
            metrics_dict, calib_metrics = evaluate_agent(model_path, test_df, config)
            report['steps']['evaluation'] = 'SUCCESS'
            report['metrics'] = metrics_dict
        else:
            report['steps']['rl_training'] = 'SKIPPED'
            report['steps']['evaluation'] = 'SKIPPED'
        
    except Exception as e:
        logger.exception("Pipeline failed")
        report['error'] = str(e)
        sys.exit(1)
    
    # Save reports
    os.makedirs("reports", exist_ok=True)
    
    with open(f"reports/{run_id}.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    md_content = f"""# End-to-End Evaluation Report

**Run ID**: `{run_id}`  
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

```yaml
encoder_epochs: {config['encoder_epochs']}
rl_steps: {config['rl_steps']}
embedding_dim: {config['embedding_dim']}
quick_run: {config['quick_run']}
```

## Pipeline Steps

{chr(10).join(f"- **{k}**: {v}" for k, v in report['steps'].items())}

## Metrics

| Metric | Value |
|--------|-------|
| Precision | {report['metrics'].get('precision', 'N/A'):.4f} |
| Recall | {report['metrics'].get('recall', 'N/A'):.4f} |
| F1 Score | {report['metrics'].get('f1', 'N/A'):.4f} |
| AUC | {report['metrics'].get('auc', 'N/A'):.4f} |
| Coverage | {report['metrics'].get('coverage', 'N/A'):.4f} |
| Brier Score | {report['metrics'].get('brier_score', 'N/A'):.4f} |
| Stability Drift | {report['metrics'].get('stability_drift', 'N/A'):.4f} |

## Artifacts

- **Encoder**: `{report['artifacts'].get('encoder_path', 'N/A')}`
- **Embeddings**: `{report['artifacts'].get('embeddings_path', 'N/A')}`
- **RL Model**: `{report['artifacts'].get('rl_model_path', 'N/A')}`
- **Figures**: `reports/figures/{run_id}_*.png`
"""
    
    with open(f"reports/{run_id}.md", 'w') as f:
        f.write(md_content)
    
    logger.info(f"Reports saved to reports/{run_id}.(json|md)")
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
