# Phase 5 Execution

Phase 5 is additive. Phase 4C, A2, 4H, and 4I artifacts are read-only inputs.
The 2024 split is development evidence. The 2025 split stays locked until the
adapter, optional alignment pass, policy, and international configuration are frozen.

## 1. Decision adapter

`phase5_v1` is the retained failed control: its train target used fixed 63-session
returns while validation used simulated A2 exits. Do not resume or overwrite it.
The corrected deeper experiment is `phase5_v2`.

Run all available fold/seed replicas. This builds causal decision outcomes,
trains the 128-to-32 adapter, and backtests the adapted retrieval geometry with
the unchanged A2 policy.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_phase5_decision_alignment.py --config configs\phase5_decision_alignment_v2.yaml --run-id phase5_v2
```

Review:

```text
reports/phase5/phase5_v2/adapter_summary.csv
reports/phase5/phase5_v2/adapter_promotion.json
```

Do not run Phase 5C unless `adapter_promotion.json` says
`eligible_for_alignment` and the declared OOF Sharpe/drawdown gates are met.

### Seed-consensus diagnostic

The rejected adapter is not promoted, but its seed-consistent rally discoveries
can be tested without retraining or touching latent coordinates. This bounded
experiment compares median evidence, majority entry, causal persistent exit, and
upper-consensus persistent exit on validation caches only:

```powershell
python scripts\run_phase5_consensus_policy.py --config configs\phase5_consensus_policy.yaml --run-id phase5_consensus_v1
```

Review `leaderboard.csv`, `selection.json`, and each fold's
`trade_explanations.csv`. Test and holdout remain sealed.

## 2. One gated alignment pass

Run this only for the selected fold/seed. It initializes static memory from the
legacy runtime state, unfreezes the final transformer/patch blocks and memory,
and performs four PCGrad epochs.

```powershell
python scripts\run_phase5_transformer_alignment.py `
  --base-checkpoint reports\phase4c\phase4c_v1\fold_01\seed_7\model\final_model.pt `
  --base-config reports\phase4c\phase4c_v1\fold_01\seed_7\config_snapshot.yaml `
  --adapter-checkpoint reports\phase5\phase5_v2\fold_01\seed_7\decision_adapter.pt `
  --output reports\phase5\phase5_v2\alignment\fold_01_seed_7.pt
```

Reject the result if its adjacent JSON reports more than five percent future
MAE degradation. Representation weights are frozen permanently after this step.

## 3. Shared offline-policy data

```powershell
python scripts\export_phase5_offline_datasets.py `
  --phase5-run reports\phase5\phase5_v2 `
  --output reports\phase5\policy_datasets
```

The contextual bandit can run in the core environment:

```powershell
python scripts\train_phase5_offline_policy.py `
  --dataset reports\phase5\policy_datasets\fold_01\seed_7\offline_policy_dataset.npz `
  --algorithm bandit --steps 100000 `
  --output reports\phase5\policies\fold_01\seed_7\bandit.pt
```

Create a separate environment for CQL, IQL, and TD3+BC so d3rlpy cannot replace
the encoder's pinned CUDA PyTorch:

```powershell
py -3.11 -m venv .venv-phase5-rl
.\.venv-phase5-rl\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-phase5-rl.txt
python scripts\train_phase5_offline_policy.py --dataset reports\phase5\policy_datasets\fold_01\seed_7\offline_policy_dataset.npz --algorithm cql --steps 100000 --output reports\phase5\policies\fold_01\seed_7\cql.d3
python scripts\train_phase5_offline_policy.py --dataset reports\phase5\policy_datasets\fold_01\seed_7\offline_policy_dataset.npz --algorithm iql --steps 100000 --output reports\phase5\policies\fold_01\seed_7\iql.d3
python scripts\train_phase5_offline_policy.py --dataset reports\phase5\policy_datasets\fold_01\seed_7\offline_policy_dataset.npz --algorithm td3bc --steps 100000 --output reports\phase5\policies\fold_01\seed_7\td3bc.d3
```

Evaluate each model with `scripts/evaluate_phase5_offline_policy.py`. It merges
the decision embedding into cached A2 signals and delegates all position holds,
stops, and exits to `run_long_only_backtest`.

## 4. International development evaluation

Generate local-listing download commands, download, audit, and freeze the 48-job
within/pairwise/leave-one-market-out matrix:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\run_phase5_international.py --stage prepare --run-id phase5_markets_v1
python scripts\run_phase5_international.py --stage download --run-id phase5_markets_v1
python scripts\run_phase5_international.py --stage audit --run-id phase5_markets_v1
python scripts\run_phase5_international.py --stage training-configs --run-id phase5_markets_v1
python scripts\run_phase5_international.py --stage manifest --run-id phase5_markets_v1
```

`market_training_commands.txt` contains the identical architecture runs for
three seeds in each source market. Zero-shot evaluation must use only the source
market standardizer. Do not fit normalization, thresholds, adapters, or policies
on the target market.

For each source model, train its adapter from its train/validation latent exports,
then export each target using that source checkpoint and adapter:

```powershell
python scripts\train_phase5_adapter_from_latents.py --train-latents <source-train-latents> --val-latents <source-val-latents> --precomputed-glob "data\international\India\*.parquet" --output <source-adapter-dir> --seed 7
python scripts\export_phase5_transfer_latents.py --encoder-checkpoint <source-model> --adapter-checkpoint <source-adapter> --target-glob "data/international/UK/*.parquet" --start 2024-01-01 --end 2024-12-31 --output <source-to-UK-decisions.parquet>
python scripts\evaluate_phase5_transfer.py --source-memory <source-train-decisions.parquet> --target-decisions <source-to-UK-decisions.parquet> --target-precomputed-glob "data/international/UK/*.parquet" --output <source-to-UK-report-dir>
```

## 5. Final confirmation

After selecting one frozen representation and one frozen policy, create the lock:

```powershell
python scripts\run_phase5_international.py --stage lock --run-id phase5_markets_v1 `
  --artifacts <promoted-model> <promoted-policy> configs\phase5_decision_alignment.yaml
```

Run 2025 exactly once across all six markets, then close the lock with the result
files. Passing requires pooled Sharpe at least 2.0, maximum drawdown at most 20%,
positive local excess in at least five markets, positive off-diagonal transfer in
at least 60% of pairs, and no market contributing over 35% of total profit.
