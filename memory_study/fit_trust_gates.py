"""
Refit Selective Trust Gates on H2 2021 for MLP and Transformer Backbones.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from memory_study.mixing_gate import fit_trust_gate, load_trust_gate
from memory_study.retrieval import batch_retrieve_m2

PROJECT_ROOT = Path("c:/Users/rohil/OneDrive/Desktop/Core-RL-Agent")
CACHE_DIR = PROJECT_ROOT / "research_runs" / "memory_study" / "cache"
CONFIG_PATH = PROJECT_ROOT / "memory_study" / "configs" / "final_comparison.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_all_trust_gates(dev_start: str = "2021-07-01", dev_end: str = "2021-12-31"):
    print("=" * 80)
    print(f"FITTING SELECTIVE TRUST GATES ON H2 2021 ({dev_start} to {dev_end})")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    import yaml
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    backbone_seeds = cfg["backbones"]["seeds"]
    k_val = cfg["retrieval"]["k"]

    # 1. Load caches
    query_meta = pd.read_parquet(CACHE_DIR / "query_meta.parquet")
    outcomes_df = pd.read_parquet(CACHE_DIR / "outcomes.parquet")
    outcomes_map = dict(zip(outcomes_df["query_id"], outcomes_df["realized_return_63"]))

    mem_meta = pd.read_parquet(CACHE_DIR / "memory_meta.parquet")
    mem_tickers = mem_meta["ticker"].values
    mem_sessions = mem_meta["ticker_session_index"].values.astype(int)
    mem_returns = mem_meta["return_63"].values.astype(np.float32)

    mem_raw_arr = np.load(CACHE_DIR / "memory_raw_windows.npy")
    mem_windows_gpu = torch.tensor(mem_raw_arr, dtype=torch.float32, device=DEVICE)
    mem_sq_norms_gpu = torch.sum(mem_windows_gpu ** 2, dim=1)

    query_raw_windows = np.load(CACHE_DIR / "query_raw_windows.npy")

    # Filter H2 2021
    h2_mask = (query_meta["decision_timestamp"] >= dev_start) & (query_meta["decision_timestamp"] <= dev_end)
    h2_queries = query_meta[h2_mask].copy()
    h2_indices = h2_queries["array_index"].values
    h2_tickers = h2_queries["ticker"].values
    h2_qids = h2_queries["query_id"].values
    h2_windows = query_raw_windows[h2_indices]
    y_h2 = np.array([outcomes_map.get(qid, 0.0) for qid in h2_qids], dtype=np.float32)

    print(f"[+] Retrieving M2 memory predictions for {len(h2_queries)} queries in H2 2021...")
    h2_mem_preds, _ = batch_retrieve_m2(
        query_windows=h2_windows,
        mem_windows_gpu=mem_windows_gpu,
        mem_sq_norms_gpu=mem_sq_norms_gpu,
        query_tickers=h2_tickers,
        mem_tickers=mem_tickers,
        mem_sessions=mem_sessions,
        mem_returns=mem_returns,
        k=k_val,
        batch_size=500,
        device=DEVICE,
    )

    query_preds = {}
    for bb in ["transformer", "mlp"]:
        query_preds[bb] = {}
        for s in backbone_seeds:
            query_preds[bb][s] = np.load(CACHE_DIR / f"query_preds_{bb}_seed_{s}.npy")

    # Fit gates
    for bb in ["transformer", "mlp"]:
        print(f"\nFitting trust gates for {bb.upper()}...")
        for s in backbone_seeds:
            b_preds_h2 = query_preds[bb][s][h2_indices]
            vols_h2 = np.ones_like(b_preds_h2) * 0.015
            fit_trust_gate(
                backbone=bb,
                backbone_seed=s,
                base_preds_h2=b_preds_h2,
                mem_preds_h2=h2_mem_preds,
                vols_h2=vols_h2,
                y_true_h2=y_h2,
                epochs=50,
                lr=0.05,
                device=DEVICE,
            )
            mod, u_m, u_s, best_lam = load_trust_gate(bb, s, device=DEVICE)
            print(f"   [+] Loaded and verified trust gate for {bb.upper()} seed {s} (Grid lambda: {best_lam})")

    print("\n[+] All trust gates successfully fitted and saved to models/")


if __name__ == "__main__":
    fit_all_trust_gates()
