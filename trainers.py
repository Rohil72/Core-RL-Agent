"""
Train a candidate scorer model from precomputed parquet files in data/precomputed.

Usage (examples):
  python trainers.py --data-dir data/precomputed --epochs 6 --batch-size 256 --device cuda

The script expects precomputed parquet files produced by scripts/precompute_ground_truth.py
which include oracle_cycle_id (>=0 means in-cycle). Candidates are generated from the price
series and labeled positive if any of their span overlaps any oracle cycle rows.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.cycle.candidate_builder import generate_candidates, candidate_to_feature_vector
from src.models.candidate_scorer import CandidateScorer


class CandidateDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.float32)

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int):
        return self.xs[idx], self.ys[idx]


def collect_training_data(data_dir: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    xs: List[np.ndarray] = []
    ys: List[float] = []
    count = 0
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        prices = df["close"].astype(float)
        # generate candidates with permissive params
        context, candidates = generate_candidates(prices, df, min_duration_days=10, max_duration_days=42, min_return=0.0)
        if not candidates:
            continue
        oracle_ids = df.get("oracle_cycle_id", pd.Series([-1]*len(df), index=df.index)).to_numpy()
        for c in candidates:
            vec = candidate_to_feature_vector(c, context, df)
            # label positive if any position within the candidate span has oracle_cycle_id >=0
            span_ids = oracle_ids[c.start_idx : c.end_idx + 1]
            positive = float((span_ids >= 0).any())
            xs.append(vec)
            ys.append(positive)
        count += 1
        if limit and count >= limit:
            break
    if not xs:
        return np.zeros((0,13), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(xs), np.array(ys, dtype=np.float32)


def train(args):
    xs, ys = collect_training_data(args.data_dir, limit=args.limit)
    if len(xs) == 0:
        print("No training data found in", args.data_dir)
        return

    ds = CandidateDataset(xs, ys)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = CandidateScorer(input_dim=xs.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        pos = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            total += xb.size(0)
            pos += int(yb.sum().item())
        avg_loss = total_loss / max(1, total)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} positive={pos}/{total}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "candidate_scorer.pt")
    torch.save(model.state_dict(), out_path)
    print("Saved model to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/precomputed")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    train(args)
