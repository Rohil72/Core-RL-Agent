from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch

from src.cycle.candidate_builder import generate_candidates, candidate_to_feature_vector
from src.models.candidate_scorer import CandidateScorer
from src.cycle.cycle_detector import Cycle


def score_candidates_with_model(candidates: List[Cycle], context, feature_frame: pd.DataFrame | None, model_path: str, device: str | torch.device = "cpu"):
    device = torch.device(device)
    # Load state dict first so we can infer architecture (hidden size) if needed
    state = torch.load(model_path, map_location=device)
    # attempt to find the first linear weight in the saved state
    w0 = None
    for k, v in state.items():
        if k.endswith("net.0.weight"):
            w0 = v
            break
    if w0 is None:
        # fallback: pick the first weight tensor
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                w0 = v
                break
    if w0 is None:
        raise RuntimeError("Could not infer model input/hidden dims from checkpoint")

    in_dim = int(w0.shape[1])
    hidden = int(w0.shape[0])

    model = CandidateScorer(input_dim=in_dim, hidden=hidden)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    feats = [candidate_to_feature_vector(c, context, feature_frame) for c in candidates]
    if not feats:
        return []
    xs = np.stack(feats)
    with torch.no_grad():
        x = torch.from_numpy(xs).to(device)
        scores = model(x).cpu().numpy()
    scored = list(zip(candidates, scores))
    return scored


def select_non_overlapping_greedy(scored_candidates: List[tuple[Cycle, float]]) -> List[Cycle]:
    # sort by score desc
    sorted_candidates = sorted(scored_candidates, key=lambda x: float(x[1]), reverse=True)
    selected: List[Cycle] = []
    occupied = set()
    for cand, score in sorted_candidates:
        span = range(cand.start_idx, cand.end_idx + 1)
        if any(i in occupied for i in span):
            continue
        selected.append(cand)
        for i in span:
            occupied.add(i)
    # return sorted by start_idx
    return sorted(selected, key=lambda c: c.start_idx)


def detect_cycles_learned(
    prices: pd.Series,
    feature_frame: pd.DataFrame | None = None,
    model_path: str | None = None,
    device: str | torch.device = "cpu",
    # candidate generation params
    min_duration_days: int = 21,
    max_duration_days: int = 252,
    min_return: float = 0.0,
    soft_pullback_limit: float = 0.05,
    hard_pullback_limit: float = 0.12,
    volatility_window: int = 21,
    volatility_multiplier: float = 2.0,
) -> List[Cycle]:
    """Generate candidates then score with a learned model (if provided), otherwise fall back to heuristic detect_cycles."""
    context, candidates = generate_candidates(
        prices=prices,
        feature_frame=feature_frame,
        min_duration_days=min_duration_days,
        max_duration_days=max_duration_days,
        min_return=min_return,
        soft_pullback_limit=soft_pullback_limit,
        hard_pullback_limit=hard_pullback_limit,
        volatility_window=volatility_window,
        volatility_multiplier=volatility_multiplier,
    )

    if not candidates:
        return []

    if model_path is None:
        # no model provided: return candidates filtered by a modest heuristic score
        # keep highest-scoring candidate per start (by cycle_score)
        best_per_start = {}
        for c in candidates:
            key = c.start_idx
            if key not in best_per_start or c.cycle_score > best_per_start[key].cycle_score:
                best_per_start[key] = c
        return sorted(best_per_start.values(), key=lambda x: x.start_idx)

    scored = score_candidates_with_model(candidates, context, feature_frame, model_path, device=device)
    # scored is list of (candidate, score)
    selected = select_non_overlapping_greedy(scored)
    return selected
