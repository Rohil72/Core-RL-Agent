"""Train one Phase 5 policy in either the core or isolated d3rlpy environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.offline_policy import train_contextual_bandit  # noqa: E402


def _d3rlpy_train(algorithm: str, data: dict[str, np.ndarray], output: Path, steps: int, device: str) -> None:
    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError("Install requirements-phase5-rl.txt in the isolated .venv-phase5-rl environment.") from exc
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 1:
        raise ValueError(f"Expected continuous exposure actions with shape (N, 1), got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError("Offline-policy actions contain non-finite values.")
    if np.any((actions < 0.0) | (actions > 1.0)):
        raise ValueError("Offline-policy exposure actions must lie in [0, 1].")
    dataset = d3rlpy.dataset.MDPDataset(
        observations=data["observations"],
        actions=actions,
        rewards=data["rewards"],
        terminals=data["terminals"],
        action_space=d3rlpy.constants.ActionSpace.CONTINUOUS,
        action_size=1,
    )
    configs = {
        "cql": d3rlpy.algos.CQLConfig,
        "iql": d3rlpy.algos.IQLConfig,
        "td3bc": d3rlpy.algos.TD3PlusBCConfig,
    }
    if algorithm not in configs:
        raise ValueError(f"Unsupported d3rlpy policy: {algorithm}")
    use_gpu = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    learner = configs[algorithm](batch_size=256).create(device="cuda:0" if use_gpu else "cpu")
    learner.fit(dataset, n_steps=steps, experiment_name=f"phase5_{algorithm}", with_timestamp=False)
    learner.save(str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", choices=("bandit", "cql", "iql", "td3bc"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    data = dict(np.load(args.dataset, allow_pickle=False))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.algorithm == "bandit":
        model = train_contextual_bandit(data, args.seed, args.steps, device=args.device)
        torch.save({"state_dict": model.state_dict(), "observation_dim": data["observations"].shape[1]}, output)
    else:
        _d3rlpy_train(args.algorithm, data, output, args.steps, args.device)
    print(json.dumps({"algorithm": args.algorithm, "output": str(output), "steps": args.steps}, indent=2))


if __name__ == "__main__":
    main()
