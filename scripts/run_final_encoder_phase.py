from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import load_evaluation_config, run_market_memory_evaluation


LOGGER = logging.getLogger("final_encoder_phase")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "final_encoder_training.yaml"))
    parser.add_argument("--memory-config", default=str(ROOT / "configs" / "market_memory_backtest.yaml"))
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--train-latents", default=None)
    parser.add_argument("--test-latents", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run-memory", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_summary = None
    latent_exports = {}
    if not args.skip_training:
        from src.trainers.train_cycle_model import train

        train_summary = train(args.train_config)
        latent_exports = train_summary.get("latent_exports", {})
        LOGGER.info("Training complete. Latent exports: %s", latent_exports)

    train_latents = args.train_latents or latent_exports.get("train")
    test_latents = args.test_latents or latent_exports.get("test")
    if not train_latents or not test_latents:
        raise ValueError(
            "Need train/test latent paths. Either run training with latent_export enabled "
            "or pass --skip-training --train-latents ... --test-latents ..."
        )

    memory_cfg = load_evaluation_config(args.memory_config)
    memory_cfg["data"]["train_latents"] = _relative_to_root(train_latents)
    memory_cfg["data"]["test_latents"] = _relative_to_root(test_latents)
    out_dir = run_market_memory_evaluation(
        memory_cfg,
        ROOT,
        dry_run=args.dry_run_memory,
        run_id=args.run_id,
    )
    summary = {
        "training": train_summary,
        "train_latents": train_latents,
        "test_latents": test_latents,
        "market_memory_run_dir": str(out_dir),
    }
    (out_dir / "final_encoder_phase_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    LOGGER.info("Final encoder phase output: %s", out_dir)


def _relative_to_root(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


if __name__ == "__main__":
    main()
