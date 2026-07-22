from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest import load_evaluation_config, run_market_memory_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "market_memory_backtest.yaml"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_evaluation_config(args.config)
    out_dir = run_market_memory_evaluation(config, ROOT, dry_run=args.dry_run, run_id=args.run_id)
    logging.getLogger("market_memory_backtest").info("Wrote market-memory backtest to %s", out_dir)


if __name__ == "__main__":
    main()
