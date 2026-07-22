"""Prepare, audit, and seal six-market Phase 5 transfer experiments."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.confirmation_lock import create_confirmation_lock, mark_confirmation_executed  # noqa: E402


def _load(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_market_universes(config: dict[str, Any], output: Path) -> list[Path]:
    """Materialize downloader-compatible local-listing universe files."""
    paths = []
    for market, values in config["markets"].items():
        path = output / "universes" / f"{market}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"tickers": values["tickers"]}, sort_keys=False), encoding="utf-8")
        paths.append(path)
    return paths


def prepare_data(config: dict[str, Any], output: Path, execute: bool) -> list[str]:
    """Create or execute one reproducible precompute command per local market."""
    universe_paths = write_market_universes(config, output)
    experiment = config["experiment"]
    commands = []
    for path in universe_paths:
        market = path.stem
        command = [
            sys.executable, "scripts/precompute_ground_truth.py", "--config-path", str(path),
            "--output-dir", str(PROJECT_ROOT / experiment["data_root"] / market),
            "--start", str(experiment["start"]), "--end", str(experiment["end"]),
        ]
        commands.append(subprocess.list2cmdline(command))
        if execute:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    (output / "precompute_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    return commands


def audit_data(config: dict[str, Any], output: Path) -> pd.DataFrame:
    """Fail closed on missing bars, weak coverage, or ticker/exchange mismatches."""
    experiment = config["experiment"]
    rows = []
    for market, metadata in config["markets"].items():
        data_dir = PROJECT_ROOT / experiment["data_root"] / market
        expected = {str(ticker) for ticker in metadata["tickers"]}
        for ticker in sorted(expected):
            path = data_dir / f"{ticker}.parquet"
            if not path.exists():
                rows.append({"market": market, "ticker": ticker, "status": "missing", "sessions": 0, "fundamental_coverage": 0.0})
                continue
            frame = pd.read_parquet(path)
            coverage = float(pd.to_numeric(frame.get("fund_report_available", 0.0), errors="coerce").fillna(0.0).mean())
            status = "ok" if len(frame) >= int(experiment["minimum_sessions"]) and coverage >= float(experiment["minimum_fundamental_coverage"]) else "insufficient"
            rows.append({"market": market, "ticker": ticker, "status": status, "sessions": len(frame), "fundamental_coverage": coverage})
    audit = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output / "data_audit.csv", index=False)
    summary = audit.groupby("market").agg(available=("status", lambda values: int((values == "ok").sum())), requested=("ticker", "size")).reset_index()
    summary["passes"] = summary["available"] >= int(experiment["minimum_tickers"])
    summary.to_csv(output / "market_audit_summary.csv", index=False)
    return summary


def write_transfer_manifest(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Declare every within, source-target, and leave-one-market-out job before results exist."""
    markets = list(config["markets"])
    jobs = []
    for source in markets:
        jobs.append({"mode": "within_market", "train_markets": [source], "test_market": source})
        for target in markets:
            jobs.append({"mode": "pairwise_zero_shot", "train_markets": [source], "test_market": target})
    for target in markets:
        jobs.append({"mode": "leave_one_market_out", "train_markets": [market for market in markets if market != target], "test_market": target})
    manifest = {
        "jobs": jobs,
        "job_count": len(jobs),
        "development_test_year": config["experiment"]["development_test_year"],
        "normalization": "source_only",
        "architecture_tuning_allowed": False,
        "policy_tuning_allowed": False,
    }
    (output / "transfer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_training_configs(config: dict[str, Any], output: Path, base_path: str) -> list[str]:
    """Freeze one encoder-training recipe across markets and seeds."""
    base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
    commands = []
    for market in config["markets"]:
        for seed in (7, 17, 37):
            resolved = copy.deepcopy(base)
            resolved["data"]["precomputed_dir"] = str(PROJECT_ROOT / config["experiment"]["data_root"] / market / "*.parquet")
            resolved["data"]["exclude_years"] = []
            resolved["model"]["memory_mode"] = "static_parameter"
            resolved["model"]["memory_update_rate"] = 0.0
            resolved["training"]["seed"] = seed
            resolved["training"]["model_dir"] = str(output / "models" / market / f"seed_{seed}")
            resolved["training"]["auto_resume"] = True
            resolved["training"]["checkpoint_interval_steps"] = 250
            resolved["training"]["allow_hardware_mismatch_resume"] = False
            resolved["split"].update({
                "train_years": 5, "val_years": 1, "test_years": 1,
                "step_years": 1, "selected_fold": 0,
                "ticker_holdout_fraction": 0.0, "seed": seed,
            })
            resolved["evaluation"]["reports_dir"] = str(output / "encoder_reports" / market / f"seed_{seed}")
            resolved.setdefault("latent_export", {})["output_dir"] = str(output / "latents" / market / f"seed_{seed}")
            config_path = output / "training_configs" / market / f"seed_{seed}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
            commands.append(subprocess.list2cmdline([
                sys.executable, "-m", "src.trainers.train_cycle_model", "--config-path", str(config_path), "--resume"
            ]))
    (output / "market_training_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_markets.yaml")
    parser.add_argument("--run-id", default="phase5_markets_v1")
    parser.add_argument("--stage", choices=("prepare", "download", "audit", "training-configs", "manifest", "lock", "close"), required=True)
    parser.add_argument("--base-training-config", default="configs/final_encoder_training.yaml")
    parser.add_argument("--artifacts", nargs="*", default=[])
    parser.add_argument("--results", nargs="*", default=[])
    parser.add_argument("--idempotent-close", action="store_true")
    args = parser.parse_args()
    config = _load(args.config)
    output = PROJECT_ROOT / config["experiment"]["output_root"] / args.run_id
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "download"}:
        result = {"commands": prepare_data(config, output, args.stage == "download")}
    elif args.stage == "audit":
        result = {"audit": audit_data(config, output).to_dict(orient="records")}
    elif args.stage == "training-configs":
        result = {"commands": write_training_configs(config, output, args.base_training_config)}
    elif args.stage == "manifest":
        result = write_transfer_manifest(config, output)
    elif args.stage == "lock":
        result = create_confirmation_lock(output / "confirmation_lock.json", [args.config, *args.artifacts], config["markets"], config["experiment"]["confirmation_year"])
    else:
        result = mark_confirmation_executed(
            output / "confirmation_lock.json",
            args.results,
            idempotent=args.idempotent_close,
        )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
