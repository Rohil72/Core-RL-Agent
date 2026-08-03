"""Evaluate conventional decoders on the same frozen states and execution policy."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.consensus_policy import (  # noqa: E402
    ConsensusSignalConfig,
    build_consensus_signals,
    consensus_row_filter,
)
from src.backtest.market_memory_backtester import PolicyConfig  # noqa: E402
from src.eval.policy_baselines import (  # noqa: E402
    neutral_baseline_policy,
    run_scored_policy,
    save_policy_result,
    write_json,
)
from src.eval.tabular_decoders import (  # noqa: E402
    DecoderSuiteConfig,
    fit_predict_frozen_decoders,
)


def _key(config: dict, market: str, seed: int) -> str:
    prefix = "global" if config["experiment"].get("source_layout") == "global_shared" else "regional"
    return f"{prefix}_{market}_seed_{seed}"


def run(config_path: str, run_id: str, market: str, period: str) -> dict:
    """Fit each baseline per seed, then evaluate a cross-seed rank ensemble."""

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = PROJECT_ROOT / config["experiment"]["output_root"] / run_id
    decoder_config = DecoderSuiteConfig(**config.get("tabular_baselines", {}).get("models", {}))
    reference_variant = config.get("tabular_baselines", {}).get(
        "reference_variant", config["variants"][0]["id"]
    )
    per_model: dict[str, dict[int, pd.DataFrame]] = {}
    audits: dict[str, object] = {}
    for seed_value in config["experiment"]["seeds"]:
        seed = int(seed_value)
        key = _key(config, market, seed)
        view_root = output / "views" / key
        training = pd.read_parquet(view_root / "static_raw.parquet")
        query = pd.read_parquet(view_root / period / "query_raw.parquet")
        predictions, audit = fit_predict_frozen_decoders(
            training,
            query,
            decoder_config,
        )
        audits[f"seed_{seed}"] = audit
        signals = pd.read_parquet(
            output
            / "retrieval"
            / reference_variant
            / key
            / period
            / "eval"
            / "signals.parquet"
        )
        signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)
        signals["ticker"] = signals["ticker"].astype(str)
        identity = query[["ticker", "timestamp"]].copy()
        identity["timestamp"] = pd.to_datetime(identity["timestamp"], utc=True)
        identity["ticker"] = identity["ticker"].astype(str)
        for name, values in predictions.items():
            scores = identity.copy()
            scores["decoder_score"] = values
            frame = signals.drop(columns=["opportunity_score"], errors="ignore").merge(
                scores,
                on=["ticker", "timestamp"],
                how="inner",
                validate="one_to_one",
            )
            frame["opportunity_score"] = frame["decoder_score"]
            frame["retrieval_expected_upside"] = frame["decoder_score"].clip(lower=0.0)
            frame["retrieval_expected_downside"] = frame["decoder_score"].clip(upper=0.0)
            frame["retrieval_confidence"] = 1.0
            frame["retrieval_ood_pass"] = True
            per_model.setdefault(name, {})[seed] = frame

    testbed = yaml.safe_load(
        (PROJECT_ROOT / config["experiment"]["source_testbed_config"]).read_text(
            encoding="utf-8"
        )
    )
    policy_values = dict(config["policy"])
    policy_values["slippage_bps"] = float(
        testbed["markets"][market]["execution_cost_bps"]
    )
    policy = neutral_baseline_policy(PolicyConfig(**policy_values))
    consensus_config = ConsensusSignalConfig(**config["consensus"])
    destination = output / "tabular_baselines" / period / market
    metrics: dict[str, object] = {}
    for name, seed_frames in sorted(per_model.items()):
        consensus = build_consensus_signals(seed_frames, policy, consensus_config)
        result = run_scored_policy(
            consensus,
            policy,
            "consensus_entry_rank",
            exit_score_col="consensus_exit_score",
            row_filter=lambda row, cfg, _score: consensus_row_filter(
                row,
                cfg,
                consensus_config.minimum_votes,
            ),
        )
        save_policy_result(destination, name, result)
        consensus.to_parquet(
            destination / "baselines" / name / "consensus_signals.parquet",
            index=False,
        )
        metrics[name] = result["metrics"]
    payload = {
        "status": "completed",
        "market": market,
        "period": period,
        "representation": "frozen_raw_encoder_latent",
        "policy": asdict(policy),
        "decoder_config": asdict(decoder_config),
        "audits": audits,
        "metrics": metrics,
    }
    write_json(destination / "summary.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--period", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.run_id, args.market, args.period), indent=2))


if __name__ == "__main__":
    main()
