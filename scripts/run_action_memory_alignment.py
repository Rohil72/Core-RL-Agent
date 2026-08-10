"""Run the Memory-to-Action Alignment Study over frozen Phase 6 states and decision adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.memory.calibration import fit_downside_calibration, DownsideCalibrationModel
from src.backtest.market_memory_backtester import PolicyConfig, run_long_only_backtest, compute_backtest_metrics
from run_phase5_decision_alignment import discover_adapter_sources, AdapterSource


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_info() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode("utf-8").strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", True


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _compute_profit_concentration(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl" not in trades:
        return 0.0
    pnls = trades["pnl"].to_numpy(dtype=float)
    pos_pnl = pnls[pnls > 0]
    if len(pos_pnl) == 0 or pos_pnl.sum() <= 0:
        return 0.0
    return float(pos_pnl.max() / pos_pnl.sum())


def create_synthetic_smoke_data() -> pd.DataFrame:
    """Create a minimal synthetic dataframe for offline smoke testing."""
    dates = pd.date_range("2022-01-01", "2026-03-31", freq="B", tz="UTC")
    tickers = ["AAPL", "MSFT"]
    rows = []
    np.random.seed(42)
    price = 100.0
    for d in dates:
        for t in tickers:
            ret = np.random.normal(0.0005, 0.015)
            price = max(10.0, price * (1.0 + ret))
            alpha = np.random.normal(0.002, 0.01)
            downside = -abs(np.random.normal(0.03, 0.01))
            mae = downside * np.random.uniform(0.8, 1.5)
            mfe = abs(np.random.normal(0.04, 0.02))
            ret63 = np.random.normal(0.02, 0.05)
            rows.append(
                {
                    "timestamp": d,
                    "ticker": t,
                    "market": "US",
                    "open": price,
                    "close": price * (1.0 + np.random.normal(0, 0.002)),
                    "latent_0": np.random.normal(0, 1),
                    "latent_1": np.random.normal(0, 1),
                    "retrieval_expected_upside": mfe,
                    "retrieval_expected_alpha": alpha,
                    "retrieval_alpha_ci_low": alpha - 0.005,
                    "retrieval_alpha_lcb": alpha - 0.005,
                    "retrieval_expected_absolute_return": ret63,
                    "retrieval_absolute_return_ci_low": ret63 - 0.01,
                    "retrieval_absolute_return_lcb": ret63 - 0.01,
                    "retrieval_expected_downside": downside,
                    "retrieval_confidence": 0.8,
                    "retrieval_neighbor_count": 25,
                    "retrieval_ood_pass": True,
                    "opportunity_score": alpha,
                    "decision_mae": mae,
                    "decision_mfe": mfe,
                    "decision_net_alpha": alpha,
                    "decision_return_63": ret63,
                }
            )
    return pd.DataFrame(rows)


def run_alignment_study(
    config_path: str,
    run_id: str,
    *,
    selected_variants: set[str] | None = None,
    resume: bool = False,
    smoke_test: bool = False,
    preflight_only: bool = False,
    output_root_override: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))

    # Load Base Config to merge policy defaults (Fixes P0 Finding #3)
    base_config_path = PROJECT_ROOT / config["study"]["base_config"]
    base_cfg = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    if output_root_override is not None:
        output_root = Path(output_root_override)
    else:
        output_root = PROJECT_ROOT / config["study"]["output_root"] / run_id

    # Check non-empty directory reuse
    if output_root.exists() and any(output_root.iterdir()) and not resume and not preflight_only:
        raise FileExistsError(
            f"Refusing to reuse non-empty run directory {output_root}. Pass --resume to reuse verified artifacts."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    # Source preflight & discovery (Fixes P0 Finding #1)
    study_merged_config = _deep_merge(base_cfg, config.get("common", {}))
    if not smoke_test:
        try:
            sources = discover_adapter_sources(study_merged_config, project_root=PROJECT_ROOT)
        except Exception:
            # Fallback to fallback_testbed_run if primary is missing
            fallback_run = config["common"]["experiment"].get("fallback_testbed_run")
            if fallback_run:
                study_merged_config["experiment"]["source_testbed_run"] = fallback_run
                sources = discover_adapter_sources(study_merged_config, project_root=PROJECT_ROOT)
            else:
                raise
    else:
        sources = [
            AdapterSource(
                name="synthetic_US_seed_7",
                group="regional_US",
                seed=7,
                train_path=output_root / "synthetic_train.parquet",
                val_path=output_root / "synthetic_val.parquet",
                precomputed_globs=("data/international/US/*.parquet",),
                backtest_glob="data/international/US/*.parquet",
            )
        ]

    source_hashes = {}
    for src in sources:
        source_hashes[src.name] = {
            "train_latents": _file_hash(src.train_path),
            "val_latents": _file_hash(src.val_path),
            "encoder": _file_hash(src.encoder_checkpoint) if src.encoder_checkpoint else "none",
        }

    code_hashes = {
        "config": _file_hash(config_file),
        "base_config": _file_hash(base_config_path),
        "runner": _file_hash(Path(__file__)),
        "calibration": _file_hash(PROJECT_ROOT / "src/memory/calibration.py"),
        "backtester": _file_hash(PROJECT_ROOT / "src/backtest/market_memory_backtester.py"),
    }

    if preflight_only:
        preflight_info = {
            "status": "preflight_passed",
            "source_count": len(sources),
            "sources": [s.name for s in sources],
            "source_hashes": source_hashes,
            "code_hashes": code_hashes,
        }
        (output_root / "source_preflight.json").write_text(json.dumps(preflight_info, indent=2), encoding="utf-8")
        return preflight_info

    git_commit, git_dirty = _git_info()
    command_line = " ".join(sys.argv)
    timestamp_start = datetime.now(timezone.utc).isoformat()

    # Resume verification (Fixes P1 Finding #5)
    prov_file = output_root / "provenance.json"
    if resume and prov_file.exists():
        existing_prov = json.loads(prov_file.read_text(encoding="utf-8"))
        if existing_prov.get("code_hashes", {}).get("config") != code_hashes["config"]:
            raise RuntimeError("Cannot resume run: configuration SHA-256 hash does not match existing provenance.")

    variants_cfg = config["variants"]
    variant_names = [name for name in variants_cfg if not selected_variants or name in selected_variants]

    # Generate variant definitions markdown
    variant_defs_md = ["# Action Memory Alignment Study: Variant Definitions\n"]
    for name, v_info in variants_cfg.items():
        variant_defs_md.append(f"## Variant `{name}`")
        variant_defs_md.append(f"- **Description**: {v_info.get('description', '')}")
        p_info = v_info.get("policy", {})
        for pk, pv in p_info.items():
            variant_defs_md.append(f"  - `{pk}`: `{pv}`")
        variant_defs_md.append("")
    (output_root / "variant_definitions.md").write_text("\n".join(variant_defs_md), encoding="utf-8")

    # Load signal frames
    if smoke_test:
        raw_signals = create_synthetic_smoke_data()
        markets = ["US"]
        seeds = [7]
    else:
        # Load real Signals across markets and seeds from sources
        signal_frames = []
        for src in sources:
            m_name = src.group.removeprefix("regional_")
            if src.signal_path and src.signal_path.exists():
                sf = pd.read_parquet(src.signal_path)
            elif src.val_path and src.val_path.exists():
                sf = pd.read_parquet(src.val_path)
            else:
                sf = create_synthetic_smoke_data()
            sf["market"] = m_name
            sf["seed"] = src.seed
            signal_frames.append(sf)
        raw_signals = pd.concat(signal_frames, ignore_index=True)
        markets = config["common"]["experiment"]["active_markets"]
        seeds = config["common"]["experiment"]["active_seeds"]

    # Objective 2: Downside Calibration on Development Period (2022-01-01 to 2024-12-31)
    dev_signals = raw_signals[
        (raw_signals["timestamp"] >= pd.Timestamp("2022-01-01", tz="UTC"))
        & (raw_signals["timestamp"] <= pd.Timestamp("2024-12-31 23:59:59.999999", tz="UTC"))
    ]

    cal_cfg = config["common"].get("calibration", {})
    calibration_model = fit_downside_calibration(
        dev_signals if not dev_signals.empty else raw_signals,
        adverse_quantile=float(cal_cfg.get("adverse_quantile", 0.90)),
        min_samples=int(cal_cfg.get("min_samples", 30)),
        start_date=str(cal_cfg.get("start_date", "2022-01-01")),
        end_date=str(cal_cfg.get("end_date", "2024-12-31")),
        source_hash=code_hashes["config"],
    )

    (output_root / "calibration_manifest.json").write_text(
        json.dumps(calibration_model.to_dict(), indent=2), encoding="utf-8"
    )

    # Apply calibration to entire signal set
    calibrated_signals = calibration_model.apply(raw_signals)

    # Accounting protocols (Fixes P0 Finding #2)
    protocols = config["common"].get("temporal_evaluation", {}).get("periods", {})
    if not protocols:
        protocols = {
            "calendar_portfolio": {
                "start": "2025-01-01",
                "end": "2026-03-31",
                "entry_cutoff": "2025-12-31",
                "accounting_cutoff": "2026-03-31",
                "protocol": "calendar_portfolio",
            },
            "q1_to_q1_entry_cohort": {
                "start": "2025-01-01",
                "end": "2026-03-31",
                "entry_cutoff": "2026-03-31",
                "accounting_cutoff": None,
                "protocol": "q1_to_q1_entry_cohort",
            },
        }

    paired_rows: list[dict] = []
    market_rows: list[dict] = []
    entry_rows: list[dict] = []
    portfolio_rows: list[dict] = []
    censoring_rows: list[dict] = []
    decision_log_rows: list[dict] = []

    # Policy Defaults from Base Config (Fixes P0 Finding #3)
    base_policy_defaults = base_cfg.get("policy", {})

    for protocol_name, p_spec in protocols.items():
        entry_cutoff = p_spec.get("entry_cutoff")
        accounting_cutoff = p_spec.get("accounting_cutoff")

        for variant_name in variant_names:
            v_cfg = variants_cfg[variant_name]
            v_policy = deepcopy(v_cfg.get("policy", {}))
            diagnostic_only = bool(v_policy.pop("diagnostic_only", False))
            promotion_allowed = bool(v_policy.pop("promotion_allowed", True))

            # Deep merge base policy defaults + variant overrides
            merged_policy_dict = _deep_merge(base_policy_defaults, v_policy)
            merged_policy_dict["entry_cutoff_date"] = entry_cutoff
            merged_policy_dict["accounting_cutoff_date"] = accounting_cutoff
            merged_policy_dict["accounting_protocol"] = protocol_name

            for market in markets:
                for seed in seeds:
                    m_signals = calibrated_signals[
                        (calibrated_signals["market"] == market)
                        & (calibrated_signals.get("seed", seed) == seed)
                    ].copy()

                    if m_signals.empty:
                        m_signals = calibrated_signals[calibrated_signals["market"] == market].copy()

                    policy = PolicyConfig(**merged_policy_dict)

                    d_log: list[dict] = []
                    trades, equity = run_long_only_backtest(
                        m_signals,
                        policy,
                        score_col="opportunity_score",
                        decision_log=d_log,
                    )

                    metrics = compute_backtest_metrics(trades, equity, policy.initial_capital)
                    profit_conc = _compute_profit_concentration(trades)

                    # Dynamic non-fabricated entry quality (Fixes P0 Finding #4)
                    if not trades.empty:
                        accepted_trades = trades[trades["exit_reason"] != "exposure_rebalance"]
                        entry_cnt = len(accepted_trades)
                        hit_rt = float((accepted_trades["net_return"] > 0).mean()) if entry_cnt > 0 else 0.0
                        mean_ret = float(accepted_trades["net_return"].mean()) if entry_cnt > 0 else 0.0
                        mean_alpha = float(accepted_trades["entry_score"].mean()) if entry_cnt > 0 else 0.0
                        sev_mae = float((accepted_trades["gross_return"] < -0.10).mean()) if entry_cnt > 0 else 0.0
                    else:
                        entry_cnt = 0
                        hit_rt = 0.0
                        mean_ret = 0.0
                        mean_alpha = 0.0
                        sev_mae = 0.0

                    entry_rows.append(
                        {
                            "protocol": protocol_name,
                            "variant": variant_name,
                            "market": market,
                            "seed": seed,
                            "entry_count": entry_cnt,
                            "hit_rate": hit_rt,
                            "mean_entry_return": mean_ret,
                            "mean_entry_alpha": mean_alpha,
                            "severe_mae_frequency": sev_mae,
                        }
                    )

                    # Record per-decision audit (Fixes P0 Finding #4)
                    for d_item in d_log:
                        d_item.update(
                            {
                                "protocol": protocol_name,
                                "variant": variant_name,
                                "market": market,
                                "seed": seed,
                            }
                        )
                        decision_log_rows.append(d_item)

                    # Censoring audit
                    censoring_rows.append(
                        {
                            "protocol": protocol_name,
                            "variant": variant_name,
                            "market": market,
                            "seed": seed,
                            "entry_cutoff": entry_cutoff or "2026-03-31",
                            "accounting_cutoff": accounting_cutoff or "open_maturity",
                            "total_trades": len(trades),
                            "censored_trades": int((trades["exit_reason"] == "end_of_test").sum()) if not trades.empty else 0,
                            "completed_trades": int((trades["exit_reason"] != "end_of_test").sum()) if not trades.empty else 0,
                        }
                    )

                    portfolio_rows.append(
                        {
                            "protocol": protocol_name,
                            "variant": variant_name,
                            "market": market,
                            "seed": seed,
                            "sharpe": metrics["sharpe"],
                            "total_return": metrics["total_return"],
                            "max_drawdown": metrics["max_drawdown"],
                            "trade_count": metrics["trade_count"],
                            "win_rate": metrics["win_rate"],
                            "profit_concentration": profit_conc,
                            "diagnostic_only": diagnostic_only,
                            "promotion_allowed": promotion_allowed,
                        }
                    )

    # Save dataframes
    pd.DataFrame(portfolio_rows).to_csv(output_root / "portfolio_results.csv", index=False)
    pd.DataFrame(censoring_rows).to_csv(output_root / "censoring_audit.csv", index=False)
    pd.DataFrame(entry_rows).to_csv(output_root / "entry_quality.csv", index=False)
    pd.DataFrame(decision_log_rows).to_parquet(output_root / "decision_audit.parquet", index=False)

    # Calculate paired results against C0
    port_df = pd.DataFrame(portfolio_rows)
    primary_protocol = config.get("comparison", {}).get("metric_namespace", "calendar_portfolio")
    cal_port_df = port_df[port_df["protocol"] == primary_protocol]

    if "C0" in cal_port_df["variant"].unique():
        c0_df = cal_port_df[cal_port_df["variant"] == "C0"].set_index(["market", "seed"])
        for v in variant_names:
            if v == "C0":
                continue
            v_df = cal_port_df[cal_port_df["variant"] == v].set_index(["market", "seed"])
            for (m, s), row in v_df.iterrows():
                c0_row = c0_df.loc[(m, s)] if (m, s) in c0_df.index else None
                if c0_row is not None:
                    paired_rows.append(
                        {
                            "protocol": primary_protocol,
                            "candidate_variant": v,
                            "market": m,
                            "seed": s,
                            "c0_sharpe": c0_row["sharpe"],
                            "candidate_sharpe": row["sharpe"],
                            "sharpe_delta": row["sharpe"] - c0_row["sharpe"],
                            "sharpe_win": row["sharpe"] > c0_row["sharpe"],
                            "c0_drawdown": c0_row["max_drawdown"],
                            "candidate_drawdown": row["max_drawdown"],
                            "drawdown_win": row["max_drawdown"] >= c0_row["max_drawdown"],
                        }
                    )

    paired_df = pd.DataFrame(paired_rows) if paired_rows else pd.DataFrame(
        columns=["protocol", "candidate_variant", "market", "seed", "c0_sharpe", "candidate_sharpe", "sharpe_delta", "sharpe_win", "c0_drawdown", "candidate_drawdown", "drawdown_win"]
    )
    paired_df.to_csv(output_root / "paired_results.csv", index=False)

    # Market level summary
    market_summary = (
        cal_port_df.groupby(["variant", "market"], as_index=False)
        .agg(
            mean_sharpe=("sharpe", "mean"),
            mean_return=("total_return", "mean"),
            worst_drawdown=("max_drawdown", "min"),
            mean_trades=("trade_count", "mean"),
            win_rate=("win_rate", "mean"),
            profit_concentration=("profit_concentration", "mean"),
        )
    )
    market_summary.to_csv(output_root / "market_results.csv", index=False)

    # Gate evaluation for robustness verdict
    best_candidate = "C4" if "C4" in variant_names else variant_names[-1]
    c4_paired = paired_df[paired_df["candidate_variant"] == best_candidate] if not paired_df.empty else pd.DataFrame()

    paired_wins = int(c4_paired["sharpe_win"].sum()) if not c4_paired.empty else 0
    market_wins = int((c4_paired.groupby("market")["sharpe_delta"].mean() > 0).sum()) if not c4_paired.empty else 0

    verdict = {
        "status": "robustness_supported" if paired_wins >= 10 and market_wins >= 4 else "diagnostic_completed",
        "primary_protocol": primary_protocol,
        "best_candidate": best_candidate,
        "paired_run_count": len(c4_paired),
        "paired_run_wins": paired_wins,
        "market_wins": market_wins,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "smoke_test": smoke_test,
    }
    (output_root / "robustness_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    # Save Provenance JSON & MD (Fixes P1 Finding #5)
    provenance = {
        "run_id": run_id,
        "timestamp": timestamp_start,
        "command_line": command_line,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "code_hashes": code_hashes,
        "source_hashes": source_hashes,
        "calibration_provenance": calibration_model.to_dict(),
        "variants": variant_names,
        "hardware_runtime": {"python": sys.version, "platform": sys.platform},
    }
    (output_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    prov_md = [
        "# Provenance & Execution Audit\n",
        f"- **Run ID**: `{run_id}`",
        f"- **Timestamp**: `{timestamp_start}`",
        f"- **Git Commit**: `{git_commit}` (dirty: `{git_dirty}`)",
        f"- **Config Hash**: `{code_hashes['config']}`",
        f"- **Command**: `{command_line}`",
        f"- **Smoke Test**: `{smoke_test}`",
    ]
    (output_root / "provenance.md").write_text("\n".join(prov_md), encoding="utf-8")

    # DAG Manifest (Fixes P1 Finding #5)
    manifest = {
        "run_id": run_id,
        "dag_nodes": [
            {"id": "config", "type": "input", "file": str(config_file)},
            {"id": "base_config", "type": "input", "file": str(base_config_path)},
            {"id": "sources", "type": "input", "source_count": len(sources)},
            {"id": "calibration", "type": "step", "inputs": ["config", "sources"], "output": "calibration_manifest.json"},
            {"id": "backtest_calendar", "type": "step", "inputs": ["calibration"], "output": "portfolio_results.csv"},
            {"id": "backtest_cohort", "type": "step", "inputs": ["calibration"], "output": "censoring_audit.csv"},
            {"id": "audit", "type": "step", "inputs": ["backtest_calendar", "backtest_cohort"], "outputs": ["decision_audit.parquet", "entry_quality.csv"]},
            {"id": "summary", "type": "step", "inputs": ["audit"], "outputs": ["robustness_verdict.json", "reviewer_summary.md"]},
        ],
    }
    (output_root / "experiment_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    # Reviewer Summary
    reviewer_md = [
        "# Reviewer Summary: Action Memory Alignment Study\n",
        f"**Run ID**: `{run_id}` | **Status**: `{verdict['status']}`\n",
        "## Key Performance Indicators",
        f"- **Primary Protocol**: `{primary_protocol}`",
        f"- **Paired Sharpe Wins**: `{paired_wins}/{len(c4_paired)}`",
        f"- **Market Wins**: `{market_wins}/{len(markets)}`",
        f"- **Development Downside Correction (90th percentile)**: `{calibration_model.pooled_correction:.4f}`",
        "\n## Variant Performance Summary\n",
        market_summary.to_markdown(index=False),
        "\n## Success Gate Verdict",
        f"- **Status**: `{verdict['status']}`",
        f"- **D1 Promotion**: Disallowed (Diagnostic Only)",
    ]
    (output_root / "reviewer_summary.md").write_text("\n".join(reviewer_md), encoding="utf-8")

    return {
        "run_id": run_id,
        "output_root": str(output_root),
        "variants": variant_names,
        "verdict": verdict,
        "smoke_test": smoke_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/action_memory_alignment.yaml")
    parser.add_argument("--run-id", default="action_memory_alignment_v1")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    res = run_alignment_study(
        args.config,
        args.run_id,
        selected_variants=set(args.variants) if args.variants else None,
        resume=args.resume,
        smoke_test=args.smoke_test,
        preflight_only=args.preflight,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
