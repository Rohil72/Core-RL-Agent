import argparse
import copy
import json
import logging
import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml
import numpy as np
import scipy.stats as stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trainers.train_cycle_model import train, load_config as load_cycle_config
from src.backtest.market_memory_evaluator import run_market_memory_evaluation, load_evaluation_config
from src.eval.causal_memory import attach_outcome_availability

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VARIANTS = {
    "huber_baseline_v2": {"lambda_reg": 1.00, "lambda_analogue": 0.00},
    "huber_analogue_v2": {"lambda_reg": 1.00, "lambda_analogue": 0.10},
}
SEEDS = [7, 17, 37]
MODES = {
    "validation": [
        ("mixed", {"same_ticker_mode": "allow", "exclude_query_sector": False}),
        ("cross_ticker_only", {"same_ticker_mode": "exclude", "exclude_query_sector": False}),
        ("sector_excluded", {"same_ticker_mode": "exclude", "exclude_query_sector": True})
    ],
    "test": [
        ("mixed", {"same_ticker_mode": "allow", "exclude_query_sector": False}),
        ("cross_ticker_only", {"same_ticker_mode": "exclude", "exclude_query_sector": False}),
        ("sector_excluded", {"same_ticker_mode": "exclude", "exclude_query_sector": True})
    ],
    "holdout": [
        ("cross_ticker_only", {"same_ticker_mode": "exclude", "exclude_query_sector": False})
    ]
}

def load_market_universe(yaml_path: str):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    mapping = {}
    for ind, info in data.get("industries", {}).items():
        sector = info.get("sector", "")
        for t in info.get("tickers", []):
            mapping[t] = {"sector": sector, "industry": ind}
    return mapping

def enrich_latents(latent_path: Path, mapping: dict, is_memory: bool, precomputed_glob: str):
    if not latent_path.exists():
        return
    df = pd.read_parquet(latent_path)
    df["sector"] = df["ticker"].map(lambda x: mapping.get(x, {}).get("sector", "Unknown"))
    df["industry"] = df["ticker"].map(lambda x: mapping.get(x, {}).get("industry", "Unknown"))
    if is_memory:
        df = attach_outcome_availability(df, precomputed_glob, horizon_sessions=126)
        # Drop rows without mapped outcome_available_timestamp to enforce causal separation
        df = df.dropna(subset=["outcome_available_timestamp"])
    df.to_parquet(latent_path, index=False)

def calculate_ticker_hhi(neighbors_path: Path) -> float:
    if not neighbors_path.exists():
        return 0.0
    try:
        df = pd.read_parquet(neighbors_path)
        if df.empty or "query_id" not in df.columns or "neighbor_ticker" not in df.columns:
            return 0.0
        counts = df.groupby(['query_id', 'neighbor_ticker']).size().reset_index(name='count')
        totals = df.groupby('query_id').size().reset_index(name='total')
        merged = counts.merge(totals, on='query_id')
        merged['p_sq'] = (merged['count'] / merged['total']) ** 2
        hhi_per_query = merged.groupby('query_id')['p_sq'].sum()
        val = hhi_per_query.mean()
        return float(val) if pd.notna(val) else 0.0
    except Exception as e:
        logger.error(f"Error calculating HHI: {e}")
        return 0.0

def calculate_weekly_rank_ic(signals_path: Path) -> float:
    if not signals_path.exists():
        return 0.0
    try:
        df = pd.read_parquet(signals_path)
        required = ["timestamp", "opportunity_score", "future_max_return_63", "future_min_return_63", "event_upside_before_drawdown_126", "retrieval_confidence"]
        for col in required:
            if col not in df.columns:
                return 0.0
                
        df = df.dropna(subset=required).copy()
        if df.empty:
            return 0.0
            
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        # realised-utility = (upside - |downside|) * (0.5 + 0.5 * confidence) + 0.05 * path_quality
        df["realized_utility"] = (df["future_max_return_63"] - df["future_min_return_63"].abs()) * (0.5 + 0.5 * df["retrieval_confidence"]) + 0.05 * df["event_upside_before_drawdown_126"]
        
        df["week"] = df["timestamp"].dt.to_period("W")
        weekly_ics = []
        for week, wdf in df.groupby("week"):
            if len(wdf) < 3:
                continue
            scores = pd.to_numeric(wdf["opportunity_score"], errors="coerce").to_numpy()
            utility = pd.to_numeric(wdf["realized_utility"], errors="coerce").to_numpy()
            valid = np.isfinite(scores) & np.isfinite(utility)
            if valid.sum() < 2:
                continue
            sp = float(stats.spearmanr(scores[valid], utility[valid])[0])
            if np.isfinite(sp):
                weekly_ics.append(sp)
                
        if not weekly_ics:
            return 0.0
        return float(np.mean(weekly_ics))
    except Exception as e:
        logger.error(f"Error calculating weekly rank IC: {e}")
        return 0.0

def build_sweep_summary(results, reports_base: Path):
    summary_json_path = reports_base / "sweep_summary.json"
    summary_md_path = reports_base / "sweep_summary.md"
    
    with open(summary_json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    df = pd.DataFrame(results)
    with open(summary_md_path, "w") as f:
        f.write("# Phase 4B Winner Sweep Summary\n\n")
        f.write(df.to_markdown(index=False))

def evaluate_run(run_dir: Path, variant_name: str, seed: int, base_eval_config: dict, splits_to_evaluate: list[str]):
    metrics_summary = {}
    latent_dir = run_dir / "latents"
    
    universe_mapping = load_market_universe(os.path.join(PROJECT_ROOT, "config/market_universe.yaml"))
    precomputed_glob = os.path.join(PROJECT_ROOT, "data/precomputed_v2/*.parquet")
    
    # Enrich memory (train) only once
    train_latents_path = latent_dir / "train_latents.parquet"
    if train_latents_path.exists():
        df_temp = pd.read_parquet(train_latents_path)
        if "sector" not in df_temp.columns:
            enrich_latents(train_latents_path, universe_mapping, True, precomputed_glob)
            
    # Enrich query sets
    for split in splits_to_evaluate:
        enrich_latents(latent_dir / f"{split}_latents.parquet", universe_mapping, False, precomputed_glob)
        
    for split in splits_to_evaluate:
        query_latent_name = "val_latents.parquet" if split == "validation" else f"{split}_latents.parquet"
        query_path = latent_dir / query_latent_name
        if not query_path.exists():
            continue
            
        split_modes = MODES.get(split, [])
        for mode_name, memory_overrides in split_modes:
            mode_dir = run_dir / split / mode_name
            mode_dir.mkdir(parents=True, exist_ok=True)
            
            mode_config = copy.deepcopy(base_eval_config)
            mode_config["data"] = {
                "train_latents": str(latent_dir / "train_latents.parquet"),
                "test_latents": str(query_path),
                "output_dir": str(mode_dir),
                "precomputed_glob": precomputed_glob,
            }
            if "memory" not in mode_config:
                mode_config["memory"] = {}
            mode_config["memory"].update(memory_overrides)
            mode_config["memory"]["minimum_neighbor_separation_sessions"] = 21
            
            try:
                run_market_memory_evaluation(mode_config, Path(PROJECT_ROOT), dry_run=False, run_id="eval")
                
                # Load the generated metrics to aggregate
                metrics_file = mode_dir / "eval" / "metrics.json"
                neighbors_file = mode_dir / "eval" / "neighbors.parquet"
                signals_file = mode_dir / "eval" / "signals.parquet"
                
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        m = json.load(f)
                        metrics_summary[f"{split}_{mode_name}_return"] = m.get("total_return")
                        metrics_summary[f"{split}_{mode_name}_sharpe"] = m.get("sharpe")
                        metrics_summary[f"{split}_{mode_name}_max_drawdown"] = m.get("max_drawdown")
                        
                        # Real ticker HHI calculation from neighbors.parquet
                        metrics_summary[f"{split}_{mode_name}_hhi"] = calculate_ticker_hhi(neighbors_file)
                        metrics_summary[f"{split}_{mode_name}_cross_rate"] = m.get("retrieval_cross_ticker_rate_mean")
                        
                        # Real weekly Rank IC calculation using signals.parquet
                        metrics_summary[f"{split}_{mode_name}_spearman"] = calculate_weekly_rank_ic(signals_file)
            except Exception as e:
                logger.error(f"Eval failed for {variant_name} {seed} {split} {mode_name}: {e}")
                metrics_summary[f"{split}_{mode_name}_error"] = str(e)
                
    return metrics_summary

def run_sweep(base_config_path: str, eval_config_path: str, is_mock: bool = False):
    base_config = load_cycle_config(base_config_path)
    base_eval_config = load_evaluation_config(eval_config_path)
    
    import datetime
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    reports_base = Path(f"reports/phase4b/sweep_{run_id}")
    reports_base.mkdir(parents=True, exist_ok=True)
    
    validation_results = []

    # Step 1: Run training and validation stage for all variants and seeds
    for variant_name, loss_params in VARIANTS.items():
        for seed in SEEDS:
            logger.info(f"=== Starting Run: Variant {variant_name} | Seed {seed} ===")
            run_dir = reports_base / variant_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            run_config = copy.deepcopy(base_config)
            run_config["training"]["seed"] = seed
            run_config["training"]["loss"] = loss_params
            run_config["training"]["model_dir"] = str(run_dir / "model")
            run_config["evaluation"]["reports_dir"] = str(run_dir / "reports")
            
            run_config["latent_export"] = {
                "output_dir": str(run_dir / "latents"),
                "splits": ["train", "val", "test", "holdout"]
            }
            run_config["_export_latents"] = True
            
            if is_mock:
                run_config["training"]["epochs"] = 1
                run_config["training"]["batch_size"] = 16
                run_config["data"]["precomputed_dir"] = "data/precomputed_v2/AAPL.parquet"
                
            config_snapshot = run_dir / "config_snapshot.yaml"
            with open(config_snapshot, "w") as f:
                yaml.dump(run_config, f)
            
            try:
                temp_config_path = run_dir / "temp_train_config.yaml"
                with open(temp_config_path, "w") as f:
                    yaml.dump(run_config, f)
                
                train(config_path=str(temp_config_path))
                
                # Rename latents to match what evaluate_run expects (strip timestamp)
                latents_dir = run_dir / "latents"
                for split_prefix in ["train_latents", "val_latents", "test_latents", "holdout_latents"]:
                    split_files = list(latents_dir.glob(f"{split_prefix}_*.parquet"))
                    if split_files:
                        split_files.sort()
                        latest_file = split_files[-1]
                        target_file = latents_dir / f"{split_prefix}.parquet"
                        latest_file.replace(target_file)
                        for f in split_files[:-1]:
                            try:
                                f.unlink()
                            except Exception:
                                pass
                
                # Run validation split evaluation only
                val_metrics = evaluate_run(run_dir, variant_name, seed, base_eval_config, ["validation"])
                val_metrics["variant"] = variant_name
                val_metrics["seed"] = seed
                validation_results.append(val_metrics)
                
            except Exception as e:
                logger.error(f"Run failed for {variant_name} seed {seed}: {e}")
                traceback.print_exc()
                validation_results.append({
                    "variant": variant_name,
                    "seed": seed,
                    "error": str(e)
                })

    # Step 2: Write validation selection summary
    val_json_path = reports_base / "validation_selection_summary.json"
    val_md_path = reports_base / "validation_selection_summary.md"
    
    with open(val_json_path, "w") as f:
        json.dump(validation_results, f, indent=2)
        
    val_df = pd.DataFrame(validation_results)
    
    # Step 3: Apply the promotion contract
    def is_valid_run(r: dict) -> bool:
        # True if there's no top-level error and no split-level errors (like sector_excluded_error)
        return "error" not in r and not any(k.endswith("_error") for k in r.keys())

    baseline_runs = [r for r in validation_results if r.get("variant") == "huber_baseline_v2" and is_valid_run(r)]
    analogue_runs = [r for r in validation_results if r.get("variant") == "huber_analogue_v2" and is_valid_run(r)]
    
    baseline_sharpe = float(np.median([r.get("validation_cross_ticker_only_sharpe", 0.0) for r in baseline_runs])) if baseline_runs else -999.0
    analogue_sharpe = float(np.median([r.get("validation_cross_ticker_only_sharpe", 0.0) for r in analogue_runs])) if analogue_runs else -999.0
    
    baseline_ic = float(np.mean([r.get("validation_cross_ticker_only_spearman", 0.0) for r in baseline_runs])) if baseline_runs else -999.0
    analogue_ic = float(np.mean([r.get("validation_cross_ticker_only_spearman", 0.0) for r in analogue_runs])) if analogue_runs else -999.0
    
    promoted = (
        len(analogue_runs) == len(SEEDS) and 
        analogue_ic > 0.0 and 
        analogue_sharpe > 0.0
    )
    winner_variant = "huber_analogue_v2" if promoted else "huber_baseline_v2"
    
    with open(val_md_path, "w") as f:
        f.write("# Phase 4B Validation Selection Summary\n\n")
        f.write(val_df.to_markdown(index=False))
        f.write("\n\n## Promotion Decision Summary\n\n")
        f.write(f"- **Huber Baseline v2 Median Sharpe**: {baseline_sharpe:.4f}\n")
        f.write(f"- **Huber Analogue v2 Median Sharpe**: {analogue_sharpe:.4f}\n")
        f.write(f"- **Huber Baseline v2 Mean weekly Rank IC**: {baseline_ic:.4f}\n")
        f.write(f"- **Huber Analogue v2 Mean weekly Rank IC**: {analogue_ic:.4f}\n\n")
        if promoted:
            f.write(f"### **PROMOTED**: `huber_analogue_v2` successfully completed all seeds, achieved positive rank IC, positive median Sharpe, and had no split evaluation errors!\n")
        else:
            f.write(f"### **RETAINED**: `huber_analogue_v2` failed the strict promotion criteria. `huber_baseline_v2` is selected as the winner.\n")

    # Step 4: Evaluate test and holdout ONLY for the winner variant
    logger.info(f"Selected winner variant: {winner_variant}")
    final_sweep_results = []
    
    for seed in SEEDS:
        run_dir = reports_base / winner_variant / f"seed_{seed}"
        logger.info(f"Running test/holdout evaluation for {winner_variant} | Seed {seed}")
        test_metrics = evaluate_run(run_dir, winner_variant, seed, base_eval_config, ["test", "holdout"])
        
        # Merge with validation results
        val_metrics = next((r for r in validation_results if r.get("variant") == winner_variant and r.get("seed") == seed), {})
        combined = {"variant": winner_variant, "seed": seed, **val_metrics, **test_metrics}
        final_sweep_results.append(combined)
        
        # Save run_metrics.json for the seed
        with open(run_dir / "run_metrics.json", "w") as f:
            json.dump(combined, f, indent=2)
            
    build_sweep_summary(final_sweep_results, reports_base)
    logger.info(f"Sweep complete. Summary written to {reports_base}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs/cycle_model.yaml", help="Path to base config")
    parser.add_argument("--eval-config", default="configs/market_memory_backtest.yaml", help="Path to eval config")
    parser.add_argument("--mock", action="store_true", help="Run a quick mock")
    args = parser.parse_args()
    
    run_sweep(args.config_path, args.eval_config, args.mock)
