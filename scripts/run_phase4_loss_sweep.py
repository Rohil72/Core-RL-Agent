import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trainers.train_cycle_model import train, load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VARIANTS = {
    "huber_baseline": {"lambda_reg": 1.00, "lambda_analogue": 0.00, "lambda_rank": 0.00, "lambda_var": 0.00},
    "huber_rank": {"lambda_reg": 1.00, "lambda_analogue": 0.00, "lambda_rank": 0.10, "lambda_var": 0.00},
    "huber_analogue": {"lambda_reg": 1.00, "lambda_analogue": 0.20, "lambda_rank": 0.00, "lambda_var": 0.00},
    "huber_supcon_control": {"lambda_reg": 1.00, "lambda_analogue": 0.20, "lambda_rank": 0.00, "lambda_var": 0.00, "is_supcon_control": True},
    "huber_triplet_control": {"lambda_reg": 1.00, "lambda_analogue": 0.20, "lambda_rank": 0.00, "lambda_var": 0.00, "is_triplet_control": True},
    "huber_analogue_rank": {"lambda_reg": 1.00, "lambda_analogue": 0.20, "lambda_rank": 0.10, "lambda_var": 0.00},
    "huber_analogue_rank_vc": {"lambda_reg": 1.00, "lambda_analogue": 0.20, "lambda_rank": 0.10, "lambda_var": 0.01},
}

SEEDS = [7, 17, 37]


def run_sweep(base_config_path: str):
    base_config = load_config(base_config_path)
    
    reports_base = Path("reports/phase4")
    reports_base.mkdir(parents=True, exist_ok=True)
    
    sweep_results = []

    for variant_name, loss_params in VARIANTS.items():
        for seed in SEEDS:
            logger.info(f"=== Starting Run: Variant {variant_name} | Seed {seed} ===")
            
            # Create isolated config
            run_config = copy.deepcopy(base_config)
            
            # Override seed
            run_config["training"]["seed"] = seed
            
            # Inject loss configuration
            run_config["training"]["loss"] = loss_params
            
            # Override export and report directories to ensure isolation
            run_dir = reports_base / variant_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            
            run_config["training"]["model_dir"] = str(run_dir / "model")
            run_config["evaluation"]["reports_dir"] = str(run_dir / "reports")
            
            # Write config snapshot
            config_snapshot = run_dir / "config_snapshot.yaml"
            with open(config_snapshot, "w") as f:
                yaml.dump(run_config, f)
            
            try:
                # Invoke the trainer (which internally evaluates validation/test splits)
                # To pass the path, we can either monkey-patch load_config or write out the temp config
                # and pass that to train()
                temp_config_path = run_dir / "temp_train_config.yaml"
                with open(temp_config_path, "w") as f:
                    yaml.dump(run_config, f)
                
                # Execute training
                summary = train(config_path=str(temp_config_path))
                
                # Evaluate Market Memory (Stubbed here for pipeline completeness)
                # In full execution, we'd invoke the evaluator on the latents produced
                
                # Collect metrics
                val_metrics = summary["metrics"].get("val", {})
                
                sweep_results.append({
                    "variant": variant_name,
                    "seed": seed,
                    "val_future_target_mae": val_metrics.get("future_target_mae", float('inf')),
                    "val_action_accuracy": val_metrics.get("action_accuracy", 0.0),
                    # Other proxy metrics would be pulled here
                })
                
                # Clean up temp file
                if temp_config_path.exists():
                    temp_config_path.unlink()
                    
            except Exception as e:
                logger.error(f"Run failed for {variant_name} seed {seed}: {e}")
                sweep_results.append({
                    "variant": variant_name,
                    "seed": seed,
                    "error": str(e)
                })

    # Write Sweep Summary
    summary_df = pd.DataFrame(sweep_results)
    
    summary_json_path = reports_base / "sweep_summary.json"
    summary_md_path = reports_base / "sweep_summary.md"
    
    with open(summary_json_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
        
    with open(summary_md_path, "w") as f:
        f.write("# Phase 4 Loss Sweep Summary\n\n")
        f.write(summary_df.to_markdown(index=False))

    logger.info(f"Sweep complete. Summary written to {reports_base}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default="configs/cycle_model.yaml", help="Path to base config")
    args = parser.parse_args()
    
    run_sweep(args.config_path)
