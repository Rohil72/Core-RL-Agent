"""
Runner script to execute Phase 2 analyses.

This script calls the four analysis scripts and writes outputs into reports/phase2/ and artifacts/phase2/.

Usage:
    python scripts/phase2_run_all.py --precomputed-dir <path> [--latent-dir <path>]

Note: scripts assume standard Python packages: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, lightgbm, umap-learn, shap (optional).
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

def run(script_name, args=None):
    cmd = [sys.executable, str(SCRIPTS / script_name)]
    if args:
        cmd += args
    print("Running:", cmd)
    subprocess.check_call(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precomputed-dir', required=False, help='glob or dir for precomputed parquet files')
    parser.add_argument('--latent-dir', required=False, help='dir containing exported latent parquet files')
    args = parser.parse_args()

    common_args = []
    if args.precomputed_dir:
        common_args += ["--precomputed-dir", args.precomputed_dir]
    if args.latent_dir:
        common_args += ["--latent-dir", args.latent_dir]

    run('phase2_target_audit.py', common_args)
    run('phase2_baseline_learnability.py', common_args)
    run('phase2_feature_audit.py', common_args)
    run('phase2_latent_audit.py', common_args)

    print('\nAll Phase 2 scripts completed. Check reports/phase2 for outputs.')

