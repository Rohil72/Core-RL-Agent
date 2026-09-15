#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p outputs/ops
mkdir -p rebuild_plan/a30_production_out

PYTHON_EXEC="/root/miniconda3/envs/py3.10/bin/python3"
LOG_FILE="outputs/ops/production.log"
OPS_LOG="outputs/ops/operational_log.txt"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] LAUNCH: Starting production experiment run..." | tee -a "${OPS_LOG}"

${PYTHON_EXEC} -u scripts/launch_a30_production.py \
    --config rebuild_plan/config.proposed.json \
    --data-dir data/cache/ohlcv \
    --output-dir outputs/a30-c0f895f \
    --sample-ids-dir rebuild_plan/sample_ids \
    --folds 2020 2021 2022 2023 2024 2025 \
    --authorize-production \
    --mode production 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] FINISH: Production process exited with code ${EXIT_CODE}" | tee -a "${OPS_LOG}"
exit ${EXIT_CODE}
