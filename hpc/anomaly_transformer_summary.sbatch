#!/usr/bin/env bash
#SBATCH --job-name=at-summary
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
: "${PROJECT_ROOT:?請設定 PROJECT_ROOT}"
: "${DATA_ROOT:?請設定 DATA_ROOT}"
: "${RESULTS_ROOT:?請設定 RESULTS_ROOT}"
: "${PYTHON_BIN:?請設定原 CPU environment}"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" scripts/analyze_anomaly_transformer.py --hpc
