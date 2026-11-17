#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
"$PYTHON_BIN" scripts/find_failed_seed_replication_runs.py --hpc
"$PYTHON_BIN" scripts/analyze_seed_replication.py --hpc \
  --reference-results-root "${SEED1_RESULTS_ROOT:-$RESULTS_ROOT}"
