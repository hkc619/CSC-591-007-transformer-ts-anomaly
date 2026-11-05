#!/usr/bin/env bash
#SBATCH --job-name=tranad-cal-smoke
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/calibration_common.sh"
"$PYTHON_BIN" -u scripts/run_target_calibration.py \
  --hpc --target machine-1-1 --fraction 0.05 --smoke-test
