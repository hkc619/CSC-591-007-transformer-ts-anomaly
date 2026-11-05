#!/usr/bin/env bash
# 預備範本：須使用者確認 machine-1-1 smoke 後，才可手動提交。
#SBATCH --job-name=tranad-calibration
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --array=0-83%8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/calibration_common.sh"
: "${SLURM_ARRAY_TASK_ID:?需要 Slurm array task ID}"
"$PYTHON_BIN" -u scripts/run_target_calibration.py --hpc \
  --manifest configs/calibration_runs.csv --task-id "$SLURM_ARRAY_TASK_ID"
