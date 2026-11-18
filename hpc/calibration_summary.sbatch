#!/usr/bin/env bash
# 僅在 84 tasks 實際成功後使用；本次不提交。
#SBATCH --job-name=tranad-cal-summary
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --open-mode=append
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/calibration_common.sh"
read -r -a calibration_seeds <<< "${CALIBRATION_SEEDS:-1}"
"$PYTHON_BIN" -u scripts/analyze_calibration.py --hpc --seeds "${calibration_seeds[@]}"
