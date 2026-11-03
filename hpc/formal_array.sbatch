#!/usr/bin/env bash
#SBATCH --job-name=tranad-v2
#SBATCH --array=0-55%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
source "${PROJECT_ROOT:?請先設定 PROJECT_ROOT}/hpc/common.sh"
: "${SLURM_ARRAY_TASK_ID:?此腳本必須作為 Slurm array 執行}"
"$PYTHON_BIN" -u scripts/run_experiment.py --hpc --manifest configs/formal_28_machine_runs.csv --task-id "$SLURM_ARRAY_TASK_ID"
