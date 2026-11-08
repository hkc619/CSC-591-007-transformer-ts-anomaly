#!/usr/bin/env bash
#SBATCH --job-name=at-smoke
#SBATCH --account=csc591f26_gpu
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/anomaly_transformer_common.sh"
: "${AT_SMOKE_TASK_ID:?只允許 0=per-machine 或 1=cross-machine}"
[[ "$AT_SMOKE_TASK_ID" == 0 || "$AT_SMOKE_TASK_ID" == 1 ]] || exit 2
nvidia-smi
"$AT_PYTHON_BIN" -u scripts/run_anomaly_transformer.py --hpc --device cuda \
  --manifest configs/anomaly_transformer_runs.csv --task-id "$AT_SMOKE_TASK_ID" --phase smoke_gate
