#!/usr/bin/env bash
# 僅準備；必須先 review tasks 0/1，不能本次自動提交其餘 10 runs。
#SBATCH --job-name=at-robustness
#SBATCH --account=csc591f26_gpu
#SBATCH --partition=gpu_partners
#SBATCH --qos=short_gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=2-11%1
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/anomaly_transformer_common.sh"
: "${SLURM_ARRAY_TASK_ID:?需要 array task ID}"
"$AT_PYTHON_BIN" -u scripts/run_anomaly_transformer.py --hpc --device cuda \
  --manifest configs/anomaly_transformer_runs.csv --task-id "$SLURM_ARRAY_TASK_ID"
