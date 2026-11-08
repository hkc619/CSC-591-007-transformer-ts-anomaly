#!/usr/bin/env bash
# 不 source CPU common.sh，避免清空 Slurm 配置的 CUDA_VISIBLE_DEVICES。
set -euo pipefail
: "${PROJECT_ROOT:?請設定 PROJECT_ROOT}"
: "${DATA_ROOT:?請設定 DATA_ROOT}"
: "${RESULTS_ROOT:?請設定 RESULTS_ROOT}"
: "${AT_PYTHON_BIN:?請設定獨立 GPU environment 的 AT_PYTHON_BIN；保留原 TranAD CPU environment}"
[[ -x "$AT_PYTHON_BIN" ]] || { echo "AT_PYTHON_BIN 不可執行" >&2; exit 2; }
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RESULTS_ROOT/anomaly_transformer/cache/matplotlib/${SLURM_JOB_ID:-local}"
if [[ "${CUDA_VISIBLE_DEVICES+x}" == x && -z "$CUDA_VISIBLE_DEVICES" ]]; then
  echo "CUDA_VISIBLE_DEVICES 被清空；不要使用 CPU common.sh。請保留 Slurm 指派的 GPU。" >&2
  exit 2
fi
