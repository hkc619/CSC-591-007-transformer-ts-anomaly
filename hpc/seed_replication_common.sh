#!/usr/bin/env bash
# 保留共用 RESULTS_ROOT，不改舊 CPU/AT/calibration scripts 的環境或輸出位置。
set -euo pipefail
: "${PROJECT_ROOT:?請設定 PROJECT_ROOT}"
: "${DATA_ROOT:?請設定 DATA_ROOT}"
: "${RESULTS_ROOT:?請設定共同外部 RESULTS_ROOT}"
: "${PYTHON_BIN:?請設定既有 CPU float64 Python environment}"
[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN 不可執行" >&2; exit 2; }
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
# 每個 job 獨立 matplotlib cache；不要沿用舊 summaries 或 figures directory。
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tranad-seed-mpl-${SLURM_JOB_ID:-$$}-${SLURM_ARRAY_TASK_ID:-single}"
