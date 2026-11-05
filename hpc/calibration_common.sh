#!/usr/bin/env bash
# Calibration 專用環境；不改原 formal scripts 或其 cache。
set -euo pipefail
: "${PROJECT_ROOT:?請設定 PROJECT_ROOT}"
: "${DATA_ROOT:?請設定 DATA_ROOT}"
: "${RESULTS_ROOT:?請設定 RESULTS_ROOT}"
: "${PYTHON_BIN:?請設定 PYTHON_BIN}"
[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN 不可執行" >&2; exit 2; }
cd "$PROJECT_ROOT"
export FORMAL_RESULTS_ROOT="${FORMAL_RESULTS_ROOT:-$RESULTS_ROOT}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RESULTS_ROOT/calibration/cache/matplotlib/${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-single}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""
