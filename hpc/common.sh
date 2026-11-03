#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ROOT:?請設定 PROJECT_ROOT}"
: "${DATA_ROOT:?請設定 DATA_ROOT}"
: "${RESULTS_ROOT:?請設定 RESULTS_ROOT}"
: "${PYTHON_BIN:?請設定外部 Python environment 的 PYTHON_BIN}"
[[ -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN 不可執行" >&2; exit 2; }
cd "$PROJECT_ROOT"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$RESULTS_ROOT/cache/matplotlib/${SLURM_JOB_ID:-local}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$MPLCONFIGDIR" "$RESULTS_ROOT/slurm-logs"
