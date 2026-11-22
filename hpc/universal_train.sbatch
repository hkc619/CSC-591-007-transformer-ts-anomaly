#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# 單一 seed；無 array，不自動啟動另一個 seed。
set -euo pipefail
source "${PROJECT_ROOT:?請先設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
SEED="${SEED:-1}"
case "$SEED" in 1|2|3) ;; *) echo "SEED 必須為 1、2 或 3" >&2; exit 2 ;; esac
export PYTHONHASHSEED="$SEED"
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tranad-universal-mpl-${SLURM_JOB_ID:-$$}"
"$PYTHON_BIN" -u scripts/run_universal_training.py --hpc --seed "$SEED"
