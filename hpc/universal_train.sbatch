#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Stage 1 只有 seed=1；沒有 array，不自動啟動 seeds 2/3。
set -euo pipefail
source "${PROJECT_ROOT:?請先設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
SEED="${SEED:-1}"
[[ "$SEED" == "1" ]] || { echo "Stage 1 只准備 seed=1；seeds 2/3 留待 review" >&2; exit 2; }
export PYTHONHASHSEED="$SEED"
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tranad-universal-mpl-${SLURM_JOB_ID:-$$}"
"$PYTHON_BIN" -u scripts/run_universal_training.py --hpc --seed "$SEED"
