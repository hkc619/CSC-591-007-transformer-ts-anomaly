#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-111%4
#SBATCH --open-mode=append
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
: "${SLURM_ARRAY_TASK_ID:?需要 Slurm array task ID}"
for option in "$@"; do
  case "$option" in
    --retry|--recover-interrupted) ;;
    *) echo "只接受 --retry / --recover-interrupted，不能覆寫科學設定" >&2; exit 2 ;;
  esac
done
"$PYTHON_BIN" -u scripts/run_seed_replication.py --hpc --task-id "$SLURM_ARRAY_TASK_ID" "$@"
