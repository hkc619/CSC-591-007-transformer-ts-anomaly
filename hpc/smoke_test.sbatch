#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PROJECT_ROOT:?請先設定 PROJECT_ROOT}/hpc/common.sh"
"$PYTHON_BIN" -u scripts/run_experiment.py --hpc --machine "${SMOKE_MACHINE:-machine-1-1}" --smoke-test --epochs 1 --progress
