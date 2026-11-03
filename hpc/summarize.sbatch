#!/usr/bin/env bash
#SBATCH --job-name=tranad-summary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
source "${PROJECT_ROOT:?請先設定 PROJECT_ROOT}/hpc/common.sh"
"$PYTHON_BIN" scripts/analyze_formal_experiments.py --hpc
