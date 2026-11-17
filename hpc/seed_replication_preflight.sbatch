#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
"$PYTHON_BIN" scripts/build_seed_replication_manifest.py
# 只做原 loader dry-run：不初始化模型、不訓練、不 POT fitting、不建立 run。
for task in 0 28 56 84; do
  "$PYTHON_BIN" -u scripts/run_seed_replication.py --hpc --task-id "$task" --dry-run
done
