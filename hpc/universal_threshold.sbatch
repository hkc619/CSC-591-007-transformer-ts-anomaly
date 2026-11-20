#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# 單一 job 順序處理 TRAIN；預設只做 machine-1-1 local_05 smoke，不提交其他 jobs。
set -euo pipefail
source "${PROJECT_ROOT:?請設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/universal-threshold-mpl-${SLURM_JOB_ID:-$$}"
args=(--hpc --seed 1 --phase "${UNIVERSAL_THRESHOLD_PHASE:-smoke}")
[[ -z "${UNIVERSAL_STAGE1_RUN:-}" ]] || args+=(--stage1-run "$UNIVERSAL_STAGE1_RUN")
[[ -z "${UNIVERSAL_STAGE2_RUN:-}" ]] || args+=(--stage2-run "$UNIVERSAL_STAGE2_RUN")
[[ -z "${UNIVERSAL_FORMAL_SUMMARY:-}" ]] || args+=(--formal-summary "$UNIVERSAL_FORMAL_SUMMARY")
[[ "${UNIVERSAL_THRESHOLD_RETRY:-0}" != "1" ]] || args+=(--retry)
"$PYTHON_BIN" -u scripts/run_universal_thresholding.py "${args[@]}"
