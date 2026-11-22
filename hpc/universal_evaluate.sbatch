#!/usr/bin/env bash
#SBATCH --account=csc591f26_cpu
#SBATCH --partition=compute_partners
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
# 預設只跑 smoke。full 要由使用者明確指定，且程式會確認 matching smoke 已成功。
set -euo pipefail
source "${PROJECT_ROOT:?請先設定 PROJECT_ROOT}/hpc/seed_replication_common.sh"
SEED="${SEED:-1}"
case "$SEED" in 1|2|3) ;; *) echo "SEED 必須為 1、2 或 3" >&2; exit 2 ;; esac
export PYTHONHASHSEED="$SEED"
if [[ "$SEED" != "1" ]]; then
  args=(--hpc --seed "$SEED" --phase evaluate)
  [[ "${UNIVERSAL_REPLICATION_RETRY:-0}" != "1" ]] || args+=(--retry)
  "$PYTHON_BIN" -u scripts/run_universal_replication.py "${args[@]}"
  exit 0
fi
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/tranad-universal-eval-mpl-${SLURM_JOB_ID:-$$}"
args=(--hpc --seed 1)
case "${UNIVERSAL_EVAL_MODE:-smoke}" in
  smoke) args+=(--target machine-1-1) ;;
  full) args+=(--all) ;;
  *) echo "UNIVERSAL_EVAL_MODE 必須為 smoke 或 full" >&2; exit 2 ;;
esac
[[ -z "${UNIVERSAL_STAGE1_RUN:-}" ]] || args+=(--stage1-run "$UNIVERSAL_STAGE1_RUN")
[[ -z "${UNIVERSAL_FORMAL_SUMMARY:-}" ]] || args+=(--formal-summary "$UNIVERSAL_FORMAL_SUMMARY")
[[ "${UNIVERSAL_EVAL_RETRY:-0}" != "1" ]] || args+=(--retry)
"$PYTHON_BIN" -u scripts/run_universal_evaluation.py "${args[@]}"
