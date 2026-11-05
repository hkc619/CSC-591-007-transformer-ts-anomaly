#!/usr/bin/env bash
# 本機 synthetic smoke：無 SMD、無正式 checkpoint、無 training、無真實 POT fitting。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
SMOKE_PYTHON="${PYTHON_BIN:-python3}"
command -v "$SMOKE_PYTHON" >/dev/null 2>&1 || {
  echo "找不到 Python；請設定 PYTHON_BIN 指向已安裝 requirements-hpc.txt 的 environment。" >&2
  exit 2
}
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tranad-calibration-local-mpl"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
echo "本機 synthetic calibration smoke：fixtures 暫存並自動清除；POT 使用 stub。"
echo "Hazel / 真實 checkpoint / 真實 POT validation 仍待後續執行。"
exec "$SMOKE_PYTHON" -m unittest discover -s tests -p 'test*calibration*.py' -v
