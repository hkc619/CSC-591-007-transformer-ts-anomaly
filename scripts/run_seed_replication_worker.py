"""全新 subprocess 的 args adapter；執行原 main.py，不複製 training / evaluation。"""
import argparse
import os
from pathlib import Path
import runpy
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.seed_replication import claim_worker, manifest, worker_args


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    roots = resolve_roots(options)
    rows = manifest(ROOT)
    if not 0 <= options.task_id < len(rows):
        parser.error("task-id 必須為 0..111")
    args = worker_args(roots, rows[options.task_id], options.attempt_dir, options.dry_run)
    if not options.dry_run:
        claim_worker(options.attempt_dir, rows[options.task_id], [sys.executable, *sys.argv], args)
    if "src.parser" in sys.modules:
        raise RuntimeError("worker 必須在乾淨 subprocess 啟動，不能沿用其他實驗的 args")
    # 原程式以 from src.parser import args 共用 CLI 狀態。
    # 僅在此專用 subprocess 提供已驗證的新 seed args，原 parser.py 原文保持不變。
    adapter = types.ModuleType("src.parser")
    adapter.args = args
    sys.modules["src.parser"] = adapter
    os.environ.update(PROJECT_ROOT=str(roots.project), DATA_ROOT=str(roots.data), RESULTS_ROOT=args.results_root)
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()
