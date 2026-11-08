"""單一 AT ranking-only run；不呼叫 threshold / calibration，也不提交 jobs。"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.anomaly_transformer.protocol import configuration, validate_manifest
from src.anomaly_transformer.experiment import describe, execute


def main():
    parser = argparse.ArgumentParser(description="Anomaly Transformer 6-target ranking-only robustness")
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--setting", choices=("per_machine", "cross_machine"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--phase", choices=("smoke_gate", "formal"), default="formal")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    roots = resolve_roots(args)
    os.environ.update(PROJECT_ROOT=str(roots.project), DATA_ROOT=str(roots.data), RESULTS_ROOT=str(roots.results))
    config, canonical = configuration(roots.project)
    rows = validate_manifest(roots.project / "configs/anomaly_transformer_runs.csv", config)
    if args.manifest:
        if args.target is not None or args.setting is not None:
            parser.error("manifest mode 不接受 target/setting 覆寫")
        path = args.manifest if args.manifest.is_absolute() else roots.project / args.manifest
        rows = validate_manifest(path, config)
        if args.task_id is None or not 0 <= args.task_id < 12:
            parser.error("需要 task-id=0..11")
        row = rows[args.task_id]
        args.target, args.setting = row["target"], row["setting"]
    else:
        if args.task_id is not None or args.target is None or args.setting is None:
            parser.error("請指定 target / setting，或 manifest + task-id")
        matches = [row for row in rows if row["target"] == args.target and row["setting"] == args.setting]
        if len(matches) != 1:
            parser.error("target 不在預先指定的六台中")
        args.task_id = int(matches[0]["task_id"])
    if args.phase == "smoke_gate" and args.target != "machine-1-1":
        parser.error("smoke gate 只允許 machine-1-1")
    if args.dry_run:
        print(json.dumps(describe(roots, config, canonical, args.target, args.setting, args.device),
                         ensure_ascii=False, indent=2))
        return 0
    execute(roots, config, canonical, args.target, args.setting, args.device,
            args.task_id, args.phase, args.retry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
