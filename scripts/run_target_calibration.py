"""單次 target threshold calibration CLI；不提交 jobs、不啟動 training。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_protocol import canonical_fraction, load_configuration, validate_manifest
from src.paths import overlaps, resolve_roots
from src.target_calibration import execute_calibration, inspect_experiment


def main():
    parser = argparse.ArgumentParser(description="Frozen TranAD target TRAIN-prefix threshold calibration")
    for name in ("project-root", "data-root", "results-root", "formal-results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--fraction")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true", help="保留舊 attempt，明確建立新 attempt；不 resume model")
    parser.add_argument("--smoke-test", action="store_true",
                        help="只標記 smoke artifact；完整指定 prefix，無 training / TEST inference")
    args = parser.parse_args()
    roots = resolve_roots(args)
    config, machines = load_configuration(roots.project)
    formal = Path(args.formal_results_root or os.environ.get("FORMAL_RESULTS_ROOT") or roots.results).expanduser().resolve()
    if overlaps(formal, roots.project) or overlaps(formal, roots.data):
        parser.error("FORMAL_RESULTS_ROOT 必須位於 code / dataset 外")
    if args.manifest:
        if args.target is not None or args.fraction is not None or args.smoke_test:
            parser.error("manifest mode 不接受 target/fraction 覆寫或 smoke 標記")
        path = args.manifest if args.manifest.is_absolute() else roots.project / args.manifest
        rows = validate_manifest(path, machines)
        if args.task_id is None or not 0 <= args.task_id < 84:
            parser.error("manifest mode 需要 task-id=0..83")
        row = rows[args.task_id]
        args.target, args.fraction = row["target"], row["fraction"]
    elif args.task_id is not None or args.target is None or args.fraction is None:
        parser.error("請指定 --target 與 --fraction，或 manifest + task-id")
    fraction = canonical_fraction(args.fraction)
    if args.dry_run:
        description, *_ = inspect_experiment(roots, formal, config, machines, args.target, fraction)
        print(json.dumps(description, ensure_ascii=False, indent=2))
        print("NOT EXECUTED — requires Hazel validation；未載入 model、未 inference、未 fit POT、未建立 outputs")
        return 0
    execute_calibration(roots, formal, config, machines, args.target, fraction,
                        smoke=args.smoke_test, task_id=args.task_id, retry=args.retry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
