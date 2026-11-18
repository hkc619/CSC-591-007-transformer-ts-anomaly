"""單次 target threshold calibration CLI；不提交 jobs、不啟動 training。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_protocol import canonical_fraction, load_configuration, manifest_seeds, validate_manifest
from src.paths import overlaps, resolve_roots
from src.target_calibration import execute_calibration, inspect_experiment


def main():
    parser = argparse.ArgumentParser(description="Frozen TranAD target TRAIN-prefix threshold calibration")
    for name in ("project-root", "data-root", "results-root", "formal-results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--fraction")
    parser.add_argument("--seed", type=int, help="formal model seed，預設 1；本次新增 2/3")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true", help="保留舊 attempt，明確建立新 attempt；不 resume model")
    parser.add_argument("--smoke-test", action="store_true",
                        help="只標記 smoke artifact；完整指定 prefix，無 training / TEST inference")
    args = parser.parse_args()
    roots = resolve_roots(args)
    config, machines = load_configuration(roots.project)
    seeds = None
    if args.manifest:
        if args.target is not None or args.fraction is not None or args.smoke_test or args.seed is not None:
            parser.error("manifest mode 不接受 target/fraction 覆寫或 smoke 標記")
        path = args.manifest if args.manifest.is_absolute() else roots.project / args.manifest
        seeds = manifest_seeds(path)
        if 1 in seeds and len(seeds) != 1:
            parser.error("seed=1 必須維持獨立舊 manifest；新陣列不可混入 seed=1")
        rows = validate_manifest(path, machines, seeds)
        if args.task_id is None or not 0 <= args.task_id < len(rows):
            parser.error(f"manifest mode 需要 task-id=0..{len(rows)-1}")
        row = rows[args.task_id]
        args.target, args.fraction = row["target"], row["fraction"]
        args.seed = int(row["seed"])
    elif args.task_id is not None or args.target is None or args.fraction is None:
        parser.error("請指定 --target 與 --fraction，或 manifest + task-id")
    fraction = canonical_fraction(args.fraction)
    seed = 1 if args.seed is None else args.seed
    if seed not in (1, 2, 3):
        parser.error("目前 formal checkpoint resolver 僅支援已完成的 seeds=1,2,3")
    formal_env = "FORMAL_RESULTS_ROOT" if seed == 1 else "SEED23_FORMAL_RESULTS_ROOT"
    formal = Path(args.formal_results_root or os.environ.get(formal_env) or roots.results).expanduser().resolve()
    if overlaps(formal, roots.project) or overlaps(formal, roots.data):
        parser.error("formal results root 必須位於 code / dataset 外")
    if args.dry_run:
        description, *_ = inspect_experiment(roots, formal, config, machines, args.target, fraction, seed)
        print(json.dumps(description, ensure_ascii=False, indent=2))
        print("NOT EXECUTED — requires Hazel validation；未載入 model、未 inference、未 fit POT、未建立 outputs")
        return 0
    execute_calibration(roots, formal, config, machines, args.target, fraction,
                        smoke=args.smoke_test, task_id=args.task_id, retry=args.retry,
                        seed=seed, manifest_seed_values=seeds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
