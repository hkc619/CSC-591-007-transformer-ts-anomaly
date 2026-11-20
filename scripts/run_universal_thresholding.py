#!/usr/bin/env python3
"""Universal Stage 3：先 local_05 smoke，再 TRAIN scores／完整 global + local threshold comparison。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.universal_thresholding import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["smoke", "train-scores", "full"], default="smoke")
    parser.add_argument("--seed", type=int, choices=[1], default=1)
    parser.add_argument("--stage1-run")
    parser.add_argument("--stage2-run", help="指定 completed Stage-2 full attempt；不得使用 smoke")
    parser.add_argument("--formal-summary")
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--"+name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true")
    args = parser.parse_args()
    execute(resolve_roots(args), phase=args.phase, seed=args.seed, stage1_run=args.stage1_run,
            stage2_run=args.stage2_run, formal_summary=args.formal_summary,
            dry_run=args.dry_run, retry=args.retry, command=[sys.executable, *sys.argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
