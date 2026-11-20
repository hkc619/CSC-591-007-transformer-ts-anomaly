#!/usr/bin/env python3
"""Stage 2：先 machine-1-1 smoke，成功後才允許單一 job 完成 28 台 continuous-score evaluation。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import resolve_roots
from src.universal_evaluation import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target", help="先只允許 machine-1-1 smoke")
    mode.add_argument("--all", action="store_true", dest="all_targets", help="需要同 checkpoint / implementation 的成功 smoke")
    parser.add_argument("--seed", type=int, choices=[1], default=1)
    parser.add_argument("--stage1-run", help="明確指定 completed Stage-1 attempt；否則必須恰有一個 completed attempt")
    parser.add_argument("--formal-summary", help="可選 seed-1 formal CSV；會核對原 run metadata / evaluation / TEST hashes")
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true", help="只重試失敗 evaluation，建立全新 attempt，不覆寫原 scores")
    args = parser.parse_args()
    execute(resolve_roots(args), target=args.target, all_targets=args.all_targets, seed=args.seed,
        stage1_run=args.stage1_run, retry=args.retry, dry_run=args.dry_run,
        formal_summary=args.formal_summary, command=[sys.executable, *sys.argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
