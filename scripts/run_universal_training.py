#!/usr/bin/env python3
"""Universal TRAIN-only 入口；不執行 TEST、threshold 或 calibration。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.paths import resolve_roots
from src.universal_training import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["TranAD"], default="TranAD")
    parser.add_argument("--dataset", choices=["SMD"], default="SMD")
    parser.add_argument("--seed", type=int, choices=[1, 2, 3], default=1)
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true", help="只允許在已失敗 run 下新增 attempt，保留舊 attempt")
    args = parser.parse_args()
    execute(resolve_roots(args), seed=args.seed, retry=args.retry, dry_run=args.dry_run,
            command=[sys.executable, *sys.argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
