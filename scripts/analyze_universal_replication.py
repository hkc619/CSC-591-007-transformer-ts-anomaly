#!/usr/bin/env python3
"""唯讀完整三 seed Universal 結果，另存 multiseed/analysis-*。"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.paths import resolve_roots
from src.universal_replication_analysis import analyze

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ("project-root","data-root","results-root"):parser.add_argument("--"+n)
    parser.add_argument("--hpc",action="store_true")
    parser.add_argument("--seed1-threshold-run",help="可選：明確 pin 歷史 seed-1 Stage-3 completed full attempt")
    args=parser.parse_args()
    print(analyze(resolve_roots(args),args.seed1_threshold_run))
    return 0

if __name__=="__main__":raise SystemExit(main())
