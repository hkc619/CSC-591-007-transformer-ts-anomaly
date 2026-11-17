"""多 seed 分析入口；seed=1 唯讀，新數據和圖一律另存新 analysis attempt。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.seed_replication_analysis import analyze


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--reference-results-root", type=Path,
                        help="舊 seed=1 的外部 RESULTS_ROOT，預設與 RESULTS_ROOT 相同；只讀")
    args = parser.parse_args()
    roots = resolve_roots(args)
    print("新的分析目錄：", analyze(roots, args.reference_results_root or roots.results))


if __name__ == "__main__":
    main()
