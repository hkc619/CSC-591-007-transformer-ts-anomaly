"""僅列出 112-task matrix 缺失/失敗；不自動重試或提交 jobs。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.seed_replication import manifest
from src.seed_replication_analysis import failed_tasks


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    args = parser.parse_args()
    roots = resolve_roots(args)
    result = failed_tasks(roots.results, manifest(ROOT))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
