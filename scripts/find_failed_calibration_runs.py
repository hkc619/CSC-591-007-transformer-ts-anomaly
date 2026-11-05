"""只列出 calibration 缺失／失敗 tasks，不重試、不提交 jobs、不改 formal artifacts。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_artifacts import failed_calibration_tasks
from src.calibration_protocol import calibration_directory, load_configuration, validate_manifest
from src.paths import overlaps, resolve_roots


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root", "formal-results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    args = parser.parse_args()
    roots = resolve_roots(args)
    formal = Path(args.formal_results_root or os.environ.get("FORMAL_RESULTS_ROOT") or roots.results).expanduser().resolve()
    if overlaps(formal, roots.project) or overlaps(formal, roots.data):
        raise ValueError("formal root 不可與 code / data 重疊")
    namespace = calibration_directory(roots.results, formal)
    _, machines = load_configuration(roots.project)
    rows = validate_manifest(roots.project / "configs/calibration_runs.csv", machines)
    result, _ = failed_calibration_tasks(namespace, rows)
    print(json.dumps(result, indent=2))
    return 1 if result["count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
