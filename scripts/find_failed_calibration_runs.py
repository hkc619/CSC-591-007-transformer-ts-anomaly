"""只列出 calibration 缺失／失敗 tasks，不重試、不提交 jobs、不改 formal artifacts。"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_artifacts import failed_calibration_tasks
from src.calibration_protocol import calibration_directory, load_configuration, manifest_seeds, validate_manifest
from src.paths import overlaps, resolve_roots


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root", "formal-results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/calibration_runs.csv")
    args = parser.parse_args()
    roots = resolve_roots(args)
    formal = Path(args.formal_results_root or os.environ.get("FORMAL_RESULTS_ROOT") or roots.results).expanduser().resolve()
    if overlaps(formal, roots.project) or overlaps(formal, roots.data):
        raise ValueError("formal root 不可與 code / data 重疊")
    _, machines = load_configuration(roots.project)
    path = args.manifest if args.manifest.is_absolute() else roots.project / args.manifest
    seeds = manifest_seeds(path)
    if 1 in seeds and len(seeds) > 1:
        raise ValueError("failure check 請分開 seed=1 與 seed=2/3 manifests")
    namespace = calibration_directory(roots.results, formal, seeds[0])
    rows = validate_manifest(path, machines, seeds)
    result, _ = failed_calibration_tasks(namespace, rows)
    print(json.dumps(result, indent=2))
    return 1 if result["count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
