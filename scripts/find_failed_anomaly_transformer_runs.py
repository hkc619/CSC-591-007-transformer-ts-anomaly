"""列出 AT missing/failed tasks；不重試、不提交 jobs。"""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.anomaly_transformer.protocol import configuration, namespace, validate_manifest
from src.anomaly_transformer.artifacts import discover


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    args = parser.parse_args()
    roots = resolve_roots(args)
    config, _ = configuration(roots.project)
    rows = validate_manifest(roots.project / "configs/anomaly_transformer_runs.csv", config)
    failures, _ = discover(namespace(roots.results), rows)
    print(json.dumps(failures, indent=2))
    return 1 if failures["count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
