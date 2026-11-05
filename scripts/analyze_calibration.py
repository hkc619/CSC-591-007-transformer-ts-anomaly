"""84 個 calibration tasks 全部成功後才寫獨立摘要；不執行 inference / POT。"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.calibration_analysis import summarize_run, write_outputs
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
        raise ValueError("formal root 不能位於 code / data")
    namespace = calibration_directory(roots.results, formal)
    config, machines = load_configuration(roots.project)
    rows = validate_manifest(roots.project / "configs/calibration_runs.csv", machines)
    failures, statuses = failed_calibration_tasks(namespace, rows)
    if failures["count"]:
        raise ValueError(f"84 tasks 尚未全部成功；{failures}")
    results, identities = [], {}
    for row in rows:
        folder = Path(statuses[int(row["task_id"])]["run_path"])
        record, meta = summarize_run(folder, row, config, namespace, machines)
        target = row["target"]
        identity = (meta["formal_run_id"], meta["formal_checkpoint_hash"],
                    meta["formal_score_artifact_hash"], meta["test_scores_sha256"], meta["labels_sha256"],
                    meta["protected_input_sha256"]["evaluation.json"])
        if target in identities and identity != identities[target]:
            raise ValueError("同 target 的三個 fractions 必須使用相同 formal run / model / scores / labels / evaluation")
        identities[target] = identity
        results.append(record)
    folder = namespace / "summaries" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    write_outputs(folder, results, config)
    print("Calibration summary：", folder)


if __name__ == "__main__":
    main()
