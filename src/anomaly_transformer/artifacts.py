"""AT attempts 與 failure discovery；只讀既有結果，沒有自動 retry。"""
import json
from pathlib import Path
from src.anomaly_transformer.protocol import logical_id


REQUIRED = ("status.json", "metadata.json", "evaluation.json", "scores.npz", "training_history.json", "run.log")


def discover(out, rows):
    records, failed = {}, []
    for row in rows:
        task = int(row["task_id"])
        parent = Path(out) / "runs" / logical_id(row["target"], row["setting"])
        if parent.is_symlink():
            raise ValueError("run directory 不接受 symlink")
        attempts = sorted(parent.glob("attempt-*"))
        if not attempts:
            failed.append(task)
            continue
        folder = attempts[-1]
        if folder.is_symlink() or folder.resolve().parent != parent.resolve():
            raise ValueError("attempt path 不合法")
        status_path = folder / "status.json"
        if not status_path.is_file():
            failed.append(task)
            continue
        status = json.loads(status_path.read_text())
        if (status["task_id"] != task or status["target"] != row["target"]
                or status["setting"] != row["setting"] or Path(status["run_path"]).resolve() != folder.resolve()):
            raise ValueError("status / manifest identity 不符")
        records[task] = status
        if not status["completed"] or status["exit_code"] != 0 or any(not (folder / name).is_file() for name in REQUIRED):
            failed.append(task)
    return dict(failed_or_missing_task_ids=failed, count=len(failed)), records
