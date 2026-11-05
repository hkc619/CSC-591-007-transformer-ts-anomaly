"""Calibration 固定 identity、獨立 attempts 與 failure discovery；不載入研究模型。"""
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
from uuid import uuid4

from src.calibration_protocol import canonical_fraction


REQUIRED_ARTIFACTS = ("launch.json", "metadata.json", "status.json", "evaluation.json",
                      "threshold.json", "validation.json", "scores.npz", "calibration_scores.npz",
                      "config.json", "run.log", "smoke_summary.json")


def experiment_id(target, fraction, smoke=False):
    if not re.fullmatch(r"machine-\d+-\d+", target):
        raise ValueError("target 名稱不合法")
    percent = int(Decimal(canonical_fraction(fraction)) * 100)
    return f"{'smoke-' if smoke else ''}calib-{target}-f{percent:03d}-seed1"


def new_attempt(namespace, target, fraction, smoke=False, retry=False):
    logical_id = experiment_id(target, fraction, smoke)
    parent = Path(namespace) / "runs" / logical_id
    if parent.is_symlink() or parent.resolve().parent != (Path(namespace) / "runs").resolve():
        raise ValueError("experiment directory 不可離開 calibration namespace")
    # 固定 experiment ID；預設重複操作不產生新實驗，重試需明確 --retry。
    if parent.exists() and not retry:
        raise FileExistsError(f"experiment 已存在：{parent}；先檢查結果，需重試時明確指定 --retry")
    parent.mkdir(parents=True, exist_ok=retry)
    attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8]
    folder = parent / attempt_id
    folder.mkdir(exist_ok=False)
    return logical_id, attempt_id, folder


def validate_attempt_path(namespace, folder, target, fraction, smoke=False):
    folder = Path(folder)
    expected_parent = Path(namespace) / "runs" / experiment_id(target, fraction, smoke)
    if (folder.is_symlink() or expected_parent.is_symlink()
            or folder.resolve().parent != expected_parent.resolve()
            or expected_parent.resolve().parent != (Path(namespace) / "runs").resolve()
            or not re.fullmatch(r"attempt-\d{8}T\d{12}Z-[0-9a-f]{8}", folder.name)):
        raise ValueError("calibration attempt 路徑或 identity 不合法")
    return folder.resolve()


def latest_calibration_status(namespace, rows):
    """掃描 attempts，連被強制中止且無結尾 status 的最近 attempt 也不忽略。"""
    records = {}
    for row in rows:
        task_id = int(row["task_id"])
        parent = Path(namespace) / "runs" / experiment_id(row["target"], row["fraction"])
        if not parent.exists():
            continue
        attempts = sorted(path for path in parent.iterdir() if path.is_dir())
        if not attempts:
            continue
        folder = validate_attempt_path(namespace, attempts[-1], row["target"], row["fraction"])
        path = folder / "status.json"
        if not path.is_file():
            records[task_id] = dict(task_id=task_id, run_path=str(folder), completed=False,
                                    exit_code=None, reason="最近 attempt 缺 status，可能仍執行中或被中止")
            continue
        if path.is_symlink():
            raise ValueError("status 不接受 symlink")
        record = json.loads(path.read_text())
        if (record.get("task_id") != task_id or record.get("target") != row["target"]
                or canonical_fraction(record["fraction"]) != row["fraction"]
                or Path(record["run_path"]).resolve() != folder or record.get("smoke_test") is not False):
            raise ValueError("status identity 與 manifest / attempt 不一致")
        records[task_id] = record
    return records


def failed_calibration_tasks(namespace, rows):
    records = latest_calibration_status(namespace, rows)
    failed = []
    for row in rows:
        task = int(row["task_id"])
        record = records.get(task)
        if not record or not record.get("completed") or record.get("exit_code") != 0:
            failed.append(task)
            continue
        folder = Path(record["run_path"])
        if any(not (folder / name).is_file() or (folder / name).is_symlink() for name in REQUIRED_ARTIFACTS):
            failed.append(task)
    return dict(failed_or_missing_task_ids=failed, count=len(failed)), records
