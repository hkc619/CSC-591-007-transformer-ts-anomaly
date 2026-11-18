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


def experiment_id(target, fraction, smoke=False, seed=1):
    if not re.fullmatch(r"machine-\d+-\d+", target):
        raise ValueError("target 名稱不合法")
    percent = int(Decimal(canonical_fraction(fraction)) * 100)
    if isinstance(seed, bool) or int(seed) != seed or seed < 0:
        raise ValueError("seed 必須為非負整數")
    return f"{'smoke-' if smoke else ''}calib-{target}-f{percent:03d}-seed{seed}"


def new_attempt(namespace, target, fraction, smoke=False, retry=False, seed=1):
    logical_id = experiment_id(target, fraction, smoke, seed)
    parent = Path(namespace) / "runs" / logical_id
    if parent.is_symlink() or parent.resolve().parent != (Path(namespace) / "runs").resolve():
        raise ValueError("experiment directory 不可離開 calibration namespace")
    # 固定 experiment ID；預設重複操作不產生新實驗，重試需明確 --retry。
    if parent.exists() and not retry:
        raise FileExistsError(f"experiment 已存在：{parent}；先檢查結果，需重試時明確指定 --retry")
    if seed != 1 and parent.exists():
        for status in parent.glob("attempt-*/status.json"):
            try:
                record = json.loads(status.read_text())
            except (ValueError, OSError):
                continue  # 明確 retry 可另開 attempt，保留損壞的舊 status 供稽核。
            if record.get("completed") and record.get("exit_code") == 0:
                raise FileExistsError("此 seed 的 calibration 已成功，不重新執行；舊 attempt 保持不變")
    parent.mkdir(parents=True, exist_ok=retry)
    attempt_id = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8]
    folder = parent / attempt_id
    folder.mkdir(exist_ok=False)
    return logical_id, attempt_id, folder


def validate_attempt_path(namespace, folder, target, fraction, smoke=False, seed=1):
    folder = Path(folder)
    expected_parent = Path(namespace) / "runs" / experiment_id(target, fraction, smoke, seed)
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
        seed = int(row["seed"])
        parent = Path(namespace) / "runs" / experiment_id(row["target"], row["fraction"], seed=seed)
        if not parent.exists():
            continue
        attempts = sorted(path for path in parent.iterdir() if path.is_dir())
        if not attempts:
            continue
        folder = validate_attempt_path(namespace, attempts[-1], row["target"], row["fraction"], seed=seed)
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
                or record.get("seed", 1) != seed
                or Path(record["run_path"]).resolve() != folder or record.get("smoke_test") is not False):
            raise ValueError("status identity 與 manifest / attempt 不一致")
        records[task_id] = record
    return records


def failed_calibration_tasks(namespace, rows):
    records, failed, details = {}, [], []
    for row in rows:
        task = int(row["task_id"])
        try:
            records.update(latest_calibration_status(namespace, [row]))
        except (ValueError, OSError, KeyError) as error:
            failed.append(task)
            details.append(dict(task_id=task, reason=str(error)))
            continue
        record = records.get(task)
        if not record or not record.get("completed") or record.get("exit_code") != 0:
            failed.append(task)
            continue
        folder = Path(record["run_path"])
        if any(not (folder / name).is_file() or (folder / name).is_symlink() for name in REQUIRED_ARTIFACTS):
            failed.append(task)
            continue
        try:
            metadata = json.loads((folder / "metadata.json").read_text())
            checks = json.loads((folder / "validation.json").read_text())
            if (metadata["seed"] != int(row["seed"]) or metadata["target"] != row["target"]
                    or canonical_fraction(metadata["fraction"]) != row["fraction"]
                    or metadata["smoke_test"] is not False or metadata["model_retrained"] is not False
                    or metadata["optimizer_steps"] != 0 or metadata["parameters_unchanged"] is not True
                    or metadata["pot"]["stream_score_count"] != 0):
                raise ValueError("metadata seed / identity / frozen / POT invariant 不符")
            for key in ("parameter_freeze", "test_scores_unchanged", "labels_unchanged",
                        "pr_auc_unchanged", "ap_unchanged", "roc_auc_unchanged", "formal_artifacts_unchanged"):
                if checks.get(key) is not True:
                    raise ValueError(f"validation 不通過：{key}")
        except (ValueError, OSError, KeyError) as error:
            failed.append(task)
            details.append(dict(task_id=task, reason=str(error)))
    return dict(failed_or_missing_task_ids=failed, count=len(failed), invalid_attempts=details), records
