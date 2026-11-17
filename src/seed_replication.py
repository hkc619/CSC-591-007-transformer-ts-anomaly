"""多 seed orchestration 的純設定與輸出隔離；不更改既有 research implementation。"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

FIELDS = ["task_id", "base_task_id", "setting", "target_machine", "source_machines",
          "protocol", "seed", "epochs", "batch_size", "eval_batch_size"]
PROTECTED = ("main.py", "preprocess.py", "src/models.py", "src/dlutils.py", "src/constants.py",
             "src/pot.py", "src/spot.py", "src/training_protocol.py", "src/cross_machine.py",
             "src/experiment.py", "src/parser.py", "src/deployment.py")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration(project):
    config = read_json(Path(project) / "configs/seed_replication.json")
    base = read_json(Path(project) / "configs/formal_protocol_v2.json")
    expected = dict(experiment="tranad_seed_replication_v1", base_protocol="v2", reference_seed=1,
        new_seeds=[2, 3], analysis_seeds=[1, 2, 3], namespace="seed-replication", model="TranAD",
        dataset="SMD", epochs=5, batch_size=128, eval_batch_size=128, window=10,
        learning_rate=0.0001, device="cpu", dtype="float64", training_shuffle=True,
        sampling="natural_source_size", calibration_included=False)
    if config != expected:
        raise ValueError("seed replication 固定新增 seeds=2,3；其他科學設定不可覆寫")
    keys = ("model", "dataset", "epochs", "batch_size", "eval_batch_size", "window",
            "learning_rate", "device", "dtype", "training_shuffle", "sampling")
    if (any(base[key] != config[key] for key in keys) or base["seed"] != 1 or base["protocol"] != "v2"
            or base["pot"] != dict(q=1e-5, level=0.99995, multiplier=1.06,
                                  mode="source_only_frozen_SPOT_extreme_quantile")):
        raise ValueError("原 formal protocol 與 seed replication 設定不一致")
    machines = base["machines"]
    if len(machines) != 28 or len(set(machines)) != 28:
        raise ValueError("canonical machine list 必須有 28 台")
    return config, machines


def expected_rows(project):
    config, machines = configuration(project)
    base = []
    for setting in ("per_machine", "cross_machine_zero_shot"):
        for target in machines:
            sources = [target] if setting == "per_machine" else [m for m in machines if m != target]
            base.append(dict(task_id=str(len(base)), setting=setting, target_machine=target,
                source_machines=json.dumps(sources), protocol="v2", seed="1", epochs="5",
                batch_size="128", eval_batch_size="128"))
    with (Path(project) / "configs/formal_28_machine_runs.csv").open(newline="") as stream:
        if list(csv.DictReader(stream)) != base:
            raise ValueError("原 seed=1 manifest 不符合 canonical 56-run mapping")
    rows = []
    for seed in config["new_seeds"]:
        for original in base:
            row = dict(original, base_task_id=original["task_id"], task_id=str(len(rows)), seed=str(seed))
            rows.append({key: row[key] for key in FIELDS})
    return rows


def manifest(project):
    path = Path(project) / "configs/seed_replication_runs.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("seed manifest schema 不符")
        rows = list(reader)
    if rows != expected_rows(project):
        raise ValueError("seed manifest 必須為 112 tasks，且 canonical order / sources / seed 完全一致")
    return rows


def safe_path(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError(f"禁止 symlink 改寫輸出 namespace：{path}")
    return path


def namespace(results):
    base = Path(results).resolve()
    forbidden = {"seed-replication", "calibration", "anomaly_transformer", "runs", "checkpoints", "summaries", "task_status"}
    if forbidden.intersection(base.parts):
        raise ValueError("RESULTS_ROOT 應為共同外部 root，不能指向既有 experiment 子目錄")
    return safe_path(base / "seed-replication")


def task_parent(results, row):
    return safe_path(namespace(results) / "seeds" / ("seed-" + row["seed"]) / "tasks" /
                     f"{int(row['base_task_id']):03d}-{row['setting']}-{row['target_machine']}")


def unique_name(prefix):
    return datetime.now(timezone.utc).strftime(prefix + "-%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8]


def latest_attempt(results, row):
    attempts = sorted(task_parent(results, row).glob("attempt-*"))
    if not attempts:
        return None
    folder = safe_path(attempts[-1])
    path = folder / "status.json"
    if not path.is_file():
        raise ValueError(f"最近 attempt 缺 status，禁止回退舊結果：{folder}")
    status = read_json(path)
    if any(status.get(key) != row[key] for key in ("task_id", "base_task_id", "seed", "setting", "target_machine")):
        raise ValueError("attempt status identity / seed 不符")
    return folder, status


def claim_attempt(results, row, retry=False, recover_interrupted=False):
    if recover_interrupted and not retry:
        raise ValueError("recover-interrupted 必須搭配 --retry 並先人工確認 job 已終止")
    parent = task_parent(results, row)
    if parent.exists() and not retry:
        raise FileExistsError("task 已存在，預設拒絕重跑；失敗重試須明確 --retry，且保留原 attempt")
    parent.mkdir(parents=True, exist_ok=retry)
    lock = parent / "active.lock"
    with lock.open("x") as stream:
        json.dump(dict(row=row, acquired_at=datetime.now(timezone.utc).isoformat()), stream)
    try:
        existing = latest_attempt(results, row)
        if existing and existing[1].get("completed"):
            raise FileExistsError("task 已完成，即使 --retry 也不覆寫或重訓成功結果")
        if existing and existing[1].get("state") == "running" and not recover_interrupted:
            raise RuntimeError("最近 attempt 仍標記 running；先人工確認 job 狀態")
        folder = safe_path(parent / unique_name("attempt"))
        folder.mkdir(exist_ok=False)
        return folder, lock
    except BaseException:
        lock.unlink()
        raise


def worker_args(roots, row, attempt, dry_run=False):
    """獨立入口的固定 args adapter；不修改或放寬原 seed=1 parser。"""
    if row not in manifest(roots.project):
        raise ValueError("worker row 不在固定 manifest")
    attempt = safe_path(attempt)
    if attempt.parent != task_parent(roots.results, row):
        raise ValueError("worker attempt 不在此 seed / task 的獨立 namespace")
    if not dry_run and (not attempt.is_dir() or not (attempt / "launch.json").is_file()):
        raise ValueError("worker 只能執行已由 launcher claim 的新 attempt")
    cross = row["setting"] == "cross_machine_zero_shot"
    return argparse.Namespace(dataset="SMD", model="TranAD", machine=row["target_machine"],
        train_machines=json.loads(row["source_machines"]) if cross else None,
        test_machine=row["target_machine"] if cross else None,
        epochs=5, batch_size=128, eval_batch_size=128, seed=int(row["seed"]), protocol="v2",
        source_only_pot=True, progress=True, retrain=True, test=False, less=False, checkpoint=None,
        output_dir=str(attempt / "runs"), project_root=str(roots.project), data_root=str(roots.data),
        results_root=str(attempt), hpc=True, dry_run=dry_run, smoke_test=False)


def claim_worker(attempt, row, command, args):
    """即使直接重放內部 worker command，也不能在相同 attempt 再執行一次。"""
    attempt = safe_path(attempt)
    launch = read_json(attempt / "launch.json")
    status = read_json(attempt / "status.json")
    lock = attempt.parent / "active.lock"
    if (launch.get("row") != row or launch.get("worker_results_root") != str(attempt)
            or status.get("state") != "running" or status.get("completed") is not False
            or not lock.is_file() or read_json(lock).get("row") != row):
        raise ValueError("worker 必須屬於目前由 launcher 持有的 running attempt")
    write_json(attempt / "worker_start.json", dict(command=command, effective_args=vars(args),
        row=row, source_parser_adapter="new subprocess only; original parser file unchanged"))
