"""以 canonical seed-replication manifest 定位同 seed LOMO；禁止跨 seed fallback。"""
import json
from pathlib import Path

from src.seed_replication import latest_attempt, manifest, read_json, safe_path, task_parent


def seed_formal_run(project, formal_root, target, seed):
    rows = manifest(project)
    selected = [row for row in rows if int(row["seed"]) == seed and row["target_machine"] == target
                and row["setting"] == "cross_machine_zero_shot"]
    if len(selected) != 1:
        raise ValueError(f"沒有 canonical formal mapping：target={target}, seed={seed}, cross_machine")
    row = selected[0]
    record = latest_attempt(Path(formal_root), row)
    if record is None:
        raise FileNotFoundError(f"缺少此 seed 的 formal task：{task_parent(formal_root, row)}")
    attempt, status = record
    if not status.get("completed") or status.get("exit_code") != 0 or not status.get("run_path"):
        raise ValueError("同 seed 的最近 formal attempt 尚未成功；不回退其他 seed 或舊成功")
    completed = []
    for path in task_parent(formal_root, row).glob("attempt-*/status.json"):
        candidate = read_json(path)
        if candidate.get("completed") and candidate.get("exit_code") == 0:
            completed.append(path)
    if len(completed) != 1:
        raise ValueError("同 target/seed 有多個 successful formal attempts，必須先人工釐清，不猜選")
    run = safe_path(Path(status["run_path"]))
    if run.parent != attempt / "runs":
        raise ValueError("formal run_path 離開同 seed 的 attempt，禁止 alias 其他 seed")
    launch = read_json(attempt / "launch.json")
    worker = read_json(attempt / "worker_start.json")
    if launch.get("row") != row or worker.get("row") != row:
        raise ValueError("formal launch/worker seed manifest identity 不符")
    args = worker["effective_args"]
    fixed = dict(seed=seed, model="TranAD", dataset="SMD", protocol="v2", epochs=5,
                 batch_size=128, eval_batch_size=128, smoke_test=False, checkpoint=None,
                 test_machine=target, train_machines=json.loads(row["source_machines"]))
    if any(args.get(key) != value for key, value in fixed.items()):
        raise ValueError("formal worker 實際 args 的 seed/target/sources/config 不符")
    return run, attempt / "status.json", row, attempt
