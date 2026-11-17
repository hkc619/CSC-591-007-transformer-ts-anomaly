"""獨立 seeds=2,3 launcher：attempt / lock / log 隔離；不提交 Slurm jobs。"""
import argparse
import codecs
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.seed_replication import (PROTECTED, claim_attempt, file_hash, manifest, namespace,
                                  task_parent, write_json)


def worker_command(roots, row, attempt, dry_run=False):
    command = [sys.executable, "-u", str(ROOT / "scripts/run_seed_replication_worker.py"),
        "--project-root", str(roots.project), "--data-root", str(roots.data),
        "--results-root", str(roots.results), "--task-id", row["task_id"], "--attempt-dir", str(attempt)]
    if dry_run:
        command.append("--dry-run")
    return command


def execute(roots, row, retry=False, dry_run=False, recover_interrupted=False):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="")
    if dry_run:
        # 不 claim、不建 cache/results，不初始化模型；原 describe_experiment 只讀資料。
        attempt = task_parent(roots.results, row) / "dry-run"
        command = worker_command(roots, row, attempt, True)
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        description = json.loads(result.stdout)
        description.update(seed_replication_task=row, task_output=str(task_parent(roots.results, row)),
            output_namespace=str(namespace(roots.results)), checkpoint_root=str(attempt / "checkpoints"),
            training_started=False, worker_command=command)
        print(json.dumps(description, ensure_ascii=False, indent=2))
        return 0
    attempt, lock = claim_attempt(roots.results, row, retry, recover_interrupted)
    status = dict(row, state="running", completed=False, exit_code=None, run_path=None,
                  started_at=datetime.now(timezone.utc).isoformat(), attempt=str(attempt))
    child, code = None, 1
    def terminate(signum, frame):
        raise KeyboardInterrupt("收到 SIGTERM，終止 worker 並保留失敗 attempt")
    previous_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        command = worker_command(roots, row, attempt)
        snapshot = attempt / "deployment_source"
        files = [*PROTECTED, "configs/formal_protocol_v2.json", "configs/formal_28_machine_runs.csv",
                 "configs/seed_replication.json", "configs/seed_replication_runs.csv", "src/seed_replication.py",
                 "scripts/run_seed_replication.py", "scripts/run_seed_replication_worker.py"]
        hashes = {}
        for name in files:
            path = ROOT / name
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            hashes[name] = file_hash(path)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        write_json(attempt / "launch.json", dict(command=command, launcher_command=[sys.executable, *sys.argv],
            row=row, source_sha256=hashes,
            git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
            git_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            project_root=str(roots.project), data_root=str(roots.data), base_results_root=str(roots.results),
            worker_results_root=str(attempt), retry=retry, recover_interrupted=recover_interrupted,
            slurm_job_id=os.getenv("SLURM_JOB_ID")))
        write_json(attempt / "status.json", status)
        print("Seed replication attempt:", attempt, flush=True)
        pending = ""
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        with (attempt / "run.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while True:
                    chunk = os.read(child.stdout.fileno(), 8192)
                    if not chunk:
                        break
                    log.write(chunk)
                    log.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    if status["run_path"] is None:
                        pending += decoder.decode(chunk)
                        found = re.search(r"Run directory: ([^\r\n]+)[\r\n]", pending)
                        if found:
                            status["run_path"] = found.group(1)
                code = child.wait()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
            finally:
                child.stdout.close()
        if code == 0:
            run = Path(status["run_path"] or "").resolve()
            if run.parent != attempt / "runs" or not (run / "evaluation.json").is_file():
                raise RuntimeError("worker 成功但缺少合法 run / evaluation，不標記 completed")
            metadata = json.loads((run / "metadata.json").read_text())
            expected = dict(seed=int(row["seed"]), setting=row["setting"], target_machine=row["target_machine"],
                            epochs=5, batch_size=128, eval_batch_size=128, smoke_test=False)
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise RuntimeError("worker metadata 與 seed manifest 不一致")
            if any(file_hash(ROOT / name) != value for name, value in hashes.items()):
                raise RuntimeError("執行期間 checkout source 改變，不能標記 completed")
        status.update(state="completed" if code == 0 else "failed", completed=code == 0)
    except BaseException as error:
        code = code or 1
        status.update(state="failed", completed=False, error=repr(error))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        status.update(exit_code=code, finished_at=datetime.now(timezone.utc).isoformat())
        # 只更新本次 attempt 的 live status；不動任何舊 attempt。
        temporary = attempt / "final-status.json"
        write_json(temporary, status)
        temporary.replace(attempt / "status.json")
        lock.unlink()
    return code


def main():
    parser = argparse.ArgumentParser(description="TranAD v2 seeds=2,3，原 seed=1 保持唯讀")
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry", action="store_true")
    parser.add_argument("--recover-interrupted", action="store_true",
                        help="人工確認已終止並保存 stale lock 後，允許保留 running 的舊 attempt 另開新 attempt")
    args = parser.parse_args()
    roots = resolve_roots(args)
    rows = manifest(ROOT)
    if not 0 <= args.task_id < len(rows):
        parser.error("task-id 必須為 0..111")
    return execute(roots, rows[args.task_id], args.retry, args.dry_run, args.recover_interrupted)


if __name__ == "__main__":
    raise SystemExit(main())
