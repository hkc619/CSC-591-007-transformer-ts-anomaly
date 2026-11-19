"""Universal Stage 1：沿用 formal 計算，只協調 TRAIN-only I/O 與獨立 artifacts。"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib
from importlib import metadata as packages
import json
import math
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
from time import perf_counter
import types
import traceback

import numpy as np

from src.seed_replication import PROTECTED, file_hash, read_json, safe_path, unique_name, write_json

ROOT = Path(__file__).resolve().parents[1]
TRAINING_KEYS = ("protocol", "model", "dataset", "epochs", "batch_size", "window",
                 "learning_rate", "device", "dtype", "training_shuffle", "sampling")


def configuration(project, seed):
    config = read_json(Path(project) / "configs/universal_training.json")
    expected = dict(experiment="universal_train_only_v1",
        protocol_source="configs/formal_protocol_v2.json", namespace="universal",
        allowed_seeds=[1, 2, 3], stage1_validation_seed=1,
        test_evaluation=False, threshold_fitting=False, calibration=False)
    if config != expected or type(seed) is not int or seed not in config["allowed_seeds"]:
        raise ValueError("Universal 設定不符；只接受 seeds=1,2,3，Stage 1 只驗證 seed=1")
    base = read_json(Path(project) / config["protocol_source"])
    training = {key: base[key] for key in TRAINING_KEYS}
    if training != dict(protocol="v2", model="TranAD", dataset="SMD", epochs=5,
            batch_size=128, window=10, learning_rate=1e-4, device="cpu", dtype="float64",
            training_shuffle=True, sampling="natural_source_size"):
        raise ValueError("Universal 必須沿用 formal protocol-v2 training 設定")
    if base["eval_batch_size"] != 128:
        raise ValueError("Formal adapter 的 eval batch 設定不符（此流程不執行 evaluation）")
    machines = base["machines"]
    if len(machines) != 28 or len(set(machines)) != 28:
        raise ValueError("Canonical source list 必須恰有 28 台、不重複")
    # 名單只取自 formal config；不另定義第二套 machine ordering。
    if any(not isinstance(m, str) or Path(m).name != m or not m.startswith("machine-")
           for m in machines):
        raise ValueError("不合法的 canonical machine 名稱")
    return dict(training, seed=seed, sources=machines, target=None,
                training_mode="universal", eval_batch_size=base["eval_batch_size"])


def array_hash(array):
    import hashlib
    array = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def inspect_train(roots, config):
    """只讀 TRAIN；dry-run 不 import model、不建立 cache 或 result directory。"""
    train = roots.data / "train"
    if train.resolve() != train.absolute():
        raise ValueError("TRAIN directory 不可透過 symlink 指向其他 split")
    actual = {p.stem for p in train.glob("machine-*.txt")}
    expected = set(config["sources"])
    if actual != expected:
        raise ValueError(f"TRAIN machine 集合不符；missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    records, offset = [], 0
    for machine in config["sources"]:
        path = train / (machine + ".txt")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"TRAIN 檔案不存在或為 symlink：{path}")
        # 與 preprocess.load_and_save('train', ...) 使用同一讀法；不新增 normalization。
        raw = np.genfromtxt(path, dtype=np.float64, delimiter=",")
        if raw.ndim != 2 or raw.shape[1] != 38 or not len(raw) or not np.isfinite(raw).all():
            raise ValueError(f"TRAIN 必須非空、38 features 且 finite：{path}")
        count = len(raw)  # validated padded windowing 每 raw timestamp 產生一個 window。
        records.append(dict(machine=machine, train_path=str(path), raw_train_length=count,
            window_count=count, feature_dimension=38, offset_start=offset,
            offset_end_exclusive=offset+count, dropped_points=0,
            raw_file_sha256=file_hash(path), raw_array_sha256=array_hash(raw)))
        offset += count
    return dict(machine_count=len(records), total_training_windows=offset,
                window_size=config["window"], machines=records)


def namespace(results):
    root = Path(results).resolve()
    forbidden = {"universal", "calibration", "seed-replication", "anomaly_transformer",
                 "formal-28-machine", "runs", "checkpoints", "summaries", "task_status"}
    if forbidden.intersection(root.parts):
        raise ValueError("RESULTS_ROOT 必須是共同外部 root，不可指向既有 experiment 子目錄")
    return safe_path(root / "universal")


def run_identity(seed):
    if type(seed) is not int or seed not in (1, 2, 3):
        raise ValueError("不合法的 seed")
    return f"universal_TranAD_SMD_v2_e5_bs128_seed{seed}"


def describe(roots, config, manifest):
    logical = run_identity(config["seed"])
    base = namespace(roots.results)
    return dict(config=config, train_manifest=manifest,
        run_parent=str(base / "runs" / logical),
        checkpoint_pattern=str(base / "checkpoints" / logical / "<attempt-id>" / "model.ckpt"),
        window_validation="實際執行時由 windows_for_machine 逐 window 比對原始 TRAIN timestamps",
        raw_concatenation_before_windowing=False, test_data_required=False,
        test_evaluation=False, threshold_fitting=False, calibration=False,
        estimated_window_tensor_bytes=manifest["total_training_windows"]*config["window"]*38*8)


def claim_attempt(results, seed, retry=False):
    parent = safe_path(namespace(results) / "runs" / run_identity(seed))
    if parent.exists() and not retry:
        raise FileExistsError(f"Universal run 已存在；先檢查 status，失敗重試明確指定 --retry：{parent}")
    parent.mkdir(parents=True, exist_ok=retry)
    lock = parent / "active.lock"
    with lock.open("x") as stream:
        stream.write(datetime.now(timezone.utc).isoformat())
    try:
        attempts = sorted(parent.glob("attempt-*"))
        for folder in attempts:
            status_file = safe_path(folder / "status.json")
            if not status_file.is_file():
                raise RuntimeError("既有 attempt 缺 status；請先人工確認，不可靜默重跑")
            status = read_json(status_file)
            if status.get("state") != "failed" or status.get("completed") is not False:
                raise FileExistsError("既有 attempt 已完成或狀態未確認；禁止重跑／覆寫")
        folder = safe_path(parent / unique_name("attempt"))
        folder.mkdir(exist_ok=False)
        checkpoint = safe_path(namespace(results) / "checkpoints" / run_identity(seed) / folder.name / "model.ckpt")
        return folder, checkpoint, lock
    except BaseException:
        lock.unlink()
        raise


def update_status(folder, **values):
    """只更新目前獨立 attempt 的狀態，已成功的 attempt 不可再改。"""
    path = safe_path(folder / "status.json")
    if path.exists() and read_json(path).get("completed"):
        raise FileExistsError("已完成的 universal attempt 為唯讀")
    temporary = folder / "status.pending.json"
    write_json(temporary, dict(values, updated_at=datetime.now(timezone.utc).isoformat()))
    temporary.replace(path)


def core_args(roots, config, folder, checkpoint):
    # I/O adapter：原 main.py / src.parser 保持逐字不變，不假造 held-out target。
    return argparse.Namespace(model="TranAD", dataset="SMD", machine=None, test_machine=None,
        train_machines=list(config["sources"]), epochs=5, batch_size=128, eval_batch_size=128,
        seed=config["seed"], protocol="v2", source_only_pot=True,
        progress=True, retrain=True, test=False, less=False, checkpoint=None,
        save_checkpoint=str(checkpoint), output_dir=str(folder),
        project_root=str(roots.project), data_root=str(roots.data), results_root=str(roots.results),
        hpc=False, dry_run=False, smoke_test=False)


def import_core(args):
    """與既有 seed worker 相同的 fresh-process args adapter；不執行 main guard。"""
    if "main" in sys.modules or "src.parser" in sys.modules:
        raise RuntimeError("Universal training 必須在新的 Python process 執行，避免混用其他實驗 args")
    adapter = types.ModuleType("src.parser")
    adapter.args = args
    sys.modules["src.parser"] = adapter
    return importlib.import_module("main")


def assemble_training(config, manifest, model, core):
    """重用原 TRAIN loader、windowing 及逐 window 驗證後，才 concatenate。"""
    import torch
    from src.cross_machine import load_training_machine, windows_for_machine
    chunks, raw_chunks, records = [], [], []
    offset = 0
    for index, record in enumerate(manifest["machines"]):
        machine = record["machine"]
        raw = load_training_machine(machine)
        if (len(raw) != record["raw_train_length"] or not torch.isfinite(raw).all()
                or raw.dtype != torch.float64 or array_hash(raw.numpy()) != record["raw_array_sha256"]):
            raise ValueError(f"TRAIN 在 preflight 後改變：{machine}")
        windows = windows_for_machine(raw, model, core.convert_to_windows)
        if tuple(windows.shape) != (len(raw), 10, 38):
            raise ValueError(f"Window shape 不符：{machine}")
        if record["offset_start"] != offset or record["window_count"] != len(windows):
            raise ValueError("Concatenation offsets / count 不一致")
        chunks.append(windows)
        raw_chunks.append(raw)
        records.append(dict(record, windowed_before_concatenation=True,
            all_windows_single_machine=True, all_window_contents_validated=True,
            final_element_equals_target=True, raw_timestamp_start=0,
            raw_timestamp_end_exclusive=len(raw)))
        offset += len(windows)
        print(f"Windowing {index+1}/28: {machine}, windows={len(windows)}, offset=[{record['offset_start']},{offset})", flush=True)
    windows = torch.cat(chunks)
    raw = torch.cat(raw_chunks)  # 僅在各 machine 完成 windowing 後，配合原 backprop 的 dataO 介面。
    if offset != manifest["total_training_windows"] or len(windows) != offset or len(raw) != offset:
        raise ValueError("總 windows 不等於各 machine window counts 的總和")
    if [r["machine"] for r in records] != config["sources"]:
        raise ValueError("Sources 與 canonical ordering 不符")
    return windows, raw, dict(machine_count=len(records), total_training_windows=offset,
        window_size=10, all_windows_single_machine=True, windowed_before_concatenation=True,
        raw_concatenation_before_windowing=False, sampling="natural_source_size",
        sample_id_mapping="offset_start <= sample_id < offset_end_exclusive; timestamp=sample_id-offset_start",
        timestamp_mapping="window(t) = raw[max(0,t-9):t+1]，左側不足重複 raw[0]；target=raw[t]",
        machines=records)


def assert_state_equal(left, right):
    """Roundtrip 同時驗證 model、AdamW、scheduler 與原 checkpoint history。"""
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or not torch.equal(left, right):
            raise ValueError("Checkpoint tensor roundtrip 不一致")
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError("Checkpoint dict roundtrip 不一致")
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError("Checkpoint sequence roundtrip 不一致")
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    elif left != right:
        raise ValueError("Checkpoint value roundtrip 不一致")


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False


def git_info(project):
    def git(*args):
        result = subprocess.run(["git", *args], cwd=project, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    return dict(commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                dirty_status=git("status", "--short"))


def train(roots, config, manifest, folder, checkpoint):
    import torch
    args = core_args(roots, config, folder, checkpoint)
    core = import_core(args)
    from src.training_protocol import seeded_generator
    import preprocess
    # 與 prepare_run 相同；放在 src.models 的 import-time manual_seed(1) 之後。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.training_generator = seeded_generator(args.seed)
    preprocess.output_folder = str(folder / "cache/processed")

    snapshot = folder / "source_snapshot"
    paths = (*PROTECTED, "src/universal_training.py", "scripts/run_universal_training.py",
             "configs/universal_training.json", "configs/formal_protocol_v2.json")
    hashes = {}
    for name in paths:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(roots.project / name, destination)
        hashes[name] = file_hash(destination)
    metadata = dict(experiment="universal_train_only_v1", run_id=run_identity(args.seed),
        attempt_id=folder.name, **config, machine_count=28, target_train_exclusion=False,
        total_training_windows=manifest["total_training_windows"],
        no_held_out_target=True, test_loaded=False, test_evaluation=False,
        threshold_fitting=False, calibration=False, normalization_fitted=False,
        checkpoint_path=str(checkpoint), train_manifest=str(folder / "train_manifest.json"),
        provenance_path=str(folder / "window_provenance.json"), git=git_info(roots.project),
        python=sys.version, platform=platform.platform(), torch_version=torch.__version__,
        numpy_version=np.__version__, torch_num_threads=torch.get_num_threads(),
        installed_dependencies=sorted(f"{d.metadata['Name']}=={d.version}" for d in packages.distributions()),
        source_file_sha256=hashes,
        optimizer=dict(name="AdamW", learning_rate=1e-4, weight_decay=1e-5),
        scheduler=dict(name="StepLR", step_size=5, gamma=0.9),
        window_tensor_bytes=manifest["total_training_windows"]*10*38*8)
    write_json(folder / "metadata.json", metadata)
    model, optimizer, scheduler, epoch, history = core.load_model("TranAD", 38)
    if (epoch != -1 or history or model.n_window != 10 or model.batch != 128
            or model.lr != 1e-4 or any(p.dtype != torch.float64 or p.device.type != "cpu"
                                     for p in model.parameters())):
        raise ValueError("原 TranAD 初始化與固定 training protocol 不一致")
    window_start = perf_counter()
    windows, raw, provenance = assemble_training(config, manifest, model, core)
    write_json(folder / "window_provenance.json", provenance)
    print(f"TRAIN only: 28 machines, {len(windows)} windows; TEST/POT/calibration 均不執行", flush=True)
    window_seconds = perf_counter() - window_start
    model.train()
    epoch_records = []
    start = perf_counter()
    for epoch in range(config["epochs"]):
        update_status(folder, state="running", completed=False, seed=args.seed,
                      phase="training", epoch=epoch+1, epochs=5)
        epoch_start = perf_counter()
        loss, lr = core.backprop(epoch, model, windows, raw, optimizer, scheduler, training=True)
        if not math.isfinite(float(loss)):
            raise ValueError(f"Epoch {epoch+1} loss 非 finite")
        history.append((float(loss), float(lr)))
        epoch_records.append(dict(epoch=epoch+1, loss=float(loss), learning_rate=float(lr),
            seconds=perf_counter()-epoch_start, training_batches=math.ceil(len(windows)/128)))
        write_json(folder / f"epoch_{epoch+1}.json", epoch_records[-1])
        print(f"完成 epoch {epoch+1}/5: loss={float(loss):.10g}, seconds={epoch_records[-1]['seconds']:.3f}", flush=True)
    training_seconds = perf_counter()-start
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Training 後 model parameters 非 finite")
    core.save_model(model, optimizer, scheduler, epoch, history)
    before = file_hash(checkpoint)
    del windows, raw
    args.checkpoint, args.retrain = str(checkpoint), False
    try:
        reloaded, restored_optimizer, restored_scheduler, restored_epoch, restored_history = core.load_model("TranAD", 38)
    finally:
        args.checkpoint, args.retrain = None, True
    assert_state_equal(model.state_dict(), reloaded.state_dict())
    assert_state_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    assert_state_equal(scheduler.state_dict(), restored_scheduler.state_dict())
    assert_state_equal((epoch, history), (restored_epoch, restored_history))
    if file_hash(checkpoint) != before:
        raise ValueError("Checkpoint reload 修改了檔案")
    for name, digest in hashes.items():
        if file_hash(roots.project / name) != digest:
            raise ValueError(f"Training 期間 source code 改變：{name}")
    for record in manifest["machines"]:
        if file_hash(record["train_path"]) != record["raw_file_sha256"]:
            raise ValueError("Training 期間 TRAIN 檔案改變")
    validation = dict(checkpoint_reload_passed=True, model_state_equal=True,
        optimizer_state_equal=True, scheduler_state_equal=True, epoch_history_equal=True,
        checkpoint_sha256=before, checkpoint_size_bytes=checkpoint.stat().st_size,
        all_windows_single_machine=True, machine_count=28,
        total_training_windows=manifest["total_training_windows"], seed=args.seed,
        epochs_completed=5, test_loaded=False, threshold_fitting=False, calibration=False)
    write_json(folder / "training_history.json", dict(epochs=epoch_records,
        train_time_sec=training_seconds, windowing_time_sec=window_seconds))
    write_json(folder / "validation.json", validation)
    print(f"Checkpoint reload 完成：{checkpoint}\nTraining seconds: {training_seconds:.3f}", flush=True)
    return validation


def execute(roots, seed=1, retry=False, dry_run=False, command=None):
    config = configuration(roots.project, seed)
    manifest = inspect_train(roots, config)
    description = describe(roots, config, manifest)
    if dry_run:
        print(json.dumps(description, ensure_ascii=False, indent=2, allow_nan=False))
        return description
    folder, checkpoint, lock = claim_attempt(roots.results, seed, retry)
    try:
        update_status(folder, state="running", completed=False, seed=seed, phase="initializing")
        write_json(folder / "launch.json", dict(command=command or sys.argv,
            python_executable=sys.executable, project_root=str(roots.project),
            data_root=str(roots.data), results_root=str(roots.results), **description))
        write_json(folder / "train_manifest.json", manifest)
        print(f"Universal run directory: {folder}", flush=True)
        with (folder / "run.log").open("x") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    validation = train(roots, config, manifest, folder, checkpoint)
                except BaseException:
                    traceback.print_exc()
                    raise
        update_status(folder, state="completed", completed=True, seed=seed,
                      checkpoint=str(checkpoint), validation=validation)
        return folder
    except BaseException as error:
        update_status(folder, state="failed", completed=False, seed=seed,
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.unlink()
