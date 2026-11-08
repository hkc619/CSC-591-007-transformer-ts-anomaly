"""獨立 AT experiment：先完成全部 TRAIN，再讀取 target TEST；無 validation/threshold。"""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from src.anomaly_transformer.dataset import (MachineWindows, load_sources, load_values,
    normalization_metadata, select_sources, evaluation_plan)
from src.anomaly_transformer.protocol import logical_id, namespace
from src.anomaly_transformer.runtime import infer, seed_everything, train_fixed
from src.continuous_metrics import continuous_metrics


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def describe(roots, config, canonical, target, setting, device):
    if target not in config["targets"]:
        raise ValueError("只允許固定六台 targets")
    names = {path.stem for path in (roots.data / "train").glob("machine-*.txt")}
    if names != set(canonical):
        raise ValueError("DATA_ROOT TRAIN machine names 與 canonical 28 台不一致")
    sources = select_sources(canonical, target, setting)
    arrays = load_sources(roots.data, sources, target, setting)
    windows = MachineWindows(arrays, config["model"]["win_size"], config["train_stride"])
    test_path = roots.data / "test" / (target + ".txt")
    labels = roots.data / "interpretation_label" / (target + ".txt")
    if not test_path.is_file() or not labels.is_file():
        raise FileNotFoundError("缺少 target TEST / interpretation_label")
    with test_path.open() as stream:
        count = sum(1 for line in stream if line.strip())
    plan = evaluation_plan(count, config["model"]["win_size"])
    out = namespace(roots.results)
    return dict(detector="AnomalyTransformer", protocol=config["protocol"], target=target,
        setting=setting, sources=sources, seed=1, device=device, dtype="float32",
        training_data_paths={m: str(roots.data / "train" / (m + ".txt")) for m in sources},
        test_path=str(test_path), label_path=str(labels), train_windows=len(windows),
        source_window_offsets=windows.offsets, test_points=count, test_windows=len(plan),
        output_path=str(out / "runs" / logical_id(target, setting)), model_config=config["model"],
        epochs=10, batch_size=256, eval_batch_size=256, lr=1e-4,
        training_shuffle=True, source_sampling="natural_complete_window_count",
        window=100, train_stride=100, normalization=config["normalization"],
        target_train_used_for_training=target in sources, target_test_used_for_training=False,
        target_test_used_for_validation=False, threshold_evaluation=False, point_adjustment=False,
        **normalization_metadata())


def load_labels(data_root, target, shape, folder):
    # Reuse the exact validated SMD label generator, redirecting only its output path.
    import importlib
    for line in (Path(data_root) / "interpretation_label" / (target + ".txt")).read_text().splitlines():
        span, features = line.split(":")
        start, stop = map(int, span.split("-"))
        columns = [int(value) for value in features.split(",")]
        if not 0 <= start < stop <= shape[0] or any(not 1 <= column <= shape[1] for column in columns):
            raise ValueError("interpretation interval 不符合原 validated zero-based / end-exclusive semantics")
    argv = sys.argv
    try:
        sys.argv = ["at-label-generator", "--dataset", "SMD", "--model", "TranAD"]
        preprocess = importlib.import_module("preprocess")
    finally:
        sys.argv = argv
    output = folder / "cache/processed"
    (output / "SMD").mkdir(parents=True, exist_ok=True)
    previous = preprocess.output_folder
    try:
        preprocess.output_folder = str(output)
        preprocess.load_and_save2("labels", target + ".txt", target, str(data_root), shape)
    finally:
        preprocess.output_folder = previous
    features = np.load(output / "SMD" / (target + "_labels.npy"), allow_pickle=False)
    if features.shape != shape or not np.isin(features, [0, 1]).all():
        raise ValueError("labels shape / binary check 失敗")
    return features.any(axis=1).astype(np.int64)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute(roots, config, canonical, target, setting, device_name, task_id, phase="formal", retry=False):
    if target not in config["targets"] or setting not in ("per_machine", "cross_machine"):
        raise ValueError("未知 target / setting")
    if phase == "smoke_gate" and target != "machine-1-1":
        raise ValueError("smoke gate 只允許 machine-1-1")
    out = namespace(roots.results)
    identity = logical_id(target, setting)
    parent = out / "runs" / identity
    if parent.is_symlink():
        raise ValueError("run directory 不接受 symlink")
    if parent.exists() and not retry:
        raise FileExistsError("run 已存在；先檢查結果，需要重試時明確 --retry")
    parent.mkdir(parents=True, exist_ok=retry)
    attempt = datetime.now(timezone.utc).strftime("attempt-%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8]
    folder = parent / attempt
    folder.mkdir(exist_ok=False)
    status = dict(task_id=task_id, run_id=identity, attempt_id=attempt, run_path=str(folder),
                  target=target, setting=setting, phase=phase, completed=False, exit_code=1)
    write_json(folder / "launch.json", dict(command=[sys.executable, *sys.argv],
        project_root=str(roots.project), data_root=str(roots.data), results_root=str(roots.results), **status))
    with (folder / "run.log").open("x") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        print("AT Run directory:", folder, flush=True)
        try:
            if device_name == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用；請確認 GPU allocation 與獨立 GPU PyTorch environment，不自動改 CPU")
            result = _execute(roots, config, canonical, target, setting, device_name, folder, out, identity, attempt)
            result.update(task_id=task_id, phase=phase, run_id=identity, attempt_id=attempt)
            write_json(folder / "metadata.json", result)
            status.update(completed=True, exit_code=0)
        except BaseException as error:
            import traceback
            traceback.print_exc()
            status["error"] = repr(error)
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(folder / "status.json", status)
            statuses = out / "task_status"
            statuses.mkdir(exist_ok=True)
            write_json(statuses / f"{task_id:02d}-{identity}-{attempt}.json", status)
    return folder


def _execute(roots, config, canonical, target, setting, device_name, folder, out, identity, attempt):
    from src.anomaly_transformer.model import AnomalyTransformer
    start = perf_counter()
    metadata = describe(roots, config, canonical, target, setting, device_name)
    write_json(folder / "config.json", config)
    seed_everything(config["seed"])
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=roots.project, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=roots.project, capture_output=True, text=True)
    metadata.update(git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        git_dirty=bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
        python=sys.version, platform=platform.platform(), torch_version=torch.__version__, numpy_version=np.__version__,
        cuda_version=torch.version.cuda, gpu_model=torch.cuda.get_device_name() if device.type == "cuda" else None,
        optimizer_config=config["optimizer"], learning_rate_schedule=config["lr_schedule"],
        test_mapping=config["test_mapping"], validation="none_fixed_10_epochs", early_stopping=False,
        temperature=config["temperature"], k=config["k"], upstream_commit=config["upstream_commit"])
    snapshot = folder / "source"
    source_files = [*sorted((roots.project / "src/anomaly_transformer").glob("*.py")),
                    roots.project / "src/continuous_metrics.py", roots.project / "preprocess.py",
                    roots.project / "src/anomaly_transformer/LICENSE", roots.project / "src/anomaly_transformer/PROVENANCE.json",
                    roots.project / "configs/anomaly_transformer.json", roots.project / "configs/anomaly_transformer_runs.csv"]
    metadata["source_sha256"] = {}
    for path in source_files:
        relative = path.relative_to(roots.project)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        metadata["source_sha256"][str(relative)] = file_hash(path)
    metadata["data_sha256"] = {path: file_hash(path) for path in metadata["training_data_paths"].values()}
    write_json(folder / "metadata.json", metadata)
    sources = load_sources(roots.data, metadata["sources"], target, setting)
    dataset = MachineWindows(sources, config["model"]["win_size"], config["train_stride"])
    if setting == "cross_machine" and target in dataset.names:
        raise RuntimeError("target TRAIN isolation 失敗")
    write_json(folder / "window_provenance.json", dict(source_offsets=dataset.offsets,
        all_windows_single_machine=True, proof="each item slices exactly one machine array; no raw concatenation"))
    model = AnomalyTransformer(**config["model"]).to(device=device, dtype=torch.float32)
    train_start = perf_counter()
    history = train_fixed(model, dataset, config, device,
                          lambda values: write_json(folder / "training_history.json", values))
    training_seconds = perf_counter()-train_start
    checkpoint_parent = out / "checkpoints" / identity
    if checkpoint_parent.is_symlink():
        raise ValueError("checkpoint directory 不接受 symlink")
    checkpoint_dir = checkpoint_parent / attempt
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = checkpoint_dir / "model.ckpt"
    with checkpoint.open("xb") as stream:
        torch.save(dict(model_state_dict=model.state_dict(), config=config, epoch=10,
                        target=target, setting=setting, sources=metadata["sources"]), stream)
    model.eval()
    before = state_hash(model)
    # First numerical read of target TEST is strictly after all optimizer updates.
    test_raw = load_values(roots.data / "test" / (target + ".txt"))
    labels = load_labels(roots.data, target, test_raw.shape, folder)
    inference_start = perf_counter()
    scores, mapping, plan = infer(model, test_raw, config, device)
    inference_seconds = perf_counter()-inference_start
    if len(scores) != len(labels) or state_hash(model) != before:
        raise RuntimeError("score/label alignment 或 frozen model invariant 失敗")
    evaluation = continuous_metrics(scores, labels)
    diagnostics = dict(min=float(scores.min()), max=float(scores.max()), mean=float(scores.mean()),
                       std=float(scores.std()), nonzero_count=int(np.count_nonzero(scores)),
                       unique_count=int(len(np.unique(scores))), constant=bool(np.ptp(scores) == 0))
    write_json(folder / "evaluation.json", evaluation)
    write_json(folder / "timestamp_mapping.json", dict(window_plan=plan, score_start_timestamp=0,
        label_slice=[0, len(labels)], final_count=len(scores), energy_selection=config["test_mapping"]))
    np.savez_compressed(folder / "scores.npz", scores=scores, labels=labels, **mapping)
    for split in ("test", "interpretation_label"):
        path = roots.data / split / (target + ".txt")
        metadata["data_sha256"][str(path)] = file_hash(path)
    metadata.update(test_score_count=len(scores), label_count=len(labels), anomalous_points=int(labels.sum()),
        score_diagnostics=diagnostics,
        train_windows=len(dataset), training_seconds=training_seconds, inference_seconds=inference_seconds,
        runtime_seconds=perf_counter()-start, checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint),
        checkpoint_size_bytes=checkpoint.stat().st_size, frozen_model_unchanged=True,
        target_train_loaded=target in metadata["sources"], target_test_loaded_after_training=True,
        all_windows_single_machine=True, score_label_alignment=True,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == "cuda" else 0)
    print(json.dumps(evaluation, indent=2), flush=True)
    print("Native score diagnostics:", diagnostics, flush=True)
    return metadata
