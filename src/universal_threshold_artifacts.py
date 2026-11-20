"""Stage 3 的唯讀 Stage-2 resolution、獨立 attempts 與 TRAIN-score cache。"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.seed_replication import file_hash, read_json, safe_path, unique_name
from src.target_calibration import array_hash
from src.universal_evaluation import EVALUATION_SOURCES, evaluation_root, resolve_stage1
from src.universal_evaluation_analysis import validate_target
from src.universal_training import configuration, namespace

CONDITIONS = ("global", "local_05", "local_10", "local_20", "local_full")
SOURCES = (*EVALUATION_SOURCES, "src/calibration_protocol.py", "src/target_calibration.py",
    "configs/target_calibration.json", "src/universal_threshold_artifacts.py",
    "src/universal_thresholding.py", "src/universal_threshold_analysis.py", "scripts/run_universal_thresholding.py")


def code_hashes(project):
    return {name: file_hash(Path(project) / name) for name in SOURCES}


def resolve_inputs(roots, stage1_run=None, stage2_run=None):
    stage1 = resolve_stage1(roots, stage1_run)
    parent = evaluation_root(roots.results, stage1) / "full"
    if (parent / "active.lock").exists():
        raise ValueError("Stage 2 仍有 active.lock")
    if stage2_run:
        folder = safe_path(Path(stage2_run).expanduser().absolute())
        if folder.parent != parent or not folder.name.startswith("attempt-"):
            raise ValueError("Stage-2 run 必須屬於相同 universal checkpoint／evaluation code 的 full attempt")
    else:
        candidates = [p for p in parent.glob("attempt-*")
                      if read_json(p / "status.json").get("completed") is True]
        if len(candidates) != 1:
            raise ValueError(f"Stage 2 必須有唯一 completed full attempt，目前 {len(candidates)} 個；可用 --stage2-run pin")
        folder = safe_path(candidates[0])
    status, validation, launch = [read_json(folder / name) for name in
                                 ("status.json", "validation.json", "launch.json")]
    canonical = configuration(roots.project, 1)["sources"]
    if (status.get("state") != "completed" or status.get("completed") is not True
            or status.get("seed") != 1 or status.get("mode") != "full" or status.get("target_count") != 28
            or status.get("validation") != validation or launch.get("stage1") != stage1
            or launch.get("targets") != canonical or launch.get("mode") != "full"
            or validation.get("checkpoint_sha256") != stage1["checkpoint_sha256"]
            or validation.get("unique_universal_checkpoint_hashes") != 1
            or validation.get("target_count") != 28 or validation.get("seed") != 1
            or validation.get("model_retrained") is not False or validation.get("optimizer_steps") != 0
            or not validation.get("parameter_hash_before")
            or validation["parameter_hash_before"] != validation.get("parameter_hash_after")):
        raise ValueError("Stage-2 completion／seed／checkpoint／freeze validation 不符")
    if {p.name for p in (folder / "targets").iterdir()} != set(canonical):
        raise ValueError("Stage-2 targets 不等於 canonical 28 台")
    protected = dict(stage1["input_hashes"])
    rows = {}
    for target in canonical:
        out = safe_path(folder / "targets" / target)
        rows[target] = validate_target(out, target, stage1)
        proof = read_json(out / "validation.json")
        if proof["parameter_hash_before"] != validation["parameter_hash_before"]:
            raise ValueError("Stage-2 per-target model-state hash 與全域不符")
        for name in ("metadata.json", "metrics.json", "validation.json", "scores.npz"):
            path = safe_path(out / name)
            protected[str(path)] = file_hash(path)
    summary = read_json(folder / "summary/summary.json")
    if summary.get("target_count") != 28 or summary.get("checkpoint_sha256") != stage1["checkpoint_sha256"]:
        raise ValueError("Stage-2 summary 不符")
    with (folder / "summary/target_metrics.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    if len(table) != 28 or [r["target"] for r in table] != canonical:
        raise ValueError("Stage-2 summary CSV targets 不符")
    for row in table:
        expected = rows[row["target"]]
        if any(row[key] != str(expected[key]) for key in expected):
            raise ValueError("Stage-2 summary CSV 與 canonical scores 不符")
    for path in [folder / name for name in ("status.json", "validation.json", "launch.json")]:
        protected[str(path)] = file_hash(path)
    for path in (folder / "summary").iterdir():
        if path.is_file():
            protected[str(safe_path(path))] = file_hash(path)
    return dict(stage1=stage1, stage2_run=str(folder), stage2_code_sha256=stage1["evaluation_code_sha256"],
        checkpoint_sha256=stage1["checkpoint_sha256"], model_state_sha256=validation["parameter_hash_before"],
        targets=canonical, protected_input_sha256=protected)


def check_inputs(inputs):
    for filename, digest in inputs["protected_input_sha256"].items():
        if file_hash(filename) != digest:
            raise ValueError(f"Stage-1／Stage-2 input 改變：{filename}")


def root_path(roots, inputs):
    digest = hashlib.sha256(json.dumps(code_hashes(roots.project), sort_keys=True).encode()).hexdigest()
    return safe_path(namespace(roots.results) / "thresholding/seed1" / inputs["checkpoint_sha256"] / digest)


def claim(parent, retry=False):
    parent = safe_path(parent)
    if parent.exists() and not retry:
        raise FileExistsError(f"已有 artifacts；先檢查，失敗重試須 --retry：{parent}")
    parent.mkdir(parents=True, exist_ok=retry)
    lock = parent / "active.lock"
    with lock.open("x"):
        pass
    try:
        for old in parent.glob("attempt-*"):
            status = read_json(old / "status.json")
            if status.get("state") != "failed" or status.get("completed") is not False:
                raise FileExistsError("成功或狀態未明的 attempt 不可覆寫／重跑")
        folder = safe_path(parent / unique_name("attempt"))
        folder.mkdir(exist_ok=False)
        return folder, lock
    except BaseException:
        lock.unlink()
        raise


def completed(parent):
    parent = safe_path(parent)
    if (parent / "active.lock").exists():
        raise ValueError("該 artifact 仍有 active.lock")
    attempts = list(parent.glob("attempt-*"))
    successes = [p for p in attempts if read_json(p / "status.json").get("completed") is True]
    if not attempts:
        return None
    if len(successes) != 1:
        raise ValueError("找不到唯一成功 artifact；失敗重試須明確 --retry，不使用 latest fallback")
    return safe_path(successes[0])


def read_train_scores(folder, record, inputs, require_completed=True):
    status, meta, proof = [read_json(folder / name) for name in ("status.json", "metadata.json", "validation.json")]
    fixed = dict(seed=1, split="train", machine=record["machine"], checkpoint_sha256=inputs["checkpoint_sha256"],
                 raw_file_sha256=record["raw_file_sha256"], raw_array_sha256=record["raw_array_sha256"],
                 raw_train_length=record["raw_train_length"], window=10, eval_batch_size=128, dtype="float64", device="cpu")
    expected_state, expected_completed = ("completed", True) if require_completed else ("running", False)
    if (status.get("state") != expected_state or status.get("completed") is not expected_completed
            or any(meta.get(k) != v for k, v in fixed.items())
            or proof.get("parameters_unchanged") is not True or proof.get("all_windows_single_machine") is not True
            or proof.get("optimizer_steps") != 0 or proof.get("model_retrained") is not False
            or proof.get("parameter_hash_before") != inputs["model_state_sha256"]
            or proof.get("parameter_hash_after") != inputs["model_state_sha256"]):
        raise ValueError("TRAIN scores 的 source／checkpoint／model freeze metadata 不符")
    path = safe_path(folder / "scores.npz")
    if file_hash(path) != proof["scores_npz_sha256"]:
        raise ValueError("TRAIN scores archive 改變")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"scores", "timestamps"}:
            raise ValueError("TRAIN archive 不得包含 TEST／labels")
        scores, timestamps = saved["scores"].copy(), saved["timestamps"].copy()
    if (scores.shape != (record["raw_train_length"],) or scores.dtype != np.float64
            or not np.isfinite(scores).all() or not np.array_equal(timestamps, np.arange(len(scores)))
            or array_hash(scores) != meta["scores_array_sha256"]
            or array_hash(timestamps) != meta["timestamps_array_sha256"]):
        raise ValueError("TRAIN score alignment／hash／finite 不符")
    return scores, meta


def test_arrays(inputs, target):
    """唯一 TEST 入口：只讀 Stage-2 saved scores，不讀 raw TEST、不呼叫 model。"""
    folder = safe_path(Path(inputs["stage2_run"]) / "targets" / target)
    validate_target(folder, target, inputs["stage1"])
    with np.load(folder / "scores.npz", allow_pickle=False) as saved:
        scores, labels, timestamps = [saved[k].copy() for k in ("scores", "labels", "timestamps")]
    return scores, labels, timestamps, read_json(folder / "metrics.json")
