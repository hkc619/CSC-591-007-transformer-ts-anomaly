"""Frozen TranAD 的 target TRAIN prefix threshold calibration；formal artifacts 唯讀。"""
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter

import numpy as np

from src.calibration_protocol import calibration_directory, canonical_fraction, prefix_length
from src.calibration_artifacts import new_attempt


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((value.dtype.str, value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def select_formal_run(formal_root, machines, target):
    """與正式 summary 一樣採最近 attempt；失敗時不靜默回退舊 checkpoint。"""
    task_id = 28 + machines.index(target)
    candidates = []
    for path in (formal_root / "task_status").glob("*.json"):
        record = read_json(path)
        if record.get("task_id") == task_id:
            candidates.append((path.stat().st_mtime_ns, path, record))
    if not candidates:
        raise FileNotFoundError(f"找不到 target={target} 的正式 task {task_id} status")
    _, status_path, status = max(candidates, key=lambda item: (item[0], str(item[1])))
    if status.get("exit_code") != 0 or not status.get("run_path"):
        raise ValueError("最近 formal attempt 未成功；不使用更舊結果")
    run = Path(status["run_path"]).resolve()
    if run.parent != (formal_root / "runs").resolve():
        raise ValueError("formal run_path 必須位於 FORMAL_RESULTS_ROOT/runs 的直接子目錄")
    return run, status_path


def validate_formal_bundle(project, formal_root, run, target, machines):
    paths = {name: run / name for name in
             ("metadata.json", "evaluation.json", "threshold.json", "scores.npz")}
    for path in paths.values():
        if not path.is_file() or path.resolve().parent != run:
            raise ValueError(f"缺少 formal artifact 或連結指向 run 外：{path}")
    meta = read_json(paths["metadata.json"])
    fixed = dict(setting="cross_machine_zero_shot", target_machine=target,
                 protocol="v2", seed=1, epochs=5, batch_size=128, eval_batch_size=128,
                 device="cpu", dtype="float64", window=10, learning_rate=0.0001,
                 training_shuffle=True, smoke_test=False,
                 threshold_source="source_machine_training_scores",
                 threshold_protocol="source_only_frozen_SPOT_extreme_quantile",
                 label_source="corrected_interpretation_label_union")
    if any(meta.get(key) != value for key, value in fixed.items()):
        raise ValueError("formal metadata 不符合既有 zero-shot protocol")
    # 舊 formal metadata 未另存 model/dataset；以原 CLI 和相同 parser snapshot 明確補足證據。
    # 有顯式欄位或 CLI 參數時，絕不可忽略不相符值。
    command = meta.get("command")
    if not isinstance(command, list):
        raise ValueError("formal metadata 缺少原始 command，無法審查 model / dataset")
    for key, expected in (("model", "TranAD"), ("dataset", "SMD")):
        if key in meta and meta[key] != expected:
            raise ValueError(f"formal {key} 錯誤")
        for index, token in enumerate(command):
            if token == "--" + key:
                if index + 1 >= len(command) or command[index + 1] != expected:
                    raise ValueError(f"formal command {key} 錯誤")
            elif token.startswith("--" + key + "=") and token.split("=", 1)[1] != expected:
                raise ValueError(f"formal command {key} 錯誤")
    if meta["source_machines"] != [machine for machine in machines if machine != target]:
        raise ValueError("formal sources 必須為另外 27 台，且順序與 manifest 相同")
    checks = meta["leakage_checks"]
    for key in ("target_train_loaded", "target_in_optimizer_updates", "target_in_threshold_fitting", "normalization_fitted"):
        if checks.get(key) is not False:
            raise ValueError(f"formal isolation 證據不足：{key}")
    for key in ("target_used_for_testing", "all_windows_single_machine", "target_score_label_alignment"):
        if checks.get(key) is not True:
            raise ValueError(f"formal validation 證據不足：{key}")
    if any(Path(path).parent.name == "train" and Path(path).name == target + ".txt"
           for path in meta["data_sha256"]):
        raise ValueError("formal provenance 包含 target TRAIN，拒絕當作 zero-shot baseline")
    pot = meta["pot_fit"]
    expected_pot = dict(q=1e-5, requested_level=0.99995, multiplier=1.06, stream_score_count=0)
    if any(pot.get(key) != value for key, value in expected_pot.items()):
        raise ValueError("formal threshold 並非固定 source-only POT")
    if pot.get("source_score_count") != meta["train_windows"]:
        raise ValueError("formal POT source score count 不一致")
    threshold_info = read_json(paths["threshold.json"])
    baseline = read_json(paths["evaluation.json"])
    threshold = meta["threshold_value"]
    if not np.isfinite(threshold) or baseline["threshold"] != threshold or threshold_info["threshold_value"] != threshold:
        raise ValueError("formal threshold artifacts 不一致")
    checkpoint = Path(meta["checkpoint"]).resolve()
    expected_checkpoint = (formal_root / "checkpoints" / run.name / "model.ckpt").resolve()
    if (checkpoint != expected_checkpoint or not checkpoint.is_file()
            or not checkpoint.is_relative_to((formal_root / "checkpoints").resolve())):
        raise ValueError("checkpoint 必須為同一 formal run 的既有 model.ckpt；不自動 remap paths")
    paths["checkpoint"] = checkpoint
    # 使用已訓練 checkpoint；阻止目前 inference implementation 與 formal snapshot 不同。
    for name in ("main.py", "src/models.py", "src/dlutils.py", "src/constants.py",
                 "src/pot.py", "src/spot.py", "src/training_protocol.py",
                 "src/cross_machine.py", "src/experiment.py", "src/parser.py"):
        snapshot = run / "source" / name
        if not snapshot.is_file() or file_hash(snapshot) != file_hash(project / name):
            raise ValueError(f"與 formal snapshot 的 implementation 不一致：{name}")
        paths["source/" + name] = snapshot
    return meta, baseline, paths


def inspect_prefix(path, fraction):
    """只計算 TRAIN row 數；不讀 test、labels，也不載入 TRAIN suffix 數值。"""
    total = 0
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("target TRAIN 有空白 row；不能猜測 timestamp 對齊")
            total += 1
    length = prefix_length(total, fraction)
    return dict(train_length=total, prefix_length=length, start=0, stop_exclusive=length,
                fraction=canonical_fraction(fraction), selection="first_floor_fraction_rows",
                train_suffix_used_for_scores=False, test_used_for_fitting=False)


def load_prefix(path, length):
    with Path(path).open() as stream:
        prefix = np.loadtxt(islice(stream, length), delimiter=",", dtype=np.float64, ndmin=2)
    if prefix.shape != (length, 38) or not np.isfinite(prefix).all():
        raise ValueError("target TRAIN prefix 需為 finite float64、38 features 且長度正確")
    return prefix


def load_formal_scores(path, test_points):
    # 僅讀 formal 的 test arrays；不讀 raw TEST，也不重新計算 TEST inference。
    with np.load(path, allow_pickle=False) as archive:
        scores = archive["scores"].copy()
        labels = archive["labels_final"].copy()
        errors = archive["test_scores"]
        feature_labels = archive["labels"]
        if scores.shape != (test_points,) or labels.shape != scores.shape:
            raise ValueError("formal score / label 長度不一致")
        if errors.shape != (test_points, 38) or feature_labels.shape != errors.shape:
            raise ValueError("formal feature score / label shape 錯誤")
        if scores.dtype != np.float64 or not np.isfinite(scores).all() or not np.isfinite(errors).all():
            raise ValueError("formal scores 非 finite float64")
        if not np.isin(feature_labels, [0, 1]).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("formal labels 不是 binary")
        np.testing.assert_array_equal(scores, errors.mean(axis=1))
        np.testing.assert_array_equal(labels, feature_labels.any(axis=1).astype(int))
    scores.flags.writeable = False
    labels.flags.writeable = False
    return scores, labels


def check_auc_invariance(before, after, tolerance):
    for key in ("pr_auc_trapezoidal", "average_precision", "roc_auc"):
        np.testing.assert_allclose(before["continuous_score"][key], after["continuous_score"][key],
                                   rtol=0, atol=tolerance, equal_nan=False, err_msg=key)


def check_baseline_metrics(expected, actual, tolerance):
    check_auc_invariance(expected, actual, tolerance)
    for section in ("raw", "point_adjusted"):
        for key in ("precision", "recall", "f1", "TP", "TN", "FP", "FN"):
            np.testing.assert_allclose(expected[section][key], actual[section][key],
                                       rtol=0, atol=tolerance, equal_nan=False)


def inspect_experiment(roots, formal_root, config, machines, target, fraction):
    if target not in machines:
        raise ValueError("target 不在固定 28-machine list")
    output = calibration_directory(roots.results, formal_root)
    run, status_path = select_formal_run(formal_root, machines, target)
    meta, baseline, paths = validate_formal_bundle(roots.project, formal_root, run, target, machines)
    paths["formal_status"] = status_path
    train_path = roots.data / "train" / (target + ".txt")
    selection = inspect_prefix(train_path, fraction)
    description = dict(experiment=config["experiment"], base_protocol="v2", target_machine=target,
                       experiment_type="target_threshold_calibration", target=target,
                       fraction=float(canonical_fraction(fraction)), fraction_key=canonical_fraction(fraction),
                       prefix=selection, calibration_source="target_train_temporal_prefix",
                       calibration_train_file=str(train_path.resolve()),
                       target_train_total_raw_points=selection["train_length"],
                       test_points=meta["test_timestamps"],
                       calibration_raw_points=selection["prefix_length"],
                       calibration_windows=selection["prefix_length"],
                       formal_task_id=len(machines) + machines.index(target), formal_run_id=run.name,
                       formal_checkpoint=str(paths["checkpoint"]),
                       formal_score_artifact=str(paths["scores.npz"]),
                       formal_evaluation_artifact=str(paths["evaluation.json"]),
                       formal_run=str(run), checkpoint=str(paths["checkpoint"]),
                       output_namespace=str(output), threshold_source="target_train_prefix_scores",
                       pot=config["pot"], device="cpu", dtype="float64", seed=1,
                       window=10, eval_batch_size=128,
                       window_size=10, protocol="v2", model="TranAD", dataset="SMD",
                       model_retrained=False, continuous_scores_reused=True,
                       model_identity_evidence="formal CLI + identical parser/model snapshots; strict checkpoint load required at runtime",
                       optimizer_steps=0, reuse_formal_test_scores=True,
                       runtime_validation="NOT EXECUTED — requires Hazel validation")
    return description, meta, baseline, paths, train_path


def threshold_change(zero_shot, calibrated, epsilon=1e-12):
    if not np.isfinite(zero_shot) or not np.isfinite(calibrated):
        raise ValueError("threshold 必須有限")
    signed = float(calibrated - zero_shot)
    if not np.isfinite(signed):
        raise ValueError("threshold shift 非有限值")
    ratio = None
    reason = "zero_shot_threshold_near_zero"
    if abs(zero_shot) > epsilon:
        ratio = float(calibrated / zero_shot)
        reason = "defined"
        if not np.isfinite(ratio):
            ratio, reason = None, "ratio_overflow"
    return dict(zero_shot_threshold=float(zero_shot), calibrated_threshold=float(calibrated),
                absolute_shift=abs(signed), signed_shift=signed, relative_ratio=ratio,
                relative_ratio_status=reason)


def metric_recovery(zero_shot, calibrated):
    return {f"raw_{name}_delta": calibrated["raw"][name] - zero_shot["raw"][name]
            for name in ("precision", "recall", "f1")}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def execute_calibration(roots, formal_root, config, machines, target, fraction, smoke=False,
                        task_id=None, command=None, retry=False):
    if target not in machines:
        raise ValueError("target 不在固定 machine list")
    namespace = calibration_directory(roots.results, formal_root)
    expected_task = machines.index(target) * 3 + config["fractions"].index(canonical_fraction(fraction))
    if task_id is not None and (task_id != expected_task or smoke):
        raise ValueError("calibration task ID / target / fraction / smoke 不一致")
    logical_id, attempt_id, folder = new_attempt(namespace, target, fraction, smoke, retry)
    status = dict(task_id=task_id, target=target, target_machine=target,
                  fraction=canonical_fraction(fraction), run_id=logical_id, attempt_id=attempt_id,
                  smoke_test=smoke, run_path=str(folder), exit_code=1, completed=False,
                  created_at=datetime.now(timezone.utc).isoformat())
    # 所有 source artifacts 唯讀；新增 outputs 全在 calibration namespace。
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=roots.project,
                                capture_output=True, text=True)
        git_commit = commit.stdout.strip() if commit.returncode == 0 else None
    except OSError:
        git_commit = None
    launch = dict(command=command or [sys.executable, *sys.argv], cwd=str(Path.cwd()),
                  project_root=str(roots.project), data_root=str(roots.data),
                  results_root=str(roots.results), formal_results_root=str(formal_root),
                  deployment_git_commit=git_commit,
                  git_commit_status="available" if git_commit else "unavailable", retry=retry, **status)
    write_json(folder / "launch.json", launch)
    with (folder / "run.log").open("x") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        print("Calibration run directory:", folder, flush=True)
        try:
            description, formal_meta, baseline, paths, train_path = inspect_experiment(
                roots, formal_root, config, machines, target, fraction)
            protected = {key: file_hash(path) for key, path in paths.items()}
            _execute(folder, roots, config, description, formal_meta, baseline, paths,
                     protected, train_path, smoke)
            status.update(exit_code=0, completed=True)
        except BaseException as error:
            import traceback
            traceback.print_exc()
            status["error"] = repr(error)
            raise
        finally:
            status["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(folder / "status.json", status)
            if task_id is not None:
                statuses = namespace / "task_status"
                statuses.mkdir(parents=True, exist_ok=True)
                write_json(statuses / f"{task_id:03d}-{logical_id}-{attempt_id}.json", status)
    return folder


def _execute(folder, roots, config, description, formal_meta, baseline, paths,
             protected, train_path, smoke):
    from src.calibration_runtime import (baseline_runtime, evaluate, fit_threshold,
                                         load_frozen_model, score_prefix, state_hash)
    import importlib.metadata
    import torch
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    write_json(folder / "config.json", config)
    prefix = load_prefix(train_path, description["prefix"]["prefix_length"])
    prefix_digest = array_hash(prefix)
    start = perf_counter()
    with baseline_runtime(progress=True) as core:
        print("[1/4] 載入既有 checkpoint 並 freeze；不建立 optimizer", flush=True)
        model, before = load_frozen_model(core, paths["checkpoint"])
        print("[2/4] 只對 target TRAIN prefix inference", prefix.shape, flush=True)
        errors, calibration_scores = score_prefix(core, model, prefix)
        if (calibration_scores.shape != (len(prefix),) or errors.shape != prefix.shape
                or not np.isfinite(calibration_scores).all()):
            raise ValueError("POT input 必須一對一對應 target TRAIN prefix 的 scores")
        inference_seconds = perf_counter() - start
        print("[3/4] TRAIN-prefix-only POT；TEST stream 為空", flush=True)
        np.random.seed(1)
        threshold, pot_info = fit_threshold(core, calibration_scores)
        if state_hash(model) != before or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise RuntimeError("model parameters / buffers / freeze state 改變")
        # 只有 threshold 固定後，才開啟原有 TEST score archive。
        print("[4/4] 重用 formal TEST scores / labels；只重算 thresholded metrics", flush=True)
        scores, labels = load_formal_scores(paths["scores.npz"], formal_meta["test_timestamps"])
        original_scores_hash, original_labels_hash = array_hash(scores), array_hash(labels)
        baseline_recomputed, _, _ = evaluate(core, scores, labels, baseline["threshold"])
        check_baseline_metrics(baseline, baseline_recomputed, config["auc_tolerance"])
        calibrated, raw, adjusted = evaluate(core, scores, labels, threshold)
        check_auc_invariance(baseline, calibrated, config["auc_tolerance"])
    if array_hash(scores) != original_scores_hash or array_hash(labels) != original_labels_hash:
        raise RuntimeError("TEST scores / labels 被修改")
    if array_hash(prefix) != prefix_digest:
        raise RuntimeError("TRAIN prefix 被修改")
    for key, path in paths.items():
        if file_hash(path) != protected[key]:
            raise RuntimeError(f"formal input 被外部修改：{path}")
    if state_hash(model) != before:
        raise RuntimeError("evaluation 後 model state 改變")
    validation = dict(parameter_freeze=True, optimizer_steps=0, test_scores_unchanged=True,
                      labels_unchanged=True, pr_auc_unchanged=True, ap_unchanged=True,
                      roc_auc_unchanged=True, pot_test_stream_count=pot_info["stream_score_count"],
                      formal_artifacts_unchanged=True, test_inference_executed=False,
                      formal_test_score_count=len(scores), calibrated_test_score_count=len(scores),
                      test_score_count_unchanged=True,
                      scope="runtime checks of this attempt; TEST arrays reused, not re-inferred")
    # 只有實際執行上述 checks 成功後，才會產生本 validation.json；本次開發沒有執行。
    write_json(folder / "validation.json", validation)
    change = threshold_change(baseline["threshold"], threshold)
    recovery = metric_recovery(baseline, calibrated)
    write_json(folder / "threshold.json", dict(**change, threshold_value=threshold,
               baseline_threshold=baseline["threshold"], threshold_source="target_train_prefix_scores",
               target_machine=description["target_machine"], fraction=description["fraction"],
               prefix=description["prefix"], **pot_info))
    write_json(folder / "evaluation.json", dict(zero_shot=baseline, calibrated=calibrated,
               continuous_score=calibrated["continuous_score"], recovery=recovery))
    np.savez_compressed(folder / "calibration_scores.npz", errors=errors, scores=calibration_scores,
                        timestamps=np.arange(len(prefix)))
    np.savez_compressed(folder / "scores.npz", scores=scores, labels_final=labels,
                        raw_predictions=raw, adjusted_predictions=adjusted)
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "tqdm")}
    calibration_source = {name: file_hash(roots.project / name) for name in (
        "src/calibration_protocol.py", "src/calibration_runtime.py", "src/target_calibration.py",
        "src/calibration_artifacts.py",
        "scripts/run_target_calibration.py", "configs/target_calibration.json",
        "configs/calibration_runs.csv")}
    metadata = dict(**description, setting="cross_machine_target_threshold_calibration",
                    smoke_test=smoke, base_setting="cross_machine_zero_shot",
                    base_run_id=Path(description["formal_run"]).name,
                    source_machines=formal_meta["source_machines"],
                    base_checkpoint=str(paths["checkpoint"]), model_state_hash_before=before,
                    model_state_hash_after=state_hash(model), model_frozen=True,
                    target_train_used_for_threshold=True, target_train_used_for_training=False,
                    normalization_fitted=False, target_test_used_for_threshold=False,
                    target_test_raw_loaded=False, test_inference_executed=False,
                    prefix_array_sha256=prefix_digest, test_scores_sha256=original_scores_hash,
                    labels_sha256=original_labels_hash, protected_input_sha256=protected,
                    calibration_source_sha256=calibration_source,
                    input_paths={key: str(path) for key, path in paths.items()},
                    pot_fit=pot_info, threshold_value=threshold, python=sys.version,
                    platform=platform.platform(), packages=versions,
                    prefix_inference_and_model_load_seconds=inference_seconds,
                    total_seconds=perf_counter() - start,
                    validation_status="EXECUTED checks passed for this attempt")
    # 覆寫 description 的 pending 描述僅限本次實際成功執行所產生的 metadata。
    metadata["runtime_validation"] = "EXECUTED — see validation.json for scope"
    pot_metadata = dict(q=pot_info["q"], requested_level=pot_info["requested_level"],
                        effective_level=pot_info["effective_level"], multiplier=pot_info["multiplier"],
                        attempts=pot_info["initialization_attempts"], stream_score_count=pot_info["stream_score_count"])
    metadata.update(model_retrained=False, parameters_unchanged=True,
                    parameter_hash_before=before, parameter_hash_after=state_hash(model),
                    formal_checkpoint_hash=protected["checkpoint"],
                    formal_score_artifact_hash=protected["scores.npz"], pot=pot_metadata,
                    parameter_hash_scope="all state_dict parameters and buffers",
                    continuous_scores_reused=True, calibration_windows=len(calibration_scores))
    launch = read_json(folder / "launch.json")
    metadata.update(run_id=launch["run_id"], attempt_id=launch["attempt_id"],
                    git_commit=launch["deployment_git_commit"], git_commit_status=launch["git_commit_status"])
    write_json(folder / "metadata.json", metadata)
    smoke_summary = dict(formal_run_id=metadata["formal_run_id"], formal_task_id=metadata["formal_task_id"],
        formal_checkpoint=metadata["formal_checkpoint"], target=metadata["target"], fraction=metadata["fraction"],
        target_train_total_raw_points=metadata["target_train_total_raw_points"],
        calibration_raw_points=len(prefix), calibration_windows=len(calibration_scores),
        model_retrained=False, optimizer_steps=0, parameters_unchanged=True,
        parameter_hash_before=before, parameter_hash_after=state_hash(model),
        thresholds=change, zero_shot_raw=baseline["raw"], calibrated_raw=calibrated["raw"],
        recovery=recovery, continuous_score_before=baseline["continuous_score"],
        continuous_score_after=calibrated["continuous_score"], pot_stream_score_count=pot_info["stream_score_count"])
    write_json(folder / "smoke_summary.json", smoke_summary)
    print(json.dumps(smoke_summary, ensure_ascii=False, indent=2), flush=True)
    print("Calibration 完成：", folder, flush=True)
