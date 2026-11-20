"""Universal Stage 2：同一個 checkpoint、獨立 TEST windows、只有 continuous metrics。"""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np

from src.seed_replication import PROTECTED, file_hash, read_json, safe_path, unique_name, write_json
from src.universal_training import (Tee, configuration, core_args, git_info, import_core,
    namespace, run_identity, update_status)

EVALUATION_SOURCES = (*PROTECTED, "src/calibration_runtime.py", "src/continuous_metrics.py",
    "src/universal_training.py", "src/universal_evaluation.py", "src/universal_evaluation_analysis.py",
    "scripts/run_universal_evaluation.py", "configs/formal_protocol_v2.json",
    "configs/universal_training.json")


def source_hashes(project):
    return {name: file_hash(Path(project) / name) for name in EVALUATION_SOURCES}


def resolve_stage1(roots, stage1_run=None):
    """只接受指定 attempt 或唯一 completed attempt，不使用 latest / mtime。"""
    config = configuration(roots.project, 1)
    logical = run_identity(1)
    parent = safe_path(namespace(roots.results) / "runs" / logical)
    if (parent / "active.lock").exists():
        raise ValueError("Stage-1 run 仍有 active.lock；先確認訓練已結束")
    if stage1_run:
        run = safe_path(Path(stage1_run).absolute())
        if run.parent != parent or not run.name.startswith("attempt-"):
            raise ValueError("Stage-1 attempt 必須位於 seed-1 universal logical run 下")
    else:
        completed = []
        for folder in parent.glob("attempt-*"):
            status = read_json(safe_path(folder / "status.json"))
            if status.get("state") == "completed" and status.get("completed") is True:
                completed.append(folder)
        if len(completed) != 1:
            raise ValueError(f"需要恰好一個 completed Stage-1 attempt，目前 {len(completed)} 個；可用 --stage1-run 明確指定")
        run = completed[0]
    paths = {name: safe_path(run / name) for name in
        ("status.json", "validation.json", "metadata.json", "launch.json", "train_manifest.json",
         "window_provenance.json", "training_history.json")}
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError("Stage-1 artifacts 缺失；禁止改用其他 checkpoint")
    status, validation, meta = [read_json(paths[name]) for name in
                              ("status.json", "validation.json", "metadata.json")]
    if status.get("state") != "completed" or status.get("completed") is not True or status.get("seed") != 1:
        raise ValueError("Stage-1 status 未完成或 seed 不是 1")
    if any(meta.get(key) != value for key, value in config.items()):
        raise ValueError("Stage-1 model / sources / seed / protocol / dtype 設定不符")
    if (meta.get("run_id") != logical or meta.get("attempt_id") != run.name
            or meta.get("machine_count") != 28 or meta.get("test_loaded") is not False):
        raise ValueError("Stage-1 identity 或 TRAIN-only provenance 不符")
    for key in ("checkpoint_reload_passed", "model_state_equal", "optimizer_state_equal",
                "scheduler_state_equal", "epoch_history_equal", "all_windows_single_machine"):
        if validation.get(key) is not True:
            raise ValueError(f"Stage-1 validation 缺少成功證據：{key}")
    if validation.get("seed") != 1 or validation.get("machine_count") != 28 or validation.get("epochs_completed") != 5:
        raise ValueError("Stage-1 validation 的 seed / machine count / epochs 不符")
    if status.get("validation") != validation:
        raise ValueError("Stage-1 status 與 validation.json 不一致")
    checkpoint = safe_path(namespace(roots.results) / "checkpoints" / logical / run.name / "model.ckpt")
    if meta.get("checkpoint_path") != str(checkpoint) or status.get("checkpoint") != str(checkpoint):
        raise ValueError("Checkpoint 不屬於這個 Stage-1 seed-1 universal attempt")
    if not checkpoint.is_file() or file_hash(checkpoint) != validation.get("checkpoint_sha256"):
        raise ValueError("Stage-1 checkpoint 缺失或 SHA256 不符")
    if checkpoint.stat().st_size != validation.get("checkpoint_size_bytes"):
        raise ValueError("Stage-1 checkpoint size 不符")
    provenance = read_json(paths["window_provenance.json"])
    manifest = read_json(paths["train_manifest.json"])
    if (provenance.get("all_windows_single_machine") is not True
            or provenance.get("windowed_before_concatenation") is not True
            or provenance.get("raw_concatenation_before_windowing") is not False):
        raise ValueError("Stage-1 machine-isolated windowing 證據不足")
    offset = 0
    for table in (manifest, provenance):
        records = table.get("machines", [])
        if table.get("machine_count") != 28 or [r["machine"] for r in records] != config["sources"]:
            raise ValueError("Stage-1 28-machine provenance 不符")
        offset = 0
        for record in records:
            count = record["raw_train_length"]
            if (not isinstance(count, int) or count <= 0 or record["window_count"] != count
                    or record["offset_start"] != offset or record["offset_end_exclusive"] != offset+count
                    or record["dropped_points"] != 0):
                raise ValueError("Stage-1 natural window counts / offsets 不一致")
            offset += count
        if table["total_training_windows"] != offset:
            raise ValueError("Stage-1 總 windows 不一致")
    if (manifest["total_training_windows"] != provenance["total_training_windows"]
            or meta.get("total_training_windows") != offset or validation.get("total_training_windows") != offset):
        raise ValueError("Stage-1 metadata / validation / provenance counts 不一致")
    # 不讀取 TRAIN；只核對已保存的 source snapshot，避免 inference semantics 漂移。
    for name in PROTECTED:
        snapshot = safe_path(run / "source_snapshot" / name)
        digest = meta["source_file_sha256"].get(name)
        if not digest or file_hash(snapshot) != digest or file_hash(roots.project / name) != digest:
            raise ValueError(f"Stage-1 與目前 scientific implementation 不一致：{name}")
        paths["source/" + name] = snapshot
    paths["checkpoint"] = checkpoint
    return dict(run=str(run), run_id=logical, seed=1, sources=config["sources"],
        checkpoint=str(checkpoint), checkpoint_sha256=validation["checkpoint_sha256"],
        evaluation_code_sha256=hashlib.sha256(json.dumps(source_hashes(roots.project), sort_keys=True).encode()).hexdigest(),
        input_hashes={str(path): file_hash(path) for path in paths.values()})


def unchanged(bundle):
    for path, digest in bundle["input_hashes"].items():
        if file_hash(path) != digest:
            raise ValueError(f"Stage-1 input 在 evaluation 期間改變：{path}")


def evaluation_root(results, bundle):
    # 程式更新時使用新 namespace，避免舊成功 smoke 阻擋修正版，或被修正版覆寫。
    return safe_path(namespace(results) / "evaluation" / "seed1" / bundle["checkpoint_sha256"] /
                     bundle["evaluation_code_sha256"])


def claim_evaluation(results, bundle, mode, retry):
    if mode not in ("smoke", "full"):
        raise ValueError("未知 evaluation mode")
    parent = safe_path(evaluation_root(results, bundle) / mode)
    if parent.exists() and not retry:
        raise FileExistsError(f"Evaluation 已存在；檢查 status 後，失敗重試使用 --retry：{parent}")
    parent.mkdir(parents=True, exist_ok=retry)
    lock = parent / "active.lock"
    with lock.open("x"):
        pass
    try:
        for attempt in parent.glob("attempt-*"):
            status = read_json(safe_path(attempt / "status.json"))
            if status.get("state") != "failed" or status.get("completed") is not False:
                raise FileExistsError("既有 evaluation 已完成或狀態不明；禁止覆寫")
        folder = safe_path(parent / unique_name("attempt"))
        folder.mkdir(exist_ok=False)
        return folder, lock
    except BaseException:
        lock.unlink()
        raise


def describe_test(roots, target):
    test = safe_path(roots.data / "test" / (target + ".txt"))
    label = safe_path(roots.data / "interpretation_label" / (target + ".txt"))
    if not test.is_file() or not label.is_file():
        raise FileNotFoundError(f"缺少 {target} TEST 或 interpretation_label")
    with test.open() as stream:
        count = sum(1 for _ in stream)
    if count == 0:
        raise ValueError("TEST 不可為空")
    # 只審查原有 label semantics；實際 labels 一律由 preprocess.load_and_save2 生成。
    for line in label.read_text().splitlines():
        span, columns = line.split(":")
        start, stop = map(int, span.split("-"))
        features = [int(value) for value in columns.split(",")]
        if not 0 <= start < stop <= count or any(not 1 <= f <= 38 for f in features):
            raise ValueError(f"{target} interpretation interval 超出 validated 範圍")
    return dict(target=target, test_path=str(test), test_sha256=file_hash(test),
        label_source="corrected_interpretation_label_union", label_path=str(label),
        label_file_sha256=file_hash(label), raw_test_length=count)


def passed_smoke(roots, bundle, hashes):
    from src.universal_evaluation_analysis import validate_target
    parent = evaluation_root(roots.results, bundle) / "smoke"
    attempts = []
    for attempt in parent.glob("attempt-*"):
        status = read_json(safe_path(attempt / "status.json"))
        if status.get("state") == "completed" and status.get("completed") is True:
            attempts.append(attempt)
    if len(attempts) != 1:
        raise ValueError("完整 28 台 evaluation 前，必須先完成唯一 machine-1-1 smoke")
    folder = attempts[0]
    status = read_json(folder / "status.json")
    launch = read_json(folder / "launch.json")
    validation = read_json(folder / "validation.json")
    target = configuration(roots.project, 1)["sources"][0]
    if (launch.get("mode") != "smoke" or launch.get("targets") != [target]
            or launch.get("source_file_sha256") != hashes or launch.get("stage1") != bundle
            or not validation.get("parameter_hash_before")
            or validation.get("parameter_hash_before") != validation.get("parameter_hash_after")
            or validation.get("parameters_unchanged") is not True
            or validation.get("optimizer_steps") != 0 or validation.get("model_retrained") is not False
            or validation.get("checkpoint_sha256") != bundle["checkpoint_sha256"]
            or validation.get("unique_universal_checkpoint_hashes") != 1
            or validation.get("target_count") != 1 or status.get("seed") != 1
            or status.get("mode") != "smoke" or status.get("validation") != validation):
        raise ValueError("Smoke 與目前 checkpoint / implementation 不一致")
    validate_target(folder / "targets" / target, target, bundle)
    metadata = read_json(folder / "targets" / target / "metadata.json")
    if metadata.get("source_file_sha256") != hashes or metadata["test_input"] != describe_test(roots, target):
        raise ValueError("Smoke 之後 TEST / labels 改變；禁止直接啟動 full evaluation")
    return str(folder)


def load_test(roots, target, folder, description):
    import preprocess
    cache = folder / "cache/processed"
    (cache / "SMD").mkdir(parents=True, exist_ok=False)
    previous = preprocess.output_folder
    try:
        preprocess.output_folder = str(cache)
        shape = preprocess.load_and_save("test", target + ".txt", target, str(roots.data))
        preprocess.load_and_save2("labels", target + ".txt", target, str(roots.data), shape)
    finally:
        preprocess.output_folder = previous
    raw = np.load(cache / "SMD" / (target + "_test.npy"), allow_pickle=False)
    feature_labels = np.load(cache / "SMD" / (target + "_labels.npy"), allow_pickle=False)
    if (raw.shape != (description["raw_test_length"], 38) or raw.dtype != np.float64
            or not np.isfinite(raw).all() or feature_labels.shape != raw.shape
            or not np.isin(feature_labels, [0, 1]).all()):
        raise ValueError("TEST / labels 必須 finite、38 features、長度一致及 binary labels")
    if description != describe_test(roots, target):
        raise ValueError("TEST / labels 在讀取期間改變")
    return raw, feature_labels.any(axis=1).astype(np.int64)


def evaluate_target(roots, bundle, core, model, target, folder, hashes):
    import torch
    from src.calibration_runtime import state_hash
    from src.continuous_metrics import continuous_metrics
    from src.cross_machine import windows_for_machine
    from src.target_calibration import array_hash
    if model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Evaluation 前 model 必須 frozen + eval")
    before = state_hash(model)
    description = describe_test(roots, target)
    raw, labels = load_test(roots, target, folder, description)
    raw_tensor = torch.from_numpy(raw)
    windows = windows_for_machine(raw_tensor, model, core.convert_to_windows)
    core.args.progress_desc = f"Universal {target} TEST"
    start = perf_counter()
    with torch.no_grad():
        errors, _ = core.backprop(0, model, windows, raw_tensor, None, None, training=False)
    seconds = perf_counter()-start
    if errors.shape != raw.shape or not np.isfinite(errors).all():
        raise ValueError("TEST feature scores shape / finite check 失敗")
    scores = errors.mean(axis=1)
    if scores.dtype != np.float64 or scores.shape != labels.shape or len(scores) != len(raw):
        raise ValueError("TEST score/label/timestamp alignment 不符")
    metrics = continuous_metrics(scores, labels)
    after = state_hash(model)
    if before != after or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Evaluation 改變 model state / freeze")
    unchanged(bundle)
    metadata = dict(experiment="universal_continuous_evaluation_v1", setting="universal_checkpoint_evaluation",
        target=target, group=int(target.split("-")[1]), seed=1, protocol="v2", model="TranAD", dataset="SMD",
        checkpoint_id=bundle["run_id"], checkpoint_path=bundle["checkpoint"],
        checkpoint_sha256=bundle["checkpoint_sha256"], stage1_run=bundle["run"],
        evaluation_code_sha256=bundle["evaluation_code_sha256"],
        sources=bundle["sources"], source_count=28, target_train_used_in_stage1=True,
        zero_shot=False, test_input=description, num_test_points=len(raw), score_count=len(scores),
        label_count=len(labels), num_positive_labels=int(labels.sum()), anomaly_prevalence=float(labels.mean()),
        window=10, eval_batch_size=128, dtype="float64", device="cpu",
        score_definition="main.backprop(training=False) feature MSE, errors.mean(axis=1)",
        label_mapping="interpretation_label: zero-based, end-exclusive, feature union",
        score_start_timestamp=0, score_end_exclusive=len(raw), model_retrained=False, optimizer_steps=0,
        model_eval=True, requires_grad=False, normalization_fitted=False, test_used_for_training=False,
        test_used_for_validation=False, test_used_for_adaptation=False,
        git=git_info(roots.project), source_file_sha256=hashes, inference_time_sec=seconds,
        torch_version=torch.__version__, numpy_version=np.__version__, python=sys.version,
        score_array_sha256=array_hash(scores), label_array_sha256=array_hash(labels))
    metrics.update(anomaly_prevalence=metadata["anomaly_prevalence"],
        pr_auc_minus_prevalence=metrics["pr_auc_trapezoidal"]-metadata["anomaly_prevalence"])
    with (folder / "scores.npz").open("xb") as stream:
        np.savez_compressed(stream, scores=scores, labels=labels, timestamps=np.arange(len(raw), dtype=np.int64))
    write_json(folder / "metrics.json", metrics)
    write_json(folder / "metadata.json", metadata)
    validation = dict(parameter_hash_before=before, parameter_hash_after=after, parameters_unchanged=True,
        model_retrained=False, optimizer_steps=0, all_windows_single_machine=True,
        score_label_alignment=True, scores_finite=True, labels_binary=True,
        checkpoint_sha256=bundle["checkpoint_sha256"], checkpoint_unchanged=True,
        stage1_inputs_unchanged=True, score_count=len(scores), label_count=len(labels),
        score_array_sha256=metadata["score_array_sha256"], label_array_sha256=metadata["label_array_sha256"],
        scores_npz_sha256=file_hash(folder / "scores.npz"))
    write_json(folder / "validation.json", validation)
    print(f"{target}: points={len(scores)}, prevalence={metadata['anomaly_prevalence']:.6f}, "
          f"PR-AUC={metrics['pr_auc_trapezoidal']:.8f}, AP={metrics['average_precision']:.8f}, "
          f"ROC-AUC={metrics['roc_auc']:.8f}", flush=True)


def execute(roots, target=None, all_targets=False, seed=1, stage1_run=None, retry=False,
            dry_run=False, formal_summary=None, command=None):
    if seed != 1:
        raise ValueError("Stage 2 只允許 seed=1")
    config = configuration(roots.project, 1)
    if all_targets == bool(target):
        raise ValueError("請擇一指定 --target 或 --all")
    if target and target != config["sources"][0]:
        raise ValueError("單台 smoke 只允許 canonical 第一台 machine-1-1；其餘使用 --all")
    mode = "full" if all_targets else "smoke"
    targets = config["sources"] if all_targets else [target]
    bundle = resolve_stage1(roots, stage1_run)
    hashes = source_hashes(roots.project)
    smoke = None
    if all_targets and not dry_run:
        smoke = passed_smoke(roots, bundle, hashes)
    description = dict(experiment="universal_continuous_evaluation_v1", mode=mode,
        targets=targets, stage1=bundle, source_file_sha256=hashes, seed=1,
        window=10, eval_batch_size=128, device="cpu", dtype="float64", completed_smoke=smoke,
        test_inputs=[describe_test(roots, name) for name in targets],
        output_parent=str(evaluation_root(roots.results, bundle) / mode),
        full_requires_completed_matching_smoke=True, model_retrained=False, optimizer_steps=0)
    if dry_run:
        print(json.dumps(description, ensure_ascii=False, indent=2, allow_nan=False))
        return description
    folder, lock = claim_evaluation(roots.results, bundle, mode, retry)
    try:
        update_status(folder, state="running", completed=False, seed=1, phase="loading_checkpoint")
        write_json(folder / "launch.json", dict(description, command=command or sys.argv,
                   python_executable=sys.executable))
        print(f"Universal evaluation directory: {folder}", flush=True)
        with (folder / "run.log").open("x") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    result = run_targets(roots, config, bundle, folder, targets, hashes, formal_summary, mode)
                except BaseException:
                    traceback.print_exc()
                    raise
        update_status(folder, state="completed", completed=True, seed=1, mode=mode,
                      target_count=len(targets), checkpoint_sha256=bundle["checkpoint_sha256"], validation=result)
        return folder
    except BaseException as error:
        update_status(folder, state="failed", completed=False, seed=1, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.unlink()


def run_targets(roots, config, bundle, folder, targets, hashes, formal_summary, mode):
    import torch
    from src.calibration_runtime import load_frozen_model, state_hash
    from src.universal_evaluation_analysis import summarize, validate_target
    args = core_args(roots, config, folder, Path(bundle["checkpoint"]))
    args.retrain, args.test, args.checkpoint = False, True, bundle["checkpoint"]
    core = import_core(args)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    unchanged(bundle)
    model, before = load_frozen_model(core, bundle["checkpoint"])
    # 只載入一次；各 target 永遠使用這個 model instance，沒有 optimizer。
    for i, target in enumerate(targets):
        print(f"[{i+1}/{len(targets)}] Universal checkpoint → {target} TEST", flush=True)
        update_status(folder, state="running", completed=False, seed=1,
                      phase="test_inference", target=target, target_index=i+1, target_count=len(targets))
        out = safe_path(folder / "targets" / target)
        out.mkdir(parents=True, exist_ok=False)
        evaluate_target(roots, bundle, core, model, target, out, hashes)
        validate_target(out, target, bundle)
    after = state_hash(model)
    if before != after or hashes != source_hashes(roots.project):
        raise ValueError("跨 target evaluation 的 model / implementation 改變")
    unchanged(bundle)
    rows = [validate_target(folder / "targets" / name, name, bundle) for name in targets]
    if len({row["checkpoint_sha256"] for row in rows}) != 1:
        raise ValueError("不同 target 使用了不同 checkpoint")
    validation = dict(unique_universal_checkpoint_hashes=1, checkpoint_sha256=bundle["checkpoint_sha256"],
        parameter_hash_before=before, parameter_hash_after=after, parameters_unchanged=True,
        model_retrained=False, optimizer_steps=0, stage1_inputs_unchanged=True,
        target_count=len(targets), seed=1, mode=mode)
    if mode == "full":
        summarize(roots, bundle, folder, targets, formal_summary)
    unchanged(bundle)
    write_json(folder / "validation.json", validation)
    print("Universal continuous-score evaluation 完成；沒有 threshold / calibration。", flush=True)
    return validation
