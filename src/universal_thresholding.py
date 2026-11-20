"""Stage 3：一個 frozen universal model，TRAIN-only POT，唯讀重用 Stage-2 TEST scores。"""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np

from src.calibration_protocol import load_configuration, prefix_length
from src.seed_replication import file_hash, read_json, write_json
from src.target_calibration import array_hash, threshold_change
from src.universal_training import (Tee, array_hash as raw_hash, configuration, core_args,
    git_info, import_core, update_status)
from src.universal_threshold_artifacts import (CONDITIONS, check_inputs, claim, code_hashes,
    completed, read_train_scores, resolve_inputs, root_path)
from src.universal_threshold_analysis import summarize, target_row

FRACTIONS = dict(local_05="0.05", local_10="0.10", local_20="0.20")


def inspect_train(roots, inputs, targets):
    records = read_json(Path(inputs["stage1"]["run"]) / "train_manifest.json")["machines"]
    result = []
    for record in records:
        if record["machine"] not in targets:
            continue
        path = roots.data / "train" / (record["machine"]+".txt")
        if path.resolve() != path.absolute() or str(path) != record["train_path"] or file_hash(path) != record["raw_file_sha256"]:
            raise ValueError("TRAIN path／hash 與 Stage 1 不同，不自動更換 dataset")
        result.append(record)
    if [r["machine"] for r in result] != targets:
        raise ValueError("TRAIN manifest 未包含全部預定 machines")
    return result


def frozen_scores(core, model, raw):
    import torch
    from src.cross_machine import windows_for_machine
    if (model.training or any(p.requires_grad or p.grad is not None or p.dtype != torch.float64
                             or p.device.type != "cpu" for p in model.parameters())):
        raise ValueError("TRAIN inference 必須 frozen CPU float64")
    windows = windows_for_machine(raw, model, core.convert_to_windows)
    with torch.no_grad():
        errors, _ = core.backprop(0, model, windows, raw, None, None, training=False)
    if errors.shape != tuple(raw.shape) or not np.isfinite(errors).all():
        raise ValueError("TRAIN errors 長度／finite 不符")
    return errors.mean(axis=1)


def obtain_scores(roots, inputs, core, model, record, root, retry):
    from src.calibration_runtime import state_hash
    parent = root / "train_scores" / record["machine"]
    existing = completed(parent) if not retry else None
    if retry and parent.exists():
        successes = [p for p in parent.glob("attempt-*") if read_json(p / "status.json").get("completed") is True]
        if successes:
            existing = completed(parent)
    if existing:
        scores, meta = read_train_scores(existing, record, inputs)
        print(f"重用 TRAIN scores：{record['machine']}，{len(scores)} points", flush=True)
        return scores, existing
    folder, lock = claim(parent, retry)
    try:
        update_status(folder, state="running", completed=False, seed=1)
        import preprocess
        from src.cross_machine import load_training_machine
        previous_output, previous_cache = core.args.output_dir, preprocess.output_folder
        try:
            core.args.output_dir = str(folder)
            preprocess.output_folder = str(folder / "cache/processed")
            raw = load_training_machine(record["machine"])
        finally:
            core.args.output_dir, preprocess.output_folder = previous_output, previous_cache
        if len(raw) != record["raw_train_length"] or raw_hash(raw.numpy()) != record["raw_array_sha256"]:
            raise ValueError("TRAIN raw values 與 Stage-1 provenance 不同")
        before = state_hash(model)
        start = perf_counter()
        core.args.progress_desc = f"Universal {record['machine']} TRAIN scores"
        scores = frozen_scores(core, model, raw)
        seconds = perf_counter()-start
        after = state_hash(model)
        if before != after or before != inputs["model_state_sha256"]:
            raise ValueError("TRAIN inference 改變 model state")
        timestamps = np.arange(len(scores), dtype=np.int64)
        with (folder / "scores.npz").open("xb") as stream:
            np.savez_compressed(stream, scores=scores, timestamps=timestamps)
        write_json(folder / "metadata.json", dict(seed=1, machine=record["machine"], split="train",
            checkpoint_sha256=inputs["checkpoint_sha256"], checkpoint=inputs["stage1"]["checkpoint"],
            raw_file_sha256=record["raw_file_sha256"], raw_array_sha256=record["raw_array_sha256"],
            raw_train_length=len(raw), train_path=record["train_path"], window=10, eval_batch_size=128,
            dtype="float64", device="cpu", scores_array_sha256=array_hash(scores),
            timestamps_array_sha256=array_hash(timestamps), inference_time_sec=seconds,
            prefix_reuse="window(t) only depends on same-machine raw[0:t+1]; eval mode is window-local",
            git=git_info(roots.project)))
        write_json(folder / "validation.json", dict(parameters_unchanged=True, parameter_hash_before=before,
            parameter_hash_after=after, all_windows_single_machine=True, optimizer_steps=0,
            model_retrained=False, scores_npz_sha256=file_hash(folder / "scores.npz")))
        read_train_scores(folder, record, inputs, require_completed=False)
        update_status(folder, state="completed", completed=True, seed=1)
        return scores, folder
    except BaseException as error:
        update_status(folder, state="failed", completed=False, seed=1, error=str(error))
        raise
    finally:
        lock.unlink()


def fit_train_threshold(core, scores, provenance):
    """這個 API 沒有 TEST/labels 參數，來源必須為已驗證的 TRAIN-score artifact。"""
    from src.calibration_runtime import fit_threshold
    if (provenance.get("split") != "train" or provenance.get("seed") != 1
            or provenance.get("score_count") != len(scores) or len(scores) == 0
            or not provenance.get("sources") or len(set(provenance["sources"])) != len(provenance["sources"])
            or scores.ndim != 1 or scores.dtype != np.float64 or not np.isfinite(scores).all()):
        raise ValueError("POT 只接受 finite、來源明確且 count 一致的 TRAIN scores")
    if provenance.get("condition") == "global" and len(provenance["sources"]) != 28:
        raise ValueError("不可用單台／部分 machines 的 TRAIN scores 假造 global threshold")
    print(f"POT 開始：{provenance.get('condition')}，{len(provenance['sources'])} 台 TRAIN，{len(scores):,} scores", flush=True)
    np.random.seed(1)  # 沿用舊 target-calibration，每次 fit 都使用固定 RNG。
    threshold, info = fit_threshold(core, scores)
    expected = dict(q=1e-5, requested_level=0.99995, multiplier=1.06,
                    source_score_count=len(scores), stream_score_count=0)
    if any(info.get(k) != v for k, v in expected.items()) or not np.isfinite(threshold):
        raise ValueError("POT parameters／score count／empty TEST stream 不符")
    if not 0 < info["effective_level"] <= info["requested_level"] or info["initialization_attempts"] < 1:
        raise ValueError("POT retry metadata 不符")
    print(f"POT 完成：threshold={float(threshold):.10g}，initialization attempts={info['initialization_attempts']}", flush=True)
    return dict(threshold=float(threshold), **info, score_count=len(scores), machine_count=len(provenance["sources"]),
                natural_weighting=True, provenance=provenance, pot_rng_seed=1)


def smoke_gate(root, inputs, hashes):
    folder = completed(root / "smoke")
    if folder is None:
        raise ValueError("Full Stage 3 前必須先完成 machine-1-1 local_05 smoke")
    launch, validation = read_json(folder / "launch.json"), read_json(folder / "validation.json")
    if (launch.get("inputs") != inputs or launch.get("source_file_sha256") != hashes
            or validation.get("parameters_unchanged") is not True
            or validation.get("stage2_inputs_unchanged") is not True
            or validation.get("prefix_equivalence_verified") is not True
            or validation.get("checkpoint_sha256") != inputs["checkpoint_sha256"]):
        raise ValueError("Smoke checkpoint／Stage-2 inputs／implementation 不符")
    return str(folder)


def execute(roots, phase="smoke", seed=1, stage1_run=None, stage2_run=None, retry=False,
            dry_run=False, formal_summary=None, command=None):
    if seed != 1 or phase not in ("smoke", "train-scores", "full"):
        raise ValueError("Stage 3 只接受 seed=1、smoke／train-scores／full")
    load_configuration(roots.project)  # 核對既有固定 fractions／POT／protocol，不修改舊設定。
    inputs = resolve_inputs(roots, stage1_run, stage2_run)
    hashes = code_hashes(roots.project)
    root = root_path(roots, inputs)
    targets = inputs["targets"][:1] if phase == "smoke" else inputs["targets"]
    records = inspect_train(roots, inputs, targets)
    description = dict(experiment="universal_thresholding_v1", phase=phase, seed=1, inputs=inputs,
        sources=targets, train_records=records, source_file_sha256=hashes, conditions=list(CONDITIONS),
        window=10, eval_batch_size=128, dtype="float64", device="cpu", output_root=str(root),
        test_inference_executed=False, model_retrained=False, optimizer_steps=0)
    if dry_run:
        import json
        print(json.dumps(description, ensure_ascii=False, indent=2, allow_nan=False))
        return description
    if phase == "full":
        description["completed_smoke"] = smoke_gate(root, inputs, hashes)
    folder, lock = claim(root / phase, retry)
    try:
        update_status(folder, state="running", completed=False, seed=1, phase=phase)
        write_json(folder / "launch.json", dict(description, command=command or [sys.executable, *sys.argv]))
        print(f"Stage-3 directory: {folder}", flush=True)
        with (folder / "run.log").open("x") as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    validation = run(roots, inputs, records, root, folder, phase, retry, formal_summary)
                except BaseException:
                    traceback.print_exc()
                    raise
        if hashes != code_hashes(roots.project):
            raise ValueError("Stage-3 implementation 在執行期間改變")
        check_inputs(inputs)
        write_json(folder / "validation.json", validation)
        update_status(folder, state="completed", completed=True, seed=1, phase=phase)
        return folder
    except BaseException as error:
        update_status(folder, state="failed", completed=False, seed=1, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.unlink()


def run(roots, inputs, records, root, folder, phase, retry, formal_summary):
    import torch
    from src.calibration_runtime import load_frozen_model, score_prefix, state_hash
    args = core_args(roots, configuration(roots.project, 1), folder, Path(inputs["stage1"]["checkpoint"]))
    args.retrain, args.checkpoint = False, inputs["stage1"]["checkpoint"]
    core = import_core(args)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    model, before = load_frozen_model(core, inputs["stage1"]["checkpoint"])
    if before != inputs["model_state_sha256"]:
        raise ValueError("Model state 與 Stage 2 不同")
    start = perf_counter()
    pools, manifest = {}, []
    for index, record in enumerate(records):
        target = record["machine"]
        print(f"[{index+1}/{len(records)}] TRAIN only：{target}", flush=True)
        update_status(folder, state="running", completed=False, seed=1, phase="train_scoring", machine=target)
        scores, source = obtain_scores(roots, inputs, core, model, record, root, retry)
        pools[target] = scores
        manifest.append(dict(machine=target, raw_train_length=len(scores), score_count=len(scores),
            scores_path=str(source / "scores.npz"), scores_npz_sha256=file_hash(source / "scores.npz"),
            scores_array_sha256=array_hash(scores), timestamps_array_sha256=array_hash(np.arange(len(scores), dtype=np.int64)),
            checkpoint_sha256=inputs["checkpoint_sha256"], seed=1))
    equivalent = None
    if phase == "smoke":
        record = records[0]
        target = record["machine"]
        length = prefix_length(len(pools[target]), "0.05")
        from src.target_calibration import load_prefix
        _, separately_scored = score_prefix(core, model, load_prefix(record["train_path"], length))
        np.testing.assert_allclose(pools[target][:length], separately_scored, rtol=0, atol=1e-12)
        equivalent = dict(raw_prefix_points=length, max_absolute_difference=float(np.max(np.abs(pools[target][:length]-separately_scored))),
                          tolerance=1e-12, prefix_first_equals_full_score_prefix=True)
        info = fit_train_threshold(core, pools[target][:length], dict(split="train", seed=1, sources=[target],
            score_count=length, raw_prefix_start=0, raw_prefix_stop_exclusive=length, condition="local_05"))
        row, predictions = target_row(inputs, target, "local_05", info["threshold"])
        write_json(folder / "local_threshold.json", info)
        write_json(folder / "smoke_metrics.json", row)
        write_json(folder / "prefix_equivalence.json", equivalent)
        print(f"Smoke Raw Precision={row['raw_precision']:.8f}, Recall={row['raw_recall']:.8f}, F1={row['raw_f1']:.8f}", flush=True)
    elif phase == "full":
        if len(pools) != 28 or list(pools) != inputs["targets"]:
            raise ValueError("Global pool 必須包含全部 28 TRAIN machines")
        pooled = np.concatenate([pools[t] for t in inputs["targets"]])
        provenance = dict(split="train", seed=1, sources=inputs["targets"], score_count=len(pooled),
            condition="global", score_array_sha256=array_hash(pooled), train_score_manifest=manifest)
        update_status(folder, state="running", completed=False, seed=1, phase="global_pot_fit")
        global_info = fit_train_threshold(core, pooled, provenance)
        global_info["checkpoint_sha256"] = inputs["checkpoint_sha256"]
        write_json(folder / "global_threshold.json", global_info)
        rows, thresholds = [], []
        for target_index, target in enumerate(inputs["targets"]):
            print(f"[{target_index+1}/28] Local thresholds／Stage-2 TEST-score reuse：{target}", flush=True)
            values = {"global": global_info["threshold"]}
            for condition in CONDITIONS[1:]:
                length = len(pools[target]) if condition == "local_full" else prefix_length(len(pools[target]), FRACTIONS[condition])
                update_status(folder, state="running", completed=False, seed=1, phase="local_pot_fit",
                              target=target, target_index=target_index+1, condition=condition)
                info = fit_train_threshold(core, pools[target][:length], dict(split="train", seed=1, sources=[target],
                    score_count=length, raw_prefix_start=0, raw_prefix_stop_exclusive=length, condition=condition,
                    full_train_score_sha256=array_hash(pools[target]), calibration_score_sha256=array_hash(pools[target][:length])))
                values[condition] = info["threshold"]
                out = folder / "targets" / target
                out.mkdir(parents=True, exist_ok=True)
                write_json(out / (condition+"_threshold.json"), dict(info, checkpoint_sha256=inputs["checkpoint_sha256"]))
            diagnostics = {}
            for condition, threshold in values.items():
                row, predictions = target_row(inputs, target, condition, threshold)
                rows.append(row)
                print(f"{target} {condition}: Raw P={row['raw_precision']:.6f}, R={row['raw_recall']:.6f}, F1={row['raw_f1']:.6f}", flush=True)
                change = threshold_change(values["global"], threshold)
                diagnostics[condition] = dict(global_threshold=change["zero_shot_threshold"],
                    local_threshold=change["calibrated_threshold"], absolute_shift=change["absolute_shift"],
                    signed_shift=change["signed_shift"], relative_ratio=change["relative_ratio"],
                    relative_ratio_reason="global_threshold_near_zero" if change["relative_ratio_status"] ==
                        "zero_shot_threshold_near_zero" else change["relative_ratio_status"])
                with (folder / "targets" / target / (condition+"_predictions.npz")).open("xb") as stream:
                    np.savez_compressed(stream, predictions=predictions)
            write_json(folder / "targets" / target / "threshold_diagnostics.json", diagnostics)
            thresholds.append(dict(target=target, group=int(target.split("-")[1]), seed=1,
                **{c+"_threshold": values[c] for c in CONDITIONS}))
        summarize(roots, inputs, folder, rows, thresholds, manifest, formal_summary)
    else:
        from src.universal_evaluation_analysis import write_csv
        write_csv(folder / "train_score_manifest.csv", manifest)
    after = state_hash(model)
    if before != after or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("Stage 3 model state／freeze 改變")
    for item, record in zip(manifest, records):
        read_train_scores(Path(item["scores_path"]).parent, record, inputs)
        if file_hash(record["train_path"]) != record["raw_file_sha256"]:
            raise ValueError("TRAIN file 在 Stage 3 改變")
    check_inputs(inputs)
    return dict(seed=1, phase=phase, parameters_unchanged=True, parameter_hash_before=before,
        parameter_hash_after=after, checkpoint_sha256=inputs["checkpoint_sha256"],
        unique_universal_checkpoint_hashes=1, optimizer_steps=0, model_retrained=False,
        test_inference_executed=False, stage2_inputs_unchanged=True, test_scores_reused=True,
        prefix_equivalence_verified=equivalent is not None if phase == "smoke" else phase == "full",
        prefix_equivalence_basis="smoke prefix-first inference + regression window/score proof" if phase != "train-scores" else "not_fitted",
        threshold_conditions=list(CONDITIONS) if phase == "full" else (["local_05"] if phase == "smoke" else []),
        target_condition_rows=140 if phase == "full" else (1 if phase == "smoke" else 0),
        train_machine_count=len(pools), total_train_score_count=sum(len(s) for s in pools.values()),
        pot_stream_score_count=0, continuous_metrics_tolerance=1e-12, elapsed_work_sec=perf_counter()-start)
