"""唯讀驗證 universal TEST artifacts，另存 continuous-score summary；不計算 threshold metrics。"""
import csv
from pathlib import Path

import numpy as np

from src.continuous_metrics import continuous_metrics
from src.seed_replication import file_hash, read_json, safe_path, write_json
from src.target_calibration import array_hash
from src.universal_training import configuration


def validate_target(folder, target, bundle):
    folder = safe_path(folder)
    meta = read_json(safe_path(folder / "metadata.json"))
    validation = read_json(safe_path(folder / "validation.json"))
    metrics = read_json(safe_path(folder / "metrics.json"))
    fixed = dict(experiment="universal_continuous_evaluation_v1", setting="universal_checkpoint_evaluation",
        target=target, seed=1, protocol="v2", model="TranAD", dataset="SMD", window=10,
        eval_batch_size=128, device="cpu", dtype="float64", model_retrained=False, optimizer_steps=0,
        model_eval=True, requires_grad=False, normalization_fitted=False, zero_shot=False,
        target_train_used_in_stage1=True, test_used_for_training=False, test_used_for_validation=False,
        test_used_for_adaptation=False, checkpoint_id=bundle["run_id"], checkpoint_path=bundle["checkpoint"],
        checkpoint_sha256=bundle["checkpoint_sha256"], sources=bundle["sources"], source_count=28,
        evaluation_code_sha256=bundle["evaluation_code_sha256"],
        stage1_run=bundle["run"], label_mapping="interpretation_label: zero-based, end-exclusive, feature union")
    if any(meta.get(key) != value for key, value in fixed.items()):
        raise ValueError(f"{target} evaluation metadata / universal checkpoint identity 不符")
    for key in ("parameters_unchanged", "all_windows_single_machine", "score_label_alignment", "scores_finite",
                "labels_binary", "checkpoint_unchanged", "stage1_inputs_unchanged"):
        if validation.get(key) is not True:
            raise ValueError(f"{target} 缺少 runtime invariant：{key}")
    if (not validation.get("parameter_hash_before")
            or validation["parameter_hash_before"] != validation.get("parameter_hash_after")
            or validation.get("model_retrained") is not False or validation.get("optimizer_steps") != 0
            or validation.get("checkpoint_sha256") != bundle["checkpoint_sha256"]):
        raise ValueError("Model freeze / checkpoint validation 不符")
    artifact = safe_path(folder / "scores.npz")
    if file_hash(artifact) != validation.get("scores_npz_sha256"):
        raise ValueError("scores.npz 已改變")
    with np.load(artifact, allow_pickle=False) as archive:
        if set(archive.files) != {"scores", "labels", "timestamps"}:
            raise ValueError("Universal scores archive schema 不符")
        scores, labels, timestamps = [archive[name] for name in ("scores", "labels", "timestamps")]
    count = meta["num_test_points"]
    if (scores.shape != (count,) or labels.shape != (count,) or scores.dtype != np.float64
            or labels.dtype != np.int64 or not np.isfinite(scores).all()
            or not np.isin(labels, [0, 1]).all() or not np.array_equal(timestamps, np.arange(count))):
        raise ValueError("Scores / labels / timestamps 的 dtype、長度或數值不符")
    if (any(meta.get(key) != count for key in ("score_count", "label_count", "score_end_exclusive"))
            or meta.get("score_start_timestamp") != 0 or meta["test_input"]["raw_test_length"] != count
            or validation.get("score_count") != count or validation.get("label_count") != count):
        raise ValueError("Raw TEST / score / label 對齊 count 不一致")
    for name, value in (("score_array_sha256", scores), ("label_array_sha256", labels)):
        if array_hash(value) != meta.get(name) or meta.get(name) != validation.get(name):
            raise ValueError("Score / label array hash 不一致")
    actual = continuous_metrics(scores, labels)
    for key, value in actual.items():
        if not np.isfinite(metrics[key]) or not np.isclose(value, metrics[key], atol=1e-12, rtol=0):
            raise ValueError("Continuous metrics 與保存的 scores / labels 不一致")
    prevalence = float(labels.mean())
    if (meta["num_positive_labels"] != int(labels.sum()) or meta["anomaly_prevalence"] != prevalence
            or metrics["anomaly_prevalence"] != prevalence
            or not np.isclose(metrics["pr_auc_minus_prevalence"], actual["pr_auc_trapezoidal"]-prevalence,
                              atol=1e-12, rtol=0)):
        raise ValueError("Anomaly prevalence 不一致")
    return dict(target=target, group=int(target.split("-")[1]), seed=1, test_points=count,
        anomaly_points=int(labels.sum()), anomaly_prevalence=prevalence,
        pr_auc=actual["pr_auc_trapezoidal"], average_precision=actual["average_precision"],
        roc_auc=actual["roc_auc"], pr_auc_minus_prevalence=actual["pr_auc_trapezoidal"]-prevalence,
        checkpoint_id=bundle["run_id"], checkpoint_sha256=bundle["checkpoint_sha256"])


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def formal_comparison(roots, universal_rows, path, target_folders):
    """讀指定 seed-1 summary，並以其 checkpoint/run identity 找回原 metadata；不使用 latest。"""
    path = safe_path(Path(path).expanduser().absolute())
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    canonical = configuration(roots.project, 1)["sources"]
    expected = {(target, setting) for target in canonical for setting in
                ("per_machine", "cross_machine_zero_shot")}
    if (len(rows) != 56 or {(r["target_machine"], r["setting"]) for r in rows} != expected
            or any(r["seed"] != "1" for r in rows)):
        raise ValueError("Formal comparison 需要恰好 56 個 seed-1 target/setting，不接受 smoke 或其他 seed")
    inputs = {str(path): file_hash(path)}
    values = {}
    for row in rows:
        target, setting = row["target_machine"], row["setting"]
        checkpoint = safe_path(Path(row["checkpoint_path"]))
        if checkpoint.name != "model.ckpt" or checkpoint.parent.parent.name != "checkpoints":
            raise ValueError("Formal summary checkpoint path 不符合原 namespace")
        run = checkpoint.parent.parent.parent / "runs" / checkpoint.parent.name
        metadata_path, evaluation_path = safe_path(run / "metadata.json"), safe_path(run / "evaluation.json")
        meta, evaluation = read_json(metadata_path), read_json(evaluation_path)
        fixed = dict(target_machine=target, setting=setting, seed=1, protocol="v2", epochs=5,
            batch_size=128, eval_batch_size=128, window=10, learning_rate=1e-4, device="cpu", dtype="float64",
            training_shuffle=True, smoke_test=False, label_source="corrected_interpretation_label_union")
        if any(meta.get(key) != value for key, value in fixed.items()):
            raise ValueError("Formal comparison metadata 並非 validated seed-1 protocol-v2")
        sources = [target] if setting == "per_machine" else [m for m in canonical if m != target]
        if (meta.get("source_machines") != sources or row["source_machines"].split(";") != sources
                or meta.get("checkpoint") != str(checkpoint)):
            raise ValueError("Formal comparison sources / checkpoint identity 不符")
        target_meta = read_json(target_folders[target] / "metadata.json")
        description = target_meta["test_input"]
        data_hashes = meta["data_sha256"]
        for split, key in (("test", "test_sha256"), ("interpretation_label", "label_file_sha256")):
            digests = [digest for filename, digest in data_hashes.items()
                       if Path(filename).parent.name == split and Path(filename).name == target+".txt"]
            if digests != [description[key]]:
                raise ValueError("Formal / universal 的 TEST 或 interpretation_label 來源不同")
        value = float(row["pr_auc"])
        if (not np.isfinite(value) or not 0 <= value <= 1
                or not np.isclose(value, evaluation["continuous_score"]["pr_auc_trapezoidal"], atol=1e-12, rtol=0)
                or int(row["test_points"]) != target_meta["num_test_points"]
                or meta["test_timestamps"] != target_meta["num_test_points"]):
            raise ValueError("Formal PR-AUC / TEST counts 與 artifacts 不一致")
        values[target, setting] = value
        inputs[str(metadata_path)] = file_hash(metadata_path)
        inputs[str(evaluation_path)] = file_hash(evaluation_path)
    comparison = [dict(target=row["target"], per_machine_pr_auc=values[row["target"], "per_machine"],
        lomo_pr_auc=values[row["target"], "cross_machine_zero_shot"], universal_pr_auc=row["pr_auc"])
        for row in universal_rows]
    macro = [dict(setting=setting, seed=1, targets=28, mean_pr_auc=float(np.mean([r[key] for r in comparison])))
             for setting, key in (("per_machine", "per_machine_pr_auc"), ("lomo_cross_machine", "lomo_pr_auc"),
                                  ("universal_checkpoint", "universal_pr_auc"))]
    for filename, digest in inputs.items():
        if file_hash(filename) != digest:
            raise ValueError("Formal comparison inputs 在讀取期間改變")
    return comparison, macro, inputs


def summarize(roots, bundle, folder, targets, formal_summary=None):
    canonical = configuration(roots.project, 1)["sources"]
    if targets != canonical or len(set(targets)) != 28:
        raise ValueError("Formal universal summary 必須恰為 canonical 28 台，不能混入單台 smoke")
    target_folders = {target: safe_path(folder / "targets" / target) for target in canonical}
    actual = {p.name for p in (folder / "targets").iterdir()}
    if actual != set(canonical):
        raise ValueError("Summary target directories 有 missing / extra / duplicates")
    rows = [validate_target(target_folders[target], target, bundle) for target in canonical]
    if len({row["checkpoint_sha256"] for row in rows}) != 1:
        raise ValueError("28 targets 未使用同一 checkpoint")
    hashes = [read_json(target_folders[target] / "validation.json")["parameter_hash_before"] for target in canonical]
    if len(set(hashes)) != 1:
        raise ValueError("28 targets 的 model-state hash 不同")
    summary = dict(seed=1, setting="universal_checkpoint_evaluation", target_count=28,
        checkpoint_id=bundle["run_id"], checkpoint_path=bundle["checkpoint"],
        checkpoint_sha256=bundle["checkpoint_sha256"], unique_universal_checkpoint_hashes=1,
        evaluation_code_sha256=bundle["evaluation_code_sha256"],
        mean_pr_auc=float(np.mean([r["pr_auc"] for r in rows])),
        mean_average_precision=float(np.mean([r["average_precision"] for r in rows])),
        mean_roc_auc=float(np.mean([r["roc_auc"] for r in rows])),
        median_pr_auc=float(np.median([r["pr_auc"] for r in rows])),
        sample_std_pr_auc=float(np.std([r["pr_auc"] for r in rows], ddof=1)),
        min_pr_auc_target=min(rows, key=lambda r: r["pr_auc"])["target"], min_pr_auc=min(r["pr_auc"] for r in rows),
        max_pr_auc_target=max(rows, key=lambda r: r["pr_auc"])["target"], max_pr_auc=max(r["pr_auc"] for r in rows),
        statistics_unit="28 targets 的 descriptive macro statistics；只有一個 seed，不是 28 seeds",
        random_ranking_reference="prevalence 為近似 descriptive reference，不代表 statistical significance")
    candidate = Path(formal_summary) if formal_summary else roots.results / "summaries/formal_28_machine_results.csv"
    comparison = None
    if formal_summary or candidate.exists():
        comparison, macro, inputs = formal_comparison(roots, rows, candidate, target_folders)
        summary.update(formal_comparison="validated_seed1_summary_and_run_metadata", formal_input_sha256=inputs)
    else:
        summary.update(formal_comparison="unavailable", formal_comparison_reason=f"找不到 {candidate}；不填入手抄數值")
    out = safe_path(folder / "summary")
    out.mkdir(exist_ok=False)
    write_csv(out / "target_metrics.csv", rows)
    if comparison is not None:
        write_csv(out / "seed1_comparison.csv", comparison)
        write_csv(out / "seed1_macro_comparison.csv", macro)
    write_json(out / "summary.json", summary)
    print(f"28 台 macro PR-AUC={summary['mean_pr_auc']:.8f}, AP={summary['mean_average_precision']:.8f}, "
          f"ROC-AUC={summary['mean_roc_auc']:.8f}", flush=True)
    return summary
