"""Stage 3 Raw-only adapter 與 descriptive aggregation，不做 PA／最佳 fraction 選擇。"""
import csv
from pathlib import Path

import numpy as np

from src.continuous_metrics import continuous_metrics
from src.seed_replication import file_hash, read_json, write_json
from src.target_calibration import array_hash, check_auc_invariance
from src.universal_evaluation_analysis import formal_comparison, write_csv
from src.universal_threshold_artifacts import CONDITIONS, test_arrays


def raw_metrics(scores, labels, threshold):
    """原 src.experiment.additional_metrics 的 Raw 分支原公式；不需要 PA payload。"""
    if not np.isfinite(threshold):
        raise ValueError("threshold 必須 finite")
    raw = scores > threshold
    tp = int(np.sum(raw & (labels == 1)))
    fp = int(np.sum(raw & (labels == 0)))
    fn = int(np.sum(~raw & (labels == 1)))
    tn = int(np.sum(~raw & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
    return dict(precision=precision, recall=recall, f1=f1, TP=tp, TN=tn, FP=fp, FN=fn), raw


def target_row(inputs, target, condition, threshold):
    if condition not in CONDITIONS:
        raise ValueError("不得新增或挑選 best-fraction condition")
    scores, labels, timestamps, original = test_arrays(inputs, target)
    continuous = continuous_metrics(scores, labels)
    check_auc_invariance({"continuous_score": original}, {"continuous_score": continuous}, 1e-12)
    raw, predictions = raw_metrics(scores, labels, threshold)
    meta = read_json(Path(inputs["stage2_run"]) / "targets" / target / "metadata.json")
    return dict(target=target, group=int(target.split("-")[1]), seed=1, condition=condition,
        threshold=float(threshold), raw_precision=raw["precision"], raw_recall=raw["recall"], raw_f1=raw["f1"],
        tp=raw["TP"], tn=raw["TN"], fp=raw["FP"], fn=raw["FN"],
        pr_auc=continuous["pr_auc_trapezoidal"], average_precision=continuous["average_precision"],
        roc_auc=continuous["roc_auc"], checkpoint_sha256=inputs["checkpoint_sha256"],
        test_scores_sha256=meta["score_array_sha256"], test_labels_sha256=meta["label_array_sha256"],
        test_timestamps_sha256=array_hash(timestamps),
        test_archive_sha256=file_hash(Path(inputs["stage2_run"]) / "targets" / target / "scores.npz")), predictions


def validate_rows(rows, targets, checkpoint_sha256):
    expected = {(m, c) for m in targets for c in CONDITIONS}
    if (len(targets) != 28 or len(set(targets)) != 28 or len(rows) != 140
            or {(r["target"], r["condition"]) for r in rows} != expected
            or any(r["seed"] != 1 or r["checkpoint_sha256"] != checkpoint_sha256 for r in rows)):
        raise ValueError("Stage 3 必須恰為 28 targets × 5 conditions、seed=1、同一 checkpoint")
    global_values = {r["threshold"] for r in rows if r["condition"] == "global"}
    if len(global_values) != 1:
        raise ValueError("Global threshold 必須為所有 targets 共用的一個 scalar")
    for target in targets:
        records = [r for r in rows if r["target"] == target]
        for key in ("test_scores_sha256", "test_labels_sha256", "test_timestamps_sha256", "test_archive_sha256"):
            if len({r[key] for r in records}) != 1:
                raise ValueError("同 target 的 Stage-2 TEST arrays／timestamps 改變")
        for key in ("pr_auc", "average_precision", "roc_auc"):
            if not np.allclose([r[key] for r in records], records[0][key], atol=1e-12, rtol=0):
                raise ValueError("Threshold conditions 改變 continuous metrics")


def aggregate(rows, targets):
    validate_rows(rows, targets, rows[0]["checkpoint_sha256"])
    baseline = {r["target"]: r for r in rows if r["condition"] == "global"}
    macros, recovery, groups = [], [], []
    for condition in CONDITIONS:
        records = [r for r in rows if r["condition"] == condition]
        values = [r["raw_f1"] for r in records]
        deltas = [r["raw_f1"]-baseline[r["target"]]["raw_f1"] for r in records]
        counts = dict(improved_vs_global=sum(d > 0 for d in deltas), degraded_vs_global=sum(d < 0 for d in deltas),
                      unchanged_vs_global=sum(d == 0 for d in deltas))
        macros.append(dict(condition=condition, seed=1, mean_raw_precision=float(np.mean([r["raw_precision"] for r in records])),
            mean_raw_recall=float(np.mean([r["raw_recall"] for r in records])), mean_raw_f1=float(np.mean(values)),
            median_raw_f1=float(np.median(values)), sample_std_raw_f1=float(np.std(values, ddof=1)),
            **{k: (None if condition == "global" else v) for k, v in counts.items()}))
        if condition != "global":
            recovery.append(dict(condition=condition, seed=1, mean_recovery=float(np.mean(deltas)),
                median_recovery=float(np.median(deltas)), **counts))
        for group in (1, 2, 3):
            subset = [r for r in records if r["group"] == group]
            groups.append(dict(condition=condition, group=group, seed=1, target_count=len(subset),
                mean_raw_precision=float(np.mean([r["raw_precision"] for r in subset])),
                mean_raw_recall=float(np.mean([r["raw_recall"] for r in subset])),
                mean_raw_f1=float(np.mean([r["raw_f1"] for r in subset])),
                mean_recovery_vs_global=float(np.mean([r["raw_f1"]-baseline[r["target"]]["raw_f1"] for r in subset]))))
    return macros, recovery, groups


def comparisons(roots, inputs, rows, path):
    target_folders = {t: Path(inputs["stage2_run"]) / "targets" / t for t in inputs["targets"]}
    from src.universal_evaluation_analysis import validate_target
    ranking_rows = [validate_target(target_folders[t], t, inputs["stage1"]) for t in inputs["targets"]]
    _, _, hashes = formal_comparison(roots, ranking_rows, path, target_folders)
    with Path(path).open(newline="") as stream:
        formal_rows = list(csv.DictReader(stream))
    references, metadata_hashes = {}, dict(hashes)
    for row in formal_rows:
        checkpoint = Path(row["checkpoint_path"])
        evaluation = read_json(checkpoint.parent.parent.parent / "runs" / checkpoint.parent.name / "evaluation.json")
        raw = evaluation["raw"]
        for key in ("precision", "recall", "f1"):
            if not np.isfinite(raw[key]) or not 0 <= raw[key] <= 1 or not np.isclose(
                    raw[key], float(row["raw_"+key]), rtol=0, atol=1e-12):
                raise ValueError("Formal Raw metric 與 summary 不一致")
        references[row["target_machine"], row["setting"]] = raw
    paired = [dict(target=r["target"], seed=1, condition=r["condition"],
        per_machine_raw_f1=references[r["target"], "per_machine"]["f1"], universal_raw_f1=r["raw_f1"],
        per_machine_minus_universal=references[r["target"], "per_machine"]["f1"]-r["raw_f1"]) for r in rows]
    statistics = []
    for condition in CONDITIONS:
        values = [r["per_machine_minus_universal"] for r in paired if r["condition"] == condition]
        statistics.append(dict(condition=condition, seed=1, mean_gap=float(np.mean(values)), median_gap=float(np.median(values)),
            universal_better=sum(v < 0 for v in values), per_machine_better=sum(v > 0 for v in values), ties=sum(v == 0 for v in values)))
    deployments = []
    for setting, threshold_type in (("per_machine", "local_full"), ("cross_machine_zero_shot", "source_only")):
        values = [references[t, setting] for t in inputs["targets"]]
        deployments.append(dict(deployment=setting, models=28, threshold_type=threshold_type, seed=1,
            mean_raw_precision=float(np.mean([r["precision"] for r in values])),
            mean_raw_recall=float(np.mean([r["recall"] for r in values])), mean_raw_f1=float(np.mean([r["f1"] for r in values]))))
    return paired, statistics, deployments, metadata_hashes


def summarize(roots, inputs, folder, rows, thresholds, train_manifest, formal_summary=None):
    macros, recovery, groups = aggregate(rows, inputs["targets"])
    outputs = folder / "summary"
    outputs.mkdir(exist_ok=False)
    for name, table in (("target_condition_metrics.csv", rows), ("target_thresholds.csv", thresholds),
                        ("train_score_manifest.csv", train_manifest), ("macro_condition_metrics.csv", macros),
                        ("recovery_statistics.csv", recovery), ("group_condition_metrics.csv", groups)):
        write_csv(outputs / name, table)
    deployments = [dict(deployment="universal", models=1, threshold_type=r["condition"], seed=1,
        mean_raw_precision=r["mean_raw_precision"], mean_raw_recall=r["mean_raw_recall"], mean_raw_f1=r["mean_raw_f1"])
        for r in macros]
    candidate = Path(formal_summary) if formal_summary else roots.results / "summaries/formal_28_machine_results.csv"
    comparison_status, protected = "unavailable", {}
    if formal_summary or candidate.exists():
        paired, gaps, reference, protected = comparisons(roots, inputs, rows, candidate)
        write_csv(outputs / "per_machine_comparison.csv", paired)
        write_csv(outputs / "per_machine_gap_statistics.csv", gaps)
        deployments = reference + deployments
        comparison_status = "validated_seed1_artifacts"
    write_csv(outputs / "deployment_comparison.csv", deployments)
    summary = dict(seed=1, target_count=28, condition_count=5, result_rows=140, conditions=list(CONDITIONS),
        unique_universal_checkpoint_hashes=1, checkpoint_sha256=inputs["checkpoint_sha256"],
        stage2_run=inputs["stage2_run"], stage2_code_sha256=inputs["stage2_code_sha256"],
        model_retrained=False, optimizer_steps=0, test_inference_executed=False, test_scores_reused=True,
        auc_tolerance=1e-12, variability="跨 targets descriptive variability，非多 seed uncertainty／顯著性",
        formal_comparison=comparison_status, formal_input_sha256=protected,
        lomo_calibration_comparison="未加入；optional context，不影響核心 universal comparison",
        actual_slurm_elapsed=None, actual_slurm_maxrss=None, systems_metrics_status="待使用者由 sacct 補充；request 不是測量值")
    write_json(folder / "summary.json", summary)
    for filename, digest in protected.items():
        if file_hash(filename) != digest:
            raise ValueError("Formal baseline artifacts 在分析期間改變")
    return summary
