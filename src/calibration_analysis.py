"""Calibration 的唯讀結果稽核、Raw recovery 統計與圖表；不執行模型或 POT。"""
import csv
from pathlib import Path

import numpy as np

from src.calibration_artifacts import validate_attempt_path
from src.calibration_protocol import canonical_fraction, prefix_length
from src.target_calibration import (array_hash, check_auc_invariance, check_baseline_metrics, file_hash,
                                    load_formal_scores, metric_recovery, read_json,
                                    threshold_change, write_json)


def summarize_run(folder, row, config, namespace, machines):
    target, fraction = row["target"], row["fraction"]
    folder = validate_attempt_path(namespace, folder, target, fraction)
    meta, evaluation = read_json(folder / "metadata.json"), read_json(folder / "evaluation.json")
    status, checks = read_json(folder / "status.json"), read_json(folder / "validation.json")
    expected = dict(experiment_type="target_threshold_calibration", base_protocol="v2",
        target=target, seed=1, model="TranAD", dataset="SMD", device="cpu", dtype="float64",
        model_retrained=False, optimizer_steps=0, parameters_unchanged=True,
        model_frozen=True, continuous_scores_reused=True, normalization_fitted=False,
        test_inference_executed=False, target_test_used_for_threshold=False,
        target_train_used_for_training=False, calibration_source="target_train_temporal_prefix",
        threshold_source="target_train_prefix_scores", window_size=10, eval_batch_size=128,
        formal_task_id=len(machines) + machines.index(target))
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("calibration metadata 不符合固定 protocol / target / frozen design")
    if canonical_fraction(meta["fraction"]) != fraction or status["task_id"] != int(row["task_id"]):
        raise ValueError("fraction / task ID 與 manifest 不同")
    if not status["completed"] or status["exit_code"] != 0 or meta["smoke_test"]:
        raise ValueError("不接受失敗、未完成或 smoke 結果")
    expected_pot = dict(q=1e-5, requested_level=0.99995, multiplier=1.06, stream_score_count=0)
    if any(meta["pot"].get(key) != value for key, value in expected_pot.items()):
        raise ValueError("calibration POT 參數或 test-stream isolation 不符")
    length = prefix_length(meta["target_train_total_raw_points"], fraction)
    if (meta["calibration_raw_points"] != length or meta["calibration_windows"] != length
            or meta["prefix"]["start"] != 0 or meta["prefix"]["stop_exclusive"] != length):
        raise ValueError("calibration raw-prefix / window boundaries 不符")
    for key in ("parameter_freeze", "test_scores_unchanged", "labels_unchanged", "pr_auc_unchanged",
                "ap_unchanged", "roc_auc_unchanged", "formal_artifacts_unchanged"):
        if checks.get(key) is not True:
            raise ValueError(f"runtime validation 不通過：{key}")
    if checks["optimizer_steps"] != 0 or checks["pot_test_stream_count"] != 0:
        raise ValueError("optimizer / POT stream invariant 不符")
    if meta["parameter_hash_before"] != meta["parameter_hash_after"]:
        raise ValueError("parameters 改變")
    if (meta["run_id"] != folder.parent.name or meta["attempt_id"] != folder.name
            or status["run_id"] != meta["run_id"] or status["attempt_id"] != folder.name):
        raise ValueError("metadata / status / attempt identity 不符")
    if (not checks["test_score_count_unchanged"] or checks["formal_test_score_count"] != meta["test_points"]
            or checks["calibrated_test_score_count"] != meta["test_points"]
            or meta["pot_fit"]["source_score_count"] != length):
        raise ValueError("TEST score count 或 calibration POT input count 不符")
    for key, original in meta["input_paths"].items():
        if file_hash(original) != meta["protected_input_sha256"][key]:
            raise ValueError(f"formal input 不再與 calibration provenance 一致：{key}")
    if (meta["formal_checkpoint_hash"] != meta["protected_input_sha256"]["checkpoint"]
            or meta["formal_score_artifact_hash"] != meta["protected_input_sha256"]["scores.npz"]):
        raise ValueError("formal checkpoint / score hash aliases 不一致")
    original_scores, original_labels = load_formal_scores(meta["formal_score_artifact"], meta["test_points"])
    with np.load(folder / "scores.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["scores"], original_scores)
        np.testing.assert_array_equal(saved["labels_final"], original_labels)
        np.testing.assert_array_equal(saved["raw_predictions"], original_scores > evaluation["calibrated"]["threshold"])
    predictions = original_scores > evaluation["calibrated"]["threshold"]
    tp = int(np.sum(predictions & (original_labels == 1)))
    tn = int(np.sum(~predictions & (original_labels == 0)))
    fp = int(np.sum(predictions & (original_labels == 0)))
    fn = int(np.sum(~predictions & (original_labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    expected_raw = dict(TP=tp, TN=tn, FP=fp, FN=fn, precision=precision, recall=recall,
                        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    for key, value in expected_raw.items():
        np.testing.assert_allclose(evaluation["calibrated"]["raw"][key], value, rtol=0,
                                   atol=config["auc_tolerance"], equal_nan=False)
    if array_hash(original_scores) != meta["test_scores_sha256"] or array_hash(original_labels) != meta["labels_sha256"]:
        raise ValueError("saved score / label hash 不一致")
    baseline, calibrated = evaluation["zero_shot"], evaluation["calibrated"]
    original_evaluation = read_json(meta["input_paths"]["evaluation.json"])
    if baseline["threshold"] != original_evaluation["threshold"] or calibrated["threshold"] != meta["threshold_value"]:
        raise ValueError("evaluation threshold 與 formal / calibration metadata 不一致")
    check_baseline_metrics(original_evaluation, baseline, config["auc_tolerance"])
    check_auc_invariance(baseline, calibrated, config["auc_tolerance"])
    if evaluation["continuous_score"] != calibrated["continuous_score"]:
        raise ValueError("continuous metric aliases 不一致")
    recovery = metric_recovery(baseline, calibrated)
    if recovery != evaluation["recovery"]:
        raise ValueError("Raw recovery 計算不一致")
    changes = threshold_change(baseline["threshold"], calibrated["threshold"])
    threshold = read_json(folder / "threshold.json")
    if any(threshold.get(key) != value for key, value in changes.items()):
        raise ValueError("threshold shifts / ratio 不一致")
    result = dict(target=target, group=int(target.split("-")[1]), fraction=float(fraction), seed=1,
                  calibration_raw_points=length, calibration_windows=length,
                  target_train_total_raw_points=meta["target_train_total_raw_points"],
                  **changes, **recovery, formal_run_id=meta["formal_run_id"],
                  formal_task_id=meta["formal_task_id"], formal_checkpoint=meta["formal_checkpoint"],
                  run_path=str(folder))
    for section, prefix in (("raw", "raw"), ("point_adjusted", "pa")):
        for metric in ("precision", "recall", "f1", "TP", "TN", "FP", "FN"):
            result[f"zero_shot_{prefix}_{metric.lower()}"] = baseline[section][metric]
            result[f"calibrated_{prefix}_{metric.lower()}"] = calibrated[section][metric]
    result.update(calibrated["continuous_score"])
    result["total_seconds"] = meta["total_seconds"]
    return result, meta


def count_changes(values):
    values = np.asarray(values)
    return dict(improved=int(np.sum(values > 0)), degraded=int(np.sum(values < 0)),
                unchanged=int(np.sum(values == 0)))


def build_tables(rows, fractions):
    # Zero-shot 每台只算一次，不把同 target 的三個 calibration rows 當獨立樣本。
    zero = [row for row in rows if canonical_fraction(row["fraction"]) == fractions[0]]
    if len(zero) != 28 or len({row["target"] for row in zero}) != 28:
        raise ValueError("zero-shot summary 必須有 28 個不同 targets")
    settings = [("zero_shot", zero, "zero_shot")]
    settings += [(fraction, [row for row in rows if canonical_fraction(row["fraction"]) == fraction], "calibrated")
                 for fraction in fractions]
    statistics = []
    for setting, selected, prefix in settings:
        if len(selected) != 28:
            raise ValueError("每個 setting 需有 28 targets")
        record = dict(setting=setting, targets=len(selected))
        for metric in ("precision", "recall", "f1"):
            values = [row[f"{prefix}_raw_{metric}"] for row in selected]
            record[f"mean_raw_{metric}"] = float(np.mean(values))
            record[f"median_raw_{metric}"] = float(np.median(values))
        values = [row[f"{prefix}_raw_f1"] for row in selected]
        record["sample_std_raw_f1"] = float(np.std(values, ddof=1))
        record["iqr_raw_f1"] = float(np.quantile(values, .75) - np.quantile(values, .25))
        statistics.append(record)
    recovery_rows = [{key: row[key] for key in ("target", "group", "fraction", "seed",
        "raw_precision_delta", "raw_recall_delta", "raw_f1_delta", "zero_shot_threshold",
        "calibrated_threshold", "absolute_shift", "signed_shift", "relative_ratio", "relative_ratio_status")}
                     for row in rows]
    groups, counts = [], {}
    for fraction in fractions:
        selected = [row for row in rows if canonical_fraction(row["fraction"]) == fraction]
        counts[fraction] = {metric: count_changes([row[f"raw_{metric}_delta"] for row in selected])
                            for metric in ("precision", "recall", "f1")}
        counts[fraction]["mean_raw_f1_recovery"] = float(np.mean([row["raw_f1_delta"] for row in selected]))
        for group in (1, 2, 3):
            subset = [row for row in selected if row["group"] == group]
            record = dict(fraction=float(fraction), group=group, targets=len(subset))
            for metric in ("f1", "recall"):
                values = [row[f"raw_{metric}_delta"] for row in subset]
                record[f"mean_raw_{metric}_recovery"] = float(np.mean(values))
                record[f"median_raw_{metric}_recovery"] = float(np.median(values))
                for label, count in count_changes(values).items():
                    record[f"raw_{metric}_{label}"] = count
            groups.append(record)
    return recovery_rows, statistics, groups, counts


def write_csv(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figures(rows, fractions, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    out.mkdir(exist_ok=False)
    zero = [row for row in rows if canonical_fraction(row["fraction"]) == fractions[0]]
    by_fraction = [[row for row in rows if canonical_fraction(row["fraction"]) == fraction] for fraction in fractions]
    labels = ["Zero-shot", "5%", "10%", "20%"]

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=200)
        plt.close(fig)

    for metric in ("f1", "recall"):
        fig, ax = plt.subplots(figsize=(7, 5))
        values = [[row[f"zero_shot_raw_{metric}"] for row in zero]]
        values += [[row[f"calibrated_raw_{metric}"] for row in subset] for subset in by_fraction]
        ax.boxplot(values, tick_labels=labels)
        ax.set_ylabel(f"Raw {metric.upper() if metric == 'f1' else 'Recall'}")
        save(fig, f"raw_{metric}_distribution")
    fig, ax = plt.subplots(figsize=(14, 5))
    positions = np.arange(28)
    for index, subset in enumerate(by_fraction):
        ax.bar(positions + (index - 1) * .25, [row["raw_f1_delta"] for row in subset],
               width=.25, label=labels[index + 1])
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(positions, [row["target"].replace("machine-", "") for row in zero], rotation=65)
    ax.set_ylabel("Raw F1 recovery (calibrated - zero-shot)")
    ax.legend()
    save(fig, "raw_f1_recovery_by_target")
    fig, ax = plt.subplots(figsize=(6, 6))
    for index, subset in enumerate(by_fraction):
        ax.scatter([row["zero_shot_threshold"] for row in subset],
                   [row["calibrated_threshold"] for row in subset], label=labels[index + 1], alpha=.75)
    values = [row[key] for row in rows for key in ("zero_shot_threshold", "calibrated_threshold")]
    low, high = min(values), max(values)
    if low == high:
        high = low + 1
    ax.plot([low, high], [low, high], "k--", lw=1)
    ax.set(xlabel="Zero-shot source threshold", ylabel="Target-calibrated threshold")
    ax.legend()
    save(fig, "threshold_comparison")
    fig, ax = plt.subplots(figsize=(8, 5))
    for index, subset in enumerate(by_fraction):
        means = [np.mean([row["raw_f1_delta"] for row in subset if row["group"] == group]) for group in (1, 2, 3)]
        ax.bar(np.arange(3) + (index - 1) * .25, means, width=.25, label=labels[index + 1])
    ax.set_xticks(range(3), ["Group 1", "Group 2", "Group 3"])
    ax.axhline(0, color="black", lw=.8)
    ax.set_ylabel("Mean Raw F1 recovery")
    ax.legend()
    save(fig, "group_raw_f1_recovery")


def write_outputs(folder, rows, config):
    recovery, statistics, groups, counts = build_tables(rows, config["fractions"])
    write_csv(folder / "calibration_results.csv", rows)
    write_csv(folder / "calibration_recovery.csv", recovery)
    write_csv(folder / "calibration_statistics.csv", statistics)
    write_csv(folder / "calibration_groups.csv", groups)
    write_json(folder / "analysis.json", dict(counts=counts, protocol=config,
        raw_f1_recovery="calibrated - zero_shot", absolute_shift="abs(calibrated - zero_shot)",
        sample_std_ddof=1, iqr="Q75-Q25", zero_shot_unique_targets=28,
        ranking_improvement_claimed=False, all_fractions_reported=True))
    figures(rows, config["fractions"], folder / "figures")
    report = ["# Target threshold calibration 結果", "",
              "這是使用 target TRAIN prefix 的 lightweight target adaptation，不是 zero-shot；model 未重新訓練。",
              "所有 fractions 事先固定，沒有依 TEST metrics 替各 target 挑最佳 fraction。", "",
              "| Setting | Mean Raw Precision | Median Raw Precision | Mean Raw Recall | Median Raw Recall | Mean Raw F1 | Median Raw F1 | Sample std Raw F1 | IQR Raw F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in statistics:
        fields = [row[key] for key in ("mean_raw_precision", "median_raw_precision", "mean_raw_recall",
                  "median_raw_recall", "mean_raw_f1", "median_raw_f1", "sample_std_raw_f1", "iqr_raw_f1")]
        report.append("| " + row["setting"] + " | " + " | ".join(f"{value:.6f}" for value in fields) + " |")
    report += ["", "## Raw F1 recovery", "", "正值為 calibrated Raw F1 高於 zero-shot。"]
    for fraction in config["fractions"]:
        values = counts[fraction]
        report.append(f"\n- {float(fraction):.0%}：mean recovery={values['mean_raw_f1_recovery']:.6f}；"
                      f"提升 {values['f1']['improved']} 台、下降 {values['f1']['degraded']} 台、不變 {values['f1']['unchanged']} 台。")
    report += ["", "## 判讀與限制", "",
        "PR-AUC / AP / ROC-AUC 不變是 TEST continuous scores 重用的必要條件，不能宣稱 ranking 改善。",
        "若 Raw metrics 提升，只支持 threshold calibration 可在未重新訓練下恢復部分 operating-point 表現；"
        "若結果 mixed，應分 target/group 描述；若幾乎無改善，只能說 threshold-only calibration 在此 protocol 下不足。"
        "沒有證據時不能把未改善的唯一原因斷定為 representation shift。",
        "PA 僅作 legacy comparison，不作主圖或主要結論。",
        "限制：single seed、TranAD only、TRAIN 視為 normal 的假設、temporal prefixes 有巢狀依賴、"
        "POT 小樣本 tail estimation、無獨立 validation split、無 target normalization/fine-tuning。",
        "完整 84-row results、recovery、statistics、group tables 與五類圖表位於同一獨立 summary directory。"]
    (folder / "CALIBRATION_RESULTS.md").write_text("\n".join(report) + "\n")
