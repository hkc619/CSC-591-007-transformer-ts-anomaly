"""完整 12-run ranking-only summary，唯讀同六台 TranAD 結果；不修改任何既有結果。"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import resolve_roots
from src.continuous_metrics import continuous_metrics
from src.anomaly_transformer.protocol import configuration, namespace, validate_manifest
from src.anomaly_transformer.artifacts import discover
from src.anomaly_transformer.experiment import file_hash, write_json
from src.anomaly_transformer.dataset import select_sources


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(roots, tranad_path):
    config, canonical = configuration(roots.project)
    manifest = validate_manifest(roots.project / "configs/anomaly_transformer_runs.csv", config)
    out = namespace(roots.results)
    failures, statuses = discover(out, manifest)
    if failures["count"]:
        raise ValueError(f"12 tasks 尚未全部成功，不產生完整 robustness summary：{failures}")
    rows, devices = [], set()
    for row in manifest:
        folder = Path(statuses[int(row["task_id"])]["run_path"])
        metadata = json.loads((folder / "metadata.json").read_text())
        evaluation = json.loads((folder / "evaluation.json").read_text())
        history = json.loads((folder / "training_history.json").read_text())
        fixed = dict(detector="AnomalyTransformer", protocol=config["protocol"], target=row["target"],
            setting=row["setting"], seed=1, dtype="float32", window=100, batch_size=256,
            epochs=10, lr=1e-4, eval_batch_size=256, train_stride=100, training_shuffle=True,
            normalization_source=config["normalization"], normalization_fitted=False,
            temperature=50, k=3, learning_rate_schedule=config["lr_schedule"],
            target_train_used_for_training=row["setting"] == "per_machine",
            target_test_used_for_training=False, target_test_used_for_validation=False,
            target_train_used_for_normalization=False, target_test_used_for_normalization=False,
            threshold_evaluation=False, point_adjustment=False, frozen_model_unchanged=True,
            all_windows_single_machine=True, score_label_alignment=True, target_test_loaded_after_training=True)
        if any(metadata.get(key) != value for key, value in fixed.items()) or metadata["model_config"] != config["model"]:
            raise ValueError("AT run 不符合固定 protocol / isolation")
        if metadata["optimizer_config"] != config["optimizer"] or metadata["upstream_commit"] != config["upstream_commit"]:
            raise ValueError("optimizer / upstream provenance 不符")
        if metadata["sources"] != select_sources(canonical, row["target"], row["setting"]):
            raise ValueError("source list 不正確")
        if row["setting"] == "cross_machine" and (metadata["target_train_loaded"] or metadata["target_train_used_for_training"]):
            raise ValueError("cross-machine target TRAIN leakage")
        if len(history) != 10 or [h["epoch"] for h in history] != list(range(1, 11)):
            raise ValueError("未完成固定 10 epochs")
        for h in history:
            if not all(np.isfinite(h[key]) for key in ("loss1", "loss2", "reconstruction", "series_loss", "prior_loss")):
                raise ValueError("history NaN/Inf")
        if metadata["checkpoint_sha256"] != file_hash(metadata["checkpoint"]):
            raise ValueError("checkpoint hash 不一致")
        with np.load(folder / "scores.npz", allow_pickle=False) as archive:
            scores, labels = archive["scores"], archive["labels"]
            if len(scores) != metadata["test_score_count"] or len(labels) != metadata["label_count"] or not np.all(archive["coverage"] == 1):
                raise ValueError("timestamp alignment 不一致")
            computed = continuous_metrics(scores, labels)
        if set(evaluation) != set(computed):
            raise ValueError("AT evaluation 必須只有三個 continuous metrics")
        for key in computed:
            np.testing.assert_allclose(computed[key], evaluation[key], rtol=0, atol=1e-12)
        devices.add(metadata["device"])
        rows.append(dict(target=row["target"], setting=row["setting"], seed=1, **evaluation,
            train_windows=metadata["train_windows"], test_points=len(labels),
            training_seconds=metadata["training_seconds"], inference_seconds=metadata["inference_seconds"],
            runtime_seconds=metadata["runtime_seconds"], peak_gpu_allocated_bytes=metadata["peak_gpu_allocated_bytes"],
            checkpoint=metadata["checkpoint"], run_path=str(folder)))
    if len(devices) != 1:
        raise ValueError("12 runs 的 device 不一致；先審查比較設定")
    with tranad_path.open(newline="") as stream:
        tranad = list(csv.DictReader(stream))
    gaps, comparisons = [], []
    for target in config["targets"]:
        per = next(r for r in rows if r["target"] == target and r["setting"] == "per_machine")
        cross = next(r for r in rows if r["target"] == target and r["setting"] == "cross_machine")
        gap = dict(target=target, group=int(target.split("-")[1]))
        for metric in ("pr_auc_trapezoidal", "average_precision", "roc_auc"):
            gap["per_machine_" + metric] = per[metric]
            gap["cross_machine_" + metric] = cross[metric]
            gap[metric + "_gap"] = per[metric] - cross[metric]
        gaps.append(gap)
        selected = {}
        for setting in ("per_machine", "cross_machine_zero_shot"):
            candidates = [r for r in tranad if r["target_machine"] == target and r["setting"] == setting and r["seed"] == "1"]
            if len(candidates) != 1:
                raise ValueError("TranAD summary 缺少或重複同 target / setting / seed")
            candidate = candidates[0]
            expected = [target] if setting == "per_machine" else [m for m in canonical if m != target]
            if candidate["source_machines"].split(";") != expected:
                raise ValueError("TranAD sources 與 27-source comparison 不符")
            selected[setting] = float(candidate["pr_auc"])
        comparisons.append(dict(target=target, tranad_per=selected["per_machine"],
            tranad_cross=selected["cross_machine_zero_shot"],
            tranad_gap=selected["per_machine"]-selected["cross_machine_zero_shot"],
            at_per=per["pr_auc_trapezoidal"], at_cross=cross["pr_auc_trapezoidal"],
            at_gap=gap["pr_auc_trapezoidal_gap"]))
    def aggregate(values):
        return dict(mean_gap=float(np.mean(values)), median_gap=float(np.median(values)),
            degraded=int(np.sum(np.array(values) > 0)), improved=int(np.sum(np.array(values) < 0)),
            unchanged=int(np.sum(np.array(values) == 0)))
    summary = dict(targets=config["targets"], runs=12, primary="pr_auc_trapezoidal",
        mean_per_machine_pr_auc=float(np.mean([r["at_per"] for r in comparisons])),
        mean_cross_machine_pr_auc=float(np.mean([r["at_cross"] for r in comparisons])),
        anomaly_transformer=aggregate([r["at_gap"] for r in comparisons]),
        tranad_same_six_targets=aggregate([r["tranad_gap"] for r in comparisons]),
        tranad_summary_path=str(tranad_path), tranad_summary_sha256=file_hash(tranad_path),
        interpretation="cross-machine degradation robustness; no absolute detector ranking claim")
    groups = [dict(group=group, targets=2, descriptive_only=True,
                   **aggregate([r["pr_auc_trapezoidal_gap"] for r in gaps if r["group"] == group])) for group in (1, 2, 3)]
    destination = out / "summaries" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "anomaly_transformer_results.csv", rows)
    write_csv(destination / "anomaly_transformer_gaps.csv", gaps)
    write_csv(destination / "tranad_vs_anomaly_transformer_gaps.csv", comparisons)
    write_csv(destination / "group_summary.csv", groups)
    write_json(destination / "aggregate.json", summary)
    lines = ["# Anomaly Transformer secondary robustness experiment", "",
        "固定六個 targets、每組兩台；group analysis 為 descriptive only, very small subset。", "",
        "| Target | TranAD per | TranAD cross | TranAD gap | AT per | AT cross | AT gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append("| " + row["target"] + " | " + " | ".join(f"{row[k]:.6f}" for k in
            ("tranad_per", "tranad_cross", "tranad_gap", "at_per", "at_cross", "at_gap")) + " |")
    lines += ["", f"AT PR-AUC 在 {summary['anomaly_transformer']['degraded']}/6 targets 下降；"
        f"同六台 TranAD 下降台數為 {summary['tranad_same_six_targets']['degraded']}/6。",
        "這只描述是否觀察到類似 cross-machine degradation；不能宣稱 AT 比 TranAD 更好或更差。",
        "限制：single seed、固定小 subset、不同 native architecture/window/objective、identity preprocessing、"
        "固定 epochs 而非相同 optimizer updates；不是原論文完整 reproduction。"]
    (destination / "ROBUSTNESS_RESULTS.md").write_text("\n".join(lines) + "\n")
    print("AT summary：", destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--tranad-results", type=Path)
    parser.add_argument("--hpc", action="store_true")
    args = parser.parse_args()
    roots = resolve_roots(args)
    summarize(roots, args.tranad_results or roots.results / "summaries/formal_28_machine_results.csv")


if __name__ == "__main__":
    main()
