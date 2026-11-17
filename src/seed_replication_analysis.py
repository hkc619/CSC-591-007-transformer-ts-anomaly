"""讀取 seed=1 舊結果及 seeds=2,3 新結果；所有 CSV/圖另存唯一 analysis directory。"""
import csv
from pathlib import Path

import numpy as np

from src.seed_replication import (PROTECTED, configuration, file_hash, latest_attempt,
    manifest, namespace, read_json, safe_path, unique_name, write_json)


def failed_tasks(results, rows):
    failures = []
    for row in rows:
        try:
            record = latest_attempt(results, row)
            if record is None:
                reason = "missing"
            elif not record[1].get("completed") or record[1].get("exit_code") != 0:
                reason = record[1].get("state", "failed")
            elif not record[1].get("run_path") or not (Path(record[1]["run_path"]) / "evaluation.json").is_file():
                reason = "missing evaluation"
            else:
                continue
        except (ValueError, OSError, KeyError) as error:
            reason = str(error)
        failures.append(dict(task_id=int(row["task_id"]), seed=int(row["seed"]),
                             base_task_id=int(row["base_task_id"]), reason=reason))
    return dict(failed_or_missing_task_ids=[row["task_id"] for row in failures], count=len(failures), failures=failures)


def validate_bundle(project, row, run, run_root):
    run = safe_path(run)
    if run.parent != safe_path(Path(run_root) / "runs"):
        raise ValueError("run_path 不在此 attempt / reference root 的 runs 下")
    meta = read_json(run / "metadata.json")
    evaluation = read_json(run / "evaluation.json")
    fixed = dict(seed=int(row["seed"]), setting=row["setting"], target_machine=row["target_machine"],
        protocol="v2", epochs=5, batch_size=128, eval_batch_size=128, device="cpu", dtype="float64",
        window=10, learning_rate=0.0001, training_shuffle=True, smoke_test=False,
        threshold_source="source_machine_training_scores", threshold_protocol="source_only_frozen_SPOT_extreme_quantile",
        normalization="none_added; supplied SMD values unchanged", label_source="corrected_interpretation_label_union")
    if any(meta.get(key) != value for key, value in fixed.items()):
        raise ValueError(f"run scientific metadata / seed 不符：{run}")
    import json
    if meta["source_machines"] != json.loads(row["source_machines"]):
        raise ValueError("run sources / canonical order 不符")
    pot = meta["pot_fit"]
    if any(pot.get(key) != value for key, value in dict(q=1e-5, requested_level=0.99995,
                                                     multiplier=1.06, stream_score_count=0).items()):
        raise ValueError("不是 source-only frozen POT")
    if pot["source_score_count"] != meta["train_windows"]:
        raise ValueError("POT source count 不一致")
    checks = meta["leakage_checks"]
    if any(checks.get(key) is not True for key in
           ("target_used_for_testing", "all_windows_single_machine", "target_score_label_alignment")):
        raise ValueError("window / target test alignment 證據不足")
    if checks.get("normalization_fitted") is not False:
        raise ValueError("不可新增 normalization fitting")
    if row["setting"] == "cross_machine_zero_shot":
        if any(checks.get(key) is not False for key in
               ("target_train_loaded", "target_in_optimizer_updates", "target_in_threshold_fitting")):
            raise ValueError("cross-machine target leakage")
        if any(Path(path).parent.name == "train" and Path(path).stem == row["target_machine"]
               for path in meta["data_sha256"]):
            raise ValueError("cross-machine provenance 包含 target TRAIN")
    for name in PROTECTED:
        if file_hash(run / "source" / name) != file_hash(Path(project) / name):
            raise ValueError(f"training core 與 source snapshot 不同：{name}")
    checkpoint = safe_path(Path(meta["checkpoint"]))
    if checkpoint != Path(run_root) / "checkpoints" / run.name / "model.ckpt" or not checkpoint.is_file():
        raise ValueError("checkpoint path / run 不符")
    if checkpoint.stat().st_size != meta["checkpoint_size_bytes"]:
        raise ValueError("checkpoint size 與 metadata 不符")
    result = dict(target_machine=row["target_machine"], group=int(row["target_machine"].split("-")[1]),
        setting=row["setting"], seed=int(row["seed"]), source_machines=";".join(meta["source_machines"]),
        run_path=str(run), checkpoint_path=str(checkpoint))
    for key in ("precision", "recall", "f1"):
        result["raw_" + key] = float(evaluation["raw"][key])
        result["pa_" + key] = float(evaluation["point_adjusted"][key])
    for key, name in (("pr_auc", "pr_auc_trapezoidal"), ("average_precision", "average_precision"), ("roc_auc", "roc_auc")):
        result[key] = float(evaluation["continuous_score"][name])
    if any(not np.isfinite(result[key]) or not 0 <= result[key] <= 1 for key in result
           if key.startswith(("raw_", "pa_")) or key in ("pr_auc", "average_precision", "roc_auc")):
        raise ValueError("metric 為 NaN/Inf 或超出 [0,1]")
    result["threshold"] = float(evaluation["threshold"])
    if not np.isfinite(result["threshold"]) or result["threshold"] != meta["threshold_value"]:
        raise ValueError("threshold 不一致 / 非有限值")
    for key in ("TP", "TN", "FP", "FN"):
        value = evaluation["raw"][key]
        if not isinstance(value, int) or value < 0:
            raise ValueError("confusion count 不合法")
        result[key.lower()] = value
    if sum(result[k] for k in ("tp", "tn", "fp", "fn")) != meta["test_timestamps"]:
        raise ValueError("confusion matrix / TEST count 不一致")
    result.update(train_windows=meta["train_windows"], test_points=meta["test_timestamps"],
        train_time_sec=float(meta["training_seconds"]), inference_time_sec=float(meta["test_inference_seconds"]),
        checkpoint_size_bytes=meta["checkpoint_size_bytes"])
    if len(meta["training_loss_per_epoch"]) != 5 or not np.isfinite(meta["training_loss_per_epoch"]).all():
        raise ValueError("epoch history 不完整或 loss 非有限值")
    return result, meta


def collect(roots, reference_root):
    from scripts.find_failed_runs import latest_status
    rows = manifest(roots.project)
    failures = failed_tasks(roots.results, rows)
    if failures["count"]:
        raise ValueError(f"112 個新 runs 尚未全部完成：{failures['failed_or_missing_task_ids']}")
    reference_root = Path(reference_root).expanduser().resolve()
    if reference_root == namespace(roots.results) or reference_root.is_relative_to(namespace(roots.results)):
        raise ValueError("seed=1 reference 不可指向新 seed namespace")
    statuses = latest_status(reference_root)
    result, evidence, data_by_task, environment_by_task = [], {}, {}, {}
    base_rows = [dict(row, seed="1") for row in rows[:56]]
    for row in [*base_rows, *rows]:
        if row["seed"] == "1":
            status = statuses.get(int(row["base_task_id"]))
            if not status or status.get("exit_code") != 0 or not status.get("run_path"):
                raise ValueError("原 seed=1 最近 attempt 不完整，禁止回退舊結果")
            status_path = None
            # 保存這次選到的原 status 檔名 / hash，舊 status 只讀。
            for path in (reference_root / "task_status").glob("*.json"):
                if read_json(path) == status:
                    status_path = path
            if status_path is None:
                raise ValueError("seed=1 status provenance 找不到")
            run_root = reference_root
        else:
            attempt, status = latest_attempt(roots.results, row)
            status_path, run_root = attempt / "status.json", attempt
        output, meta = validate_bundle(roots.project, row, Path(status["run_path"]), run_root)
        key = row["base_task_id"]
        environment = dict(packages=meta["packages"], python=meta["python"].split()[0],
                           torch_threads=meta["torch_threads"])
        if row["seed"] == "1":
            data_by_task[key] = meta["data_sha256"]
            environment_by_task[key] = environment
        elif meta["data_sha256"] != data_by_task[key]:
            raise ValueError("各 seed 的 input data SHA256 不同，不可合併比較")
        elif environment != environment_by_task[key]:
            raise ValueError("各 seed 的 Python / package versions / torch_threads 不同，不能解讀為只改 seed")
        result.append(output)
        for path in (status_path, Path(output["run_path"]) / "metadata.json", Path(output["run_path"]) / "evaluation.json"):
            evidence[str(path)] = file_hash(path)
    return result, evidence


def csv_file(path, rows):
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(roots, reference_root, make_figures=True):
    config, machines = configuration(roots.project)
    rows, evidence = collect(roots, reference_root)
    metrics = ("pr_auc", "average_precision", "roc_auc", "raw_precision", "raw_recall", "raw_f1", "pa_f1")
    gaps, macro, groups = [], [], []
    for seed in config["analysis_seeds"]:
        subset = [row for row in rows if row["seed"] == seed]
        for target in machines:
            per = next(r for r in subset if r["target_machine"] == target and r["setting"] == "per_machine")
            cross = next(r for r in subset if r["target_machine"] == target and r["setting"] == "cross_machine_zero_shot")
            gap = dict(target_machine=target, group=per["group"], seed=seed, floor_effect_raw_f1=per["raw_f1"] < .05)
            for metric in metrics:
                gap.update({"per_machine_" + metric: per[metric], "cross_machine_" + metric: cross[metric],
                            metric + "_gap": per[metric]-cross[metric]})
            gaps.append(gap)
        seed_gaps = [r for r in gaps if r["seed"] == seed]
        entry = dict(seed=seed, targets=28)
        for metric in metrics:
            for prefix in ("per_machine_", "cross_machine_"):
                entry["mean_" + prefix + metric] = float(np.mean([g[prefix+metric] for g in seed_gaps]))
            entry["mean_" + metric + "_gap"] = float(np.mean([g[metric+"_gap"] for g in seed_gaps]))
            entry[metric+"_degraded"] = sum(g[metric+"_gap"] > 0 for g in seed_gaps)
            entry[metric+"_improved"] = sum(g[metric+"_gap"] < 0 for g in seed_gaps)
            entry[metric+"_unchanged"] = sum(g[metric+"_gap"] == 0 for g in seed_gaps)
        macro.append(entry)
        for group in (1, 2, 3):
            selected = [g for g in seed_gaps if g["group"] == group]
            groups.append(dict(seed=seed, group=group, targets=len(selected),
                mean_pr_auc_gap=float(np.mean([g["pr_auc_gap"] for g in selected])),
                mean_raw_f1_gap=float(np.mean([g["raw_f1_gap"] for g in selected]))))
    targets = []
    for target in machines:
        selected = [g for g in gaps if g["target_machine"] == target]
        entry = dict(target_machine=target, group=selected[0]["group"], seed_count=3)
        for metric in metrics:
            for key in ("per_machine_"+metric, "cross_machine_"+metric, metric+"_gap"):
                values = [g[key] for g in selected]
                entry[key+"_mean"] = float(np.mean(values))
                entry[key+"_std"] = float(np.std(values, ddof=1))
            entry[metric+"_degraded_seed_count"] = sum(g[metric+"_gap"] > 0 for g in selected)
        targets.append(entry)
    statistics = []
    for key in macro[0]:
        if key.startswith("mean_"):
            values = [r[key] for r in macro]
            statistics.append(dict(metric=key, seed_count=3, mean=float(np.mean(values)),
                std=float(np.std(values, ddof=1)), min=float(min(values)), max=float(max(values))))
    group_statistics = []
    for group in (1, 2, 3):
        selected = [r for r in groups if r["group"] == group]
        for metric in ("pr_auc", "raw_f1"):
            values = [r["mean_"+metric+"_gap"] for r in selected]
            group_statistics.append(dict(group=group, metric=metric+"_gap", seed_count=3,
                mean=float(np.mean(values)), std=float(np.std(values, ddof=1))))
    # 完整性檢查成功後才建立全新 analysis directory，舊 summary / figures 不碰。
    parent = safe_path(namespace(roots.results) / "summaries")
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / unique_name("analysis")
    out.mkdir(exist_ok=False)
    write_json(out / "status.json", dict(completed=False, state="running"))
    try:
        csv_file(out / "seed_results.csv", rows)
        csv_file(out / "paired_gaps.csv", gaps)
        csv_file(out / "target_seed_statistics.csv", targets)
        csv_file(out / "seed_macro_results.csv", macro)
        csv_file(out / "seed_macro_statistics.csv", statistics)
        csv_file(out / "groups_by_seed.csv", groups)
        csv_file(out / "group_seed_statistics.csv", group_statistics)
        if make_figures:
            figures(targets, macro, out / "figures")
        for path, expected in evidence.items():
            if file_hash(path) != expected:
                raise RuntimeError("分析期間 input artifacts 改變，拒絕完成")
        write_json(out / "analysis.json", dict(seeds=[1, 2, 3], runs=168, paired_gaps=84,
            reference_results_root=str(Path(reference_root).resolve()), new_results_namespace=str(namespace(roots.results)),
            aggregation="machine-paired gap within each seed; macro across 28 targets within seed; sample std across 3 seeds",
            std_ddof=1, input_sha256=evidence, config=config, figures_generated=make_figures,
            calibration_included=False, reference_artifacts_modified=False))
        text = "# TranAD 多 seed 結果\n\n原 seed=1 唯讀重用；新增 seed=2,3。其他設定維持 protocol-v2。\n\n"
        text += "| Seed | Per PR-AUC mean | Cross PR-AUC mean | Gap mean | PR-AUC 下降台數 |\n|---|---:|---:|---:|---:|\n"
        for row in macro:
            text += f"| {row['seed']} | {row['mean_per_machine_pr_auc']:.6f} | {row['mean_cross_machine_pr_auc']:.6f} | {row['mean_pr_auc_gap']:.6f} | {row['pr_auc_degraded']} / 28 |\n"
        text += "\n先在同一 machine / seed 配對計算 gap，再報跨 seed mean / sample std（ddof=1）。不可把 84 個 machine-seed pairs 當成 84 個獨立 seeds。\n"
        text += "\n限制：只有三個 seeds、TranAD / SMD、自然資料量加權、per/cross optimizer updates 不等；不包含多 seed calibration。PA 僅作 legacy comparison。\n"
        with (out / "SEED_REPLICATION_RESULTS.md").open("x") as stream:
            stream.write(text)
        final = out / "completed.json"
        write_json(final, dict(completed=True, state="completed"))
        final.replace(out / "status.json")
    except BaseException as error:
        write_json(out / "failure.json", dict(completed=False, error=repr(error)))
        raise
    return out


def figures(targets, macro, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=False)
    def save(fig, name):
        try:
            fig.tight_layout()
            for extension in ("png", "pdf"):
                # 即使函式被誤呼叫第二次，也不覆寫相同檔名。
                with (out / (name + "." + extension)).open("xb") as stream:
                    fig.savefig(stream, format=extension, dpi=160)
        finally:
            plt.close(fig)
    for metric in ("pr_auc", "raw_f1"):
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.errorbar(range(28), [r[metric+"_gap_mean"] for r in targets],
            yerr=[r[metric+"_gap_std"] for r in targets], fmt="o", capsize=3)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(28), [r["target_machine"].replace("machine-", "") for r in targets], rotation=65)
        ax.set_ylabel(metric + " gap (per - cross), mean +/- sample std across 3 seeds")
        save(fig, metric+"_gap_by_target")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["seed"] for r in macro], [r["mean_per_machine_pr_auc"] for r in macro], "o-", label="per-machine")
    ax.plot([r["seed"] for r in macro], [r["mean_cross_machine_pr_auc"] for r in macro], "o-", label="cross-machine")
    ax.set(xlabel="Seed", ylabel="Macro PR-AUC across 28 targets", xticks=[1, 2, 3])
    ax.legend()
    save(fig, "pr_auc_by_seed")
