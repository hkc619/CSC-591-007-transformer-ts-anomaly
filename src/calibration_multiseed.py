"""重用逐 run 稽核；先每 seed 對 28 targets 做 macro，再計算跨 seed sample std。"""
from pathlib import Path
import numpy as np

from src.calibration_analysis import build_tables, summarize_run, write_csv
from src.calibration_artifacts import failed_calibration_tasks
from src.calibration_protocol import (FRACTIONS, calibration_directory, canonical_fraction, canonical_seeds,
                                      manifest_rows, validate_manifest)
from src.seed_replication import safe_path, unique_name
from src.target_calibration import file_hash, write_json


def build_multiseed_tables(rows, machines, seeds=(1, 2, 3)):
    seeds = canonical_seeds(seeds)
    if len(seeds) < 2:
        raise ValueError("跨 seed sample std 至少需要兩個 seeds")
    expected = {(m, f, s) for m in machines for f in FRACTIONS for s in seeds}
    actual = [(r["target"], canonical_fraction(r["fraction"]), int(r["seed"])) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("缺少、重複或額外的 target/fraction/seed，不能產生完整統計")
    macro, recovery, groups = [], [], []
    for seed in seeds:
        selected = [r for r in rows if r["seed"] == seed]
        _, statistics, seed_groups, counts = build_tables(selected, list(FRACTIONS))
        for target in machines:
            baseline = [r for r in selected if r["target"] == target]
            for metric in ("precision", "recall", "f1"):
                values = [r["zero_shot_raw_"+metric] for r in baseline]
                np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-12)
        for row in statistics:
            macro.append(dict(seed=seed, setting=row["setting"], targets=28,
                raw_precision=row["mean_raw_precision"], raw_recall=row["mean_raw_recall"], raw_f1=row["mean_raw_f1"]))
        groups.extend(dict(seed=seed, **group) for group in seed_groups)
        for fraction in FRACTIONS:
            subset = [r for r in selected if canonical_fraction(r["fraction"]) == fraction]
            item = dict(seed=seed, fraction=fraction, targets=28, **counts[fraction]["f1"])
            for metric in ("precision", "recall", "f1"):
                for row in subset:
                    delta = row["calibrated_raw_"+metric]-row["zero_shot_raw_"+metric]
                    np.testing.assert_allclose(row["raw_"+metric+"_delta"], delta, rtol=0, atol=1e-12)
                item["raw_"+metric+"_recovery"] = float(np.mean([r["raw_"+metric+"_delta"] for r in subset]))
            recovery.append(item)
    aggregate, recovery_aggregate = [], []
    for setting in ("zero_shot", *FRACTIONS):
        for metric in ("precision", "recall", "f1"):
            values = [next(r["raw_"+metric] for r in macro if r["seed"] == seed and r["setting"] == setting) for seed in seeds]
            aggregate.append(dict(setting=setting, metric="raw_"+metric, **{f"seed_{s}": v for s,v in zip(seeds, values)},
                mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), seed_count=len(seeds), targets_per_seed=28))
    for fraction in FRACTIONS:
        for metric in ("precision", "recall", "f1"):
            values = [next(r["raw_"+metric+"_recovery"] for r in recovery if r["seed"] == seed and r["fraction"] == fraction) for seed in seeds]
            recovery_aggregate.append(dict(fraction=fraction, metric="raw_"+metric+"_recovery",
                **{f"seed_{s}": v for s,v in zip(seeds, values)}, mean=float(np.mean(values)),
                sample_std=float(np.std(values, ddof=1)), seed_count=len(seeds)))
    target_rows = []
    for target in machines:
        for fraction in FRACTIONS:
            values = [next(r["raw_f1_delta"] for r in rows if r["target"] == target
                           and r["seed"] == seed and canonical_fraction(r["fraction"]) == fraction) for seed in seeds]
            positive, negative, zero = sum(v > 0 for v in values), sum(v < 0 for v in values), sum(v == 0 for v in values)
            category = ("improved_all_seeds" if positive == len(seeds) else
                        "degraded_all_seeds" if negative == len(seeds) else
                        "unchanged_all_seeds" if zero == len(seeds) else
                        "mixed_direction" if positive and negative else
                        "improved_or_unchanged" if positive else "degraded_or_unchanged")
            target_rows.append(dict(target=target, group=int(target.split("-")[1]), fraction=fraction,
                **{f"seed_{s}_recovery": v for s,v in zip(seeds, values)}, mean_recovery=float(np.mean(values)),
                sample_std_recovery=float(np.std(values, ddof=1)), improved_seeds=positive,
                degraded_seeds=negative, unchanged_seeds=zero, category=category, seed_count=len(seeds)))
    return dict(seed_macro_results=macro, seed_macro_statistics=aggregate, seed_recovery=recovery,
                seed_recovery_statistics=recovery_aggregate, target_recovery_robustness=target_rows, groups_by_seed=groups)


def collect(roots, formal_root, config, machines, seeds):
    seeds = canonical_seeds(seeds)
    rows, evidence, identities, prefixes, targets = [], {}, {}, {}, {}
    new_rows = validate_manifest(roots.project / "configs/calibration_runs_seed23.csv", machines, (2, 3))
    for seed in seeds:
        namespace = calibration_directory(roots.results, formal_root, seed)
        expected = manifest_rows(machines) if seed == 1 else [r for r in new_rows if int(r["seed"]) == seed]
        failures, records = failed_calibration_tasks(namespace, expected)
        if failures["count"]:
            raise ValueError(f"seed={seed} calibration 尚未完整：{failures}")
        for row in expected:
            folder = Path(records[int(row["task_id"])]["run_path"])
            result, meta = summarize_run(folder, row, config, namespace, machines)
            key = (row["target"], seed)
            identity = (meta["formal_run_id"], meta["formal_checkpoint_hash"], meta["formal_score_artifact_hash"],
                        meta["test_scores_sha256"], meta["labels_sha256"], meta["protected_input_sha256"]["evaluation.json"])
            if key in identities and identities[key] != identity:
                raise ValueError("同 target/seed 的三個 fractions 不得混用 formal model / TEST artifacts")
            identities[key] = identity
            prefix_key = (row["target"], row["fraction"])
            prefix_identity = (meta["prefix_array_sha256"], meta["target_train_total_raw_points"], meta["calibration_raw_points"])
            if prefix_key in prefixes and prefixes[prefix_key] != prefix_identity:
                raise ValueError("同 target/fraction 在不同 seeds 使用了不同 TRAIN prefix")
            prefixes[prefix_key] = prefix_identity
            label_identity = (meta["labels_sha256"], meta["test_points"])
            if row["target"] in targets and targets[row["target"]] != label_identity:
                raise ValueError("同 target 在不同 seeds 的 TEST labels/count 不一致")
            targets[row["target"]] = label_identity
            for name in ("status.json", "metadata.json", "evaluation.json", "validation.json", "threshold.json"):
                path = folder / name
                evidence[str(path)] = file_hash(path)
            rows.append(result)
    return rows, evidence


def analyze_multiseed(roots, formal_root, config, machines, seeds=(1, 2, 3)):
    seeds = canonical_seeds(seeds)
    rows, evidence = collect(roots, formal_root, config, machines, seeds)
    tables = build_multiseed_tables(rows, machines, seeds)
    parent = safe_path(calibration_directory(roots.results, formal_root, 2) / "summaries")
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / unique_name("analysis")
    folder.mkdir(exist_ok=False)
    write_json(folder / "status.json", dict(completed=False, state="running"))
    try:
        write_csv(folder / "calibration_results.csv", rows)
        for name, values in tables.items():
            write_csv(folder / (name+".csv"), values)
        figures(tables, folder / "figures", machines, seeds)
        for path, digest in evidence.items():
            if file_hash(path) != digest:
                raise ValueError("分析期間原 artifacts 改變，不標記成功")
        write_json(folder / "analysis.json", dict(seeds=list(seeds), calibration_runs=len(rows),
            targets_per_seed=28, fractions=list(FRACTIONS), sample_std_ddof=1,
            aggregation="macro across 28 targets within each seed; sample std across seeds",
            input_sha256=evidence, calibration_rng_seed=1, ranking_improvement_claimed=False,
            all_fractions_reported=True, statistical_significance_claimed=False))
        report = ["# 多 Seed Target-Threshold Calibration 結果", "",
                  "此設定為 lightweight target adaptation，不是 zero-shot。模型 frozen，TEST scores / labels 重用，只有 threshold 改變。", ""]
        for metric in ("raw_f1", "raw_precision", "raw_recall"):
            report += ["## "+metric, "", "| Setting | " + " | ".join(f"Seed {s}" for s in seeds) + " | Mean | Sample Std |",
                       "|---|" + "---:|" * (len(seeds)+2)]
            for row in tables["seed_macro_statistics"]:
                if row["metric"] == metric:
                    values = [row[f"seed_{s}"] for s in seeds] + [row["mean"], row["sample_std"]]
                    report.append("| "+row["setting"]+" | "+" | ".join(f"{v:.6f}" for v in values)+" |")
            report.append("")
        report += ["## Raw F1 Recovery", "", "| Fraction | Mean Recovery | Sample Std |", "|---|---:|---:|"]
        for row in tables["seed_recovery_statistics"]:
            if row["metric"] == "raw_f1_recovery":
                report.append(f"| {row['fraction']} | {row['mean']:.6f} | {row['sample_std']:.6f} |")
        report += ["", "先各 seed 對 28 targets 做 macro，再計算跨 seed sample std（ddof=1）；不是把 84 個 machine-seed pairs 當成 84 個 seeds。",
            "", "每台 target/fraction 的 recovery mean/std、改善/下降/不變 seed 數與方向分類見 target_recovery_robustness.csv。",
            "", "此為描述性分析，不宣稱統計顯著、5% 普遍最佳、每台都改善或 threshold mismatch 是唯一原因。所有 5%/10%/20% 都呈現，不能依 TEST 結果挑最佳 fraction。",
            "", "PR-AUC / AP / ROC-AUC 不變是必要 invariant，不是 ranking 改善；calibration/POT RNG 維持原本固定值 1。"]
        with (folder / "CALIBRATION_MULTI_SEED_RESULTS.md").open("x") as stream:
            stream.write("\n".join(report)+"\n")
        write_json(folder / "completed.json", dict(completed=True, state="completed"))
        (folder / "completed.json").replace(folder / "status.json")
    except BaseException as error:
        write_json(folder / "failure.json", dict(error=repr(error), completed=False))
        raise
    return folder


def figures(tables, out, machines, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=False)
    def save(fig, name):
        try:
            fig.tight_layout()
            for extension in ("png", "pdf"):
                with (out / (name+"."+extension)).open("xb") as stream:
                    fig.savefig(stream, format=extension, dpi=160)
        finally:
            plt.close(fig)
    settings, labels = ("zero_shot", *FRACTIONS), ["Zero-shot", "5%", "10%", "20%"]
    selected = [r for r in tables["seed_macro_statistics"] if r["metric"] == "raw_f1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(range(4), [r["mean"] for r in selected], yerr=[r["sample_std"] for r in selected], fmt="o-", capsize=4)
    ax.set(xticks=range(4), xticklabels=labels, ylabel="Macro Raw F1 (mean +/- sample std across seeds)")
    save(fig, "raw_f1_by_setting")
    fig, ax = plt.subplots(figsize=(14, 5))
    for index, fraction in enumerate(FRACTIONS):
        values = [next(r for r in tables["target_recovery_robustness"] if r["target"] == m and r["fraction"] == fraction) for m in machines]
        ax.errorbar(np.arange(28)+(index-1)*.18, [r["mean_recovery"] for r in values],
            yerr=[r["sample_std_recovery"] for r in values], fmt="o", capsize=2, label=labels[index+1])
    ax.axhline(0, color="black", lw=.8)
    ax.set_xticks(range(28), [m.replace("machine-", "") for m in machines], rotation=65)
    ax.set_ylabel("Raw F1 recovery, mean +/- sample std across seeds")
    ax.legend()
    save(fig, "raw_f1_recovery_by_target")
    fig, ax = plt.subplots(figsize=(7, 4))
    for seed in seeds:
        values = [next(r["raw_f1"] for r in tables["seed_macro_results"] if r["seed"] == seed and r["setting"] == s) for s in settings]
        ax.plot(range(4), values, "o-", label=f"Seed {seed}")
    ax.set(xticks=range(4), xticklabels=labels, ylabel="Macro Raw F1 across 28 targets")
    ax.legend()
    save(fig, "per_seed_calibration_summary")
