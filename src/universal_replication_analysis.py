"""唯讀三 seed artifacts；先對 28 targets macro，再對三個 seed 計算 ddof=1。"""
import csv
from pathlib import Path

import numpy as np

from src.seed_replication import file_hash, read_json, safe_path, unique_name, write_json
from src.universal_training import namespace
from src.universal_replication_protocol import digest, valid_seed, validate_target
from src.universal_evaluation_analysis import write_csv
from src.universal_threshold_artifacts import CONDITIONS, resolve_inputs, check_inputs, read_train_scores
from src.universal_threshold_analysis import validate_rows, aggregate, comparisons, target_row


def read_table(path):
    with Path(path).open(newline="") as stream:return list(csv.DictReader(stream))


def completed_threshold(roots,inputs,seed,pin=None):
    parent=safe_path(namespace(roots.results)/"thresholding"/f"seed{seed}"/inputs["checkpoint_sha256"])
    if pin:
        candidate=safe_path(Path(pin).expanduser().absolute())
        candidates=[candidate]
    else:
        candidates=[p.parent for p in parent.glob("*/full/attempt-*/status.json") if read_json(p).get("completed") is True]
    if len(candidates)!=1:
        raise ValueError(f"seed={seed} 需要唯一 completed Stage-3 full attempt，可明確 pin；目前 {len(candidates)} 個")
    folder=candidates[0]
    if not folder.is_relative_to(parent) or folder.parent.name!="full" or folder.parent.parent.parent!=parent:
        raise ValueError("Stage-3 path 跨 seed／checkpoint")
    status,launch,proof=[read_json(folder/n) for n in ("status.json","launch.json","validation.json")]
    if seed==1:
        from src.universal_threshold_artifacts import code_hashes
        approved=read_json(roots.project/"configs/universal_seed1_compatibility.json")
        if digest(approved["source_file_sha256"])!=approved["stage3_code_sha256"]:
            raise ValueError("Seed-1 compatibility manifest 損壞")
        if folder.parent.parent.name not in {approved["stage3_code_sha256"],digest(code_hashes(roots.project))}:
            raise ValueError("未知 seed-1 Stage-3 code fingerprint；不猜測 compatibility")
    if (status.get("state")!="completed" or status.get("completed") is not True or status.get("seed")!=seed
            or launch.get("seed")!=seed or launch.get("phase")!="full" or launch.get("inputs")!=inputs
            or digest(launch["source_file_sha256"])!=folder.parent.parent.name
            or proof.get("checkpoint_sha256")!=inputs["checkpoint_sha256"] or proof.get("seed")!=seed
            or proof.get("parameter_hash_before")!=inputs["model_state_sha256"]
            or proof.get("parameter_hash_after")!=inputs["model_state_sha256"]
            or proof.get("parameters_unchanged") is not True or proof.get("stage2_inputs_unchanged") is not True
            or proof.get("optimizer_steps")!=0 or proof.get("model_retrained") is not False
            or proof.get("test_inference_executed") is not False or proof.get("target_condition_rows")!=140
            or proof.get("threshold_conditions")!=list(CONDITIONS)):
        raise ValueError("Stage-3 completion／seed／input hashes／freeze 不符")
    return folder


def collect_seed(roots,seed,threshold_run=None):
    seed=valid_seed(seed)
    inputs=resolve_inputs(roots,seed=seed)
    folder=completed_threshold(roots,inputs,seed,threshold_run)
    evidence=dict(inputs["protected_input_sha256"])
    for p in folder.rglob("*"):
        if p.is_file():evidence[str(safe_path(p))]=file_hash(p)
    rows=read_table(folder/"summary/target_condition_metrics.csv")
    for row in rows:
        for k in ("seed","group","tp","tn","fp","fn"):row[k]=int(row[k])
        for k in ("threshold","raw_precision","raw_recall","raw_f1","pr_auc","average_precision","roc_auc"):row[k]=float(row[k])
    validate_rows(rows,inputs["targets"],inputs["checkpoint_sha256"])
    if any(r["seed"]!=seed for r in rows):raise ValueError("Stage-3 CSV 跨 seed")
    global_info=read_json(folder/"global_threshold.json")
    pool=[]
    train_records=read_json(Path(inputs["stage1"]["run"])/"train_manifest.json")["machines"]
    saved_manifest=read_table(folder/"summary/train_score_manifest.csv")
    if [r["machine"] for r in saved_manifest]!=inputs["targets"]:raise ValueError("TRAIN cache manifest 不是 28 canonical machines")
    for cached,record in zip(saved_manifest,train_records):
        cache=safe_path(Path(cached["scores_path"]).parent)
        scores,_=read_train_scores(cache,record,inputs)
        if cached["checkpoint_sha256"]!=inputs["checkpoint_sha256"] or int(cached["seed"])!=seed or file_hash(cache/"scores.npz")!=cached["scores_npz_sha256"]:
            raise ValueError("TRAIN cache／manifest 跨 seed 或 hash 不符")
        pool.append(scores)
        for p in (cache/n for n in ("metadata.json","validation.json","status.json","scores.npz")):
            evidence[str(p)]=file_hash(p)
    pooled=np.concatenate(pool)
    pot_fixed=dict(q=1e-5,requested_level=.99995,multiplier=1.06,stream_score_count=0,pot_rng_seed=1)
    if (global_info.get("machine_count")!=28 or global_info.get("score_count")!=len(pooled)
            or global_info.get("natural_weighting") is not True or global_info.get("checkpoint_sha256")!=inputs["checkpoint_sha256"]
            or any(global_info.get(k)!=v for k,v in pot_fixed.items())):
        raise ValueError("Global POT provenance 不是 fixed natural 28-TRAIN pool")
    from src.target_calibration import array_hash
    if array_hash(pooled)!=global_info["provenance"]["score_array_sha256"]:raise ValueError("Global pool hash 不符")
    for row in rows:
        if row["condition"]=="global":threshold=global_info["threshold"]
        else:
            info=read_json(folder/"targets"/row["target"]/(row["condition"]+"_threshold.json"))
            if any(info.get(k)!=v for k,v in pot_fixed.items()):raise ValueError("Local POT protocol 改變")
            n=train_records[inputs["targets"].index(row["target"])]["raw_train_length"]
            from src.calibration_protocol import prefix_length
            expected=n if row["condition"]=="local_full" else prefix_length(n,dict(local_05=".05",local_10=".10",local_20=".20")[row["condition"]])
            if info["source_score_count"]!=expected or info["provenance"]["raw_prefix_stop_exclusive"]!=expected:
                raise ValueError("Local prefix budget 改變")
            threshold=info["threshold"]
        if row["threshold"]!=threshold or not np.isfinite(threshold):raise ValueError("Threshold 改變／非有限值")
        actual,_=target_row(inputs,row["target"],row["condition"],threshold)
        for k,v in actual.items():
            if isinstance(v,float):
                if not np.isclose(row[k],v,atol=1e-12,rtol=0):raise ValueError("Stage-3 metrics 與 TEST artifacts 不符")
            elif row[k]!=v:raise ValueError("Stage-3 row provenance 不符")
    ranking=[validate_target(Path(inputs["stage2_run"])/"targets"/t,t,inputs["stage1"]) for t in inputs["targets"]]
    _,_,deployments,reference_evidence=comparisons(roots,inputs,rows,roots.results/"summaries/formal_28_machine_results.csv")
    evidence.update(reference_evidence)
    # Matching-seed per-target PR-AUC，seed 1 由既有 summary + metadata 安全解析。
    if seed==1:
        from src.universal_evaluation_analysis import formal_comparison
        paired,_,ref=formal_comparison(roots,ranking,roots.results/"summaries/formal_28_machine_results.csv",
            {t:Path(inputs["stage2_run"])/"targets"/t for t in inputs["targets"]})
        evidence.update(ref)
    else:
        from src.universal_replication_protocol import formal_references
        references,ref=formal_references(roots,seed,inputs["stage2_run"]);evidence.update(ref)
        lookup={(r["target_machine"],r["setting"]):r for r in references}
        paired=[dict(target=t,per_machine_pr_auc=lookup[t,"per_machine"]["pr_auc"],
            lomo_pr_auc=lookup[t,"cross_machine_zero_shot"]["pr_auc"],universal_pr_auc=ranking[i]["pr_auc"]) for i,t in enumerate(inputs["targets"])]
    check_inputs(inputs)
    for p,value in evidence.items():
        if file_hash(p)!=value:raise ValueError("Aggregation input 改變")
    return dict(seed=seed,inputs=inputs,folder=str(folder),rows=rows,ranking=paired,deployments=deployments,evidence=evidence)


def tables(records):
    if [r["seed"] for r in records]!=[1,2,3]:raise ValueError("Aggregation 必須是 seeds=1,2,3，不能把 targets 當 seeds")
    canonical=[r["target"] for r in records[0]["ranking"]]
    if len(canonical)!=28 or len(set(canonical))!=28 or any([r["target"] for r in record["ranking"]]!=canonical for record in records):
        raise ValueError("每 seed 必須有相同 28 unique targets")
    ranking,thresholds,gaps,directions=[],[],[],[]
    for record in records:
        seed=record["seed"]
        for setting,key in (("per_machine","per_machine_pr_auc"),("lomo","lomo_pr_auc"),("universal","universal_pr_auc")):
            ranking.append(dict(seed=seed,setting=setting,metric="pr_auc",macro_mean=float(np.mean([r[key] for r in record["ranking"]]))))
        macros,_,_=aggregate(record["rows"],[r["target"] for r in record["ranking"]])
        for macro in macros:
            for metric in ("precision","recall","f1"):
                thresholds.append(dict(seed=seed,condition=macro["condition"],metric="raw_"+metric,macro_mean=macro["mean_raw_"+metric]))
        reference=next(r for r in record["deployments"] if r["deployment"]=="per_machine")
        for metric in ("precision","recall","f1"):
            thresholds.append(dict(seed=seed,condition="per_machine",metric="raw_"+metric,macro_mean=reference["mean_raw_"+metric]))
        rank={r["setting"]:r["macro_mean"] for r in ranking if r["seed"]==seed}
        f1={r["condition"]:r["macro_mean"] for r in thresholds if r["seed"]==seed and r["metric"]=="raw_f1"}
        values=dict(universal_minus_lomo_pr_auc=rank["universal"]-rank["lomo"],per_minus_universal_pr_auc=rank["per_machine"]-rank["universal"],
            local05_minus_global_f1=f1["local_05"]-f1["global"],local20_minus_global_f1=f1["local_20"]-f1["global"],
            per_minus_local05_f1=f1["per_machine"]-f1["local_05"],per_minus_local20_f1=f1["per_machine"]-f1["local_20"])
        gaps.extend(dict(seed=seed,comparison=k,gap=v) for k,v in values.items())
    def summarize(table,key,value):
        result=[]
        for identity in sorted({tuple(r[k] for k in key) for r in table}):
            subset=sorted([r for r in table if tuple(r[k] for k in key)==identity],key=lambda r:r["seed"])
            if [r["seed"] for r in subset]!=[1,2,3]:raise ValueError("每個 summary cell 必須恰有三個 seed-level means")
            values=[r[value] for r in subset]
            result.append(dict(zip(key,identity),seed1=values[0],seed2=values[1],seed3=values[2],
                mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)),seed_count=3))
        return result
    targets=[r["target"] for r in records[0]["ranking"]]
    for target in targets:
        for comparison in ("universal_minus_lomo_pr_auc",*CONDITIONS[1:]):
            values=[]
            for record in records:
                if comparison=="universal_minus_lomo_pr_auc":
                    r=next(r for r in record["ranking"] if r["target"]==target);value=r["universal_pr_auc"]-r["lomo_pr_auc"]
                else:
                    subset={r["condition"]:r for r in record["rows"] if r["target"]==target}
                    value=subset[comparison]["raw_f1"]-subset["global"]["raw_f1"]
                values.append(value)
            category="all_improved" if all(v>0 for v in values) else "all_degraded" if all(v<0 for v in values) else "all_unchanged" if all(v==0 for v in values) else "mixed_or_ties"
            directions.append(dict(target=target,comparison=comparison,seed1=values[0],seed2=values[1],seed3=values[2],
                improved_seeds=sum(v>0 for v in values),degraded_seeds=sum(v<0 for v in values),unchanged_seeds=sum(v==0 for v in values),
                direction_changed=any(v>0 for v in values) and any(v<0 for v in values),category=category))
    return dict(ranking_by_seed=ranking,threshold_by_seed=thresholds,seed_level_gaps=gaps,machine_direction_consistency=directions,
        multiseed_ranking_summary=summarize(ranking,("setting","metric"),"macro_mean"),
        multiseed_threshold_summary=summarize(thresholds,("condition","metric"),"macro_mean"),
        multiseed_gap_summary=summarize(gaps,("comparison",),"gap"))


def analyze(roots,seed1_threshold_run=None):
    records=[collect_seed(roots,s,seed1_threshold_run if s==1 else None) for s in (1,2,3)]
    # 資料與 execution environment 相同；只容許 training seed 不同。
    profiles=[];data=[]
    for r in records:
        stage1=Path(r["inputs"]["stage1"]["run"])
        meta=read_json(stage1/"metadata.json")
        profiles.append({k:meta[k] for k in ("python","torch_version","numpy_version","torch_num_threads","installed_dependencies")})
        data.append([(x["machine"],x["raw_file_sha256"]) for x in read_json(stage1/"train_manifest.json")["machines"]])
    if any(p!=profiles[0] for p in profiles) or any(d!=data[0] for d in data):
        raise ValueError("Universal 三 seed 的環境或 TRAIN data 不同，不能當成只改 seed 的 replication")
    for target in records[0]["inputs"]["targets"]:
        signatures=[]
        for record in records:
            meta=read_json(Path(record["inputs"]["stage2_run"])/"targets"/target/"metadata.json")["test_input"]
            signatures.append(tuple(meta[k] for k in ("test_sha256","label_file_sha256","raw_test_length")))
        if any(value!=signatures[0] for value in signatures):raise ValueError("不同 seed 的 TEST／labels 不一致")
    output=tables(records)
    parent=safe_path(namespace(roots.results)/"multiseed");parent.mkdir(parents=True,exist_ok=True)
    folder=safe_path(parent/unique_name("analysis"));folder.mkdir()
    evidence={p:v for r in records for p,v in r["evidence"].items()}
    for name,rows in output.items():write_csv(folder/(name+".csv"),rows)
    counts={comparison:{category:sum(r["comparison"]==comparison and r["category"]==category for r in output["machine_direction_consistency"])
        for category in ("all_improved","all_degraded","all_unchanged","mixed_or_ties")}
        for comparison in {r["comparison"] for r in output["machine_direction_consistency"]}}
    for comparison in counts:
        counts[comparison]["direction_changed_targets"]=sum(r["comparison"]==comparison and r["direction_changed"]
            for r in output["machine_direction_consistency"])
    for p,value in evidence.items():
        if file_hash(p)!=value:raise ValueError("Multi-seed aggregation 改變 inputs")
    write_json(folder/"multiseed_summary.json",dict(seeds=[1,2,3],seed_count=3,targets_per_seed=28,
        aggregation="先每 seed 對 28 targets macro，再對三個 macro means 計算 mean/sample std(ddof=1)",
        machine_direction_counts=counts,input_sha256=evidence,statistical_significance_claimed=False))
    return folder
