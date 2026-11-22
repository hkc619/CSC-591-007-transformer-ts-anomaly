"""Universal replication I/O orchestration；數值計算重用 frozen TranAD helpers。"""
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import random
import sys
from time import perf_counter
import traceback

import numpy as np

from src.seed_replication import file_hash, read_json, safe_path, write_json
from src.universal_training import configuration, core_args, import_core, Tee, update_status, git_info
from src.universal_evaluation import describe_test, load_test, unchanged
from src.universal_evaluation_analysis import write_csv
from src.universal_replication_protocol import (valid_seed, resolve_training, evaluation_root, source_hashes,
    validate_target, formal_references)


def completed_attempt(parent):
    if (parent/"active.lock").exists():
        raise ValueError("Phase 仍有 active.lock")
    records=[p for p in parent.glob("attempt-*") if read_json(p/"status.json").get("completed") is True]
    if len(records)>1:
        raise ValueError("多個 successful attempts；不能自行挑選")
    return safe_path(records[0]) if records else None


def evaluate(roots, seed, mode="full", dry_run=False, retry=False, stage1_run=None):
    valid_seed(seed)
    if seed==1:
        from src.universal_evaluation import execute
        return execute(roots,target="machine-1-1" if mode=="smoke" else None,all_targets=mode=="full",
                       seed=1,stage1_run=stage1_run,dry_run=dry_run,retry=retry)
    if mode not in ("smoke","full"):
        raise ValueError("Evaluation mode 不符")
    bundle=resolve_training(roots,seed,stage1_run)
    targets=bundle["sources"][:1] if mode=="smoke" else bundle["sources"]
    hashes=source_hashes(roots.project)
    root=evaluation_root(roots.results,bundle)
    description=dict(mode=mode,seed=seed,targets=targets,stage1=bundle,source_file_sha256=hashes,
        test_inputs=[describe_test(roots,t) for t in targets],output_parent=str(root/mode),
        device="cpu",dtype="float64",window=10,eval_batch_size=128)
    if dry_run:
        print(json.dumps(description,ensure_ascii=False,indent=2));return description
    if mode=="full":
        smoke=completed_attempt(root/"smoke")
        if smoke is None or read_json(smoke/"launch.json").get("stage1") != bundle:
            raise ValueError("同 seed／checkpoint 的 evaluation smoke 尚未完成")
        validate_target(smoke/"targets"/targets[0],targets[0],bundle)
    from src.universal_threshold_artifacts import claim
    folder,lock=claim(root/mode,retry)
    try:
        update_status(folder,state="running",completed=False,seed=seed,mode=mode)
        write_json(folder/"launch.json",dict(description,command=[sys.executable,*sys.argv]))
        print("Universal evaluation directory:",folder,flush=True)
        with (folder/"run.log").open("x") as log:
            with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
                try:
                    validation=evaluate_targets(roots,bundle,targets,folder,mode,hashes)
                except BaseException:
                    traceback.print_exc();raise
        unchanged(bundle)
        if source_hashes(roots.project)!=hashes:
            raise ValueError("Evaluation source 改變")
        write_json(folder/"validation.json",validation)
        update_status(folder,state="completed",completed=True,seed=seed,mode=mode,target_count=len(targets),validation=validation,
                      checkpoint_sha256=bundle["checkpoint_sha256"])
        return folder
    except BaseException as error:
        update_status(folder,state="failed",completed=False,seed=seed,error=str(error));raise
    finally:
        lock.unlink()


def evaluate_targets(roots,bundle,targets,folder,mode,hashes):
    import torch
    from src.calibration_runtime import load_frozen_model,state_hash
    from src.universal_thresholding import frozen_scores
    from src.continuous_metrics import continuous_metrics
    from src.target_calibration import array_hash
    seed=bundle["seed"]
    core=import_core(core_args(roots,configuration(roots.project,seed),folder,Path(bundle["checkpoint"])))
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model,before=load_frozen_model(core,bundle["checkpoint"])
    rows=[]
    for index,target in enumerate(targets):
        print(f"seed={seed} [{index+1}/{len(targets)}] TEST {target}",flush=True)
        out=safe_path(folder/"targets"/target);out.mkdir(parents=True)
        description=describe_test(roots,target)
        raw,labels=load_test(roots,target,out,description)
        start=perf_counter()
        scores=frozen_scores(core,model,torch.from_numpy(raw))
        seconds=perf_counter()-start
        after=state_hash(model)
        if before!=after or model.training or any(p.requires_grad or p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen TEST inference 改變 model")
        metrics=continuous_metrics(scores,labels)
        metrics.update(anomaly_prevalence=float(labels.mean()),pr_auc_minus_prevalence=metrics["pr_auc_trapezoidal"]-float(labels.mean()))
        meta=dict(experiment="universal_continuous_evaluation_v1",setting="universal_checkpoint_evaluation",target=target,
            group=int(target.split("-")[1]),seed=seed,protocol="v2",model="TranAD",dataset="SMD",window=10,
            eval_batch_size=128,device="cpu",dtype="float64",checkpoint_id=bundle["run_id"],checkpoint_path=bundle["checkpoint"],
            checkpoint_sha256=bundle["checkpoint_sha256"],evaluation_code_sha256=bundle["evaluation_code_sha256"],
            stage1_run=bundle["run"],sources=bundle["sources"],source_count=28,zero_shot=False,target_train_used_in_stage1=True,
            test_input=description,num_test_points=len(raw),score_count=len(scores),label_count=len(labels),
            score_start_timestamp=0,score_end_exclusive=len(raw),num_positive_labels=int(labels.sum()),anomaly_prevalence=float(labels.mean()),
            model_retrained=False,optimizer_steps=0,model_eval=True,requires_grad=False,normalization_fitted=False,
            test_used_for_training=False,test_used_for_validation=False,test_used_for_adaptation=False,
            score_array_sha256=array_hash(scores),label_array_sha256=array_hash(labels),source_file_sha256=hashes,
            git=git_info(roots.project),inference_time_sec=seconds,torch_version=torch.__version__,numpy_version=np.__version__)
        with (out/"scores.npz").open("xb") as stream:
            np.savez_compressed(stream,scores=scores,labels=labels,timestamps=np.arange(len(raw),dtype=np.int64))
        write_json(out/"metadata.json",meta);write_json(out/"metrics.json",metrics)
        write_json(out/"validation.json",dict(seed=seed,parameter_hash_before=before,parameter_hash_after=after,
            parameters_unchanged=True,optimizer_steps=0,model_retrained=False,all_windows_single_machine=True,
            score_label_alignment=True,scores_finite=True,labels_binary=True,checkpoint_unchanged=True,stage1_inputs_unchanged=True,
            checkpoint_sha256=bundle["checkpoint_sha256"],score_count=len(scores),label_count=len(labels),
            score_array_sha256=meta["score_array_sha256"],label_array_sha256=meta["label_array_sha256"],scores_npz_sha256=file_hash(out/"scores.npz")))
        rows.append(validate_target(out,target,bundle))
        unchanged(bundle)
    if mode=="full":
        references,protected=formal_references(roots,seed,folder)
        lookup={(r["target_machine"],r["setting"]):r for r in references}
        out=folder/"summary";out.mkdir()
        comparison=[dict(target=r["target"],seed=seed,per_machine_pr_auc=lookup[r["target"],"per_machine"]["pr_auc"],
            lomo_pr_auc=lookup[r["target"],"cross_machine_zero_shot"]["pr_auc"],universal_pr_auc=r["pr_auc"]) for r in rows]
        write_csv(out/"target_metrics.csv",rows)
        write_csv(out/"matching_seed_comparison.csv",comparison)
        summary=dict(seed=seed,target_count=28,checkpoint_sha256=bundle["checkpoint_sha256"],unique_universal_checkpoint_hashes=1,
            mean_pr_auc=float(np.mean([r["pr_auc"] for r in rows])),median_pr_auc=float(np.median([r["pr_auc"] for r in rows])),
            sample_std_pr_auc=float(np.std([r["pr_auc"] for r in rows],ddof=1)),
            mean_average_precision=float(np.mean([r["average_precision"] for r in rows])),mean_roc_auc=float(np.mean([r["roc_auc"] for r in rows])),
            min_pr_auc_target=min(rows,key=lambda r:r["pr_auc"])["target"],max_pr_auc_target=max(rows,key=lambda r:r["pr_auc"])["target"],
            min_pr_auc=min(r["pr_auc"] for r in rows),max_pr_auc=max(r["pr_auc"] for r in rows),formal_input_sha256=protected)
        write_json(out/"summary.json",summary)
        for filename,value in protected.items():
            if file_hash(filename)!=value:raise ValueError("Formal references 改變")
    after=state_hash(model)
    if before!=after or len({r["checkpoint_sha256"] for r in rows})!=1:
        raise ValueError("28 targets 未保持同一 checkpoint")
    return dict(seed=seed,mode=mode,target_count=len(targets),checkpoint_sha256=bundle["checkpoint_sha256"],
        unique_universal_checkpoint_hashes=1,parameter_hash_before=before,parameter_hash_after=after,
        parameters_unchanged=True,model_retrained=False,optimizer_steps=0,stage1_inputs_unchanged=True)


def threshold_comparisons(roots,inputs,rows):
    references,protected=formal_references(roots,inputs["seed"],inputs["stage2_run"])
    lookup={(r["target_machine"],r["setting"]):r for r in references}
    paired=[dict(target=r["target"],seed=inputs["seed"],condition=r["condition"],
        per_machine_raw_f1=lookup[r["target"],"per_machine"]["raw_f1"],universal_raw_f1=r["raw_f1"],
        per_machine_minus_universal=lookup[r["target"],"per_machine"]["raw_f1"]-r["raw_f1"]) for r in rows]
    from src.universal_threshold_artifacts import CONDITIONS
    gaps=[]
    for condition in CONDITIONS:
        values=[r["per_machine_minus_universal"] for r in paired if r["condition"]==condition]
        gaps.append(dict(condition=condition,seed=inputs["seed"],mean_gap=float(np.mean(values)),median_gap=float(np.median(values)),
            universal_better=sum(v<0 for v in values),per_machine_better=sum(v>0 for v in values),ties=sum(v==0 for v in values)))
    deployments=[]
    for setting,kind in (("per_machine","local_full"),("cross_machine_zero_shot","source_only")):
        subset=[r for r in references if r["setting"]==setting]
        deployments.append(dict(deployment=setting,models=28,threshold_type=kind,seed=inputs["seed"],
            mean_raw_precision=float(np.mean([r["raw_precision"] for r in subset])),mean_raw_recall=float(np.mean([r["raw_recall"] for r in subset])),
            mean_raw_f1=float(np.mean([r["raw_f1"] for r in subset]))))
    return paired,gaps,deployments,protected
