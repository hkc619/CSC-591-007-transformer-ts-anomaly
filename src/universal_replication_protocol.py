"""多 seed 的 metadata／path adapter；保留 seed-1 frozen entrypoints 及其 fingerprints。"""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.seed_replication import (PROTECTED, file_hash, latest_attempt, manifest, read_json,
    safe_path, task_parent)
from src.universal_training import configuration, namespace, run_identity


def valid_seed(seed):
    if type(seed) is not int or seed not in (1, 2, 3):
        raise ValueError("Universal 只接受 seeds=1,2,3")
    return seed


def digest(mapping):
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


def source_hashes(project):
    from src.universal_evaluation import EVALUATION_SOURCES
    names = (*EVALUATION_SOURCES, "src/universal_replication_protocol.py", "src/universal_replication.py",
             "src/universal_replication_analysis.py", "scripts/run_universal_replication.py",
             "configs/universal_seed1_compatibility.json",
             "src/universal_threshold_artifacts.py", "src/universal_thresholding.py", "src/universal_threshold_analysis.py")
    return {n: file_hash(Path(project)/n) for n in names}


def resolve_training(roots, seed=1, stage1_run=None):
    valid_seed(seed)
    if seed == 1:
        from src.universal_evaluation import resolve_stage1
        return resolve_stage1(roots, stage1_run)
    parent = safe_path(namespace(roots.results)/"runs"/run_identity(seed))
    if (parent/"active.lock").exists():
        raise ValueError("同 seed Stage 1 仍在執行")
    if stage1_run:
        run = safe_path(Path(stage1_run).expanduser().absolute())
        if run.parent != parent:
            raise ValueError("Stage-1 attempt 屬於其他 seed／namespace")
    else:
        completed = [p for p in parent.glob("attempt-*") if read_json(p/"status.json").get("completed") is True]
        if len(completed) != 1:
            raise ValueError(f"seed={seed} 需要唯一 completed Stage-1 attempt；目前 {len(completed)} 個")
        run = safe_path(completed[0])
    paths = [run/n for n in ("status.json", "metadata.json", "validation.json", "launch.json",
                             "train_manifest.json", "window_provenance.json", "training_history.json")]
    status, meta, proof = [read_json(p) for p in paths[:3]]
    config = configuration(roots.project, seed)
    if (status.get("state") != "completed" or status.get("completed") is not True or status.get("seed") != seed
            or any(meta.get(k) != v for k,v in config.items()) or meta.get("run_id") != run_identity(seed)
            or meta.get("attempt_id") != run.name or meta.get("machine_count") != 28
            or status.get("validation") != proof or proof.get("seed") != seed or proof.get("machine_count") != 28
            or proof.get("epochs_completed") != 5 or meta.get("test_loaded") is not False):
        raise ValueError("Stage-1 completion／seed／protocol 不符")
    for key in ("checkpoint_reload_passed", "model_state_equal", "optimizer_state_equal", "scheduler_state_equal",
                "epoch_history_equal", "all_windows_single_machine"):
        if proof.get(key) is not True:
            raise ValueError("Stage-1 validation 未成功："+key)
    checkpoint = safe_path(namespace(roots.results)/"checkpoints"/run_identity(seed)/run.name/"model.ckpt")
    if (status.get("checkpoint") != str(checkpoint) or meta.get("checkpoint_path") != str(checkpoint)
            or not checkpoint.is_file() or file_hash(checkpoint) != proof.get("checkpoint_sha256")
            or checkpoint.stat().st_size != proof.get("checkpoint_size_bytes")):
        raise ValueError("Checkpoint seed／path／SHA256／size 不符")
    manifest_data, provenance = read_json(paths[4]), read_json(paths[5])
    for table in (manifest_data, provenance):
        records = table["machines"]
        if table.get("machine_count") != 28 or [r["machine"] for r in records] != config["sources"]:
            raise ValueError("Sources 必須包含全部 canonical 28 台")
        offset = 0
        for r in records:
            n = r["raw_train_length"]
            if n <= 0 or r["window_count"] != n or r["offset_start"] != offset or r["offset_end_exclusive"] != offset+n or r["dropped_points"] != 0:
                raise ValueError("TRAIN window offsets／natural weighting 不符")
            offset += n
        if table["total_training_windows"] != offset or meta["total_training_windows"] != offset or proof["total_training_windows"] != offset:
            raise ValueError("Stage-1 window counts 不一致")
    if (provenance.get("all_windows_single_machine") is not True or provenance.get("windowed_before_concatenation") is not True
            or provenance.get("raw_concatenation_before_windowing") is not False):
        raise ValueError("Machine-isolated windowing 證據不足")
    for name in PROTECTED:
        snapshot = safe_path(run/"source_snapshot"/name)
        value = meta["source_file_sha256"].get(name)
        if file_hash(snapshot) != value or file_hash(roots.project/name) != value:
            raise ValueError("Scientific core 與 Stage 1 不符："+name)
        paths.append(snapshot)
    paths.append(checkpoint)
    return dict(run=str(run), run_id=run_identity(seed), seed=seed, sources=config["sources"],
        checkpoint=str(checkpoint), checkpoint_sha256=proof["checkpoint_sha256"],
        evaluation_code_sha256=digest(source_hashes(roots.project)), input_hashes={str(p):file_hash(p) for p in paths})


def evaluation_root(results, bundle):
    if bundle["seed"] == 1:
        from src.universal_evaluation import evaluation_root as legacy
        return legacy(results, bundle)
    return safe_path(namespace(results)/"evaluation"/f"seed{bundle['seed']}"/bundle["checkpoint_sha256"]/bundle["evaluation_code_sha256"])


def validate_target(folder, target, bundle):
    if bundle["seed"] == 1:
        from src.universal_evaluation_analysis import validate_target as legacy
        return legacy(folder, target, bundle)
    from src.continuous_metrics import continuous_metrics
    from src.target_calibration import array_hash
    meta, proof, metrics = [read_json(safe_path(folder/n)) for n in ("metadata.json", "validation.json", "metrics.json")]
    fixed = dict(seed=bundle["seed"], target=target, protocol="v2", model="TranAD", dataset="SMD", window=10,
        eval_batch_size=128, device="cpu", dtype="float64", checkpoint_id=bundle["run_id"], checkpoint_path=bundle["checkpoint"],
        checkpoint_sha256=bundle["checkpoint_sha256"], evaluation_code_sha256=bundle["evaluation_code_sha256"],
        sources=bundle["sources"], source_count=28, stage1_run=bundle["run"], model_retrained=False, optimizer_steps=0,
        model_eval=True, requires_grad=False, normalization_fitted=False, test_used_for_training=False,
        test_used_for_validation=False, test_used_for_adaptation=False, zero_shot=False, target_train_used_in_stage1=True)
    if any(meta.get(k) != v for k,v in fixed.items()) or proof.get("seed") != bundle["seed"]:
        raise ValueError("Stage-2 TEST artifact 屬於其他 seed／checkpoint／protocol")
    for key in ("parameters_unchanged", "all_windows_single_machine", "score_label_alignment", "scores_finite", "labels_binary", "checkpoint_unchanged", "stage1_inputs_unchanged"):
        if proof.get(key) is not True:
            raise ValueError("Stage-2 invariant 未成功："+key)
    if (not proof.get("parameter_hash_before") or proof["parameter_hash_before"] != proof.get("parameter_hash_after")
            or proof.get("optimizer_steps") != 0 or proof.get("model_retrained") is not False
            or proof.get("checkpoint_sha256") != bundle["checkpoint_sha256"]):
        raise ValueError("Stage-2 model freeze／checkpoint 不符")
    path = safe_path(folder/"scores.npz")
    if file_hash(path) != proof["scores_npz_sha256"]:
        raise ValueError("Stage-2 archive 已改變")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"scores", "labels", "timestamps"}:
            raise ValueError("Stage-2 archive schema 不符")
        scores, labels, times = [saved[k] for k in ("scores", "labels", "timestamps")]
    n = meta["num_test_points"]
    if (scores.shape != (n,) or scores.dtype != np.float64 or labels.shape != (n,) or labels.dtype != np.int64
            or times.dtype != np.int64 or not np.array_equal(times,np.arange(n)) or not np.isfinite(scores).all()
            or not np.isin(labels,[0,1]).all() or meta["test_input"]["raw_test_length"] != n
            or any(meta[k] != n for k in ("score_count","label_count","score_end_exclusive")) or meta["score_start_timestamp"] != 0
            or proof["score_count"] != n or proof["label_count"] != n):
        raise ValueError("Stage-2 TEST arrays／timestamps alignment 不符")
    for key, arr in (("score_array_sha256",scores),("label_array_sha256",labels)):
        if array_hash(arr) != meta[key] or meta[key] != proof[key]:
            raise ValueError("Stage-2 array SHA256 不符")
    continuous = continuous_metrics(scores,labels)
    for key,value in continuous.items():
        if not np.isclose(value,metrics[key],atol=1e-12,rtol=0):
            raise ValueError("Stage-2 continuous metrics 不符")
    prevalence=float(labels.mean())
    if meta["anomaly_prevalence"] != prevalence or meta["num_positive_labels"] != int(labels.sum()) or metrics["anomaly_prevalence"] != prevalence:
        raise ValueError("Stage-2 prevalence 不符")
    return dict(target=target,group=int(target.split("-")[1]),seed=bundle["seed"],test_points=n,anomaly_points=int(labels.sum()),
        anomaly_prevalence=prevalence,pr_auc=continuous["pr_auc_trapezoidal"],average_precision=continuous["average_precision"],
        roc_auc=continuous["roc_auc"],pr_auc_minus_prevalence=continuous["pr_auc_trapezoidal"]-prevalence,
        checkpoint_id=bundle["run_id"],checkpoint_sha256=bundle["checkpoint_sha256"])


def formal_references(roots, seed, stage2_run):
    """Matching-seed canonical formal manifest + worker + validated bundle；不回退 seed 1。"""
    from src.seed_replication_analysis import validate_bundle
    seed=valid_seed(seed)
    if seed == 1:
        raise ValueError("Seed-1 baseline 由 frozen formal summary adapter 讀取")
    rows=[r for r in manifest(roots.project) if int(r["seed"]) == seed]
    result, protected = [], {}
    for row in rows:
        parent=task_parent(roots.results,row)
        successes=[p for p in parent.glob("attempt-*/status.json") if read_json(p).get("completed")]
        record=latest_attempt(roots.results,row)
        if len(successes)!=1 or record is None or record[1].get("exit_code")!=0 or not record[1].get("completed"):
            raise ValueError(f"缺少唯一 matching-seed formal completion：seed={seed}, task={row['task_id']}")
        attempt,status=record
        launch,worker=read_json(attempt/"launch.json"),read_json(attempt/"worker_start.json")
        if launch.get("row") != row or worker.get("row") != row or worker["effective_args"].get("seed") != seed:
            raise ValueError("Formal worker／manifest seed 不符")
        cross=row["setting"]=="cross_machine_zero_shot"
        fixed=dict(seed=seed,model="TranAD",dataset="SMD",protocol="v2",epochs=5,batch_size=128,eval_batch_size=128,
            machine=row["target_machine"],test_machine=row["target_machine"] if cross else None,
            train_machines=json.loads(row["source_machines"]) if cross else None,checkpoint=None,smoke_test=False,test=False,less=False)
        if any(worker["effective_args"].get(k)!=v for k,v in fixed.items()):
            raise ValueError("Formal worker 實際 model／sources／protocol 不符")
        run=safe_path(Path(status["run_path"]))
        reference,meta=validate_bundle(roots.project,row,run,attempt)
        target_meta=read_json(Path(stage2_run)/"targets"/row["target_machine"]/"metadata.json")
        training=read_json(Path(target_meta["stage1_run"])/"train_manifest.json")["machines"]
        by_machine={r["machine"]:r for r in training}
        expected_train={m:by_machine[m]["raw_file_sha256"] for m in meta["source_machines"]}
        recorded_train=[(Path(p).stem,v) for p,v in meta["data_sha256"].items() if Path(p).parent.name=="train"]
        if (len(recorded_train)!=len(expected_train) or dict(recorded_train)!=expected_train
                or meta["train_windows"]!=sum(by_machine[m]["raw_train_length"] for m in meta["source_machines"])):
            raise ValueError("Formal／Universal TRAIN splits／自然 windows 數量不一致")
        for split,key in (("test","test_sha256"),("interpretation_label","label_file_sha256")):
            values=[v for p,v in meta["data_sha256"].items() if Path(p).parent.name==split and Path(p).stem==row["target_machine"]]
            if values != [target_meta["test_input"][key]]:
                raise ValueError("Formal／Universal TEST 或 labels 不同")
        result.append(reference)
        for p in (attempt/"status.json",attempt/"launch.json",attempt/"worker_start.json",run/"metadata.json",run/"evaluation.json"):
            protected[str(p)]=file_hash(p)
    if len(result)!=56:
        raise ValueError("Matching-seed formal references 必須為 56 rows")
    return result,protected
