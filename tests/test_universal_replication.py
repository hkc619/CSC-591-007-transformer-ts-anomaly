"""Replication tests：random model/schema fixtures、stub POT；不執行真實 seeds 2/3 experiments。"""
import ast
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_universal_thresholding import fixture as legacy_fixture, guarded_phase
from test_universal_evaluation import ROOT, python, env
from test_seed_replication import completed_fixture
from src.paths import Roots
from src.seed_replication import (file_hash,manifest,read_json,task_parent,worker_args,PROTECTED)
from src.universal_training import configuration,run_identity
from src.universal_replication_protocol import resolve_training,evaluation_root,valid_seed,validate_target,formal_references
from src.universal_replication_analysis import tables,collect_seed,analyze
from src.universal_threshold_artifacts import resolve_inputs,root_path,read_train_scores


def fixtures(base):
    roots,stage1,stage2,checkpoint=legacy_fixture(base)
    # Formal synthetic schema 補上相同真實 fixture TRAIN/TEST hashes 與 point counts。
    # 這些不是使用者已完成的正式 artifacts。
    completed_fixture(roots)
    for row in manifest(ROOT):
        attempt=next(task_parent(roots.results,row).glob("attempt-*"))
        status=read_json(attempt/"status.json");run=Path(status["run_path"])
        meta=read_json(run/"metadata.json")
        target=row["target_machine"]
        description=read_json(stage2/"targets"/target/"metadata.json")["test_input"]
        meta["data_sha256"][description["test_path"]]=description["test_sha256"]
        meta["data_sha256"][description["label_path"]]=description["label_file_sha256"]
        meta["test_timestamps"]=description["raw_test_length"]
        records=read_json(stage1/"train_manifest.json")["machines"]
        lookup={r["machine"]:r for r in records}
        for m in meta["source_machines"]:
            meta["data_sha256"][str(roots.data/"train"/(m+".txt"))]=lookup[m]["raw_file_sha256"]
        meta["train_windows"]=sum(lookup[m]["raw_train_length"] for m in meta["source_machines"])
        meta["pot_fit"]["source_score_count"]=meta["train_windows"]
        evaluation=read_json(run/"evaluation.json")
        # 只調整 synthetic TN，使 validation count 等於實際 TEST count。
        evaluation["raw"]["TN"]=meta["test_timestamps"]-5
        (run/"metadata.json").write_text(json.dumps(meta));(run/"evaluation.json").write_text(json.dumps(evaluation))
        args=worker_args(roots,row,attempt,dry_run=True)
        (attempt/"launch.json").write_text(json.dumps(dict(row=row)))
        (attempt/"worker_start.json").write_text(json.dumps(dict(row=row,effective_args=vars(args))))
    # Seed-1 formal reference 使用相同 28 TEST/label hashes，另建立原 schema summary。
    references=[]
    for status_path in (roots.results/"task_status").glob("*.json"):
        status=read_json(status_path);run=Path(status["run_path"]);meta=read_json(run/"metadata.json")
        target=meta["target_machine"];description=read_json(stage2/"targets"/target/"metadata.json")["test_input"]
        meta["data_sha256"][description["test_path"]]=description["test_sha256"]
        meta["data_sha256"][description["label_path"]]=description["label_file_sha256"]
        meta["test_timestamps"]=description["raw_test_length"]
        evaluation=read_json(run/"evaluation.json")
        evaluation["raw"]["TN"]=meta["test_timestamps"]-5
        (run/"metadata.json").write_text(json.dumps(meta));(run/"evaluation.json").write_text(json.dumps(evaluation))
        references.append(dict(target_machine=target,setting=meta["setting"],seed=1,source_machines=";".join(meta["source_machines"]),
            checkpoint_path=meta["checkpoint"],test_points=meta["test_timestamps"],pr_auc=evaluation["continuous_score"]["pr_auc_trapezoidal"],
            raw_precision=evaluation["raw"]["precision"],raw_recall=evaluation["raw"]["recall"],raw_f1=evaluation["raw"]["f1"]))
    (roots.results/"summaries").mkdir()
    with (roots.results/"summaries/formal_28_machine_results.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(references[0]));writer.writeheader();writer.writerows(references)
    for seed in (1,2,3):
        destination=roots.results/"universal/runs"/run_identity(seed)/"attempt-replication-fixture"
        cp=roots.results/"universal/checkpoints"/run_identity(seed)/destination.name/"model.ckpt"
        if seed==1:
            destination=stage1;cp=checkpoint
        else:
            shutil.copytree(stage1,destination)
            cp.parent.mkdir(parents=True)
            code='''
import sys
from pathlib import Path
import torch
from src.paths import Roots
from src.universal_training import configuration,core_args,import_core
roots=Roots(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]));seed=int(sys.argv[4])
core=import_core(core_args(roots,configuration(roots.project,seed),Path(sys.argv[5]),Path(sys.argv[6])))
torch.manual_seed(seed)
model=core.TranAD(38).double()
with Path(sys.argv[6]).open("xb") as f:torch.save(dict(epoch=4,model_state_dict=model.state_dict(),optimizer_state_dict={},scheduler_state_dict={},accuracy_list=[(0.,0.)]*5),f)
'''
            result=python(code,ROOT,roots.data,roots.results,seed,destination,cp)
            if result.returncode:raise AssertionError(result.stdout+result.stderr)
        meta,proof,status=[read_json(destination/n) for n in ("metadata.json","validation.json","status.json")]
        meta.update(configuration(ROOT,seed),run_id=run_identity(seed),attempt_id=destination.name,checkpoint_path=str(cp),
            python="synthetic fixture",torch_version="synthetic",numpy_version="synthetic",torch_num_threads=1,installed_dependencies=["synthetic==1"])
        proof.update(seed=seed,checkpoint_sha256=file_hash(cp),checkpoint_size_bytes=cp.stat().st_size)
        status.update(seed=seed,checkpoint=str(cp),validation=proof)
        for n,value in (("metadata.json",meta),("validation.json",proof),("status.json",status)):(destination/n).write_text(json.dumps(value))
    # 上方只補 metadata profile，舊 Stage-2 launch Stage-1 hashes 必須與現在 fixture 一致。
    # 重建 test-owned seed-1 evaluation，沒有修改任何使用者實驗目錄。
    shutil.rmtree(roots.results/"universal/evaluation")
    from test_universal_evaluation import guard_evaluation
    guard_evaluation(roots,"smoke");stage2=guard_evaluation(roots,"full")
    return roots,stage1,stage2


def phase(roots,seed,kind,step):
    code='''
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.paths import Roots
import src.universal_replication as replication
import src.universal_thresholding as thresholding
roots=Roots(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]));seed=int(sys.argv[4]);kind=sys.argv[5];step=sys.argv[6]
def forbidden(*a,**kw):raise AssertionError("optimizer/backward/PA/TEST reinference 不允許")
def stub(core,scores):
    # 有限極端值必須原樣保留；不是實際 POT fitting。
    threshold=1e250 if len(scores)<5 else float(scores.mean()*1.06)
    return threshold,dict(q=1e-5,requested_level=.99995,effective_level=.99995,multiplier=1.06,
        initialization_attempts=1,source_score_count=len(scores),stream_score_count=0)
with patch.object(torch.optim.AdamW,"__init__",side_effect=forbidden),patch.object(torch.Tensor,"backward",side_effect=forbidden),patch("src.calibration_runtime.evaluate",side_effect=forbidden),patch("src.calibration_runtime.fit_threshold",side_effect=stub):
    if kind=="evaluate":
        folder=replication.evaluate(roots,seed,mode=step)
    else:
        from src.universal_threshold_artifacts import resolve_inputs
        inputs=resolve_inputs(roots,seed=seed)
        original=thresholding.frozen_scores
        def train_only(core,model,raw):
            assert len(raw)>=2
            return original(core,model,raw)
        with patch.object(thresholding,"frozen_scores",side_effect=train_only):
            folder=thresholding.execute(roots,seed=seed,phase=step)
print("RESULT="+str(folder))
'''
    result=python(code,ROOT,roots.data,roots.results,seed,kind,step)
    if result.returncode:raise AssertionError(result.stdout+result.stderr)
    return Path(next(s.split("=",1)[1] for s in result.stdout.splitlines() if s.startswith("RESULT=")))


class ReplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.roots,cls.stage1,cls.stage2=fixtures(Path(cls.temp.name).resolve())
        cls.legacy_smoke=guarded_phase(cls.roots,"smoke");cls.legacy_full=guarded_phase(cls.roots,"full")
        cls.seed1_protected={str(p):file_hash(p) for base in (cls.stage1,cls.stage2,cls.legacy_full) for p in base.rglob("*") if p.is_file()}
        cls.records={}
        for seed in (2,3):
            phase(cls.roots,seed,"evaluate","smoke")
            ev=phase(cls.roots,seed,"evaluate","full")
            phase(cls.roots,seed,"threshold","smoke")
            th=phase(cls.roots,seed,"threshold","full")
            cls.records[seed]=(ev,th)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_seeds_accepted_unsupported_rejected_and_configs_identical_except_seed(self):
        configs=[configuration(ROOT,s) for s in (1,2,3)]
        for s in (1,2,3):self.assertEqual(valid_seed(s),s)
        for bad in (0,4,-1,True,"2"):
            with self.assertRaises(ValueError):valid_seed(bad)
        for config in configs:
            self.assertEqual(len(config["sources"]),28)
            self.assertEqual({k:v for k,v in config.items() if k!="seed"},{k:v for k,v in configs[0].items() if k!="seed"})

    def test_checkpoint_namespaces_and_model_weights_differ(self):
        bundles=[resolve_training(self.roots,s) for s in (1,2,3)]
        self.assertEqual(len({b["checkpoint"] for b in bundles}),3)
        self.assertEqual(len({b["checkpoint_sha256"] for b in bundles}),3)
        self.assertEqual(len({run_identity(s) for s in (1,2,3)}),3)
        for b in bundles:self.assertIn(f"seed{b['seed']}",b["checkpoint"])

    def test_cross_seed_checkpoint_test_and_train_caches_rejected(self):
        b2,b3=resolve_training(self.roots,2),resolve_training(self.roots,3)
        with self.assertRaises(ValueError):resolve_training(self.roots,3,b2["run"])
        with self.assertRaises(ValueError):validate_target(self.records[2][0]/"targets/machine-1-1","machine-1-1",b3)
        with self.assertRaises(ValueError):validate_target(self.stage2/"targets/machine-1-1","machine-1-1",b2)
        i2,i3=resolve_inputs(self.roots,seed=2),resolve_inputs(self.roots,seed=3)
        cache=next((root_path(self.roots,i2)/"train_scores/machine-1-1").glob("attempt-*"))
        record=read_json(Path(b3["run"])/"train_manifest.json")["machines"][0]
        with self.assertRaises(ValueError):read_train_scores(cache,record,i3)
        with self.assertRaises(ValueError):resolve_inputs(self.roots,stage2_run=self.records[2][0],seed=3)

    def test_each_seed_one_checkpoint_140_conditions_invariants_and_extreme_threshold_retained(self):
        for seed in (2,3):
            record=collect_seed(self.roots,seed)
            self.assertEqual(len(record["rows"]),140)
            self.assertEqual({r["seed"] for r in record["rows"]},{seed})
            self.assertEqual(len({r["checkpoint_sha256"] for r in record["rows"]}),1)
            self.assertEqual({r["condition"] for r in record["rows"]},{"global","local_05","local_10","local_20","local_full"})
            self.assertTrue(any(r["threshold"]==1e250 for r in record["rows"]))
            proof=read_json(self.records[seed][1]/"validation.json")
            self.assertFalse(proof["test_inference_executed"]);self.assertFalse(proof["model_retrained"])
            self.assertEqual(proof["optimizer_steps"],0);self.assertEqual(proof["parameter_hash_before"],proof["parameter_hash_after"])
            self.assertEqual(read_json(self.records[seed][1]/"global_threshold.json")["pot_rng_seed"],1)

    def test_matching_seed_formal_references_do_not_fall_back(self):
        for seed in (2,3):
            refs,_=formal_references(self.roots,seed,self.records[seed][0])
            self.assertEqual(len(refs),56);self.assertEqual({r["seed"] for r in refs},{seed})
        row=next(r for r in manifest(ROOT) if r["seed"]=="2")
        status=next(task_parent(self.roots.results,row).glob("attempt-*/status.json"))
        saved=status.read_bytes()
        try:
            status.write_text(json.dumps(dict(read_json(status),completed=False,exit_code=1)))
            with self.assertRaises(ValueError):formal_references(self.roots,2,self.records[2][0])
        finally:status.write_bytes(saved)

    def test_wrong_formal_source_or_train_hash_rejected(self):
        row=next(r for r in manifest(ROOT) if r["seed"]=="3")
        attempt=next(task_parent(self.roots.results,row).glob("attempt-*"))
        worker_path=attempt/"worker_start.json";original=worker_path.read_bytes()
        try:
            worker=read_json(worker_path);worker["effective_args"]["machine"]="machine-9-9"
            worker_path.write_text(json.dumps(worker))
            with self.assertRaises(ValueError):formal_references(self.roots,3,self.records[3][0])
        finally:worker_path.write_bytes(original)
        run=Path(read_json(attempt/"status.json")["run_path"])
        path=run/"metadata.json";original=path.read_bytes()
        try:
            meta=read_json(path)
            train_key=next(p for p in meta["data_sha256"] if Path(p).parent.name=="train")
            meta["data_sha256"][train_key]="0"*64;path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):formal_references(self.roots,3,self.records[3][0])
        finally:path.write_bytes(original)

    def test_seed1_artifacts_and_frozen_training_evaluation_files_unchanged(self):
        for path,value in self.seed1_protected.items():self.assertEqual(file_hash(path),value,path)
        for name in (*PROTECTED,"src/universal_training.py","src/universal_evaluation.py","src/universal_evaluation_analysis.py",
                     "scripts/run_universal_training.py","scripts/run_universal_evaluation.py","scripts/run_universal_thresholding.py"):
            self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(["git","show","HEAD:"+name],cwd=ROOT),name)

    def test_multiseed_aggregation_seed_level_ddof1_and_immutable_inputs(self):
        before={str(p):file_hash(p) for p in self.roots.results.rglob("*") if p.is_file()}
        folder=analyze(self.roots)
        summary=read_json(folder/"multiseed_summary.json")
        self.assertEqual(summary["seed_count"],3)
        for name in ("ranking_by_seed.csv","threshold_by_seed.csv","seed_level_gaps.csv","machine_direction_consistency.csv",
                     "multiseed_ranking_summary.csv","multiseed_threshold_summary.csv","multiseed_gap_summary.csv"):
            self.assertTrue((folder/name).is_file())
        with (folder/"multiseed_threshold_summary.csv").open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(len(rows),18)
        for row in rows:
            values=[float(row["seed"+str(s)]) for s in (1,2,3)]
            self.assertAlmostEqual(float(row["sample_std"]),np.std(values,ddof=1))
            self.assertEqual(row["seed_count"],"3")
        for p,value in before.items():self.assertEqual(file_hash(p),value,p)

    def test_approved_historical_seed1_code_namespace_still_readable(self):
        approved=read_json(ROOT/"configs/universal_seed1_compatibility.json")
        original_root=self.legacy_full.parent.parent
        legacy_root=original_root.parent/approved["stage3_code_sha256"]
        shutil.copytree(original_root,legacy_root)
        full=legacy_root/"full"/self.legacy_full.name
        try:
            launch=read_json(full/"launch.json")
            launch["source_file_sha256"]=approved["source_file_sha256"]
            (full/"launch.json").write_text(json.dumps(launch))
            record=collect_seed(self.roots,1,full)
            self.assertEqual(len(record["rows"]),140)
            self.assertEqual(record["seed"],1)
            for p,value in self.seed1_protected.items():self.assertEqual(file_hash(p),value,p)
        finally:shutil.rmtree(legacy_root)

    def test_dependency_gates_and_dry_run_plan_write_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();roots=Roots(ROOT,self.roots.data,root/"results")
            with self.assertRaises(ValueError):resolve_training(roots,2)
            result=subprocess.run([sys.executable,"scripts/run_universal_replication.py","--seed","2","--phase","plan",
                "--data-root",str(roots.data),"--results-root",str(roots.results)],cwd=ROOT,env=env(),text=True,capture_output=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)["config"]["seed"],2)
            self.assertIn("seed2",json.loads(result.stdout)["replication"]["stage2_path"])
            self.assertIn("seed2",json.loads(result.stdout)["replication"]["stage3_path"])
            self.assertFalse(roots.results.exists())

    def test_completed_stage_dependencies_are_required(self):
        from src.universal_replication import evaluate
        from src.universal_thresholding import execute
        bundle=resolve_training(self.roots,2)
        stage1_status=Path(bundle["run"])/"status.json"
        stage2_status=self.records[2][0]/"status.json"
        for path,action in ((stage1_status,lambda:evaluate(self.roots,2,dry_run=True)),
                            (stage2_status,lambda:execute(self.roots,seed=2,dry_run=True))):
            original=path.read_bytes()
            try:
                status=read_json(path);status.update(completed=False,state="running")
                path.write_text(json.dumps(status))
                with self.assertRaises(ValueError):action()
            finally:path.write_bytes(original)


class SeedLevelStatisticsTests(unittest.TestCase):
    def test_frozen_numerical_functions_and_pot_rng_tail_unchanged(self):
        for filename,names in (("src/universal_threshold_analysis.py",("raw_metrics",)),
                               ("src/universal_thresholding.py",("frozen_scores",))):
            old=ast.parse(subprocess.check_output(["git","show","HEAD:"+filename],cwd=ROOT,text=True))
            new=ast.parse((ROOT/filename).read_text())
            for name in names:
                a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
                b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
                self.assertEqual(ast.dump(a),ast.dump(b))
        filename="src/universal_thresholding.py"
        functions=[]
        for text in (subprocess.check_output(["git","show","HEAD:"+filename],cwd=ROOT,text=True),(ROOT/filename).read_text()):
            fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="fit_train_threshold")
            index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                       and ast.unparse(n.value.func)=="np.random.seed")
            functions.append([ast.dump(n) for n in fn.body[index:]])
        self.assertEqual(functions[0],functions[1])

    def test_macro_std_and_direction_counts_use_three_seeds_not_84(self):
        from src.universal_threshold_artifacts import CONDITIONS
        records=[]
        for seed in (1,2,3):
            ranking=[];rows=[]
            for i,target in enumerate(configuration(ROOT,seed)["sources"]):
                ranking.append(dict(target=target,per_machine_pr_auc=.4+i*.001,lomo_pr_auc=.1+i*.001,
                                    universal_pr_auc=.2+i*.001+seed*.01))
                base=.1+i*.001
                deltas=dict(global_=0,local_05=.05 if seed!=2 else -.05,local_10=.1,local_20=0,local_full=-.05)
                for condition in CONDITIONS:
                    f1=base+(0 if condition=="global" else deltas[condition])
                    rows.append(dict(target=target,group=int(target.split("-")[1]),seed=seed,condition=condition,
                        threshold=.5,raw_precision=f1,raw_recall=f1,raw_f1=f1,pr_auc=ranking[-1]["universal_pr_auc"],
                        average_precision=.2,roc_auc=.6,checkpoint_sha256=str(seed),test_scores_sha256=target,
                        test_labels_sha256=target,test_timestamps_sha256=target,test_archive_sha256=target))
            records.append(dict(seed=seed,rows=rows,ranking=ranking,
                deployments=[dict(deployment="per_machine",mean_raw_precision=.5,mean_raw_recall=.5,mean_raw_f1=.5)]))
        output=tables(records)
        row=next(r for r in output["multiseed_ranking_summary"] if r["setting"]=="universal")
        self.assertAlmostEqual(row["mean"],.2335)
        self.assertAlmostEqual(row["sample_std"],.01)
        self.assertEqual(row["seed_count"],3)
        directions=output["machine_direction_consistency"]
        for comparison,category in (("universal_minus_lomo_pr_auc","all_improved"),("local_05","mixed_or_ties"),
            ("local_10","all_improved"),("local_20","all_unchanged"),("local_full","all_degraded")):
            self.assertEqual(sum(r["comparison"]==comparison and r["category"]==category for r in directions),28)
        self.assertEqual(sum(r["comparison"]=="local_05" and r["direction_changed"] for r in directions),28)
        self.assertEqual(sum(r["comparison"]=="local_20" and r["direction_changed"] for r in directions),0)
        with self.assertRaises(ValueError):tables(records*28)


if __name__=="__main__":unittest.main()
