"""Seed 2/3 fixtures 與多 seed 分析；不使用真實 formal checkpoint、不訓練。"""
from contextlib import redirect_stderr, redirect_stdout
import csv
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_target_calibration as fixtures
from src.calibration_analysis import summarize_run
from src.calibration_artifacts import experiment_id, failed_calibration_tasks, new_attempt
from src.calibration_multiseed import analyze_multiseed, build_multiseed_tables, collect
from src.calibration_protocol import (calibration_directory, canonical_fraction, load_configuration, manifest_rows,
                                      manifest_seeds, validate_manifest)
from src.calibration_runtime import baseline_runtime, evaluate
from src.seed_replication import manifest, task_parent, unique_name
from src.target_calibration import (array_hash, execute_calibration, file_hash, inspect_experiment,
                                     load_formal_scores, select_formal_run)

ROOT = Path(__file__).resolve().parents[1]


def fixture(temp, seed=2, target="machine-1-1"):
    roots, config, machines, old_run, train = fixtures.CalibrationArtifactTests().create_fixture(temp)
    created = {}
    for value in (2, 3):
        row = next(r for r in manifest(ROOT) if int(r["seed"]) == value and r["target_machine"] == target
                   and r["setting"] == "cross_machine_zero_shot")
        attempt = task_parent(roots.results, row) / unique_name("attempt")
        run = attempt / "runs" / f"formal_seed{value}_fixture"
        shutil.copytree(old_run, run)
        checkpoint = attempt / "checkpoints" / run.name / "model.ckpt"
        checkpoint.parent.mkdir(parents=True)
        with baseline_runtime() as core:
            torch.manual_seed(100+value)
            model = core.TranAD(38).double()
            torch.save(dict(epoch=4, model_state_dict=model.state_dict()), checkpoint)
            with np.load(old_run / "scores.npz") as old:
                errors = old["test_scores"] * (1 + .1 * value)
                labels = old["labels"].copy()
                union = old["labels_final"].copy()
            scores = errors.mean(axis=1)
            baseline, _, _ = evaluate(core, scores, union, .85)
        np.savez_compressed(run / "scores.npz", scores=scores, labels_final=union, test_scores=errors, labels=labels)
        meta = json.loads((run / "metadata.json").read_text())
        meta.update(seed=value, checkpoint=str(checkpoint))
        (run / "metadata.json").write_text(json.dumps(meta))
        (run / "evaluation.json").write_text(json.dumps(baseline))
        (attempt / "status.json").write_text(json.dumps(dict(row, completed=True, exit_code=0, run_path=str(run))))
        (attempt / "launch.json").write_text(json.dumps(dict(row=row)))
        effective = dict(seed=value, model="TranAD", dataset="SMD", protocol="v2", epochs=5,
            batch_size=128, eval_batch_size=128, smoke_test=False, checkpoint=None,
            test_machine=target, train_machines=json.loads(row["source_machines"]))
        (attempt / "worker_start.json").write_text(json.dumps(dict(row=row, effective_args=effective)))
        created[value] = (run, attempt, row)
    return roots, config, machines, old_run, train, created


class MultiSeedManifestTests(unittest.TestCase):
    def test_new_168_rows_and_original_84_unchanged(self):
        _, machines = load_configuration(ROOT)
        original = validate_manifest(ROOT / "configs/calibration_runs.csv", machines)
        rows = validate_manifest(ROOT / "configs/calibration_runs_seed23.csv", machines, (2, 3))
        self.assertEqual(len(original), 84)
        self.assertEqual(len(rows), 168)
        self.assertEqual(len({r["task_id"] for r in rows}), 168)
        self.assertEqual(len({(r["target"],r["fraction"],r["seed"]) for r in rows}), 168)
        self.assertEqual(rows[0], dict(task_id="0", target="machine-1-1", fraction="0.05", seed="2"))
        self.assertEqual(rows[84], dict(task_id="84", target="machine-1-1", fraction="0.05", seed="3"))
        for seed in ("2", "3"):
            for fraction in ("0.05", "0.10", "0.20"):
                self.assertEqual([r["target"] for r in rows if r["seed"]==seed and r["fraction"]==fraction], machines)

    def test_generator_cli_idempotent_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seed23.csv"
            command = [sys.executable, str(ROOT / "scripts/build_calibration_manifest.py"),
                       "--seeds", "2", "3", "--output", str(path)]
            before=None
            for _ in range(2):
                subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                digest=file_hash(path)
                if before is not None:self.assertEqual(digest,before)
                before=digest
            self.assertEqual(manifest_seeds(path), (2, 3))
            path.write_text("existing-invalid-content")
            result = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), "existing-invalid-content")


class SeedIsolationTests(unittest.TestCase):
    def test_checkpoint_scores_and_baseline_lookup_are_seed_specific(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, old, _, runs = fixture(temp)
            for seed in (2, 3):
                description, meta, baseline, paths, _ = inspect_experiment(
                    roots, roots.results, config, machines, "machine-1-1", ".05", seed)
                self.assertEqual(meta["seed"], seed)
                self.assertEqual(description["seed"], seed)
                self.assertEqual(description["formal_task_id"], 28 if seed==2 else 84)
                self.assertEqual(paths["scores.npz"], runs[seed][0] / "scores.npz")
                self.assertEqual(paths["checkpoint"].parent.parent.parent, runs[seed][1])
                self.assertEqual(baseline, json.loads((runs[seed][0] / "evaluation.json").read_text()))
                self.assertNotEqual(paths["scores.npz"], old / "scores.npz")
            first, _ = load_formal_scores(runs[2][0] / "scores.npz", 6)
            second, _ = load_formal_scores(runs[3][0] / "scores.npz", 6)
            self.assertFalse(np.array_equal(first, second))

    def test_cross_seed_metadata_checkpoint_and_status_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, old, _, runs = fixture(temp)
            for seed, wrong in ((2, 1), (3, 2)):
                run, attempt, row = runs[seed]
                metadata = json.loads((run / "metadata.json").read_text())
                for key, value in (("seed", wrong), ("checkpoint", str((old if wrong==1 else runs[wrong][0]).parent.parent / "checkpoints" / (old if wrong==1 else runs[wrong][0]).name / "model.ckpt"))):
                    altered = dict(metadata, **{key:value})
                    (run / "metadata.json").write_text(json.dumps(altered))
                    with self.subTest(seed=seed, key=key), self.assertRaises((ValueError, FileNotFoundError)):
                        inspect_experiment(roots, roots.results, config, machines, "machine-1-1", ".05", seed)
                (run / "metadata.json").write_text(json.dumps(metadata))
                if seed == 3:
                    archive=run/"scores.npz"
                    archive.rename(run/"scores-original.npz")
                    archive.symlink_to(runs[2][0]/"scores.npz")
                    with self.assertRaises(ValueError):
                        inspect_experiment(roots,roots.results,config,machines,"machine-1-1",".05",seed)
                    archive.unlink()
                    (run/"scores-original.npz").rename(archive)
                    status = json.loads((attempt / "status.json").read_text())
                    status["run_path"] = str(runs[2][0])
                    (attempt / "status.json").write_text(json.dumps(status))
                    with self.assertRaises(ValueError):
                        select_formal_run(roots.results, machines, "machine-1-1", seed)

    def test_seed2_smoke_real_frozen_inference_stub_pot_and_readonly_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, old, _, runs = fixture(temp)
            # 包含 seed=1 calibration 的 sentinel；新 smoke 不可更動。
            sentinel = roots.results / "calibration/runs/calib-machine-1-1-f005-seed1/old-artifact"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("immutable seed1 calibration")
            protected = {p:file_hash(p) for p in roots.results.rglob("*") if p.is_file()}
            def stub_pot(core, scores):
                self.assertEqual(scores.shape, (10,))
                self.assertTrue(np.isfinite(scores).all())
                return .35, dict(q=1e-5, requested_level=.99995, effective_level=.99995, multiplier=1.06,
                    initialization_attempts=1, source_score_count=len(scores), stream_score_count=0)
            with patch("src.calibration_runtime.fit_threshold", side_effect=stub_pot), \
                 patch.object(torch.optim.AdamW,"__init__",side_effect=AssertionError("optimizer forbidden")), \
                 patch.object(torch.Tensor,"backward",side_effect=AssertionError("backward forbidden")), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                output = execute_calibration(roots, roots.results, config, machines,
                    "machine-1-1", ".05", smoke=True, seed=2)
            self.assertEqual(output.parent.name, "smoke-calib-machine-1-1-f005-seed2")
            self.assertTrue(output.is_relative_to(roots.results / "calibration/seed-replication"))
            for path, digest in protected.items():
                self.assertEqual(file_hash(path), digest)
            meta = json.loads((output / "metadata.json").read_text())
            checks = json.loads((output / "validation.json").read_text())
            self.assertEqual(meta["seed"], 2)
            self.assertEqual(meta["calibration_rng_seed"], 1)
            self.assertEqual(meta["formal_run_id"], runs[2][0].name)
            self.assertEqual(meta["parameter_hash_before"], meta["parameter_hash_after"])
            self.assertEqual(meta["optimizer_steps"], 0)
            self.assertFalse(meta["model_retrained"])
            self.assertTrue(all(checks[k] for k in ("test_scores_unchanged","labels_unchanged","pr_auc_unchanged","ap_unchanged","roc_auc_unchanged")))
            self.assertEqual(checks["pot_test_stream_count"], 0)
            with np.load(output / "scores.npz") as new, np.load(runs[2][0] / "scores.npz") as original:
                np.testing.assert_array_equal(new["scores"], original["scores"])
                np.testing.assert_array_equal(new["labels_final"], original["labels_final"])
            failures,_=failed_calibration_tasks(calibration_directory(roots.results,roots.results,2),
                                               manifest_rows(machines,(2,3)))
            self.assertEqual(failures["count"],168)  # smoke 絕不可計入 formal matrix。
            with self.assertRaises(FileExistsError):
                execute_calibration(roots, roots.results, config, machines, "machine-1-1", ".05", smoke=True, seed=2, retry=True)

    def test_seed2_and_seed3_formal_calibration_outputs_pass_existing_deep_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            roots,config,machines,old,_,runs=fixture(temp)
            def stub_pot(core,scores):
                return .35,dict(q=1e-5,requested_level=.99995,effective_level=.99995,multiplier=1.06,
                    initialization_attempts=1,source_score_count=len(scores),stream_score_count=0)
            rows=manifest_rows(machines,(2,3))
            with patch("src.calibration_runtime.fit_threshold",side_effect=stub_pot), \
                 patch.object(torch.optim.AdamW,"__init__",side_effect=AssertionError("optimizer forbidden")), \
                 patch.object(torch.Tensor,"backward",side_effect=AssertionError("backward forbidden")), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
                for seed,task in ((2,0),(3,84)):
                    output=execute_calibration(roots,roots.results,config,machines,"machine-1-1",".05",
                        task_id=task,seed=seed,manifest_seed_values=(2,3))
                    result,meta=summarize_run(output,rows[task],config,
                        calibration_directory(roots.results,roots.results,seed),machines)
                    self.assertEqual(result["seed"],seed)
                    self.assertEqual(result["formal_run_id"],runs[seed][0].name)
                    self.assertEqual(meta["formal_seed"],seed)
                    scores,labels=load_formal_scores(runs[seed][0]/"scores.npz",6)
                    self.assertEqual(meta["test_scores_sha256"],array_hash(scores))
                    self.assertEqual(meta["labels_sha256"],array_hash(labels))

    def test_dry_cli_uses_manifest_seed3_and_writes_no_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, _, _, _, _, runs = fixture(temp)
            env = dict(os.environ, PROJECT_ROOT=str(ROOT), DATA_ROOT=str(roots.data), RESULTS_ROOT=str(roots.results),
                       PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, str(ROOT / "scripts/run_target_calibration.py"), "--hpc", "--manifest",
                       "configs/calibration_runs_seed23.csv", "--task-id", "84", "--dry-run"]
            result = subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            description,_ = json.JSONDecoder().raw_decode(result.stdout)
            self.assertEqual(description["seed"],3)
            self.assertEqual(description["formal_run"],str(runs[3][0]))
            self.assertFalse((roots.results / "calibration").exists())

    def test_invalid_latest_status_is_reported_and_smoke_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            _, machines=load_configuration(ROOT)
            rows=manifest_rows(machines,(2,3))
            namespace=Path(temp).resolve()/"calibration/seed-replication"
            _,_,folder=new_attempt(namespace,"machine-1-1",".05",seed=2)
            (folder/"status.json").write_text("invalid json")
            failure,_=failed_calibration_tasks(namespace,rows)
            self.assertIn(0,failure["failed_or_missing_task_ids"])
            self.assertTrue(failure["invalid_attempts"])
            _,_,retried=new_attempt(namespace,"machine-1-1",".05",seed=2,retry=True)
            self.assertNotEqual(folder,retried)
            self.assertNotEqual(experiment_id("machine-1-1",".05",seed=2),experiment_id("machine-1-1",".05",seed=3))


def synthetic_rows():
    _,machines=load_configuration(ROOT)
    rows=[]
    for seed in (1,2,3):
        for machine in machines:
            for fraction in ("0.05","0.10","0.20"):
                zero=.1*seed
                delta=.02*seed*float(fraction)/.05
                row=dict(target=machine,group=int(machine.split("-")[1]),seed=seed,fraction=float(fraction),
                         zero_shot_threshold=.8,calibrated_threshold=.4,absolute_shift=.4,signed_shift=-.4,
                         relative_ratio=.5,relative_ratio_status="defined")
                for metric in ("precision","recall","f1"):
                    row["zero_shot_raw_"+metric]=zero
                    row["calibrated_raw_"+metric]=zero+delta
                    row["raw_"+metric+"_delta"]=delta
                rows.append(row)
    return rows,machines


class MultiSeedAnalysisTests(unittest.TestCase):
    def test_collection_pairs_fractions_and_rejects_changed_prefix_or_labels_across_seeds(self):
        from src.paths import Roots
        rows,machines=synthetic_rows()
        source={(r["target"],canonical_fraction(r["fraction"]),r["seed"]):r for r in rows}
        config,_=load_configuration(ROOT)
        with tempfile.TemporaryDirectory() as temp:
            roots=Roots(ROOT,Path(temp).resolve()/"data",Path(temp).resolve()/"results")
            def statuses(namespace,expected):
                return dict(count=0),{int(r["task_id"]):dict(run_path=str(namespace/"runs"/(r["seed"]+"-"+r["task_id"]))) for r in expected}
            def audit(folder,row,*args):
                seed=int(row["seed"]);target=row["target"];fraction=row["fraction"]
                identity=target+"-"+str(seed)
                meta=dict(formal_run_id=identity,formal_checkpoint_hash=identity,
                    formal_score_artifact_hash=identity,test_scores_sha256=identity,labels_sha256=target,
                    protected_input_sha256={"evaluation.json":identity},prefix_array_sha256=target+fraction,
                    target_train_total_raw_points=200,calibration_raw_points=int(200*float(fraction)),test_points=6)
                return source[(target,fraction,seed)],meta
            with patch("src.calibration_multiseed.failed_calibration_tasks",side_effect=statuses), \
                 patch("src.calibration_multiseed.file_hash",return_value="fixture-file-hash"), \
                 patch("src.calibration_multiseed.summarize_run",side_effect=audit):
                result,_=collect(roots,roots.results,config,machines,(1,2,3))
            self.assertEqual(len(result),252)
            for field in ("prefix_array_sha256","labels_sha256"):
                def changed(folder,row,*args):
                    record,meta=audit(folder,row,*args)
                    if int(row["seed"])==3:meta[field]="different-data"
                    return record,meta
                with patch("src.calibration_multiseed.failed_calibration_tasks",side_effect=statuses), \
                     patch("src.calibration_multiseed.file_hash",return_value="fixture-file-hash"), \
                     patch("src.calibration_multiseed.summarize_run",side_effect=changed),self.assertRaises(ValueError):
                    collect(roots,roots.results,config,machines,(1,2,3))

    def test_seed_level_macro_sample_std_and_machine_recovery(self):
        rows,machines=synthetic_rows()
        tables=build_multiseed_tables(rows,machines)
        self.assertEqual(len(tables["seed_macro_results"]),12)
        item=next(r for r in tables["seed_macro_statistics"] if r["setting"]=="zero_shot" and r["metric"]=="raw_f1")
        self.assertAlmostEqual(item["mean"],.2)
        self.assertAlmostEqual(item["sample_std"],.1)
        item=next(r for r in tables["seed_recovery_statistics"] if r["fraction"]=="0.05" and r["metric"]=="raw_f1_recovery")
        self.assertAlmostEqual(item["mean"],.04)
        self.assertAlmostEqual(item["sample_std"],.02)
        self.assertEqual(len(tables["target_recovery_robustness"]),84)
        self.assertTrue(all(r["improved_seeds"]==3 for r in tables["target_recovery_robustness"]))
        self.assertTrue(all(r["targets_per_seed"]==28 for r in tables["seed_macro_statistics"]))
        for invalid in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(ValueError):build_multiseed_tables(invalid,machines)

    def test_mixed_machine_directions_and_all_fractions_preserved(self):
        rows,machines=synthetic_rows()
        selected=next(r for r in rows if r["target"]==machines[0] and r["seed"]==3 and r["fraction"]==.05)
        for metric in ("precision","recall","f1"):
            selected["calibrated_raw_"+metric]=.1
            selected["raw_"+metric+"_delta"]=-.2
        tables=build_multiseed_tables(rows,machines)
        first=tables["target_recovery_robustness"][0]
        self.assertEqual(first["category"],"mixed_direction")
        self.assertEqual((first["improved_seeds"],first["degraded_seeds"]),(2,1))
        self.assertEqual({r["setting"] for r in tables["seed_macro_results"]},{"zero_shot","0.05","0.10","0.20"})

    def test_summary_outputs_new_figures_and_does_not_touch_seed1(self):
        rows,machines=synthetic_rows()
        with tempfile.TemporaryDirectory() as temp:
            from src.paths import Roots
            roots=Roots(ROOT,Path(temp).resolve()/"data",Path(temp).resolve()/"results")
            old=roots.results/"calibration/summaries/original"
            old.mkdir(parents=True)
            sentinel=old/"seed1.png";sentinel.write_bytes(b"preserve seed1 graph")
            before=file_hash(sentinel)
            config,_=load_configuration(ROOT)
            with patch("src.calibration_multiseed.collect",return_value=(rows,{})):
                output=analyze_multiseed(roots,roots.results,config,machines)
                second=analyze_multiseed(roots,roots.results,config,machines)
            self.assertNotEqual(output,second)
            self.assertEqual(file_hash(sentinel),before)
            self.assertTrue(json.loads((output/"status.json").read_text())["completed"])
            self.assertEqual(len(list((output/"figures").glob("*.png"))),3)
            self.assertEqual(len(list((output/"figures").glob("*.pdf"))),3)
            self.assertEqual((output/"seed_macro_statistics.csv").read_bytes(),(second/"seed_macro_statistics.csv").read_bytes())


if __name__=="__main__":unittest.main()
