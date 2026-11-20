"""Stage 3 synthetic tests；POT orchestration 使用 stub，另測原 SPOT adapter 的 empty-stream contract。"""
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

from test_universal_evaluation import ROOT, create_fixture, env, guard_evaluation, python
from src.paths import Roots
from src.seed_replication import PROTECTED, file_hash, read_json
from src.target_calibration import array_hash
from src.universal_training import inspect_train as stage1_inspect, configuration
from src.universal_threshold_artifacts import CONDITIONS, check_inputs, resolve_inputs, root_path
from src.universal_threshold_analysis import aggregate, raw_metrics, validate_rows
from src.universal_thresholding import execute, fit_train_threshold


def fixture(base):
    roots, stage1, checkpoint = create_fixture(base)
    (roots.data / "train").mkdir()
    for i, machine in enumerate(configuration(ROOT, 1)["sources"]):
        count = 40+i%3
        raw = .2+i*.002 + np.arange(count)[:, None]*.002 + np.arange(38)[None, :]*.0007
        np.savetxt(roots.data / "train" / (machine+".txt"), raw, delimiter=",")
    manifest = stage1_inspect(roots, configuration(ROOT, 1))
    (stage1 / "train_manifest.json").write_text(json.dumps(manifest))
    provenance = dict(manifest, all_windows_single_machine=True, windowed_before_concatenation=True,
                      raw_concatenation_before_windowing=False)
    (stage1 / "window_provenance.json").write_text(json.dumps(provenance))
    metadata, validation, status = [read_json(stage1 / name) for name in ("metadata.json", "validation.json", "status.json")]
    for record in (metadata, validation):
        record["total_training_windows"] = manifest["total_training_windows"]
    status["validation"] = validation
    for name, value in (("metadata.json", metadata), ("validation.json", validation), ("status.json", status)):
        (stage1 / name).write_text(json.dumps(value))
    guard_evaluation(roots, "smoke")
    stage2 = guard_evaluation(roots, "full")
    return roots, stage1, stage2, checkpoint


def guarded_phase(roots, phase):
    code = '''
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import torch
from src.paths import Roots
import src.universal_thresholding as thresholding
from src.calibration_runtime import load_frozen_model
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
original_import = thresholding.import_core
calls, inference_calls = [], []
def forbidden(*args, **kwargs):
    raise AssertionError("禁止 TEST inference、training、optimizer、backward 或 PA")
def guarded_import(args):
    core = original_import(args)
    import src.cross_machine as cm
    import preprocess
    original_loader = cm.load_and_save
    def train_only(category, *args, **kwargs):
        assert category == "train", category
        return original_loader(category, *args, **kwargs)
    cm.load_and_save = train_only
    preprocess.load_and_save2 = forbidden
    core.load_model = core.save_model = core.adjust_predicts = forbidden
    original_backprop = core.backprop
    def infer(*args, **kwargs):
        assert kwargs.get("training") is False and args[4] is None and args[5] is None
        inference_calls.append(len(args[2]))
        return original_backprop(*args, **kwargs)
    core.backprop = infer
    return core
def stub_pot(core, scores):
    calls.append(scores.tolist())
    # 明確的 test stub：不宣稱是真實 POT 結果。
    return float(np.mean(scores)*1.06), dict(q=1e-5, requested_level=.99995,
        effective_level=.99995*.999, multiplier=1.06, initialization_attempts=2,
        source_score_count=len(scores), stream_score_count=0)
with patch.object(thresholding, "import_core", side_effect=guarded_import), \
     patch.object(torch.optim.AdamW, "__init__", side_effect=forbidden), \
     patch.object(torch.Tensor, "backward", side_effect=forbidden), \
     patch("src.calibration_runtime.evaluate", side_effect=forbidden), \
     patch("src.calibration_runtime.fit_threshold", side_effect=stub_pot), \
     patch("src.calibration_runtime.load_frozen_model", wraps=load_frozen_model) as load:
    folder = thresholding.execute(roots, phase=sys.argv[4])
    assert load.call_count == 1
(folder/"test_fit_calls.json").write_text(json.dumps(dict(pot_inputs=calls, inference_lengths=inference_calls)))
print("STAGE3="+str(folder))
'''
    result = python(code, ROOT, roots.data, roots.results, phase)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return Path(next(s.split("=", 1)[1] for s in result.stdout.splitlines() if s.startswith("STAGE3=")))


class UniversalThresholdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.roots, cls.stage1, cls.stage2, cls.checkpoint = fixture(Path(cls.temp.name).resolve())
        cls.inputs = resolve_inputs(cls.roots)
        cls.protected = dict(cls.inputs["protected_input_sha256"])
        cls.smoke = guarded_phase(cls.roots, "smoke")
        cls.train = guarded_phase(cls.roots, "train-scores")
        cls.full = guarded_phase(cls.roots, "full")
        with (cls.full / "summary/target_condition_metrics.csv").open() as stream:
            cls.csv_rows = list(csv.DictReader(stream))
        cls.rows = []
        for original in cls.csv_rows:
            row = dict(original)
            for key in ("group", "seed", "tp", "tn", "fp", "fn"):
                row[key] = int(row[key])
            for key in ("threshold", "raw_precision", "raw_recall", "raw_f1", "pr_auc", "average_precision", "roc_auc"):
                row[key] = float(row[key])
            cls.rows.append(row)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_same_checkpoint_140_rows_five_conditions_no_fraction_selection(self):
        validate_rows(self.rows, self.inputs["targets"], self.inputs["checkpoint_sha256"])
        self.assertEqual(len(self.rows), 140)
        self.assertEqual({r["condition"] for r in self.rows}, set(CONDITIONS))
        self.assertEqual({r["checkpoint_sha256"] for r in self.rows}, {file_hash(self.checkpoint)})
        self.assertFalse(list(self.full.rglob("*best*")))
        self.assertEqual({r["seed"] for r in self.rows}, {1})

    def test_stage2_scores_labels_timestamps_archives_and_checkpoint_immutable(self):
        check_inputs(self.inputs)
        for path, digest in self.protected.items():
            self.assertEqual(file_hash(path), digest, path)
        for target in self.inputs["targets"]:
            subset = [r for r in self.rows if r["target"] == target]
            for key in ("test_scores_sha256", "test_labels_sha256", "test_timestamps_sha256", "test_archive_sha256"):
                self.assertEqual(len({r[key] for r in subset}), 1)

    def test_model_frozen_and_no_optimizer_no_test_inference(self):
        for folder in (self.smoke, self.train, self.full):
            validation = read_json(folder / "validation.json")
            self.assertEqual(validation["parameter_hash_before"], validation["parameter_hash_after"])
            self.assertEqual(validation["optimizer_steps"], 0)
            self.assertIs(validation["model_retrained"], False)
            self.assertIs(validation["test_inference_executed"], False)
        self.assertEqual(read_json(self.full / "test_fit_calls.json")["inference_lengths"], [])

    def test_global_fit_all_28_train_streams_natural_weighting_and_one_scalar(self):
        global_info = read_json(self.full / "global_threshold.json")
        manifest = global_info["provenance"]["train_score_manifest"]
        arrays = []
        for item in manifest:
            with np.load(item["scores_path"]) as saved:
                arrays.append(saved["scores"])
                np.testing.assert_array_equal(saved["timestamps"], np.arange(len(saved["scores"])))
                self.assertEqual(set(saved.files), {"scores", "timestamps"})
        pooled = np.concatenate(arrays)
        calls = read_json(self.full / "test_fit_calls.json")["pot_inputs"]
        self.assertEqual(len(calls), 1+28*4)
        np.testing.assert_array_equal(calls[0], pooled)
        self.assertEqual(global_info["score_count"], sum(40+i%3 for i in range(28)))
        self.assertEqual(global_info["machine_count"], 28)
        self.assertTrue(global_info["natural_weighting"])
        self.assertEqual(len({r["threshold"] for r in self.rows if r["condition"] == "global"}), 1)
        self.assertEqual({r["threshold"] for r in self.rows if r["condition"] == "global"}, {global_info["threshold"]})

    def test_local_05_10_20_and_full_exact_prefix_counts_and_pot_inputs(self):
        from src.calibration_protocol import prefix_length
        global_info = read_json(self.full / "global_threshold.json")
        calls = read_json(self.full / "test_fit_calls.json")["pot_inputs"]
        for index, item in enumerate(global_info["provenance"]["train_score_manifest"]):
            with np.load(item["scores_path"]) as saved:
                scores = saved["scores"]
            for j, condition in enumerate(CONDITIONS[1:]):
                count = len(scores) if condition == "local_full" else prefix_length(len(scores), {"local_05":".05", "local_10":".10", "local_20":".20"}[condition])
                np.testing.assert_array_equal(calls[1+index*4+j], scores[:count])
                info = read_json(self.full / "targets" / item["machine"] / (condition+"_threshold.json"))
                self.assertEqual(info["score_count"], count)
                self.assertEqual(info["provenance"]["raw_prefix_start"], 0)
                self.assertEqual(info["provenance"]["raw_prefix_stop_exclusive"], count)
                self.assertEqual(info["stream_score_count"], 0)

    def test_continuous_metrics_identical_to_stage2_and_all_conditions(self):
        for row in self.rows:
            stage2 = read_json(self.stage2 / "targets" / row["target"] / "metrics.json")
            for new, old in (("pr_auc", "pr_auc_trapezoidal"), ("average_precision", "average_precision"), ("roc_auc", "roc_auc")):
                self.assertAlmostEqual(row[new], stage2[old], delta=1e-12)

    def test_saved_raw_predictions_strict_greater_and_no_pa(self):
        for row in self.rows:
            with np.load(self.stage2 / "targets" / row["target"] / "scores.npz") as source:
                expected = source["scores"] > row["threshold"]
                raw, _ = raw_metrics(source["scores"], source["labels"], row["threshold"])
            with np.load(self.full / "targets" / row["target"] / (row["condition"]+"_predictions.npz")) as saved:
                np.testing.assert_array_equal(saved["predictions"], expected)
            self.assertEqual(raw["f1"], row["raw_f1"])
        self.assertFalse(list(self.full.rglob("*adjusted*")))

    def test_raw_adapter_matches_original_validated_function_and_prefix_scores_equivalence(self):
        code = '''
from pathlib import Path
import sys
import numpy as np
import torch
from src.paths import Roots
from src.universal_training import configuration, core_args, import_core
from src.calibration_runtime import load_frozen_model, score_prefix
from src.calibration_protocol import prefix_length
from src.universal_threshold_analysis import raw_metrics
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
core = import_core(core_args(roots, configuration(roots.project, 1), roots.results/"probe", Path(sys.argv[4])))
model, _ = load_frozen_model(core, sys.argv[4])
from src.experiment import additional_metrics
scores, labels = np.array([.1,.2,.2,.3,.4]), np.array([0,1,0,1,1])
for threshold in (.1,.2, .3, 100.):
    legacy = dict.fromkeys(("precision","recall","f1","TP","TN","FP","FN"), 0)
    original, original_pred = additional_metrics(scores, labels, threshold, legacy)
    new, pred = raw_metrics(scores, labels, threshold)
    assert original["raw"] == new
    np.testing.assert_array_equal(original_pred, pred)
for count in (40, 2600):
    raw = np.arange(count*38, dtype=np.float64).reshape(count,38)/(count*100)
    _, full = score_prefix(core, model, raw)
    full_windows = core.convert_to_windows(torch.from_numpy(raw), model)
    for fraction in (".05", ".10", ".20"):
        n = prefix_length(count, fraction)
        prefix_windows = core.convert_to_windows(torch.from_numpy(raw[:n]), model)
        assert torch.equal(prefix_windows, full_windows[:n])
        _, prefix_scores = score_prefix(core, model, raw[:n])
        np.testing.assert_allclose(prefix_scores, full[:n], atol=1e-12, rtol=0)
'''
        result = python(code, ROOT, self.roots.data, self.roots.results, self.checkpoint)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_original_spot_helper_receives_only_train_empty_stream_fixed_parameters(self):
        code = '''
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from src.paths import Roots
from src.universal_training import configuration, core_args, import_core
from src.universal_thresholding import fit_train_threshold
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
core = import_core(core_args(roots, configuration(roots.project,1), roots.results/"probe", Path(sys.argv[4])))
seen=[]
class Spot:
    def __init__(self,q): assert q==1e-5
    def fit(self,scores,stream):
        assert len(stream)==0
        np.testing.assert_array_equal(scores,np.arange(20,dtype=np.float64))
        seen.append(True)
    def initialize(self,level,min_extrema,verbose):
        assert level==.99995
        self.extreme_quantile=2.
with patch("src.cross_machine.SPOT",Spot):
    out=fit_train_threshold(core,np.arange(20,dtype=np.float64),dict(split="train",seed=1,sources=["machine-1-1"],score_count=20))
assert out["threshold"]==2.*1.06 and out["stream_score_count"]==0 and len(seen)==1
'''
        result = python(code, ROOT, self.roots.data, self.roots.results, self.checkpoint)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_wrong_test_provenance_cannot_enter_pot(self):
        with patch("src.calibration_runtime.fit_threshold", side_effect=AssertionError("不得到達 POT")):
            with self.assertRaises(ValueError):
                fit_train_threshold(None, np.ones(10), dict(split="test", seed=1, sources=["machine-1-1"], score_count=10))
            with self.assertRaises(ValueError):
                fit_train_threshold(None, np.ones(10), dict(split="train", seed=1, sources=["machine-1-1"],
                    score_count=10, condition="global"))

    def test_per_machine_comparison_reads_validated_raw_baselines_and_pairs_all_conditions(self):
        from src.universal_threshold_analysis import comparisons
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            reference = []
            for target in self.inputs["targets"]:
                stage2meta = read_json(self.stage2 / "targets" / target / "metadata.json")
                description = stage2meta["test_input"]
                for setting in ("per_machine", "cross_machine_zero_shot"):
                    name = setting+"-"+target
                    folder = root / "runs" / name
                    folder.mkdir(parents=True)
                    checkpoint = root / "checkpoints" / name / "model.ckpt"
                    sources = [target] if setting == "per_machine" else [t for t in self.inputs["targets"] if t != target]
                    meta = dict(target_machine=target, setting=setting, seed=1, protocol="v2", epochs=5,
                        batch_size=128, eval_batch_size=128, window=10, learning_rate=1e-4, device="cpu",
                        dtype="float64", training_shuffle=True, smoke_test=False, source_machines=sources,
                        label_source="corrected_interpretation_label_union", checkpoint=str(checkpoint),
                        test_timestamps=stage2meta["num_test_points"], data_sha256={description["test_path"]:description["test_sha256"],
                        description["label_path"]:description["label_file_sha256"]})
                    raw = dict(precision=.5, recall=.4, f1=2*.5*.4/(.5+.4))
                    (folder / "metadata.json").write_text(json.dumps(meta))
                    (folder / "evaluation.json").write_text(json.dumps(dict(raw=raw, continuous_score=dict(pr_auc_trapezoidal=.4))))
                    reference.append(dict(target_machine=target, setting=setting, seed=1, source_machines=";".join(sources),
                        checkpoint_path=str(checkpoint), test_points=stage2meta["num_test_points"], pr_auc=.4,
                        raw_precision=raw["precision"], raw_recall=raw["recall"], raw_f1=raw["f1"]))
            summary = root / "formal.csv"
            def save():
                with summary.open("w", newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(reference[0]))
                    writer.writeheader();writer.writerows(reference)
            save()
            before = {str(p):file_hash(p) for p in root.rglob("*") if p.is_file()}
            paired, gaps, deployments, protected = comparisons(self.roots,self.inputs,self.rows,summary)
            self.assertEqual(len(paired),140)
            self.assertEqual(len(gaps),5)
            self.assertEqual(len(deployments),2)
            self.assertEqual({g["condition"] for g in gaps},set(CONDITIONS))
            for row in paired:
                self.assertAlmostEqual(row["per_machine_minus_universal"],2*.5*.4/(.5+.4)-row["universal_raw_f1"])
            for gap in gaps:
                self.assertEqual(gap["universal_better"]+gap["per_machine_better"]+gap["ties"],28)
            self.assertEqual(before,{str(p):file_hash(p) for p in root.rglob("*") if p.is_file()})
            reference[0]["raw_f1"] = .9
            save()
            with self.assertRaises(ValueError):comparisons(self.roots,self.inputs,self.rows,summary)

    def test_namespace_success_no_overwrite_and_full_smoke_gate(self):
        root = root_path(self.roots, self.inputs)
        self.assertTrue(self.full.is_relative_to(root))
        self.assertFalse(self.full.is_relative_to(self.stage2))
        self.assertFalse(self.full.is_relative_to(self.stage1))
        with self.assertRaises(FileExistsError):
            execute(self.roots, phase="full")
        with self.assertRaises(FileExistsError):
            execute(self.roots, phase="smoke", retry=True)
        with patch("src.universal_thresholding.smoke_gate", side_effect=ValueError("smoke missing")):
            with self.assertRaisesRegex(ValueError,"smoke missing"):
                execute(self.roots, phase="full")

    def test_macro_recovery_groups_all_conditions_and_no_undefined_json(self):
        macros, recovery, groups = aggregate(self.rows, self.inputs["targets"])
        self.assertEqual(len(macros), 5)
        self.assertEqual(len(recovery), 4)
        self.assertEqual(len(groups), 15)
        for row in recovery:
            self.assertEqual(sum(row[k] for k in ("improved_vs_global", "degraded_vs_global", "unchanged_vs_global")), 28)
        for path in self.full.rglob("*.json"):
            content = path.read_text()
            self.assertNotIn("NaN",content)
            self.assertNotIn("Infinity",content)
        from src.target_calibration import threshold_change
        result = threshold_change(0.,1.)
        self.assertIsNone(result["relative_ratio"])
        self.assertEqual(result["relative_ratio_status"],"zero_shot_threshold_near_zero")

    def test_bad_140_rows_or_metric_drift_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows(self.rows[:-1], self.inputs["targets"], self.inputs["checkpoint_sha256"])
        rows = [dict(r) for r in self.rows]
        rows[1]["pr_auc"] += .1
        with self.assertRaises(ValueError):
            validate_rows(rows, self.inputs["targets"], self.inputs["checkpoint_sha256"])

    def test_stage2_archive_mutation_rejected_without_inference_fallback(self):
        path = self.stage2 / "targets/machine-1-1/scores.npz"
        original = path.read_bytes()
        try:
            with path.open("ab") as stream: stream.write(b"tamper")
            with self.assertRaises(ValueError): resolve_inputs(self.roots)
        finally:
            path.write_bytes(original)

    def test_cli_dry_run_no_output_no_model_and_seed2_rejection(self):
        command = [sys.executable,"scripts/run_universal_thresholding.py","--phase","smoke","--dry-run",
            "--data-root",str(self.roots.data),"--results-root",str(self.roots.results)]
        before = {str(p):file_hash(p) for p in root_path(self.roots,self.inputs).rglob("*") if p.is_file()}
        result = subprocess.run(command,cwd=ROOT,env=env(),capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        description=json.loads(result.stdout)
        self.assertEqual(description["inputs"]["checkpoint_sha256"],file_hash(self.checkpoint))
        self.assertEqual(description["sources"],["machine-1-1"])
        self.assertEqual(before,{str(p):file_hash(p) for p in root_path(self.roots,self.inputs).rglob("*") if p.is_file()})
        with self.assertRaises(ValueError):execute(self.roots,seed=2)

    def test_old_scientific_stage1_stage2_code_unchanged(self):
        for name in (*PROTECTED,"src/universal_training.py","src/universal_evaluation.py",
                     "src/universal_evaluation_analysis.py","src/calibration_runtime.py"):
            self.assertEqual(subprocess.check_output(["git","show","HEAD:"+name],cwd=ROOT),(ROOT/name).read_bytes(),name)


if __name__ == "__main__":
    unittest.main()
