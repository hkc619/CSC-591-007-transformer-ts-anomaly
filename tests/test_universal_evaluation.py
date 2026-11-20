"""Stage 2 synthetic regression：只有 frozen inference，不執行 training / threshold / calibration。"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.paths import Roots
from src.seed_replication import PROTECTED, file_hash, read_json, write_json
from src.universal_training import configuration, run_identity, update_status
from src.universal_evaluation import (EVALUATION_SOURCES, claim_evaluation, describe_test,
    evaluation_root, execute, resolve_stage1)
from src.universal_evaluation_analysis import formal_comparison, summarize, validate_target

ROOT = Path(__file__).resolve().parents[1]


def env():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg",
                MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / "tranad-universal-eval-tests-mpl"))


def python(code, *arguments):
    return subprocess.run([sys.executable, "-c", code, *map(str, arguments)], cwd=ROOT,
                          env=env(), text=True, capture_output=True, timeout=180)


def create_fixture(base):
    """純 synthetic Stage-1 schema / random model fixture，並非已訓練研究 checkpoint。"""
    roots = Roots(ROOT, base / "SMD", base / "results")
    config = configuration(ROOT, 1)
    run = roots.results / "universal/runs" / run_identity(1) / "attempt-fixture"
    run.mkdir(parents=True)
    checkpoint = roots.results / "universal/checkpoints" / run_identity(1) / run.name / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    records, offset = [], 0
    for i, target in enumerate(config["sources"]):
        for split in ("test", "interpretation_label"):
            (roots.data / split).mkdir(parents=True, exist_ok=True)
        count = 6 + i % 3
        values = i/28 + np.arange(count)[:, None]*0.003 + np.arange(38)[None, :]*0.001
        np.savetxt(roots.data / "test" / (target+".txt"), values, delimiter=",")
        (roots.data / "interpretation_label" / (target+".txt")).write_text("1-3:1,2\n")
        records.append(dict(machine=target, raw_train_length=count, window_count=count,
            offset_start=offset, offset_end_exclusive=offset+count, dropped_points=0,
            windowed_before_concatenation=True, all_windows_single_machine=True))
        offset += count
    provenance = dict(machine_count=28, total_training_windows=offset, machines=records,
        window_size=10, all_windows_single_machine=True, windowed_before_concatenation=True,
        raw_concatenation_before_windowing=False)
    for name in ("train_manifest.json", "window_provenance.json"):
        write_json(run / name, provenance)
    hashes = {}
    for name in PROTECTED:
        destination = run / "source_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        hashes[name] = file_hash(destination)
    meta = dict(config, run_id=run_identity(1), attempt_id=run.name, machine_count=28,
                test_loaded=False, checkpoint_path=str(checkpoint), total_training_windows=offset,
                source_file_sha256=hashes, synthetic_test_fixture=True)
    write_json(run / "metadata.json", meta)
    write_json(run / "launch.json", {"synthetic_test_fixture": True})
    write_json(run / "training_history.json", {"synthetic_test_fixture": True})
    code = '''
import sys
from pathlib import Path
import torch
from src.paths import Roots
from src.universal_training import configuration, core_args, import_core
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
args = core_args(roots, configuration(roots.project, 1), Path(sys.argv[4]), Path(sys.argv[5]))
core = import_core(args)
model = core.TranAD(38).double().cpu()
# Random-state fixture only：沒有建立 optimizer、沒有 training、沒有 backward。
with Path(sys.argv[5]).open("xb") as stream:
    torch.save(dict(epoch=4, model_state_dict=model.state_dict(), optimizer_state_dict={},
                    scheduler_state_dict={}, accuracy_list=[(0., 0.)]*5), stream)
'''
    result = python(code, ROOT, roots.data, roots.results, run, checkpoint)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    validation = dict(checkpoint_reload_passed=True, model_state_equal=True, optimizer_state_equal=True,
        scheduler_state_equal=True, epoch_history_equal=True, all_windows_single_machine=True,
        seed=1, machine_count=28, epochs_completed=5, total_training_windows=offset,
        checkpoint_sha256=file_hash(checkpoint), checkpoint_size_bytes=checkpoint.stat().st_size)
    write_json(run / "validation.json", validation)
    write_json(run / "status.json", dict(state="completed", completed=True, seed=1,
        checkpoint=str(checkpoint), validation=validation))
    return roots, run, checkpoint


def guard_evaluation(roots, mode):
    code = '''
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from src.paths import Roots
import src.universal_evaluation as evaluation
from src.calibration_runtime import load_frozen_model
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
original_import = evaluation.import_core
def forbidden(*args, **kwargs):
    raise AssertionError("training / optimizer / threshold / calibration 不得執行")
def guarded_import(args):
    core = original_import(args)
    import src.cross_machine as cm
    import preprocess
    original_loader = preprocess.load_and_save
    def test_only(category, *args, **kwargs):
        assert category == "test", category
        return original_loader(category, *args, **kwargs)
    preprocess.load_and_save = test_only
    cm.load_training_machine = cm.fit_source_threshold = cm.run_smd_experiment = forbidden
    core.load_model = core.save_model = forbidden
    original_backprop = core.backprop
    def infer(*args, **kwargs):
        assert kwargs.get("training") is False
        assert args[4] is None and args[5] is None
        return original_backprop(*args, **kwargs)
    core.backprop = infer
    return core
with patch.object(evaluation, "import_core", side_effect=guarded_import), \
     patch.object(torch.optim.AdamW, "__init__", side_effect=forbidden), \
     patch.object(torch.optim.Adam, "__init__", side_effect=forbidden), \
     patch.object(torch.Tensor, "backward", side_effect=forbidden), \
     patch("src.calibration_runtime.fit_threshold", side_effect=forbidden), \
     patch("src.calibration_runtime.evaluate", side_effect=forbidden), \
     patch("src.calibration_runtime.load_frozen_model", wraps=load_frozen_model) as load:
    mode = sys.argv[4]
    folder = evaluation.execute(roots, target="machine-1-1" if mode == "smoke" else None,
                                all_targets=mode == "full")
    assert load.call_count == 1, load.call_count
print("EVALUATION="+str(folder))
'''
    result = python(code, ROOT, roots.data, roots.results, mode)
    if result.returncode:
        raise AssertionError(result.stdout+result.stderr)
    return Path(next(line.split("=", 1)[1] for line in result.stdout.splitlines() if line.startswith("EVALUATION=")))


class UniversalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.roots, cls.stage1, cls.checkpoint = create_fixture(Path(cls.temp.name).resolve())
        cls.bundle = resolve_stage1(cls.roots)
        cls.stage1_before = {str(p): file_hash(p) for p in cls.stage1.rglob("*") if p.is_file()}
        cls.stage1_before[str(cls.checkpoint)] = file_hash(cls.checkpoint)
        cls.smoke = guard_evaluation(cls.roots, "smoke")
        cls.full = guard_evaluation(cls.roots, "full")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_28_canonical_targets_and_single_checkpoint_hash(self):
        folders = list((self.full / "targets").iterdir())
        canonical = configuration(ROOT, 1)["sources"]
        self.assertEqual({p.name for p in folders}, set(canonical))
        self.assertEqual(len(folders), 28)
        rows = [validate_target(self.full / "targets" / m, m, self.bundle) for m in canonical]
        self.assertEqual({r["checkpoint_sha256"] for r in rows}, {file_hash(self.checkpoint)})
        self.assertEqual({r["seed"] for r in rows}, {1})

    def test_frozen_no_optimizer_and_stage1_inputs_unchanged(self):
        for path, digest in self.stage1_before.items():
            self.assertEqual(file_hash(path), digest, path)
        validation = read_json(self.full / "validation.json")
        self.assertEqual(validation["parameter_hash_before"], validation["parameter_hash_after"])
        self.assertEqual(validation["unique_universal_checkpoint_hashes"], 1)
        self.assertEqual(validation["optimizer_steps"], 0)
        self.assertIs(validation["model_retrained"], False)
        self.assertTrue(read_json(self.full / "status.json")["completed"])

    def test_machine_isolation_alignment_finite_scores_labels_and_metrics(self):
        for i, target in enumerate(configuration(ROOT, 1)["sources"]):
            folder = self.full / "targets" / target
            with np.load(folder / "scores.npz", allow_pickle=False) as archive:
                self.assertEqual(set(archive.files), {"scores", "labels", "timestamps"})
                self.assertEqual(len(archive["scores"]), 6+i%3)
                self.assertEqual(archive["scores"].shape, archive["labels"].shape)
                self.assertTrue(np.isfinite(archive["scores"]).all())
                self.assertEqual(archive["scores"].dtype, np.float64)
                np.testing.assert_array_equal(archive["timestamps"], np.arange(6+i%3))
                np.testing.assert_array_equal(archive["labels"], [0, 1, 1]+[0]*(3+i%3))
            val = read_json(folder / "validation.json")
            self.assertTrue(val["all_windows_single_machine"])
            meta = read_json(folder / "metadata.json")
            self.assertEqual(Path(meta["test_input"]["test_path"]).name, target+".txt")
            metrics = read_json(folder / "metrics.json")
            for key in ("pr_auc_trapezoidal", "average_precision", "roc_auc"):
                self.assertTrue(0 <= metrics[key] <= 1)

    def test_no_threshold_calibration_or_checkpoint_outputs(self):
        for folder in (self.smoke, self.full):
            self.assertFalse(list(folder.rglob("*threshold*")))
            self.assertFalse(list(folder.rglob("*calibrat*")))
            self.assertFalse(list(folder.rglob("*.ckpt")))
            for path in (folder / "targets").glob("*/metrics.json"):
                self.assertEqual(set(read_json(path)), {"pr_auc_trapezoidal", "average_precision", "roc_auc",
                                                     "anomaly_prevalence", "pr_auc_minus_prevalence"})

    def test_same_model_test_input_gives_identical_smoke_and_full_scores(self):
        target = configuration(ROOT, 1)["sources"][0]
        with np.load(self.smoke / "targets" / target / "scores.npz") as smoke:
            with np.load(self.full / "targets" / target / "scores.npz") as full:
                for key in ("scores", "labels", "timestamps"):
                    np.testing.assert_array_equal(smoke[key], full[key])

    def test_summary_exactly_28_unique_targets_macro_std_and_prevalence(self):
        with (self.full / "summary/target_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 28)
        self.assertEqual(len({r["target"] for r in rows}), 28)
        summary = read_json(self.full / "summary/summary.json")
        values = [float(r["pr_auc"]) for r in rows]
        self.assertEqual(summary["mean_pr_auc"], np.mean(values))
        self.assertEqual(summary["sample_std_pr_auc"], np.std(values, ddof=1))
        self.assertEqual(summary["unique_universal_checkpoint_hashes"], 1)
        for row in rows:
            self.assertAlmostEqual(float(row["anomaly_prevalence"]), int(row["anomaly_points"])/int(row["test_points"]))
            self.assertAlmostEqual(float(row["pr_auc_minus_prevalence"]), float(row["pr_auc"])-float(row["anomaly_prevalence"]))
        self.assertFalse((self.smoke / "summary").exists())

    def test_dry_run_no_model_import_or_output_writes_and_seed2_rejected(self):
        before = {str(p): file_hash(p) for p in evaluation_root(self.roots.results, self.bundle).rglob("*") if p.is_file()}
        code = '''
import sys
from pathlib import Path
from src.paths import Roots
from src.universal_evaluation import execute
execute(Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), target="machine-1-1", dry_run=True)
assert "torch" not in sys.modules and "main" not in sys.modules
'''
        result = python(code, ROOT, self.roots.data, self.roots.results)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        description = json.loads(result.stdout)
        self.assertEqual(description["stage1"]["checkpoint_sha256"], file_hash(self.checkpoint))
        self.assertEqual(description["targets"], ["machine-1-1"])
        self.assertEqual(before, {str(p): file_hash(p) for p in evaluation_root(self.roots.results, self.bundle).rglob("*") if p.is_file()})
        with self.assertRaises(ValueError):
            execute(self.roots, target="machine-1-1", seed=2)
        cli = subprocess.run([sys.executable, "scripts/run_universal_evaluation.py", "--target", "machine-1-1",
            "--data-root", str(self.roots.data), "--results-root", str(self.roots.results), "--dry-run"],
            cwd=ROOT, env=env(), capture_output=True, text=True, timeout=30)
        self.assertEqual(cli.returncode, 0, cli.stderr)

    def test_namespace_and_completed_attempts_cannot_be_overwritten(self):
        root = evaluation_root(self.roots.results, self.bundle)
        self.assertTrue(self.smoke.is_relative_to(root / "smoke"))
        self.assertTrue(self.full.is_relative_to(root / "full"))
        self.assertFalse(self.full.is_relative_to(self.stage1))
        for mode in ("smoke", "full"):
            for retry in (False, True):
                with self.assertRaises(FileExistsError):
                    claim_evaluation(self.roots.results, self.bundle, mode, retry)

    def test_checkpoint_and_incomplete_or_wrong_seed_stage1_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            new_results = base / "results"
            shutil.copytree(self.roots.results / "universal", new_results / "universal",
                            ignore=shutil.ignore_patterns("evaluation"))
            # 原 artifact 的 absolute path 不可被 silent remap。
            with self.assertRaises(ValueError):
                resolve_stage1(Roots(ROOT, self.roots.data, new_results))
        path = self.stage1 / "metadata.json"
        original = path.read_bytes()
        try:
            metadata = read_json(path)
            metadata["seed"] = 2
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                resolve_stage1(self.roots)
        finally:
            path.write_bytes(original)
        path = self.stage1 / "status.json"
        original = path.read_bytes()
        try:
            status = read_json(path)
            status.update(state="running", completed=False)
            path.write_text(json.dumps(status))
            with self.assertRaises(ValueError):
                resolve_stage1(self.roots, self.stage1)
        finally:
            path.write_bytes(original)
        original = self.checkpoint.read_bytes()
        try:
            with self.checkpoint.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                resolve_stage1(self.roots)
        finally:
            self.checkpoint.write_bytes(original)

    def test_full_requires_successful_matching_smoke_before_claim(self):
        from unittest.mock import patch
        with patch("src.universal_evaluation.passed_smoke", side_effect=ValueError("smoke missing")):
            with self.assertRaisesRegex(ValueError, "smoke missing"):
                execute(self.roots, all_targets=True)
        self.assertEqual(len(list((evaluation_root(self.roots.results, self.bundle)/"full").glob("attempt-*"))), 1)

    def test_tampered_scores_wrong_hash_and_partial_summary_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            target = configuration(ROOT, 1)["sources"][0]
            folder = Path(temp).resolve() / "target"
            shutil.copytree(self.full / "targets" / target, folder)
            path = folder / "metadata.json"
            metadata = read_json(path)
            metadata["checkpoint_sha256"] = "0"*64
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                validate_target(folder, target, self.bundle)
            shutil.copyfile(self.full / "targets" / target / "metadata.json", path)
            with (folder / "scores.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                validate_target(folder, target, self.bundle)
        with self.assertRaises(ValueError):
            summarize(self.roots, self.bundle, self.smoke, ["machine-1-1"])

    def test_nonfinite_and_misaligned_scores_fail_even_with_new_archive_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            target = configuration(ROOT, 1)["sources"][0]
            folder = Path(temp).resolve() / "target"
            shutil.copytree(self.full / "targets" / target, folder)
            with np.load(folder / "scores.npz") as archive:
                scores, labels, timestamps = [archive[k].copy() for k in ("scores", "labels", "timestamps")]
            for invalid in (np.full_like(scores, np.nan), scores[:-1]):
                with (folder / "scores.npz").open("wb") as stream:
                    np.savez_compressed(stream, scores=invalid, labels=labels, timestamps=timestamps)
                validation = read_json(folder / "validation.json")
                validation["scores_npz_sha256"] = file_hash(folder / "scores.npz")
                (folder / "validation.json").write_text(json.dumps(validation))
                with self.assertRaisesRegex(ValueError, "Scores / labels"):
                    validate_target(folder, target, self.bundle)

    def test_seed1_formal_comparison_read_only_metadata_and_test_hash_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            rows = []
            canonical = configuration(ROOT, 1)["sources"]
            for i, target in enumerate(canonical):
                description = describe_test(self.roots, target)
                for setting in ("per_machine", "cross_machine_zero_shot"):
                    name = setting+"-"+target
                    run = root / "runs" / name
                    run.mkdir(parents=True)
                    checkpoint = root / "checkpoints" / name / "model.ckpt"
                    sources = [target] if setting == "per_machine" else [m for m in canonical if m != target]
                    value = (.4 if setting == "per_machine" else .2)+i*.001
                    meta = dict(target_machine=target, setting=setting, seed=1, protocol="v2", epochs=5,
                        batch_size=128, eval_batch_size=128, window=10, learning_rate=1e-4, device="cpu",
                        dtype="float64", training_shuffle=True, smoke_test=False,
                        label_source="corrected_interpretation_label_union", checkpoint=str(checkpoint),
                        source_machines=sources, test_timestamps=description["raw_test_length"],
                        data_sha256={description["test_path"]: description["test_sha256"],
                                     description["label_path"]: description["label_file_sha256"]})
                    write_json(run / "metadata.json", meta)
                    write_json(run / "evaluation.json", dict(continuous_score=dict(pr_auc_trapezoidal=value)))
                    rows.append(dict(target_machine=target, setting=setting, seed="1", checkpoint_path=str(checkpoint),
                        source_machines=";".join(sources), pr_auc=value, test_points=description["raw_test_length"]))
            summary = root / "formal.csv"
            def write_rows():
                with summary.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            write_rows()
            before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
            folders = {t: self.full / "targets" / t for t in canonical}
            universal = [validate_target(folders[t], t, self.bundle) for t in canonical]
            comparison, macro, provenance = formal_comparison(self.roots, universal, summary, folders)
            self.assertEqual(len(comparison), 28)
            self.assertEqual(len(macro), 3)
            self.assertAlmostEqual(macro[0]["mean_pr_auc"], np.mean([.4+i*.001 for i in range(28)]))
            self.assertEqual(before, {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()})
            self.assertEqual(before, provenance)
            rows[0]["seed"] = "2"
            write_rows()
            with self.assertRaises(ValueError):
                formal_comparison(self.roots, universal, summary, folders)
            rows[0]["seed"] = "1"
            write_rows()
            path = root / "runs" / ("per_machine-"+canonical[0]) / "metadata.json"
            meta = read_json(path)
            meta["data_sha256"][describe_test(self.roots, canonical[0])["label_path"]] = "0"*64
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                formal_comparison(self.roots, universal, summary, folders)

    def test_original_scientific_and_stage1_files_unchanged(self):
        for name in (*PROTECTED, "src/universal_training.py", "configs/universal_training.json"):
            committed = subprocess.check_output(["git", "show", "HEAD:"+name], cwd=ROOT)
            self.assertEqual(committed, (ROOT/name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
