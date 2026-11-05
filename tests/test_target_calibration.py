"""Synthetic runtime / artifact tests；不使用正式 SMD 或正式 checkpoint。"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.calibration_protocol import load_configuration
from src.calibration_runtime import (baseline_runtime, evaluate, fit_threshold,
                                     freeze_model, load_frozen_model, score_prefix, state_hash)
from src.paths import Roots
from src.target_calibration import (check_auc_invariance, execute_calibration, file_hash,
                                    inspect_experiment, inspect_prefix, load_prefix,
                                    load_formal_scores, select_formal_run, threshold_change)

ROOT = Path(__file__).resolve().parents[1]


class CalibrationRuntimeTests(unittest.TestCase):
    def test_prefix_loader_never_parses_train_suffix_or_test(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.txt"
            first = np.arange(10 * 38, dtype=np.float64).reshape(10, 38)
            np.savetxt(path, first, delimiter=",")
            with path.open("a") as stream:
                stream.write("suffix_must_not_be_parsed\n" * 190)
            selection = inspect_prefix(path, "0.05")
            self.assertEqual(selection["prefix_length"], 10)
            np.testing.assert_array_equal(load_prefix(path, 10), first)

    def test_actual_frozen_tranad_scores_repeat_without_optimizer(self):
        with baseline_runtime() as core, tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "model.ckpt"
            original = core.TranAD(38).double()
            torch.save(dict(epoch=4, model_state_dict=original.state_dict()), checkpoint)
            with patch.object(core, "load_model", side_effect=AssertionError("不可建立 optimizer")), \
                 patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("不可建立 optimizer")), \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("不可 backward")):
                model, before = load_frozen_model(core, checkpoint)
                raw = np.random.default_rng(1).random((12, 38))
                first_errors, first = score_prefix(core, model, raw)
                second_errors, second = score_prefix(core, model, raw)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first_errors, second_errors)
            self.assertEqual(state_hash(model), before)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_pot_receives_only_calibration_scores_and_empty_stream(self):
        seen = {}
        prefix_scores = np.linspace(0.01, 0.1, 100)

        class FakeSPOT:
            def __init__(self, q):
                seen["q"] = q

            def fit(self, initial, stream):
                seen["initial"] = initial.copy()
                seen["stream"] = stream.copy()

            def initialize(self, level, min_extrema, verbose):
                seen["level"] = level
                self.extreme_quantile = 0.4

            def run(self, *args, **kwargs):
                raise AssertionError("不可執行 target TEST stream")

        with baseline_runtime() as core:
            with patch("src.cross_machine.SPOT", FakeSPOT):
                threshold, info = fit_threshold(core, prefix_scores)
        np.testing.assert_array_equal(seen["initial"], prefix_scores)
        self.assertEqual(seen["stream"].size, 0)
        self.assertEqual((seen["q"], seen["level"]), (1e-5, 0.99995))
        self.assertEqual(threshold, 0.4 * 1.06)
        self.assertEqual(info["stream_score_count"], 0)

    def test_only_threshold_changes_not_scores_labels_or_auc(self):
        scores = np.array([0.1, 0.4, 0.8, 0.2, 0.6, 0.9], dtype=np.float64)
        labels = np.array([0, 1, 1, 0, 0, 1])
        score_copy, label_copy = scores.copy(), labels.copy()
        with baseline_runtime() as core:
            baseline, raw_before, _ = evaluate(core, scores, labels, 0.85)
            calibrated, raw_after, _ = evaluate(core, scores, labels, 0.35)
        self.assertFalse(np.array_equal(raw_before, raw_after))
        np.testing.assert_array_equal(scores, score_copy)
        np.testing.assert_array_equal(labels, label_copy)
        check_auc_invariance(baseline, calibrated, 0)
        self.assertNotEqual(baseline["raw"]["recall"], calibrated["raw"]["recall"])

    def test_archive_alignment_rejects_shifted_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.npz"
            errors = np.arange(12, dtype=np.float64).reshape(12, 1).repeat(38, axis=1)
            labels = np.zeros_like(errors)
            np.savez(path, scores=np.roll(errors.mean(axis=1), 1), test_scores=errors,
                     labels=labels, labels_final=labels.any(axis=1).astype(int))
            with self.assertRaises(AssertionError):
                load_formal_scores(path, 12)

    def test_threshold_ratio_near_zero_is_json_null_not_inf(self):
        for zero in (0.0, 1e-14):
            result = threshold_change(zero, .5)
            self.assertIsNone(result["relative_ratio"])
            json.dumps(result, allow_nan=False)
        self.assertEqual(threshold_change(.5, .25)["relative_ratio"], .5)
        self.assertEqual(threshold_change(.5, .25)["absolute_shift"], .25)

    def test_checkpoint_with_wrong_model_state_rejected(self):
        with baseline_runtime() as core, tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "wrong_model.ckpt"
            torch.save(dict(epoch=4, model_state_dict=torch.nn.Linear(38, 38).double().state_dict()), checkpoint)
            with self.assertRaises(RuntimeError):
                load_frozen_model(core, checkpoint)

    def test_continuous_metric_drift_is_rejected(self):
        baseline = dict(continuous_score=dict(pr_auc_trapezoidal=.5, average_precision=.5, roc_auc=.5))
        for key in baseline["continuous_score"]:
            after = dict(continuous_score=dict(baseline["continuous_score"]))
            after["continuous_score"][key] += .001
            with self.assertRaises(AssertionError):
                check_auc_invariance(baseline, after, 1e-12)

    def test_short_raw_prefix_uses_original_padding_and_one_window_per_point(self):
        with baseline_runtime() as core:
            model = core.TranAD(38).double()
            raw = torch.arange(5, dtype=torch.float64)[:, None].repeat(1, 38)
            windows = core.convert_to_windows(raw, model)
        self.assertEqual(tuple(windows.shape), (5, 10, 38))
        torch.testing.assert_close(windows[:, -1], raw, rtol=0, atol=0)
        self.assertEqual(windows[0, :, 0].tolist(), [0.0] * 10)
        self.assertEqual(windows[-1, :, 0].tolist(), [0.0] * 6 + [1.0, 2.0, 3.0, 4.0])


class CalibrationArtifactTests(unittest.TestCase):
    def create_fixture(self, temp):
        root = Path(temp).resolve()
        results, data = root / "results", root / "data"
        run = results / "runs/formal_fixture"
        run.mkdir(parents=True)
        checkpoint = results / "checkpoints/formal_fixture/model.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fixture-only; model loading mocked")
        (data / "train").mkdir(parents=True)
        train = np.arange(200 * 38, dtype=np.float64).reshape(200, 38) / 10000
        np.savetxt(data / "train/machine-1-1.txt", train, delimiter=",")
        config, machines = load_configuration(ROOT)
        values = np.array([0.1, 0.4, 0.8, 0.2, 0.6, 0.9], dtype=np.float64)
        errors = np.repeat(values[:, None], 38, axis=1)
        scores = errors.mean(axis=1)
        feature_labels = np.zeros((6, 38))
        feature_labels[[1, 2, 5], 0] = 1
        labels = feature_labels.any(axis=1).astype(int)
        with baseline_runtime() as core:
            baseline, _, _ = evaluate(core, scores, labels, 0.85)
        np.savez(run / "scores.npz", scores=scores, labels_final=labels,
                 test_scores=errors, labels=feature_labels)
        meta = dict(setting="cross_machine_zero_shot", target_machine="machine-1-1",
            command=["python", "main.py", "--model", "TranAD", "--dataset", "SMD"],
            source_machines=machines[1:], protocol="v2", seed=1, epochs=5,
            batch_size=128, eval_batch_size=128, device="cpu", dtype="float64", window=10,
            learning_rate=0.0001, training_shuffle=True, smoke_test=False,
            threshold_source="source_machine_training_scores", threshold_value=0.85,
            threshold_protocol="source_only_frozen_SPOT_extreme_quantile",
            label_source="corrected_interpretation_label_union", train_windows=12345,
            checkpoint=str(checkpoint), test_timestamps=6, data_sha256={},
            pot_fit=dict(q=1e-5, requested_level=0.99995, multiplier=1.06,
                         stream_score_count=0, source_score_count=12345),
            leakage_checks=dict(target_train_loaded=False, target_in_optimizer_updates=False,
                target_in_threshold_fitting=False, normalization_fitted=False,
                target_used_for_testing=True, all_windows_single_machine=True,
                target_score_label_alignment=True))
        for name, value in (("metadata.json", meta), ("evaluation.json", baseline),
                            ("threshold.json", dict(threshold_value=0.85))):
            (run / name).write_text(json.dumps(value))
        for name in ("main.py", "src/models.py", "src/dlutils.py", "src/constants.py",
                     "src/pot.py", "src/spot.py", "src/training_protocol.py",
                     "src/cross_machine.py", "src/experiment.py", "src/parser.py"):
            dest = run / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT / name).read_bytes())
        status = results / "task_status/028-fixture.json"
        status.parent.mkdir(parents=True)
        status.write_text(json.dumps(dict(task_id=28, exit_code=0, run_path=str(run))))
        return Roots(ROOT, data, results), config, machines, run, train

    def test_dry_description_does_not_load_model_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, run, _ = self.create_fixture(temp)
            with patch("src.calibration_runtime.load_frozen_model", side_effect=AssertionError("不可載入 model")), \
                 patch("src.target_calibration.load_formal_scores", side_effect=AssertionError("不可讀 TEST scores")):
                description, *_ = inspect_experiment(roots, roots.results, config, machines, "machine-1-1", "0.05")
            self.assertEqual(description["prefix"]["prefix_length"], 10)
            self.assertEqual(description["formal_task_id"], 28)
            self.assertEqual(description["formal_run_id"], run.name)
            self.assertEqual(description["calibration_windows"], 10)
            self.assertFalse((roots.results / "calibration").exists())

    def test_orchestration_isolates_artifacts_and_never_reads_raw_test(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, run, train = self.create_fixture(temp)
            protected = {p: file_hash(p) for p in roots.results.rglob("*") if p.is_file()}
            model = torch.nn.Linear(38, 38).double()
            before = freeze_model(model)
            events = []

            def prefix_scores(core, actual_model, prefix):
                self.assertIs(model, actual_model)
                np.testing.assert_array_equal(prefix, train[:10])
                events.append("prefix")
                errors = np.full((10, 38), 0.15)
                return errors, errors.mean(axis=1)

            def fit(core, scores):
                self.assertEqual(events, ["prefix"])
                self.assertEqual(len(scores), 10)
                events.append("threshold_fixed")
                return 0.35, dict(q=1e-5, requested_level=0.99995, effective_level=0.99995,
                                 multiplier=1.06, source_score_count=10, stream_score_count=0,
                                 initialization_attempts=1)

            original_load = load_formal_scores

            def test_arrays(path, count):
                self.assertEqual(events, ["prefix", "threshold_fixed"])
                events.append("formal_arrays")
                return original_load(path, count)

            # fixture 沒有 raw test 或 labels 目錄；若意外讀取會失敗。
            with patch("src.calibration_runtime.load_frozen_model", return_value=(model, before)), \
                 patch("src.calibration_runtime.score_prefix", side_effect=prefix_scores), \
                 patch("src.calibration_runtime.fit_threshold", side_effect=fit), \
                 patch("src.target_calibration.load_formal_scores", side_effect=test_arrays), \
                 patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("不可建立 optimizer")):
                folder = execute_calibration(roots, roots.results, config, machines,
                    "machine-1-1", "0.05", smoke=True)
            self.assertEqual(folder.parent.name, "smoke-calib-machine-1-1-f005-seed1")
            self.assertEqual(folder.parent.parent, roots.results / "calibration/runs")
            self.assertEqual(events, ["prefix", "threshold_fixed", "formal_arrays"])
            for path, digest in protected.items():
                self.assertEqual(file_hash(path), digest)
            metadata = json.loads((folder / "metadata.json").read_text())
            evaluation = json.loads((folder / "evaluation.json").read_text())
            self.assertEqual(metadata["optimizer_steps"], 0)
            self.assertEqual(metadata["model_state_hash_before"], metadata["model_state_hash_after"])
            self.assertFalse(metadata["model_retrained"])
            self.assertTrue(metadata["parameters_unchanged"])
            self.assertEqual(metadata["formal_task_id"], 28)
            self.assertEqual(metadata["calibration_windows"], 10)
            self.assertEqual(metadata["pot"]["stream_score_count"], 0)
            self.assertIn("raw_f1_delta", evaluation["recovery"])
            check_auc_invariance(evaluation["zero_shot"], evaluation["calibrated"], 0)
            with np.load(folder / "scores.npz") as calibrated, np.load(run / "scores.npz") as original:
                np.testing.assert_array_equal(calibrated["scores"], original["scores"])
                np.testing.assert_array_equal(calibrated["labels_final"], original["labels_final"])

    def test_rejects_baseline_with_target_training_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, run, _ = self.create_fixture(temp)
            path = run / "metadata.json"
            meta = json.loads(path.read_text())
            meta["leakage_checks"]["target_train_loaded"] = True
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                inspect_experiment(roots, roots.results, config, machines, "machine-1-1", "0.05")

    def test_failed_latest_attempt_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, _, machines, run, _ = self.create_fixture(temp)
            status = roots.results / "task_status/028-fixture.json"
            status.write_text(json.dumps(dict(task_id=28, exit_code=1, run_path=str(run))))
            with self.assertRaises(ValueError):
                select_formal_run(roots.results, machines, "machine-1-1")

    def test_wrong_formal_identity_or_missing_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, run, _ = self.create_fixture(temp)
            path = run / "metadata.json"
            original = json.loads(path.read_text())
            variants = [dict(original, **{key: value}) for key, value in (
                ("target_machine", "machine-2-1"), ("setting", "per_machine"),
                ("seed", 2), ("protocol", "v1"), ("source_machines", ["machine-1-1"]),
                ("model", "OtherModel"), ("dataset", "OtherDataset"),
                ("dtype", "float32"), ("device", "cuda"),
                ("checkpoint", str(roots.results / "missing.ckpt")))]
            for value in variants:
                with self.subTest(value=value):
                    path.write_text(json.dumps(value))
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        inspect_experiment(roots, roots.results, config, machines, "machine-1-1", "0.05")

    def test_pre_model_failure_is_recorded_as_failed_calibration_task(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, machines, run, _ = self.create_fixture(temp)
            path = run / "metadata.json"
            meta = json.loads(path.read_text())
            meta["leakage_checks"]["target_train_loaded"] = True
            path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                execute_calibration(roots, roots.results, config, machines,
                                    "machine-1-1", "0.05", task_id=0)
            records = list((roots.results / "calibration/task_status").glob("*.json"))
            self.assertEqual(len(records), 1)
            status = json.loads(records[0].read_text())
            self.assertFalse(status["completed"])
            self.assertEqual(status["exit_code"], 1)
            self.assertEqual(status["task_id"], 0)


if __name__ == "__main__":
    unittest.main()
