"""AT synthetic correctness checks；不執行真實 SMD training 或提交 HPC jobs。"""
from contextlib import redirect_stdout, redirect_stderr
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.anomaly_transformer.dataset import (MachineWindows, assemble_energies, evaluation_plan,
    load_sources, load_values, normalization_metadata, select_sources)
from src.anomaly_transformer.model import AnomalyTransformer
from src.anomaly_transformer.protocol import configuration, namespace, validate_manifest
from src.anomaly_transformer.protocol import logical_id
from src.anomaly_transformer.experiment import write_json
from src.anomaly_transformer.runtime import infer, minimax_step, seed_everything, training_loader
from src.anomaly_transformer.scoring import my_kl_loss, native_energy, training_objectives
from src.anomaly_transformer.experiment import execute, file_hash
from src.continuous_metrics import continuous_metrics
from src.paths import Roots

ROOT = Path(__file__).resolve().parents[1]


def fixture(temp):
    root = Path(temp).resolve()
    data, results = root / "data", root / "results"
    for split in ("train", "test", "interpretation_label"):
        (data / split).mkdir(parents=True)
    config, canonical = configuration(ROOT)
    raw = np.random.default_rng(1).random((205, 38)).astype(np.float32)
    for name in canonical:
        np.savetxt(data / "train" / (name + ".txt"), raw, delimiter=",")
    np.savetxt(data / "test/machine-1-1.txt", np.random.default_rng(2).random((257, 38)), delimiter=",")
    (data / "interpretation_label/machine-1-1.txt").write_text("2-7:1,3\n240-245:2\n")
    return Roots(ROOT, data, results), config, canonical


class ModelAndScoreTests(unittest.TestCase):
    def test_native_forward_batch2_window100_features38(self):
        seed_everything(1)
        config, _ = configuration(ROOT)
        model = AnomalyTransformer(**config["model"]).eval()
        inputs = torch.randn(2, 100, 38)
        with torch.no_grad():
            output, series, priors, sigmas = model(inputs)
            energy = native_energy(inputs, output, series, priors)
        self.assertEqual(tuple(output.shape), (2, 100, 38))
        self.assertEqual(tuple(energy.shape), (2, 100))
        self.assertEqual(len(series), 3)
        for s, p in zip(series, priors):
            self.assertEqual(tuple(s.shape), (2, 8, 100, 100))
            self.assertTrue(torch.isfinite(p).all())
            torch.testing.assert_close(s.sum(-1), torch.ones(2, 8, 100))
        self.assertTrue(torch.isfinite(energy).all())
        self.assertTrue(all(buffer.device.type == "cpu" for buffer in model.buffers()))

    def test_native_minimax_step_has_finite_loss_gradients_and_updates(self):
        seed_everything(1)
        config, _ = configuration(ROOT)
        model = AnomalyTransformer(**config["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = model.projection.weight.detach().clone()
        values = minimax_step(model, torch.randn(2, 100, 38), optimizer)
        self.assertTrue(np.isfinite(values).all())
        self.assertFalse(torch.equal(before, model.projection.weight))

    def test_minimax_detach_and_two_reconstruction_contributions(self):
        torch.manual_seed(1)
        inputs = torch.randn(2, 5, 38)
        outputs = torch.randn(2, 5, 38, requires_grad=True)
        series = torch.softmax(torch.randn(2, 2, 5, 5), -1).detach().requires_grad_()
        prior = torch.rand(2, 2, 5, 5, requires_grad=True)
        loss1, loss2, rec, sl, pl = training_objectives(inputs, outputs, [series], [prior])
        normalized = prior / prior.sum(-1, keepdim=True).repeat(1, 1, 1, 5)
        def official_kl(p, q):
            return torch.mean(torch.sum(p * (torch.log(p + .0001)-torch.log(q + .0001)), -1), 1)
        ref_sl = official_kl(series, normalized.detach()).mean()+official_kl(normalized.detach(), series).mean()
        ref_pl = official_kl(normalized, series.detach()).mean()+official_kl(series.detach(), normalized).mean()
        torch.testing.assert_close(loss1, rec - 3*ref_sl)
        torch.testing.assert_close(loss2, rec + 3*ref_pl)
        gradients = torch.autograd.grad(loss1 + loss2, (outputs, series, prior))
        torch.testing.assert_close(gradients[0], 4*(outputs-inputs)/inputs.numel())

    def test_native_energy_matches_official_layer_sum_temperature(self):
        x, output = torch.randn(2, 100, 38), torch.randn(2, 100, 38)
        series = [torch.softmax(torch.randn(2, 8, 100, 100), -1) for _ in range(3)]
        priors = [torch.rand(2, 8, 100, 100) for _ in range(3)]
        discrepancy = 0
        for s, p in zip(series, priors):
            normalized = p / p.sum(-1, keepdim=True).repeat(1, 1, 1, 100)
            discrepancy = discrepancy + 50*my_kl_loss(s, normalized.detach()) + 50*my_kl_loss(normalized, s.detach())
        expected = torch.softmax(-discrepancy, -1)*torch.mean((x-output)**2, -1)
        torch.testing.assert_close(native_energy(x, output, series, priors), expected)

    def test_metrics_match_validated_tranad_continuous_values(self):
        from src.experiment import additional_metrics
        scores, labels = np.array([.1, .3, .4, .8, .5]), np.array([0, 0, 1, 1, 1])
        # Reference wrapper only inside unit test; AT experiment never calls thresholded metrics.
        baseline, _ = additional_metrics(scores, labels, 0.0,
            {key: 0 for key in ("precision", "recall", "f1", "TP", "TN", "FP", "FN")})
        self.assertEqual(continuous_metrics(scores, labels), baseline["continuous_score"])


class DatasetTests(unittest.TestCase):
    def test_no_cross_machine_window_and_seeded_whole_window_shuffle(self):
        arrays = {"A": np.full((205, 38), 10, np.float32), "B": np.full((205, 38), 20, np.float32)}
        dataset = MachineWindows(arrays)
        self.assertEqual(len(dataset), 4)
        for index in range(len(dataset)):
            window, sample_id = dataset[index]
            name, start = dataset.provenance(index)
            self.assertEqual(sample_id, index)
            self.assertLessEqual(start + 100, len(arrays[name]))
            self.assertTrue(torch.all(window == (10 if name == "A" else 20)))
        orders = [[ids.tolist() for _, ids in training_loader(dataset, 2, seed)] for seed in (1, 1, 2)]
        self.assertEqual(orders[0], orders[1])
        self.assertNotEqual(orders[0], orders[2])

    def test_cross_machine_train_and_normalization_isolation(self):
        config, canonical = configuration(ROOT)
        target = "machine-1-1"
        sources = select_sources(canonical, target, "cross_machine")
        seen = []
        original = np.full((100, 38), 3.5, np.float32)
        def reader(path):
            self.assertNotEqual(path.name, target + ".txt")
            self.assertEqual(path.parent.name, "train")
            seen.append(path)
            return original.copy()
        arrays = load_sources(Path("/unused"), sources, target, "cross_machine", reader)
        self.assertEqual(len(seen), 27)
        self.assertNotIn(target, arrays)
        for values in arrays.values():
            np.testing.assert_array_equal(values, original)
        info = normalization_metadata()
        self.assertFalse(info["normalization_fitted"])
        self.assertFalse(info["target_train_used_for_normalization"])
        self.assertFalse(info["target_test_used_for_normalization"])
        with self.assertRaises(ValueError):
            load_sources(Path("/unused"), [target], target, "cross_machine", reader)

    def test_full_timestamp_alignment_including_right_aligned_tail(self):
        for length in (100, 200, 201, 257):
            plan = evaluation_plan(length)
            energies = [np.arange(row["start"], row["stop"], dtype=np.float32) for row in plan]
            scores, mapping = assemble_energies(energies, plan, length)
            np.testing.assert_array_equal(scores, np.arange(length))
            np.testing.assert_array_equal(mapping["coverage"], np.ones(length))
            self.assertEqual(len(scores), length)
            for t in range(length):
                row = plan[mapping["window_index"][t]]
                self.assertEqual(row["start"] + mapping["window_offset"][t], t)
        with self.assertRaises(ValueError):
            evaluation_plan(99)

    def test_artifact_namespace_and_manifest(self):
        config, _ = configuration(ROOT)
        rows = validate_manifest(ROOT / "configs/anomaly_transformer_runs.csv", config)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({row["target"] for row in rows}), 6)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.assertEqual(namespace(root), root / "anomaly_transformer")
            for name in ("calibration", "runs", "checkpoints", "summaries"):
                with self.assertRaises(ValueError):
                    namespace(root / name)

    def test_source_file_cannot_alias_test_or_target_train(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "train").mkdir()
            (root / "test").mkdir()
            (root / "train/target.txt").write_text("never read")
            (root / "test/target.txt").write_text("never read")
            source = root / "train/source.txt"
            for forbidden in (root / "train/target.txt", root / "test/target.txt"):
                source.symlink_to(forbidden)
                with self.assertRaises(ValueError):
                    load_sources(root, ["source"], "target", "cross_machine", lambda p: None)
                source.unlink()


class CLITests(unittest.TestCase):
    def test_two_dry_runs_source_mapping_and_no_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, canonical = fixture(temp)
            env = dict(os.environ, PROJECT_ROOT=str(ROOT), DATA_ROOT=str(roots.data),
                       RESULTS_ROOT=str(roots.results), PYTHONDONTWRITEBYTECODE="1")
            for setting, count in (("per_machine", 1), ("cross_machine", 27)):
                result = subprocess.run([sys.executable, str(ROOT / "scripts/run_anomaly_transformer.py"),
                    "--hpc", "--device", "cpu", "--target", "machine-1-1", "--setting", setting, "--dry-run"],
                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                info = json.loads(result.stdout)
                self.assertEqual(len(info["sources"]), count)
                self.assertEqual(info["target"], "machine-1-1")
                self.assertFalse(info["target_test_used_for_validation"])
                self.assertEqual(info["model_config"]["win_size"], 100)
                self.assertFalse(roots.results.exists())

    def test_all_cli_imports(self):
        for name in ("run_anomaly_transformer.py", "build_anomaly_transformer_manifest.py",
                     "find_failed_anomaly_transformer_runs.py", "analyze_anomaly_transformer.py"):
            result = subprocess.run([sys.executable, str(ROOT/"scripts"/name), "--help"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_cuda_unavailable_records_failed_attempt_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, canonical = fixture(temp)
            with patch("torch.cuda.is_available", return_value=False), redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    execute(roots, config, canonical, "machine-1-1", "per_machine", "cuda", 0, "smoke_gate")
            parent = roots.results / "anomaly_transformer/runs" / logical_id("machine-1-1", "per_machine")
            status = json.loads(next(parent.glob("*/status.json")).read_text())
            self.assertFalse(status["completed"])
            self.assertEqual(status["exit_code"], 1)

    def test_synthetic_12_run_analysis_and_missing_task_guard(self):
        # Artificial completed records test analysis only; no 12-run model training occurs.
        with tempfile.TemporaryDirectory() as temp:
            roots, config, canonical = fixture(temp)
            out = namespace(roots.results)
            rows = validate_manifest(ROOT / "configs/anomaly_transformer_runs.csv", config)
            last_status = None
            for row in rows:
                target, setting = row["target"], row["setting"]
                parent = out / "runs" / logical_id(target, setting)
                folder = parent / "attempt-20000101T000000000000Z-deadbeef"
                folder.mkdir(parents=True)
                checkpoint = folder / "fixture-checkpoint"
                checkpoint.write_bytes(b"synthetic analysis fixture; never loaded as model")
                scores = np.array([.1,.8,.3,.9,.2] if setting == "per_machine" else [.9,.2,.3,.1,.8], np.float32)
                labels = np.array([0,1,0,1,0])
                np.savez(folder / "scores.npz", scores=scores, labels=labels, coverage=np.ones(5, dtype=int))
                write_json(folder / "evaluation.json", continuous_metrics(scores, labels))
                write_json(folder / "training_history.json", [dict(epoch=i+1, loss1=.1, loss2=.1,
                    reconstruction=.1, series_loss=.1, prior_loss=.1) for i in range(10)])
                (folder / "run.log").write_text("synthetic summary fixture")
                meta = dict(detector="AnomalyTransformer", protocol=config["protocol"], target=target,
                    setting=setting, seed=1, dtype="float32", window=100, batch_size=256, eval_batch_size=256,
                    train_stride=100, epochs=10, lr=1e-4, device="cpu", training_shuffle=True,
                    normalization_source=config["normalization"], normalization_fitted=False,
                    temperature=50, k=3, learning_rate_schedule=config["lr_schedule"],
                    optimizer_config=config["optimizer"], upstream_commit=config["upstream_commit"],
                    sources=select_sources(canonical, target, setting), model_config=config["model"],
                    target_train_used_for_training=setting == "per_machine", target_train_loaded=setting == "per_machine",
                    target_test_used_for_training=False, target_test_used_for_validation=False,
                    target_train_used_for_normalization=False, target_test_used_for_normalization=False,
                    threshold_evaluation=False, point_adjustment=False, frozen_model_unchanged=True,
                    all_windows_single_machine=True, score_label_alignment=True, target_test_loaded_after_training=True,
                    checkpoint=str(checkpoint), checkpoint_sha256=file_hash(checkpoint), test_score_count=5, label_count=5,
                    train_windows=2, training_seconds=0., inference_seconds=0., runtime_seconds=0., peak_gpu_allocated_bytes=0)
                write_json(folder / "metadata.json", meta)
                last_status = folder / "status.json"
                write_json(last_status, dict(task_id=int(row["task_id"]), target=target, setting=setting,
                    run_path=str(folder), completed=True, exit_code=0))
            reference = Path(temp).resolve() / "tranad.csv"
            with reference.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["target_machine", "setting", "seed", "source_machines", "pr_auc"])
                writer.writeheader()
                for target in config["targets"]:
                    for setting in ("per_machine", "cross_machine_zero_shot"):
                        sources = [target] if setting == "per_machine" else [m for m in canonical if m != target]
                        writer.writerow(dict(target_machine=target, setting=setting, seed=1,
                            source_machines=";".join(sources), pr_auc=.7 if setting == "per_machine" else .5))
            original = reference.read_bytes()
            env = dict(os.environ, PROJECT_ROOT=str(ROOT), DATA_ROOT=str(roots.data), RESULTS_ROOT=str(roots.results))
            command = [sys.executable, str(ROOT / "scripts/analyze_anomaly_transformer.py"),
                       "--hpc", "--tranad-results", str(reference)]
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            folders = list((out / "summaries").iterdir())
            self.assertEqual(len(folders), 1)
            with (folders[0] / "tranad_vs_anomaly_transformer_gaps.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            self.assertEqual(reference.read_bytes(), original)
            last_status.unlink()
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(list((out / "summaries").iterdir())), 1)

    def test_mock_training_pipeline_no_target_train_or_early_test_read(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, config, canonical = fixture(temp)
            # Any accidental target TRAIN read will fail numeric parsing.
            (roots.data / "train/machine-1-1.txt").write_text("TARGET_TRAIN_MUST_NOT_BE_READ\n")
            sentinels = []
            for name in ("runs", "checkpoints", "calibration", "summaries"):
                path = roots.results / name / "keep.txt"
                path.parent.mkdir(parents=True)
                path.write_text("immutable")
                sentinels.append(path)
            done = False
            def synthetic_train(model, dataset, cfg, device, callback):
                nonlocal done
                self.assertNotIn("machine-1-1", dataset.names)
                history = [dict(epoch=i+1, loss1=1., loss2=1., reconstruction=1., series_loss=0.,
                                prior_loss=0., seconds=0., batches=1, learning_rate=.0001) for i in range(10)]
                callback(history)
                done = True
                return history
            def read_test(path):
                self.assertTrue(done, "TEST numeric values read before training completed")
                self.assertEqual(path.parent.name, "test")
                return load_values(path)
            env = dict(PROJECT_ROOT=str(ROOT), DATA_ROOT=str(roots.data), RESULTS_ROOT=str(roots.results))
            with patch.dict(os.environ, env), patch("src.anomaly_transformer.experiment.train_fixed", synthetic_train), \
                 patch("src.anomaly_transformer.experiment.load_values", side_effect=read_test), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                folder = execute(roots, config, canonical, "machine-1-1", "cross_machine", "cpu", 1, "smoke_gate")
            metadata = json.loads((folder / "metadata.json").read_text())
            evaluation = json.loads((folder / "evaluation.json").read_text())
            self.assertEqual(set(evaluation), {"pr_auc_trapezoidal", "average_precision", "roc_auc"})
            self.assertFalse(metadata["target_train_loaded"])
            self.assertFalse(metadata["target_train_used_for_training"])
            self.assertFalse(metadata["target_test_used_for_training"])
            self.assertFalse(metadata["target_test_used_for_validation"])
            self.assertEqual(metadata["test_score_count"], 257)
            self.assertTrue(np.isfinite(metadata["score_diagnostics"]["min"]))
            self.assertTrue(np.isfinite(metadata["score_diagnostics"]["max"]))
            with np.load(folder / "scores.npz") as archive:
                self.assertEqual(len(archive["scores"]), len(archive["labels"]))
                np.testing.assert_array_equal(np.flatnonzero(archive["labels"]), [2,3,4,5,6,240,241,242,243,244])
            for path in sentinels:
                self.assertEqual(path.read_text(), "immutable")
            checkpoint = torch.load(metadata["checkpoint"], map_location="cpu", weights_only=True)
            model = AnomalyTransformer(**config["model"])
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            with redirect_stderr(io.StringIO()):
                scores, _, _ = infer(model, load_values(roots.data / "test/machine-1-1.txt"), config, torch.device("cpu"))
            with np.load(folder / "scores.npz") as archive:
                np.testing.assert_array_equal(scores, archive["scores"])


if __name__ == "__main__":
    unittest.main()
