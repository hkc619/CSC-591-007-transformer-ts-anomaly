"""只做 synthetic frozen inference / 分解測試；不啟動任何 training。"""
import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import diagnose_anomaly_transformer_scores as diagnostic
from src.anomaly_transformer.dataset import TestWindows, assemble_energies
from src.anomaly_transformer.model import AnomalyTransformer
from src.anomaly_transformer.scoring import native_energy
from src.continuous_metrics import continuous_metrics
from src.paths import Roots

ROOT = Path(__file__).resolve().parents[1]


class DecompositionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_native_layer_sum_and_softmax_normalization(self):
        windows, output = torch.randn(2, 100, 38), torch.randn(2, 100, 38)
        series = [torch.softmax(torch.randn(2, 2, 100, 100), -1) for _ in range(3)]
        priors = [torch.rand(2, 2, 100, 100) for _ in range(3)]
        result = diagnostic.decompose(windows, output, series, priors, 50)
        torch.testing.assert_close(result["native_energy_score"], native_energy(windows, output, series, priors, 50), rtol=0, atol=0)
        torch.testing.assert_close(result["association_weight_score"].sum(-1), torch.ones(2))
        torch.testing.assert_close(result["negative_discrepancy_score"], -result["discrepancy"], rtol=0, atol=0)
        torch.testing.assert_close(result["reconstruction_score"], (windows-output).square().mean(-1))

    def test_alignment_and_reproduction_rejects_score_label_mapping_and_metric_changes(self):
        raw = np.zeros((257, 38), dtype=np.float32)
        dataset = TestWindows(raw, 100)
        energies = [np.arange(row["start"], row["stop"], dtype=np.float32) for row in dataset.plan]
        scores, mapping = assemble_energies(energies, dataset.plan, len(raw))
        labels = (np.arange(len(raw)) % 3 == 0).astype(np.int64)
        np.testing.assert_array_equal(scores, np.arange(len(raw)))
        saved = dict(scores=scores.copy(), labels=labels.copy(), **mapping)
        evaluation = continuous_metrics(scores, labels)
        check = diagnostic.validate_reproduction(scores, labels, mapping, saved, evaluation)
        self.assertTrue(check["passed"])
        self.assertEqual(check["max_absolute_difference"], 0)
        for mutation in ("score", "shape", "label", "mapping", "metric"):
            bad, metrics = deepcopy(saved), dict(evaluation)
            if mutation == "score":
                bad["scores"][0] += 1
            elif mutation == "shape":
                bad["scores"] = bad["scores"][:-1]
            elif mutation == "label":
                bad["labels"][0] ^= 1
            elif mutation == "mapping":
                bad["window_offset"][0] = 99
            else:
                metrics["pr_auc_trapezoidal"] += .05
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                diagnostic.validate_reproduction(scores, labels, mapping, bad, metrics)

    def test_concentration_statistics_include_zeros_and_tail_scopes(self):
        weights = np.zeros((3, 100), dtype=np.float32)
        weights[:, 0] = 1
        arrays = dict(association_weight_score=weights, discrepancy=np.zeros_like(weights),
                      negative_discrepancy_score=np.zeros_like(weights))
        scores = dict(association_weight_score=weights.ravel(), reconstruction_score=np.ones(300),
                      native_energy_score=weights.ravel())
        stats, windows = diagnostic.diagnostics(scores, arrays)
        self.assertEqual(stats["association_weight"]["zero_count"], 297)
        self.assertEqual(stats["association_weight"]["fraction_below_1e-30"], .99)
        np.testing.assert_array_equal(windows["window_effective_support"], np.ones(3))
        self.assertEqual(stats["per_window"]["top1_mass"]["mean"], 1)
        json.dumps(stats, allow_nan=False)

    def test_machine_1_1_length_28479_four_streams_emit_once(self):
        dataset = TestWindows(np.zeros((28479, 38), dtype=np.float32))
        self.assertEqual(len(dataset.plan), 285)
        for key in diagnostic.STREAMS:
            energies = [np.arange(row["start"], row["stop"], dtype=np.float32) for row in dataset.plan]
            scores, mapping = assemble_energies(energies, dataset.plan, 28479)
            with self.subTest(stream=key):
                np.testing.assert_array_equal(scores, np.arange(28479))
                self.assertTrue(np.isfinite(scores).all())
                self.assertTrue(np.all(mapping["coverage"] == 1))

    def test_output_namespace_isolated_unique_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            identity = "at-per-machine-machine-1-1-seed1"
            first = diagnostic.diagnostic_directory(root, identity)
            second = diagnostic.diagnostic_directory(root, identity)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_relative_to(root / "anomaly_transformer/diagnostics"))
            self.assertFalse({"runs", "checkpoints"} & set(first.relative_to(root).parts))
            link = root / "other/anomaly_transformer"
            link.mkdir(parents=True)
            (link / "diagnostics").symlink_to(root / "anomaly_transformer/runs", target_is_directory=True)
            with self.assertRaises(ValueError):
                diagnostic.diagnostic_directory(root / "other", identity)
            with self.assertRaises(ValueError):
                diagnostic.diagnostic_directory(root, "../runs")

    def test_script_contains_no_optimizer_backward_or_training_calls(self):
        tree = ast.parse(Path(diagnostic.__file__).read_text())
        forbidden = {"Adam", "SGD", "Optimizer", "backward", "step", "train", "train_fixed",
                     "minimax_step", "training_loader", "load_sources", "describe", "infer"}
        calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                 for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, (ast.Attribute, ast.Name))}
        self.assertFalse(forbidden & calls)


class FrozenIntegrationTests(unittest.TestCase):
    def test_readonly_existing_checkpoint_test_only_full_diagnostic_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            roots = Roots(ROOT, root / "data", root / "results")
            target, setting = "machine-1-1", "cross_machine"
            identity = diagnostic.logical_id(target, setting)
            folder = roots.results / "anomaly_transformer/runs" / identity / "attempt-synthetic"
            folder.mkdir(parents=True)
            # 使用小型 synthetic checkpoint 測 I/O，不是 native model training 或真實 smoke。
            config = json.loads((ROOT / "configs/anomaly_transformer.json").read_text())
            config["model"].update(d_model=8, n_heads=2, e_layers=2, d_ff=8)
            config["eval_batch_size"] = 2
            canonical = json.loads((ROOT / "configs/formal_protocol_v2.json").read_text())["machines"]
            sources = [machine for machine in canonical if machine != target]
            diagnostic.seed_everything(1)
            model = AnomalyTransformer(**config["model"]).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            checkpoint = roots.results / "anomaly_transformer/checkpoints" / identity / folder.name / "model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(dict(model_state_dict=model.state_dict(), config=config, epoch=10,
                            target=target, setting=setting, sources=sources), checkpoint)
            raw = np.random.default_rng(1).random((257, 38)).astype(np.float32)
            test = roots.data / "test" / (target + ".txt")
            label = roots.data / "interpretation_label" / (target + ".txt")
            test.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            np.savetxt(test, raw, delimiter=",")
            label.write_text("2-7:1,3\n240-245:2\n")
            labels = np.zeros(257, dtype=np.int64)
            labels[2:7] = 1
            labels[240:245] = 1
            dataset = TestWindows(raw)
            energies = []
            with torch.no_grad():
                for start in range(0, len(dataset), config["eval_batch_size"]):
                    windows = torch.stack([dataset[i][0] for i in range(start, min(start+2, len(dataset)))])
                    output, series, priors, _ = model(windows)
                    energies.extend(native_energy(windows, output, series, priors, 50).numpy())
            scores, mapping = assemble_energies(energies, dataset.plan, len(raw))
            np.savez_compressed(folder / "scores.npz", scores=scores, labels=labels, **mapping)
            (folder / "evaluation.json").write_text(json.dumps(continuous_metrics(scores, labels)))
            source_paths = ["src/anomaly_transformer/" + name + ".py"
                            for name in ("model", "attention", "embedding", "scoring", "dataset", "runtime")]
            source_paths += ["src/continuous_metrics.py", "preprocess.py"]
            metadata = dict(target=target, setting=setting, sources=sources, detector="AnomalyTransformer",
                dtype="float32", device="cpu", checkpoint=str(checkpoint), checkpoint_sha256=diagnostic.file_hash(checkpoint),
                run_id=identity, attempt_id=folder.name, test_path=str(test), label_path=str(label),
                data_sha256={str(path): diagnostic.file_hash(path) for path in (test, label)},
                source_sha256={path: diagnostic.file_hash(ROOT / path) for path in source_paths},
                test_score_count=257, label_count=257, git_commit="synthetic-only",
                model_config=config["model"], protocol=config["protocol"], seed=1, temperature=50,
                eval_batch_size=2, batch_size=256, epochs=10)
            (folder / "metadata.json").write_text(json.dumps(metadata))
            (folder / "config.json").write_text(json.dumps(config))
            (folder / "status.json").write_text(json.dumps(dict(completed=True, exit_code=0,
                target=target, setting=setting, run_id=identity, attempt_id=folder.name)))
            original = {str(path): diagnostic.file_hash(path) for path in roots.results.rglob("*") if path.is_file()}
            reader = diagnostic.load_values
            def test_only(path):
                self.assertEqual(Path(path), test)
                return reader(path)
            # 禁止 optimizer / backward / 原 training API；TEST-only fixture 沒有 TRAIN directory。
            with patch.object(torch.optim, "Adam", side_effect=AssertionError("optimizer 禁止")), \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward 禁止")), \
                 patch("src.anomaly_transformer.runtime.train_fixed", side_effect=AssertionError("training 禁止")), \
                 patch.object(diagnostic, "load_values", side_effect=test_only):
                output = diagnostic.diagnose(roots, folder, "cpu")
            self.assertEqual(original, {path: diagnostic.file_hash(path) for path in original})
            result = json.loads((output / "diagnostic_metadata.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["native_score_reproduction_passed"])
            self.assertTrue(result["parameters_unchanged"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertFalse(result["target_train_loaded"])
            with np.load(output / "diagnostic_scores.npz") as saved:
                for key in diagnostic.STREAMS:
                    self.assertEqual(saved[key].shape, labels.shape)
                    self.assertTrue(np.isfinite(saved[key]).all())
                np.testing.assert_allclose(saved["native_energy_score"], scores,
                                           rtol=diagnostic.RTOL, atol=diagnostic.ATOL)
            with self.assertRaisesRegex(ValueError, "target"):
                diagnostic.validate_checkpoint(dict(config=config, target="machine-2-1", setting=setting,
                                                      sources=sources), metadata, config)
            with self.assertRaisesRegex(ValueError, "SHA256"):
                changed = deepcopy(metadata)
                changed["checkpoint_sha256"] = "wrong"
                (folder / "metadata.json").write_text(json.dumps(changed))
                diagnostic.resolve_source(roots, folder)
            # 刻意損壞 synthetic reference，確認整個 runner 拒絕而非只 helper 拒絕。
            (folder / "metadata.json").write_text(json.dumps(metadata))
            corrupted = scores.copy()
            corrupted[0] += 1
            np.savez_compressed(folder / "scores.npz", scores=corrupted, labels=labels, **mapping)
            with self.assertRaisesRegex(ValueError, "native score reproduction"):
                diagnostic.diagnose(roots, folder, "cpu")
            attempts = list(output.parent.glob("score-decomposition-*"))
            failed = [p for p in attempts if p != output]
            self.assertEqual(len(failed), 1)
            failed_metadata = json.loads((failed[0] / "diagnostic_metadata.json").read_text())
            self.assertEqual(failed_metadata["status"], "failed")
            self.assertFalse(failed_metadata["native_score_reproduction_passed"])
            self.assertFalse((failed[0] / "diagnostic_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
