"""Synthetic CLI smoke；真實 frozen inference，POT 使用明確的 stub，不做研究校準。"""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_target_calibration as fixtures
from src.calibration_runtime import baseline_runtime

ROOT = Path(__file__).resolve().parents[1]


class CalibrationCLITests(unittest.TestCase):
    def test_all_calibration_cli_help_entrypoints(self):
        for name in ("run_target_calibration.py", "build_calibration_manifest.py",
                     "find_failed_calibration_runs.py", "analyze_calibration.py"):
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), "--help"],
                    cwd=ROOT, text=True, capture_output=True, timeout=60,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("usage:", result.stdout)

    def fixture(self, temp):
        roots, config, machines, run, train = fixtures.CalibrationArtifactTests().create_fixture(temp)
        env = dict(os.environ, PROJECT_ROOT=str(ROOT), DATA_ROOT=str(roots.data),
                   RESULTS_ROOT=str(roots.results), FORMAL_RESULTS_ROOT=str(roots.results),
                   PYTHONDONTWRITEBYTECODE="1")
        return roots, run, env

    def cli(self, env, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "scripts/run_target_calibration.py"), *arguments],
                              cwd=ROOT, env=env, text=True, capture_output=True, timeout=60)

    def test_single_and_manifest_dry_run_cli_with_external_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, run, env = self.fixture(temp)
            for arguments in (("--target", "machine-1-1", "--fraction", "0.05"),
                              ("--manifest", "configs/calibration_runs.csv", "--task-id", "0")):
                with self.subTest(arguments=arguments):
                    result = self.cli(env, "--hpc", *arguments, "--dry-run")
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    description, _ = json.JSONDecoder().raw_decode(result.stdout)
                    self.assertEqual(description["formal_task_id"], 28)
                    self.assertEqual(description["formal_run_id"], run.name)
                    self.assertEqual(description["calibration_raw_points"], 10)
                    self.assertEqual(description["calibration_windows"], 10)
                    self.assertEqual(description["target"], "machine-1-1")
                    self.assertFalse((roots.results / "calibration").exists())

    def test_invalid_fraction_cli_fails_without_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, _, env = self.fixture(temp)
            result = self.cli(env, "--hpc", "--target", "machine-1-1", "--fraction", "0.15", "--dry-run")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fraction", result.stderr)
            self.assertFalse((roots.results / "calibration").exists())

    def test_cli_smoke_frozen_inference_stub_pot_and_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            roots, run, env = self.fixture(temp)
            with baseline_runtime() as core:
                # 這是隨機、未訓練的 fixture weights；epoch=4 只模擬已完成 formal schema。
                model = core.TranAD(38).double()
                torch.save(dict(epoch=4, model_state_dict=model.state_dict()),
                           roots.results / "checkpoints/formal_fixture/model.ckpt")

            def stub_pot(core, scores):
                self.assertEqual(scores.shape, (10,))
                self.assertTrue(np.isfinite(scores).all())
                return .35, dict(q=1e-5, requested_level=.99995, effective_level=.99995,
                                 multiplier=1.06, initialization_attempts=1,
                                 source_score_count=len(scores), stream_score_count=0)

            argv = [str(ROOT / "scripts/run_target_calibration.py"), "--hpc", "--target",
                    "machine-1-1", "--fraction", "0.05", "--smoke-test"]
            output = io.StringIO()
            with patch.dict(os.environ, env), patch.object(sys, "argv", argv), \
                 patch("src.calibration_runtime.fit_threshold", side_effect=stub_pot), \
                 patch.object(torch.optim.AdamW, "__init__", side_effect=AssertionError("不可建立 optimizer")), \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("不可 backward")), \
                 redirect_stdout(output), redirect_stderr(output):
                with self.assertRaises(SystemExit) as exit_info:
                    runpy.run_path(argv[0], run_name="__main__")
            self.assertEqual(exit_info.exception.code, 0, output.getvalue())
            parent = roots.results / "calibration/runs/smoke-calib-machine-1-1-f005-seed1"
            attempts = list(parent.iterdir())
            self.assertEqual(len(attempts), 1)
            folder = attempts[0]
            metadata = json.loads((folder / "metadata.json").read_text())
            status = json.loads((folder / "status.json").read_text())
            summary = json.loads((folder / "smoke_summary.json").read_text())
            self.assertTrue(status["completed"])
            self.assertEqual(status["exit_code"], 0)
            self.assertFalse(metadata["model_retrained"])
            self.assertEqual(metadata["parameter_hash_before"], metadata["parameter_hash_after"])
            self.assertEqual(metadata["optimizer_steps"], 0)
            self.assertEqual(metadata["pot"]["stream_score_count"], 0)
            self.assertEqual(summary["continuous_score_before"], summary["continuous_score_after"])
            with np.load(folder / "scores.npz") as after, np.load(run / "scores.npz") as before:
                np.testing.assert_array_equal(after["scores"], before["scores"])
                np.testing.assert_array_equal(after["labels_final"], before["labels_final"])


if __name__ == "__main__":
    unittest.main()
