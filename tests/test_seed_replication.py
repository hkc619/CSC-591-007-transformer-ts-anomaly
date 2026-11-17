"""多 seed 的 routing、RNG、唯讀分析與不覆寫測試；不執行 training / POT fitting。"""
from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.paths import Roots
from src.seed_replication import (PROTECTED, claim_attempt, claim_worker, configuration, expected_rows, file_hash,
    latest_attempt, manifest, namespace, read_json, task_parent, unique_name, worker_args, write_json)
from src.seed_replication_analysis import analyze, failed_tasks
from scripts import run_seed_replication as launcher

ROOT = Path(__file__).resolve().parents[1]


def environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                MPLBACKEND="Agg", CUDA_VISIBLE_DEVICES="")


def data_fixture(root):
    data = root / "SMD"
    _, machines = configuration(ROOT)
    for split in ("train", "test", "labels", "interpretation_label"):
        (data / split).mkdir(parents=True)
    for machine in machines:
        values = np.arange(12 * 38, dtype=np.float64).reshape(12, 38) / 1000
        np.savetxt(data / "train" / (machine + ".txt"), values, delimiter=",")
        np.savetxt(data / "test" / (machine + ".txt"), values, delimiter=",")
        np.savetxt(data / "labels" / (machine + ".txt"), np.arange(12) % 2, fmt="%d")
        (data / "interpretation_label" / (machine + ".txt")).write_text("1-3:1\n")
    return Roots(ROOT, data, root / "results")


class ProtocolTests(unittest.TestCase):
    def test_manifest_exactly_112_tasks_and_sources_exclude_target(self):
        rows = manifest(ROOT)
        self.assertEqual(rows, expected_rows(ROOT))
        self.assertEqual(len(rows), 112)
        self.assertEqual(rows[0]["seed"], "2")
        self.assertEqual(rows[56]["seed"], "3")
        self.assertEqual(rows[28]["base_task_id"], "28")
        self.assertEqual(rows[84]["target_machine"], "machine-1-1")
        for row in rows:
            sources = json.loads(row["source_machines"])
            if row["setting"] == "cross_machine_zero_shot":
                self.assertEqual(len(sources), 27)
                self.assertNotIn(row["target_machine"], sources)
            else:
                self.assertEqual(sources, [row["target_machine"]])
        self.assertEqual(len({(r["seed"], r["base_task_id"]) for r in rows}), 112)

    def test_adapter_matches_original_cli_defaults_except_explicit_replication_routing(self):
        with patch.object(sys, "argv", ["original-defaults"]):
            original = runpy.run_path(str(ROOT / "src/parser.py"))["args"]
        with tempfile.TemporaryDirectory() as temporary:
            roots = data_fixture(Path(temporary).resolve())
            row = manifest(ROOT)[0]
            args = worker_args(roots, row, task_parent(roots.results, row) / "dry-run", True)
            self.assertEqual(set(vars(original)), set(vars(args)))
            changed = {"seed", "progress", "retrain", "output_dir", "project_root", "data_root", "results_root", "hpc", "dry_run"}
            for key in set(vars(original))-changed:
                self.assertEqual(getattr(original, key), getattr(args, key), key)
            self.assertEqual(args.seed, 2)
            self.assertFalse(args.smoke_test)
            with self.assertRaises(ValueError):
                worker_args(roots, dict(row, seed="1"), task_parent(roots.results, row) / "dry-run", True)

    def test_protected_scientific_files_unchanged_from_committed_baseline(self):
        for path in PROTECTED:
            original = subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
            self.assertEqual(original, (ROOT / path).read_bytes(), path)

    def test_atomic_task_claim_refuses_overwrite_and_preserves_retry_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            row = manifest(ROOT)[0]
            first, lock = claim_attempt(base, row)
            write_json(first / "status.json", dict(row, state="failed", completed=False, exit_code=1))
            with self.assertRaises(FileExistsError):
                claim_attempt(base, row, retry=True)
            lock.unlink()
            before = file_hash(first / "status.json")
            with self.assertRaises(FileExistsError):
                claim_attempt(base, row)
            second, lock = claim_attempt(base, row, retry=True)
            self.assertNotEqual(first, second)
            self.assertEqual(file_hash(first / "status.json"), before)
            write_json(second / "status.json", dict(row, state="completed", completed=True, exit_code=0))
            lock.unlink()
            with self.assertRaises(FileExistsError):
                claim_attempt(base, row, retry=True)
            self.assertEqual(latest_attempt(base, row)[0], second)

    def test_symlink_namespace_and_using_old_run_as_results_root_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            old = base / "old"
            old.mkdir()
            (base / "seed-replication").symlink_to(old, target_is_directory=True)
            with self.assertRaises(ValueError):
                namespace(base)
            with self.assertRaises(ValueError):
                namespace(base / "runs")

    def test_internal_worker_can_be_claimed_only_once_per_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = Roots(ROOT, Path(temporary).resolve()/"SMD", Path(temporary).resolve()/"results")
            row = manifest(ROOT)[0]
            attempt, lock = claim_attempt(roots.results, row)
            write_json(attempt / "launch.json", dict(row=row, worker_results_root=str(attempt)))
            write_json(attempt / "status.json", dict(row, state="running", completed=False))
            args = worker_args(roots, row, attempt)
            claim_worker(attempt, row, ["python", "worker"], args)
            before = file_hash(attempt / "worker_start.json")
            with self.assertRaises(FileExistsError):
                claim_worker(attempt, row, ["python", "worker"], args)
            self.assertEqual(file_hash(attempt / "worker_start.json"), before)
            lock.unlink()
            with self.assertRaises(ValueError):
                claim_worker(attempt, row, ["python", "worker"], args)

    def test_launcher_records_success_and_failure_without_actual_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = Roots(ROOT, Path(temporary).resolve()/"SMD", Path(temporary).resolve()/"results")
            rows = manifest(ROOT)
            def fake_worker(roots, row, attempt, dry_run=False):
                # 明確 stub：只產生 launcher 所需的 synthetic files，不執行 main/model。
                code = r'''
import json, pathlib, sys
attempt = pathlib.Path(sys.argv[1]); row = json.loads(sys.argv[2])
if row['task_id'] == '1': sys.exit(7)
run = attempt/'runs'/'synthetic-launcher-check'; run.mkdir(parents=True)
(run/'evaluation.json').write_text('{}')
meta = dict(seed=int(row['seed']), setting=row['setting'], target_machine=row['target_machine'], epochs=5, batch_size=128, eval_batch_size=128, smoke_test=False)
(run/'metadata.json').write_text(json.dumps(meta))
print('Run directory:', run, flush=True)
'''
                return [sys.executable, "-u", "-c", code, str(attempt), json.dumps(row)]
            with patch.object(launcher, "worker_command", side_effect=fake_worker):
                self.assertEqual(launcher.execute(roots, rows[0]), 0)
                self.assertEqual(launcher.execute(roots, rows[1]), 7)
            completed, good = latest_attempt(roots.results, rows[0])
            failed, bad = latest_attempt(roots.results, rows[1])
            self.assertTrue(good["completed"])
            self.assertFalse(bad["completed"])
            self.assertEqual(bad["exit_code"], 7)
            self.assertFalse((completed.parent / "active.lock").exists())
            self.assertFalse((failed.parent / "active.lock").exists())
            self.assertTrue((completed / "run.log").is_file())
            with self.assertRaises(FileExistsError):
                launcher.execute(roots, rows[0])

    def test_real_cli_dry_runs_seeds_2_3_both_settings_write_no_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = data_fixture(Path(temporary).resolve())
            for task in (0, 28, 56, 84):
                command = [sys.executable, str(ROOT / "scripts/run_seed_replication.py"),
                    "--hpc", "--project-root", str(ROOT), "--data-root", str(roots.data),
                    "--results-root", str(roots.results), "--task-id", str(task), "--dry-run"]
                output = subprocess.check_output(command, cwd=ROOT, env=environment(), text=True)
                result = json.loads(output)
                self.assertEqual(result["seed"], 2 if task < 56 else 3)
                cross = task in (28, 84)
                self.assertEqual(len(result["sources"]), 27 if cross else 1)
                self.assertEqual(result["window_count"], 12 * (27 if cross else 1))
                self.assertFalse(result["training_started"])
                self.assertFalse(result["smoke_test"])
                self.assertEqual(result["pot_parameters"]["multiplier"], 1.06)
                self.assertTrue(Path(result["results_root"]).is_relative_to(namespace(roots.results)))
            self.assertFalse(roots.results.exists())

    def test_actual_prepare_run_seed_controls_model_init_and_shuffle_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = data_fixture(Path(temporary).resolve())
            code = r'''
import hashlib, json, os, sys, types
from pathlib import Path
from src.paths import Roots
from src.seed_replication import manifest, task_parent, worker_args
root = Path(sys.argv[1])
roots = Roots(Path.cwd(), root/'SMD', root/'rng-results')
row = manifest(Path.cwd())[0]
args = worker_args(roots, row, task_parent(roots.results,row)/'dry-run', True)
args.dry_run = False
adapter = types.ModuleType('src.parser'); adapter.args = args
sys.modules['src.parser'] = adapter
os.environ.update(PROJECT_ROOT=str(roots.project), DATA_ROOT=str(roots.data), RESULTS_ROOT=args.results_root)
import main
from src.experiment import prepare_run
from src.training_protocol import shuffled_loader
import torch
results = []
for seed in (2, 2, 3):
    args.seed = seed
    args.output_dir = str(Path(args.results_root)/'runs')
    prepare_run(args)
    model = main.TranAD(38).double()
    h = hashlib.sha256()
    for value in model.state_dict().values(): h.update(value.detach().numpy().tobytes())
    loader = shuffled_loader(torch.zeros(24,10,38), 5, args.training_generator)
    order = [int(i) for _, ids in loader for i in ids]
    results.append(dict(seed=seed, model=h.hexdigest(), order=order))
print(json.dumps(results))
'''
            completed = subprocess.check_output([sys.executable, "-c", code, str(roots.data.parent)],
                cwd=ROOT, env=environment(), text=True)
            samples = json.loads(completed.splitlines()[-1])
            self.assertEqual(samples[0], samples[1])
            self.assertNotEqual(samples[0]["model"], samples[2]["model"])
            self.assertNotEqual(samples[0]["order"], samples[2]["order"])
            self.assertEqual(sorted(samples[2]["order"]), list(range(24)))


def completed_fixture(roots):
    """產生假的 completed artifacts 測分析，不產生模型訓練或 anomaly scores。"""
    rows = manifest(ROOT)
    for row in [*[dict(r, seed="1") for r in rows[:56]], *rows]:
        seed = int(row["seed"])
        if seed == 1:
            run_root = roots.results
        else:
            run_root = task_parent(roots.results, row) / unique_name("attempt")
        run = run_root / "runs" / ("synthetic-seed" + str(seed) + "-task" + row["base_task_id"])
        run.mkdir(parents=True)
        checkpoint = run_root / "checkpoints" / run.name / "model.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic checkpoint; no training")
        cross = row["setting"] == "cross_machine_zero_shot"
        sources = json.loads(row["source_machines"])
        data_hashes = {str(roots.data / "train" / (m+".txt")): "synthetic-data-hash" for m in sources}
        data_hashes[str(roots.data / "test" / (row["target_machine"]+".txt"))] = "synthetic-data-hash"
        meta = dict(seed=seed, setting=row["setting"], target_machine=row["target_machine"], protocol="v2",
            python="synthetic-python test", packages={"torch": "synthetic", "numpy": "synthetic"}, torch_threads=4,
            epochs=5, batch_size=128, eval_batch_size=128, device="cpu", dtype="float64", window=10,
            learning_rate=0.0001, training_shuffle=True, smoke_test=False,
            threshold_source="source_machine_training_scores", threshold_protocol="source_only_frozen_SPOT_extreme_quantile",
            normalization="none_added; supplied SMD values unchanged", label_source="corrected_interpretation_label_union",
            source_machines=sources, checkpoint=str(checkpoint), threshold_value=.4, data_sha256=data_hashes,
            train_windows=20 * len(sources), test_timestamps=10, training_seconds=1, test_inference_seconds=.1,
            checkpoint_size_bytes=checkpoint.stat().st_size, training_loss_per_epoch=[.1]*5,
            pot_fit=dict(q=1e-5, requested_level=.99995, multiplier=1.06, stream_score_count=0, source_score_count=20 * len(sources)),
            leakage_checks=dict(target_used_for_testing=True, all_windows_single_machine=True,
                target_score_label_alignment=True, normalization_fitted=False, target_train_loaded=not cross,
                target_in_optimizer_updates=not cross, target_in_threshold_fitting=not cross))
        pr = (.3 + .005 * seed) if cross else (.4 + .01 * seed)
        evaluation = dict(raw=dict(precision=.5, recall=.25, f1=1/3, TP=1, TN=5, FP=1, FN=3),
            point_adjusted=dict(precision=.8, recall=.6, f1=.7), threshold=.4,
            continuous_score=dict(pr_auc_trapezoidal=pr, average_precision=pr, roc_auc=.6))
        write_json(run / "metadata.json", meta)
        write_json(run / "evaluation.json", evaluation)
        for name in PROTECTED:
            destination = run / "source" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, destination)
        if seed == 1:
            status_root = roots.results / "task_status"
            status_root.mkdir(exist_ok=True)
            write_json(status_root / (row["base_task_id"]+".json"),
                       dict(task_id=int(row["base_task_id"]), exit_code=0, run_path=str(run)))
        else:
            write_json(run_root / "status.json", dict(row, state="completed", completed=True, exit_code=0, run_path=str(run)))


class AnalysisTests(unittest.TestCase):
    def test_missing_runs_prevent_summary_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = Roots(ROOT, Path(temporary).resolve()/"SMD", Path(temporary).resolve()/"results")
            self.assertEqual(failed_tasks(roots.results, manifest(ROOT))["count"], 112)
            with self.assertRaises(ValueError):
                analyze(roots, roots.results, False)
            self.assertFalse(roots.results.exists())

    def test_analysis_pairing_std_all_figures_are_new_and_original_files_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = Roots(ROOT, Path(temporary).resolve()/"SMD", Path(temporary).resolve()/"results")
            completed_fixture(roots)
            old = roots.results / "summaries"
            old.mkdir()
            (old / "formal_28_machine_results.csv").write_text("keep original csv\n")
            (old / "existing_figure.png").write_bytes(b"keep original figure")
            before = {str(p): file_hash(p) for p in roots.results.rglob("*") if p.is_file()}
            first = analyze(roots, roots.results, True)
            self.assertEqual(before, {p: file_hash(p) for p in before})
            self.assertEqual(len(list((first / "figures").glob("*.png"))), 3)
            self.assertEqual(len(list((first / "figures").glob("*.pdf"))), 3)
            with (first / "target_seed_statistics.csv").open() as stream:
                target = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(target["pr_auc_gap_mean"]), .11)
            self.assertAlmostEqual(float(target["pr_auc_gap_std"]), .005)
            original_analysis = {str(p): file_hash(p) for p in first.rglob("*") if p.is_file()}
            second = analyze(roots, roots.results, False)
            self.assertNotEqual(first, second)
            self.assertEqual(original_analysis, {p: file_hash(p) for p in original_analysis})
            self.assertEqual(read_json(first / "analysis.json")["runs"], 168)
            self.assertTrue(read_json(first / "status.json")["completed"])
            self.assertEqual((first / "seed_results.csv").read_bytes(), (second / "seed_results.csv").read_bytes())
            # 新的失敗 attempt 不得回退至剛才成功的舊 attempt。
            row = manifest(ROOT)[0]
            failure = task_parent(roots.results, row) / unique_name("attempt")
            failure.mkdir()
            write_json(failure / "status.json", dict(row, state="failed", completed=False, exit_code=1))
            self.assertEqual(failed_tasks(roots.results, manifest(ROOT))["failed_or_missing_task_ids"], [0])
            with self.assertRaises(ValueError):
                analyze(roots, roots.results, False)
            self.assertEqual(len(list(first.parent.glob("analysis-*"))), 2)

    def test_wrong_seed_metadata_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = Roots(ROOT, Path(temporary).resolve()/"SMD", Path(temporary).resolve()/"results")
            completed_fixture(roots)
            attempt, status = latest_attempt(roots.results, manifest(ROOT)[0])
            path = Path(status["run_path"]) / "metadata.json"
            meta = read_json(path)
            meta["seed"] = 1
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "seed"):
                analyze(roots, roots.results, False)
            self.assertFalse((namespace(roots.results) / "summaries").exists())
            meta["seed"] = 2
            meta["packages"]["torch"] = "different-environment"
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "package"):
                analyze(roots, roots.results, False)


if __name__ == "__main__":
    unittest.main()
