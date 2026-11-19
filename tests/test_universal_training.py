"""Stage 1 的 TRAIN-only 邊界／I/O／原 checkpoint regression；只訓練 synthetic seed=1。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from src.paths import Roots
from src.seed_replication import PROTECTED, file_hash, read_json
from src.universal_training import (ROOT, claim_attempt, configuration, core_args,
    inspect_train, namespace, run_identity, update_status)


def environment():
    return dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg",
        MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"tranad-universal-test-mpl")))


def fixture(base):
    roots = Roots(ROOT, base / "SMD", base / "results")
    (roots.data / "train").mkdir(parents=True)
    for i, name in enumerate(configuration(ROOT, 1)["sources"]):
        count = 2 + i % 5  # 不等長且短於 window；同時測試自然 weighting 與 padding。
        values = i/28 + np.arange(count)[:, None]*0.001 + np.arange(38)[None, :]*0.0001
        np.savetxt(roots.data / "train" / (name + ".txt"), values, delimiter=",")
    # 刻意不建立 TEST 或 label，驗證此階段完全不依賴它們。
    return roots


def run_python(code, *arguments):
    return subprocess.run([sys.executable, "-c", code, *map(str, arguments)], cwd=ROOT,
                          env=environment(), capture_output=True, text=True, timeout=180)


class UniversalTests(unittest.TestCase):
    def test_canonical_28_sources_no_duplicates_no_exclusion_and_fixed_protocol(self):
        config = configuration(ROOT, 1)
        base = read_json(ROOT / "configs/formal_protocol_v2.json")
        self.assertEqual(config["sources"], base["machines"])
        self.assertEqual(len(set(config["sources"])), 28)
        self.assertIsNone(config["target"])
        for key in ("protocol", "model", "dataset", "epochs", "batch_size", "window",
                    "learning_rate", "dtype", "device", "sampling", "training_shuffle"):
            self.assertEqual(config[key], base[key], key)

    def test_seed_support_propagates_without_training_seeds_2_or_3(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = Roots(ROOT, Path(temp).resolve()/"SMD", Path(temp).resolve()/"results")
            for seed in (1, 2, 3):
                config = configuration(ROOT, seed)
                args = core_args(roots, config, roots.results / "unused", roots.results / "model.ckpt")
                self.assertEqual(args.seed, seed)
                self.assertIsNone(args.machine)
                self.assertIsNone(args.test_machine)
                self.assertEqual(args.train_machines, config["sources"])
                self.assertIn(f"seed{seed}", run_identity(seed))
            with self.assertRaises(ValueError):
                configuration(ROOT, 0)

    def test_train_manifest_counts_natural_weighting_and_deterministic_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = fixture(Path(temp).resolve())
            config = configuration(ROOT, 1)
            manifest = inspect_train(roots, config)
            self.assertEqual(manifest, inspect_train(roots, config))
            self.assertEqual(manifest["machine_count"], 28)
            self.assertEqual(manifest["total_training_windows"], sum(2+i%5 for i in range(28)))
            offset = 0
            for i, row in enumerate(manifest["machines"]):
                self.assertEqual(row["raw_train_length"], 2+i%5)
                self.assertEqual(row["window_count"], 2+i%5)
                self.assertEqual(row["offset_start"], offset)
                offset += row["window_count"]
                self.assertEqual(row["offset_end_exclusive"], offset)
                self.assertEqual(row["dropped_points"], 0)
            self.assertFalse(roots.results.exists())

    def test_missing_extra_nonfinite_wrong_dimension_train_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = fixture(Path(temp).resolve())
            config = configuration(ROOT, 1)
            path = roots.data / "train" / (config["sources"][0]+".txt")
            original = path.read_bytes()
            for invalid in (np.ones((4, 37)), np.full((4, 38), np.nan), np.full((4, 38), np.inf)):
                np.savetxt(path, invalid, delimiter=",")
                with self.assertRaises(ValueError):
                    inspect_train(roots, config)
            path.unlink()
            with self.assertRaises(ValueError):
                inspect_train(roots, config)
            path.write_bytes(original)
            extra = roots.data / "train/machine-9-9.txt"
            extra.write_bytes(original)
            with self.assertRaises(ValueError):
                inspect_train(roots, config)

    def test_artifact_isolation_seed_ids_attempts_and_success_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sentinel = root / "runs/formal-checkpoint.ckpt"
            sentinel.parent.mkdir()
            sentinel.write_bytes(b"immutable formal input")
            before = file_hash(sentinel)
            first, checkpoint, lock = claim_attempt(root, 1)
            self.assertTrue(checkpoint.is_relative_to(root / "universal/checkpoints"))
            self.assertIn("seed1", str(checkpoint))
            update_status(first, state="failed", completed=False)
            with self.assertRaises(FileExistsError):
                claim_attempt(root, 1, retry=True)
            lock.unlink()
            with self.assertRaises(FileExistsError):
                claim_attempt(root, 1)
            second, _, lock = claim_attempt(root, 1, retry=True)
            self.assertNotEqual(first, second)
            update_status(second, state="completed", completed=True)
            lock.unlink()
            with self.assertRaises(FileExistsError):
                update_status(second, state="failed", completed=False)
            with self.assertRaises(FileExistsError):
                claim_attempt(root, 1, retry=True)
            self.assertEqual(file_hash(sentinel), before)
            with self.assertRaises(ValueError):
                namespace(root / "calibration")
            (root / "elsewhere").mkdir()
            (root / "elsewhere/universal").symlink_to(root / "runs", target_is_directory=True)
            with self.assertRaises(ValueError):
                namespace(root / "elsewhere")

    def test_dry_run_reads_only_train_without_model_or_result_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = fixture(Path(temp).resolve())
            command = [sys.executable, "scripts/run_universal_training.py", "--seed", "1", "--hpc",
                "--project-root", str(ROOT), "--data-root", str(roots.data),
                "--results-root", str(roots.results), "--dry-run"]
            a = subprocess.run(command, cwd=ROOT, env=environment(), text=True, capture_output=True, timeout=30)
            b = subprocess.run(command, cwd=ROOT, env=environment(), text=True, capture_output=True, timeout=30)
            self.assertEqual(a.returncode, 0, a.stderr)
            self.assertEqual(b.returncode, 0, b.stderr)
            self.assertEqual(a.stdout, b.stdout)
            result = json.loads(a.stdout)
            self.assertIsNone(result["config"]["target"])
            self.assertEqual(len(result["config"]["sources"]), 28)
            self.assertFalse(result["test_data_required"])
            self.assertFalse(roots.results.exists())
            code = '''
import sys
from pathlib import Path
from src.paths import Roots
from src.universal_training import execute
execute(Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])), dry_run=True)
assert "main" not in sys.modules and "src.parser" not in sys.modules
assert "torch" not in sys.modules
'''
            probe = run_python(code, ROOT, roots.data, roots.results)
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_independent_windows_targets_provenance_shuffle_and_boundary_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = fixture(Path(temp).resolve())
            code = '''
import sys
from pathlib import Path
from unittest.mock import patch
import torch
from src.paths import Roots
from src.universal_training import configuration, inspect_train, core_args, import_core, assemble_training
from src.training_protocol import seeded_generator, shuffled_loader
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
config = configuration(roots.project, 1)
manifest = inspect_train(roots, config)
args = core_args(roots, config, roots.results/"probe", roots.results/"probe.ckpt")
core = import_core(args)
import preprocess
preprocess.output_folder = str(roots.results/"probe/cache/processed")
model, *_ = core.load_model("TranAD", 38)
windows, raw, provenance = assemble_training(config, manifest, model, core)
assert provenance["all_windows_single_machine"]
assert provenance["machine_count"] == 28
assert torch.equal(windows[:, -1], raw)
for record in provenance["machines"]:
    start, end = record["offset_start"], record["offset_end_exclusive"]
    machine_raw = raw[start:end]
    for t in range(end-start):
        indices = (torch.arange(-9, 1)+t).clamp(min=0)
        assert torch.equal(windows[start+t], machine_raw[indices])
def order():
    return torch.cat([ids for _, ids in shuffled_loader(windows, 128, seeded_generator(1))])
first, second = order(), order()
assert torch.equal(first, second)
assert torch.equal(first.sort().values, torch.arange(len(windows)))
assert torch.equal(windows[first][:, -1], raw[first])
original = core.convert_to_windows
with patch.object(core, "convert_to_windows", side_effect=lambda raw, model: original(raw, model).roll(1, 0)):
    try:
        assemble_training(config, manifest, model, core)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt windows accepted")
'''
            result = run_python(code, ROOT, roots.data, roots.results)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_synthetic_seed1_five_epoch_checkpoint_roundtrip_no_test_or_pot(self):
        with tempfile.TemporaryDirectory() as temp:
            roots = fixture(Path(temp).resolve())
            code = '''
import builtins
from pathlib import Path
import sys
from unittest.mock import patch
from src.paths import Roots
import src.universal_training as universal
roots = Roots(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
original_import = universal.import_core
def guarded_import(args):
    core = original_import(args)
    import src.cross_machine as cm
    def forbidden(*args, **kwargs):
        raise AssertionError("TEST / POT / evaluation path was invoked")
    cm.fit_source_threshold = forbidden
    cm.load_and_save2 = forbidden
    cm.run_smd_experiment = forbidden
    original_backprop = core.backprop
    def training_only(*args, **kwargs):
        assert kwargs.get("training") is True
        return original_backprop(*args, **kwargs)
    core.backprop = training_only
    original_conversion = cm.load_and_save
    def train_only(category, *args, **kwargs):
        assert category == "train"
        return original_conversion(category, *args, **kwargs)
    cm.load_and_save = train_only
    return core
original_open = builtins.open
def guarded_open(file, *args, **kwargs):
    if isinstance(file, (str, Path)):
        path = Path(file).absolute()
        if path.is_relative_to(roots.data):
            assert path.is_relative_to(roots.data / "train"), path
    return original_open(file, *args, **kwargs)
with patch.object(universal, "import_core", side_effect=guarded_import), patch("builtins.open", guarded_open):
    folder = universal.execute(roots, seed=1, command=["synthetic-seed1-regression"])
print("SYNTHETIC_RUN="+str(folder))
'''
            result = run_python(code, ROOT, roots.data, roots.results)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            parent = roots.results / "universal/runs" / run_identity(1)
            folders = list(parent.glob("attempt-*"))
            self.assertEqual(len(folders), 1)
            folder = folders[0]
            status = read_json(folder / "status.json")
            validation = read_json(folder / "validation.json")
            meta = read_json(folder / "metadata.json")
            self.assertTrue(status["completed"])
            self.assertTrue(validation["checkpoint_reload_passed"])
            self.assertTrue(validation["optimizer_state_equal"])
            self.assertTrue(validation["scheduler_state_equal"])
            self.assertEqual(validation["epochs_completed"], 5)
            self.assertEqual(meta["sources"], configuration(ROOT, 1)["sources"])
            self.assertFalse(meta["test_loaded"])
            self.assertIsNone(meta["target"])
            checkpoint = Path(meta["checkpoint_path"])
            self.assertTrue(checkpoint.is_file())
            self.assertEqual(file_hash(checkpoint), validation["checkpoint_sha256"])
            self.assertEqual(len(read_json(folder / "training_history.json")["epochs"]), 5)
            self.assertEqual(len(list((folder / "training_orders").glob("*.npz"))), 5)
            for filename in ("metadata.json", "launch.json", "run.log", "train_manifest.json",
                             "window_provenance.json", "training_history.json", "status.json", "validation.json"):
                self.assertTrue((folder / filename).is_file(), filename)
            self.assertFalse(list(folder.rglob("*scores*")))
            self.assertFalse(list(folder.rglob("*evaluation*")))

    def test_original_scientific_files_unchanged(self):
        for name in PROTECTED:
            committed = subprocess.check_output(["git", "show", "HEAD:"+name], cwd=ROOT)
            self.assertEqual(committed, (ROOT/name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
