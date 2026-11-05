"""Calibration protocol tests；可在本機使用 synthetic fixtures 執行。"""
import csv
from pathlib import Path
import tempfile
import unittest

from src.calibration_protocol import (FIELDS, calibration_directory, canonical_fraction,
                                      load_configuration, manifest_rows, prefix_length,
                                      validate_manifest)

ROOT = Path(__file__).resolve().parents[1]


class CalibrationProtocolTests(unittest.TestCase):
    def test_fraction_slicing_is_floor_and_nested(self):
        self.assertEqual([prefix_length(23703, f) for f in ("0.05", "0.10", "0.20")],
                         [1185, 2370, 4740])
        self.assertEqual(prefix_length(28479, "0.05"), 1423)
        self.assertEqual(canonical_fraction("0.1"), "0.10")
        self.assertEqual(prefix_length(100, "0.05"), 5)  # 不足 window=10 仍沿用原 padding。
        for fraction in ("0", "0.01", "0.15", "0.50", "1", "-0.05", "NaN"):
            with self.assertRaises((ValueError, ArithmeticError)):
                canonical_fraction(fraction)
        with self.assertRaises(ValueError):
            prefix_length(1, "0.05")

    def test_committed_manifest_has_exact_84_unique_tasks(self):
        config, machines = load_configuration(ROOT)
        rows = validate_manifest(ROOT / "configs/calibration_runs.csv", machines)
        self.assertEqual(len(rows), 84)
        self.assertEqual(len({(row["target"], row["fraction"], row["seed"]) for row in rows}), 84)
        self.assertEqual(len({row["target"] for row in rows}), 28)
        self.assertEqual(rows[0]["target"], "machine-1-1")
        self.assertEqual(rows[-1]["fraction"], "0.20")
        self.assertEqual(config["optimizer_steps"], 0)

    def test_manifest_rejects_missing_duplicate_and_wrong_seed(self):
        _, machines = load_configuration(ROOT)
        original = manifest_rows(machines)
        variants = [original[:-1], original + [original[0]],
                    [dict(row, seed="2") if i == 0 else row for i, row in enumerate(original)]]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            for rows in variants:
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    validate_manifest(path, machines)

    def test_artifact_namespace_separate_and_rejects_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.assertEqual(calibration_directory(root, root), root / "calibration")
            self.assertEqual(calibration_directory(root / "new", root / "old"), root / "new/calibration")
            with self.assertRaises(ValueError):
                calibration_directory(root, root / "calibration/formal-input")
            (root / "calibration").mkdir()
            (root / "runs").mkdir()
            (root / "calibration/runs").symlink_to(root / "runs", target_is_directory=True)
            with self.assertRaises(ValueError):
                calibration_directory(root, root)


if __name__ == "__main__":
    unittest.main()
