"""Synthetic attempts / summary tests；不使用正式實驗結果。"""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.calibration_analysis import build_tables, write_outputs
from src.calibration_artifacts import (experiment_id, failed_calibration_tasks,
                                       latest_calibration_status, new_attempt)
from src.calibration_protocol import load_configuration, manifest_rows

ROOT = Path(__file__).resolve().parents[1]


class CalibrationAttemptsTests(unittest.TestCase):
    def test_deterministic_identity_no_overwrite_and_explicit_retry(self):
        self.assertEqual(experiment_id("machine-1-1", ".05"), "calib-machine-1-1-f005-seed1")
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp).resolve() / "calibration"
            logical, first_id, first = new_attempt(namespace, "machine-1-1", ".05")
            sentinel = first / "existing-artifact"
            sentinel.write_text("preserve")
            with self.assertRaises(FileExistsError):
                new_attempt(namespace, "machine-1-1", ".05")
            logical2, second_id, second = new_attempt(namespace, "machine-1-1", ".05", retry=True)
            self.assertEqual(logical, logical2)
            self.assertNotEqual(first_id, second_id)
            self.assertNotEqual(first, second)
            self.assertEqual(sentinel.read_text(), "preserve")

    def test_incomplete_new_attempt_blocks_older_success(self):
        with tempfile.TemporaryDirectory() as temp:
            namespace = Path(temp).resolve() / "calibration"
            _, machines = load_configuration(ROOT)
            rows = manifest_rows(machines)
            _, _, first = new_attempt(namespace, "machine-1-1", ".05")
            (first / "status.json").write_text(json.dumps(dict(task_id=0, target="machine-1-1",
                fraction="0.05", smoke_test=False, run_path=str(first), completed=True, exit_code=0)))
            _, _, latest = new_attempt(namespace, "machine-1-1", ".05", retry=True)
            records = latest_calibration_status(namespace, rows)
            self.assertFalse(records[0]["completed"])
            self.assertEqual(Path(records[0]["run_path"]), latest)
            result, _ = failed_calibration_tasks(namespace, rows)
            self.assertIn(0, result["failed_or_missing_task_ids"])
            self.assertEqual(result["count"], 84)

    def synthetic_rows(self):
        config, machines = load_configuration(ROOT)
        rows = []
        for machine in machines:
            for fraction in config["fractions"]:
                row = dict(target=machine, group=int(machine.split("-")[1]), fraction=float(fraction), seed=1,
                    raw_precision_delta=.1, raw_recall_delta=.2, raw_f1_delta=.3,
                    zero_shot_threshold=.8, calibrated_threshold=.4, absolute_shift=.4,
                    signed_shift=-.4, relative_ratio=.5, relative_ratio_status="defined")
                for metric in ("precision", "recall", "f1"):
                    row[f"zero_shot_raw_{metric}"] = .1
                    row[f"calibrated_raw_{metric}"] = .4
                rows.append(row)
        return config, rows

    def test_statistics_zero_shot_counted_once_and_groups_not_assumed(self):
        config, rows = self.synthetic_rows()
        recovery, statistics, groups, counts = build_tables(rows, config["fractions"])
        self.assertEqual(len(recovery), 84)
        self.assertEqual(len(statistics), 4)
        self.assertTrue(all(row["targets"] == 28 for row in statistics))
        self.assertEqual(len(groups), 9)
        self.assertTrue(all(row["mean_raw_f1_recovery"] > 0 for row in groups))
        self.assertEqual(counts["0.05"]["f1"]["improved"], 28)

    def test_synthetic_summary_writes_tables_report_and_five_figures(self):
        config, rows = self.synthetic_rows()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            write_outputs(folder, rows, config)
            for name in ("calibration_results.csv", "calibration_recovery.csv", "calibration_statistics.csv",
                         "calibration_groups.csv", "analysis.json", "CALIBRATION_RESULTS.md"):
                self.assertGreater((folder / name).stat().st_size, 0)
            with (folder / "calibration_results.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 84)
            for name in ("raw_f1_distribution", "raw_recall_distribution", "raw_f1_recovery_by_target",
                         "threshold_comparison", "group_raw_f1_recovery"):
                for extension in ("png", "pdf"):
                    self.assertGreater((folder / "figures" / f"{name}.{extension}").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
