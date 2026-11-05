"""Calibration 的純設定、manifest 與 prefix 規則；不 import baseline CLI。"""
import csv
from decimal import Decimal
import json
from pathlib import Path


FIELDS = ["task_id", "target", "fraction", "seed"]
FRACTIONS = ("0.05", "0.10", "0.20")


def canonical_fraction(value):
    number = Decimal(str(value))
    for fraction in FRACTIONS:
        if number == Decimal(fraction):
            return fraction
    raise ValueError("fraction 僅允許 0.05 / 0.10 / 0.20")


def prefix_length(total, fraction):
    if total < 1:
        raise ValueError("target TRAIN 不可為空")
    length = int(Decimal(total) * Decimal(canonical_fraction(fraction)))
    if length < 1:
        raise ValueError("calibration prefix 為空；不擴張 fraction")
    return length


def load_configuration(project):
    project = Path(project)
    formal = json.loads((project / "configs/formal_protocol_v2.json").read_text())
    config = json.loads((project / "configs/target_calibration.json").read_text())
    expected = dict(base_protocol="v2", seed=1, device="cpu", dtype="float64",
                    window=10, eval_batch_size=128, optimizer_steps=0,
                    normalization_fitting=False, slice="target_train_prefix_floor",
                    test_scores="reuse_formal_cross_machine_artifacts",
                    output_namespace="calibration", auc_tolerance=1e-12)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("calibration config 偏離固定設計")
    pot = dict(q=1e-5, level=0.99995, multiplier=1.06)
    if config["fractions"] != list(FRACTIONS) or config["pot"] != pot:
        raise ValueError("fraction / POT config 不符合固定設定")
    if any(formal["pot"][key] != value for key, value in pot.items()):
        raise ValueError("calibration 與 formal POT 參數不同")
    fixed = dict(model="TranAD", dataset="SMD", protocol="v2", seed=1, device="cpu", dtype="float64", window=10,
                 epochs=5, batch_size=128, eval_batch_size=128, learning_rate=0.0001)
    if any(formal.get(key) != value for key, value in fixed.items()):
        raise ValueError("formal config 偏離 validated protocol")
    machines = formal["machines"]
    if len(machines) != 28 or len(set(machines)) != 28:
        raise ValueError("formal config 必須有 28 個不同 targets")
    # 以既有 canonical order 與 formal manifest 核對 task mapping，不另建 machine list。
    with (project / "configs/formal_28_machine_runs.csv").open(newline="") as stream:
        formal_rows = list(csv.DictReader(stream))
    if len(formal_rows) != 56:
        raise ValueError("formal manifest 必須為 56 tasks")
    for index, row in enumerate(formal_rows):
        target = machines[index % 28]
        cross = index >= 28
        sources = [m for m in machines if m != target] if cross else [target]
        expected_row = dict(task_id=str(index), target_machine=target,
                            setting="cross_machine_zero_shot" if cross else "per_machine",
                            protocol="v2", seed="1", epochs="5", batch_size="128", eval_batch_size="128")
        if any(row.get(key) != value for key, value in expected_row.items()) or json.loads(row["source_machines"]) != sources:
            raise ValueError("formal manifest 與 canonical order 不一致")
    return config, machines


def manifest_rows(machines):
    return [dict(task_id=str(i * 3 + j), target=machine, fraction=fraction, seed="1")
            for i, machine in enumerate(machines)
            for j, fraction in enumerate(FRACTIONS)]


def validate_manifest(path, machines):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("calibration manifest 欄位或順序錯誤")
        rows = list(reader)
    if rows != manifest_rows(machines) or len(rows) != 84:
        raise ValueError("manifest 必須精確等於 28 targets × 3 fractions 的 84 tasks")
    return rows


def calibration_directory(results, formal_root):
    """只允許新 namespace；不得覆蓋 formal 輸入目錄或其祖先。"""
    results, formal_root = Path(results).resolve(), Path(formal_root).resolve()
    if (results / "calibration").is_symlink():
        raise ValueError("calibration namespace 不接受 symlink")
    output = (results / "calibration").resolve()
    for child in ("runs", "task_status", "summaries", "cache", "slurm-logs"):
        path = output / child
        if path.is_symlink() or path.resolve().parent != output:
            raise ValueError("calibration 子目錄不可經 symlink 指向其他 artifacts")
    if output == formal_root or formal_root.is_relative_to(output):
        raise ValueError("calibration output 不可包含 formal results root")
    protected = [formal_root / name for name in
                 ("runs", "checkpoints", "logs", "task_status", "summaries", "cache", "slurm-logs")]
    for path in protected:
        path = path.resolve()
        if output == path or output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("calibration output 與 formal artifacts 重疊")
    return output
