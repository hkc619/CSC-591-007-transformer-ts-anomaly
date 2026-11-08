"""固定 6 targets / 12 tasks，source list 仍為完整 SMD 的其餘 27 台。"""
import csv
import json
from pathlib import Path


def configuration(project):
    project = Path(project)
    config = json.loads((project / "configs/anomaly_transformer.json").read_text())
    formal = json.loads((project / "configs/formal_protocol_v2.json").read_text())
    canonical = formal["machines"]
    targets = [name for group in (1, 2, 3) for name in canonical
               if name in (f"machine-{group}-1", f"machine-{group}-2")]
    if len(canonical) != 28 or len(set(canonical)) != 28 or len(targets) != 6 or config["targets"] != targets:
        raise ValueError("canonical SMD list / 預定 6-target subset 不符合 protocol")
    expected = dict(seed=1, dtype="float32", epochs=10, batch_size=256, eval_batch_size=256,
                    learning_rate=1e-4, k=3, temperature=50, kl_epsilon=1e-4,
                    train_stride=100, normalization="identity_no_fitting_same_as_tranad",
                    training_shuffle=True, threshold_evaluation=False, point_adjustment=False,
                    test_mapping="nonoverlap_full_windows_then_right_aligned_tail_emit_only_uncovered",
                    validation="none_fixed_epochs_last_checkpoint",
                    lr_schedule="upstream_epoch_end_lr_base_times_0.5_pow_epoch_zero_based")
    model = dict(win_size=100, enc_in=38, c_out=38, d_model=512, n_heads=8,
                 e_layers=3, d_ff=512, activation="gelu", dropout=0.0, output_attention=True)
    if any(config.get(key) != value for key, value in expected.items()) or config["model"] != model:
        raise ValueError("Anomaly Transformer config 偏離固定 protocol")
    if config["optimizer"] != dict(name="Adam", betas=[.9, .999], eps=1e-8, weight_decay=0):
        raise ValueError("optimizer config 不符合 native Adam")
    return config, canonical


def expected_rows(config):
    return [dict(task_id=str(i * 2 + j), target=target, setting=setting, seed="1")
            for i, target in enumerate(config["targets"])
            for j, setting in enumerate(("per_machine", "cross_machine"))]


def validate_manifest(path, config):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["task_id", "target", "setting", "seed"]:
            raise ValueError("AT manifest schema 錯誤")
        rows = list(reader)
    if rows != expected_rows(config):
        raise ValueError("AT manifest 必須是固定 6×2=12 tasks")
    return rows


def namespace(results):
    root = Path(results).resolve()
    destination = root / "anomaly_transformer"
    if destination.is_symlink():
        raise ValueError("AT namespace 不接受 symlink")
    for name in ("runs", "checkpoints", "task_status", "summaries", "slurm-logs", "cache"):
        if (destination / name).is_symlink():
            raise ValueError("AT namespace 子目錄不接受 symlink")
    # 防止把 RESULTS_ROOT 指向已知既有 experiment 子目錄。
    if any(part in ("calibration", "formal-28-machine", "runs", "checkpoints", "summaries") for part in root.parts):
        raise ValueError("RESULTS_ROOT 必須是共同外部 root，不能指向既有 experiment directory")
    return destination


def logical_id(target, setting):
    if setting not in ("per_machine", "cross_machine"):
        raise ValueError("未知 setting")
    return f"at-{setting.replace('_', '-')}-{target}-seed1"
