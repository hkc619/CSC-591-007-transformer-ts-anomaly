"""只讀既有 AT checkpoint / TEST，分解 native score；不訓練、不選 threshold。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anomaly_transformer.dataset import TestWindows, assemble_energies, load_values
from src.anomaly_transformer.experiment import file_hash, load_labels, state_hash, write_json
from src.anomaly_transformer.model import AnomalyTransformer
from src.anomaly_transformer.protocol import logical_id
from src.anomaly_transformer.runtime import seed_everything
from src.anomaly_transformer.scoring import my_kl_loss, normalized_prior
from src.continuous_metrics import continuous_metrics
from src.paths import resolve_roots

STREAMS = ("reconstruction_score", "association_weight_score",
           "negative_discrepancy_score", "native_energy_score")
METRICS = ("pr_auc_trapezoidal", "average_precision", "roc_auc")
RTOL, ATOL = 1e-5, 1e-8


def decompose(inputs, output, series, priors, temperature):
    """與 native_energy 相同的 layer SUM；額外保留中間值供診斷。"""
    if not series or len(series) != len(priors):
        raise ValueError("association layers 不完整")
    reconstruction = (inputs - output).square().mean(dim=-1)
    series_loss, prior_loss = 0.0, 0.0
    for s, prior in zip(series, priors):
        p = normalized_prior(prior)
        series_loss = series_loss + my_kl_loss(s, p.detach()) * temperature
        prior_loss = prior_loss + my_kl_loss(p, s.detach()) * temperature
    discrepancy = series_loss + prior_loss
    # 保留 native_energy 的原運算順序，避免引入額外 rounding 差異。
    logits = -series_loss - prior_loss
    weight = torch.softmax(logits, dim=-1)
    parts = dict(zip(STREAMS, (reconstruction, weight, logits, weight * reconstruction)))
    parts["discrepancy"] = discrepancy
    if any(not torch.isfinite(value).all() for value in parts.values()):
        raise FloatingPointError("decomposition 出現 NaN/Inf")
    return parts


def infer_decomposition(model, raw, config, device):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = TestWindows(raw, config["model"]["win_size"])
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False, num_workers=0)
    before = state_hash(model)
    windows_by_stream = {key: [] for key in (*STREAMS, "discrepancy")}
    with torch.no_grad():
        for windows, ids in tqdm(loader, desc="AT diagnostic TEST only", unit="batch"):
            if model.training or torch.is_grad_enabled():
                raise RuntimeError("diagnostic 必須是 frozen eval/no_grad")
            windows = windows.to(device=device, dtype=torch.float32)
            output, series, priors, _ = model(windows)
            parts = decompose(windows, output, series, priors, config["temperature"])
            for key, values in parts.items():
                windows_by_stream[key].extend(values.cpu().numpy())
    after = state_hash(model)
    if before != after:
        raise RuntimeError("diagnostic model state 改變")
    arrays = {key: np.stack(values) for key, values in windows_by_stream.items()}
    scores, mapping = {}, None
    for key in STREAMS:
        scores[key], current = assemble_energies(arrays[key], dataset.plan, len(raw))
        if mapping is not None and any(not np.array_equal(mapping[k], current[k]) for k in mapping):
            raise RuntimeError("score streams timestamp mapping 不一致")
        mapping = current
    return scores, arrays, mapping, dataset.plan, before, after


def basic_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("statistics 需要 finite 非空資料")
    return dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                std=float(values.std()), median=float(np.median(values)))


def distribution_stats(values, association=False):
    values = np.asarray(values)
    result = basic_stats(values)
    zeros = int(np.count_nonzero(values == 0))
    result.update(zero_count=zeros, nonzero_count=int(values.size-zeros),
                  unique_count=int(np.unique(values).size), fraction_zero=zeros / values.size)
    if association:
        for power in (30, 20, 12):
            result[f"fraction_below_1e-{power}"] = float(np.mean(values < 10.0 ** -power))
        for threshold in (.5, .9):
            result[f"fraction_above_{threshold}"] = float(np.mean(values > threshold))
    return result


def diagnostics(scores, arrays):
    # float64 僅用於統計 entropy，模型與四條 score stream 仍為原 float32。
    weights = arrays["association_weight_score"].astype(np.float64)
    entropy = -(weights * np.log(np.maximum(weights, np.finfo(np.float32).tiny))).sum(-1)
    support, top1 = np.exp(entropy), weights.max(-1)
    logits = arrays["negative_discrepancy_score"]
    ranges = logits.max(-1).astype(np.float64) - logits.min(-1).astype(np.float64)
    sums = weights.sum(-1)
    if not np.allclose(sums, 1, rtol=RTOL, atol=ATOL):
        raise RuntimeError("association softmax 每個 window sum 必須約為 1")
    stats = dict(
        association_weight=distribution_stats(scores["association_weight_score"], association=True),
        association_weight_all_window_positions=distribution_stats(weights, association=True),
        reconstruction=distribution_stats(scores["reconstruction_score"]),
        native_energy=distribution_stats(scores["native_energy_score"]),
        discrepancy=basic_stats(arrays["discrepancy"]), association_logits=basic_stats(logits),
        per_window=dict(entropy=basic_stats(entropy), effective_support=basic_stats(support),
                        top1_mass=basic_stats(top1), logit_range=basic_stats(ranges),
                        weight_sum=basic_stats(sums)),
        scopes=dict(association_weight="one emitted value per raw TEST timestamp",
                    reconstruction="one emitted value per raw TEST timestamp",
                    per_window="all original full TEST windows, including right-aligned tail context",
                    logits_and_discrepancy="all original window positions",
                    entropy="natural log; float64 statistics; clamp_min=float32.tiny"))
    return stats, dict(window_entropy=entropy, window_effective_support=support,
                       window_top1_mass=top1, window_logit_range=ranges)


def validate_reproduction(scores, labels, mapping, saved, evaluation):
    reference = np.asarray(saved["scores"])
    if reference.shape != scores.shape or not np.isfinite(reference).all():
        raise ValueError("既有 native scores shape / finite check 失敗")
    if not np.array_equal(labels, saved["labels"]):
        raise ValueError("diagnostic labels 與既有 labels 不一致")
    for key, values in mapping.items():
        if key not in saved or not np.array_equal(values, saved[key]):
            raise ValueError(f"timestamp mapping 不一致：{key}")
    difference = float(np.max(np.abs(scores.astype(np.float64) - reference.astype(np.float64))))
    if not np.allclose(scores, reference, rtol=RTOL, atol=ATOL):
        raise ValueError(f"native score reproduction 失敗；max_abs_difference={difference}")
    original_metrics = continuous_metrics(reference, labels)
    current_metrics = continuous_metrics(scores, labels)
    for key in METRICS:
        value = evaluation[key]
        if not np.isfinite(value) or not all(np.isclose(metric[key], value, rtol=RTOL, atol=ATOL)
                                           for metric in (original_metrics, current_metrics)):
            raise ValueError(f"native metric reproduction 失敗：{key}")
    return dict(passed=True, rtol=RTOL, atol=ATOL, max_absolute_difference=difference,
                exact_score_equality=bool(np.array_equal(scores, reference)),
                original_evaluation=evaluation, saved_score_metrics=original_metrics,
                diagnostic_native_metrics=current_metrics, labels_unchanged=True,
                timestamp_mapping_unchanged=True, score_count=len(scores))


def diagnostic_directory(results_root, identity):
    if Path(identity).name != identity or identity in (".", ".."):
        raise ValueError("diagnostic run ID 必須是單一 directory name")
    root = Path(results_root).resolve() / "anomaly_transformer"
    parent = root / "diagnostics" / identity
    expected = root / "diagnostics"
    # resolve() 也拒絕 diagnostics 的任何 symlink 導向 runs/checkpoints 等位置。
    if parent.resolve() != parent or not parent.is_relative_to(expected):
        raise ValueError("diagnostic output path 不可透過 symlink 改寫 namespace")
    name = datetime.now(timezone.utc).strftime("score-decomposition-%Y%m%dT%H%M%S%fZ-") + uuid4().hex[:8]
    folder = parent / name
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_source(roots, run_dir):
    folder = Path(run_dir).expanduser().resolve()
    metadata, status = read_json(folder / "metadata.json"), read_json(folder / "status.json")
    targets = read_json(roots.project / "configs/anomaly_transformer.json")["targets"]
    if metadata["target"] not in targets:
        raise ValueError("source target 不在固定六台內")
    identity = logical_id(metadata["target"], metadata["setting"])
    expected_parent = roots.results / "anomaly_transformer/runs" / identity
    if folder.parent != expected_parent.resolve() or expected_parent.resolve() != expected_parent:
        raise ValueError("run-dir 必須是指定 RESULTS_ROOT 的原 AT attempt directory")
    if (not status["completed"] or status["exit_code"] != 0
            or any(status[key] != metadata[key] for key in ("target", "setting", "run_id", "attempt_id"))
            or metadata["run_id"] != identity or metadata["attempt_id"] != folder.name):
        raise ValueError("source run 並非已完成或 identity 不符")
    if metadata["detector"] != "AnomalyTransformer" or metadata["dtype"] != "float32":
        raise ValueError("source detector / dtype 不符")
    checkpoint = Path(metadata["checkpoint"]).resolve()
    expected_checkpoint = roots.results / "anomaly_transformer/checkpoints" / identity / folder.name / "model.ckpt"
    if checkpoint != expected_checkpoint or file_hash(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("checkpoint path / SHA256 不符")
    test = roots.data / "test" / (metadata["target"] + ".txt")
    label = roots.data / "interpretation_label" / (metadata["target"] + ".txt")
    if (test.parent.resolve() == (roots.data / "train").resolve()
            or not test.resolve().is_relative_to(test.parent.resolve())):
        raise ValueError("TEST 不可 alias TRAIN 或其他目錄")
    if (label.parent.resolve() == (roots.data / "train").resolve()
            or not label.resolve().is_relative_to(label.parent.resolve())):
        raise ValueError("label path 不可 alias TRAIN 或其他目錄")
    for name, path in (("test_path", test), ("label_path", label)):
        supplied = Path(metadata[name])
        if supplied.resolve() != path.resolve() or not path.is_file():
            raise ValueError(f"{name} 與 DATA_ROOT / target 不符")
        if file_hash(path) != metadata["data_sha256"][str(supplied)]:
            raise ValueError(f"{name} SHA256 與原 inference 資料不符")
    # 所有讀取都僅是 TEST、labels、checkpoint/config/artifacts；不呼叫 describe/load_sources。
    tracked = [folder / name for name in ("metadata.json", "status.json", "config.json",
                                          "scores.npz", "evaluation.json")]
    tracked += [checkpoint, test, label]
    fingerprints = {str(path): file_hash(path) for path in tracked}
    # 驗證影響 numerical result 的 source code；diagnostic 新檔不在原 snapshot 中。
    for relative in ("src/anomaly_transformer/model.py", "src/anomaly_transformer/attention.py",
                     "src/anomaly_transformer/embedding.py", "src/anomaly_transformer/scoring.py",
                     "src/anomaly_transformer/dataset.py", "src/anomaly_transformer/runtime.py",
                     "src/continuous_metrics.py", "preprocess.py"):
        if file_hash(roots.project / relative) != metadata["source_sha256"][relative]:
            raise ValueError(f"原 run source fingerprint 不符：{relative}")
    return folder, metadata, checkpoint, test, fingerprints


def validate_checkpoint(checkpoint, metadata, saved_config):
    for key in ("target", "setting", "sources"):
        if checkpoint[key] != metadata[key]:
            raise ValueError(f"checkpoint / metadata {key} 不符")
    config = checkpoint["config"]
    if config != saved_config or checkpoint["epoch"] != config["epochs"]:
        raise ValueError("checkpoint config / epoch 與 source run 不符")
    fields = {"model": "model_config", "protocol": "protocol", "seed": "seed",
              "temperature": "temperature", "eval_batch_size": "eval_batch_size",
              "batch_size": "batch_size", "epochs": "epochs", "dtype": "dtype"}
    if any(config[key] != metadata[name] for key, name in fields.items()):
        raise ValueError("checkpoint scientific config 與 metadata 不符")
    if (metadata["setting"] == "cross_machine" and metadata["target"] in checkpoint["sources"]):
        raise ValueError("checkpoint cross-machine sources 含有 target")
    if metadata["setting"] == "per_machine" and checkpoint["sources"] != [metadata["target"]]:
        raise ValueError("checkpoint per-machine source 不符")
    if config["temperature"] != 50 or config["model"]["win_size"] != 100 or config["dtype"] != "float32":
        raise ValueError("diagnostic 不允許修改 native temperature / window / dtype")
    if any(t.is_floating_point() and t.dtype != torch.float32 for t in checkpoint["model_state_dict"].values()):
        raise ValueError("checkpoint floating tensors 必須為 float32，不可靜默轉換")
    return config


def diagnose(roots, run_dir, device_name):
    folder, source, checkpoint_path, test_path, fingerprints = resolve_source(roots, run_dir)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；不自動 fallback CPU")
    # weights_only=True：只載入 tensor / primitive checkpoint，拒絕任意 pickle code。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = validate_checkpoint(checkpoint, source, read_json(folder / "config.json"))
    seed_everything(config["seed"])
    device = torch.device(device_name)
    model = AnomalyTransformer(**config["model"]).to(device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    output = diagnostic_directory(roots.results, source["run_id"])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=roots.project, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=roots.project, capture_output=True, text=True)
    metadata = dict(analysis_type="anomaly_transformer_score_decomposition", status="running",
        source_run=str(folder), source_run_id=source["run_id"], source_checkpoint=str(checkpoint_path),
        source_checkpoint_sha256=fingerprints[str(checkpoint_path)], target=source["target"], setting=source["setting"],
        device=str(device), source_device=source["device"], dtype="float32", temperature=config["temperature"],
        model_config=config["model"], eval_batch_size=config["eval_batch_size"], seed=config["seed"],
        gpu_model=torch.cuda.get_device_name() if device.type == "cuda" else None,
        torch_version=str(torch.__version__), numpy_version=np.__version__, cuda_version=torch.version.cuda,
        diagnostic_git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        diagnostic_git_dirty=bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        source_experiment_git_commit=source["git_commit"], source_input_sha256=fingerprints,
        native_score_reproduction_passed=False, target_train_loaded=False,
        model_retrained=False, optimizer_steps=0, diagnostic_only_streams=["negative_discrepancy_score"],
        created_at=datetime.now(timezone.utc).isoformat())
    print("Diagnostic output:", output, flush=True)
    write_json(output / "diagnostic_metadata.json", metadata)
    try:
        raw = load_values(test_path)
        labels = load_labels(roots.data, source["target"], raw.shape, output)
        if len(raw) != source["test_score_count"] or len(labels) != source["label_count"]:
            raise ValueError("TEST / label count 與原 run 不符")
        scores, arrays, mapping, plan, before, after = infer_decomposition(model, raw, config, device)
        if any(values.shape != labels.shape or not np.isfinite(values).all() for values in scores.values()):
            raise ValueError("四條 score stream / labels 長度或 finite check 失敗")
        with np.load(folder / "scores.npz", allow_pickle=False) as saved:
            reproduction = validate_reproduction(scores["native_energy_score"], labels, mapping,
                                                 saved, read_json(folder / "evaluation.json"))
        metrics = {key: continuous_metrics(values, labels) for key, values in scores.items()}
        stats, per_window = diagnostics(scores, arrays)
        for path, expected in fingerprints.items():
            if file_hash(path) != expected:
                raise RuntimeError(f"source input 在 diagnostic 期間改變：{path}")
        write_json(output / "diagnostic_metrics.json", dict(scores=metrics, native_reproduction=reproduction,
            primary_streams=[key for key in STREAMS if key != "negative_discrepancy_score"],
            diagnostic_only_streams=["negative_discrepancy_score"]))
        write_json(output / "diagnostic_stats.json", stats)
        with (output / "diagnostic_scores.npz").open("xb") as stream:
            np.savez_compressed(stream, **scores, labels=labels, **mapping, **per_window,
                window_plan=np.asarray([[r[k] for k in ("start", "stop", "emit_start", "emit_stop")]
                                        for r in plan], dtype=np.int64))
        metadata.update(status="completed", native_score_reproduction_passed=True,
            parameter_hash_before=before, parameter_hash_after=after, parameters_unchanged=before == after,
            source_inputs_unchanged=True, score_count=len(labels), label_count=len(labels),
            window_count=len(plan), window_size=config["model"]["win_size"],
            test_path=str(test_path), label_path=source["label_path"], window_plan_columns=["start", "stop", "emit_start", "emit_stop"])
        print(json.dumps(dict(metrics=metrics, native_reproduction=reproduction), ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), native_score_reproduction_passed=False)
        raise
    finally:
        write_json(output / "diagnostic_metadata.json", metadata)
    return output


def main():
    parser = argparse.ArgumentParser(description="既有 AT TEST native energy decomposition；只做 inference")
    for name in ("project-root", "data-root", "results-root"):
        parser.add_argument("--" + name)
    parser.add_argument("--hpc", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--run-dir", required=True, type=Path, help="既有已完成 smoke attempt directory")
    args = parser.parse_args()
    roots = resolve_roots(args)
    # validated label generator 的 legacy paths 必須維持與本次明確指定的 roots 一致。
    os.environ.update(PROJECT_ROOT=str(roots.project), DATA_ROOT=str(roots.data), RESULTS_ROOT=str(roots.results))
    diagnose(roots, args.run_dir, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
