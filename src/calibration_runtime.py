"""只在 Hazel 執行時載入 baseline；不建立 optimizer，不改原 scientific functions。"""
from contextlib import contextmanager
import hashlib
import importlib
import sys


@contextmanager
def baseline_runtime(progress=False):
    # 舊 parser 在 import 時 parse_args；暫時隔離 calibration / unittest 的 CLI。
    argv = sys.argv
    try:
        sys.argv = ["calibration-baseline", "--model", "TranAD", "--dataset", "SMD"]
        core = importlib.import_module("main")
    finally:
        sys.argv = argv
    if core.args.model != "TranAD" or core.args.dataset != "SMD":
        raise ValueError("已載入的 baseline parser 並非 TranAD / SMD")
    previous = {key: getattr(core.args, key) for key in ("progress", "eval_batch_size")}
    core.args.progress = progress
    core.args.eval_batch_size = 128
    try:
        yield core
    finally:
        for key, value in previous.items():
            setattr(core.args, key, value)


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def freeze_model(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("無法 freeze model")
    return state_hash(model)


def load_frozen_model(core, checkpoint_path):
    import torch
    # 不呼叫原 load_model，因為該函式會建立 optimizer / scheduler。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("epoch") != 4:
        raise ValueError("calibration 需要完成原正式 5 epochs 的 checkpoint")
    state = checkpoint["model_state_dict"]
    if any(t.is_floating_point() and t.dtype != torch.float64 for t in state.values()):
        raise ValueError("checkpoint 並非 float64；禁止靜默轉換訓練 precision")
    model = core.TranAD(38).double().cpu()
    model.load_state_dict(state, strict=True)
    if model.n_window != 10 or model.lr != 0.0001:
        raise ValueError("model configuration 與 formal protocol 不同")
    return model, freeze_model(model)


def score_prefix(core, model, prefix):
    import numpy as np
    import torch
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("prefix inference 前 model 必須 frozen + eval")
    if prefix.dtype != np.float64 or prefix.ndim != 2 or prefix.shape[1] != 38:
        raise ValueError("prefix 必須為 38-feature float64；不靜默轉換 precision")
    if any(p.device.type != "cpu" or p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError("calibration model 必須維持 CPU float64")
    raw = torch.from_numpy(prefix)
    windows = core.convert_to_windows(raw, model)
    if not torch.equal(windows[:, -1], raw):
        raise ValueError("calibration target timestamp 錯位")
    # 重用正式 inference；沒有 optimizer object，也沒有 training=True 路徑。
    with torch.no_grad():
        errors, _ = core.backprop(0, model, windows, raw, None, None, training=False)
    if errors.shape != prefix.shape or not np.isfinite(errors).all():
        raise ValueError("calibration errors 非有限值或長度錯誤")
    return errors, errors.mean(axis=1)


def fit_threshold(core, scores):
    from src.cross_machine import fit_source_threshold
    # 使用同一份 q / level / multiplier / retry implementation；輸入只含 target TRAIN prefix scores。
    threshold, info = fit_source_threshold(scores)
    if info["stream_score_count"] != 0:
        raise ValueError("POT 不可讀取 TEST stream")
    return threshold, info


def evaluate(core, scores, labels, threshold):
    from src.experiment import additional_metrics
    raw = scores > threshold
    adjusted = core.adjust_predicts(scores, labels, pred=raw.copy())
    values = core.calc_point2point(adjusted, labels)
    legacy = dict(zip(["f1", "precision", "recall", "TP", "TN", "FP", "FN", "legacy_binary_roc_auc"], values))
    evaluation, checked_raw = additional_metrics(scores, labels, threshold, legacy)
    import numpy as np
    np.testing.assert_array_equal(raw, checked_raw)
    return evaluation, raw, adjusted
