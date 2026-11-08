"""SMD identity preprocessing、machine-aware 完整 windows 與一對一 TEST emission plan。"""
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_values(path):
    raw = np.genfromtxt(path, delimiter=",", dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 38 or not np.isfinite(raw).all():
        raise ValueError(f"SMD 必須為 finite [N,38]：{path}")
    values = raw.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("float32 conversion overflow")
    return values


def select_sources(canonical, target, setting):
    if target not in canonical or setting not in ("per_machine", "cross_machine"):
        raise ValueError("未知 target / setting")
    sources = [target] if setting == "per_machine" else [m for m in canonical if m != target]
    if setting == "cross_machine" and (target in sources or len(sources) != 27):
        raise ValueError("cross-machine 必須使用另外 27 台")
    return sources


def load_sources(data_root, sources, target, setting, reader=load_values):
    if len(sources) != len(set(sources)):
        raise ValueError("sources 不可重複")
    if setting == "cross_machine" and target in sources:
        raise ValueError("禁止載入 cross-machine target TRAIN")
    if setting == "per_machine" and sources != [target]:
        raise ValueError("per-machine sources 必須只有 target")
    # 不採用官方 StandardScaler；與既有 TranAD 一樣，不 fit 任何資料的 normalization。
    base = Path(data_root).resolve()
    train_root = (base / "train").resolve()
    if train_root == (base / "test").resolve():
        raise ValueError("TRAIN directory 不可 alias TEST directory")
    arrays = {}
    for machine in sources:
        path = base / "train" / (machine + ".txt")
        if not path.resolve().is_relative_to(train_root):
            raise ValueError("TRAIN file 不可 alias 到 TEST 或其他目錄")
        if setting == "cross_machine" and path.resolve() == (base / "train" / (target + ".txt")).resolve():
            raise ValueError("source file 不可 alias target TRAIN")
        arrays[machine] = reader(path)
    return arrays


def normalization_metadata():
    return dict(normalization_source="identity_no_fitting_same_as_tranad",
                normalization_fit_machines=[], normalization_fitted=False,
                target_train_used_for_normalization=False, target_test_used_for_normalization=False)


class MachineWindows(Dataset):
    """邏輯上 concatenate 每台完整 windows；不建立任何跨 machine raw sequence。"""
    def __init__(self, arrays, window=100, stride=100):
        self.arrays, self.names = arrays, list(arrays)
        self.window, self.stride = window, stride
        self.offsets, self.stops = [], []
        total = 0
        for name, raw in arrays.items():
            if raw.ndim != 2 or raw.shape[1] != 38 or raw.dtype != np.float32:
                raise ValueError("training arrays 必須是 [N,38] float32")
            count = (len(raw) - window) // stride + 1
            if count < 1:
                raise ValueError(f"{name} 少於一個完整 window")
            self.offsets.append(dict(machine=name, start=total, stop=total + count,
                                     train_length=len(raw), windows=count,
                                     dropped_tail_points=len(raw) - ((count - 1) * stride + window)))
            total += count
            self.stops.append(total)
        if not total:
            raise ValueError("sources 不可為空")

    def __len__(self):
        return self.stops[-1]

    def provenance(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        machine_id = bisect_right(self.stops, index)
        local = index - self.offsets[machine_id]["start"]
        return self.names[machine_id], local * self.stride

    def __getitem__(self, index):
        name, start = self.provenance(index)
        stop = start + self.window
        if stop > len(self.arrays[name]):
            raise RuntimeError("window 跨越 machine boundary")
        window = self.arrays[name][start:stop]
        if window.shape != (self.window, 38):
            raise RuntimeError("window shape 錯誤")
        return torch.from_numpy(window), index


def evaluation_plan(length, window=100):
    if length < window:
        raise ValueError("TEST 少於一個完整 native window；停止，不靜默裁切 labels")
    plan = [dict(start=start, stop=start + window, emit_start=0, emit_stop=window)
            for start in range(0, length - window + 1, window)]
    remainder = length % window
    if remainder:
        plan.append(dict(start=length - window, stop=length,
                         emit_start=window - remainder, emit_stop=window))
    return plan


class TestWindows(Dataset):
    def __init__(self, raw, window=100):
        self.raw = raw
        self.plan = evaluation_plan(len(raw), window)

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        row = self.plan[index]
        return torch.from_numpy(self.raw[row["start"]:row["stop"]]), index


def assemble_energies(energies, plan, length):
    scores = np.empty(length, dtype=np.float32)
    coverage = np.zeros(length, dtype=np.int32)
    window_index = np.empty(length, dtype=np.int64)
    window_offset = np.empty(length, dtype=np.int64)
    if len(energies) != len(plan):
        raise ValueError("energy / window count 不一致")
    for index, row in enumerate(plan):
        energy = np.asarray(energies[index])
        if energy.shape != (row["stop"] - row["start"],) or not np.isfinite(energy).all():
            raise ValueError("native energy shape / finite check 失敗")
        lo, hi = row["emit_start"], row["emit_stop"]
        timestamps = np.arange(row["start"] + lo, row["start"] + hi)
        scores[timestamps] = energy[lo:hi]
        coverage[timestamps] += 1
        window_index[timestamps] = index
        window_offset[timestamps] = np.arange(lo, hi)
    if not np.all(coverage == 1):
        raise ValueError("TEST timestamp 必須精確 emit 一次")
    return scores, dict(timestamps=np.arange(length), window_index=window_index,
                        window_offset=window_offset, coverage=coverage)
