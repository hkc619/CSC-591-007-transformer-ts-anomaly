"""固定 epochs、官方 minimax 雙 backward；完全沒有 validation/test-based selection。"""
import hashlib
import random
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.anomaly_transformer.dataset import TestWindows, assemble_energies
from src.anomaly_transformer.scoring import native_energy, training_objectives
from src.training_protocol import seeded_generator


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def training_loader(dataset, batch_size, seed=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      generator=seeded_generator(seed), num_workers=0, drop_last=False)


def minimax_step(model, inputs, optimizer, k=3):
    optimizer.zero_grad()
    output, series, priors, _ = model(inputs)
    values = training_objectives(inputs, output, series, priors, k)
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError("training loss NaN/Inf")
    # Preserve upstream two reconstruction contributions and detach/minimax gradients.
    values[0].backward(retain_graph=True)
    values[1].backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("training gradient NaN/Inf")
    optimizer.step()
    return [float(value.detach()) for value in values]


def train_fixed(model, dataset, config, device, progress_callback):
    options = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
        betas=tuple(options["betas"]), eps=options["eps"], weight_decay=options["weight_decay"])
    loader = training_loader(dataset, config["batch_size"], config["seed"])
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        start = perf_counter()
        total, seen = np.zeros(5), 0
        order = hashlib.sha256()
        lr = optimizer.param_groups[0]["lr"]
        batches = tqdm(loader, desc=f"AT epoch {epoch+1}/{config['epochs']}", unit="batch")
        for windows, ids in batches:
            order.update(ids.numpy().tobytes())
            values = minimax_step(model, windows.to(device=device, dtype=torch.float32), optimizer, config["k"])
            total += np.asarray(values)
            seen += 1
            batches.set_postfix(reconstruction=f"{values[2]:.6f}")
        means = total / seen
        entry = dict(epoch=epoch + 1, loss1=float(means[0]), loss2=float(means[1]),
            reconstruction=float(means[2]), series_loss=float(means[3]), prior_loss=float(means[4]),
            learning_rate=lr, seconds=perf_counter()-start, batches=seen, sample_order_sha256=order.hexdigest())
        history.append(entry)
        progress_callback(history)
        print(entry, flush=True)
        # Official solver adjusts at END of epoch: epoch 1 -> base lr; epoch 2 -> base/2.
        for group in optimizer.param_groups:
            group["lr"] = config["learning_rate"] * .5 ** epoch
    return history


def infer(model, raw, config, device):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = TestWindows(raw, config["model"]["win_size"])
    loader = DataLoader(dataset, batch_size=config["eval_batch_size"], shuffle=False, num_workers=0)
    energies = []
    with torch.no_grad():
        for windows, ids in tqdm(loader, desc="AT TEST native energy", unit="batch"):
            windows = windows.to(device=device, dtype=torch.float32)
            output, series, priors, _ = model(windows)
            energy = native_energy(windows, output, series, priors, config["temperature"])
            if not torch.isfinite(energy).all():
                raise FloatingPointError("TEST energy NaN/Inf")
            energies.extend(energy.cpu().numpy())
    scores, mapping = assemble_energies(energies, dataset.plan, len(raw))
    return scores, mapping, dataset.plan
