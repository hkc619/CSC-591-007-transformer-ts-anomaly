"""Protocol v2：只排列完整 windows，不改變視窗內的時間順序。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def seeded_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def shuffled_loader(windows, batch_size, generator):
    dataset = TensorDataset(windows, torch.arange(len(windows), dtype=torch.int64))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator,
                      num_workers=0, drop_last=False)


def save_epoch_order(folder, epoch, batches, sample_count, generator_state):
    order = torch.cat(batches).numpy()
    assert len(order) == sample_count
    np.testing.assert_array_equal(np.sort(order), np.arange(sample_count))
    path = Path(folder)/'training_orders'
    path.mkdir(exist_ok=True)
    file = path/f'epoch_{epoch}.npz'
    with file.open('xb') as f:
        np.savez_compressed(f, sample_ids=order.astype(np.int32),
                            generator_state_before_epoch=generator_state.numpy())
    digest = hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
    info = dict(epoch=epoch, sample_count=sample_count, order_sha256=digest,
                unique_sample_count=len(order), sample_order_file=str(file),
                first_sample_ids=order[:20].tolist())
    (path/f'epoch_{epoch}.json').write_text(json.dumps(info, indent=2))
