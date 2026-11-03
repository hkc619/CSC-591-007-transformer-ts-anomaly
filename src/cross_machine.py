"""SMD pilot：沿用 TranAD 計算，只擴充資料來源與 source-only threshold。"""
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter
import resource

import numpy as np
import torch

from preprocess import load_and_save, load_and_save2
from src.paths import resolve_roots, scalar_label_path
from src.parser import args

def cache_dir():
    return Path(args.output_dir)/"cache/processed/SMD"

from src.constants import lm
from src.experiment import additional_metrics, prepare_run
from src.pot import adjust_predicts, calc_point2point
from src.spot import SPOT


def validate_protocol(args):
    if args.dataset != 'SMD' or args.model != 'TranAD':
        raise ValueError('此實驗路徑僅支援 SMD / TranAD')
    if args.checkpoint or args.test or args.less:
        raise ValueError('pilot 必須從頭訓練完整 source splits，不接受 checkpoint/test/less')
    target = args.test_machine if args.train_machines else args.machine
    sources = list(args.train_machines or [args.machine])
    if not target or len(set(sources)) != len(sources):
        raise ValueError('必須指定 target，且 sources 不可重複')
    if args.train_machines and target in sources:
        raise ValueError('zero-shot target 不可包含在 source machines')
    if not args.train_machines and args.test_machine and args.test_machine != args.machine:
        raise ValueError('per-machine target 必須等於 --machine')
    for name in [*sources, target]:
        if not re.fullmatch(r'machine-\d+-\d+', name):
            raise ValueError(f'不合法的 machine 名稱：{name}')
        if not (resolve_roots().data/'train'/(name+'.txt')).is_file():
            raise ValueError(f'找不到 machine：{name}')
    return sources, target


def load_training_machine(machine):
    """只讀取指定 source 的 train，不讀取該 machine 的 test 或 labels。"""
    cache_dir().mkdir(parents=True, exist_ok=True)
    load_and_save('train', machine+'.txt', machine, str(resolve_roots().data))
    raw = np.load(cache_dir()/f'{machine}_train.npy')
    assert raw.ndim == 2 and raw.shape[1] == 38 and np.isfinite(raw).all()
    return torch.from_numpy(raw)


def windows_for_machine(raw, model, convert):
    windows = convert(raw, model)
    assert len(windows) == len(raw)
    # 逐一驗證所有視窗內每個時間點都屬於同一份 machine raw array。
    for start in range(0, len(raw), 512):
        end = min(start+512, len(raw))
        indices = torch.arange(start, end)[:,None] + torch.arange(-model.n_window+1, 1)[None,:]
        expected = raw[indices.clamp(min=0)]
        assert torch.equal(windows[start:end], expected), '發現跨 machine 或時間錯位的視窗'
    assert torch.equal(windows[:,-1], raw), 'target timestamp 錯位'
    return windows


def fit_source_threshold(scores):
    """SPOT 僅 fit source training scores；不執行 target stream。"""
    scores = np.asarray(scores)
    assert scores.ndim == 1 and np.isfinite(scores).all()
    level = lm[0]
    for attempts in range(10000):
        spot = SPOT(1e-5)
        spot.fit(scores, np.empty(0, dtype=np.float64))
        try:
            spot.initialize(level=level, min_extrema=False, verbose=False)
        except Exception:
            level *= .999  # 原 pot_eval 相同的 initialization retry 行為。
            continue
        threshold = float(spot.extreme_quantile * lm[1])
        if not np.isfinite(threshold):
            raise RuntimeError('source POT threshold 非有限值；停止，不以 target 調參')
        return threshold, dict(q=1e-5, requested_level=lm[0], effective_level=level,
                              multiplier=lm[1], initialization_attempts=attempts+1,
                              source_score_count=len(scores), stream_score_count=0,
                              fitted_extreme_quantile=float(spot.extreme_quantile))
    raise RuntimeError('source POT 初始化失敗；停止，不調整研究設定')


def run_smd_experiment(args, convert, load_model, backprop, save_model):
    sources, target = validate_protocol(args)
    protocol = getattr(args, 'protocol', 'v1')
    def stage(message):
        if getattr(args, 'progress', False):
            print(message, flush=True)

    stage(f'實驗：sources={sources} → target={target}；epochs={args.epochs}，seed={args.seed}')
    metadata = prepare_run(args)
    output = Path(args.output_dir)
    metadata.update(threshold_source='source_machine_training_scores',
                    source_machines=sources, target_machine=target,
                    normalization='none_added; supplied SMD values unchanged',
                    label_source='corrected_interpretation_label_union',
                    source_order=sources, training_shuffle=protocol == 'v2')
    # 在任何 target test 載入前完成全部 optimizer updates 和 threshold fitting。
    stage('[1/7] 載入 source training data')
    source_raw = [load_training_machine(name) for name in sources]
    model, optimizer, scheduler, epoch, history = load_model('TranAD', source_raw[0].shape[1])
    assert next(model.parameters()).device.type == 'cpu'
    assert next(model.parameters()).dtype == torch.float64
    chunks, offsets, origin_ids, offset = [], [], [], 0
    stage('[2/7] 各 machine 獨立 windowing 與邊界驗證')
    for source_id, (name, raw) in enumerate(zip(sources, source_raw)):
        chunk = windows_for_machine(raw, model, convert)
        chunks.append(chunk)
        offsets.append(dict(machine=name, start=offset, stop=offset+len(raw), windows=len(raw)))
        origin_ids.append(np.full(len(raw), source_id, dtype=np.int16))
        offset += len(raw)
        stage(f'  {name}：{len(chunk):,} windows，檢查通過')
    train_windows = torch.cat(chunks)
    train_raw = torch.cat(source_raw)
    del chunks, source_raw
    provenance = np.concatenate(origin_ids)
    assert len(provenance) == len(train_windows) == sum(x['windows'] for x in offsets)
    assert not args.train_machines or target not in [x['machine'] for x in offsets]
    (output/'window_provenance.json').write_text(json.dumps(dict(source_offsets=offsets,
        all_windows_checked=True, window_method='window each machine independently, then concatenate',
        target_in_training=bool(target in sources)), indent=2))
    model.train()
    stage(f'[3/7] 訓練：{len(train_windows):,} windows，共 {args.epochs} epochs')
    start = perf_counter()
    epoch_times = []
    for epoch in range(args.epochs):
        epoch_start = perf_counter()
        loss, learning_rate = backprop(epoch, model, train_windows, train_raw, optimizer, scheduler)
        history.append((float(loss), float(learning_rate)))
        epoch_times.append(perf_counter()-epoch_start)
        if protocol == 'v2':
            (output/'training_progress.json').write_text(json.dumps(dict(
                completed_epochs=epoch+1, total_epochs=args.epochs,
                epoch_training_seconds=epoch_times, losses=[x[0] for x in history],
                elapsed_training_seconds=perf_counter()-start,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024)), indent=2))
    training_seconds = perf_counter()-start
    save_model(model, optimizer, scheduler, epoch, history)
    stage(f'Checkpoint 已儲存：{args.save_checkpoint}')
    model.eval()
    stage('[4/7] 計算 source training scores')
    args.progress_desc = 'Source scores 推論'
    start = perf_counter()
    train_errors, _ = backprop(0, model, train_windows, train_raw, optimizer, scheduler, training=False)
    source_inference_seconds = perf_counter()-start
    source_scores = train_errors.mean(axis=1)
    stage('[5/7] Source-only POT fitting；此步驟無 batch 進度條')
    threshold, pot_info = fit_source_threshold(source_scores)
    metadata.update(threshold_value=threshold, threshold_protocol='source_only_frozen_SPOT_extreme_quantile',
                    pot_fit=pot_info, source_offsets=offsets,
                    target_loaded_after_threshold=True)
    (output/'threshold.json').write_text(json.dumps(dict(threshold_value=threshold,
        threshold_source=metadata['threshold_source'], source_machines=sources,
        target_machine=target, **pot_info), indent=2))
    print('source-only threshold 已固定：', threshold, flush=True)
    # 此後 target 只提供 test observations 與最終評估用的 labels。
    stage(f'[6/7] 載入並測試 target：{target}')
    shape = load_and_save('test', target+'.txt', target, str(resolve_roots().data))
    load_and_save2('labels', target+'.txt', target, str(resolve_roots().data), shape)
    test_raw = torch.from_numpy(np.load(cache_dir()/f'{target}_test.npy'))
    labels = np.load(cache_dir()/f'{target}_labels.npy')
    assert labels.shape == tuple(test_raw.shape) and np.isfinite(test_raw.numpy()).all()
    test_windows = windows_for_machine(test_raw, model, convert)
    args.progress_desc = f'Target {target} 測試'
    start = perf_counter()
    errors, reconstructions = backprop(0, model, test_windows, test_raw, optimizer, scheduler, training=False)
    inference_seconds = perf_counter()-start
    assert errors.shape == labels.shape and len(errors) == len(test_raw)
    assert np.isfinite(errors).all()
    stage('[7/7] 計算 Raw / PA / AUC 並儲存結果')
    scores, label_union = errors.mean(axis=1), labels.any(axis=1).astype(int)
    raw_predictions = scores > threshold
    adjusted = adjust_predicts(scores, label_union, pred=raw_predictions.copy())
    pa_values = calc_point2point(adjusted, label_union)
    legacy = dict(zip(['f1','precision','recall','TP','TN','FP','FN','legacy_binary_roc_auc'], pa_values))
    evaluation, raw_check = additional_metrics(scores, label_union, threshold, legacy)
    np.testing.assert_array_equal(raw_predictions, raw_check)
    np.testing.assert_array_equal(adjusted, adjust_predicts(scores, label_union, threshold))
    np.savez_compressed(output/'scores.npz', train_scores=train_errors, source_scores=source_scores,
                       source_machine_ids=provenance,
                       source_timestamps=np.concatenate([np.arange(x['windows']) for x in offsets]), test_scores=errors, scores=scores,
                       labels=labels, labels_final=label_union, predictions=reconstructions,
                       raw_predictions=raw_predictions, adjusted_predictions=adjusted)
    (output/'evaluation.json').write_text(json.dumps(evaluation, indent=2))
    scalar = np.loadtxt(scalar_label_path(resolve_roots().data,target), dtype=int)
    metadata.update(training_loss_per_epoch=[x[0] for x in history], training_seconds=training_seconds,
                    epoch_training_seconds=epoch_times,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024),
                    checkpoint_size_bytes=Path(args.save_checkpoint).stat().st_size,
                    test_inference_seconds=inference_seconds, train_score_inference_seconds=source_inference_seconds,
                    learning_rate=model.lr, window=model.n_window, batch_size=model.batch,
                    pot_parameters=list(lm), train_windows=len(train_windows), test_timestamps=len(test_raw),
                    scalar_vs_interpretation_disagreement_count=int(np.sum(scalar != label_union)),
                    leakage_checks=dict(target_in_optimizer_updates=bool(target in sources),
                        target_in_threshold_fitting=bool(target in sources),
                        normalization_fitted=False, target_used_for_testing=True,
                        target_train_loaded=bool(target in sources),
                        all_windows_single_machine=True, target_score_label_alignment=True))
    for split in ['test','interpretation_label','labels']:
        path=scalar_label_path(resolve_roots().data,target) if split=='labels' else resolve_roots().data/split/(target+'.txt')
        metadata['data_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(evaluation, indent=2), flush=True)
    print('已完成：', output, flush=True)
