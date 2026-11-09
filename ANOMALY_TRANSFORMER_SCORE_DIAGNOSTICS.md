# Anomaly Transformer：唯讀 score decomposition 診斷

## 目的與範圍

使用者已完成 machine-1-1 的 per-machine 與 cross-machine smoke，並保存 checkpoint。
目前回報的 PR-AUC 約為 0.10054 / 0.09832，ROC-AUC 約為 0.5123 / 0.50035；
anomaly prevalence 約為 0.09365。這些是使用者提供的結果，本次未在本機重新驗證。

新增 `scripts/diagnose_anomaly_transformer_scores.py`，只從既有 checkpoint 進行 TEST inference，
用來區分 reconstruction ranking 薄弱，以及 native association weighting 抑制 reconstruction signal 的可能性。
本診斷不訓練、不建立 optimizer、不 backward、不 validation、不選 threshold、不 PA、不讀 target TRAIN。
不修改既有 `scoring.py`、`runtime.py`、任何訓練設定或原 smoke artifacts。

## 數學與 timestamp mapping

使用 checkpoint 保存的完整 model config、temperature=50、window=100 與原 eval batch size。
不提供調整 temperature、window 或 batch 的 CLI。

逐 window 計算：

```text
reconstruction = feature-mean squared reconstruction error
discrepancy = temperature × sum_layers(KL(series, normalized_prior)
                                     + KL(normalized_prior, series))
association_logits = -discrepancy
association_weight = softmax(association_logits, window time dimension)
native_energy = association_weight × reconstruction
```

KL 與 prior normalization 直接使用既有 scoring helper，保留 detach 及 layer SUM。
logits 的實際運算順序沿用原函式的 `-series_loss - prior_loss`。
四條 score 都經由既有 `TestWindows.plan` 與 `assemble_energies()` 回到 raw timestamps：

- `reconstruction_score`
- `association_weight_score`
- `negative_discrepancy_score`：只作 diagnostic-only 輔助。
- `native_energy_score`

完整 windows 依原方式 emit；最後的右對齊 window 只 emit 未覆蓋的 tail。
machine-1-1 應得到 285 個原始 TEST windows，每條 score 與 labels 都是 28,479 點，coverage 全為 1。
這仍是 bidirectional offline scoring，沒有改成 causal detector。

## 必須重現原結果

程式會檢查：

- 原 run 已完成、run/attempt/target/setting 一致。
- checkpoint path / SHA256、checkpoint target / setting / sources、保存的 config 與 metadata 一致。
- checkpoint floating tensors 是 float32。
- model、attention、embedding、scoring、runtime、dataset、continuous metrics 與 preprocessing 的原 source SHA256 一致。
- TEST 與 interpretation labels 的 path / SHA256 與原 inference 資料一致。
- 只讀 TEST，不呼叫會載入 TRAIN 的 `describe()` 或 `load_sources()`。
- `model.eval()`、`requires_grad_(False)`、`torch.no_grad()`，inference 前後 model state hash 相同。
- labels 與原 `scores.npz` 完全相同，timestamp mapping 完全相同。
- native score 與原 `scores.npz['scores']` 在 `rtol=1e-5, atol=1e-8` 下 allclose。
- 重算原 scores 與 diagnostic native scores 的三項 continuous metrics，兩者均須與原 `evaluation.json` 在相同 tolerance 下相符。
- 原 checkpoint、config、metadata、status、scores、evaluation、TEST 與原始 labels 在診斷前後 SHA256 不變。

任一項不符即拋出錯誤；不會默默忽略、重新訓練或改用其他 checkpoint。
如果已建立診斷 directory，失敗會記錄為 `diagnostic_metadata.json: status=failed`，
`native_score_reproduction_passed=false`。不可把該 attempt 當成有效診斷。

GPU / PyTorch 差異可能導致重現檢查失敗。即使 metrics 看起來接近，也不能繞過檢查；
先核對原 smoke 使用的 GPU、套件版本與 source fingerprint。

## 統計與輸出隔離

每次執行建立唯一的新目錄：

```text
$RESULTS_ROOT/anomaly_transformer/diagnostics/
  <original-run-id>/score-decomposition-<UTC timestamp>-<unique suffix>/
```

保存：

```text
diagnostic_metrics.json
diagnostic_stats.json
diagnostic_metadata.json
diagnostic_scores.npz
cache/processed/SMD/<target>_labels.npy
```

label cache 是呼叫原 validated `experiment.load_labels()` 所需的 I/O 適配；
只將 output directory 指向本次 diagnostic directory，不寫回原 smoke。
model、checkpoint 與 scores 的原檔案均只讀。重跑產生新 diagnostic attempt，不覆寫既有結果。

`diagnostic_metrics.json` 保存四條 stream 的 PR-AUC trapezoidal、AP、ROC-AUC，以及 native 重現差異。
negative discrepancy 明確標為 diagnostic-only；沒有 threshold、Raw F1、PA 或 POT。

`diagnostic_stats.json` 保存：

- 最終逐 timestamp association weights 的 min/max/mean/std/median、zero/nonzero/unique count、
  fraction zero、低於 1e-30/1e-20/1e-12、高於 0.5/0.9 的比例。
- 完整原始 TEST windows 的 association weight 統計，與最終逐 timestamp 統計分開。
- 每個原始 window 的 entropy、effective support、top1 mass 與 logit range，彙整 mean/median/min/max/std。
- discrepancy / logits 的 min/max/mean/std/median，以及 reconstruction 分布、zero/unique count。

entropy 使用 natural log，僅統計階段轉 float64，log clamp 使用 float32.tiny；
模型、softmax 與 score 計算均保留 float32。完整 window 統計包含 tail 的重疊 context，
逐 timestamp score metrics 則仍精確計入每個 timestamp 一次。

NPZ 保存四條 timestamp-aligned score、labels、coverage、window index/offset/timestamps、
per-window entropy/support/top1/range、window plan。metadata 保存 checkpoint hash、原 run / source commit、
diagnostic commit/dirty flag、device/GPU/versions/config、freeze hashes 與 source 檔案 fingerprints。

## 本機 regression checks

只執行 synthetic tests，不啟動任何 training：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../TranAD/.venv/bin/python -m unittest discover \
  -s tests -p 'test_anomaly_transformer_diagnostics.py' -v
```

測試包括：與 `native_energy()` 相同的 layer-summed 分解、softmax sum≈1、
禁止 optimizer/backward/training calls、四條 28,479 點 synthetic mapping、
原 scores/labels/mapping/metrics 不符時拒絕、namespace / symlink 隔離，以及 synthetic checkpoint 唯讀 inference。
整合 fixture 使用小型未訓練模型，只測 I/O 與控制流程；不代表 Hazel 上原 smoke 已重現。

本次本機執行結果：7 tests，0.526 秒，全部通過。Synthetic integration 的 native score
max absolute difference 為 0，並測到刻意損壞 reference scores 時 runner 失敗、
只留下 failed metadata，不輸出成功的 diagnostic metrics。
真實 per-machine / cross-machine checkpoint 的重現與科學診斷：PENDING HAZEL。

## Hazel 手動執行：只診斷兩個已存在的 checkpoint

若新檔還沒 push，請先 review 後自行 commit/push，再到 Hazel 更新。
以下命令由使用者執行；本次沒有 SSH、提交 job 或執行 Hazel inference。

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
export AT_PYTHON_BIN=/path/to/the-existing-AT-CUDA-environment/bin/python

# 指定已完成的確切 attempt，禁止用模糊 glob 或自動改選其他結果。
export AT_PER_RUN="$RESULTS_ROOT/anomaly_transformer/runs/at-per-machine-machine-1-1-seed1/實際已完成attempt目錄"
export AT_CROSS_RUN="$RESULTS_ROOT/anomaly_transformer/runs/at-cross-machine-machine-1-1-seed1/實際已完成attempt目錄"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=1
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

沿用兩個 smoke 已成功使用的 GPU environment 與 Slurm 資源。
以下資源名稱沿用現有 AT script；若 smoke 使用不同的有效 allocation，應使用該 allocation。
`srun` 取得 compute-node GPU，不在 login node 執行 inference：

```bash
srun --account=csc591f26_gpu --partition=gpu_partners --qos=short_gpu \
  --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 \
  "$AT_PYTHON_BIN" scripts/diagnose_anomaly_transformer_scores.py \
  --hpc --device cuda --run-dir "$AT_PER_RUN"

srun --account=csc591f26_gpu --partition=gpu_partners --qos=short_gpu \
  --gres=gpu:a100:1 --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=00:30:00 \
  "$AT_PYTHON_BIN" scripts/diagnose_anomaly_transformer_scores.py \
  --hpc --device cuda --run-dir "$AT_CROSS_RUN"
```

每個命令會印出 `Diagnostic output:` 的完整路徑，以及 TEST batch 進度。
把實際路徑填入下方，檢視並提供三份 JSON 給 ChatGPT 審查：

```bash
export AT_DIAG=/實際印出的score-decomposition目錄
cat "$AT_DIAG/diagnostic_metadata.json"
cat "$AT_DIAG/diagnostic_metrics.json"
cat "$AT_DIAG/diagnostic_stats.json"
```

沒有新增 dry-run：本程式是依要求重跑 frozen TEST inference 的診斷入口。
CLI 結構與控制流程由 synthetic regression check 檢查；真實 checkpoint 重現仍待 Hazel 執行。

## 結果解讀與停止點

先確認 `status=completed`、`native_score_reproduction_passed=true`、parameters/source inputs 不變，
再比較 reconstruction 與 native energy 的三項 ranking metrics。

- reconstruction 排序明顯優於 native，且 weights 極端集中／多數為零：支持 weighting 抑制訊號的解釋。
- reconstruction 與 native 都接近無 discrimination：表示 reconstruction 本身也可能薄弱；仍需搭配其他分解統計判斷。
- softmax 集中本身不能證明它造成低 ranking，必須與 reconstruction/native metric 差異一起解讀。

association weights 是 window-relative probability；不同 windows 的 logits offset / normalization 不同，
negative discrepancy 與 weight 的全域排序不保證相同。
這些是固定兩個 checkpoint 的 post-hoc 診斷，不是新 detector 結果，不用 TEST labels 調 temperature 或挑設定。

完成兩個 diagnostics 後停止：不啟動 tasks 2–11、不重訓、不改 temperature/window/batch、也不覆寫原 smoke。
