# Anomaly Transformer Secondary Detector Robustness Experiment

日期：2026-11-07。這是既有 TranAD RQ1/RQ2/RQ3 之外的 additional robustness experiment，
不是新的主要 RQ，不重做 calibration，也不比較哪個 detector 的 absolute performance 較好。

## 1. 固定研究設計

Targets 事先固定為每個 SMD group 的前兩台，沒有根據 TranAD 結果挑選：

```text
machine-1-1  machine-1-2
machine-2-1  machine-2-2
machine-3-1  machine-3-2
```

每台各跑 per_machine 與 cross_machine，共 12 runs。Cross-machine 使用完整 SMD 的其餘
27 台 TRAIN，不是只使用另外五個 targets。觀察 PR-AUC gap = per-machine − cross-machine，
再與同六台的 TranAD gap 比較是否也出現 degradation。

輸出只有 continuous PR-AUC trapezoidal、Average Precision、ROC-AUC。沒有 threshold、
POT/SPOT、Raw F1、thresholded Precision/Recall、PA 或 target calibration。

| 設定 | 值 |
| --- | --- |
| Features / native window | 38 / 100 |
| d_model / heads / encoder layers / d_ff | 512 / 8 / 3 / 512 |
| Activation / dropout | GELU / 0.0 |
| Epochs / batch / evaluation batch | 10 / 256 / 256 |
| Initial LR / optimizer | 1e-4 / native Adam，betas=(0.9,0.999)、eps=1e-8、weight_decay=0 |
| k / score temperature / KL epsilon | 3 / 50 / 1e-4 |
| Seed / dtype | 1 / float32 |
| Intended formal device | CUDA；CPU 僅供本機 tests/dry-run 或明確指定的執行 |
| Validation / early stopping | 無；固定 10 epochs，使用最後 epoch checkpoint |

## 2. 官方來源、保留行為與 compatibility

Architecture 根據 [THUML 官方 repository](https://github.com/thuml/Anomaly-Transformer/tree/b0ee470c8012bff857fb600462aec6209c4a18d9)，
固定 commit `b0ee470c8012bff857fb600462aec6209c4a18d9`。MIT LICENSE 與 upstream blob IDs
保存在 `src/anomaly_transformer/LICENSE`、`PROVENANCE.json`。

核心 model、attention、embedding 由官方檔案改寫 import paths 後納入；沒有發明新的 attention。
Compatibility 變更只有：

1. 官方 distance tensor 的硬編碼 `.cuda()` 改為 nonpersistent registered buffer，讓 `.to(device)`
   正確管理 CPU/CUDA，保持 distance 數值與 checkpoint 中不保存該常數的行為。
2. Modern PyTorch 的 circular Conv1d padding 固定為 1；與原程式對 PyTorch 2.8 的結果相同，
   避免以版本字串比較決定 padding。
3. Import paths 改為 package-relative `attention.py`、`embedding.py`。

沒有改原 TranAD 的 PyTorch environment、architecture 或任何 source fingerprints。
新 GPU environment 必須獨立，不能把既有 CPU environment 降級或替換。

### 官方 objective / energy 保留

Minimax 使用原本兩個 losses：`rec − k × series_discrepancy` 與 `rec + k × prior_discrepancy`，
保留雙 backward、detach 方向與兩次 reconstruction gradient contribution。不能簡化成單一重建 loss。
LR 沿用官方 **epoch 結束後**調整的規則：第 1、2 epoch 使用 base LR，第 3 epoch 使用 base/2，
之後逐 epoch 減半。History 保存實際使用的 learning rate。

Native energy 保留 [官方 solver](https://github.com/thuml/Anomaly-Transformer/blob/b0ee470c8012bff857fb600462aec6209c4a18d9/solver.py)
的 `softmax(-temperature × layer-summed bidirectional association KL) × feature-mean reconstruction error`。
Scoring 時 layers 為 sum，不取 average；training discrepancy 才對 layers average。

### 刻意不採用的官方流程

官方 `train()` 呼叫 `vali(self.test_loader)`，並據此 early stop / 選 checkpoint；官方 test 還混合
train/test energies 選 percentile threshold 並做 PA。新 runner 完全不使用這些流程，也不匯入官方
Solver 或 legacy dataset loader。這不是原論文完整 reproduction。

## 3. Normalization 與 leakage isolation

Audit 發現現有 validated TranAD/SMD preprocessing 不另 fit normalization，使用供應的數值。
因此 AT 也採 **identity preprocessing**，只做 float32 conversion；不啟用官方 StandardScaler。

Metadata 明確記錄：

```text
normalization_source = identity_no_fitting_same_as_tranad
normalization_fitted = false
normalization_fit_machines = []
target_train_used_for_normalization = false
target_test_used_for_normalization = false
```

Per-machine 只載入 target TRAIN；cross-machine 只載入另外 27 台 TRAIN。Source file 不接受 alias
到 TEST 或 target TRAIN。沒有 validation loader；沒有 TEST-driven selection。
Dry-run/規劃階段會計算 TEST 文字行數以描述 coverage，但不將 TEST 數值輸入 optimizer 或 validation；
真正 TEST 數值與 interpretation labels 在全部 optimizer updates 完成後才載入。

原 label generator 的 import 會解析 TranAD CLI；AT adapter 僅在 import 時暫時隔離 argv，立刻還原。
呼叫原 `preprocess.load_and_save2`，只把 label cache 改放 AT run directory，不改 zero-based、
end-exclusive `start:end` 或 feature-label union。合法 interval 邊界另有 fail-loudly 檢查。

## 4. Machine-aware windows 與 timestamp mapping

官方 SMD loader 的預設 train stride=100；本實驗保留這個 native 完整 window policy。
各 machine 獨立產生完整 windows，再在 Dataset index space 合併：每個 item 只 slice 一個 array。
不 concatenate raw machines，也不 shuffle window 裡面的 timestamps。TRAIN 每台最後不足 100 點
的尾段不產生 window，dropped point count 與 source window offsets 都寫入 metadata。

TEST 必須完整覆蓋原始 N timestamps：

1. 從 0 開始每 100 點一個完整 window，各輸出 100 個 native energies。
2. 若有餘數 r，最後新增 `[N−100,N)` 的右對齊完整 window，只輸出 offset `[100−r,100)`。
3. 重疊區不平均、不重複計算到最終 score vector；每個 raw timestamp 恰好採用一個 energy。

例：N=257，windows=[0,100)、[100,200)、[157,257)。最後一個只採用 offsets 43–99，
對應 raw timestamps 200–256。Synthetic values 等於 timestamps 的 alignment test 驗證此 mapping。

```text
raw t → window_index[t] → window_offset[t] → native energy → score[t] → label[t]
```

`scores.npz` 保存 scores、labels、timestamps、window_index、window_offset、coverage。
`timestamp_mapping.json` 保存每個 window start/stop 與 emission slice。
`score_start_timestamp=0`、`label_slice=[0,N]`，沒有 warmup label 裁切或 roll。
TEST 少於 100 點時明確停止，不偷偷換 window 或 padding。
這是 native bidirectional window 的 offline scoring；不宣稱能在 timestamp t 當下完成 causal/online detection。

## 5. Reuse 與新 components

| Component | 做法 |
| --- | --- |
| Paths | 重用 src.paths.resolve_roots；保持 code/data/results 分離 |
| Labels | 重用原 preprocessing 的 interpretation-label generator |
| Shuffle | 重用 src.training_protocol.seeded_generator，完整 windows shuffle |
| Continuous metrics | 新增純 helper，與 validated additional_metrics 的三個公式一致，另做 exact numeric regression |
| Formal metrics wrapper | 保留原檔，避免執行 Raw/PA 與破壞 calibration snapshot fingerprints |
| Model / training / scoring | 獨立 AT package，來自 pinned upstream architecture / native formulas |
| Artifacts | 獨立 AT namespace；deterministic IDs + unique attempts；預設拒絕覆寫，retry 需明確指定 |

```text
RESULTS_ROOT/anomaly_transformer/
  runs/at-per-machine-machine-1-1-seed1/attempt-.../
  runs/at-cross-machine-machine-1-1-seed1/attempt-.../
  checkpoints/<run_id>/<attempt_id>/model.ckpt
  task_status/
  summaries/<unique-summary-id>/
  slurm-logs/
```

每 run 保存 status、metadata、evaluation、scores、training_history、run.log、launch、config、
window provenance、timestamp mapping、source snapshots 與 checkpoint。Metadata 記錄 source/target、
normalization/isolation flags、model/optimizer config、seed、versions、CUDA/GPU、Git commit、
data/source/checkpoint hashes、runtime、training/inference time、peak GPU allocated/reserved memory。
Native score 的 min/max/mean/std、nonzero/unique count、constant flag 也保存於 metadata，供 smoke review
判斷是否出現退化 score distribution；不自動依 TEST labels 調整任何設定。
CUDA 不可用時記錄 failed attempt，不靜默改 CPU；最新 attempt 不完整時不回退舊成功結果。

## 6. 本機測試與 dry-runs

詳細實測結果見 `ANOMALY_TRANSFORMER_LOCAL_VALIDATION.md`。測試使用 synthetic inputs；沒有執行
真實 SMD 的 10-epoch training。Orchestration test 會 mock training completion，僅驗證 stage order、
真正的 frozen inference、label reuse、artifact isolation 與 checkpoint reload，不能當作訓練成功證據。
Analysis test 的 12-record matrix 也是純 temporary fixtures，沒有執行其餘十個正式 runs。

本機重新執行：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../TranAD/.venv/bin/python -m unittest discover -s tests -p 'test_anomaly_transformer.py' -v
```

以實際 SMD 對 machine-1-1 的 local dry-run，已確認 per-machine 1 source / 284 train windows，
cross-machine 27 sources / 6,785 train windows；兩者 TEST=28,479 points / 285 windows。
沒有 model training 或 result directory 產生。上述 counts 是 native 100-point windows，不能與
TranAD 每 timestamp 一個 window 的 counts 混淆。

## 7. Hazel environment：獨立 GPU env，不破壞既有 CPU env

本機沒有 hpc/env.sh，也沒有連線 Hazel；無法從本機確認 private account association、GPU type
可用量、driver 或你現有 GPU environment。下列 account 採使用者提供的 csc591f26_gpu。

[NCSU GPU 文件](https://hpc.ncsu.edu/RunningJobs/Gpu.php) 要求 typed GPU request；本範本使用
gpu_partners / short_gpu / gpu:a100:1、1 node、1 task、4 CPUs、16G RAM、1 小時。
這些是待實測的初始 requests，不是 runtime/VRAM 保證；實際 account/partition/QOS 權限須在 Hazel 確認。

先 source 原 env，再額外設定獨立 GPU Python，原 PYTHON_BIN 保留：

```bash
cd "$PROJECT_ROOT"
source hpc/env.sh
export AT_PYTHON_BIN=/path/to/separate-cuda-environment/bin/python
mkdir -p "$RESULTS_ROOT/anomaly_transformer/slurm-logs"
```

若需要建立新 environment，請使用 Hazel 可用的 Python，不要修改舊 env。以下只是在**新的外部 env**
安裝 PyTorch 2.8 CUDA 12.8 wheel 的示範，driver 相容性仍需確認；本次未安裝任何套件：

```bash
python3.13 -m venv /path/to/separate-cuda-environment
export AT_PYTHON_BIN=/path/to/separate-cuda-environment/bin/python
"$AT_PYTHON_BIN" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
"$AT_PYTHON_BIN" -m pip install -r requirements-hpc.txt
```

版本安裝語法參考 [PyTorch 官方說明](https://pytorch.org/get-started/previous-versions/)。沒有新增
einops 等額外 dependency，也沒有安裝或降級本機／舊 CPU environment。

新的 HPC common 不 source CPU common.sh，也不清空 Slurm 的 CUDA_VISIBLE_DEVICES。
若 GPU environment 不支援 CUDA、deterministic operation 或 memory 不足，先停止並檢查，不自動改
dtype、window、batch 或 model。PYTHONHASHSEED、Python/NumPy/Torch/CUDA RNG、DataLoader seed、
deterministic algorithms、CUBLAS config 均固定；跨硬體 bitwise equality 仍未驗證。

## 8. 只準備兩個完整設定的 smoke

Smoke 使用正式相同的 10 epochs / batch 256 / native model，不偷偷截短 TRAIN/TEST。
Manifest task 0=machine-1-1 per_machine，task 1=machine-1-1 cross_machine。
以 phase=smoke_gate 標記，review 通過後可保留為 12-run matrix 的前兩個結果；不用重訓相同設定。
剩餘 array 範本只有 tasks 2–11，這次不提交。

先在合適 compute allocation 做兩個 dry-runs（不在 login node 訓練）：

```bash
"$AT_PYTHON_BIN" scripts/build_anomaly_transformer_manifest.py
"$AT_PYTHON_BIN" scripts/run_anomaly_transformer.py --hpc --device cuda \
  --target machine-1-1 --setting per_machine --dry-run
"$AT_PYTHON_BIN" scripts/run_anomaly_transformer.py --hpc --device cuda \
  --target machine-1-1 --setting cross_machine --dry-run
```

由使用者手動提交 **只有以下兩個** smoke jobs：

```bash
AT_SMOKE_TASK_ID=0 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/anomaly_transformer/slurm-logs/per-smoke-%j.out" \
  --error="$RESULTS_ROOT/anomaly_transformer/slurm-logs/per-smoke-%j.err" \
  hpc/anomaly_transformer_smoke.sbatch

AT_SMOKE_TASK_ID=1 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/anomaly_transformer/slurm-logs/cross-smoke-%j.out" \
  --error="$RESULTS_ROOT/anomaly_transformer/slurm-logs/cross-smoke-%j.err" \
  hpc/anomaly_transformer_smoke.sbatch
```

如 allocation 不支援範本的 A100/QOS，先確認核准資源，再以 sbatch resource overrides 調整；
不得在不同 setting 靜默使用不同 scientific hyperparameters。

```bash
squeue -u "$USER"
tail -f "$RESULTS_ROOT/anomaly_transformer/slurm-logs/per-smoke-JOBID.out"
tail -f "$RESULTS_ROOT/anomaly_transformer/slurm-logs/cross-smoke-JOBID.out"
```

把 JOBID 與 attempt 名稱換成實際值，兩個 run 都要檢查：

```bash
export AT_RUN="$RESULTS_ROOT/anomaly_transformer/runs/at-cross-machine-machine-1-1-seed1/實際attempt"
cat "$AT_RUN/status.json"
cat "$AT_RUN/training_history.json"
cat "$AT_RUN/metadata.json"
cat "$AT_RUN/evaluation.json"
cat "$AT_RUN/window_provenance.json"
cat "$AT_RUN/timestamp_mapping.json"
```

Review：10 epochs 完成、loss/gradient/energy finite、cross target TRAIN excluded、無 TEST validation、
machine boundaries、scores=labels count、三個 ranking metrics、training runtime 與 peak GPU memory。
真實 GPU smoke 及其 performance 目前全部 PENDING HAZEL VALIDATION。

## 9. 完整分析（只準備，不啟動剩餘 runs）

Failure checker 不自動重試；summary 缺任何一個 task 即停止。後續必須等兩個 smoke review 後才決定
提交剩餘十個 jobs；本文件不提供立即提交剩餘 array 的命令。

完整 summary 會產生 per-target PR-AUC/AP/ROC-AUC per/cross/gaps、aggregate、三組 descriptive tables，
以及同六台 TranAD vs AT PR-AUC gaps。TranAD table 預設唯讀：
`RESULTS_ROOT/summaries/formal_28_machine_results.csv`；CLI 支援 --tranad-results 指定其他現有 table。
輸入 file hash 保存，原 table 不改寫。Group 每組只有兩台，明確標示 descriptive only, very small subset。

不能以這份實驗直接說哪個 architecture 較準，也不能推廣成完整 SMD 的 AT 28-machine 結論。
限制包含 single seed、fixed six targets、不同 native window/objective、identity preprocessing、
固定 epochs 但不同 training volume/update counts，以及 TEST tail mapping 與官方丟尾段行為不同。

**STOP：先 review machine-1-1 per-machine 與 cross-machine 兩個 smoke。**
本次沒有 SSH、sbatch、正式 10-epoch training、Git commit 或 push。
