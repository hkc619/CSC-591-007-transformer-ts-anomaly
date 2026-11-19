# Universal Shared Checkpoint：Stage 1

本階段只新增 `28 台 SMD TRAIN → 一個 TranAD checkpoint`。沒有 held-out target，沒有
TEST inference、metrics、threshold 或 calibration。這不是新的 zero-shot 評估：未來若測試
這 28 台，必須說明各台 TRAIN 已參與 shared representation training。

目前完成實作、本機 synthetic regression 與真實 SMD TRAIN-only dry-run。
**尚未執行完整 SMD universal training，未連線 Hazel、未提交 job；真實 universal checkpoint 尚未產生。**

## 固定 protocol 與重用範圍

| 項目 | 設定 |
| --- | --- |
| Canonical machines | 直接讀取 `configs/formal_protocol_v2.json` 的 28 台及原排序 |
| Model / dataset | TranAD / SMD |
| Epochs / batch / window | 5 / 128 / 10 |
| 初始 learning rate | 0.0001 |
| Optimizer / scheduler | 原 `main.load_model()` 的 AdamW、StepLR，完全不修改 |
| Loss / architecture | 原 `main.backprop(training=True)`、`src.models.TranAD` |
| Device / dtype | CPU / float64 |
| Sampling | Natural source-size weighting；每台保留所有 windows |
| Shuffle | 原 seeded generator / shuffled DataLoader；只排列完整 windows |
| Seed | CLI 支援 1、2、3；Stage 1 只驗證及準備提交 seed 1 |
| Preprocessing | 原 `preprocess.load_and_save('train', ...)`，沒有新增 normalization |
| Checkpoint | 原 `main.save_model()` / `main.load_model()` 與原五個 serialization keys |

為什麼另設入口：原 `run_smd_experiment()` 要求 held-out target，且會自動執行 source scoring、
POT、TEST。本階段直接重用其中的 TRAIN loader、window validator，以及原 `main.py` 的初始化、
每 epoch training 計算與 save/load；另加一層 training-only orchestration，完成 reload 後即結束。
沒有假造 target，也沒有修改原 scientific core 或舊 CLI。

`src.models` 有 import-time `manual_seed(1)`；新入口在 import 後，依原 `prepare_run()` 的方式設定
Python、NumPy、PyTorch seed 及獨立 DataLoader generator，確保 CLI seed 不被 import 覆蓋。
不額外改變 PyTorch deterministic 設定；跨 PyTorch 版本或硬體的 bitwise equivalence 尚未確認。

## Machine-isolated windowing 與 provenance

每台依序：讀 TRAIN → 原 `convert_to_windows()` → 原 `windows_for_machine()` 完整比對所有內容
→ 保存該台 windows。28 台都完成後，才 `torch.cat(window_chunks)`，然後 seeded shuffle。
為配合原 `backprop(data, dataO, ...)` 介面，raw tensors 也會在這個階段合併；它們不會再被 windowing。

原 mapping 不變：`window(t) = raw[max(0,t-9):t+1]`，左側不足時重複該台 `raw[0]`；
最後一項及 training target 都是該台 `raw[t]`。每 raw point 產生一個 padded window，dropped points=0。
DataLoader 的 sample ID 是 concatenated window index；不會打亂 window 內 timestamps。

`window_provenance.json` 記錄各 machine 的 raw length、window count、起始 offset、exclusive end offset、
TRAIN path、檔案及 array SHA256、dropped points，以及逐 window 比對及 target equality 的確認。
全域記錄 `all_windows_single_machine=true`、`windowed_before_concatenation=true`。
每個 shuffled sample ID 都可由 offset range 找回 machine，再以 `sample_id-offset_start` 找回 timestamp。
每 epoch 原有 `training_orders/epoch_*.npz` 和 SHA256 繼續保存完整 sample order。

dry-run 只檢查 TRAIN 檔、shape、finite、長度與 hash，並**預測** window counts；
實際逐 window 內容驗證在 training 前進行，不能把 dry-run 說成已實測完整 window tensors。
Synthetic tests 則已實際驗證 window contents、padding、boundary、target 及 shuffled provenance。

## 真實本機 SMD TRAIN-only dry-run

此表來自本機實際 TRAIN 檔；不讀 TEST 或 labels。Window count 沿用一 timestamp 一 window。
HPC 執行時會重新檢查檔案並保存自己的 hashes；未宣稱本機和 Hazel dataset 已逐檔比對。

| Machine | Raw TRAIN points / windows |
| --- | ---: |
| machine-1-1 | 28479 |
| machine-1-2 | 23694 |
| machine-1-3 | 23702 |
| machine-1-4 | 23706 |
| machine-1-5 | 23705 |
| machine-1-6 | 23688 |
| machine-1-7 | 23697 |
| machine-1-8 | 23698 |
| machine-2-1 | 23693 |
| machine-2-2 | 23699 |
| machine-2-3 | 23688 |
| machine-2-4 | 23689 |
| machine-2-5 | 23688 |
| machine-2-6 | 28743 |
| machine-2-7 | 23696 |
| machine-2-8 | 23702 |
| machine-2-9 | 28722 |
| machine-3-1 | 28700 |
| machine-3-2 | 23702 |
| machine-3-3 | 23703 |
| machine-3-4 | 23687 |
| machine-3-5 | 23690 |
| machine-3-6 | 28726 |
| machine-3-7 | 28705 |
| machine-3-8 | 28703 |
| machine-3-9 | 28713 |
| machine-3-10 | 23692 |
| machine-3-11 | 28695 |
| **合計：28 台** | **708405** |

## 本次 I/O 適配與輸出保護

新增檔案：

- `configs/universal_training.json`：指向原 formal config，不另列 machines，也不引入 threshold 設定。
- `src/universal_training.py`：TRAIN-only inspection、組裝、provenance、training orchestration、reload、attempt I/O。
- `scripts/run_universal_training.py`：獨立 CLI；不放寬既有 formal parser。
- `hpc/universal_train.sbatch`：單一 CPU job；Stage 1 明確拒絕 seed 2/3，不是 array。
- `tests/test_universal_training.py`：focused synthetic tests。
- 本文件；`hpc/README_HPC.md` 只新增導引。

Adapter 在 fresh Python process 以 `src.parser.args` 注入明確 TRAIN-only args，再 import `main`；
其 main guard 不會執行。`machine=None`、`test_machine=None`、sources=28 台。
`eval_batch_size` 等舊介面欄位只為相容而保留，並不觸發 evaluation。原 import chain 仍會載入
POT 函式定義，但此入口不呼叫 fitting 或 evaluation。
`preprocess.output_folder` 指向當次 attempt 專用 cache；不寫舊 formal cache。
入口先用共用 `resolve_roots()` 檢查外部 paths；TRAIN 及 output symlink 拒絕使用，避免 split／namespace 別名。

每次 outputs：

```text
RESULTS_ROOT/universal/
├── runs/universal_TranAD_SMD_v2_e5_bs128_seed1/
│   ├── active.lock                    # 僅執行中存在
│   └── attempt-<UTC timestamp>-<uuid>/
│       ├── launch.json
│       ├── metadata.json
│       ├── train_manifest.json
│       ├── window_provenance.json
│       ├── epoch_1.json … epoch_5.json
│       ├── training_history.json
│       ├── training_orders/
│       ├── source_snapshot/
│       ├── cache/processed/SMD/
│       ├── run.log
│       ├── status.json
│       └── validation.json
├── checkpoints/universal_TranAD_SMD_v2_e5_bs128_seed1/
│   └── attempt-<相同 ID>/model.ckpt
└── slurm-logs/
```

`metadata.json` 保存 seed、28 sources、無 target、所有 training settings、總 windows、optimizer、
scheduler、dependency versions、Git commit/dirty status 和 source hashes。
Checkpoint 仍只有原 keys：`epoch`、`model_state_dict`、`optimizer_state_dict`、
`scheduler_state_dict`、`accuracy_list`。
`validation.json` 在原 loader 成功 reload 後，確認 model/optimizer/scheduler/epoch/history 完全相同，
checkpoint SHA256 未改變，才將 status 設為 completed。

預設 run parent 已存在便拒絕重跑；`--retry` 只允許失敗 run 新增 attempt，保留所有舊檔。
成功 run 即使加 `--retry` 也拒絕重訓。每個 seed 有不同 logical ID。
程序被 Slurm 強制終止可能留下 lock 或 running status；先確認 job 已終止並人工檢查，不能直接重送。
沒有自動 resume optimizer；失敗重試從頭訓練，且保留失敗 checkpoint（若已寫出）。

## 本機驗證與 regression

測試使用外部 temporary directory 的 28 台不等長 synthetic TRAIN；沒有 TEST/labels directory。
只有 seed 1 執行一次 tiny 5-epoch training，確認原 save/load 的完整 roundtrip。
這不等於真實 708405-window training，也不代表 Hazel 已驗證。

本次實際結果：universal 9 項通過（最後一次 46.110 秒）、既有 seed replication 12 項通過
（99.078 秒）、calibration protocol 4 項通過（0.012 秒），合計 25 項。
`bash -n hpc/universal_train.sbatch` 與 `git diff --check` 亦成功。
本機使用既有 sibling `../TranAD/.venv/bin/python`，沒有建立或修改 Python environment。

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tranad-universal-test-mpl"
# PYTHON_BIN 指向已有 TranAD CPU environment；不安裝或降級任何套件。
"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_training.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_seed_replication.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_calibration_protocol.py -v
bash -n hpc/universal_train.sbatch
git diff --check
```

涵蓋 canonical 28 台／無 duplicates／無 target exclusion、自然 weighting、offset/counts、seed 傳遞、
independent windowing／邊界及 target、seed-1 shuffled order 重現、錯誤 window 拒絕、missing/extra/NaN/Inf/
features 拒絕、dry-run 不 import model 或寫 outputs、namespace／attempt 保護、TRAIN-only 五 epoch
及 checkpoint reload、既有 scientific files 與 HEAD 逐字不變。
既有 seed replication 及 calibration protocol regression 保留原 assertions。

## 使用者手動執行

先 Git review；本次沒有 commit 或 push。更新 Hazel 後，沿用既有外部 CPU environment：

```bash
source hpc/env.sh
cd "$PROJECT_ROOT"
```

在合適的 compute allocation 做 TRAIN-only dry-run；會掃描全部 TRAIN，但不建立 result/cache/model：

```bash
"$PYTHON_BIN" scripts/run_universal_training.py --hpc --seed 1 --dry-run
```

本機可用相同 CLI，加上明確外部 dataset/results roots，例如本 repository 的 sibling dataset：

```bash
"$PYTHON_BIN" scripts/run_universal_training.py \
  --data-root ../TranAD/data/SMD \
  --results-root ../transformer_ts_anomaly-results \
  --seed 1 --dry-run
```

真正 training CLI（Hazel 必須在 compute allocation，不在 login node）：

```bash
"$PYTHON_BIN" -u scripts/run_universal_training.py --hpc --seed 1
```

或手動提交唯一 seed-1 batch job；同一次訓練只選一種啟動方式：

```bash
mkdir -p "$RESULTS_ROOT/universal/slurm-logs"
SEED=1 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/slurm-logs/train-seed1-%j.out" \
  --error="$RESULTS_ROOT/universal/slurm-logs/train-seed1-%j.err" \
  hpc/universal_train.sbatch
```

沿用 `csc591f26_cpu / compute_partners / short`、1 node、1 task、4 CPUs、16G、2 小時。
708405 windows 比 machine-1-1 的 27-source 679926 windows 多約 4.2%。
單一 dense window tensor 約 2.01 GiB；concatenation 時各 machine tensors 與合併 tensor 可能同時存在，
再加 raw arrays、model、batch 與 runtime overhead。因此沿用既有 16G，不擅改 precision 或 architecture。
實際 universal walltime／peak RSS 尚未量測，2 小時不是完成保證；請檢查 Slurm accounting。

```bash
squeue -u "$USER"
# 將下列 JOB_ID／ATTEMPT 換成實際值。
tail -f "$RESULTS_ROOT/universal/slurm-logs/train-seed1-JOB_ID.out"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
cat "$RESULTS_ROOT/universal/runs/universal_TranAD_SMD_v2_e5_bs128_seed1/ATTEMPT/status.json"
cat "$RESULTS_ROOT/universal/runs/universal_TranAD_SMD_v2_e5_bs128_seed1/ATTEMPT/validation.json"
cat "$RESULTS_ROOT/universal/runs/universal_TranAD_SMD_v2_e5_bs128_seed1/ATTEMPT/training_history.json"
```

stdout／run.log 顯示 windowing 進度與原 batch tqdm 的 epoch、batch、loss。
完成後只 review checkpoint、provenance 與 validation；**停止 Stage 1，不執行 seeds 2/3、TEST、threshold 或 calibration。**
