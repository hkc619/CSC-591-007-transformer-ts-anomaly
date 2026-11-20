# Universal Shared Checkpoint：Stage 2

更新日期：2026-11-20。

本文件記錄 seed-1 universal TranAD checkpoint 的 **continuous-score evaluation** 實作、
本機驗證證據與 Hazel 手動操作方式。雖然沿用使用者指定的 `UNIVERSAL_TRAINING_STAGE2.md`
檔名，Stage 2 只有 frozen inference，不進行 training。

## 1. 目前進度與證據範圍

| 項目 | 狀態與證據 |
| --- | --- |
| Stage-1 universal training | 使用者已回報 Hazel 訓練及 validation 完成 |
| Stage-2 code | 已實作，尚未 commit／push |
| 本機 Stage-2 synthetic tests | 14 項通過 |
| 既有相關 regression | 60 項通過；合計 74 項 |
| CLI help、shell syntax、Git whitespace | 已檢查 |
| 真實 Stage-1 checkpoint 路徑／SHA256 | 本機沒有該 artifacts，待 Hazel dry-run 解析與核對 |
| 真實 machine-1-1 smoke | PENDING HAZEL VALIDATION |
| 真實 28-machine evaluation | PENDING HAZEL VALIDATION |
| 真實 Per／LOMO／Universal comparison | PENDING HAZEL VALIDATION |
| Hazel 連線／job submission | Codex 未執行 |

本機用真實資料路徑嘗試 dry-run 時，resolver 找到 **0 個 completed Stage-1 attempts**，因此拒絕執行。
這表示本機 RESULTS_ROOT 沒有 Hazel 上的 Stage-1 artifacts；程式沒有回退到 per-machine／LOMO
checkpoint，也沒有產生真實 evaluation 結果。

以下測試結果來自前一輪實作時的執行紀錄；本次新增文件沒有重新執行 tests 或 experiments。

## 2. 研究問題與定位

本階段只回答：

> 同一個 universal TranAD checkpoint，能否在完整 SMD 28 台 machines 上保留 anomaly-score ranking quality？

三種 setting 的差異：

| Setting | Training | TEST evaluation | Model 數量 |
| --- | --- | --- | ---: |
| Per-machine | 每台自己的 TRAIN | 該台 TEST | 28 |
| LOMO cross-machine | 其他 27 台 TRAIN | held-out target TEST | 28 個研究用模型 |
| Universal checkpoint | 全部 28 台 TRAIN | 同一 checkpoint 分別評估各台 TEST | **1** |

Universal 的各 target TRAIN 已參與 Stage-1 shared representation training，因此 **不是 zero-shot**。
本階段可使用「universal checkpoint evaluation」或「shared-model deployment」等名稱。
Stage 2 不回答單一 global threshold 是否可行，也不根據 TEST 結果選 checkpoint 或調整模型。

## 3. 固定 scientific protocol

| 項目 | 設定 |
| --- | --- |
| Model／dataset | TranAD／SMD |
| Checkpoint | Stage-1 completed seed-1 universal checkpoint |
| Targets | 原 `configs/formal_protocol_v2.json` 的全部 28 台及其排序 |
| Window | 10 |
| Eval batch | 128 |
| Seed | 1 |
| Device／dtype | CPU／float64 |
| Model mode | `eval()`、`requires_grad=False`、`torch.no_grad()` |
| Optimizer steps | 0；不建立 optimizer，不呼叫 backward |
| Preprocessing | 沿用 formal TranAD 的 TEST conversion，沒有新增 normalization |
| Labels | Validated interpretation-label feature union |
| Primary metric | PR-AUC trapezoidal |
| Supplemental metrics | Average Precision、ROC-AUC |

沒有新增或執行 universal threshold fitting、Raw Precision／Recall／F1、PA、POT/SPOT fitting、
target calibration 或 seeds 2／3 evaluation。

## 4. Checkpoint resolution 與不可變性

Stage-1 logical ID：

```text
universal_TranAD_SMD_v2_e5_bs128_seed1
```

預期原始 checkpoint 路徑：

```text
$RESULTS_ROOT/universal/checkpoints/
  universal_TranAD_SMD_v2_e5_bs128_seed1/<stage1-attempt-id>/model.ckpt
```

`resolve_stage1()` 只接受該 logical run 下唯一 completed attempt，或由 `--stage1-run` 明確指定的
completed attempt。不依賴 latest、mtime 或 TEST metrics；不自動改寫 artifact 的 absolute paths。

讀取前會核對：

- Stage-1 status 為 completed，且與 `validation.json` 一致。
- Seed=1、5 epochs、CPU float64、原 protocol-v2 與 28 個 canonical sources。
- Stage-1 reload validation、window provenance、自然 window counts／offsets。
- Checkpoint path、檔案大小及保存的 SHA256。
- 目前 scientific core 與 Stage-1 `source_snapshot/` fingerprints 一致。

原 `main.load_model()` 會建立 optimizer，因此 Stage 2 重用現有的
`src.calibration_runtime.load_frozen_model()`：讀取相同 checkpoint serialization 與原 TranAD
architecture，凍結 model，不建立 optimizer。重用的是 frozen-loading helper，沒有呼叫 calibration
的 threshold-fitting 或 thresholded-evaluation wrapper。

每台與整批 evaluation 前後均核對 model-state hash，並驗證 checkpoint 與 Stage-1 核心 artifacts
未改變。每台記錄 checkpoint absolute path／SHA256、`model_retrained=false`、`optimizer_steps=0`。
完整 summary 必須確認：

```text
parameter_hash_before == parameter_hash_after
unique_universal_checkpoint_hashes == 1
```

## 5. 每台 TEST 的資料流與 timestamp alignment

```text
同一個 frozen universal model
    ↓
一台 machine 的 TEST raw sequence
    ↓
原 preprocessing → 原 convert_to_windows()／windows_for_machine()
    ↓
原 main.backprop(training=False, optimizer=None, scheduler=None)
    ↓
逐 feature reconstruction errors
    ↓ errors.mean(axis=1)
逐 timestamp continuous anomaly score
    ↓
與同 timestamp 的 interpretation-label union 計算 ranking metrics
```

每台獨立 windowing、獨立 inference，**不 concatenate raw TEST sequences**。
原 mapping 保持：

```text
raw timestamp t → window t → target t → error t → score t → label t
```

Window 包含該台 `t-9` 到 `t` 的 observations；左側不足時重複該台第一筆 observation。
每個 raw timestamp 產生一個 window，不裁切 warmup，不 roll 或 shift labels。
`windows_for_machine()` 會逐一比對所有 windows 的內容與 target。

TEST values 沿用 `preprocess.load_and_save('test', ...)`。
Feature labels 沿用 `preprocess.load_and_save2('labels', ...)`，區間為 zero-based、end-exclusive，
再以 feature union 產生每 timestamp 的 binary label。只把 cache 改寫至新的 evaluation attempt。

必須驗證：

```text
score_count == label_count == raw_test_length
scores 為 finite float64
labels 為合法 binary vector
timestamps == 0, 1, ..., N-1
```

TEST labels 只用於 labels integrity／alignment checks 與最終 metrics，不能影響 model、checkpoint
selection、normalization、hyperparameters 或 threshold。

## 6. Metrics 與 summary

直接重用 `src.continuous_metrics.continuous_metrics()`，與 validated TranAD continuous-score
evaluation 使用相同公式。每台保存：

- PR-AUC trapezoidal、Average Precision、ROC-AUC。
- TEST points、positive-label count、anomaly prevalence。
- `PR-AUC - anomaly prevalence`，僅作 descriptive random-ranking reference。

完整 28 台成功後，`summary.json` 保存 macro mean PR-AUC／AP／ROC-AUC、median PR-AUC、
sample std（`ddof=1`）、min／max PR-AUC 與對應 targets。這些是 target-level descriptive statistics，
**不能把 28 targets 解讀成 28 independent random seeds**。

若能找到既有 seed-1 formal summary，另外產生每台 Per／LOMO／Universal PR-AUC comparison，
以及三個 setting 的 macro mean table。舊數值從 artifacts 讀取，不 hard-code 使用者提供的近似平均數。

Comparison 會核對 56 個 unique target／setting、seed=1、protocol/config、原 run metadata／evaluation、
TEST 和 interpretation-label hashes。若預設 summary 不存在，ranking summary 仍可產生，
但 comparison 明確標記 unavailable；若找到或明確指定的資料不符，則直接失敗。

不預先認定 universal 一定介於 Per 與 LOMO 之間，也不把任何正向差異稱為 statistical significance。

## 7. 新增與修改檔案

| File | 用途 |
| --- | --- |
| `src/universal_evaluation.py` | Stage-1 resolution、smoke gate、frozen inference、provenance、獨立 attempts |
| `src/universal_evaluation_analysis.py` | Score／label／hash 驗證、28-target summary、seed-1 formal comparison |
| `scripts/run_universal_evaluation.py` | Dry-run／單台 smoke／完整 28 台 CLI |
| `hpc/universal_evaluate.sbatch` | 單一 CPU inference job，預設只跑 smoke |
| `tests/test_universal_evaluation.py` | Focused synthetic tests |
| `hpc/README_HPC.md` | 既有文件新增 Stage-2 部署說明 |
| `UNIVERSAL_TRAINING_STAGE2.md` | 本文件 |

既有 Stage-1 training source/config、TranAD architecture/loss/score、POT/SPOT、formal/calibration
實作沒有修改。原有 data、results、checkpoints 不加入 Git，也不搬移或覆寫。

## 8. 本機測試紀錄

| Test suite | 結果 | 最近一次執行秒數 |
| --- | ---: | ---: |
| `test_universal_evaluation.py` | 14 通過 | 13.584 |
| `test_universal_training.py` | 9 通過 | 43.758 |
| `test_seed_replication.py` | 12 通過 | 56.453 |
| `test_calibration*.py` | 24 通過 | 39.268 |
| `test_target_calibration.py` | 15 通過 | 4.958 |
| **合計** | **74 通過** | — |

Stage-2 tests 使用明確標示的 random-model／Stage-1-schema synthetic fixture，沒有 training。
先完成一台 synthetic smoke，再執行 synthetic 28-target evaluation，確認只載入一次 model、
沒有 optimizer/backward、scores finite、labels/alignment 正確、checkpoint/hash 一致、原 inputs 不變，
以及 summary/namespace/錯誤處理。Smoke 與 full 中相同 machine-1-1 的 scores 完全一致。

既有 regression 的 synthetic training/calibration fixtures 只驗證舊程式相容性，不代表新增了
universal training 或 calibration experiment。所有舊 assertions 保持不變。

重跑本機測試，在 repository root 使用原有 CPU environment：

```bash
export PYTHON_BIN="../TranAD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/tranad-universal-eval-tests-mpl"

"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_evaluation.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_training.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_seed_replication.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_calibration*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_target_calibration.py -v
bash -n hpc/universal_evaluate.sbatch
git diff --check
```

不要在 Hazel 把 `PYTHON_BIN` 設成上述 Mac sibling path；Hazel 沿用自己的 `hpc/env.sh`。

## 9. Hazel：先 dry-run 與 machine-1-1 smoke

先由使用者完成 Git review、commit/push，再於 Hazel checkout 手動更新。以下指令未由 Codex 執行：

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/universal/evaluation/slurm-logs"

"$PYTHON_BIN" scripts/run_universal_evaluation.py \
  --hpc --seed 1 --target machine-1-1 --dry-run
```

Dry-run 不載入 model、不 inference、不建立 results/cache；會顯示 checkpoint path/SHA256、Stage-1
attempt、TEST/label 路徑/hashes、預期 count、protocol 與 output namespace。
若 resolver 發現多個 completed attempts，先審查來源，再以 `--stage1-run` 明確指定。
不要把 RESULTS_ROOT 改成 checkpoint 或 training attempt 子目錄。

單台 smoke 由使用者提交：

```bash
UNIVERSAL_EVAL_MODE=smoke sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/evaluation/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/universal/evaluation/slurm-logs/smoke-%j.err" \
  hpc/universal_evaluate.sbatch
```

已在 compute allocation 時，也可直接執行以下 CLI；與 sbatch 擇一，不在 login node 跑 inference：

```bash
"$PYTHON_BIN" -u scripts/run_universal_evaluation.py \
  --hpc --seed 1 --target machine-1-1
```

先 review smoke 的 status、validation、metadata、metrics、stdout/stderr。單台 smoke 不產生
正式 28-target summary。

## 10. Smoke 成功並 review 後，再執行完整 28 台

完整執行前，程式必須找到相同 checkpoint、相同 evaluation code fingerprint、相同 machine-1-1
TEST/labels 的成功 smoke；未符合時，在建立 full attempt 前就停止。

```bash
UNIVERSAL_EVAL_MODE=full sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/evaluation/slurm-logs/full-%j.out" \
  --error="$RESULTS_ROOT/universal/evaluation/slurm-logs/full-%j.err" \
  hpc/universal_evaluate.sbatch
```

或在已有 compute allocation 使用 CLI：

```bash
"$PYTHON_BIN" -u scripts/run_universal_evaluation.py --hpc --seed 1 --all
```

這是一個 sequential job，不是 28 個 training jobs，也不建立任何新 checkpoint。

| Slurm resource | 設定 |
| --- | --- |
| Account | `csc591f26_cpu` |
| Partition／QoS | `compute_partners`／`short` |
| Nodes／tasks／CPUs | 1／1／4 |
| Memory／walltime | 8G／30 分鐘 |

每次只保留一台 TEST windows，不保留 training gradients/optimizer，因此使用較 training 小的
memory request。真實 runtime/MaxRSS 仍須由 Hazel accounting 確認，不能把 request 當成測量值。

可選參數：

| CLI | sbatch environment variable | 用途 |
| --- | --- | --- |
| `--stage1-run /實際/attempt` | `UNIVERSAL_STAGE1_RUN` | Pin completed Stage-1 attempt |
| `--formal-summary /實際/formal_28_machine_results.csv` | `UNIVERSAL_FORMAL_SUMMARY` | 指定舊 seed-1 summary |
| `--retry` | `UNIVERSAL_EVAL_RETRY=1` | 只對 failed evaluation 建立新 attempt |

預設 comparison summary 路徑為 `$RESULTS_ROOT/summaries/formal_28_machine_results.csv`。

## 11. Outputs 與覆寫保護

```text
RESULTS_ROOT/universal/evaluation/
├── slurm-logs/
└── seed1/<checkpoint-sha256>/<evaluation-code-sha256>/
    ├── smoke/attempt-<timestamp>-<uuid>/
    └── full/attempt-<timestamp>-<uuid>/
        ├── launch.json
        ├── run.log
        ├── status.json
        ├── validation.json
        ├── targets/<machine>/
        │   ├── metadata.json
        │   ├── metrics.json
        │   ├── validation.json
        │   ├── scores.npz
        │   └── cache/processed/SMD/
        └── summary/
            ├── target_metrics.csv
            ├── summary.json
            ├── seed1_comparison.csv
            └── seed1_macro_comparison.csv
```

最後兩份 comparison CSV 只在 formal inputs 可安全核對時產生。
`evaluation-code-sha256` 由 scientific core 與 evaluation source fingerprints 計算；程式更新會使用
新 namespace，保留舊成功結果，新版本必須重新做 matching smoke。

同版本／checkpoint 已有結果時，預設拒絕重跑。成功 attempt 即使加 `--retry` 也不覆寫；
failed retry 建立全新 attempt，保留所有舊 files。若 job 強制終止留下 lock/running 狀態，
先人工確認 job 已結束，不自動刪 lock 或重送，也不接受 partial files 作為成功結果。

### `scores.npz`

| Key | Shape／dtype | 意義 |
| --- | --- | --- |
| `scores` | `(N,)`／float64 | 原 TranAD feature MSE 的 mean |
| `labels` | `(N,)`／int64 | Validated interpretation-label union |
| `timestamps` | `(N,)`／int64 | 原 raw TEST indices：0 到 N-1 |

保存 archive SHA256 與 score/label array hashes，供之後 Stage 3 唯讀重用。

### `target_metrics.csv`

```csv
target,group,seed,test_points,anomaly_points,anomaly_prevalence,pr_auc,average_precision,roc_auc,pr_auc_minus_prevalence,checkpoint_id,checkpoint_sha256
```

必須恰有 28 個 unique canonical targets，且 checkpoint SHA256 的 unique count=1。
`metadata.json` 另存每台 TEST/label 路徑與 hashes、raw/counts、固定 config、Stage-1 sources、
checkpoint ID/path/hash、Git commit/dirty state、套件版本、inference seconds 及 no-adaptation 證據。

## 12. Smoke／完整結果的檢查方式

`EVALUATION_DIR` 設為 stdout 印出的實際 smoke 或 full attempt directory；JOB_ID 換成實際 Slurm ID：

```bash
squeue -u "$USER"
tail -f "$RESULTS_ROOT/universal/evaluation/slurm-logs/smoke-JOB_ID.out"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode

cat "$EVALUATION_DIR/status.json"
cat "$EVALUATION_DIR/validation.json"
cat "$EVALUATION_DIR/targets/machine-1-1/metadata.json"
cat "$EVALUATION_DIR/targets/machine-1-1/metrics.json"
cat "$EVALUATION_DIR/targets/machine-1-1/validation.json"

# 完整 28 台成功後才會有：
cat "$EVALUATION_DIR/summary/summary.json"
```

Review 必須確認：

- `state=completed`、seed=1；完整 run 的 `target_count=28`。
- checkpoint SHA256 與 Stage-1 validation 一致，所有 targets 只有同一 SHA256。
- `parameters_unchanged=true`，全體 before/after hashes 相同，optimizer steps=0。
- 每台 score/label/counts/timestamps 對齊，scores finite，labels binary。
- PR-AUC/AP/ROC-AUC 與 anomaly prevalence 均存在；沒有 threshold/calibration artifacts。
- Comparison 若 unavailable，先補確認 formal summary 位置，不把缺值當成研究結果。

可交給 ChatGPT review 的資料：本文件、dry-run 輸出、smoke stdout/stderr、smoke 的
`status.json`／`validation.json`／target metadata 與 metrics。完整結果出來後，再提供
`target_metrics.csv`、`summary.json` 與 comparison CSV。

## 13. 停止點與尚未驗證項目

目前狀態是 **Stage-2 code 與本機 synthetic validation 完成，真實 Hazel evaluation 待執行**。
沒有可回報的真實 universal PR-AUC/AP/ROC-AUC，不能用 synthetic metrics 代替。

Stage 2 的真實完成條件是：28 台成功、同一 checkpoint、逐台 scores/labels 保存、完整 ranking
summary 及可核對的 seed-1 comparison，且 Stage-1／formal artifacts 保持不變。

完成後停止，先 review ranking results；不自動開始 global/local thresholds、Raw F1、PA、
target calibration、seeds 2/3 或其他 architecture。
