# Universal Shared Checkpoint：Stage 3

更新日期：2026-11-20。

本文件記錄 seed-1 universal TranAD 的 **Global Threshold vs Local Threshold Calibration**
實作、驗證證據與 Hazel 執行方式。雖沿用 `UNIVERSAL_TRAINING_STAGE3.md` 檔名，
Stage 3 不 training、不 fine-tuning，也不重新做 TEST inference。

## 1. 目前進度與證據範圍

| 項目 | 目前狀態 |
| --- | --- |
| Stage 1 | 使用者已回報：全部 28 台 TRAIN 訓練出一個 seed-1 checkpoint，validation 完成 |
| Stage 2 | 使用者已回報：同一 checkpoint 完成 28 台 TEST evaluation，scores/labels 已保存 |
| Stage-2 mean PR-AUC | 使用者回報約 0.34964；本機沒有原 artifacts，未獨立核對 |
| Stage-3 implementation | 已完成，目前待 Git review／commit／push |
| Stage-3 synthetic tests | 17 項通過 |
| 相關既有 regression | 74 項通過；合計 91 項 |
| 真實 Stage-3 dry-run／smoke／full | PENDING HAZEL VALIDATION |
| 真實 global threshold／Raw metrics | 尚未產生可由本機確認的結果 |
| 真實 Slurm runtime／MaxRSS | 尚未取得 |
| Codex SSH／submit jobs | 未執行 |

本機真實路徑 dry-run 因找不到 completed Stage-1 artifacts 而停止，沒有 fallback 到其他 checkpoint。
真實 checkpoint path/SHA256、Stage-2 evaluation path/code hash，須於 Hazel dry-run 解析並核對。
使用者回報的 Stage-2 mean 不會 hard-code 進程式或拿來代替 artifacts。

本文件引用前一輪已完成的測試紀錄；本次撰寫文件未重新執行 tests、POT 或 experiments。

## 2. 研究問題與固定條件

RQ3a：一個 universal checkpoint，能否使用一個共通 threshold 部署到全部 28 台？

RQ3b：保持 shared model 不變時，少量正常 target TRAIN 是否可透過 local thresholds 改善 operating point？

固定比較：

| Condition | Threshold fitting input | 定位 |
| --- | --- | --- |
| `global` | 全部 28 台完整 TRAIN scores concatenate | 一個 model、一個 threshold |
| `local_05` | 每台 raw TRAIN 前 5% 對應的 scores | Lightweight local calibration |
| `local_10` | 每台 raw TRAIN 前 10% 對應的 scores | Lightweight local calibration |
| `local_20` | 每台 raw TRAIN 前 20% 對應的 scores | Lightweight local calibration |
| `local_full` | 每台完整 TRAIN scores | Full-local reference，不是 few-shot calibration |

Universal 已使用各台 TRAIN 參與 representation training，因此這些條件都不能稱為 zero-shot。
所有條件使用相同 checkpoint、相同 Stage-2 TEST scores；只改 threshold。
不得選 target-specific best fraction、依 TEST F1 調 threshold，或建立 oracle result。

| 設定 | 固定值 |
| --- | --- |
| Model／dataset | TranAD／SMD |
| Seed | 1；本階段不支援執行 seeds 2/3 |
| Window／eval batch | 10／128 |
| Device／dtype | CPU／float64 |
| Model | `eval()`、`requires_grad=False`、`torch.no_grad()` |
| Optimizer／backward | 不建立、不呼叫；optimizer steps=0 |
| Preprocessing／score／labels | 沿用既有 validated protocol-v2 |
| Primary threshold metric | Raw pointwise Precision／Recall／F1 |
| PA | 不執行，不作主結果 |

## 3. Data flow 與 leakage 防護

```text
Stage-1 universal checkpoint：同一份 frozen weights
        ↓
每台 TRAIN 各自 preprocessing／windowing／inference
        ↓
每台完整 aligned TRAIN scores（保存 cache）
        ├── 全部 28 台 concatenate scalar scores → 一個 global POT threshold
        └── 每台 scores temporal prefix → local 5%／10%／20%／full POT thresholds

既有 Stage-2 TEST scores／labels／timestamps（唯讀）
        ↓
套用上述 thresholds：prediction = score > threshold
        ↓
Raw P/R/F1、recovery、paired baseline comparison
```

禁止 concatenate raw machines 後才 windowing。每次只處理單台 TRAIN，完成 inference 後釋放 dense
windows；global pool 只 concatenate 一維 scalar scores。每台貢獻與自己 TRAIN timestamps 數量相同
的 scores，保留 natural source-size weighting，不 oversample、不 equalize、不 normalize scores、
不平均各台 thresholds。

TEST 入口只讀 Stage-2 saved NPZ，不讀 raw TEST，不對 TEST 呼叫 model。
POT API 不接受 TEST scores/labels 參數；實際 inputs 來自驗證過的 TRAIN-score caches。
TEST labels 只供結果驗證及最終 metrics，不能參與 fitting、fraction selection 或模型更新。

## 4. Prefix reuse：為何可重用完整 TRAIN scores

沿用原 window definition：

```text
window(t) = 該台 raw[max(0,t-9):t+1]，不足 10 點時左側重複 raw[0]
target／error／score timestamp = t
```

前 n 個 windows 不依賴 raw[n:]；frozen TranAD 在 windows 間沒有持續狀態。
因此以下 window tensors 數學上相同：

```text
convert_to_windows(raw[:n]) == convert_to_windows(raw)[:n]
```

Local budget 仍以 raw observations 定義：`n=floor(N×fraction)`，使用既有 `prefix_length()`，
不 random sample、不 shuffle raw points，也不先看 TEST labels 清理 TRAIN。

Regression 已驗證短 prefix 的 padding，以及跨 eval batch=128 邊界的 window/score equivalence。
Window tensors 要求完全相同；scores 使用 atol=1e-12、rtol=0，因不同 BLAS batch shapes 不保證
bitwise 相同。真實 smoke 會額外做一次 5% prefix-first inference，保存最大差異；不符合就停止。

驗證後，正式五個條件重用完整 TRAIN scores 的前綴，避免四次重複 TRAIN inference。
`local_full` 直接使用全部 scores，不經 5/10/20% helper。

## 5. POT 與 Raw metrics

重用 `src.calibration_runtime.fit_threshold()` → 原 `src.cross_machine.fit_source_threshold()`：

| 參數 | 固定值 |
| --- | --- |
| q | 1e-5 |
| Requested level | 0.99995 |
| Multiplier | 1.06 |
| Initialization retry | 原行為，不改 level retry 規則或次數上限 |
| POT RNG | 每次 fit 重設 NumPy seed=1，與原 calibration 一致 |
| SPOT TEST stream | Empty；`stream_score_count=0` |

Global fit 必須使用全部 28 台，不接受單台或部分 machines 冒充 global。Full run 有一個 global fit
加上 28×4 local fits，共 113 次邏輯 threshold fits；內部仍可依原初始化規則 retry。

Prediction 一律為 `scores > threshold`，不是 `>=`。
原 `additional_metrics()` 將 Raw 與 PA payload 綁在一起，且原 source 受既有 checkpoint fingerprint
保護。新 Raw-only adapter 保留原 Raw 分支公式與零分母處理，已直接對照原函式測試。
沒有修改舊 core，也沒有呼叫 `adjust_predicts()`。

## 6. Immutable inputs 與 runtime invariants

Stage-1 resolver 與 Stage-2 resolver 只接受可核對的 completed artifacts，不使用 latest/mtime selection。
Stage-2 input 必須是同 checkpoint、seed=1 的完整 28-target evaluation，不接受單台 smoke。
遇到多個 completed attempts，須用 CLI 明確指定，不依 TEST performance 選 checkpoint。

必須確認：

- 同一 checkpoint SHA256，所有 conditions 的 unique checkpoint hash count=1。
- Model-state hash 與 Stage 2 相符，Stage 3 前後相同；optimizer steps=0、model retrained=false。
- Stage-2 每台 NPZ／score／label／timestamp hashes、counts 保持不變。
- Stage-1 checkpoint、核心 artifacts，以及 Stage-2 metadata/metrics/validation/summary hashes 保持不變。
- 每個 condition 的 PR-AUC/AP/ROC-AUC 與 Stage 2 相同：atol=1e-12、rtol=0。
- Full result 恰有 140 rows、28 unique targets、每台五個固定 conditions、seed=1。
- 同 target 的五列 TEST array/archive hashes 相同；所有 global rows threshold 為同一 scalar。

任何 invariant 不符都停止，不重新 TEST inference、不調參，也不 silent fallback。
Calibration 改變 operating point，不改善或重新估計 ranking quality。

## 7. 實作檔案與相容性

| File | 功能 |
| --- | --- |
| `src/universal_threshold_artifacts.py` | Stage-2 resolution、input hashes、attempt isolation、TRAIN cache validation |
| `src/universal_thresholding.py` | TRAIN scoring、prefix smoke、global/local POT、phase orchestration |
| `src/universal_threshold_analysis.py` | Raw-only metrics、140-row validation、macro/recovery/group/baseline analysis |
| `scripts/run_universal_thresholding.py` | `smoke`／`train-scores`／`full` CLI |
| `hpc/universal_threshold.sbatch` | 單一 CPU job，預設只執行 smoke |
| `tests/test_universal_thresholding.py` | Focused synthetic tests |
| `hpc/README_HPC.md` | 既有文件新增 Stage-3 部署說明 |
| `UNIVERSAL_TRAINING_STAGE3.md` | 本文件 |

重用既有 frozen loader、TRAIN loader、window validator、`main.backprop(training=False)`、
prefix helper、POT helper、continuous metrics、AUC invariance 與 threshold-shift helper。
I/O 適配僅把 TRAIN preprocessing cache 導向當次 Stage-3 cache attempt，完成後還原原 module globals。
沒有修改 Stage-1/Stage-2 code、TranAD architecture/loss、POT/SPOT 或既有 calibration implementation。

## 8. 本機測試證據

| Suite | 通過數 | 最近一次紀錄秒數 |
| --- | ---: | ---: |
| `test_universal_thresholding.py` | 17 | 22.367 |
| `test_universal_evaluation.py` | 14 | 11.367 |
| `test_universal_training.py` | 9 | 25.007 |
| `test_seed_replication.py` | 12 | 33.630 |
| `test_calibration*.py` | 24 | 22.115 |
| `test_target_calibration.py` | 15 | 18.849 |
| **合計** | **91** | — |

這些是不同 suites 的執行紀錄，不是 Hazel cost 或整體單次 walltime。
Stage-3 orchestration 使用明確的 random-model/Stage-1-schema synthetic fixtures 與 POT stub；
另以原 fitting adapter 確認 SPOT 收到正確 q/level/multiplier 與 empty TEST stream。
不把 synthetic threshold、Raw F1 或 mock baseline 說成真實研究結果。

Tests 涵蓋自然 weighting、113 個固定 fit inputs、四個 local prefixes、global scalar、140 rows、
model freeze、cache reuse、Stage-2 immutability、metric invariance、Raw equality、無 PA、
prefix equivalence、baseline pairing、namespace、不覆寫，以及錯誤 seed/inputs 拒絕。
既有 regressions 使用其原本 synthetic fixtures，不代表重跑 formal training/calibration。

在本機 repository root 重跑：

```bash
export PYTHON_BIN="../TranAD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/universal-stage3-test-mpl"

"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_thresholding.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_evaluation.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_training.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_seed_replication.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_calibration*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_target_calibration.py -v
bash -n hpc/universal_threshold.sbatch
git diff --check
```

Hazel 使用既有 `hpc/env.sh` 的 CPU environment，不使用上述 Mac sibling Python path。

## 9. Hazel：Git 更新、dry-run、單台 smoke

使用者先完成 Git review／commit／push，再在 Hazel checkout 手動更新。下列 commands 未由 Codex 執行。

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/universal/thresholding/slurm-logs"

"$PYTHON_BIN" scripts/run_universal_thresholding.py \
  --hpc --phase smoke --seed 1 --dry-run
```

Dry-run 只讀 inputs/hashes，不載入 model、不 inference、不 fit POT、不寫結果。
JSON 會列出 checkpoint path/SHA256、Stage-2 run/code SHA256、TRAIN provenance 與 Stage-3 output root。
維持共同外部 RESULTS_ROOT，不把它改成 checkpoint 或 Stage-2 attempt 子目錄。

先只提交 machine-1-1 local_05 smoke：

```bash
UNIVERSAL_THRESHOLD_PHASE=smoke sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/thresholding/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/universal/thresholding/slurm-logs/smoke-%j.err" \
  hpc/universal_threshold.sbatch
```

Smoke 生成 M1 full TRAIN scores、驗證 5% prefix equivalence、fit local_05、重用 Stage-2 TEST scores
計算 Raw P/R/F1，然後核對 hashes、freeze 和三個 continuous metrics。沒有 fake global threshold。

已有 compute allocation 時可改用 CLI，與 sbatch 擇一，不在 login node inference：

```bash
"$PYTHON_BIN" -u scripts/run_universal_thresholding.py --hpc --phase smoke --seed 1
```

先 review smoke，再決定是否進入 full；程式也會檢查 matching successful smoke。

## 10. Smoke review 後：TRAIN caches 與完整 Stage 3

可選獨立生成/驗證全部 28 台 TRAIN score caches，這個 phase 不 fit POT：

```bash
UNIVERSAL_THRESHOLD_PHASE=train-scores sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/thresholding/slurm-logs/train-scores-%j.out" \
  --error="$RESULTS_ROOT/universal/thresholding/slurm-logs/train-scores-%j.err" \
  hpc/universal_threshold.sbatch
```

Smoke review 通過，且前一個 job 已結束後，手動提交 full：

```bash
UNIVERSAL_THRESHOLD_PHASE=full sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/thresholding/slurm-logs/full-%j.out" \
  --error="$RESULTS_ROOT/universal/thresholding/slurm-logs/full-%j.err" \
  hpc/universal_threshold.sbatch
```

若在已有 compute allocation，可分別用以下 CLI，與相對應 sbatch 擇一：

```bash
"$PYTHON_BIN" -u scripts/run_universal_thresholding.py --hpc --phase train-scores --seed 1
"$PYTHON_BIN" -u scripts/run_universal_thresholding.py --hpc --phase full --seed 1
```

Full 重用已驗證的 caches，包括 smoke 的 M1。若省略獨立 train-scores job，full 會順序生成其餘 caches。
不要同時啟動這些 phases；active locks 會拒絕同一 cache 的並行寫入。

| 可選 CLI | sbatch environment variable |
| --- | --- |
| `--stage1-run /實際/completed/attempt` | `UNIVERSAL_STAGE1_RUN` |
| `--stage2-run /實際/full/attempt` | `UNIVERSAL_STAGE2_RUN` |
| `--formal-summary /實際/formal_28_machine_results.csv` | `UNIVERSAL_FORMAL_SUMMARY` |
| `--retry` | `UNIVERSAL_THRESHOLD_RETRY=1` |

資源 request：`csc591f26_cpu / compute_partners / short`、1 node、1 task、4 CPUs、8G、2 小時。
Memory request 較 training 小，因為只保留單台 dense windows，沒有 gradients/optimizer。
Walltime 比 Stage 2 長是因為需要 113 個 POT fits/retries；這些是 request，不是實測數值。

stdout/run.log 顯示 TRAIN machine/batch、POT 開始/完成、initialization attempts，以及每個
target/condition 的 Raw P/R/F1。status.json 更新 global/local POT phase 與目前 target/condition。

## 11. Outputs 與 immutable caches

```text
RESULTS_ROOT/universal/thresholding/
├── slurm-logs/
└── seed1/<checkpoint-sha256>/<stage3-code-sha256>/
    ├── train_scores/<machine>/attempt-*/
    │   ├── scores.npz
    │   ├── metadata.json / validation.json / status.json
    │   └── cache/processed/SMD/
    ├── smoke/attempt-*/
    │   ├── smoke_metrics.json
    │   ├── local_threshold.json
    │   └── prefix_equivalence.json
    ├── train-scores/attempt-*/train_score_manifest.csv
    └── full/attempt-*/
        ├── launch.json / run.log / status.json
        ├── global_threshold.json
        ├── validation.json
        ├── summary.json
        ├── targets/<machine>/
        │   ├── local_*_threshold.json
        │   ├── *_predictions.npz
        │   └── threshold_diagnostics.json
        └── summary/
            ├── train_score_manifest.csv
            ├── target_thresholds.csv
            ├── target_condition_metrics.csv
            ├── macro_condition_metrics.csv
            ├── recovery_statistics.csv
            ├── group_condition_metrics.csv
            ├── deployment_comparison.csv
            ├── per_machine_comparison.csv
            └── per_machine_gap_statistics.csv
```

所有 phases 另有 launch/run.log/status/validation。最後兩份 Per-machine CSV 只在 baseline 能安全
驗證時產生；不可把缺失當成 0 或已完成 comparison。

TRAIN NPZ 只含 `scores`（N, float64）及 `timestamps`（N, int64），沒有 labels。
Caches 保存 TRAIN file/array hashes、score/timestamp hashes、checkpoint/seed/model-state provenance。
只在全部一致時重用，不得將 seed-1 scores 用到其他 seed/checkpoint。

成功 artifacts 拒絕 overwrite 或 retrain；failed retry 新建 attempt，保留原 files。
程式更新會使用不同 code-hash namespace。強制終止留下 lock/running 時須人工確認 job 已停止，
不自動刪 lock、不自動重送、不回退舊 checkpoint 或失效 cache。

## 12. 結果格式與分析定義

`global_threshold.json` 保存 threshold、q、requested/effective level、multiplier、initialization attempts、
score/machine count、natural weighting、empty TEST stream、checkpoint hash 與完整 score-pool provenance。

`target_condition_metrics.csv` 恰有 140 rows：

```csv
target,group,seed,condition,threshold,raw_precision,raw_recall,raw_f1,tp,tn,fp,fn,pr_auc,average_precision,roc_auc,checkpoint_sha256,test_scores_sha256,test_labels_sha256,test_timestamps_sha256,test_archive_sha256
```

`target_thresholds.csv`：

```csv
target,group,seed,global_threshold,local_05_threshold,local_10_threshold,local_20_threshold,local_full_threshold
```

`macro_condition_metrics.csv`：

```csv
condition,seed,mean_raw_precision,mean_raw_recall,mean_raw_f1,median_raw_f1,sample_std_raw_f1,improved_vs_global,degraded_vs_global,unchanged_vs_global
```

Global 的 recovery counts 不適用，CSV 留空。Standard deviation 使用 ddof=1，表示跨 targets 的
descriptive variability，不是多 seed uncertainty 或 statistical significance。

四個 local recovery 定義為 `local Raw F1 - global Raw F1`，報 mean/median 與 improved/degraded/unchanged
counts。Group 1/2/3 只作 dataset descriptive grouping，不推論 server roles。

Threshold diagnostics 保存相對 global 的 signed/absolute shift 及 ratio。若 global 接近 0，
ratio=null，reason=`global_threshold_near_zero`；不輸出 NaN/Inf。

## 13. Per-machine／LOMO baseline comparison

預設讀取 `$RESULTS_ROOT/summaries/formal_28_machine_results.csv`，並核對 seed/protocol、原
metadata/evaluation、TEST/interpretation hashes、Raw metrics 與 paired targets。可用 --formal-summary
指定其他位置，不使用手抄 baseline 平均數。

Deployment table 比較原 Per-machine、原 LOMO，以及五種 Universal 條件的 macro Raw P/R/F1。
LOMO 的 28 個模型是研究用 held-out evaluation count，不表示 production 需要 28 個 shared models。

Per-machine paired gap 定義：`Per-machine Raw F1 - Universal condition Raw F1`。
所有條件均報 mean/median gap、Universal better／Per-machine better／ties，包含 local_05 與 local_full。
負 gap 表示 Universal 在該 target 的 Raw F1 較高，但不表示統計顯著。

沒有可安全驗證的 baseline 時，summary 明確標記 comparison unavailable；先確認正確來源，再做
完整解讀。Optional LOMO target-calibration context 本次沒有加入，也不重新執行舊 calibration。

## 14. 執行後檢查與交付 review

將 STAGE3_DIR 設為 stdout 顯示的實際 phase attempt，JOB_ID 換成實際 Slurm ID：

```bash
squeue -u "$USER"
tail -f "$RESULTS_ROOT/universal/thresholding/slurm-logs/smoke-JOB_ID.out"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode

cat "$STAGE3_DIR/status.json"
cat "$STAGE3_DIR/validation.json"
# smoke：
cat "$STAGE3_DIR/smoke_metrics.json"
cat "$STAGE3_DIR/prefix_equivalence.json"
# full：
cat "$STAGE3_DIR/global_threshold.json"
cat "$STAGE3_DIR/summary.json"
```

summary 的 `actual_slurm_elapsed`／`actual_slurm_maxrss` 目前保留 null；真實 cost 應由 job 完成後
的 sacct 輸出另外提供，不使用 request 值替代，也不要為填寫測量值覆寫已完成的 immutable artifacts。

Smoke review 建議提供本文件、dry-run JSON、stdout/stderr、status、validation、smoke metrics、
local threshold 與 prefix-equivalence JSON。Full review 再提供 global threshold、summary、
target/macro condition CSV、recovery、Per-machine paired gap tables，以及 sacct 輸出。

## 15. 停止點

目前是 **Stage-3 code 與本機 synthetic validation 完成，真實 seed-1 experiment 待 Hazel 執行**。
不能宣稱已取得真實 global threshold、smoke Raw F1 或五個條件的 macro results。

真實完成需：同一 checkpoint、一個 28-TRAIN global threshold、28 台四種 local thresholds、
140 rows、TEST 無 reinference、所有 hashes/continuous metrics/model invariants 通過，並具備可核對
的 baseline comparison。所有條件一起報告，不選 oracle 或 best fraction。

完成 seed-1 Stage 3 後停止並 review；不自動 training/evaluation seeds 2/3，不 fine-tuning，
不增加其他 threshold algorithm、normalization 或 architecture。
