# Hazel / Slurm 手動部署

## Universal replication：seeds 2／3（最新 workflow）

使用者已回報 seed-1 Universal Stage 1–3 完成；本節只準備同 protocol 的 seeds 2／3 replication。
沒有執行真實 seeds 2/3 training、evaluation、POT 或 Hazel job，沒有 commit/push。
本機測試只使用 synthetic data、random model/schema fixtures 與 stub POT，不代表新研究結果。

### 唯一 experimental variable 與相容性

Model training seed 為 2 或 3。其他條件完全相同：TranAD/SMD、全部 canonical 28 TRAIN、5 epochs、
train/eval batch 128、window 10、lr=1e-4、CPU float64、原 AdamW/StepLR/loss、逐 machine windowing、
natural source-size weighting、deterministic seeded shuffle、原 preprocessing/labels/score/checkpoint 格式。
沒有 normalization、balancing、額外 budget、TEST-based selection 或新模型。

POT q=1e-5、requested level=.99995、multiplier=1.06、原 initialization retries、empty TEST stream 不變。
**POT RNG 依凍結的 seed-1 protocol，每次 fit 仍為 1**；不把模型 seed 2/3 當成 POT RNG。
五個條件 `global / local_05 / local_10 / local_20 / local_full` 全部保留，每 seed 恰為 140 rows。
有限極端 thresholds 保留原值，不 clip、不刪 target、不改 POT/fraction，也不挑最佳 fraction。

舊 seed-1 source fingerprints 包含原 evaluation entrypoint；直接修改它會使已完成 artifacts 的自動
定位失效。因此保留 `src/universal_training.py`、`src/universal_evaluation.py`、
`src/universal_evaluation_analysis.py` 和原 Python CLI 逐字不變。
Seeds 2/3 改用共用 replication I/O adapter，重用原 model、TEST loader、window validator、frozen inference、
continuous metrics、Stage-3 orchestration／POT／Raw metrics。沒有另外實作 seed-specific model/loss/scoring。
原 Stage-3 functions 的 seed 身分／metadata／path 參數化，seed=1 預設仍維持原數值行為。

`configs/universal_seed1_compatibility.json` 保存凍結 commit `9982b1b56386b3c0f9ab98edd0fe99ec1f3421fd`
的 Stage-3 source map／code SHA256，供唯讀解析歷史 seed-1 namespace；未知 fingerprints 直接失敗。
不改寫 seed-1 artifacts 或改填新 code hash。原 seed-1 Python CLI 仍保留只接受 seed 1 的 evaluation/
threshold 安全介面；seeds 2/3 透過 `scripts/run_universal_replication.py` 與新的 Slurm routing 執行。
不支援的 seeds 一律拒絕。

### Gates、matching-seed references 與 caches

同 seed Stage 1 completed 才能 evaluation；同 seed Stage-2 full completed 才能 thresholding。
新 evaluate/threshold phases 各自在 fresh subprocess 依序執行 smoke → full，保留原 smoke gates；
已 completed 的同 phase 不覆寫，會重新核對其 artifacts。失敗重試需 `--retry` 或
`UNIVERSAL_REPLICATION_RETRY=1`，保留舊 attempts。不同 phases 不要同時操作相同 caches。

每 seed 的 28 TEST targets 共用一份 checkpoint，unique checkpoint SHA256 count=1。
Stage 3 不重新 TEST inference，只讀同 seed Stage-2 scores/labels/timestamps；TRAIN caches 同時核對
seed、checkpoint SHA256、model-state hash、TRAIN file/array hashes、code fingerprint/source hashes。
跨 seed cache／checkpoint／TEST artifact 不能互換。

Per-machine/LOMO 參考資料由 canonical `seed_replication_runs.csv`、同 seed task parent、唯一成功
attempt、worker args、metadata/source snapshots 定位；核對 protocol、TRAIN hashes/windows、TEST 與
interpretation-label hashes。不回退 seed 1，也不靠一個未驗證的 CSV 借用其他 seed 的結果。
缺 matching formal artifacts 會停止 comparison，而不是偷偷填 seed-1 baseline。

### 先 Seed 2 plan／dry-run

使用者 review/commit/push、Hazel 更新後，沿用既有 CPU environment：

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/universal/replication-logs"

# 不需要 checkpoint，檢查全部 28 TRAIN、固定 protocol 與各 seed namespaces。
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase plan
# 等價的 Stage-1 training preflight：
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase train --dry-run
```

Plan 中尚未產生的 checkpoint/code hash 用 placeholder，不能當成已存在 artifact。
Stage-2/3 dry-run 僅能在各自 dependency 完成後使用，缺 checkpoint/full evaluation 時直接失敗。
不載入 model、不 inference、不 fit POT、不建立結果。

### Seed 2：分階段手動提交

每個 job 完成並檢查 status/validation 後，才提交下一個；以下沒有自動 dependency submission：

```bash
SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed2-train-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed2-train-%j.err" hpc/universal_train.sbatch

# Stage 1 成功後先確認：
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase evaluate --dry-run
SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed2-evaluate-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed2-evaluate-%j.err" hpc/universal_evaluate.sbatch

# Stage 2 full 成功後先確認：
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase threshold --dry-run
SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed2-threshold-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed2-threshold-%j.err" hpc/universal_threshold.sbatch
```

如已在 compute allocation，可與 sbatch 擇一使用：

```bash
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 2 --phase train
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 2 --phase evaluate
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 2 --phase threshold
```

### Seed 2 review 後才 Seed 3

先確認 seed-2 的 checkpoint/TEST/TRAIN cache seed identities、one-checkpoint、140 rows、empty POT stream、
model/test/continuous-metric invariants 全部成立。不要因數值高低修改 seed-3 protocol。
若發現真正 protocol bug，先停止釐清；不只修後面 seed。不要同時提交 seeds 2/3。

```bash
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 3 --phase plan
SEED=3 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed3-train-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed3-train-%j.err" hpc/universal_train.sbatch

# Seed-3 Stage 1 完成後：
SEED=3 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed3-evaluate-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed3-evaluate-%j.err" hpc/universal_evaluate.sbatch

# Seed-3 Stage 2 full 完成後：
SEED=3 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed3-threshold-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed3-threshold-%j.err" hpc/universal_threshold.sbatch
```

對應 compute-allocation CLI：

```bash
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 3 --phase train
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 3 --phase evaluate
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 3 --phase threshold
```

對 seeds 2/3，evaluation/threshold sbatch 會 route 到新 wrapper，自動做該 phase 的 smoke/full；
舊 `UNIVERSAL_EVAL_MODE`／`UNIVERSAL_THRESHOLD_PHASE` 只適用 seed-1 分支，不控制新 wrapper。
資源不變：train=4 CPUs/16G/2h；evaluate=4 CPUs/8G/30min；threshold=4 CPUs/8G/2h，
同 `csc591f26_cpu / compute_partners / short`。Actual cost 未測量，job 後讀 accounting：

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

### 每 seed 的 artifacts

```text
RESULTS_ROOT/universal/
├── runs/universal_TranAD_SMD_v2_e5_bs128_seed{2,3}/attempt-*/
├── checkpoints/universal_TranAD_SMD_v2_e5_bs128_seed{2,3}/attempt-*/model.ckpt
├── evaluation/seed{2,3}/<checkpoint-sha256>/<evaluation-code-sha256>/{smoke,full}/attempt-*/
├── thresholding/seed{2,3}/<checkpoint-sha256>/<stage3-code-sha256>/
│   ├── train_scores/<machine>/attempt-*/
│   └── {smoke,full}/attempt-*/
└── multiseed/analysis-<timestamp>-<uuid>/
```

新 Stage-2 full 保存 `summary/target_metrics.csv`、`summary/summary.json`、
`summary/matching_seed_comparison.csv`；不把 seed-2/3 table 誤命名為 seed1_comparison。
Stage-3 schema 沿用原版，每 seed 140 rows，新 seeds 合計 280 rows；保留全部 threshold conditions。

### 三 seed aggregation：完成後才執行

```bash
"$PYTHON_BIN" scripts/analyze_universal_replication.py --hpc
```

如果 seed 1 有多個歷史成功 code namespaces，先人工確認，再加
`--seed1-threshold-run /實際/seed1/full/attempt`，不自行挑 latest。
Aggregation 全面核對 completed artifacts、cache provenance、score/label/hash、Raw/continuous metrics、
全域自然 score pool、prefix counts、matching formal references，以及三個 Universal seeds 相同的
TRAIN/TEST/labels/environment。只讀 individual seeds，不重算 TEST inference 或 POT。

輸出另外存至 `universal/multiseed/analysis-*`：

- `ranking_by_seed.csv`：三個 settings × 三個 seed 的 macro PR-AUC。
- `threshold_by_seed.csv`：五種 Universal conditions 與 Per-machine 的各 seed macro Raw P/R/F1。
- `multiseed_ranking_summary.csv`、`multiseed_threshold_summary.csv`：seed1/2/3、mean、sample std、seed_count=3。
- `seed_level_gaps.csv`、`multiseed_gap_summary.csv`：六種預先定義的 paired seed-level gaps。
- `machine_direction_consistency.csv`：每台 ranking/local recovery 的三 seed 方向、ties、真正正負翻轉旗標。
- `multiseed_summary.json`：input SHA256、direction counts、aggregation 定義；不聲稱 statistical significance。

一定先每 seed 對 28 targets macro，再對三個 seed-level means 做 mean ± sample std（ddof=1）。
不能把 84 machine-seed observations 當成 84 seeds；machine consistency 只是 descriptive。

### 本機驗證與停止點

原 Universal 40 項 tests 保持 assertions 原樣；seed replication 12 項、calibration 39 項亦已通過。
新增 replication 13 項 tests 通過，合計 104 項。Seed-2 真實 TRAIN-only plan 確認 28 台、708405 windows，
只讀資料，沒有訓練。新 tests 驗證 seeds/configs、cross-seed 拒絕、舊 seed-1 namespace 唯讀解析、140 rows、extreme finite
threshold 不 clipping、seed-level ddof=1、方向一致性，以及凍結數值函式/POT RNG 的 AST 對照。
Synthetic model fixtures 不代表已訓練 seed-2/3 model；stub POT 不代表真實 thresholds。

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/universal-replication-tests-mpl"
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_universal*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_seed_replication.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_calibration*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_target_calibration.py -v
```

目前停止於 replication workflow/code review；不自動執行真實實驗、submit、SSH 或 push，
不根據 seed-2/3 outcomes 修改科學設定，也不擴展到更多 seeds 或新 architecture。

## Universal Stage 3：global vs local thresholds（seed 1）

依使用者最新回報，Stage 1 universal checkpoint 與 Stage 2 全 28 台 ranking evaluation 已在 Hazel
完成；本機沒有這些真實 artifacts。下方 Stage-1／Stage-2 章節保存當時開發與驗證紀錄。
Stage 3 的真實 threshold、Raw metrics、runtime、MaxRSS 均待使用者在 Hazel 執行，不能用本機
synthetic 結果替代。本次沒有 SSH、job submission、commit 或 push。

### Scientific protocol 與實作

本階段只比較 `global / local_05 / local_10 / local_20 / local_full`：同一 seed-1 checkpoint、
同一 Stage-2 TEST scores／labels／timestamps，只有 threshold 改變。Global 使用全部 28 台 TRAIN
scores 的自然資料量權重，不 equalize、不 normalize、不平均 thresholds。Local 使用該台 raw
TRAIN 的 5／10／20% temporal prefix，以及完整 TRAIN 作為 full-local reference。
`local_full` 不是 few-shot calibration；Universal 也不是 zero-shot。

所有 TRAIN 逐台獨立 windowing 和 frozen inference；不串接 raw machines。只在 scoring 完成後
concatenate scalar scores 供 global POT 使用。固定 CPU float64、window=10、eval batch=128、seed=1，
model.eval／requires_grad=False／no_grad，不建立 optimizer、不 backward、不更新參數。
Stage 3 沒有 raw TEST loader 或 TEST model inference；TEST 入口只讀既有 Stage-2 NPZ。

重用原 `load_frozen_model()`、`windows_for_machine()`、`main.backprop(training=False)`、
`prefix_length()`、`fit_threshold()`／`fit_source_threshold()`、continuous metrics 和 AUC invariants。
POT 固定 q=1e-5、requested level=.99995、multiplier=1.06，原 initialization retry 不變；
每次 fit 依舊 calibration 做法重設 NumPy seed=1，SPOT TEST stream 必須是空的。

舊 `additional_metrics()` 將 Raw／PA payload 綁在一起，且原檔受 checkpoint fingerprints 保護。
新 Raw-only adapter 保留原 Raw 分支公式、零分母規則與嚴格 `score > threshold`，以對照測試確認
與原函式 Raw 結果相同；不呼叫 PA，不修改原 scientific core。

Prefix reuse 的依據：window(t) 只取同台 raw[max(0,t-9):t+1] 及左側 padding，與該台的未來 suffix
無關；frozen TranAD 在不同 windows 間沒有狀態。因此先切 raw prefix 再 windowing，和完整
windows[:n] 數學上相同。Regression 驗證短 prefix padding 與跨 128-batch 邊界的 scores，
absolute tolerance=1e-12、rtol=0；不宣稱不同 BLAS batch shape 必然 bitwise 相同。
真實 smoke 額外執行一次 prefix-first inference，記錄與 full-score prefix 的最大差異，未符合則停止。
正式 local 條件隨後直接重用完整 TRAIN score 前綴，避免重複 inference。

### 檔案與測試證據

新增：`src/universal_threshold_artifacts.py`（Stage-2 resolver／cache／isolation）、
`src/universal_thresholding.py`（TRAIN inference／smoke／POT orchestration）、
`src/universal_threshold_analysis.py`（Raw metrics／aggregation／baseline comparison）、
`scripts/run_universal_thresholding.py`、`hpc/universal_threshold.sbatch`、
`tests/test_universal_thresholding.py`。既有程式、Stage-1／Stage-2 及 calibration artifacts 不修改。

本機 17 項新 synthetic tests 通過（最後一次 22.367 秒）。既有 Stage-2 14 項、Stage-1 9 項、seed replication
12 項、calibration 39 項也通過，合計 91 項。新 orchestration tests 使用明確的 POT stub，另以
原 fitting adapter 確認 SPOT 收到空 TEST stream、原 q/level/multiplier；不代表真實 POT 已完成。
涵蓋 140 rows／5 conditions、全部 28 TRAIN streams、自然 weighting、prefix fractions、cache reuse、
one checkpoint／model freeze、TEST hashes／metrics invariance、Raw equality、無 PA、baseline pairing、
namespace、不覆寫及失敗拒絕。

可重跑測試（PYTHON_BIN 指向既有本機 CPU environment；本次實際使用 ../TranAD/.venv/bin/python）：

```bash
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

### Hazel 手動執行：先 smoke

使用者完成 code review、commit/push，Hazel 更新後，保留原共同 RESULTS_ROOT：

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/universal/thresholding/slurm-logs"

"$PYTHON_BIN" scripts/run_universal_thresholding.py --hpc --phase smoke --seed 1 --dry-run
```

Dry-run 只讀原 artifacts／檔案 hashes，不 model loading／inference／POT／寫結果。
會解析唯一 completed Stage-1 attempt 與同 checkpoint／evaluation code 的 completed Stage-2 full
attempt；不使用 latest 或 Stage-2 smoke。多個成功 attempts 時明確 pin，不自動挑模型。

```bash
UNIVERSAL_THRESHOLD_PHASE=smoke sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/thresholding/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/universal/thresholding/slurm-logs/smoke-%j.err" \
  hpc/universal_threshold.sbatch
```

Smoke 只生成 machine-1-1 的 full TRAIN scores，驗證 5% prefix equivalence、fit local_05 POT、
重用 Stage-2 TEST 算 Raw P/R/F1、確認三個 continuous metrics 與 hashes 不變。
不以單台 TRAIN 假造 global threshold。

### Smoke review 後，準備 TRAIN scores 與完整 Stage 3

以下 CLI 供已有 compute allocation 時使用，不在 login node inference：

```bash
# 可選獨立 Phase 1：生成並驗證全部 28 台 TRAIN score caches，不 fit POT。
"$PYTHON_BIN" -u scripts/run_universal_thresholding.py --hpc --phase train-scores --seed 1

# Smoke 通過且 review 後，完整比較五個條件。
"$PYTHON_BIN" -u scripts/run_universal_thresholding.py --hpc --phase full --seed 1
```

也可用同一 Slurm script，與上述 CLI 擇一：

```bash
# 可選：先做 TRAIN score pool。
UNIVERSAL_THRESHOLD_PHASE=train-scores sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/thresholding/slurm-logs/train-scores-%j.out" \
  --error="$RESULTS_ROOT/universal/thresholding/slurm-logs/train-scores-%j.err" hpc/universal_threshold.sbatch

# 僅在 smoke review 與前一 job 結束後手動提交。
UNIVERSAL_THRESHOLD_PHASE=full sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/thresholding/slurm-logs/full-%j.out" \
  --error="$RESULTS_ROOT/universal/thresholding/slurm-logs/full-%j.err" hpc/universal_threshold.sbatch
```

Full 自動重用已驗證的 TRAIN caches（包含 smoke 的 M1）；若未先跑 train-scores，full 會順序生成
其餘 caches。只保留 scalar TRAIN scores 供 pooled POT，不同時保留 28 個 dense window tensors。
Full 一個 job 中載入一次 model，做一個 global fit 及 28×4 local fits，共 113 次邏輯 threshold fits；
各 fit 仍可依原初始化規則 retry。沒有自動 job submission。
stdout／run.log 會顯示 TRAIN machine/batch 進度、每個 POT fit 開始／完成、initialization attempts
及逐 target/condition 的 Raw P/R/F1；status.json 會更新 global/local POT phase 與目前 target/condition。

Resource request：csc591f26_cpu／compute_partners／short、1 node、1 task、4 CPUs、8G、2 小時。
Inference 不需要 gradients/optimizer，memory 小於 universal training；walltime 較 Stage-2 長是因為
full 包含 113 個 POT fits／retries。這是保守初始 request，不是實測耗時或 MaxRSS。

可選參數與 sbatch 變數：

| CLI | Environment variable |
| --- | --- |
| `--stage1-run /實際/attempt` | `UNIVERSAL_STAGE1_RUN` |
| `--stage2-run /實際/full/attempt` | `UNIVERSAL_STAGE2_RUN` |
| `--formal-summary /實際/formal_28_machine_results.csv` | `UNIVERSAL_FORMAL_SUMMARY` |
| `--retry` | `UNIVERSAL_THRESHOLD_RETRY=1` |

### 輸出、immutability 與比較

```text
RESULTS_ROOT/universal/thresholding/
├── slurm-logs/
└── seed1/<checkpoint-sha256>/<stage3-code-sha256>/
    ├── train_scores/<machine>/attempt-*/
    │   ├── scores.npz                    # scores、timestamps；無 labels
    │   ├── metadata.json
    │   ├── validation.json
    │   ├── status.json
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
            ├── target_condition_metrics.csv  # 140 rows
            ├── macro_condition_metrics.csv
            ├── recovery_statistics.csv
            ├── group_condition_metrics.csv
            ├── deployment_comparison.csv
            ├── per_machine_comparison.csv    # 原 formal baseline 可驗證時
            └── per_machine_gap_statistics.csv
```

所有 phases 另有 launch／run.log／status／validation。TRAIN caches 只在 checkpoint、seed、code、
TRAIN file/array hashes、model state 和 timestamp/counts 都一致時重用。Seed 2/3 不可使用 seed-1 cache。
成功結果預設拒絕重跑；failed retry 新建 attempt，不改舊檔。殘留 lock/running 要先人工確認 job
已停止，不自動刪除或回退舊結果。

Stage-1 checkpoint 與 Stage-2 archive／metadata／metrics／validation／summary 全部先記錄 hashes，
執行後再核對。每 target 的 score／label／timestamp hashes 在五個條件間相同，PR-AUC/AP/ROC-AUC
與 Stage 2 差異不得超過 1e-12。這不是 ranking improvement，只有 operating point 改變。

Raw F1 recovery 定義為 local Raw F1 減 global Raw F1，四種 local 都報 mean／median 與 improved／
degraded／unchanged counts，不挑 best fraction、不產生 oracle。Group 1/2/3 只作 descriptive 分組。
Threshold diagnostics 報 signed／absolute shift、ratio；global 接近 0 時 ratio=null 並記原因，不輸出 NaN/Inf。

預設唯讀核對 RESULTS_ROOT/summaries/formal_28_machine_results.csv 及其原 run metadata/evaluation，
產生 Per／LOMO／Universal deployment macro P/R/F1，以及每 target×condition 的 Per-minus-Universal
Raw F1 gap（包含 local_05 和 local_full 的 paired gap/counts）。沒有安全可用 baseline 時明確標示
comparison unavailable，不 hard-code 或猜補舊 Raw metrics；可用 --formal-summary 指定正確來源。
LOMO 的 28 models 是研究用 held-out evaluation count，不代表 production 必須部署 28 shared models。
本次沒有加入 optional LOMO-calibration context，也沒有重跑舊 calibration。

### 完成檢查與停止點

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

將 STAGE3_DIR 設為 stdout 的實際 phase attempt，JOB_ID 換成實際 Slurm ID。summary 的 actual Slurm
elapsed/MaxRSS 目前為 null；job 完成後由 sacct 取得，提供 review 時附上輸出，不能使用 request 值替代。
完整成功需 140 rows、28 targets×5 conditions、同一 checkpoint、一個 global scalar、全部 hashes／
metrics／freeze invariants 通過；baseline comparison 若 unavailable，先補確認來源再做完整解讀。
**停在 seed-1 Stage 3 review，不自動執行 seeds 2/3、training 或 fine-tuning。**

## Universal Stage 2：同一 checkpoint 的 continuous-score evaluation

本階段新增獨立入口，只比較 shared representation 的 ranking quality：PR-AUC trapezoidal、
Average Precision、ROC-AUC。每台另存 anomaly prevalence 與 `PR-AUC - prevalence`，後者只作
descriptive reference，不表示 statistical significance。Universal 已使用各台 TRAIN，**不是 zero-shot**。
沒有新增 threshold、Raw F1、PA、POT fitting 或 calibration，也不啟動 seeds 2/3。

### 實作與本機驗證狀態

新增 `src/universal_evaluation.py`、`src/universal_evaluation_analysis.py`、
`scripts/run_universal_evaluation.py`、`hpc/universal_evaluate.sbatch` 與
`tests/test_universal_evaluation.py`。既有 scientific core、Stage-1 code/config、formal 與 calibration
實作都沒有修改。本文件是本次唯一修改的既有檔案。

選擇單一 job：只載入一次 seed-1 universal checkpoint，依 canonical 順序獨立處理 28 台 TEST，
每次只 materialize 一台的 windows。這能直接驗證整批前後 model-state hash 不變，也不需另建
28 個 model/checkpoint。資源為既有 CPU account/partition/QoS，4 CPUs、8G、30 分鐘；
比 training 減少 memory request，是因為不保留 28 台 training windows、沒有 gradients/optimizer，
不是改 precision。真實 Hazel Stage-2 runtime / memory 仍待量測。

本機通過 14 項新 synthetic tests；Stage-1 9 項、seed replication 12 項、calibration 39 項
regression 亦通過，合計 74 項。Synthetic fixture 明確使用 random model state 與測試用 Stage-1 schema，
只用來驗證 frozen inference／artifact 控制，不代表真實已訓練 universal checkpoint 的結果。
本機沒有 Hazel 上完成的 Stage-1 artifacts，因此**真實 checkpoint path/SHA256、machine-1-1 smoke
及完整 28 台 metrics 仍待 Hazel 執行**；未連線、未提交 job，未宣稱 Stage 2 真實實驗已完成。

測試實際使用既有 `../TranAD/.venv/bin/python`。可重跑的指令如下：

```bash
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

### Checkpoint 與 scientific invariants

Stage-1 logical identity 沿用 `universal_TranAD_SMD_v2_e5_bs128_seed1`。Resolver 只接受該 logical run
下唯一 completed attempt，或使用者透過 `--stage1-run` 明確指定的 completed attempt；不依賴 latest、
mtime 或 TEST metrics。必須通過 status、validation、metadata、28-machine provenance、checkpoint SHA256
與大小，以及原 scientific source snapshot 比對。缺失、不一致或多個 completed attempts 均直接失敗。
Stage-1 checkpoint 的原路徑為：

```text
RESULTS_ROOT/universal/checkpoints/
  universal_TranAD_SMD_v2_e5_bs128_seed1/<stage1-attempt-id>/model.ckpt
```

原 `main.load_model()` 會建立 optimizer，因此 Stage 2 重用既有
`src.calibration_runtime.load_frozen_model()` 讀取相同 TranAD serialization／architecture，
完全不建立 optimizer。沿用 `eval()`、`requires_grad=False`、`torch.no_grad()`，
原 `main.backprop(training=False, optimizer=None, scheduler=None)` 與 `errors.mean(axis=1)`。
不呼叫 calibration 的 threshold 或 evaluation wrapper。

TEST 資料及 feature labels 分別由原 `preprocess.load_and_save('test', ...)`、
`load_and_save2('labels', ...)` 產生，只將 cache 導向 Stage-2 當次 attempt。
Labels 使用 validated interpretation-label feature union（0-based、end-exclusive）。
每台獨立呼叫原 `windows_for_machine()`／`convert_to_windows()`，逐 window 比對所有 timestamps；
window=10，target/score/label timestamp=t，沒有裁切或跨 machine sequence。
Eval batch=128、CPU float64、seed=1 不變；Stage 2 不讀 TRAIN。

每台及全體前後驗證 model-state hash、checkpoint SHA256 與 Stage-1 核心 artifacts hashes。
每台記錄 `model_retrained=false`、`optimizer_steps=0`。全體 summary 要求
`unique_universal_checkpoint_hashes=1`，28 台 model-state hash 也相同。

### 手動順序：先 dry-run，再單台 smoke

先完成 Git review／使用者 commit/push，再在 Hazel 更新 checkout 並 source 原 CPU environment。
以下命令由使用者手動執行；inference 請放在 compute job，不在 login node 執行。

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/universal/evaluation/slurm-logs"

"$PYTHON_BIN" scripts/run_universal_evaluation.py \
  --hpc --seed 1 --target machine-1-1 --dry-run
```

Dry-run 會列出真實 Stage-1 attempt、checkpoint 絕對路徑/SHA256、TEST/label 路徑與 hashes、
預期 score count、固定 protocol、output namespace。不載入 model、不 inference、不建立結果或 cache。
若需要明確 pin attempt，CLI 加 `--stage1-run /實際/completed/attempt-path`；sbatch 則設定
`UNIVERSAL_STAGE1_RUN`，不要將 RESULTS_ROOT 改成 checkpoint 或 training attempt 目錄。

只提交 machine-1-1 smoke：

```bash
UNIVERSAL_EVAL_MODE=smoke sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/evaluation/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/universal/evaluation/slurm-logs/smoke-%j.err" \
  hpc/universal_evaluate.sbatch
```

也可在已有 compute allocation 直接使用以下 CLI；與 sbatch 擇一：

```bash
"$PYTHON_BIN" -u scripts/run_universal_evaluation.py --hpc --seed 1 --target machine-1-1
```

先檢查 stdout、`status.json`、全域及 target 的 `validation.json`：state=completed、scores/labels
長度相同、finite、parameter hashes 相同、checkpoint hash 與 Stage 1 一致、PR-AUC/AP/ROC-AUC 存在。
單台 smoke 不會產生偽裝成正式 28 台的 summary。

### Smoke review 後，才手動啟動完整 28 台

完整 evaluation 必須找到同 checkpoint、同程式 fingerprint、同 machine-1-1 TEST/labels 的成功 smoke；
程式會在建立 full attempt 前檢查，不能用 `--all` 跳過 smoke。沒有自動 dependency 或提交動作。

```bash
UNIVERSAL_EVAL_MODE=full sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/evaluation/slurm-logs/full-%j.out" \
  --error="$RESULTS_ROOT/universal/evaluation/slurm-logs/full-%j.err" \
  hpc/universal_evaluate.sbatch
```

或在已有 compute allocation：

```bash
"$PYTHON_BIN" -u scripts/run_universal_evaluation.py --hpc --seed 1 --all
```

預設若存在 `RESULTS_ROOT/summaries/formal_28_machine_results.csv`，會讀取 seed-1 Per-machine/LOMO
結果並核對 56 unique target/setting、seed/protocol/config、原 metadata/evaluation、TEST 與 interpretation
hashes，再輸出 comparison。不 hard-code 0.407517／0.291756，也不呼叫 threshold metrics。
若 summaries 位於其他地方，CLI 加 `--formal-summary /實際/formal_28_machine_results.csv`；
sbatch 使用 `UNIVERSAL_FORMAL_SUMMARY`。若預設 summary 不存在，ranking summary 仍可產生，
comparison 會明確標記 unavailable；若指定／找到的資料不符則失敗，不填補或猜測舊數值。

### 獨立且不可覆寫的 artifacts

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
            ├── seed1_comparison.csv          # formal inputs 可核對時
            └── seed1_macro_comparison.csv    # formal inputs 可核對時
```

`evaluation-code-sha256` 是既有 scientific core 與新 evaluation source fingerprints 的 hash。
程式變更使用新 namespace，舊成功 smoke/full 不會被覆寫；新版本須重新做 matching smoke。
同一版本／checkpoint 已存在時，預設拒絕重跑。`--retry` 只允許 failed evaluation 新建 attempt，
不覆寫成功結果；sbatch 對應 `UNIVERSAL_EVAL_RETRY=1`。被強制終止後若殘留 lock/running 狀態，
先人工確認 job 已終止；不自動刪 lock、不自動重送，也不把 partial files 當完成。

`scores.npz` 只有：`scores`（N, float64）、`labels`（N, int64 union labels）、
`timestamps`（N, int64，0 到 N-1）。保存 archive／score array／label array hashes，供 Stage 3 唯讀重用。
`metadata.json` 保存 raw TEST/label 路徑及 hashes、prevalence、window、batch、dtype/device、同一
universal checkpoint 路徑/hash、source list、Git/versions、freeze 及 no-adaptation 證據。

`target_metrics.csv` 欄位為：

```text
target,group,seed,test_points,anomaly_points,anomaly_prevalence,
pr_auc,average_precision,roc_auc,pr_auc_minus_prevalence,checkpoint_id,checkpoint_sha256
```

`summary.json` 保存 mean PR-AUC/AP/ROC-AUC、median PR-AUC、跨 targets sample std（ddof=1）、
min/max targets 與 scores、unique checkpoint hash count。這是 28 targets 的 descriptive statistics，
不是 28 independent seeds。Comparison 僅比較三個 setting 的 seed-1 PR-AUC，不預先指定結果解讀。

監看與檢查（替換實際 JOB_ID／EVALUATION_DIR）：

```bash
squeue -u "$USER"
tail -f "$RESULTS_ROOT/universal/evaluation/slurm-logs/smoke-JOB_ID.out"
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
cat "$EVALUATION_DIR/status.json"
cat "$EVALUATION_DIR/validation.json"
cat "$EVALUATION_DIR/targets/machine-1-1/metrics.json"
# 完整 28 台成功後才有：
cat "$EVALUATION_DIR/summary/summary.json"
```

**完整 Stage 2 結果完成後停止，先 review ranking；不自動開始 universal threshold 或 calibration。**

## Universal training Stage 1

只使用全部 28 台 SMD TRAIN、沿用原 TranAD protocol-v2，完成一個 seed-1 checkpoint 與 reload 驗證。
不執行 TEST、POT 或 calibration，也不修改下列既有實驗入口。新輸出只在 `RESULTS_ROOT/universal/`。
完整 provenance、TRAIN counts、本機測試與單一 CPU sbatch 指令見
[UNIVERSAL_TRAINING_STAGE1.md](../UNIVERSAL_TRAINING_STAGE1.md)。
先手動 dry-run，再由使用者決定提交 `hpc/universal_train.sbatch`；本次未提交。
Stage 1 停在 checkpoint review，不啟動 seeds 2/3 或 universal evaluation。

依使用者目前提供的進度，既有 RQ1 已完成 Hazel dry-run、1-epoch smoke、27→1 reproduction 與
56 個 formal runs；本次沒有重新執行或驗證這些工作。新增的 RQ2 calibration 已完成程式實作及
27 個 local synthetic tests；真實 SMD / Hazel calibration 仍為 PENDING HAZEL VALIDATION。
以下原部署章節保留作為操作參考。
Slurm 範本依 [NCSU Running Jobs](https://hpc.ncsu.edu/RunningJobs/RunningJobs.php)
與 [Array Jobs](https://hpc.ncsu.edu/RunningJobs/Arrays.php) 的官方語法；實際 partition、account、
可用 Python module 和額度需依使用者帳號確認，不假設固定 allocation。

## 1. 分別準備 code、data、results

Code 可由既有 Git remote clone 到使用者指定位置。Dataset 另外放置，不隨 Git 上傳。
不要從原研究 checkout 搬移 data 或歷史結果。

設定 `hpc/env.example` 中四個環境變數，可複製成被 ignore 的 `hpc/env.sh`，再手動 source。
`DATA_ROOT` 直接指向 SMD 根目錄。支援 scalar labels 資料夾名 `labels` 或 `test_label`；
主要異常標籤仍由 interpretation_label 以 start:end 產生。

```sh
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/slurm-logs"
```

## 2. 環境與 dry-run

依根目錄 README 安裝外部 CPU Python environment。下面要讀取 dataset 的命令應在適當的
interactive compute allocation 或 batch job 執行，不要在 login node 執行訓練。

```sh
"$PYTHON_BIN" scripts/run_experiment.py --hpc --dry-run --machine machine-1-1
"$PYTHON_BIN" scripts/build_formal_manifest.py --hpc
```

HPC 模式缺少 roots、root 指向錯誤 checkout 或 results 與 data/code 重疊時，會直接失敗。

## 3. 先提交單機 smoke

以下命令是供使用者手動執行的範例，本次沒有執行：

```sh
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/slurm-logs/smoke-%j.err" hpc/smoke_test.sbatch
```

若帳號需要 partition/account，請在 sbatch 加上核准的 `--partition=... --account=...`。
不要把 shell variable 放在 `#SBATCH` 期待 Slurm 自動展開。
Smoke 使用 1 epoch / CPU float64；單次 smoke 成功仍不等於正式結果已完成。

## 4. 檢查 smoke 後，再由使用者決定提交 array

```sh
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/slurm-logs/formal-%A_%a.out" \
  --error="$RESULTS_ROOT/slurm-logs/formal-%A_%a.err" hpc/formal_array.sbatch
```

Array 是 `0-55%4`，最多同時四個獨立 runs；不是四個 source machines，也沒有改 sampling。
每個 task 配置 4 CPU threads、16 GiB request、2 小時 walltime；這是初始資源範本，應根據 Hazel
smoke 的實測調整資源 request，不能為速度擅改模型、precision 或 scientific hyperparameters。
`CUDA_VISIBLE_DEVICES` 設為空，維持 CPU execution。

Array 各 run 使用自己獨立的 processed cache，避免同時寫共用 source `.npy`。
中斷的 epoch 不會自動 resume；完整 epochs 完成才寫 checkpoint。重新執行同一 task 會產生新的
run_id，保留舊 attempt。程序被 scheduler 強制終止時，可能沒有結尾 status，會被列為 missing。

## 5. 監看與失敗 task

```sh
squeue -u "$USER"
tail -f "$RESULTS_ROOT/slurm-logs/實際檔名.out"
"$PYTHON_BIN" scripts/find_failed_runs.py --hpc
```

若需要重跑，檢查輸出的 task IDs 後，由使用者以 `sbatch --array=...` 覆寫 array 範圍。
程式不會自動重跑或提交 jobs。

## 6. 完成後 summarize

```sh
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/slurm-logs/summary-%j.out" \
  --error="$RESULTS_ROOT/slurm-logs/summary-%j.err" hpc/summarize.sbatch
```

也可由使用者加 `--dependency=afterok:<array-job-id>`。Summary 發現任何缺失、失敗或 smoke row
混入 formal 結果時會停止；不會把未完成工作算成完整 28-machine 結果。

若省略文件中的 `--output/--error`，Slurm 可能產生預設 `slurm-*.out` 在提交位置；因此正式
提交務必使用外部 log 路徑。本 repository 的 hpc/logs/ 僅保留空白 placeholder。

## 7. RQ2 Target threshold calibration：先單次 smoke

完整 protocol、schema 與 static review 見 [CALIBRATION_PROTOCOL.md](../CALIBRATION_PROTOCOL.md)。
Calibration 使用既有 cross-machine model、target TRAIN prefix 與同一 POT；不重新 training，
重用正式 TEST scores。這是 lightweight target adaptation，不能改稱 zero-shot。

本機已有 [synthetic smoke](../LOCAL_CALIBRATION_SMOKE.md)，可用
`PYTHON_BIN=/path/to/python bash scripts/local_calibration_smoke.sh` 重跑；它不需要 SMD，
不執行真實 POT，也不能代替 Hazel smoke。

手動流程固定為：

```text
local Git review → user commit/push → Hazel git pull
→ calibration manifest/config validation
→ machine-1-1 5% dry-run
→ machine-1-1 5% smoke
→ inspect invariants → 使用者與 ChatGPT review → STOP

只有確認 smoke 後：user approval → 84-task array → failure check → summary
```

本次只準備以下 scripts，沒有提交：

| Script | Resource request |
| --- | --- |
| calibration_smoke.sbatch | 1 node、1 task、4 CPUs、8G、30 分鐘；machine-1-1 / 5% |
| calibration_array.sbatch | 0-83%8；每 task 1 node、4 CPUs、8G、30 分鐘 |
| calibration_summary.sbatch | 1 node、2 CPUs、4G、15 分鐘；要求 84 tasks 全部完成 |

Account / partition / qos 使用 csc591f26_cpu / compute_partners / short；calibration requests 尚未實測。
所有輸出位於 RESULTS_ROOT/calibration；原 formal scripts 與結果不變。

使用者 push 後，在 Hazel 手動更新：

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
export FORMAL_RESULTS_ROOT="${FORMAL_RESULTS_ROOT:-$RESULTS_ROOT}"
mkdir -p "$RESULTS_ROOT/calibration/slurm-logs"
```

在適當 compute allocation 驗證 manifest/config 與 dry-run；本次 Codex 未執行：

```bash
"$PYTHON_BIN" scripts/build_calibration_manifest.py --hpc
"$PYTHON_BIN" scripts/run_target_calibration.py \
  --hpc --target machine-1-1 --fraction 0.05 --dry-run
```

接著由使用者只提交 smoke：

```bash
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/calibration/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/calibration/slurm-logs/smoke-%j.err" \
  hpc/calibration_smoke.sbatch
```

檢查 status、smoke_summary、metadata、threshold、evaluation、validation：model hashes 不變、
optimizer_steps=0、scores/labels/AUC 不變、POT stream_count=0、Raw metrics 產生且 formal inputs 不變。
任何失敗先停止，不啟動 array。預設同 experiment 不重複寫；確定需重試時才明確加 --retry。

未來完整實驗的 failed/missing checker 為：

```bash
"$PYTHON_BIN" scripts/find_failed_calibration_runs.py --hpc
```

只做 smoke 時，84 個正式 calibration tasks 尚未執行，checker 列出缺失是預期行為。
Checker 不自動重試；summary 不接受失敗、missing 或 smoke rows。

**STOP AFTER MACHINE-1-1 5% SMOKE。** 本章不提供立即提交完整 84-task array 的命令。

## 8. Anomaly Transformer secondary robustness

新增獨立 6-target / 12-run ranking-only extension，正式設定、local test/dry-run 證據與兩個 GPU smoke
指令見 [ANOMALY_TRANSFORMER_ROBUSTNESS.md](../ANOMALY_TRANSFORMER_ROBUSTNESS.md)。
不使用 threshold、PA、POT/SPOT 或 calibration，不修改原 TranAD / calibration 結果。

AT 需額外設定獨立 CUDA environment 的 `AT_PYTHON_BIN`，保留原 CPU `PYTHON_BIN`。
範本為 csc591f26_gpu / gpu_partners / short_gpu / gpu:a100:1；本機無 hpc/env.sh，尚未驗證 private
account associations、GPU environment 或實際 VRAM/runtime。不要 source CPU common.sh 清空 GPU。

先只由使用者手動提交 machine-1-1 manifest tasks 0（per_machine）與 1（cross_machine），兩者均固定
10 epochs、batch 256。兩個 smoke review 前，不提交其餘 tasks 2–11，也不啟動完整 summary。

## TranAD 多 seed 補強（原結果不覆寫）

新增 seed=2、3 各 56 runs；原 seed=1 的 runs、checkpoints、summaries、圖檔均唯讀保留。
新入口與所有新 artifacts 位於獨立 `seed-replication/`，不修改原 formal / calibration / AT scripts。
先執行 preflight dry-runs，再由使用者手動提交 array；本次沒有自動提交 jobs。

- `hpc/seed_replication_preflight.sbatch`：四個 dry-runs，不訓練。
- `hpc/seed_replication_array.sbatch`：112 tasks，`0-111%4`，CPU float64。
- `hpc/seed_replication_summary.sbatch`：讀取 seeds=1,2,3，另存新 CSV / PNG / PDF。

完整輸出保護、retry 與命令見 [SEED_REPLICATION_HPC.md](../SEED_REPLICATION_HPC.md)。

## RQ2 Calibration 擴充至 Seeds 2 / 3

新 manifest 為 `configs/calibration_runs_seed23.csv`（168 tasks），不重跑 seed=1。
同 seed 的 formal checkpoint / TEST artifacts 由 canonical seed-replication mapping 核對，
新輸出位於 `calibration/seed-replication/`；舊 seed=1 calibration artifacts 唯讀保留。
先用 `--seed 2` 完成 machine-1-1 / 5% dry-run 與 smoke，再手動使用
`CALIBRATION_MANIFEST=... --array=0-167%8` 提交新 tasks。
三 seed 分析入口為 `scripts/analyze_calibration.py --hpc --seeds 1 2 3`。
完整指令、RNG 定義、路徑假設與驗證限制見 [CALIBRATION_MULTI_SEED_HPC.md](../CALIBRATION_MULTI_SEED_HPC.md)。
