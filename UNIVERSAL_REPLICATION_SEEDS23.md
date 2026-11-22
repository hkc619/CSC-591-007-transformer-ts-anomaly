# Universal Replication：Seeds 2 與 3

更新日期：2026-11-21。

本文件供 code review 與 ChatGPT 進度確認，記錄 Universal Stage 1–3 的 seeds 2/3 replication
實作、驗證證據、artifact 隔離與 Hazel 手動執行順序。

## 1. 目前狀態

| 項目 | 狀態 |
| --- | --- |
| Seed-1 Universal Stage 1–3 | 使用者已回報完成，protocol 凍結 |
| Seeds 2/3 replication workflow | 已實作，待 Git review／commit／push |
| 新增 replication tests | 13 項通過 |
| 原有相關 tests | 91 項通過；合計 104 項 |
| Seed-2 本機 TRAIN-only plan | 確認 28 台、708405 windows；未訓練 |
| 真實 seed-2 Stage 1–3 | 尚未執行，PENDING HAZEL VALIDATION |
| 真實 seed-3 Stage 1–3 | 尚未執行，須先 review seed 2 |
| 真實三 seed aggregation | 尚未執行，等待 seeds 2/3 artifacts |
| SSH／Slurm submission／commit／push | Codex 未執行 |

本機測試使用 temporary synthetic data、random model/schema fixtures 與 stub POT。
它們驗證流程與 invariants，不代表已訓練 seed-2/3 model，也不提供真實 PR-AUC、Raw F1 或 threshold。
本文件引用前一輪已完成的測試紀錄；本次撰寫文件沒有重新執行 tests 或 experiments。

## 2. 實驗目的與凍結 protocol

本階段是 replication，不是 redesign。只改變模型 training seed，不根據 seed-1 outcomes 調整設定。

每個 seed 執行：

```text
全部 28 台 SMD TRAIN
    ↓ 每台獨立 windowing，再 concatenate 完整 windows
seeded shuffle → 原 TranAD training → 一份 universal checkpoint_s
    ↓
28 台獨立 TEST inference → 保存 scores／labels／timestamps
    ↓
逐台 frozen TRAIN scoring
    ├── concatenate 全部 TRAIN scalar scores → 一個 global POT threshold
    └── 各台 temporal prefixes → local 5%／10%／20%／full thresholds
    ↓
重用相同 seed 的 Stage-2 TEST scores → Raw P/R/F1
```

| 設定 | 固定值 |
| --- | --- |
| Model／dataset | TranAD／SMD |
| Epochs | 5 |
| Train／eval batch | 128／128 |
| Window／初始 learning rate | 10／0.0001 |
| Device／dtype | CPU／float64 |
| Sources | 原 canonical 28 台及排序；沒有 held-out target |
| Weighting | Natural source-size weighting |
| Training order | 原 deterministic seeded window shuffle |
| Architecture／optimizer／scheduler／loss | 原 TranAD、AdamW、StepLR 與既有 loss |
| Preprocessing／score／labels／checkpoint | 保留 validated protocol-v2 |
| Model training seed | 2 或 3 |
| POT RNG | **每次 fitting 仍固定為 1**，依凍結的 seed-1 protocol |

POT 固定 q=1e-5、requested level=.99995、multiplier=1.06，原 initialization retries 與 empty TEST stream
不變。Model seed 不會取代 POT RNG seed。

保留全部五個 threshold conditions：

```text
global
local_05
local_10
local_20
local_full
```

每台 local prefix 使用 `[0:floor(N×fraction)]`，不 random sample、不 shuffle raw、不用 TEST labels
清理 TRAIN。Global pool 保留自然資料量權重，不 equalize、不 normalize、不平均 thresholds。
有限極端 thresholds 原樣保存；不 clip、不移除 machines、不只保留 5%、不選 target-specific best fraction。
Universal 已使用各台 TRAIN，因此不能稱為 zero-shot。

## 3. Seed-1 相容性與實作方式

原 seed-1 fingerprints 包含 evaluation entrypoint，舊 tests 也保護部分檔案逐字不變。
因此保留以下 source 與原 Python CLI，不直接放寬其 evaluation/threshold seed-1 parser：

```text
src/universal_training.py
src/universal_evaluation.py
src/universal_evaluation_analysis.py
scripts/run_universal_training.py
scripts/run_universal_evaluation.py
scripts/run_universal_thresholding.py
```

Stage 1 原本已接受 seeds 1/2/3，直接重用。
Seeds 2/3 的 evaluation 使用共用 seed-aware I/O adapter，重用原 TEST loader、windowing、frozen inference
與 continuous metrics；不另寫 seed-specific model/loss/scoring。
Stage 3 的 functions 參數化 seed identity／metadata／paths，保留相同 Raw/POT numerical code。
所有 seeds 2/3 操作透過 `scripts/run_universal_replication.py` 或已更新的 Slurm routing。

`configs/universal_seed1_compatibility.json` 保存凍結 commit：

```text
9982b1b56386b3c0f9ab98edd0fe99ec1f3421fd
```

其中保存原 Stage-3 source map 與 code SHA256：

```text
e38541a5bd1dec710031e533f2fefaca91b2e3a9f8fa6cd5d9a83eebc214f080
```

它用於唯讀解析歷史 seed-1 namespace，不是實驗 metric 或 checkpoint hash。
不改填既有 artifacts 的 code hash；未知歷史版本會失敗，不自行假設相容。

## 4. 新增與修改檔案

新增：

| File | 功能 |
| --- | --- |
| `configs/universal_seed1_compatibility.json` | Seed-1 frozen code fingerprint／唯讀相容性 |
| `src/universal_replication_protocol.py` | Seed-aware checkpoint／TEST resolver 與 matching formal references |
| `src/universal_replication.py` | Evaluation I/O orchestration 與 matching-seed comparisons |
| `src/universal_replication_analysis.py` | 完整 artifacts 稽核、三 seed macro aggregation／direction consistency |
| `scripts/run_universal_replication.py` | plan／train／evaluate／threshold orchestration |
| `scripts/analyze_universal_replication.py` | 獨立三 seed aggregation CLI |
| `tests/test_universal_replication.py` | 新增 13 項 focused tests |
| `UNIVERSAL_REPLICATION_SEEDS23.md` | 本文件 |

修改：

```text
hpc/README_HPC.md
hpc/universal_train.sbatch
hpc/universal_evaluate.sbatch
hpc/universal_threshold.sbatch
src/universal_threshold_artifacts.py
src/universal_thresholding.py
src/universal_threshold_analysis.py
```

Slurm scripts 只放寬受控 SEED=1/2/3 與 routing，不增加 resources。
Stage-3 修改集中在 seed metadata、paths、cache binding、matching-seed references；預設 seed=1 的數值行為
不變。Raw-only metrics、frozen scoring 與 POT RNG tail 有 AST 對照測試；原 model/core/POT source 不修改。

## 5. Dependency gates 與 seed isolation

```text
同 seed Stage 1 completed
    ↓ required
同 seed Stage 2 full completed
    ↓ required
同 seed Stage 3 completed
```

每 seed 的 28 TEST targets 必須使用同一 checkpoint，unique checkpoint SHA256 count=1。
Stage 2 因 weights 不同，必須重新生成該 seed 的 TEST scores；Stage 3 則不得重新 TEST inference。

TRAIN cache 必須核對 seed、checkpoint SHA256、model-state hash、TRAIN file/array hashes、code fingerprint
與 source hashes。Seed-2 checkpoint／TRAIN scores／TEST scores／thresholds 不可與 seeds 1/3 混用。

每個 Stage-3 full run 恰為 28 targets×5 conditions=140 rows；seeds 2/3 合計 280 新 rows。
Model frozen、optimizer steps=0、model retrained=false、TEST scores reused=true；PR-AUC/AP/ROC-AUC
與該 seed Stage 2 一致，使用原 tolerance=1e-12。

Matching Per-machine/LOMO references 由原 canonical formal manifest、同 seed task namespace、唯一成功
attempt、worker args、metadata/source snapshots 定位，核對 TRAIN hashes/counts、TEST/interpretation hashes。
缺少或不相符就失敗，不能把 Universal seed 2 和 Per-machine seed 1 當成主要 paired comparison。

## 6. 本機驗證證據

| 範圍 | 通過數 |
| --- | ---: |
| 新 Universal replication tests | 13 |
| 原 Universal Stage 1–3 | 40 |
| 原 formal seed replication | 12 |
| Calibration regression | 39 |
| **合計** | **104** |

Tests 包含：supported/unsupported seeds、config equality、run/checkpoint namespaces、cross-seed artifact
拒絕、completion gates、one checkpoint、140 rows、five conditions、TEST immutability、model freeze、
finite extreme threshold 保留、historical seed-1 解析、seed-level ddof=1、machine directions，以及
凍結數值函式/POT RNG 的 AST 對照。原 tests 的 assertions 沒有削弱。

Seed-2 真實 TRAIN-only plan 確認 28 台、708405 windows；這只有讀檔/shape/hash/namespace 檢查，
沒有 model loading/training 或真實新 checkpoint。

本機使用既有 CPU environment，可重跑：

```bash
export PYTHON_BIN="../TranAD/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/universal-replication-tests-mpl"

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_universal*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_universal_replication.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_seed_replication.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_calibration*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_target_calibration.py -v
bash -n hpc/universal_train.sbatch hpc/universal_evaluate.sbatch hpc/universal_threshold.sbatch
git diff --check
```

Standalone replication suite 已執行 13 項；combined Universal discovery 與其他 suites 的重複執行
不重複計入 104 項。Synthetic fixtures／stub thresholds 不能當成 Hazel 真實結果。

## 7. Hazel：先 Seed-2 plan

使用者先完成 code review、commit/push，然後手動更新 Hazel checkout；以下不是 Codex 已執行的動作。

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/universal/replication-logs"

"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase plan
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase train --dry-run
```

Plan 在沒有 checkpoint 時仍可檢查全部 28 TRAIN、固定 config 與各 seed namespaces。
未知 checkpoint/code hashes 以 placeholder 顯示，不表示 artifact 已存在。
Evaluate/threshold dry-run 需要先前 stage completed，沒有 dependency 時直接失敗。
Dry-run 不載入 model、不 inference、不 fit POT、不建立 experiment outputs。

## 8. Seed 2：分階段提交

每個 job 完成並 review status/validation 後，再提交下一個。不要一次貼上全部提交命令。

Stage 1：

```bash
SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed2-train-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed2-train-%j.err" \
  hpc/universal_train.sbatch
```

Stage 1 成功後：

```bash
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase evaluate --dry-run
SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed2-evaluate-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed2-evaluate-%j.err" \
  hpc/universal_evaluate.sbatch
```

Stage-2 full 成功後：

```bash
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 2 --phase threshold --dry-run
SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed2-threshold-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed2-threshold-%j.err" \
  hpc/universal_threshold.sbatch
```

Evaluation 與 threshold phases 會各自在 fresh subprocess 做 smoke → full，保留原 smoke gates，
不因 seed-1 已成功就跳過新 checkpoint 的驗證。不提交新 Slurm jobs；subprocess 都在同一 allocation 執行。

已有 compute allocation 時，可改用 CLI，與 sbatch 擇一：

```bash
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 2 --phase train
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 2 --phase evaluate
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 2 --phase threshold
```

## 9. Seed-2 review 後才 Seed 3

先確認 seed-2 checkpoint/TEST/TRAIN cache identities、28-target ranking、140 threshold rows、model/test
hashes、continuous-metric invariants 與 empty POT stream 正確。不要因數值高低改 seed-3 設定。
真正 protocol bug 必須先停止釐清，不能只修較晚 seed；不要同時啟動 seeds 2/3。

```bash
"$PYTHON_BIN" scripts/run_universal_replication.py --hpc --seed 3 --phase plan
SEED=3 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed3-train-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed3-train-%j.err" \
  hpc/universal_train.sbatch
```

Seed-3 Stage 1 完成後：

```bash
SEED=3 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed3-evaluate-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed3-evaluate-%j.err" \
  hpc/universal_evaluate.sbatch
```

Seed-3 Stage-2 full 完成後：

```bash
SEED=3 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/universal/replication-logs/seed3-threshold-%j.out" \
  --error="$RESULTS_ROOT/universal/replication-logs/seed3-threshold-%j.err" \
  hpc/universal_threshold.sbatch
```

對應 compute-allocation CLI：

```bash
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 3 --phase train
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 3 --phase evaluate
"$PYTHON_BIN" -u scripts/run_universal_replication.py --hpc --seed 3 --phase threshold
```

## 10. Resources、routing 與 retry

| Phase | CPU／memory／walltime request |
| --- | --- |
| Train | 4 CPUs／16G／2 小時 |
| Evaluate | 4 CPUs／8G／30 分鐘 |
| Threshold | 4 CPUs／8G／2 小時 |

沿用 `csc591f26_cpu / compute_partners / short`，不因種子任意增加 resources。
這些是 request，不是實測 elapsed 或 MaxRSS；job 後使用：

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

對 seeds 2/3，evaluation/threshold sbatch route 到 replication wrapper，自動做該 phase 的 smoke/full。
舊 `UNIVERSAL_EVAL_MODE`／`UNIVERSAL_THRESHOLD_PHASE` 僅控制 seed-1 分支，不控制新 wrapper。

Completed phases 不 overwrite；會重新核對現有 artifacts。Failed retry 使用 CLI `--retry`；
evaluation/threshold sbatch 可設 `UNIVERSAL_REPLICATION_RETRY=1`，保留失敗 attempts。
Training 若需要 retry，直接在 compute allocation 使用 replication CLI 的 `--phase train --retry`。
不要同時啟動相同 phase；殘留 active.lock/running 要先人工確認 job 已停止，不自動刪 lock 或重送。

## 11. 每 seed 的輸出隔離

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

新 Stage-2 full 保存 `summary/target_metrics.csv`、`summary/summary.json` 與
`summary/matching_seed_comparison.csv`，記錄該 seed 的 PR-AUC/AP/ROC-AUC、prevalence、median/std/min/max。
沒有把 seed-2/3 table 誤標成 seed1_comparison。

Stage 3 保存 global/local thresholds、TRAIN score manifests、140-row condition table、Raw P/R/F1、
recovery 與 matching-seed Per-machine paired gaps。五個條件中每台 TEST score/label/timestamp hashes 相同。
Seed-1 training/evaluation/thresholding artifacts 不覆寫，也不與新結果混成同一個 individual run。

## 12. 三 seed aggregation

Seeds 2/3 完成後才執行：

```bash
"$PYTHON_BIN" scripts/analyze_universal_replication.py --hpc
```

若 seed 1 有多個歷史 successful code namespaces，先人工 review，再明確指定：

```bash
"$PYTHON_BIN" scripts/analyze_universal_replication.py --hpc \
  --seed1-threshold-run /實際/seed1/full/attempt
```

Aggregation 只讀，驗證完整 Stage-1/2/3 artifacts、cache provenance、pool/prefix counts、Raw/continuous
metrics、TEST hashes、matching formal references，以及三 seeds 相同的 TRAIN/TEST/labels/environment。
不重新 inference、fit POT 或重寫個別 seed summaries。

正確統計單位：

```text
seed1：28 targets macro
seed2：28 targets macro
seed3：28 targets macro
    ↓
三個 seed-level values → mean ± sample std，ddof=1
```

不能把 84 machine-seed observations 當作 84 independent seeds，也不聲稱 statistical significance。

輸出在新的 `universal/multiseed/analysis-*`：

| File | 內容 |
| --- | --- |
| `ranking_by_seed.csv` | Per／LOMO／Universal 的每 seed macro PR-AUC |
| `threshold_by_seed.csv` | 五種 Universal conditions 與 Per-machine 的每 seed macro Raw P/R/F1 |
| `multiseed_ranking_summary.csv` | 三 seed ranking values、mean、sample std、seed_count=3 |
| `multiseed_threshold_summary.csv` | 各 threshold condition 的三 seed Raw metric summary |
| `seed_level_gaps.csv` | 每 seed 的六種預定 paired gaps |
| `multiseed_gap_summary.csv` | 六種 gaps 的三 seed mean/sample std |
| `machine_direction_consistency.csv` | 每 target 三 seed 的方向、ties、正負翻轉旗標 |
| `multiseed_summary.json` | Input SHA256、direction counts、aggregation 定義 |

六種 gaps：

```text
Universal PR-AUC - LOMO PR-AUC
Per-machine PR-AUC - Universal PR-AUC
Universal local_05 F1 - Universal global F1
Universal local_20 F1 - Universal global F1
Per-machine F1 - Universal local_05 F1
Per-machine F1 - Universal local_20 F1
```

Machine consistency 只作 descriptive analysis。`all_improved`／`all_degraded` 要求三個 seed 方向一致，
`all_unchanged` 表示全為 0；其他為 mixed/ties。`direction_changed` 只有出現正、負兩種方向才為 true，
不把單純出現 tie 誤算成方向翻轉。

## 13. Review 與停止點

Seed 2 review 建議提供 plan JSON、三個 phase stdout/stderr、status/validation、Stage-2 summary、
Stage-3 macro/condition CSV、global/local threshold diagnostics，以及 sacct 輸出。
確認 code parameterization 與 seed isolation 後才 Seed 3；不要根據 seed-2 performance 改 protocol。

目前停在 **replication workflow 已準備、本機驗證完成，待 Git review 與 Hazel 手動執行**。
沒有可回報的真實 seed-2/3 PR-AUC、Raw F1、thresholds、training time 或 MaxRSS。

完成後先 review 三 seed 結果，不自動開始更多 seeds、POT tuning、fraction selection、normalization、
新 architecture 或其他研究 extension。
