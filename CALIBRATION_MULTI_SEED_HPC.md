# TranAD RQ2：Seed 2 / 3 Target-Threshold Calibration

## 範圍與科學控制

RQ1 seeds=1,2,3 已依使用者回報完成；seed=1 的 84 個 calibration runs 也已完成。
本次只新增 seed=2、3：28 targets × 3 fractions × 2 seeds，共 **168 tasks**。
不重訓、不 fine-tune、不重跑 seed=1，也不修改任何既有 formal / calibration artifacts。

每個 task 只使用同 `(target, formal model seed, cross_machine_zero_shot)` 的 checkpoint、
TEST score archive、labels 與 zero-shot evaluation；模型 frozen，TEST scores 不重新 inference。
Target TRAIN 先取 `[0:floor(N*fraction)]` raw temporal prefix，再沿用原 window=10 / padding / target alignment。
Fractions 固定 0.05 / 0.10 / 0.20，沒有 random sampling、raw shuffle 或 TEST-driven fraction selection。

保留 CPU float64、eval batch=128、`eval()` / `no_grad()` / `requires_grad=False`、無 optimizer / backward，
POT q=1e-5、requested_level=0.99995、multiplier=1.06 與原 initialization retry。
POT 只接收 calibration TRAIN scores，TEST stream 必須為空。

**兩個 seed 概念必須區分：** CLI `--seed 2/3` 表示 formal checkpoint 的 training seed。
原 calibration 在 model load / POT 前固定重設 RNG=1；本次保留此行為，metadata 新增
`formal_seed=2/3` 與 `calibration_rng_seed=1`。因此只變動 formal model 的 training seed，
沒有額外變動 POT initialization RNG；不是誤用 seed=1 的模型。

## Formal Lookup 與路徑假設

使用上一階段 `configs/seed_replication_runs.csv` 的 canonical mapping：

```text
<FORMAL_BASE>/seed-replication/seeds/seed-2/tasks/<base-task-setting-target>/attempt-.../
<FORMAL_BASE>/seed-replication/seeds/seed-3/tasks/<base-task-setting-target>/attempt-.../
  status.json
  launch.json
  worker_start.json
  runs/<run-id>/{metadata.json,evaluation.json,threshold.json,scores.npz,source/...}
  checkpoints/<run-id>/model.ckpt
```

例如 machine-1-1：seed=2 的 training global task=28，seed=3 為 84；兩者 base task 都是 28。
Resolver 不掃描所有 seeds 的「最新 run」，只進入 canonical seed / target / cross-machine task。
最近 attempt 若失敗或不完整就 fail，不回退其他 seed；同一 task 有多個成功 attempts 也 fail，避免猜選。
再核對 status、launch / worker manifest row、effective args、formal metadata 的 seed / target / sources，
checkpoint 必須位於同 attempt；archive 不得 symlink 至其他 run。

新 seeds 的 formal base 預設為 `$RESULTS_ROOT`，**不套用舊 seed=1 的 `FORMAL_RESULTS_ROOT` fallback**。
若 seed=2/3 放在其他共同 root，可設定：

```bash
export SEED23_FORMAL_RESULTS_ROOT=/實際含seed-replication的共同結果根目錄
```

或在單次 CLI 使用 `--formal-results-root`。不要把此值指向單一 attempt 或 checkpoint directory。
Seed=1 舊 lookup / 既有 CLI defaults 保留；本次新陣列的 manifest 不含 seed=1。
若實際 Hazel artifacts 不符合上述已實作的 seed-replication layout，resolver 會 fail loudly，
需要先提供真實路徑 / schema；不自動 remap 或使用其他 seed。

## 輸出隔離與 Retry

```text
$RESULTS_ROOT/calibration/
├── runs/calib-...-seed1/...             # 既有，唯讀
├── summaries/...                       # 既有，唯讀
└── seed-replication/
    ├── runs/
    │   ├── smoke-calib-machine-1-1-f005-seed2/attempt-.../
    │   ├── calib-machine-1-1-f005-seed2/attempt-.../
    │   ├── calib-machine-1-1-f005-seed3/attempt-.../
    │   └── ...
    ├── task_status/
    ├── summaries/analysis-<UTC>-<suffix>/
    ├── cache/
    └── slurm-logs/
```

每個 run 保存原 metadata / launch / status / evaluation / threshold / validation / score archives / log。
Smoke 有獨立 `smoke-` identity，不計入 168 個 formal tasks。
重複 ID 預設拒絕；明確 `--retry` 才新增 attempt，舊檔保留；新 seeds 已成功的 task 即使 retry 也拒絕。
Failure checker 不回退最近失敗 / 損壞 attempt，且不自動 resubmit。
新 cache、summary、PNG / PDF 都位於新 namespace；summary 每次另建 unique directory，不改舊圖。

每個 run 的 `smoke_summary.json` / stdout 可直接看到 formal seed、training task ID、
checkpoint / score artifact 路徑、parameter hashes、threshold、Raw metrics、recovery 與 continuous metrics。
Runtime assertions 會檢查參數 / buffer hash、TEST score / label hash、count、原 artifacts file hashes 不變，
以及 PR-AUC / AP / ROC-AUC 在 absolute tolerance=1e-12 下相同。

## Manifest

```bash
"$PYTHON_BIN" scripts/build_calibration_manifest.py \
  --seeds 2 3 --output configs/calibration_runs_seed23.csv
```

檔案已建立；重複執行只驗證，不覆寫。舊 `configs/calibration_runs.csv` 的 84 rows 不變。
Task 0–83 為 seed=2，84–167 為 seed=3；各 seed 依 canonical machine order，每台依 5% / 10% / 20% 排列。
因此 task 0 與 84 都是 machine-1-1 / 5%，分別使用 seed=2、3。
原 `target_calibration.json` 保留 seed=1 default；新 run 的有效 model seed 由 CLI / manifest 傳入，
寫入該 run 自己的 config / metadata，不直接修改舊 default config。
Generator 可指定其他不重複 seeds，但 runtime resolver 只允許目前已有 formal mapping 的 seeds=1,2,3；
不代表其他 seeds 的 checkpoints 已存在。本階段執行只使用這份 168-row manifest。

## 本機 Regression Tests

使用既有 CPU environment，沒有 training 或 real SMD experiment：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/private/tmp/tranad-calibration-tests-mpl \
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test*calibration*.py' -v
```

包含全部 27 個原 tests，未移除或弱化 assertions；新 tests 使用 synthetic checkpoints，
真的執行 frozen TranAD prefix inference，POT 明確使用 stub，另由原 SPOT test 驗證空 TEST stream 與固定參數。
測試涵蓋 168 rows、seed-specific checkpoint / archive、跨 seed 指向拒絕、seed-aware smoke、
model hash / scores / labels / AUC invariants、原 seed=1 sentinel 不變、deep artifact audit、
sample std ddof=1、machine recovery 分類，以及三類新圖與原圖不覆寫。
Synthetic metrics / generated tables 不是研究結果，不得放入報告宣稱已完成真實 seed=2/3 calibration。

本次最終本機結果：**39 tests，7.904 秒，全數通過**（既有 27 + 新增 12）。
Seed=2 / 3 的 non-smoke fixture artifacts 也通過既有 deep audit；
跨 seeds 的 TRAIN prefix / TEST label 不一致會被拒絕。五個 shell / sbatch files 的 Bash syntax check 通過。
原 `main.py`、models、parser、calibration runtime、training protocol、POT/SPOT 與 seed=1 config / manifest 皆未修改。

## Hazel 執行順序

以下是使用者之後手動執行的指令。本次沒有 SSH、sbatch、training 或真實 Hazel calibration。
Code review 並由使用者 commit / push 後，才更新 Hazel；既有 RESULTS_ROOT / DATA_ROOT 不搬移。

### 1. 更新、Manifest 與 Seed-2 Dry-Run

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
mkdir -p "$RESULTS_ROOT/calibration/seed-replication/slurm-logs"

"$PYTHON_BIN" scripts/build_calibration_manifest.py \
  --seeds 2 3 --output configs/calibration_runs_seed23.csv

"$PYTHON_BIN" scripts/run_target_calibration.py --hpc \
  --target machine-1-1 --fraction 0.05 --seed 2 --dry-run
```

Dry-run 必須顯示 `seed=2`、`formal_seed=2`、`formal_task_id=28`，
checkpoint / score artifact 位於 seed-2 task 的同一 attempt，output namespace 在 calibration/seed-replication。
Dry-run 不載入模型、不 inference、不 fit POT、不建立 calibration result。

### 2. 只提交 Seed-2 / 5% Smoke

```bash
CALIBRATION_SEED=2 sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/smoke-%j.err" \
  hpc/calibration_smoke.sbatch
```

資源沿用 account=csc591f26_cpu、partition=compute_partners、qos=short、1 node、4 CPUs、8G、30 分鐘。
若使用已分配的 compute allocation，也可直接執行：

```bash
"$PYTHON_BIN" scripts/run_target_calibration.py --hpc \
  --target machine-1-1 --fraction 0.05 --seed 2 --smoke-test
```

不要在 login node 執行 inference。Smoke 完成後檢查 stdout 指出的確切 attempt：

```bash
CALIBRATION_SMOKE=/實際的smoke-calib-machine-1-1-f005-seed2/attempt-目錄
cat "$CALIBRATION_SMOKE/status.json"
cat "$CALIBRATION_SMOKE/smoke_summary.json"
cat "$CALIBRATION_SMOKE/validation.json"
cat "$CALIBRATION_SMOKE/metadata.json"
```

需確認 completed=true、seed/formal_seed=2、model_retrained=false、optimizer_steps=0、
parameter hashes 相同、continuous scores reused、AUC 不變、POT stream_score_count=0。
**目前停在這個 smoke gate。下列完整 array / retry / summary 指令僅供 smoke 確認後使用。**

### 3. Smoke 確認後才提交完整 168 Tasks

```bash
CALIBRATION_MANIFEST=configs/calibration_runs_seed23.csv \
sbatch --array=0-167%8 --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/array-%A_%a.out" \
  --error="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/array-%A_%a.err" \
  hpc/calibration_array.sbatch
```

原 array header 保留 seed=1 的 0–83 default；新命令必須同時指定 manifest 和 0–167 range。
此 array 沒有 training，也不包括 seed=1。Smoke 不替代正式 task 0，兩者 identities 不同。

### 4. 失敗檢查與明確 Retry

```bash
"$PYTHON_BIN" scripts/find_failed_calibration_runs.py --hpc \
  --manifest configs/calibration_runs_seed23.csv
```

`count=0` 表示基本 artifact / metadata / validation checks 完整；資料內容仍由後續 deep analysis 稽核。
Count 非零時 exit code=1。先確認對應 jobs 已終止，不對 active jobs 重複 retry。
例如確認只需 retry IDs 7、100：

```bash
CALIBRATION_MANIFEST=configs/calibration_runs_seed23.csv \
sbatch --array=7,100%8 --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/retry-%A_%a.out" \
  --error="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/retry-%A_%a.err" \
  hpc/calibration_array.sbatch --retry
```

這是建立新 attempts，不刪除／改寫失敗證據，不自動 resubmit。

### 5. 全部完成後：三 Seed Summary

```bash
CALIBRATION_SEEDS="1 2 3" sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/summary-%j.out" \
  --error="$RESULTS_ROOT/calibration/seed-replication/slurm-logs/summary-%j.err" \
  hpc/calibration_summary.sbatch
```

已分配 compute allocation 時，等價入口為：

```bash
"$PYTHON_BIN" scripts/analyze_calibration.py --hpc --seeds 1 2 3
```

Summary 讀取 seed=1 的 calibration/runs 與 seed=2,3 的 calibration/seed-replication/runs，
必須有全部 252 個 formal calibration runs；smoke 不接受。
同 target/seed 的三個 fractions 必須對應相同 formal model / TEST artifacts；
跨 seeds 的同 target/fraction 必須使用相同 TRAIN prefix，TEST labels / count 也必須一致。
假設舊 seed=1 calibration 仍位於同一共同 RESULTS_ROOT；如果實際放在其他 root，先說明真實路徑，
不要搬動舊結果或以 symlink 猜測 remap。

## 結果檔案與統計

每次分析寫入新的 `calibration/seed-replication/summaries/analysis-...`：

| 檔案 | 內容 |
|---|---|
| calibration_results.csv | 全部 252 個稽核後的 machine/seed/fraction 結果 |
| seed_macro_results.csv | 每 seed × 4 settings，共 12 rows 的 Raw P/R/F1 macro means |
| seed_macro_statistics.csv | 每 setting/metric 的 seed 1/2/3、mean、sample std |
| seed_recovery.csv | 各 seed/fraction 的平均 Raw P/R/F1 recovery 與改善台數 |
| seed_recovery_statistics.csv | 三 seed 的 recovery mean±sample std |
| target_recovery_robustness.csv | 每 target/fraction 的 recovery mean/std、改善/下降/不變 seed 數與方向分類 |
| groups_by_seed.csv | 各 seed / group / fraction 的描述性 recovery |
| analysis.json / status.json | aggregation、input provenance 與完成狀態 |
| CALIBRATION_MULTI_SEED_RESULTS.md | 三 seed Raw F1 / Precision / Recall 主表及 recovery 摘要 |
| figures/raw_f1_by_setting.png/pdf | 四個 settings 的 seed-level mean±sample std |
| figures/raw_f1_recovery_by_target.png/pdf | 各 target 的三個 fractions recovery mean±std |
| figures/per_seed_calibration_summary.png/pdf | 每個 seed 的完整 calibration curve |

先各 seed 對 28 targets 取平均，再對三個 seed macro means 計算 sample std（ddof=1）。
Zero-shot 每 target/seed 只計一次，不因三個 fractions 重複計數；不是 84 個獨立 seeds。
Machine-level 分類包含全 seeds 改善、全下降、全不變、正負方向混合，以及單向含不變的 cases。
只有描述性統計，沒有宣稱統計顯著。全部 5% / 10% / 20% 都呈現，不能自動宣稱 5% 最佳、
每台都改善，或 threshold mismatch 是全部 cross-machine failure 的唯一原因。

## 執行界線

本機 synthetic smoke 通過，不等於 Hazel 真實 seed=2 checkpoint / POT smoke 通過。
本機沒有新 seeds 的真實 formal checkpoint / TEST archives，故真實 dry-run、smoke、168-task array
及三 seed RQ2 conclusions 均為 **PENDING HAZEL VALIDATION**。
本次不 commit、不 push、不 SSH、不 submit jobs；先完成 code review，再由使用者執行。
