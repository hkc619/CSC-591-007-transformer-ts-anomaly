# Hazel / Slurm 手動部署

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
