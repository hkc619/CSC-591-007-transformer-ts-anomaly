# Hazel / Slurm 手動部署

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
