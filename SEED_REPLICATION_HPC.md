# TranAD 多 Seed 實驗：輸出保護與 Hazel 操作

## 範圍與不變設定

本階段只補強 RQ1 的 training-seed robustness：保留已完成的 seed=1，新增 seed=2、3。
各 seed 使用完整 28 targets 的 per-machine / 27-source cross-machine 設定，新增 112 runs；
分析時合併原 seed=1 的 56 runs，共 168 runs。

不重跑 seed=1、不擴充 AT、不執行多 seed calibration。既有 RQ2 calibration 仍為 seed=1 證據。

固定 TranAD / SMD / protocol-v2、epochs=5、train/eval batch=128、window=10、lr=0.0001、
CPU float64、natural source-size weighting、deterministic window shuffle、validated labels、
原 loss / optimizer / score，以及 source-only POT（q=1e-5、level=0.99995、multiplier=1.06）。
每台獨立 windowing；cross-machine target TRAIN 不參與 training、normalization 或 threshold fitting。

## 為何使用獨立入口

原 `src/parser.py`、formal manifest / analyzer 固定 seed=1，且 calibration 會核對這些 core files 的原文指紋。
直接修改 parser 會使舊 checkpoint 的 calibration 相容性檢查失敗，因此此實作不修改既有 research source。

新增 worker 在全新的 subprocess，以已驗證的固定 Namespace 作為 `src.parser.args` adapter，
再透過 `runpy.run_path()` 執行原 `main.py`。這是 CLI / I/O 適配，不複製 training pipeline、
不改原模型或 windowing，也不把 seed=2、3 偽裝成 smoke。
Adapter 僅在此 subprocess 有效，原 `main.py` / `scripts/run_experiment.py` 的 seed=1 CLI 保持原行為。
精確 launcher / worker commands 保存在 `launch.json` / `worker_start.json`；後者也保存實際 args。
原 core metadata 的 command 由 `runpy` 呼叫環境產生，不應拿來直接重放舊 CLI；重現請用本文件的新 launcher。

`src/models.py` import 時雖有 `torch.manual_seed(1)`，但原 `prepare_run()` 在實際建立模型前
會重新設定 Python / NumPy / PyTorch 及 DataLoader generator 的實驗 seed。
因此新 seed 影響 initialization、training RNG 與 shuffle，並非只更改資料夾名稱。
本機測試驗證相同 seed 的 initialization / order 相同，不同 seed 的兩者不同，沒有執行 training。

## Manifest 與 Task Mapping

`configs/seed_replication.json` 定義本階段固定條件；`configs/seed_replication_runs.csv` 有 112 rows。
Machine order 和 sources 來自既有 canonical config / 56-row manifest，沒有新增第二套 machine list。

| Global task ID | Seed | Setting | Base task ID |
|---|---:|---|---|
| 0–27 | 2 | per_machine | 0–27 |
| 28–55 | 2 | cross_machine_zero_shot | 28–55 |
| 56–83 | 3 | per_machine | 0–27 |
| 84–111 | 3 | cross_machine_zero_shot | 28–55 |

Task 0 / 28 / 56 / 84 分別是 seed=2 per/cross、seed=3 per/cross 的 machine-1-1。
Manifest validation 必須完全符合 112-task 設計；不接受 CLI 覆寫 epochs、batch、seed 或 sources。

## 輸出位置與不覆寫機制

`RESULTS_ROOT` 維持共同外部 root，不要改指向 `runs/`、`summaries/` 或 `seed-replication/`。

```text
$RESULTS_ROOT/
├── runs/ ...                 # 舊 seed=1，唯讀
├── checkpoints/ ...          # 舊 seed=1，唯讀
├── summaries/ ...            # 舊 CSV / figures，唯讀
├── calibration/ ...          # 完全不動
├── anomaly_transformer/ ...  # 完全不動
└── seed-replication/
    ├── seeds/
    │   ├── seed-2/tasks/<base-task-setting-target>/
    │   │   ├── active.lock
    │   │   └── attempt-<UTC>-<suffix>/
    │   │       ├── launch.json
    │   │       ├── worker_start.json
    │   │       ├── status.json
    │   │       ├── run.log
    │   │       ├── deployment_source/
    │   │       ├── runs/<unique-seed-run-id>/
    │   │       │   ├── metadata.json
    │   │       │   ├── evaluation.json
    │   │       │   ├── scores.npz
    │   │       │   ├── training_progress.json
    │   │       │   ├── training_orders/
    │   │       │   └── ...
    │   │       └── checkpoints/<unique-seed-run-id>/model.ckpt
    │   └── seed-3/tasks/...
    ├── summaries/analysis-<UTC>-<suffix>/
    │   ├── seed_results.csv
    │   ├── paired_gaps.csv
    │   ├── target_seed_statistics.csv
    │   ├── seed_macro_results.csv
    │   ├── seed_macro_statistics.csv
    │   ├── groups_by_seed.csv
    │   ├── group_seed_statistics.csv
    │   ├── analysis.json
    │   ├── status.json
    │   ├── SEED_REPLICATION_RESULTS.md
    │   └── figures/*.png, *.pdf
    └── slurm-logs/
```

保護機制：

- 任何 task 已存在時，預設拒絕重跑；完成的 task 即使加 `--retry` 也拒絕。
- 失敗重試必須明確 `--retry`，並新增 attempt，保留舊 checkpoint / log / 數據。
- Task 用 exclusive lock 防止重複執行；輸出路徑拒絕 symlink 指向其他 namespace。
- 每個 attempt 的 worker 另以 exclusive `worker_start.json` 防止內部 command 被直接重放。
- 原 core 的 run name 已含 seed、UTC 與 UUID，checkpoint 使用 exclusive create。
- Summary 每次產生新 analysis directory；CSV / JSON / PNG / PDF 使用 exclusive create。
- 僅本次 active attempt 的 status 會更新；已完成的其他 attempts / analyses 不更新。
- Slurm logs 使用 job ID / array task ID，且 `--open-mode=append` 避免 requeue 時截斷 log。
- `deployment_source/` 保存原 core、新 adapter、config、manifest、hashes 與 Git provenance。

這些保護適用於本套新入口；請勿手動將舊 analyzer 指向新資料夾，或使用外部命令覆寫檔案。

## 分析方式

每個 seed 內先對相同 target 的 per/cross metrics 計算 paired gap：

```text
gap(target, seed) = per-machine(target, seed) - cross-machine(target, seed)
```

再對每個 target 報告 seeds=1,2,3 的 mean / sample std（ddof=1）。
另外每個 seed 先對 28 targets 計算 macro mean 與 degraded / improved counts，
再對三個 seed 的 macro metrics 算 mean / std。
Group summaries 也先在各 seed 內計算，避免將 machines 間差異冒充 seed variance。

主要看 PR-AUC / gap，補充 AP、ROC-AUC、Raw Precision / Recall / F1；PA-F1 只作 legacy comparison。
共 168 results、84 paired gaps；不能把 84 個 machine-seed pairs 當成 84 個獨立 seeds。

分析必須確認所有新 runs 完成，並檢查 metadata 的 seed、setting、source list、epoch/batch/dtype、
source-only POT、leakage checks、source snapshot、checkpoint path、data SHA256。
同一 base task 在各 seed 的 data SHA256 必須一致；舊 seed=1 使用同一批資料與原 source 的證據不足時會 fail。
同一 base task 的 Python / package versions / torch_threads 也必須一致，避免將環境更換誤解為 seed variance。
最近一次 attempt 失敗或缺 status 時，不回退舊成功結果。

原 seed=1 預設從共同 `$RESULTS_ROOT` 的 `task_status/` 與 `runs/` 讀取。
若舊 formal results 另有 root，分析可指定 `--reference-results-root`，或 sbatch 設 `SEED1_RESULTS_ROOT`。
原 checkpoint、scores、summary 與圖只讀，不搬移、不重建、不覆寫。

## 本機檢查與 Dry-Run

Config / manifest 檢查不讀 dataset、不訓練：

```bash
"$PYTHON_BIN" scripts/build_seed_replication_manifest.py
```

此檔已隨 code 建立。`--write` 僅供 CSV 尚不存在時使用，已存在就拒絕，不要每次 setup 重建。

本機 regression checks：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON_BIN" -m unittest discover -s tests -p 'test_seed_replication.py' -v
```

測試包含 synthetic artifacts、真正模型的 initialization（無 optimizer / backward）、shuffle、
CLI dry-run、launcher stub、完整合成分析與實際 PNG/PDF 產生。
Synthetic test metrics 不是真實研究結果，不能放入報告當成 seed=2、3 的結果。

Dry-run 只呼叫原 loader 的 describe：不初始化模型、不 training、不 fit POT、不建立 run。
會顯示實際 seed、sources、target、window count、dtype、POT configuration 與新輸出位置。

### 本次已完成的本機驗證

- 12 個 tests 全部通過（最終一輪 33.174 秒），包括 duplicate worker / task 拒絕、failed retry、
  latest-failure 不回退、錯 seed / 不同環境拒絕、source bytes regression，以及 CSV / PNG / PDF 不覆寫。
- 原 12 個受保護的 core files 與目前已 commit 版本逐 byte 相同，沒有改 parser / training / scoring。
- 112-row manifest 驗證通過；四個新增 shell / sbatch files 的 Bash syntax check 通過。
- 真實本機 SMD dry-run：原 seed=1 per-machine 與新 seed=2 per-machine 都為 28,479 windows；
  除 seed / output path 外，所有原 dry-run contract 欄位一致。
- Seed=2 的 27-source dry-run 為 679,926 windows，排除 machine-1-1 TRAIN；TEST 為 28,479 點。
- Seeds=2,3 的 per/cross 四個 routing 均在 synthetic dataset 以真實 CLI dry-run 驗證。
- 真實 dry-run 指定的 output root 未被建立；沒有執行 training / inference / POT fitting 或提交 HPC job。

這些是 local evidence。Hazel preflight、112 個正式 runs 與真實跨 seed statistics 仍待使用者執行。

## Hazel：使用者手動流程

本次實作不 commit / push / SSH / sbatch。先 review code，再由使用者 commit / push；
以下命令是之後在 Hazel 手動執行的步驟。

### 1. 更新與環境

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
# 保留原 CPU float64 environment 與原始共同 RESULTS_ROOT。
mkdir -p "$RESULTS_ROOT/seed-replication/slurm-logs"
"$PYTHON_BIN" scripts/build_seed_replication_manifest.py
```

`hpc/seed_replication_common.sh` 只設定此 job 的 CPU environment，不修改舊 common / env files。
Matplotlib 的暫存 cache 使用 job-specific temporary directory；實驗輸出全部位於外部 RESULTS_ROOT。

### 2. 先提交 Preflight（不訓練）

```bash
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/seed-replication/slurm-logs/preflight-%j.out" \
  --error="$RESULTS_ROOT/seed-replication/slurm-logs/preflight-%j.err" \
  hpc/seed_replication_preflight.sbatch
```

Preflight 使用 tasks 0、28、56、84 做四個 dry-runs。
資源：account=`csc591f26_cpu`、partition=`compute_partners`、qos=`short`、4 CPUs、8G、20 分鐘。
確認兩個 seeds 都正確、cross sources=27 且排除 target、dtype=float64、epochs=5、batch=128。

### 3. Preflight 確認後，手動啟動新增 112 Runs

```bash
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/seed-replication/slurm-logs/train-%A_%a.out" \
  --error="$RESULTS_ROOT/seed-replication/slurm-logs/train-%A_%a.err" \
  hpc/seed_replication_array.sbatch
```

Array=`0-111%4`；每個 task 為 4 CPUs、16G、2 小時，不用 GPU。
若先手動執行過部分 tasks，之後必須以 `--array=尚未執行的ID` 排除已完成 tasks；預設不會重跑它們。
這些資源沿用原 TranAD HPC 設計，新的完整 runs 仍需 Hazel runtime 驗證。

### 4. 進度與失敗檢查

```bash
squeue -u "$USER"
sacct -j 實際ARRAY_JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -f "$RESULTS_ROOT/seed-replication/slurm-logs/train-實際ARRAY_JOB_ID_28.out"
"$PYTHON_BIN" scripts/find_failed_seed_replication_runs.py --hpc
```

Run log 會顯示 seed / sources / target、`[1/7]` 到 `[7/7]` stages、epochs / batch progress 與 loss。
`find_failed...` 有 missing/failed 時 exit code=1，完整完成才為 0；不自動重試。

若確定 IDs 為失敗 tasks（例如 2、9），可明確另開重試 attempts：

```bash
sbatch --array=2,9%2 --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/seed-replication/slurm-logs/retry-%A_%a.out" \
  --error="$RESULTS_ROOT/seed-replication/slurm-logs/retry-%A_%a.err" \
  hpc/seed_replication_array.sbatch --retry
```

SIGTERM 會嘗試停止 child 並保存失敗狀態。若 SIGKILL / node failure 留下 `active.lock` / running status，
先人工用 Slurm 確認沒有活躍 job，將 stale lock **改名保留**，再用 `--retry --recover-interrupted`。
不自動判斷 lock 過期、不刪除任何舊 attempt，也不要對仍執行的 job 使用 recovery。

### 5. 全部完成後另存分析與圖

```bash
# 若 seed=1 原本與共同 RESULTS_ROOT 相同，無需設定；不同才 export 此變數。
# export SEED1_RESULTS_ROOT=/path/to/original-seed1-results
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/seed-replication/slurm-logs/summary-%j.out" \
  --error="$RESULTS_ROOT/seed-replication/slurm-logs/summary-%j.err" \
  hpc/seed_replication_summary.sbatch
```

Summary：2 CPUs、8G、30 分鐘；缺 run / 錯 seed / source 不一致時失敗，不輸出假完整報告。
正常時印出新的 `summaries/analysis-...` directory；重複分析會新增 directory，舊 CSV / PNG / PDF 保留。

圖包括 target PR-AUC gap mean±std、Raw F1 gap mean±std，以及各 seed 的 per/cross macro PR-AUC。
圖中 error bars 為三個 seeds 的 sample std，不是 confidence intervals。

## 驗證界線

本機 checks / dry-runs 不能當成 Hazel 新 seeds 已完成或 paper reproduction。
在實際 112 runs 與分析完成前，報告仍應保留原 seed=1 的結果與限制，不能填入 synthetic tests 數據。
本階段不自動執行多 seed calibration、AT、更多 seeds 或其他架構。
