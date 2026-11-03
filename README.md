# transformer_ts_anomaly：TranAD CPU / protocol-v2 deployment

此 repository 是 validated TranAD working tree 的乾淨部署版。原本的研究紀錄、資料、checkpoint
與歷史結果保留在來源 repository；這裡只放 code、config、測試和部署文件。
來源版本與 dirty 狀態見 [PROVENANCE.md](PROVENANCE.md)。

## 固定研究設定

TranAD、SMD、protocol v2、CPU、float64、5 epochs、training/evaluation batch 128、window 10、
lr=0.0001、seed=1。每台獨立 windowing 後 concatenate，再用獨立 seeded generator shuffle
完整 windows；不做 balancing。Threshold 僅 fit source training scores，固定後才讀 target test。
Labels 為 validated interpretation-label union；Raw / PA 共用同一組原始 prediction，PR-AUC
與 ROC-AUC 使用 continuous scores。沒有 CUDA/MPS、target adaptation 或其他模型實驗。

`main.py` 的四個核心函式和 SMD conversion 函式保留來源原文；模型、attention layers、POT、
SPOT、constants 與 shuffle helper 保留原始 bytes。共用 models.py 內仍有上游未使用的 class，
CLI 只允許 TranAD，沒有新增或執行其他模型。

## Code、data、results 分離

```text
PROJECT_ROOT  → 此 Git checkout
DATA_ROOT     → 外部 SMD，內含 train/、test/、interpretation_label/、labels/ 或 test_label/
RESULTS_ROOT  → 外部 runs/、checkpoints/、logs/、cache/、summaries/
```

HPC 模式必須明確設定三個 roots；它們不能相同或互相包含。Code checkout 不存 data 或結果。
Local 模式可用 repo-relative data default，但不存在時明確失敗；沒有搜尋使用者電腦舊目錄的 fallback。
每個 run 的 processed cache 位於自己的結果目錄，避免 array jobs 競爭同一個 `.npy`。

## Python environment

原環境是 CPython 3.13.0 / torch 2.8.0 / NumPy 2.2.4。Linux/Hazel 尚未執行驗證。
先在 compute allocation 或依學校規範建立外部 Python environment：

```sh
python3.13 -m venv /path/to/external-env
/path/to/external-env/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
/path/to/external-env/bin/python -m pip install -r requirements-hpc.txt
```

`/path/to/...` 是需自行替換的 example，不是程式中的固定位置。沒有安裝 DGL、torchvision、
torchaudio 或 SciencePlots，因本部署執行路徑不需要它們。模型計算沒有因此更動。
CPU wheel 方式依 [PyTorch 官方歷史版本安裝說明](https://pytorch.org/get-started/previous-versions/)。

## 手動 dry-run

```sh
export PROJECT_ROOT=/path/to/transformer_ts_anomaly
export DATA_ROOT=/path/to/SMD
export RESULTS_ROOT=/path/to/tranad-results
export PYTHON_BIN=/path/to/external-env/bin/python

"$PYTHON_BIN" scripts/run_experiment.py --hpc --dry-run --machine machine-1-1
"$PYTHON_BIN" scripts/run_experiment.py --hpc --dry-run \
  --manifest configs/formal_28_machine_runs.csv --task-id 28
```

Dry-run 會讀資料確認來源、target、window count、protocol、POT、seed、model config、dtype；
不訓練、不寫 processed cache、不產生實驗結果。

## 小型 smoke test

先在本機使用 fixture 驗證（需要來源 checkout 僅作 read-only reference）：

```sh
"$PYTHON_BIN" scripts/local_regression_check.py \
  --source-root /path/to/validated-source --report /path/to/external-validation-report.json
```

此檢查在暫存目錄建立 80 train / 48 test / 38 features 的 fixture，只訓練 1 epoch，
比較來源與 deployment 的 loss、參數、順序、分數、標籤與 metrics；不是正式 research run。

在 Hazel 做獨立 smoke 時，使用 `hpc/smoke_test.sbatch`；它對單機跑 1 epoch，沒有偷偷截短資料。
正式 manifest 不接受任意 hyperparameter override；smoke 需要明確 `--smoke-test`。

## 56-run manifest 與 Slurm

`configs/formal_28_machine_runs.csv` 從實際 SMD 名單產生，含 28 per-machine + 28 zero-shot rows。
先用外部 dataset 驗證 manifest：

```sh
"$PYTHON_BIN" scripts/build_formal_manifest.py --hpc
```

接著依 [hpc/README_HPC.md](hpc/README_HPC.md) 手動 smoke、array、summary。
本專案不會自動 `sbatch`、SSH 或 push；沒有任何已完成 Hazel 結果的宣稱。
所有 Slurm output/error 路徑在提交時明確指向外部 `RESULTS_ROOT/slurm-logs`。

## 結果與失敗檢查

```sh
"$PYTHON_BIN" scripts/find_failed_runs.py --hpc
"$PYTHON_BIN" scripts/analyze_formal_experiments.py --hpc
```

每個 run 使用唯一 run_id、checkpoint exclusive creation。每次 array attempt 分開記錄 status。
分析僅使用完整、成功、非 smoke 的固定 protocol 結果；缺失或失敗 task 會明確報錯，不會輸出假完整 table。
Summary 包含 Raw / PA、continuous AUC、gaps、floor effect、aggregate/group statistics 和 systems count。
LOMO 的 28 shared checkpoints 是 evaluation model count，不是 production 必須部署的數量。
