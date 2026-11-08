# Anomaly Transformer 本機驗證與 Smoke Readiness

日期：2026-11-07。這是 local correctness evidence，不是 Hazel/GPU 或論文 reproduction。

## 實際執行結果

- AT suite：**15 tests 通過**，最後一輪 `Ran 15 tests in 19.739s / OK`。
- 與既有 calibration 合併 regression：**42 tests 通過**，`Ran 42 tests in 24.136s / OK`。
- 最後補上 Git dirty metadata 與 source snapshot 的 MIT license/provenance 後，另重跑 AT suite 通過。
- 新增 score distribution diagnostics 後，再執行對應 artifact/orchestration test：1 test / 1.123s / OK。
- 12-row manifest validator 成功；四個 HPC shell scripts 的 `bash -n` 成功。
- 本機實際 SMD 的兩個 dry-runs 成功，未產生 result directory。

本機：Python 3.13.0、PyTorch 2.8.0、NumPy 2.2.4、Apple arm64、CPU float32；CUDA unavailable。
沒有安裝或更動套件，使用原已存在的 Python environment。

## 每個 AT test

| Test | 結果 |
| --- | --- |
| Native forward，B=2 / W=100 / F=38，完整 512/8/3-layer model | 通過 |
| Synthetic native minimax optimizer step，finite losses/gradients、weights 更新 | 通過 |
| 官方 detach / loss 方向與兩次 reconstruction gradient contribution | 通過 |
| Native energy 的 layer sum、temperature=50 公式 | 通過 |
| 與 validated TranAD continuous metrics 的 exact numeric comparison | 通過 |
| Machine boundaries 與 seeded whole-window shuffle | 通過 |
| Cross target TRAIN / identity-normalization isolation | 通過 |
| TRAIN file 不可 alias TEST 或 target TRAIN | 通過 |
| Timestamp mapping，含非 100 整除的右對齊尾段 | 通過 |
| 固定 6×2 manifest 與 artifact namespace isolation | 通過 |
| 四個 CLI 的 import / --help | 通過 |
| Per-machine / cross-machine synthetic CLI dry-runs | 通過 |
| Mocked-training orchestration、TEST 延後載入、原 labels、checkpoint reload | 通過 |
| CUDA unavailable：failed attempt、不自動改 CPU | 通過 |
| Synthetic 12-record analysis、唯讀 TranAD comparison、missing task guard | 通過 |

重要範圍：只有一次 synthetic minibatch 做真正 minimax optimizer update；沒有用真實 SMD 訓練。
Orchestration test mock 10-epoch completion，僅驗證 runner 的 stage order、實際 frozen inference、
metadata/artifacts、checkpoint reload；不能拿它證明 10 epochs 已實際完成。
Analysis test 的 12 個人工 records 僅在 temporary directory 建立，沒有執行 12 個正式 runs。
既有 calibration regression 本身使用 synthetic fixtures / POT stub，不是重做 RQ2。

## 真實 SMD dry-run

| 項目 | machine-1-1 per_machine | machine-1-1 cross_machine |
| --- | ---: | ---: |
| Sources | target 自己 | 其餘 27 台 |
| TRAIN raw points | 28,479 | 679,926 |
| Complete TRAIN windows，W=stride=100 | 284 | 6,785 |
| TRAIN dropped tail points 合計 | 79 | 1,426 |
| TEST timestamps / labels expected | 28,479 | 28,479 |
| TEST inference windows | 285 | 285 |
| Target TRAIN used for training | true | false |
| Target TRAIN used for normalization | false | false |
| TEST used for validation | false | false |

Normalization 兩者均為 identity/no fitting，與現有 TranAD supplied values 相同。Dry-run 讀取/檢查
source TRAIN 與 TEST 文字行數，不初始化 model、不 inference、不訓練、不寫實驗結果。

本次 raw dry-run JSON 暫存於 `/tmp/at-dry-per_machine.json`、`/tmp/at-dry-cross_machine.json`。
結果 root `/tmp/at-dry-results` 沒有被建立；這些暫存檔可能被系統清除。

實際 local 命令（從 deployment repository root）：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../TranAD/.venv/bin/python -m unittest discover -s tests -p 'test_anomaly_transformer.py' -v

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../TranAD/.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v

PROJECT_ROOT="$PWD" DATA_ROOT="$PWD/../TranAD/data/SMD" RESULTS_ROOT=/tmp/at-dry-results \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  ../TranAD/.venv/bin/python scripts/run_anomaly_transformer.py --hpc --device cpu \
  --target machine-1-1 --setting per_machine --dry-run

PROJECT_ROOT="$PWD" DATA_ROOT="$PWD/../TranAD/data/SMD" RESULTS_ROOT=/tmp/at-dry-results \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  ../TranAD/.venv/bin/python scripts/run_anomaly_transformer.py --hpc --device cpu \
  --target machine-1-1 --setting cross_machine --dry-run
```

程式沒有 hard-code 上述本機資料位置；paths 由 env/CLI 解析。測試 log 位於
`/tmp/at-local-tests.log`、`/tmp/at-and-calibration-tests.log`，沒有加入 Git 或修改歷史 results。

## 尚待 Hazel 驗證

| 項目 | 狀態 |
| --- | --- |
| Private GPU account / partition / QOS association | PENDING HAZEL |
| 獨立 GPU PyTorch environment / driver compatibility | PENDING HAZEL |
| CUDA deterministic kernels / native backward | PENDING HAZEL |
| machine-1-1 per-machine 10 epochs | PENDING HAZEL |
| machine-1-1 cross-machine 10 epochs | PENDING HAZEL |
| 真實 training loss、native energies、ranking metrics | PENDING HAZEL |
| Runtime / peak GPU memory / 1-hour request 可行性 | PENDING HAZEL |
| 剩餘 targets / full 12-run robustness results | 未執行，等待兩個 smoke review |

本機沒有 hpc/env.sh，沒有 SSH、sbatch、srun、salloc、commit 或 push。
下一步精確手動命令見 [ANOMALY_TRANSFORMER_ROBUSTNESS.md](ANOMALY_TRANSFORMER_ROBUSTNESS.md)。
**停止點：只準備 tasks 0 / 1 的 Hazel smoke，review 前不提交其餘 10 runs。**
