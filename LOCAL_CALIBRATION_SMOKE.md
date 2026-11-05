# Calibration 本機 synthetic smoke

日期：2026-11-05。依使用者後續要求，執行本機小型程式檢查；這項授權取代先前「只做 static review」對 local tests 的限制。

結果：**27 tests 通過**。涵蓋四個 CLI 的 help/import、single/manifest dry-run、frozen inference、
metrics、artifact isolation、attempt/retry、synthetic summary CSV/Markdown 與五類 PNG/PDF 圖表。
經 shell smoke 入口實際執行為 `Ran 27 tests in 26.824s / OK`；使用已建立的 font cache 直接重跑
suite 約 5 秒。首次 cache 建立與主機負載會影響時間。

這是 synthetic code smoke，不是 SMD calibration experiment，也不是 Hazel validation。

## 如何在本機重跑

在 repository 根目錄，使用已安裝依賴的 Python environment：

```bash
PYTHON_BIN="../TranAD/.venv/bin/python" bash scripts/local_calibration_smoke.sh
```

上述相對路徑是本機既有 validated environment 的使用範例；script 本身不依賴 TranAD sibling
位置。其他環境請自行設定 PYTHON_BIN；未設定時使用 python3，需已安裝 requirements-hpc.txt。
不需要 DATA_ROOT、RESULTS_ROOT 或 SMD dataset。

腳本實際執行：

```bash
"$PYTHON_BIN" -m unittest discover -s tests -p 'test*calibration*.py' -v
```

Script 設定 CPU threads 與無 GUI backend，禁止寫入 bytecode；fixtures 使用暫存目錄並自動清除。
Matplotlib font cache 放在系統暫存目錄，第一次執行可能需要較久。
本次完整 console log 暫存在 `/tmp/tranad-calibration-local-tests.log`；暫存檔可能被系統清除，
此文件保留測試範圍、版本與結果紀錄。

## 測試範圍與證據

| 範圍 | 本機測試 |
| --- | --- |
| Protocol / manifest | 4 tests；fractions、84 tasks、prefix floor、namespace |
| Runtime / artifacts | 15 tests；frozen TranAD、no optimizer/backward、POT input wiring、invariants、錯誤拒絕 |
| Attempts / analysis | 4 tests；固定 ID、retry、incomplete attempt、統計與實際圖表輸出 |
| CLI | 4 tests；四個 scripts 的 --help、single/manifest dry-run、invalid fraction、CLI smoke |

CLI smoke 走真實 argparse、checkpoint loading、windowing、frozen inference、metrics 與 artifact writer。
Checkpoint 是隨機且未訓練的 fixture weights，epoch=4 只是模擬正式 metadata schema；不是正式模型。
Target fixture 為 200 train rows、38 features，5% prefix 為 10 rows；TEST archive 為 6 rows。
POT 使用明確的 stub / FakeSPOT，沒有執行真實 extreme-value fitting；只驗證輸入與程式串接。
Synthetic summary 使用 84 個人工 rows，直接測試 writer / plotting helper，不冒充 84 個完成的研究 runs。

第一輪發現三個 path-comparison failures：macOS 的 `/var/...` 與 `/private/var/...` 指向同一目錄。
已將三個 test files 的 fixture roots canonicalize，再重跑通過。沒有放寬 production isolation checks，
也沒有因測試修改模型、windowing、POT 或 evaluation semantics。

PyTorch 會提示原 custom Transformer layer 未使用 nested-tensor fast path；本次沒有因此改動原 Transformer。
失敗案例測試可能刻意印出 traceback，請以 suite 最後的 Ran 27 tests / OK 判讀結果。

## 本機環境

| 項目 | 值 |
| --- | --- |
| Python | 3.13.0 |
| PyTorch | 2.8.0 |
| NumPy | 2.2.4 |
| SciPy | 1.16.1 |
| scikit-learn | 1.7.1 |
| Matplotlib | 3.10.7 |
| OS | macOS 26.6.1 / arm64 |
| Inference | CPU / float64，synthetic frozen model |

## 尚未驗證

- 真實 Hazel checkpoint / formal artifact paths 與 source snapshots。
- 真實 target TRAIN prefix 的 POT 收斂、threshold 與 metrics。
- Hazel Linux imports / binary compatibility、Slurm、資源使用。
- machine-1-1 / 5% 真實 calibration smoke。
- 完整 84-task calibration 與 research conclusions。

以上均為 **PENDING HAZEL VALIDATION**。Tests 通過只表示目前這組測試未發現程式錯誤，不能保證所有情境無誤。

沒有 training、真實資料校準、formal-result 變更、SSH、Slurm submission、git commit 或 push。
後續仍先 review code，再由使用者更新 Hazel，停在 machine-1-1 / 5% 真實 smoke review。
