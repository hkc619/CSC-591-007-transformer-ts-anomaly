# Target threshold calibration

完整、目前有效的研究設計、artifact schema、tests 與 Hazel 手動步驟見
[CALIBRATION_PROTOCOL.md](CALIBRATION_PROTOCOL.md)。

狀態：**CODE READY FOR GIT REVIEW**。

- 既有 27-source cross-machine checkpoint，CPU float64 / seed 1，完全 frozen。
- Target TRAIN 前 5% / 10% / 20% raw observations，先切 prefix，再用原 windowing / inference。
- 同一 POT helper，只 fit prefix scores，TEST stream 為空。
- TEST scores / labels 重用原 formal artifacts；只有 threshold 與衍生 Raw / PA decisions 改變。
- Calibration 為 lightweight target adaptation，不是 zero-shot。
- 固定 experiment ID，獨立 attempts；預設拒絕覆寫，重試需明確 --retry。
- 新產物只寫 RESULTS_ROOT/calibration；舊 formal results / checkpoints / summaries 保留。

Manifest：`configs/calibration_runs.csv`，`task_id,target,fraction,seed`，預期 84 tasks。

原先只建立與審查程式碼；依使用者後續要求，已執行並通過 27 個 local synthetic tests。
涵蓋 fixture CLI dry-run / smoke、frozen inference 與圖表，POT 使用 stub。
詳見 [LOCAL_CALIBRATION_SMOKE.md](LOCAL_CALIBRATION_SMOKE.md)。

沒有 training、真實資料校準、SSH、Slurm 或 git add / commit / push / pull。
**Tests executed: YES — local synthetic only。真實 SMD / Hazel：PENDING HAZEL VALIDATION。**

使用者下一步：Git review → 自行 commit/push → Hazel pull → manifest/config validation →
machine-1-1 / 5% dry-run → smoke → inspect invariants → STOP。

**STOP AFTER MACHINE-1-1 5% SMOKE。** 只有使用者與 ChatGPT review 真實 smoke 後，才決定後續 84 tasks。
