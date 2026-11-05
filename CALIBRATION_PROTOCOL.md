# TranAD Target-Threshold Calibration Protocol

更新日期：2026-11-05。狀態：**CODE READY FOR GIT REVIEW**。

原實作階段只做 static review。依使用者後續新增的 local smoke 要求，目前已執行並通過
27 個 synthetic tests，包含 fixture CLI dry-run、frozen inference 與圖表輸出；POT 使用 stub。
詳見 [LOCAL_CALIBRATION_SMOKE.md](LOCAL_CALIBRATION_SMOKE.md)。

**Tests executed: YES — local synthetic only。真實 SMD / Hazel calibration：PENDING HAZEL VALIDATION。**
沒有 training、真實 POT fitting、SSH、Slurm 或任何 Git 寫入操作。

## 1. Research motivation 與目前階段

依使用者提供的目前狀態，RQ1 已完成 Hazel 56 個正式 runs，failed/missing=0；既有本機
`reports/` 亦有相應摘要。Cross-machine PR-AUC 在 19/28 targets 下降、9/28 提升；
mean PR-AUC 約從 0.4075 降為 0.2918，mean Raw F1 約從 0.1363 降為 0.0077。
本次沒有重跑、重新驗證或修改這些 formal results。

RQ2：少量正常 target observations 能否在不重新訓練 Transformer 下，恢復部分 thresholded
detection performance？這是在既有 zero-shot 結果之外新增的 **lightweight target adaptation**。
不得把 calibration 後結果稱為 zero-shot performance。

核心控制：model 固定、target TEST 固定、continuous TEST scores 固定，只有 target calibration
data budget 與 threshold 改變。Raw Precision / Recall / F1 為主要結果，PA 僅為 legacy comparison。

## 2. 固定 protocol

| 項目 | 設定 |
| --- | --- |
| Backbone / dataset | 原 TranAD / SMD |
| Checkpoint | 每台 target 對應的正式 27-source cross-machine v2 模型 |
| Device / dtype / seed | CPU / float64 / 1 |
| Window / inference batch | 10 / 128 |
| Fractions | 0.05、0.10、0.20，事先固定 |
| Prefix | `[0:floor(N × fraction)]`，N 只由 target TRAIN row count 決定 |
| Normalization / fine-tuning | 不新增、不執行 |
| Optimizer / backward | 不建立、不呼叫，optimizer_steps=0 |
| POT | q=1e-5，requested_level=0.99995，multiplier=1.06，保留原 retry |
| TEST scores / labels | 重用同一份 formal `scores.npz` |

先切 raw temporal prefix，再呼叫原 `convert_to_windows`。不 shuffle calibration raw points、
不隨機挑樣本、不對全長 TRAIN windowing 後才切 windows。小於 10 筆但非空時沿用原 padding；
若 floor 後是 0 筆則停止，不擴張預算。每筆 score 對應 window 最後一筆 raw timestamp。

TRAIN 被視為正常資料；不使用 TEST labels、interpretation intervals 或 TEST metrics 來清洗、
選 fraction、選時間段、調 POT 或挑最佳結果。Prefix 間巢狀依賴需要在分析中保留，不當成獨立樣本。

## 3. Checkpoint mapping 與 no-retraining

Canonical machines 只來自 `configs/formal_protocol_v2.json`。Config loader 同時核對現有
`configs/formal_28_machine_runs.csv`：前 28 個 per-machine，後 28 個 cross-machine。
Target index i 的 formal cross-machine task 為 28+i。

Mapping：target → formal task → 最近正式 attempt status → run ID → metadata → checkpoint / scores。
最近 attempt 未成功即停止；不回退舊結果。缺 checkpoint、錯 target / setting / seed / protocol /
sources / dtype / device / model / dataset 均拒絕。原 snapshots 必須與目前被重用的 core files 一致。

原 formal metadata 未獨立存 model/dataset 欄位；對舊 schema，以 metadata 的原 command、
完全相同的 parser/model snapshots，以及 runtime 的 strict TranAD state_dict load 補足證據。
若 metadata 或 command 顯式寫了其他 model/dataset，直接拒絕，不以 default 掩蓋。

不呼叫會建立 optimizer 的 `main.load_model`。Adapter 僅建立同一 TranAD class，CPU float64
strict load `model_state_dict`，要求完成原 5 epochs 的 checkpoint（epoch=4），不恢復 optimizer。
執行 `model.eval()`、所有 parameters `requires_grad=False`，在 `torch.no_grad()` 中重用
`main.backprop(training=False)`。前後 hash 包含整份 state_dict 的 parameters 與 buffers。

**既有 provenance 限制：** 舊 checkpoint 本體沒有 target 身分欄位，也沒有事先保存的 checkpoint
hash。若檔案曾被放錯到一個看似合法的 formal path，單靠 tensor shapes 不能證明其原始 target。
本設計信任既有 immutable formal bundle，核對 task/path/metadata/snapshots，並從 calibration
開始保存 checkpoint hash；不宣稱可偵測更早發生的同架構檔案替換。

## 4. Threshold 與 leakage controls

```text
target TRAIN prefix → 原 windowing → frozen inference → calibration scores
    → src.cross_machine.fit_source_threshold(scores)
    → SPOT.fit(scores, empty_stream) → initialize → extreme_quantile × 1.06
    → 固定 threshold → 讀取 formal TEST arrays → Raw / PA evaluation
```

完全重用 POT helper，不呼叫 SPOT.run。原 initialization retry（level ×0.999）保留，記錄
effective_level / attempts。若 finite threshold 無法產生即停止，不依 TEST 結果換演算法或參數。
Fitting helper 的介面只收到一個 calibration-score vector，沒有 TEST scores / labels 引數。

- Training leakage：不進 training mode、不建 optimizer、不 backward；target prefix 只做 inference。
- Threshold leakage：只有 prefix scores 傳入 POT，stream_score_count 必須為 0。
- Test leakage：raw TEST 不載入、不重新 TEST inference；現有 archive arrays 在 threshold 固定後才讀。
- Label leakage：labels 僅供固定 threshold 後的 metrics / PA；不供 fitting 或設定選擇。
- Future temporal leakage：只 parse TRAIN prefix 數值，suffix 僅掃文字行數以計算 N；不讓未來值進 window。

原 archive 會在 fitting 前後以位元組串流 hash 保護；這不會解析或使用 TEST scores 來 fit POT。

## 5. Metrics 與 invariants

原 continuous scores → 原 threshold → zero-shot Raw / PA。
同一 scores → calibrated threshold → calibrated Raw / PA。PA 從 Raw prediction 的 copy 產生。

PR-AUC 使用 continuous-score PR curve 的 trapezoidal area，AP 與 ROC-AUC 另存。
重算原 threshold 的 baseline metrics 必須與原 evaluation 一致；calibrated AUC 與原 AUC 的
absolute tolerance=1e-12，否則 fail。Scores / labels 逐值與 hash 不變，長度不變。

Raw recovery：calibrated − zero-shot，分別保存 precision / recall / F1 delta。
`absolute_shift=abs(calibrated_threshold-zero_shot_threshold)`；另存 signed_shift。
`relative_ratio=calibrated/zero_shot`；分母絕對值 ≤1e-12 或 ratio overflow 時記錄 JSON null 與原因，
不寫 inf / NaN。Threshold 或 metric 不一定改善，不依 TEST 分數挑每台最好的 fraction。

## 6. Identity、retry 與 artifact isolation

```text
RESULTS_ROOT/calibration/
  runs/calib-machine-1-1-f005-seed1/attempt-<UTC-microseconds>-<uuid>/
  runs/smoke-calib-machine-1-1-f005-seed1/attempt-<UTC-microseconds>-<uuid>/
  task_status/
  summaries/<unique-summary-id>/
  slurm-logs/
  cache/
```

Experiment/run ID 由 target / fraction / seed 決定；attempt ID 用於保留每次執行。
同 experiment 預設拒絕重複寫入；使用者明確 `--retry` 才新增 attempt，不覆寫、不自動 resume。
最新 attempt 缺 status（例如 scheduler kill）仍視為 incomplete，不能回退舊成功紀錄。
Smoke identity 與正式 calibration identity 分開，不能混入 84-task summary。

每個 attempt 保存 launch、metadata、status、config、run.log、threshold、evaluation、validation、
calibration_scores.npz、scores.npz、smoke_summary.json。Smoke summary 同時列印至 stdout，包含
formal task/run/checkpoint、prefix/window counts、前後 parameter hashes、thresholds、Raw metrics、
recovery、PR-AUC/AP/ROC-AUC before/after 和 POT stream count。

Metadata 包含指定的 model_retrained、optimizer_steps、parameters_unchanged、parameter_hash_before /
after、formal_run_id、formal_task_id、formal_checkpoint/hash、formal_score_artifact/hash、target、
fraction、seed、calibration_source/raw_points/windows、window_size、POT requested/effective/attempts /
stream_count、continuous_scores_reused、Git commit（取不到時明確 unavailable）、environment 與 source hashes。

Formal inputs 只以讀取模式開啟；checkpoint 不另存、不修改。所有新實驗與 summary outputs 都限制於
calibration namespace，拒絕與 formal paths 重疊或 symlink alias；不改既有 formal CSV / JSON / Markdown。

## 7. Manifest 與完整分析規劃

`configs/calibration_runs.csv` schema：`task_id,target,fraction,seed`。
Canonical order 直接從既有 formal manifest 的 28 個 per-machine rows 以純文字轉換建立，沒有另一份
hard-coded machine list，也沒有執行 Python generator。CSV 為 header + 84 rows；目前只有 static inspection。
Runtime validator 核對整張表與 canonical 28×3 排列，涵蓋 task IDs、unique tuples、fractions、seed、缺失與多餘 targets。

所有 84 個非 smoke tasks 成功且 invariants 一致後，analysis 才寫入新的 summary directory：

- calibration_results.csv：84 rows，zero-shot / calibrated Raw 與 PA、AUC、thresholds、provenance。
- calibration_recovery.csv：Raw deltas、absolute/signed shifts、ratio。
- calibration_statistics.csv：zero-shot、5%、10%、20% 的 mean/median Raw Precision/Recall/F1、sample std F1、IQR F1。
- calibration_groups.csv：每 fraction × group 的 mean/median F1/Recall recovery、improved/degraded/unchanged counts。
- analysis.json：各 fraction / metric 的變化台數與方法定義。
- CALIBRATION_RESULTS.md：實際資料摘要與解讀限制。
- figures/：Raw F1 distribution、每 target F1 recovery、threshold scatter、Raw Recall distribution、group F1 recovery；各有 PNG/PDF。

Zero-shot 統計每台只計一次，不將三個 fractions 重複計成 84 個獨立 targets。同 target 的三個
fractions 必須來自同 formal run / checkpoint / scores / labels / evaluation；不一致停止。
Summary 另比對保存的 calibration score archive 與原 TEST scores / labels，原 formal artifacts hashes
也必須仍一致。沒有完整結果時不產生假完整 tables 或 figures。

不能聲稱 calibration 解決 cross-machine detection。若提升，只能支持 threshold adaptation 恢復了
此 protocol 下的部分 Raw performance；mixed results 須逐 target/group 描述。無改善不能單獨證明
全部原因就是 representation shift。限制包含 single seed、TRAIN-normal 假設、POT 小樣本 tail、
prefix dependencies、沒有 target normalization/fine-tuning、沒有獨立 calibration validation set。

## 8. Tests：本機 synthetic 已執行，Hazel 仍待驗證

`tests/test_calibration_protocol.py`、`test_target_calibration.py`、`test_calibration_analysis.py`、
`test_calibration_cli.py` 在本機以 synthetic fixtures 驗證：
fraction / raw prefix / padding、invalid 0.15、84-row manifest、no TEST leakage、model freeze、
禁止 optimizer/backward、scores/labels/AUC invariance、threshold-only change、wrong formal identity /
missing checkpoint 拒絕、artifact isolation、failure status、deterministic identity、explicit retry、
incomplete-latest 不回退，以及 zero-shot 去重與 group statistics。

**Tests executed: YES — local synthetic only，27 tests 通過。**

共 27 個 test methods（4 protocol、15 runtime/artifact、4 analysis/attempt、4 CLI），完整名稱如下：

```text
test_fraction_slicing_is_floor_and_nested
test_committed_manifest_has_exact_84_unique_tasks
test_manifest_rejects_missing_duplicate_and_wrong_seed
test_artifact_namespace_separate_and_rejects_aliases
test_prefix_loader_never_parses_train_suffix_or_test
test_actual_frozen_tranad_scores_repeat_without_optimizer
test_pot_receives_only_calibration_scores_and_empty_stream
test_only_threshold_changes_not_scores_labels_or_auc
test_archive_alignment_rejects_shifted_scores
test_threshold_ratio_near_zero_is_json_null_not_inf
test_checkpoint_with_wrong_model_state_rejected
test_continuous_metric_drift_is_rejected
test_short_raw_prefix_uses_original_padding_and_one_window_per_point
test_dry_description_does_not_load_model_or_create_outputs
test_orchestration_isolates_artifacts_and_never_reads_raw_test
test_rejects_baseline_with_target_training_leakage
test_failed_latest_attempt_does_not_fall_back
test_wrong_formal_identity_or_missing_checkpoint_rejected
test_pre_model_failure_is_recorded_as_failed_calibration_task
test_deterministic_identity_no_overwrite_and_explicit_retry
test_incomplete_new_attempt_blocks_older_success
test_statistics_zero_shot_counted_once_and_groups_not_assumed
test_synthetic_summary_writes_tables_report_and_five_figures
test_all_calibration_cli_help_entrypoints
test_single_and_manifest_dry_run_cli_with_external_fixture
test_invalid_fraction_cli_fails_without_outputs
test_cli_smoke_frozen_inference_stub_pot_and_saved_artifacts
```

下表限定真實 SMD / Hazel 情境；本機 fixture 通過不表示這些研究用 runtime checks 已完成。

| Runtime 項目 | 狀態 |
| --- | --- |
| parameter freeze / hashes | PENDING HAZEL VALIDATION |
| optimizer_steps=0 | PENDING HAZEL VALIDATION |
| scores / labels unchanged | PENDING HAZEL VALIDATION |
| PR-AUC / AP / ROC-AUC unchanged | PENDING HAZEL VALIDATION |
| POT stream_score_count=0 | PENDING HAZEL VALIDATION |
| CLI / imports / checkpoint compatibility | PENDING HAZEL VALIDATION |
| manifest / failed checker / analysis / figures | PENDING HAZEL VALIDATION |
| machine-1-1 5% smoke | PENDING HAZEL VALIDATION |

## 9. 使用者 Git review 與 Hazel workflow

本次沒有 git add/commit/push/pull。Local 先 review；本次新增檔案未追蹤時，普通 `git diff` 不顯示
其內容，需同時閱讀新檔，之後由使用者自行 staging 再看 cached diff。既有 `reports/` 與
`CHATGPT_STATUS_AND_HPC_PLAN.md` 不納入本次建議 staging。

```bash
git status
git diff --stat
git diff
git add src/calibration_protocol.py src/calibration_runtime.py src/target_calibration.py \
  src/calibration_artifacts.py src/calibration_analysis.py \
  scripts/run_target_calibration.py scripts/build_calibration_manifest.py \
  scripts/analyze_calibration.py scripts/find_failed_calibration_runs.py \
  scripts/local_calibration_smoke.sh \
  configs/target_calibration.json configs/calibration_runs.csv \
  hpc/calibration_common.sh hpc/calibration_smoke.sbatch hpc/calibration_array.sbatch \
  hpc/calibration_summary.sbatch hpc/README_HPC.md \
  tests/test_calibration_protocol.py tests/test_target_calibration.py tests/test_calibration_analysis.py \
  tests/test_calibration_cli.py TARGET_CALIBRATION.md CALIBRATION_PROTOCOL.md LOCAL_CALIBRATION_SMOKE.md
git diff --cached --stat
git diff --cached
git commit -m "Add target threshold calibration experiment"
git push
```

之後由使用者在 Hazel 更新：

```bash
cd "$PROJECT_ROOT"
git pull
source hpc/env.sh
export FORMAL_RESULTS_ROOT="${FORMAL_RESULTS_ROOT:-$RESULTS_ROOT}"
mkdir -p "$RESULTS_ROOT/calibration/slurm-logs"
```

若正式結果位於另一個外部 root，明確設定 FORMAL_RESULTS_ROOT；不自動重寫歷史 absolute paths。
以下 Python 工作須在合適 compute allocation 執行，本文件不申請或執行 allocation。

**步驟 1：manifest/config validation。** 本機 synthetic tests 已通過；Hazel environment 仍應重新檢查。

```bash
"$PYTHON_BIN" scripts/build_calibration_manifest.py --hpc
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_calibration*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_target_calibration.py' -v
```

**步驟 2：machine-1-1 / 5% dry-run。** 不載入模型、inference、POT 或寫結果。
預期顯示 target/fraction/seed、TRAIN total/prefix/windows、formal task/run/checkpoint/score/evaluation、
protocol/device/dtype/POT。若 TRAIN N=28,479，prefix=[0,1423)，windows=1423。

```bash
"$PYTHON_BIN" scripts/run_target_calibration.py \
  --hpc --target machine-1-1 --fraction 0.05 --dry-run
```

**步驟 3：僅提交 machine-1-1 / 5% smoke。**

```bash
sbatch --chdir="$PROJECT_ROOT" \
  --output="$RESULTS_ROOT/calibration/slurm-logs/smoke-%j.out" \
  --error="$RESULTS_ROOT/calibration/slurm-logs/smoke-%j.err" \
  hpc/calibration_smoke.sbatch
```

帳號設定沿用使用者提供的 csc591f26_cpu / compute_partners / short；smoke 與 array 範本為
1 node、1 task、4 CPUs、8G、30 分鐘；summary 為 1 node、2 CPUs、4G、15 分鐘。
這些 calibration resource requests 尚未實測。

**步驟 4：檢查排程與 logs。** 將 JOBID 換成實際值。

```bash
squeue -u "$USER"
tail -f "$RESULTS_ROOT/calibration/slurm-logs/smoke-JOBID.out"
cat "$RESULTS_ROOT/calibration/slurm-logs/smoke-JOBID.err"
```

**步驟 5：檢查產物。** 將實際 attempt 名稱替換成 stdout 顯示的資料夾。

```bash
export CALIBRATION_RUN="$RESULTS_ROOT/calibration/runs/smoke-calib-machine-1-1-f005-seed1/實際attempt名稱"
cat "$CALIBRATION_RUN/status.json"
cat "$CALIBRATION_RUN/smoke_summary.json"
cat "$CALIBRATION_RUN/metadata.json"
cat "$CALIBRATION_RUN/threshold.json"
cat "$CALIBRATION_RUN/evaluation.json"
cat "$CALIBRATION_RUN/validation.json"
```

Smoke 不納入正式 manifest tasks，此時 failed checker 列出 84 個缺失是預期現象。
Checker 供日後完整實驗的失敗檢查，不會提交任何 job：

```bash
"$PYTHON_BIN" scripts/find_failed_calibration_runs.py --hpc
```

後續流程僅文件化為：smoke invariants → 使用者與 ChatGPT review → 使用者核准 → 84-task array
→ failure check → summary。`calibration_array.sbatch` 預備 0-83%8，但目前不提供立即提交完整 array
的命令；summary 也不應現在執行。

**STOP AFTER MACHINE-1-1 5% SMOKE。** 本次 Codex 則已更早停在 CODE READY FOR GIT REVIEW。

## 10. 規格對照與 backward compatibility

| 規格範圍 | Static implementation |
| --- | --- |
| 0、46–50、62：原先只實作 | 原階段遵守；後續依使用者要求執行 local synthetic tests，未執行真實 calibration 或 Git 寫入 |
| 1–4、42、58：RQ1/RQ2 分離與解讀 | 本文件動機、threshold-only 定義及 analysis report 限制 |
| 5–15、27：frozen / prefix / window / POT / invariants | calibration_runtime.py、target_calibration.py；reuse 原 helpers |
| 9、17、26、28、55：mapping / metadata / dry-run / provenance / smoke output | 明確 formal_task/run/path/hash、prefix/windows、smoke_summary |
| 16、35、43：namespace / retry / immutability | calibration_artifacts.py、exclusive outputs、獨立 attempt / summary |
| 18–19：threshold shift / ratio / evaluation / recovery | threshold_change、metric_recovery、zero_shot/calibrated artifacts |
| 20–21：canonical 84-row manifest | calibration_protocol.py、build_calibration_manifest.py |
| 22–25：獨立 code 與 CLI | calibration 核心分檔，四個指定 scripts，不塞入 main.py |
| 29：tests | 四個 test files，27 個 local synthetic tests；真實 Hazel checks pending |
| 30–34：HPC scripts | smoke / array / summary，指定 account/partition/qos/nodes |
| 36–41：完整 tables / statistics / recovery / groups / figures | calibration_analysis.py；只有 84 tasks 真實完成後才產生 |
| 44–45、51–56：文件與分階段部署 | 本文件、HPC README；目前 workflow 到 smoke review 即停止 |
| 57、59–60：scope / reuse / backward compatibility | 無新模型、無 training、無 normalization；formal shared source 未改 |
| 61：最終回覆 | 回報新增/修改檔案、static review、pending 與手動 commands |

唯一既有 tracked file 的修改是 `hpc/README_HPC.md`：更新使用者提供的進度並新增 calibration
workflow。沒有修改 main、models、dlutils、cross_machine、POT/SPOT、paths、experiment、formal
manifest、formal sbatch 或 formal analysis，因此原 formal CLI 與計算行為保持原程式內容。

Static 對應與 local synthetic 通過均不能替代真實 SMD / Hazel validation；該部分仍為 PENDING HAZEL VALIDATION。
