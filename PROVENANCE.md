# Scientific provenance

- Source repository path：`/Users/hkc619/Local File/School/2026 Fall/CSC 591 007/TranAD`
- Source git commit（base）：`46605a735909956cb27f35a912c3a2b764c54399`
- Source branch：`research/protocol-v2-smd28`
- Source dirty status：**dirty，尚未 commit v2 正式程式**。
- Protocol version：v2。
- Creation date（UTC）：`2026-11-03T02:34:21.971383+00:00`。

此 commit 只是來源 base，**不能單靠它還原實際採用的 working tree**。本部署使用當時 dirty
working tree 的明確檔案內容，SHA256 列於 `configs/source_provenance.json`。沒有替來源 commit。

## 來源 git status（tracked）

```text
 M AGENTS.md
 M MANUAL_TRAINING.md
 M main.py
 M plots/TranAD_SMD/output.pdf
 M plots/TranAD_SMD/training-graph.pdf
 M scripts/train.py
 M src/cross_machine.py
 M src/experiment.py
 M src/parser.py
```

## 來源 git diff --stat

```text
 AGENTS.md                           |   5 +++++
 MANUAL_TRAINING.md                  |  37 ++++++++++++++++++++++++++++++++----
 main.py                             |  14 +++++++++++++-
 plots/TranAD_SMD/output.pdf         | Bin 15303037 -> 16477922 bytes
 plots/TranAD_SMD/training-graph.pdf | Bin 16465 -> 15391 bytes
 scripts/train.py                    |   5 +++--
 src/cross_machine.py                |  19 ++++++++++++++++--
 src/experiment.py                   |  11 ++++++++---
 src/parser.py                       |   1 +
 9 files changed, 80 insertions(+), 12 deletions(-)
```

未追蹤的正式 code 包含：

- `src/training_protocol.py`
- `scripts/validate_protocol_v2.py`
- `scripts/validate_v2_artifacts.py`
- `scripts/formal_experiments.py`
- `scripts/analyze_formal_experiments.py`

歷史 artifacts 沒有搬移、刪除或複製。目標既有 `.git`、remote 與 `papers/` 是使用者原有內容；
papers 保留在本機但不納入 deployment Git payload。README 原本為空，已填入部署說明。

## Adaptation 與研究語意

- models.py、dlutils.py、pot.py、spot.py、constants.py、training_protocol.py 保留 source bytes。
- main.py 只收錄原 convert_to_windows、save_model、load_model、backprop 函式，保持 function AST / 原始計算。
- preprocess.py 保留兩個 SMD conversion 函式原文，不新增 normalization 或更改 label slicing。
- cross_machine.py 只改 I/O path 與 per-run cache，windowing / training / threshold / metrics 計算順序保留。
- experiment.py 重寫外部 run bookkeeping；additional_metrics 函式保留原文。
- 移除未使用的歷史 baseline runner、plotting 入口與 utils 中繪圖 helper，避免 import 時建立 repo-local plots。
- 支援 `PROJECT_ROOT`、`DATA_ROOT`、`RESULTS_ROOT`，HPC 模式需明確設定；不存在原電腦絕對路徑 fallback。
- Config 56 rows 由實際已列出的 28-machine 名單生成，不手動猜名稱。
- Smoke-test 是明確標示的非正式短測試，不是改動正式 protocol。

Local regression 比較 loss、trained parameters、generator state、順序、labels、windows、scores、
predictions 和 metrics；結果見 LOCAL_VALIDATION.md。尚未建立乾淨 Linux environment、連線 Hazel
或執行完整 matrix，不宣稱 Mac / Linux 位元級一致。

- Formal figures 函式取自來源 analysis script，只將輸出位置改為外部 summaries/figures；來源 hash 亦有記錄。
