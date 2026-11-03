# 本機 deployment 驗證

這是本機驗證，不代表已在 Hazel 執行。沒有 SSH、Slurm submission、Git push 或完整 matrix。

## 已通過

1. 單機 dry-run：machine-1-1，28,479 windows，CPU float64、v2、seed 1。
2. 27→1 dry-run：679,926 windows；sources、target、window count、protocol、POT mode、seed、
   dtype、device、window、lr、epochs、training/evaluation batch 全部與來源 scale-probe metadata 一致。
3. 使用外部 DATA_ROOT 實際列出全部 28 machines；train/test/features、NaN/Inf、labels 與 manifest 一致。
4. 六個 protected source files bytes 相同；核心 windowing / loss / optimizer / preprocessing /
   shuffle / POT / metrics 的 function AST 與來源相同。
5. 小型 fixture：80 train、48 test、38 features、1 epoch、batch 8。
   來源與 deployment 的 loss、訓練後全部 parameters、generator state、sample order、labels、
   windows、scores、predictions、threshold 和 metrics 完全一致。
6. 獨立 deployment CLI smoke 能完成 training → source-only threshold → testing → metrics。
7. HPC mode 缺 DATA_ROOT 時明確報錯，不會搜尋舊本機資料。
8. Python AST / shell syntax 通過，正式 source code 沒有 `/Users/`、課程本機目錄或來源 repo absolute fallback。

Fixture data、NPZ、checkpoints 和 reports 全部位於暫存目錄或外部 results root，沒有打包進 repository。
本機 source-reference regression 使用原有 validated environment；它含原 repository import 所需的
SciencePlots。Deployment 的獨立執行路徑不需要 SciencePlots。

## 尚未驗證

- 全新 Linux / Hazel Python environment 的安裝與 binary 相容性。
- Hazel Slurm 的 partition/account/queue 配置與實際 performance。
- Linux 與 Mac 的 bitwise equality。
- 全部 56 runs；只產生 config，沒有啟動它們。

下一步是使用者在 Hazel 的 compute allocation 先做 dry-run / smoke，再決定是否提交正式 array。
