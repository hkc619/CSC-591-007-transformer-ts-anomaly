# Deployment 協作規則

- 所有說明與新增文件使用繁體中文。
- 此 checkout 是 CPU float64 / TranAD / protocol-v2 deployment，不能擅改科學設定。
- 正式 matrix 需使用者明確啟動；不得自動 push、SSH 或 submit Slurm jobs。
- DATA_ROOT 與 RESULTS_ROOT 位於 code checkout 外，禁止將 dataset、結果或 checkpoint 加入 Git。
- 修改後先提供 dry-run、regression check 與手動執行方式。
- 記錄所有 I/O 適配；不可將 local smoke test 當作 Hazel 已驗證或 paper reproduction。

## Secondary robustness extension

- 既有 TranAD / calibration 維持 CPU float64；獨立 Anomaly Transformer extension 依使用者固定 protocol 使用 float32 / CUDA。
- AT 只評估預先指定六台的 per-machine vs 27-source cross-machine ranking metrics，不做 threshold、PA 或 calibration。
- AT local synthetic tests / dry-runs 可執行；Hazel 先只準備 machine-1-1 tasks 0/1 的完整 10-epoch smoke，由使用者手動提交。
- Smoke review 前不可提交其餘 tasks 2–11，不可擴到 28 targets。操作文件：ANOMALY_TRANSFORMER_ROBUSTNESS.md。
