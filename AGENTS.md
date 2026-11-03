# Deployment 協作規則

- 所有說明與新增文件使用繁體中文。
- 此 checkout 是 CPU float64 / TranAD / protocol-v2 deployment，不能擅改科學設定。
- 正式 matrix 需使用者明確啟動；不得自動 push、SSH 或 submit Slurm jobs。
- DATA_ROOT 與 RESULTS_ROOT 位於 code checkout 外，禁止將 dataset、結果或 checkpoint 加入 Git。
- 修改後先提供 dry-run、regression check 與手動執行方式。
- 記錄所有 I/O 適配；不可將 local smoke test 當作 Hazel 已驗證或 paper reproduction。
