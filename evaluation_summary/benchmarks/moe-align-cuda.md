# moe-align-cuda

## 1. Benchmark Background
- Benchmark 類型：MoE Sequence Alignment
- 主要測試內容：Sorting and prefix-sum alignment kernel for Mixture-of-Experts routing.
- 硬體 / runtime 需求：1 V100 GPU
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：No

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_MISSING / INVALID | n/a | us mean latency | PASS | Summary reports final mean but no measured baseline, so speedup cannot be audited. |
| P2 | DATA_FOUND | 19.504113 | us mean latency | PASS | Cached cumsum workspace; rejected slower variants documented. |
| P3 | DATA_FOUND | 19.366169 | us mean latency across accepted rows | PASS | Three accepted trials plus profiler notes; rejected regression excluded. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 15.948151 | us mean latency | n/a | NOT_EXPLICIT_IN_COMPARISON_CSV | PARAM_TUNE | Summary reports final mean but no measured baseline, so speedup cannot be audited. |
| P2 | 15.939143 | us mean latency | 1.2237x | PASS | PARAM_TUNE | Cached cumsum workspace; rejected slower variants documented. |
| P3 | 16.833719 | us mean latency across accepted rows | 1.1504x | PASS | PARAM_TUNE | Three accepted trials plus profiler notes; rejected regression excluded. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：AI 成功進行參數調整並記錄 final latency，但缺乏實測 baseline，且比較 CSV 缺乏明確 correctness 欄位。
- 風險：無 baseline 導致無法計算 speedup，屬於資訊缺失與正確性不明案例。
- 結果：valid_no_baseline

### P2 Medium Prompt
- 行為：優化策略為 cached cumsum workspace，並明確記錄了被拒絕的變慢版本。
- 改善：在 CSV 與 variance 上仍有欠缺。
- 限制：成功記錄 accepted/rejected attempts，排除了 regression。

### P3 Strong Prompt
- 行為：進行了三次 trial 並提供 profiler notes，排除並記錄了 1 筆 rejected regression。
- 改善：排除 regression 的邏輯有 standard CSV 與 variance 支撐。
- 是否提升可審核性：Yes，高可重現性。

## 5. Validity Assessment
- 是否有有效 baseline：Yes
- 是否有 correctness PASS：Yes (all cases PASS)
- 是否有 raw output：Yes
- 是否有 repeated trials：Yes (3 trials)
- 是否有 profiler：Yes (profiler variables/notes recorded)
- 是否存在 contradiction：No contradiction found

## 6. Interpretation
- 這是 kernel optimization、environment fix、measurement fix 還是 topology measurement？
  答：本案在 P3 被正式判定為 `PARAM_TUNE`。
- 是否可以計算 speedup？
  答：是，最佳有效加速為 1.1504x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 Volta 架構特徵調整 block/thread 數量，快取頻繁讀寫的 shared memory，並探討 occupancy 瓶頸。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
