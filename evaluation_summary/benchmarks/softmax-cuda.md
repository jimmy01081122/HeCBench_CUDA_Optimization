# softmax-cuda

## 1. Benchmark Background
- Benchmark 類型：Softmax Activation Kernel
- 主要測試內容：Softmax probability distribution computation for different grid/slice sizes.
- 硬體 / runtime 需求：1 V100 GPU
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：No

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_FOUND | 1.451723 | ms avg latency for slice=784 impl=1 | PASS | Raw logs show slice=784 impl=1 improved, but P1 has no structured summary, rejected-attempt table, or variance. |
| P2 | DATA_FOUND | 1.449279 | ms slice=784 implementation 1 | PASS | Manual warp reductions plus slice=784 specialization produce a clear implementation-1 speedup; slower attempts are documented. |
| P3 | DATA_FOUND | 1.450565 | ms slice=784 implementation 1, 3-trial mean | PASS | Block-per-slice kernel improves large-slice implementation 1; final result includes 3 trials and contradiction check. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 1.405330 | ms avg latency for slice=784 impl=1 | 1.0330x | PASS | KERNEL_OPT | Raw logs show slice=784 impl=1 improved, but P1 has no structured summary, rejected-attempt table, or variance. |
| P2 | 0.774845 | ms slice=784 implementation 1 | 1.8704x | PASS | KERNEL_OPT | Manual warp reductions plus slice=784 specialization produce a clear implementation-1 speedup; slower attempts are documented. |
| P3 | 0.995243 | ms slice=784 implementation 1, 3-trial mean | 1.4575x | PASS | KERNEL_OPT | Block-per-slice kernel improves large-slice implementation 1; final result includes 3 trials and contradiction check. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：Raw logs 顯示 slice=784 效能改善，但缺乏 rejected 記錄與 variance 數據。
- 風險：資訊單一，難以證明其優化在其他維度是否有效。
- 結果：valid_with_caution

### P2 Medium Prompt
- 行為：手動 warp reduction 與 slice=784 特化帶來 1.87x 的加速，記錄了較慢的優化嘗試。
- 改善：有了基本的優化路徑對照。
- 限制：有效整理了特化策略。

### P3 Strong Prompt
- 行為：使用 block-per-slice 特化在大 slice 上獲得 1.45x 加速，提供了完整 CSV、三次 trial 與 contradiction 檢驗。
- 改善：透過 variance check 與大/小 slice 特化策略，數據極度可靠。
- 是否提升可審核性：Yes，數據與 CSV 完全對齊。

## 5. Validity Assessment
- 是否有有效 baseline：Yes
- 是否有 correctness PASS：Yes (all cases PASS)
- 是否有 raw output：Yes
- 是否有 repeated trials：Yes (3 trials)
- 是否有 profiler：Yes (profiler variables/notes recorded)
- 是否存在 contradiction：No contradiction found

## 6. Interpretation
- 這是 kernel optimization、environment fix、measurement fix 還是 topology measurement？
  答：本案在 P3 被正式判定為 `KERNEL_OPT`。
- 是否可以計算 speedup？
  答：是，最佳有效加速為 1.4575x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 Volta 架構特徵調整 block/thread 數量，快取頻繁讀寫的 shared memory，並探討 occupancy 瓶頸。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
