# prefetch-cuda

## 1. Benchmark Background
- Benchmark 類型：Unified Memory Prefetching
- 主要測試內容：Measures CUDA Unified Memory demand paging latency with and without prefetching.
- 硬體 / runtime 需求：1 V100 GPU
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：No

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_MISSING / INVALID | n/a | ms mean with_prefetch raw samples | PASS | Only one P1 result file was available; final timing can be summarized but speedup cannot be audited against measured baseline. |
| P2 | DATA_FOUND | 1.682481 | ms repeat=100 with_prefetch | PASS | Separates prefetch setup from timed kernel execution; primary repeat=100 with_prefetch improves while no-prefetch also improves. |
| P3 | DATA_FOUND | 2.145355 | ms repeat=100 without_prefetch | PASS | No-prefetch block-size tuning improves demand-paging path; repeat=100 with_prefetch is explicitly measurement-equivalent. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 0.512447 | ms mean with_prefetch raw samples | n/a | PASS | MEASURE_FIX/PARAM_TUNE | Only one P1 result file was available; final timing can be summarized but speedup cannot be audited against measured baseline. |
| P2 | 1.036367 | ms repeat=100 with_prefetch | 1.6234x | PASS | PARAM_TUNE/MEASURE_FIX | Separates prefetch setup from timed kernel execution; primary repeat=100 with_prefetch improves while no-prefetch also improves. |
| P3 | 1.921765 | ms repeat=100 without_prefetch | 1.1163x | PASS | PARAM_TUNE | No-prefetch block-size tuning improves demand-paging path; repeat=100 with_prefetch is explicitly measurement-equivalent. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：僅有一份 raw file，可以總結時間但無法與 baseline 對比計算 speedup。
- 風險：缺乏 baseline 對照，審核困難。
- 結果：valid_no_baseline

### P2 Medium Prompt
- 行為：將 prefetch 設置與測量時間分離，以 repeat=100 with_prefetch 作為主結論。
- 改善：能區分 prefetch 造成的影響。
- 限制：結構較 P1 完整。

### P3 Strong Prompt
- 行為：指出 repeat=100 with_prefetch 是等價的，但 prefetch-cuda 的 no-prefetch 特化 (block size tuning) 獲得 1.11x 加速。
- 改善：成功區分 Unified Memory 在不同 prefetch API 下的特徵。
- 是否提升可審核性：Yes，明確區分。

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
  答：是，最佳有效加速為 1.1163x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 Volta 架構特徵調整 block/thread 數量，快取頻繁讀寫的 shared memory，並探討 occupancy 瓶頸。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
