# topk-cuda

## 1. Benchmark Background
- Benchmark 類型：Top-K Radix Selection
- 主要測試內容：Computes top-k elements along the hidden dimension using radix sort.
- 硬體 / runtime 需求：1 V100 GPU
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：No

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_FOUND | 3706.788705 | us mean over reported hidden_size/topk cases | PASS | Raw outputs report PASS and lower mean than first run, but P1 lacks accepted/rejected rationale and variance. |
| P2 | DATA_FOUND | 3718.191855 | us mean over 14 hidden_size/topk cases | PASS | Cached radix workspace removes timed cudaMalloc/cudaFree overhead; rejected block-size variants are recorded. |
| P3 | DATA_FOUND | 3702.170000 | us mean over 14 cases, 3 final trials | PASS | Hybrid workspace/block-size strategy improves mean top-k time with low trial variance; rejected block512 regression excluded. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 1238.256029 | us mean over reported hidden_size/topk cases | 2.9936x | PASS | KERNEL_OPT | Raw outputs report PASS and lower mean than first run, but P1 lacks accepted/rejected rationale and variance. |
| P2 | 3100.858841 | us mean over 14 hidden_size/topk cases | 1.1991x | PASS | KERNEL_OPT | Cached radix workspace removes timed cudaMalloc/cudaFree overhead; rejected block-size variants are recorded. |
| P3 | 3086.353000 | us mean over 14 cases, 3 final trials | 1.1995x | PASS | KERNEL_OPT | Hybrid workspace/block-size strategy improves mean top-k time with low trial variance; rejected block512 regression excluded. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：報告了 correctness PASS 與均值下降，但沒有 accepted/rejected 決策過程與 trials 數據。
- 風險：無法追溯無效嘗試的決策過程。
- 結果：valid_with_caution

### P2 Medium Prompt
- 行為：快取 radix workspace 移除了重複 allocation 的開銷 (1.199x)，排除了無效 block size。
- 改善：優化方向明確，並保留了對照組記錄。
- 限制：有效防止無效 block size 寫入。

### P3 Strong Prompt
- 行為：使用 hybrid workspace/block size 策略，在 3 次試驗中獲得穩定的 1.199x加速，排除了 regressed block size 512 的結果。
- 改善：Radix selection 優化與 CUB workspace 快取完全被 repeated trials 證明為真。
- 是否提升可審核性：Yes，極具學術審核價值。

## 5. Validity Assessment
- 是否有有效 baseline：Yes
- 是否有 correctness PASS：Yes (all cases PASS)
- 是否有 raw output：Yes
- 是否有 repeated trials：Yes (3 trials)
- 是否有 profiler：No
- 是否存在 contradiction：No contradiction found

## 6. Interpretation
- 這是 kernel optimization、environment fix、measurement fix 還是 topology measurement？
  答：本案在 P3 被正式判定為 `KERNEL_OPT`。
- 是否可以計算 speedup？
  答：是，最佳有效加速為 1.1995x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 Volta 架構特徵調整 block/thread 數量，快取頻繁讀寫的 shared memory，並探討 occupancy 瓶頸。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
