# shmembench-cuda

## 1. Benchmark Background
- Benchmark 類型：Shared Memory Microbenchmark
- 主要測試內容：Measures hardware shared memory read/write bandwidth using float4 vector operations.
- 硬體 / runtime 需求：1 V100 GPU
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：No

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_FOUND | 6.551755 | ms avg kernel time | PASS | Best valid raw run improved modestly; one faster P1 attempt had checksum failure and is excluded. |
| P2 | DATA_FOUND | 6.555778 | ms avg kernel time | PASS | Valid final optimization is only about 0.13% faster; checksum-failing faster attempt is rejected. |
| P3 | DATA_FOUND | 7.692324 | ms avg kernel time, 3-trial mean | PASS | Removing unneeded synchronization gives a modest 2.85% time improvement; checksum-failing block-size sweep is rejected. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 6.373006 | ms avg kernel time | 1.0280x | PASS | KERNEL_OPT | Best valid raw run improved modestly; one faster P1 attempt had checksum failure and is excluded. |
| P2 | 6.547344 | ms avg kernel time | 1.0013x | PASS | KERNEL_OPT | Valid final optimization is only about 0.13% faster; checksum-failing faster attempt is rejected. |
| P3 | 7.473037 | ms avg kernel time, 3-trial mean | 1.0293x | PASS | KERNEL_OPT | Removing unneeded synchronization gives a modest 2.85% time improvement; checksum-failing block-size sweep is rejected. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：最好的有效 run 提升微弱，但曾經有一次更快的嘗試因為 checksum 失敗而被排除，P1 險些漏掉此錯誤。
- 風險：若無 correctness 嚴格把關，容易誤用錯誤的優化版。
- 結果：valid_with_caution

### P2 Medium Prompt
- 行為：最終有效優化僅有 0.13% 的提升，但成功拒絕了 checksum 失敗的快速版本。
- 改善：防止了 checksum 錯誤的代碼被當作優化成果。
- 限制：正確把關 correctness。

### P3 Strong Prompt
- 行為：移除多餘的 synchronization 獲得 2.85% 的實質改善，同時利用 contradiction check 拒絕了 checksum 失敗的 block sweep。
- 改善：高度一致的 correctness gate，配合 stddev/CV 變異數分析。
- 是否提升可審核性：Yes，具有重複試驗統計。

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
  答：是，最佳有效加速為 1.0293x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 Volta 架構特徵調整 block/thread 數量，快取頻繁讀寫的 shared memory，並探討 occupancy 瓶頸。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
