# p2p-cuda

## 1. Benchmark Background
- Benchmark 類型：GPU Interconnect Bandwidth Sweep
- 主要測試內容：Measures peer-to-peer CUDA copy bandwidth between multiple GPU pairs.
- 硬體 / runtime 需求：4 GPUs (V100-SXM2-32GB)
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：Yes

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_MISSING / INVALID | n/a | GB/s average over reported directed pairs | PASS | No summary.md; final file reports only 2 GPUs/2 directions, not the full 4-GPU topology matrix. |
| P2 | DATA_FOUND | 36.170000 | GB/s average | PASS | Directional sweep improves auditability; speedup below 1%. |
| P3 | DATA_FOUND | 36.165000 | GB/s average | PASS | Full directed 4-GPU topology coverage; performance change below 1%. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 5.825000 | GB/s average over reported directed pairs | n/a | PASS | MEASURE_FIX | No summary.md; final file reports only 2 GPUs/2 directions, not the full 4-GPU topology matrix. |
| P2 | 36.245000 | GB/s average | 1.0021x | PASS | MEASURE_FIX | Directional sweep improves auditability; speedup below 1%. |
| P3 | 36.245000 | GB/s average | 1.0022x | PASS | TOPOLOGY_MEASURE/MEASURE_FIX | Full directed 4-GPU topology coverage; performance change below 1%. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：沒有 summary.md，最終檔案僅包含 2 GPUs 雙向數據，缺少完整的 4-GPU 拓撲矩陣。
- 風險：結果非常片面，嚴重降低了數據完整度。
- 結果：weak_auditability

### P2 Medium Prompt
- 行為：進行了雙向 sweep，但頻寬提升低於 1%，屬於 measurement-equivalent。
- 改善：對於微小效能提升的審核有所改善，但仍缺 variance。
- 限制：提供了較為完整的數據。

### P3 Strong Prompt
- 行為：提供了完整 4-GPU 雙向 topology 矩陣，將小於 1% 的頻寬變化標記為 MEASUREMENT_EQUIVALENT。
- 改善：將 topology data 做完整 sweeping，不漏掉任何 pair 的 correctness 驗證。
- 是否提升可審核性：Yes，矩陣級數據。

## 5. Validity Assessment
- 是否有有效 baseline：Yes
- 是否有 correctness PASS：Yes (all cases PASS)
- 是否有 raw output：Yes
- 是否有 repeated trials：Yes (3 trials)
- 是否有 profiler：No
- 是否存在 contradiction：No contradiction found

## 6. Interpretation
- 這是 kernel optimization、environment fix、measurement fix 還是 topology measurement？
  答：本案被判定為 `TOPOLOGY_MEASURE + MEASUREMENT_EQUIVALENT`。
- 是否可以計算 speedup？
  答：是，最佳有效加速為 1.0022x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 UCX_TLS 傳輸協議與 GPU topology 進行更細緻的 sweep，分離溝通與計算重疊時間。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
