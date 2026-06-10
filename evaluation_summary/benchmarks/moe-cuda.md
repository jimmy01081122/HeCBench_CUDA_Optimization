# moe-cuda

## 1. Benchmark Background
- Benchmark 類型：MoE Gate & Dispatch
- 主要測試內容：Top-k expert gating and gating probability calculation kernel for MoE models.
- 硬體 / runtime 需求：1 V100 GPU
- 是否需要 MPI：No
- 是否需要 NCCL：No
- 是否需要多 GPU：No

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_FOUND | 531.303436 | us arithmetic mean over topk 1/2/4/8 | PASS | Fused softmax+topk; P1 summary lacks variance/profiler but reports baseline and final. |
| P2 | DATA_FOUND | 530.250191 | us arithmetic mean over topk 1/2/4/8 | PASS | Hybrid path gives real topk=1 gain but mean speedup only 3.39%. |
| P3 | DATA_FOUND | 525.414238 | us arithmetic mean over all topk/trials | PASS | topk 1/2/4 improve; topk=8 explicitly classified measurement-equivalent. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 486.238289 | us arithmetic mean over topk 1/2/4/8 | 1.0927x | PASS | KERNEL_OPT | Fused softmax+topk; P1 summary lacks variance/profiler but reports baseline and final. |
| P2 | 512.859005 | us arithmetic mean over topk 1/2/4/8 | 1.0339x | PASS | KERNEL_OPT | Hybrid path gives real topk=1 gain but mean speedup only 3.39%. |
| P3 | 487.500949 | us arithmetic mean over all topk/trials | 1.0778x | PASS | KERNEL_OPT | topk 1/2/4 improve; topk=8 explicitly classified measurement-equivalent. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：實現了 softmax+topk 的融合，但摘要中缺乏變異數分析與 profiler 紀錄。
- 風險：缺乏 reproducibility 與 profiler，難以確認實際硬體瓶頸。
- 結果：valid

### P2 Medium Prompt
- 行為：使用 hybrid path 取得 topk=1 的實質加速，但 mean speedup 僅為 3.39%。
- 改善：排除 correctness fail 與 timeout 的嘗試，流程更清晰。
- 限制：有了初步的 workflow 對照。

### P3 Strong Prompt
- 行為：提供了 topk 1/2/4/8 分別的統計，指出 topk=8 為等價且不進行融合，避免 regression。
- 改善：細緻到 per-case 的 evaluation，並附帶 profiler 資訊。
- 是否提升可審核性：Yes，非常詳盡。

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
  答：是，最佳有效加速為 1.0778x。
- 是否可納入論文主要結果？
  答：是，P3 優化結果與審核資料完整且無矛盾，可直接作為論文中 AI 程式優化成果的佐證。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 Volta 架構特徵調整 block/thread 數量，快取頻繁讀寫的 shared memory，並探討 occupancy 瓶頸。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
