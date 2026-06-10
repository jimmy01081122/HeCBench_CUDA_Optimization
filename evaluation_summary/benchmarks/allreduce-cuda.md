# allreduce-cuda

## 1. Benchmark Background
- Benchmark 類型：MPI Collective Communication
- 主要測試內容：Ring-based Allreduce algorithm implemented in CUDA for multi-GPU communication.
- 硬體 / runtime 需求：2 ranks / 2 GPUs
- 是否需要 MPI：Yes
- 是否需要 NCCL：No
- 是否需要多 GPU：Yes

## 2. Baseline Summary
| Prompt Level | Baseline Status | Metric | Unit | Correctness | Notes |
|---|---:|---:|---|---|---|
| P1 | DATA_FOUND | 141747.000000 | us/iter at largest size | PASS | P1 replaced measured allreduce path with NCCL; strong speedup on many sizes, largest-size speedup modest. |
| P2 | DATA_FOUND | 1.000000 | geomean relative latency | PASS | Geomean speedup 2.728x; largest buffer is measurement-equivalent/slightly slower (120294 -> 120423 us). |
| P3 | DATA_MISSING / INVALID | n/a | us/iter repeated mean at largest size | PASS | P3 correctly treats this as launcher/environment repair with three reproducibility runs. |

## 3. Optimization Summary
| Prompt Level | Best Valid Metric | Unit | Speedup | Correctness | Result Type | Strategy |
|---|---:|---|---:|---|---|---|
| P1 | 121825.000000 | us/iter at largest size | invalid/unverified | PASS | ENV_FIX | P1 replaced measured allreduce path with NCCL; strong speedup on many sizes, largest-size speedup modest. |
| P2 | 0.366569 | geomean relative latency | invalid/unverified | PASS | ENV_FIX | Geomean speedup 2.728x; largest buffer is measurement-equivalent/slightly slower (120294 -> 120423 us). |
| P3 | 131121.666667 | us/iter repeated mean at largest size | n/a | PASS | ENV_FIX | P3 correctly treats this as launcher/environment repair with three reproducibility runs. |

## 4. Prompt Constraint Impact
### P1 Weak Prompt
- 行為：使用了 NCCL 改善，在某些尺寸上有部分加速，但 baseline run FAIL 且缺乏 raw data，可審核性極低。
- 風險：缺乏對照 baseline 與重複實驗，且將 launcher 修復誤當作程式加速，有偽加速風險。
- 結果：invalid/unverified

### P2 Medium Prompt
- 行為：提供了 P2 報告，但將其歸類為 KERNEL_OPT 且計算 2.7280x，這忽視了 baseline 在非零尺寸 failed 的事實。
- 改善：在對照上雖比 P1 進步，但仍計算了 invalid baseline 的 speedup。
- 限制：開始有 rejected/accepted attempts 記錄。

### P3 Strong Prompt
- 行為：嚴格將其分類為 ENV_FIX，不宣稱任何 kernel speedup，並在 3 次試驗中證明 launcher 修復的穩定性。
- 改善：透過 contradiction check，避免將環境修復寫成程式優化。
- 是否提升可審核性：Yes，完全符合 P3 強約束要求。

## 5. Validity Assessment
- 是否有有效 baseline：No (or partial)
- 是否有 correctness PASS：Yes (all cases PASS)
- 是否有 raw output：Yes
- 是否有 repeated trials：Yes (3 trials)
- 是否有 profiler：No
- 是否存在 contradiction：Yes (see contradiction_check.csv)

## 6. Interpretation
- 這是 kernel optimization、environment fix、measurement fix 還是 topology measurement？
  答：本案在 P3 被正式判定為 `ENV_FIX`。
- 是否可以計算 speedup？
  答：否，本題不適用計算 speedup（因 baseline 無效或其本質為環境/測量修復）。
- 是否可納入論文主要結果？
  答：是，但應標記為 ENV_FIX，作為 prompt 約束防止偽加速或進行環境修復的典型對照案例。

## 7. Next Step
- 後續 CUDA 優化建議：
  - 針對 UCX_TLS 傳輸協議與 GPU topology 進行更細緻的 sweep，分離溝通與計算重疊時間。
- 後續 prompt 改善建議：
  - 必須將 `result_type` 設定為 agent 輸出的強約束必填欄位，並強制執行矛盾檢查以杜絕偽加速。
