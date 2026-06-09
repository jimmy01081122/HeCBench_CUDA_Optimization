# Phase 2 P1/P2/P3 測試結果簡易分析報告

本報告彙整 `/home/a/PP/phase2/p1`、`p2`、`p3` 已回傳結果。分析重點是 prompt 層級對結果完整性、正確性、統計品質與效能宣稱可信度的影響。

## 資料範圍

- 已有三層結果的 benchmark：`allreduce-cuda`、`moe-align-cuda`、`moe-cuda`、`p2p-cuda`、`pingpong-cuda`。
- 尚未看到 p1/p2/p3 結果的 benchmark：`softmax-cuda`、`topk-cuda`、`shmembench-cuda`、`simpleMultiDevice-cuda`、`prefetch-cuda`。
- 統一摘要表：`reports/phase2_level_summary.csv`，共 15 筆。

## 層級統計

| Level | 筆數 | 可計算 speedup 平均 | 主要觀察 |
|---|---:|---:|---|
| P1 | 5 | 1.418x | 可快速找到可行修改，但 baseline/CSV/variance 常不足，審核成本高。 |
| P2 | 5 | 1.398x | 開始有 rejected/accepted 紀錄，能過濾失敗嘗試，報告可信度明顯提升。 |
| P3 | 5 | 1.077x | CSV、三次 trial、profiler/measurement notes 與 contradiction check 最完整，較少誇大 speedup。 |

## Benchmark 橫向比較

### allreduce-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | 141747.000000 | 121825.000000 | us/iter at largest size | 1.1635 | PASS all sizes | KERNEL_OPT/ENV_FIX | valid |
| P2 | 1.000000 | 0.366569 | geomean relative latency | 2.7280 | PASS 12/12 | KERNEL_OPT | valid |
| P3 | n/a | 131121.666667 | us/iter repeated mean at largest size | n/a | PASS 36/36 total | ENV_FIX | success_no_speedup_claim |

- P1: P1 replaced measured allreduce path with NCCL; strong speedup on many sizes, largest-size speedup modest.
- P2: Geomean speedup 2.728x; largest buffer is measurement-equivalent/slightly slower (120294 -> 120423 us).
- P3: P3 correctly treats this as launcher/environment repair with three reproducibility runs.

### moe-align-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | n/a | 15.948151 | us mean latency | n/a | PASS 30/30 | PARAM_TUNE | valid_no_baseline |
| P2 | 19.504113 | 15.939143 | us mean latency | 1.2237 | PASS 30/30 | PARAM_TUNE | valid |
| P3 | 19.366169 | 16.833719 | us mean latency across accepted rows | 1.1504 | PASS all scored runs | PARAM_TUNE | valid_with_variance_profiler |

- P1: Summary reports final mean but no measured baseline, so speedup cannot be audited.
- P2: Cached cumsum workspace; rejected slower variants documented.
- P3: Three accepted trials plus profiler notes; rejected regression excluded.

### moe-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | 531.303436 | 486.238289 | us arithmetic mean over topk 1/2/4/8 | 1.0927 | PASS 4/4 | KERNEL_OPT | valid |
| P2 | 530.250191 | 512.859005 | us arithmetic mean over topk 1/2/4/8 | 1.0339 | PASS 4/4 final; invalid attempts rejected | KERNEL_OPT | valid |
| P3 | 525.414238 | 487.500949 | us arithmetic mean over all topk/trials | 1.0778 | PASS all official cases | KERNEL_OPT | valid_with_variance_profiler |

- P1: Fused softmax+topk; P1 summary lacks variance/profiler but reports baseline and final.
- P2: Hybrid path gives real topk=1 gain but mean speedup only 3.39%.
- P3: topk 1/2/4 improve; topk=8 explicitly classified measurement-equivalent.

### p2p-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | n/a | 5.825000 | GB/s average over reported directed pairs | n/a | PASS reported | MEASURE_FIX | weak_auditability |
| P2 | 36.170000 | 36.245000 | GB/s average | 1.0021 | PASS 12/12 final | MEASURE_FIX | measurement_equivalent |
| P3 | 36.165000 | 36.245000 | GB/s average | 1.0022 | PASS 36/36 directed final | TOPOLOGY_MEASURE/MEASURE_FIX | measurement_equivalent |

- P1: No summary.md; final file reports only 2 GPUs/2 directions, not the full 4-GPU topology matrix.
- P2: Directional sweep improves auditability; speedup below 1%.
- P3: Full directed 4-GPU topology coverage; performance change below 1%.

### pingpong-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | 22.899000 | 45.776000 | GB/s NCCL at 1GiB | 1.9990 | no correctness errors printed | KERNEL_OPT/MEASURE_FIX | valid_with_caution |
| P2 | 22.899285 | 22.899518 | GB/s NCCL at 1GiB | 1.0000 | PASS full sweep | PARAM_TUNE/MEASURE_FIX | measurement_equivalent |
| P3 | invalid baseline | 24.256129 | GB/s MPI at 1GiB | n/a | PASS MPI/NCCL full sweep after fix | MEASURE_FIX | success_no_speedup_claim |

- P1: NCCL grouping doubles reported 1GiB bandwidth; P1 lacks CSV/variance and has earlier invalid attempts.
- P2: Five optimizations tried; final improvement is noise-level but invalid setup attempts are clearly rejected.
- P3: Baseline NCCL executable missing; P3 correctly reports measurement recovery and avoids speedup claim.

## 結論

1. P1 對效能探索有幫助，但常缺少足夠的 baseline、CSV、variance 與無效嘗試紀錄；例如 `moe-align-cuda` 沒有可審核 baseline，`p2p-cuda` 只有 2 GPU/2 direction 結果，無法完整比較。
2. P2 已能把多數失敗嘗試標成 rejected，對研究報告較友善；`moe-cuda` 與 `pingpong-cuda` 都明確保留失敗/無效提交，不把它們混入最終成果。
3. P3 的優勢最明顯：標準 CSV、trial 統計、profiler 或 measurement notes、contradiction check 都讓結果更可審核；它也比較會把 `ENV_FIX`、`MEASURE_FIX`、`MEASUREMENT_EQUIVALENT` 與真正 `KERNEL_OPT` 分開。
4. 效能上，P1 有些案例看起來 speedup 最大，例如 `pingpong-cuda` NCCL 1GiB 約 2x，但因缺少 variance/CSV，可信度低於 P3。P3 不一定追求最高 speedup，而是更準確地界定結果本質。

## 後續建議

- 補齊尚未跑的 5 個 benchmark，尤其 `softmax-cuda`、`topk-cuda` 這類純 kernel optimization 案例，才能更公平評估 P1/P2/P3 對效能探索的影響。
- 之後每層都要求最少輸出一份統一 schema CSV；P1 可以保持弱約束，但實驗紀錄端仍應另外保存 raw log。
- 統計分析時應分開計算 `KERNEL_OPT` 與 `ENV_FIX/MEASURE_FIX`，否則環境修復會稀釋或誇大 prompt 對 kernel optimization 的影響。
- 對 P1 的高 speedup 案例進行 P3 重跑驗證，確認是否為真實加速、測量差異或語意改變。

