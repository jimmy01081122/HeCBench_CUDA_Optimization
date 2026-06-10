# Phase 2 P1/P2/P3 測試結果簡易分析報告

本報告彙整 `/home/a/PP/phase2/p1`、`p2`、`p3` 已回傳結果。分析重點是 prompt 層級對結果完整性、正確性、統計品質與效能宣稱可信度的影響。

## 資料範圍

- 已有三層結果的 benchmark：`allreduce-cuda`、`moe-align-cuda`、`moe-cuda`、`p2p-cuda`、`pingpong-cuda`、`prefetch-cuda`、`shmembench-cuda`、`simpleMultiDevice-cuda`、`softmax-cuda`、`topk-cuda`。
- 目前摘要表中的 benchmark 均已形成 P1/P2/P3 三層結果。
- 統一摘要表：`reports/phase2_level_summary.csv`，共 30 筆。

## 層級統計

| Level | 筆數 | 可計算 speedup 平均 | 主要觀察 |
|---|---:|---:|---|
| P1 | 10 | 1.552x | 可快速找到可行修改，但 baseline/CSV/variance 常不足，審核成本高。 |
| P2 | 10 | 1.369x | 開始有 rejected/accepted 紀錄，能過濾失敗嘗試，報告可信度明顯提升。 |
| P3 | 10 | 1.131x | CSV、三次 trial、profiler/measurement notes 與 contradiction check 最完整，較少誇大 speedup。 |

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

### prefetch-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | n/a | 0.512447 | ms mean with_prefetch raw samples | n/a | PASS reported | MEASURE_FIX/PARAM_TUNE | valid_no_baseline |
| P2 | 1.682481 | 1.036367 | ms repeat=100 with_prefetch | 1.6234 | PASS all trials | PARAM_TUNE/MEASURE_FIX | valid |
| P3 | 2.145355 | 1.921765 | ms repeat=100 without_prefetch | 1.1163 | PASS 40/40 final | PARAM_TUNE | valid_with_variance_profiler |

- P1: Only one P1 result file was available; final timing can be summarized but speedup cannot be audited against measured baseline.
- P2: Separates prefetch setup from timed kernel execution; primary repeat=100 with_prefetch improves while no-prefetch also improves.
- P3: No-prefetch block-size tuning improves demand-paging path; repeat=100 with_prefetch is explicitly measurement-equivalent.

### shmembench-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | 6.551755 | 6.373006 | ms avg kernel time | 1.0280 | PASS/no checksum failed in final | KERNEL_OPT | valid_with_caution |
| P2 | 6.555778 | 6.547344 | ms avg kernel time | 1.0013 | PASS final | KERNEL_OPT | measurement_equivalent |
| P3 | 7.692324 | 7.473037 | ms avg kernel time, 3-trial mean | 1.0293 | PASS final; failed attempt rejected | KERNEL_OPT | valid_with_variance |

- P1: Best valid raw run improved modestly; one faster P1 attempt had checksum failure and is excluded.
- P2: Valid final optimization is only about 0.13% faster; checksum-failing faster attempt is rejected.
- P3: Removing unneeded synchronization gives a modest 2.85% time improvement; checksum-failing block-size sweep is rejected.

### simpleMultiDevice-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | measurement scope changed | 60.863901 | us total_us raw final | n/a | PASS reported | MEASURE_FIX/KERNEL_OPT | success_no_speedup_claim |
| P2 | 5621.596680 | 5561.413086 | us total time over 4 GPUs | 1.0108 | PASS | KERNEL_OPT | measurement_equivalent |
| P3 | 5622.520996 | 5555.228678 | us total time over 4 GPUs, 3-trial mean | 1.0121 | PASS all final trials | KERNEL_OPT | measurement_equivalent |

- P1: Raw P1 logs show a dramatic total_us drop, but H2D/D2H timing scope appears changed; no speedup claim is counted.
- P2: Block-level reduction improves kernel/D2H components, but total time remains H2D-copy-limited with about 1% speedup.
- P3: Final kernel optimization is real but total-time speedup is only about 1.2% because H2D copy dominates.

### softmax-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | 1.451723 | 1.405330 | ms avg latency for slice=784 impl=1 | 1.0330 | PASS reported | KERNEL_OPT | valid_with_caution |
| P2 | 1.449279 | 0.774845 | ms slice=784 implementation 1 | 1.8704 | PASS impl 0/1 | KERNEL_OPT | valid |
| P3 | 1.450565 | 0.995243 | ms slice=784 implementation 1, 3-trial mean | 1.4575 | PASS 42/42 final | KERNEL_OPT | valid_with_variance_profiler |

- P1: Raw logs show slice=784 impl=1 improved, but P1 has no structured summary, rejected-attempt table, or variance.
- P2: Manual warp reductions plus slice=784 specialization produce a clear implementation-1 speedup; slower attempts are documented.
- P3: Block-per-slice kernel improves large-slice implementation 1; final result includes 3 trials and contradiction check.

### topk-cuda
| Level | Baseline | Final | Unit | Speedup | Correctness | Result type | Status |
|---|---:|---:|---|---:|---|---|---|
| P1 | 3706.788705 | 1238.256029 | us mean over reported hidden_size/topk cases | 2.9936 | PASS reported | KERNEL_OPT | valid_with_caution |
| P2 | 3718.191855 | 3100.858841 | us mean over 14 hidden_size/topk cases | 1.1991 | PASS 14/14 | KERNEL_OPT | valid |
| P3 | 3702.170000 | 3086.353000 | us mean over 14 cases, 3 final trials | 1.1995 | PASS all final trials | KERNEL_OPT | valid_with_variance |

- P1: Raw outputs report PASS and lower mean than first run, but P1 lacks accepted/rejected rationale and variance.
- P2: Cached radix workspace removes timed cudaMalloc/cudaFree overhead; rejected block-size variants are recorded.
- P3: Hybrid workspace/block-size strategy improves mean top-k time with low trial variance; rejected block512 regression excluded.

## 結論

1. P1 對效能探索有幫助，但常缺少足夠的 baseline、CSV、variance 與無效嘗試紀錄；例如 `moe-align-cuda` 沒有可審核 baseline，`p2p-cuda` 只有 2 GPU/2 direction 結果，無法完整比較。
2. P2 已能把多數失敗嘗試標成 rejected，對研究報告較友善；`moe-cuda` 與 `pingpong-cuda` 都明確保留失敗/無效提交，不把它們混入最終成果。
3. P3 的優勢最明顯：標準 CSV、trial 統計、profiler 或 measurement notes、contradiction check 都讓結果更可審核；它也比較會把 `ENV_FIX`、`MEASURE_FIX`、`MEASUREMENT_EQUIVALENT` 與真正 `KERNEL_OPT` 分開。
4. 效能上，P1 有些案例看起來 speedup 最大，例如 `pingpong-cuda` NCCL 1GiB 約 2x，但因缺少 variance/CSV，可信度低於 P3。P3 不一定追求最高 speedup，而是更準確地界定結果本質。

## 後續建議

- 後續若新增 benchmark，應維持目前的三層摘要格式，並優先補上 P3 CSV、accepted/rejected attempts、variance 與 contradiction check。
- 之後每層都要求最少輸出一份統一 schema CSV；P1 可以保持弱約束，但實驗紀錄端仍應另外保存 raw log。
- 統計分析時應分開計算 `KERNEL_OPT` 與 `ENV_FIX/MEASURE_FIX`，否則環境修復會稀釋或誇大 prompt 對 kernel optimization 的影響。
- 對 P1 的高 speedup 案例進行 P3 重跑驗證，確認是否為真實加速、測量差異或語意改變。

