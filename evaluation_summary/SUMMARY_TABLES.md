# HeCBench AI Code Optimization Summary Tables

本檔案彙整了 HeCBench 10 個標準化測試在不同 AI 輔助與 Prompt 約束層級下的統計表格，提供論文數據支持。

> [!NOTE]
> 本表中使用之加速比（Speedup）與分類使用 Phase 2 正規化 P3 結果（或顯式指定的 Comparison Basis），不完全等同於早期 BASIC 實驗中 AI 產出的最高單點加速比（如 `softmax-cuda` 的 BASIC/GM slice=784 曾達 59.593x）。

## 1. Benchmark Overview
此表顯示 10 個測試案例的特徵、最佳結果分類、最佳有效加速比與正確性狀態。

| Benchmark | Category | Required GPUs | Requires MPI | Requires NCCL | Best Result Type | Best Speedup | Correctness |
|---|---|---|---|---|---|---|---|
| allreduce-cuda | MPI Collective Communication | 2 | Yes | No | ENV_FIX | n/a | PASS after tuned UCX launcher |
| moe-align-cuda | MoE Sequence Alignment | 1 | No | No | PARAM_TUNE | 1.1504x | NOT_EXPLICIT_IN_COMPARISON_CSV |
| moe-cuda | MoE Gate & Dispatch | 1 | No | No | KERNEL_OPT | 1.0778x | PASS |
| p2p-cuda | GPU Interconnect Bandwidth Sweep | 2-4 | No | No | TOPOLOGY_MEASURE + MEASUREMENT_EQUIVALENT | 1.0022x | PASS |
| pingpong-cuda | Point-to-Point Communication | 2 | Yes | Yes | TRANSPORT_COMPARISON / MEASURE_FIX | 1.059x MPI vs NCCL | MPI PASS; NCCL PASS |
| prefetch-cuda | Unified Memory Prefetching | 1 | No | No | PARAM_TUNE | 1.1163x | PASS |
| shmembench-cuda | Shared Memory Microbenchmark | 1 | No | No | KERNEL_OPT / PARAM_TUNE | 1.0293x | PASS |
| simpleMultiDevice-cuda | Multi-GPU Reduction Scaling | 1/2/4 | No | No | MULTI_GPU_SCALING | 2.232x 4GPU vs 1GPU | PASS |
| softmax-cuda | Softmax Activation Kernel | 1 | No | No | KERNEL_OPT | 1.4575x | PASS |
| topk-cuda | Top-K Radix Selection | 1 | No | No | KERNEL_OPT | 1.1995x | PASS |

## 2. Prompt Level Comparison
此表對比 P1（弱約束）、P2（中約束）、P3（強約束）下，AI agent 報告的運行狀態 (Status) 與加速比 (Speedup)。

| Benchmark | P1 Status | P2 Status | P3 Status | P1 Speedup | P2 Speedup | P3 Speedup | Auditability Winner |
|---|---|---|---|---|---|---|---|
| allreduce-cuda | valid | valid | success_no_speedup_claim | invalid/unverified | invalid/unverified | n/a | P3 (強約束無矛盾) |
| moe-align-cuda | valid_no_baseline | valid | valid_with_variance_profiler | n/a | 1.2237 | 1.1504 | P3 (強約束無矛盾) |
| moe-cuda | valid | valid | valid_with_variance_profiler | 1.0927 | 1.0339 | 1.0778 | P3 (強約束無矛盾) |
| p2p-cuda | weak_auditability | measurement_equivalent | measurement_equivalent | n/a | 1.0021 | 1.0022 | P3 (強約束無矛盾) |
| pingpong-cuda | valid_with_caution | measurement_equivalent | success_no_speedup_claim | 1.9990 | 1.0000 | n/a | P3 (強約束無矛盾) |
| prefetch-cuda | valid_no_baseline | valid | valid_with_variance_profiler | n/a | 1.6234 | 1.1163 | P3 (強約束無矛盾) |
| shmembench-cuda | valid_with_caution | measurement_equivalent | valid_with_variance | 1.0280 | 1.0013 | 1.0293 | P3 (強約束無矛盾) |
| simpleMultiDevice-cuda | success_no_speedup_claim | measurement_equivalent | measurement_equivalent | n/a | 1.0108 | 1.0121 | P3 (強約束無矛盾) |
| softmax-cuda | valid_with_caution | valid | valid_with_variance_profiler | 1.0330 | 1.8704 | 1.4575 | P3 (強約束無矛盾) |
| topk-cuda | valid_with_caution | valid | valid_with_variance | 2.9936 | 1.1991 | 1.1995 | P3 (強約束無矛盾) |

## 3. Result Type Distribution
此表展示結果在各分類下的分佈、對應 Benchmark 與學術解讀。

| Result Type | Count | Benchmarks | Interpretation |
|---|---:|---|---|
| ENV_FIX | 1 | `allreduce-cuda` | 修復或調優 launcher、MPI、UCX 傳輸層等環境配置 |
| KERNEL_OPT | 3 | `moe-cuda`, `softmax-cuda`, `topk-cuda` | 修改 CUDA kernel 或算法，在保證正確性下提升性能 |
| KERNEL_OPT / PARAM_TUNE | 1 | `shmembench-cuda` | CUDA 核函數優化與共享記憶體參數配置調整 |
| MULTI_GPU_SCALING | 1 | `simpleMultiDevice-cuda` | 多 GPU 劃分歸約，擴展性受 PCIe (H2D) 傳輸主導限制 |
| PARAM_TUNE | 2 | `moe-align-cuda`, `prefetch-cuda` | 調整 block/grid size 或快取配置等超參數 |
| TOPOLOGY_MEASURE + MEASUREMENT_EQUIVALENT | 1 | `p2p-cuda` | 拓撲頻寬掃描，且效能提升小於 1%（測量等價） |
| TRANSPORT_COMPARISON / MEASURE_FIX | 1 | `pingpong-cuda` | 不同傳輸協議（MPI vs NCCL）的性能對比與測量修正 |

## 4. Invalid Results
此表列出所有被判定為無效的實驗數據（例如 correctness FAIL、缺 raw data、 estimated baseline 等）。

| Benchmark | Prompt Level | Reason |
|---|---|---|
| moe-align-cuda | P1 | baseline missing or invalid |
| p2p-cuda | P1 | baseline missing or invalid |
| prefetch-cuda | P1 | baseline missing or invalid |
| simpleMultiDevice-cuda | P1 | baseline missing or invalid |
| allreduce-cuda | P3 | baseline missing or invalid |
| pingpong-cuda | P3 | baseline missing or invalid |
| moe-align | BASIC | correctness status: not shown in comparison csv |
| moe-align | BASIC | correctness status: not shown in comparison csv |
| moe-align | BASIC | correctness status: not shown in comparison csv |
| allreduce-cuda | BASIC | baseline run failed |
