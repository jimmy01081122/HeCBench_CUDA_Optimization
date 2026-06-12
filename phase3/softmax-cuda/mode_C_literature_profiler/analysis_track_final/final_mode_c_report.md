# Phase 3 Mode C 最終整理報告
## softmax-cuda Evidence-guided Aggressive Optimization

## 0. 摘要

本報告整理 HeCBench `softmax-cuda` 在 Phase 3 Mode C 的最終結果。Mode C 的定位是 evidence-guided aggressive optimization，目標是在保留 correctness、official cases、paired baseline 與 per-slice analysis 的前提下，嘗試超越 Mode B accepted candidate `impl=3`。

Mode C 最終 accepted candidate 為：

```text
impl4_shape_specialized_large_reduce
```

Final confirmation 結果顯示，`impl=4` 在兩個 large slices 上取得可接受的 additional speedup：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

`slice=128`、`slice=256` 與 `slice=2048` 不接受為 Mode C additional speedup claim。其中 128 與 256 是 guardrail / measurement-equivalent rows；2048 的 `speedup_vs_impl3=1.008239`，低於 1% claim gate，因此分類為 measurement-equivalent。

Final label：

```text
SUCCESS_WITH_ADDITIONAL_SPEEDUP
```

Final confirmation status：

```text
CONFIRMED
```

Profiler 結論為：

```text
LIMITED_PROFILER_EVIDENCE
```

Profiler 只支持有限的 resource observation：`impl=4` 相較 `impl=3` 在 784、1024、2048 上均降低約 `0.90 KB/block` dynamic shared memory，而 registers/thread 與 waves/SM 在收集到的 launch-resource metrics 中維持不變。Profiler timing 不用於 official timing，也不支持 causality claim。

Submission 2 的 `impl5_reduction_structure_ablation` 結果為 `BLOCKED`，不得 promotion，不支持 speedup claim，也不支持 reduction-structure attribution。

## 1. Mode C 目標與比較基準

Mode C 的研究定位是：

```text
Evidence-guided aggressive optimization
```

Mode C 的主要目標：

1. 嘗試超越 Mode B accepted candidate `impl=3`。
2. 使用 profiler、ablation 與 artifact audit 支持結果解釋。
3. 維持 correctness、official cases、paired baseline 與 per-case analysis。

Mode C 的主要比較基準是：

```text
Primary comparison:
  Mode C candidate vs Mode B impl=3
```

次要比較基準是：

```text
Secondary comparison:
  Mode C candidate vs impl=1 baseline
```

因此，本報告中的 Mode C 成功判定以 `speedup_vs_impl3` 為主。`speedup_vs_impl1` 只作為 supporting context，不能取代 per-slice 的 Mode C additional speedup 判定。

Mode B 固定背景如下：

```text
Accepted candidate:
  impl3_shape_dispatch_impl1_small_impl2_large

Dispatch policy:
  slice=128  -> impl=1
  slice=256  -> impl=1
  slice=784  -> impl=2
  slice=1024 -> impl=2
  slice=2048 -> impl=2

Result type:
  PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

Mode B final accepted large-slice speedups against `impl=1`：

```text
slice=784:  1.392x
slice=1024: 1.699x
slice=2048: 1.337x
```

Mode B 的限制是：`impl=3` 不是 universal kernel optimization；`impl=2` 不是 universal replacement；128 與 256 是 measurement-equivalent；Mode B final 的 `profiler_status=NOT_RUN`；cached-exp causality 未被證明。

## 2. Mode C 執行流程總覽

Mode C 的流程可分為四個主要階段：

1. Submission 1：提出並測試 `impl4_shape_specialized_large_reduce`。
2. Analysis Track 1：使用 profiler 進行 large-slice diagnostic analysis。
3. Submission 2：執行 `impl5_reduction_structure_ablation`，用於測試 reduction-structure hypothesis。
4. Final Confirmation 與 Final Profiler Analysis：確認 `impl=4` 的 final timing claims，並整理 profiler resource evidence。

重要執行限制：

```text
official speedup 來自 final confirmation normal timing。
profiler timing 不用於 official speedup。
final candidate 只接受 impl=4。
impl=5 僅保留為 blocked ablation artifact。
```

本報告明確區分四種資訊：

1. timing-supported speedup
2. profiler-supported resource observation
3. ablation result
4. unsupported causal claim

## 3. Submission 1：impl4_shape_specialized_large_reduce

### 3.1 Candidate 定義

Submission 1 的 candidate 為：

```text
impl4_shape_specialized_large_reduce
```

它是 Mode C 的 accepted candidate，也是 final confirmation 使用的最終 candidate。

小 slice 的處理遵循 guardrail 原則：128 與 256 不作為 additional Mode C speedup claim。Mode C 的主要 claim 集中在 large-slice comparison：`impl=4` vs Mode B `impl=3`。

### 3.2 Submission 1 可接受結果

Submission 1 最終被接受的 additional speedup claims 為：

```text
slice=784:  accepted additional speedup vs impl=3
slice=1024: accepted additional speedup vs impl=3
```

Final confirmation 後，這兩個 accepted claims 被固定為：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

### 3.3 Submission 1 限制

Submission 1 不支持以下解讀：

1. `impl=4` 是 universal kernel optimization。
2. `impl=4` 改善所有 official slices。
3. 128 或 256 有 Mode C additional speedup。
4. 2048 有 Mode C additional speedup。
5. speedup_vs_impl1 可以作為 Mode C 的主要成功指標。
6. profiler 或 timing 結果能證明 dynamic shared memory、reduction structure 或 cached-exp 的 causality。

## 4. Analysis Track 1：Profiler 輔助分析

### 4.1 Profiler 目的

Profiler 的目的不是產生 official timing，而是補充 resource-level diagnostics，協助描述 `impl=3` 與 `impl=4` 在 accepted large-slice cases 上的資源差異。

Profiler 分析問題包括：

1. `impl=3` 與 `impl=4` 在 784 的 resource differences。
2. `impl=3` 與 `impl=4` 在 1024 的 resource differences。
3. `impl=3` 與 `impl=4` 在 2048 的 resource differences。
4. profiler 是否能解釋 784/1024 improve 但 2048 measurement-equivalent。
5. profiler evidence 缺少哪些指標。

### 4.2 Profiler 有限證據

Profiler 支持的有限觀察為：

```text
impl=4 相較 impl=3，在 784、1024、2048 三個 large slices 上均降低約 0.90 KB/block 的 dynamic shared memory 使用量，而 registers/thread 與 waves/SM 維持不變。
```

這是 resource observation，不是 causality proof。

### 4.3 Profiler 無法支持的結論

Profiler 缺少以下指標：

1. memory throughput
2. warp execution efficiency
3. instruction mix
4. math / special-function indicators
5. stall / scheduler breakdown

因此 profiler conclusion 必須保守標記為：

```text
LIMITED_PROFILER_EVIDENCE
```

Profiler 不能支持以下結論：

1. dynamic shared memory reduction caused speedup。
2. reduction structure caused speedup。
3. cached-exp contribution。
4. profiler timing 是 official timing。
5. profiler 已解釋為何 784/1024 improve 但 2048 measurement-equivalent。

## 5. Submission 2：impl5_reduction_structure_ablation

### 5.1 Ablation 目的

Submission 2 的 candidate 為：

```text
impl5_reduction_structure_ablation
```

其目的是作為 partial reduction-structure ablation，嘗試檢查 reduction-structure hypothesis。然而，它不是 final candidate，也不得作為 replacement。

### 5.2 結果

Submission 2 final label：

```text
BLOCKED
```

結果摘要：

```text
slice=128:
  correctness PASS
  guardrail row
  MEASUREMENT_EQUIVALENT

slice=256:
  correctness PASS
  guardrail row
  MEASUREMENT_EQUIVALENT

slice=784:
  correctness FAIL
  INVALID

slice=1024:
  correctness FAIL / PARTIAL FAIL
  INVALID

slice=2048:
  correctness PASS
  slower than impl=3 and impl=4
  REGRESSION
```

### 5.3 BLOCKED 判定

Submission 2 被判定為 `BLOCKED`，原因是 `impl=5` 沒有通過所有 official correctness requirements，且 2048 即使 correctness PASS 也比 `impl=3` 與 `impl=4` 慢。

因此：

```text
impl=5 不得 promoted。
impl=5 不支援任何 speedup claim。
impl=5 不支援 reduction-structure causality。
impl=5 只能作為 blocked ablation artifact。
```

## 6. Final Confirmation：impl=4 最終確認

### 6.1 Final 設定

Final confirmation job：

```text
Slurm job: 950691
Node: gn1224.twcc.ai
Candidate: impl4_shape_specialized_large_reduce
Profiler status for official timing rows: NOT_RUN
Final confirmation status: CONFIRMED
```

Final confirmation 使用 official timing rows，並以 `impl=4` 對 `impl=3` 的 `speedup_vs_impl3` 作為 Mode C primary metric。

### 6.2 Final Result Table

| slice | impl=1 mean | impl=3 mean | impl=4 mean | speedup_vs_impl3 | speedup_vs_impl1 | correctness | CV | result_type |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 128 | 0.137974 | 0.136672 | 0.136548 | 1.000906 | 1.010441 | PASS | 0.005285 | MEASUREMENT_EQUIVALENT |
| 256 | 0.305982 | 0.306187 | 0.305269 | 1.003007 | 1.002335 | PASS | 0.002846 | MEASUREMENT_EQUIVALENT |
| 784 | 1.448263 | 1.031688 | 0.908544 | 1.135540 | 1.594048 | PASS | 0.000143 | MODE_C_CANDIDATE |
| 1024 | 2.107634 | 1.240982 | 1.183307 | 1.048740 | 1.781139 | PASS | 0.000211 | MODE_C_CANDIDATE |
| 2048 | 2.238517 | 1.674963 | 1.661275 | 1.008239 | 1.347470 | PASS | 0.000069 | MEASUREMENT_EQUIVALENT |

### 6.3 Accepted Claims

Accepted additional speedup claims：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

這兩個 claims 均來自 final confirmation normal timing，不來自 profiler timing。

### 6.4 Rejected Claims

Not accepted as additional Mode C speedup：

```text
slice=128
slice=256
slice=2048
```

128 與 256 是 guardrail / measurement-equivalent rows。2048 的 `speedup_vs_impl3=1.008239`，低於 1% additional-speedup claim gate，因此不接受為 Mode C additional speedup。

## 7. Final Profiler Analysis

### 7.1 Profiler 設定

Final profiler job：

```text
Slurm job: 950695
Node: gn1225.twcc.ai
```

Profiler execution safeguards：

```text
repeat_for_profiler=10
launch_skip=2
launch_count=1
kernel filters:
  softMax3 for impl=3
  softMax4 for impl=4

profiled slices:
  784
  1024
  2048

not profiled:
  128
  256

official_timing_used=false
```

### 7.2 Resource Table

| slice | impl | profiler_status | registers/thread | dynamic shared memory | static shared memory | waves/SM | profiler timing |
|---:|---:|---|---:|---:|---:|---:|---|
| 784 | 3 | AVAILABLE | 18 | 4.16 KB/block | 0 | 156.25 | 1.10 ms |
| 784 | 4 | AVAILABLE | 18 | 3.26 KB/block | 0 | 156.25 | 938.11 us |
| 1024 | 3 | AVAILABLE | 18 | 5.12 KB/block | 0 | 156.25 | 1.30 ms |
| 1024 | 4 | AVAILABLE | 18 | 4.22 KB/block | 0 | 156.25 | 1.20 ms |
| 2048 | 3 | AVAILABLE | 18 | 9.22 KB/block | 0 | 78.12 | 1.66 ms |
| 2048 | 4 | AVAILABLE | 18 | 8.32 KB/block | 0 | 78.12 | 1.64 ms |

Profiler timing in this table is diagnostic only and is not official timing.

### 7.3 Limited Profiler Evidence

Final profiler analysis supports the following limited observation:

```text
impl=4 reduces dynamic shared memory by approximately 0.90 KB/block relative to impl=3 for slices 784, 1024, and 2048.
```

Collected registers/thread remain `18` for both implementations. Collected waves/SM remain the same within each profiled slice:

```text
slice=784:  156.25 -> 156.25
slice=1024: 156.25 -> 156.25
slice=2048: 78.12  -> 78.12
```

This evidence is useful as resource context, but it does not identify the mechanism that produced the final timing differences.

### 7.4 Missing Profiler Evidence

The final profiler analysis does not include:

1. memory throughput
2. warp execution efficiency
3. instruction mix
4. math / special-function indicators
5. stall / scheduler breakdown

Because those metrics are unavailable, profiler evidence cannot explain why 784 and 1024 produce accepted additional speedups while 2048 remains measurement-equivalent.

## 8. Mode C 最終結論

### 8.1 Accepted Candidate

Mode C final accepted candidate：

```text
impl4_shape_specialized_large_reduce
```

Mode C final label：

```text
SUCCESS_WITH_ADDITIONAL_SPEEDUP
```

Final confirmation status：

```text
CONFIRMED
```

### 8.2 Accepted Additional Speedups

Accepted additional-speedup claims：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

These are timing-supported claims from final confirmation normal timing.

### 8.3 Measurement-equivalent Slices

Measurement-equivalent / no Mode C additional speedup claim：

```text
slice=128
slice=256
slice=2048
```

2048 is not accepted because:

```text
speedup_vs_impl3=1.008239 < 1.01
```

### 8.4 Blocked Ablation

Blocked ablation：

```text
impl5_reduction_structure_ablation
```

Submission 2 final label：

```text
BLOCKED
```

`impl=5` is not promotable and does not support speedup or attribution claims.

### 8.5 Causality Status

Causal attribution：

```text
NOT_PROVEN
```

The report may state that profiler observed lower dynamic shared-memory allocation for `impl=4`. It must not state that this caused the accepted speedups. Reduction-structure causality and cached-exp contribution are also not proven.

## 9. 可寫入論文或正式報告的文字

### 9.1 中文正式版

在 Phase 3 Mode C 中，本研究以 `softmax-cuda` 為對象，評估 evidence-guided aggressive optimization 是否能在 Mode B accepted candidate `impl=3` 之上取得額外效能改善。最終確認結果顯示，`impl4_shape_specialized_large_reduce` 在 `slice=784` 與 `slice=1024` 上分別達到 `1.135540x` 與 `1.048740x` 的 additional speedup relative to `impl=3`，並通過 correctness 與 artifact audit。因此，Mode C final label 判定為 `SUCCESS_WITH_ADDITIONAL_SPEEDUP`，final confirmation status 為 `CONFIRMED`。

對於 `slice=128`、`slice=256` 與 `slice=2048`，本研究不接受 Mode C additional speedup claim。其中 128 與 256 為 guardrail / measurement-equivalent rows；2048 雖 correctness PASS，但 `speedup_vs_impl3=1.008239`，低於 1% claim gate，因此分類為 measurement-equivalent。

Final profiler analysis 顯示，`impl=4` 相較 `impl=3` 在 784、1024 與 2048 上皆降低約 `0.90 KB/block` 的 dynamic shared-memory allocation，而 collected registers/thread 與 waves/SM 維持不變。由於缺少 memory throughput、warp execution efficiency、instruction mix、math/special-function 與 scheduler/stall breakdown，本研究將 profiler evidence 標記為 `LIMITED_PROFILER_EVIDENCE`。該結果只能作為 resource-level observation，不能作為 dynamic shared memory、reduction structure 或 cached-exp causality 的證明。

### 9.2 英文正式版

In Phase 3 Mode C, this study evaluated whether evidence-guided aggressive optimization could provide additional improvement over the Mode B accepted candidate `impl=3` for `softmax-cuda`. The final confirmation shows that `impl4_shape_specialized_large_reduce` achieved accepted additional speedups of `1.135540x` and `1.048740x` over `impl=3` for `slice=784` and `slice=1024`, respectively, while passing correctness and artifact-audit requirements. Therefore, the Mode C final label is `SUCCESS_WITH_ADDITIONAL_SPEEDUP`, and the final confirmation status is `CONFIRMED`.

No Mode C additional speedup claim is accepted for `slice=128`, `slice=256`, or `slice=2048`. The 128 and 256 cases are guardrail / measurement-equivalent rows. The 2048 case passes correctness, but its `speedup_vs_impl3=1.008239` is below the 1% claim gate and is therefore classified as measurement-equivalent.

The final profiler analysis shows that `impl=4` reduces dynamic shared-memory allocation by approximately `0.90 KB/block` relative to `impl=3` for slices 784, 1024, and 2048, while the collected registers/thread and waves-per-SM metrics remain unchanged. Because memory throughput, warp execution efficiency, instruction mix, math/special-function indicators, and scheduler/stall breakdowns are unavailable, the profiler evidence is labeled `LIMITED_PROFILER_EVIDENCE`. These diagnostics provide resource-level context only and do not establish causality for dynamic shared memory, reduction structure, or cached-exp behavior.

## 10. Do-Not-Claim List

The final Mode C report must not claim:

1. `impl=4` 是 universal kernel optimization。
2. `impl=4` 改善所有 official slices。
3. `slice=128` 有 Mode C additional speedup。
4. `slice=256` 有 Mode C additional speedup。
5. `slice=2048` 有 Mode C additional speedup。
6. profiler timing 是 official timing。
7. profiler 證明 dynamic shared memory reduction caused speedup。
8. profiler 證明 reduction structure caused speedup。
9. profiler 證明 cached-exp contribution。
10. `impl=5` ablation succeeded。
11. `impl=5` supports attribution。
12. `speedup_vs_impl1` 是 Mode C 主要成功指標。
13. aggregate speedup 可以取代 per-slice analysis。
14. topk-cuda 或 shmembench-cuda 已完成 Mode C optimization。

## 11. 最終建議

建議將 Mode C 結果整合進最終中文報告時，使用以下保守結論：

1. Mode C 在 `softmax-cuda` 上完成，final label 為 `SUCCESS_WITH_ADDITIONAL_SPEEDUP`。
2. Accepted candidate 為 `impl4_shape_specialized_large_reduce`。
3. Accepted additional speedup 僅限 784 與 1024，且比較基準為 Mode B `impl=3`。
4. 128、256、2048 不接受為 Mode C additional speedup。
5. Profiler evidence 為 `LIMITED_PROFILER_EVIDENCE`，只能描述 resource observation。
6. `impl=5` ablation 為 `BLOCKED`，不得用於 promotion 或 attribution。
7. Causal attribution 維持 `NOT_PROVEN`。

## 12. 最終主規劃器判定

```text
Mode C:
  COMPLETE

Final label:
  SUCCESS_WITH_ADDITIONAL_SPEEDUP

Final accepted candidate:
  impl4_shape_specialized_large_reduce

Accepted additional-speedup claims:
  slice=784: 1.135540x vs impl=3
  slice=1024: 1.048740x vs impl=3

Measurement-equivalent / no Mode C speedup claim:
  slice=128
  slice=256
  slice=2048

Profiler:
  LIMITED_PROFILER_EVIDENCE

Ablation:
  impl=5 BLOCKED

Causal attribution:
  NOT_PROVEN

Next:
  integrate Mode C into final Chinese report
```
