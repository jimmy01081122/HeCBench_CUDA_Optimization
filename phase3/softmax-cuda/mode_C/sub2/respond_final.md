## Profiler Summary Audit

* **verdict:** ACCEPT\_WITH\_LIMITATIONS
* **profiler evidence label:** LIMITED\_PROFILER\_EVIDENCE
* **Mode C final result:** unchanged
* **Mode C final candidate:** `impl4_shape_specialized_large_reduce`
* **official speedup claims:** unchanged, only `slice=784` and `slice=1024`
* **causal attribution:** NOT\_SUPPORTED

這次 profiler summary 可以作為 **補充性 resource observation**，但不能升級成 profiler-supported bottleneck conclusion。

***

## Accepted Observations

可接受的觀察只有以下幾點：

1. Profiler run 是 analysis-only。
2. `official_timing_used=false`，沒有把 profiler timing 當成 official timing。
3. 只 profile large slices：
   * 784
   * 1024
   * 2048
4. 只比較：
   * `impl=3`
   * `impl=4`
5. kernel filtering 指向：
   * `softMax3`
   * `softMax4`
6. 對三個 large slices，`impl=4` 相對 `impl=3`：
   * registers/thread 不變：18
   * waves/SM 不變
   * dynamic shared memory 每 block 減少約 `0.90 Kbyte/block`

***

## Resource Difference Table

| slice | impl=3 dynamic shared memory | impl=4 dynamic shared memory |     difference | registers/thread |         waves/SM | official speedup\_vs\_impl3 |
| ----: | ---------------------------: | ---------------------------: | -------------: | ---------------: | ---------------: | --------------------------: |
|   784 |                4.16 KB/block |                3.26 KB/block | -0.90 KB/block |         18 -> 18 | 156.25 -> 156.25 |                    1.135540 |
|  1024 |                5.12 KB/block |                4.22 KB/block | -0.90 KB/block |         18 -> 18 | 156.25 -> 156.25 |                    1.048740 |
|  2048 |                9.22 KB/block |                8.32 KB/block | -0.90 KB/block |         18 -> 18 |   78.12 -> 78.12 |                    1.008239 |

***

## Important Interpretation

這份 profiler evidence 支持的安全說法是：

> `impl=4` 在 profiled large slices 上比 `impl=3` 使用較少 dynamic shared memory per block，而 registers/thread 與 waves/SM 在相同 slice 內沒有變化。

但它**不能**支持以下因果結論：

> `impl=4` 的 speedup 是由 dynamic shared memory 減少造成。

原因很直接：

* `slice=784` 與 `slice=1024` 有有效 Mode C additional speedup。
* `slice=2048` 也有相同的 dynamic shared memory 減少量，但 official `speedup_vs_impl3=1.008239`，低於 1% 門檻，分類為 `MEASUREMENT_EQUIVALENT`。
* 因此，dynamic shared memory 減少本身不足以解釋全部 performance behavior。

這反而強化一個保守結論：

> dynamic shared memory reduction may be associated with the implementation difference, but it is not sufficient evidence for causal attribution.

***

## Missing Evidence

以下 profiler 指標仍缺失，因此不能做完整 bottleneck analysis：

* memory throughput
* warp execution efficiency
* instruction mix
* math / special-function summary
* scheduler / stall breakdown
* DRAM throughput
* SM throughput
* achieved occupancy 的直接指標

因此不能寫：

```text
Profiler shows memory bandwidth bottleneck.
Profiler shows warp efficiency improved.
Profiler shows reduction overhead was reduced.
Profiler shows expf or special-function pressure changed.
Profiler proves why 2048 did not improve.
```

***

## Profiler Timing Interpretation

表中的 profiler timing 只能作診斷參考：

| slice | impl=3 profiler timing | impl=4 profiler timing |
| ----: | ---------------------: | ---------------------: |
|   784 |                1.10 ms |              938.11 us |
|  1024 |                1.30 ms |                1.20 ms |
|  2048 |                1.66 ms |                1.64 ms |

這些 timing 方向上與 official timing 大致一致，但仍然不能作為 official speedup evidence，因為 profiler instrumentation 會改變執行環境與時間。

正式 speedup 仍只能引用 final confirmation 的 official timing：

| slice | official speedup\_vs\_impl3 | accepted? |
| ----: | --------------------------: | --------- |
|   784 |                    1.135540 | yes       |
|  1024 |                    1.048740 | yes       |
|  2048 |                    1.008239 | no        |

***

## Paper-Safe Interpretation

可放入報告：

```text
A final analysis-only Nsight Compute profiling run was performed after Mode C final confirmation. The profiler run covered only the large-slice cases 784, 1024, and 2048, comparing the Mode B `impl=3` path and the Mode C `impl=4` path. Profiler timing was not used for official speedup claims. The recovered launch/resource metrics show that `impl=4` uses approximately 0.90 KB less dynamic shared memory per block than `impl=3` for all three profiled large slices, while registers per thread and waves per SM remain unchanged. However, memory throughput, warp execution efficiency, instruction mix, special-function activity, and stall breakdown metrics were unavailable. Therefore, the profiler evidence supports only a limited resource-usage observation and does not establish a causal explanation for the observed speedups. In particular, because slice 2048 shows the same reduction in dynamic shared memory but does not pass the 1% additional-speedup threshold, dynamic shared-memory reduction alone cannot be claimed as the cause of the Mode C improvement.
```

中文版本：

```text
Mode C final confirmation 後，本研究進行了一次 analysis-only Nsight Compute profiling。該 profiler run 僅涵蓋 large-slice cases 784、1024 與 2048，並比較 Mode B `impl=3` path 與 Mode C `impl=4` path。Profiler timing 未用於 official speedup claim。回收的 launch/resource metrics 顯示，在三個 profiled large slices 上，`impl=4` 相較 `impl=3` 每 block 約少使用 0.90 KB dynamic shared memory，而 registers/thread 與 waves/SM 在相同 slice 內維持不變。然而，本次 profiler 未提供 memory throughput、warp execution efficiency、instruction mix、special-function activity 或 stall breakdown 等指標。因此，profiler evidence 僅能支持有限的 resource-usage observation，不能建立 `impl=4` 加速的因果解釋。特別是，slice 2048 也呈現相同的 dynamic shared-memory reduction，但其 official `speedup_vs_impl3` 未達 1% 門檻，因此不能宣稱 dynamic shared-memory reduction alone 是 Mode C improvement 的原因。
```

***

## Do-Not-Claim

不得寫入報告：

1. Profiler 證明 `impl=4` 的加速原因。
2. Profiler 證明 dynamic shared memory reduction caused speedup。
3. Profiler 證明 reduction structure caused speedup。
4. Profiler 證明 cached-exp contribution。
5. Profiler 證明 2048 為什麼沒有有效 speedup。
6. Profiler timing 是 official speedup。
7. `impl=4` 改善所有 large slices。
8. `impl=4` 是 universal kernel optimization。
9. `impl=5` ablation 成功。
10. 以 aggregate speedup 取代 per-slice 結果。

***

## Final State Update

```text
Mode C final candidate: impl4_shape_specialized_large_reduce
Mode C final label: SUCCESS_WITH_ADDITIONAL_SPEEDUP
Final confirmation status: CONFIRMED
Accepted additional speedup:
- slice=784: 1.135540x vs impl=3
- slice=1024: 1.048740x vs impl=3

Not accepted:
- slice=128
- slice=256
- slice=2048

Profiler evidence:
- LIMITED_PROFILER_EVIDENCE
- supports limited resource observation only
- does not support causal attribution

Ablation evidence:
- impl=5 BLOCKED
- no accepted attribution claim
```

## Recommendation

目前資料已足夠進入最終報告整理。  
不建議再開新的 optimization submission。若要提升因果解釋，必須重新設計 profiler metrics 或 ablation，但這已超出目前穩健收斂範圍。


---
這份 **Mode C Final Profiler Summary** 可以接受，但只能作為 **有限 profiler evidence**，不能作為完整瓶頸或因果解釋。

# 主規劃器判定

```text
Final profiler analysis: ACCEPT_WITH_LIMITATIONS
Profiler evidence level: LIMITED_PROFILER_EVIDENCE
Mode C final label: SUCCESS_WITH_ADDITIONAL_SPEEDUP
Final accepted candidate: impl4_shape_specialized_large_reduce
Further optimization: not required
```

目前 Mode C 可以收斂，不建議再追加 profiler 或新的 optimization submission。

***

# 1. Profiler 結果可以支持什麼

這次 profiler 可以支持一個很窄的 resource observation：

```text
impl=4 相較 impl=3，在 784、1024、2048 三個 large slices 上，
dynamic shared memory per block 均減少約 0.90 KB/block。
```

整理如下：

| slice | impl=3 dynamic shmem | impl=4 dynamic shmem |             差異 | registers/thread |        waves/SM |
| ----: | -------------------: | -------------------: | -------------: | ---------------: | --------------: |
|   784 |        4.16 KB/block |        3.26 KB/block | -0.90 KB/block |          18 → 18 | 156.25 → 156.25 |
|  1024 |        5.12 KB/block |        4.22 KB/block | -0.90 KB/block |          18 → 18 | 156.25 → 156.25 |
|  2048 |        9.22 KB/block |        8.32 KB/block | -0.90 KB/block |          18 → 18 |   78.12 → 78.12 |

這可以寫成：

```text
Profiler analysis shows that impl=4 reduces dynamic shared memory usage per block relative to impl=3 while keeping registers per thread and waves per SM unchanged.
```

中文：

```text
Profiler 分析顯示，impl=4 相較 impl=3 降低了每個 block 的 dynamic shared memory 使用量，而 registers/thread 與 waves/SM 維持不變。
```

***

# 2. Profiler 結果不能支持什麼

這份 profiler 結果不能支持以下結論：

```text
1. 不能證明 impl=4 加速是因為 dynamic shared memory 減少。
2. 不能證明 reduction structure 是加速原因。
3. 不能證明 shared-memory footprint 是瓶頸。
4. 不能證明 cached-exp 是原因。
5. 不能解釋 2048 為什麼沒有達到有效 additional speedup。
6. 不能用 profiler timing 計算 speedup。
```

原因是這次缺少：

```text
memory_throughput
warp_execution_efficiency
instruction mix
math / special-function indicators
stall / scheduler breakdown
```

而且一個關鍵觀察是：

```text
2048 也有相同的 -0.90 KB/block dynamic shared memory reduction，
但 speedup_vs_impl3 只有 1.008239，仍是 measurement-equivalent。
```

這表示：

```text
dynamic shared memory 減少本身不足以解釋為何 784/1024 有有效 speedup，而 2048 沒有。
```

因此不能做因果歸因。

***

# 3. Final confirmation 與 profiler 的正確關係

Final confirmation 的 official timing 結論是：

| slice | speedup\_vs\_impl3 | 結論                                 |
| ----: | -----------------: | ---------------------------------- |
|   128 |           1.000906 | MEASUREMENT\_EQUIVALENT            |
|   256 |           1.003007 | MEASUREMENT\_EQUIVALENT            |
|   784 |           1.135540 | accepted Mode C additional speedup |
|  1024 |           1.048740 | accepted Mode C additional speedup |
|  2048 |           1.008239 | MEASUREMENT\_EQUIVALENT            |

Profiler 只能作為補充解釋：

```text
official speedup 來自 final confirmation timing。
profiler timing 不參與 speedup 計算。
```

所以正式結論應該是：

```text
Mode C final confirmation confirms valid additional speedup on slices 784 and 1024.
Final profiler analysis provides limited resource evidence showing lower dynamic shared memory usage in impl=4, but does not establish a full causal explanation.
```

***

# 4. Mode C 最終狀態

現在可以正式收斂 Mode C：

```text
Mode C final status:
  SUCCESS_WITH_ADDITIONAL_SPEEDUP

Accepted candidate:
  impl4_shape_specialized_large_reduce

Accepted additional speedup:
  slice=784: 1.135540x vs impl=3
  slice=1024: 1.048740x vs impl=3

Not accepted as additional speedup:
  slice=128
  slice=256
  slice=2048

Profiler evidence:
  LIMITED_PROFILER_EVIDENCE

Ablation:
  impl=5 attempted
  final label: BLOCKED
  no accepted attribution claim
```

***

# 5. 報告中可直接使用的文字

## 中文正式版

```text
Mode C final confirmation 顯示，impl4_shape_specialized_large_reduce 在 slice=784 與 slice=1024 上相對 Mode B 的 impl=3 取得有效 additional speedup，分別為 1.135540x 與 1.048740x。slice=128 與 slice=256 為 small-slice guardrail cases，不宣稱 Mode C speedup；slice=2048 的 speedup_vs_impl3 為 1.008239，低於 1% 門檻，因此分類為 MEASUREMENT_EQUIVALENT。

後續 final profiler analysis 僅作為解釋用途，不用於 official timing。Profiler 顯示 impl=4 相較 impl=3 在 784、1024、2048 三個 large slices 上均降低約 0.90 KB/block 的 dynamic shared memory 使用量，而 registers/thread 與 waves/SM 維持不變。然而，本次 profiler 未提供 memory throughput、warp execution efficiency、instruction mix、math/special-function indicators 或 stall/scheduler breakdown 等指標，因此只能支持有限的 resource-usage observation，不能證明 dynamic shared memory reduction、reduction structure 或 cached exponentials 是造成 speedup 的單一原因。
```

***

## 英文正式版

```text
Mode C final confirmation shows that impl4_shape_specialized_large_reduce achieves valid additional speedup over the Mode B impl=3 dispatcher on slice=784 and slice=1024, with speedup_vs_impl3 of 1.135540x and 1.048740x respectively. The small slices 128 and 256 remain guardrail cases and do not support Mode C speedup claims. Slice=2048 achieves only 1.008239x over impl=3 and is therefore classified as measurement-equivalent.

The final profiler analysis was used only for interpretation and not for official timing. The recovered profiler metrics show that impl=4 uses approximately 0.90 KB/block less dynamic shared memory than impl=3 across the profiled large-slice cases, while registers per thread and waves per SM remain unchanged. However, memory throughput, warp execution efficiency, instruction mix, math/special-function indicators, and stall/scheduler breakdowns were not available. Therefore, the profiler evidence supports only a limited resource-usage observation and does not establish a causal explanation for the observed speedups.
```

***

# 6. Do-Not-Claim List 更新

Mode C final report 必須加入：

```text
Do not claim:
1. impl=4 is a universal kernel optimization.
2. impl=4 improves all official slices.
3. slice=128 has Mode C additional speedup.
4. slice=256 has Mode C additional speedup.
5. slice=2048 has Mode C additional speedup.
6. profiler timing is official timing.
7. profiler proves dynamic shared memory reduction caused speedup.
8. profiler proves reduction structure caused speedup.
9. profiler proves cached-exp contribution.
10. impl=5 ablation succeeded.
11. impl=5 supports attribution.
```

***

# 7. 建議下一步

現在不要再做新的 optimization 或 profiler。  
下一步應該是整理最終 Mode C 報告。

要求 CLI 更新或產出：

```text
mode_C_literature_profiler/final/mode_C_summary.md
mode_C_literature_profiler/final/final_results.csv
mode_C_literature_profiler/final/profiler_summary.md
mode_C_literature_profiler/final/limitations.md
mode_C_literature_profiler/final/do_not_claim.md
```

其中 `mode_C_summary.md` 應包含：

```text
1. Mode B reference
2. Submission 1: impl=4 additional speedup
3. Analysis Track 1: limited profiler evidence
4. Submission 2: impl=5 blocked ablation
5. Final confirmation: impl=4 confirmed
6. Final profiler analysis: limited resource observation
7. Final accepted claims
8. Rejected claims
9. Limitations
10. Final label
```

***

# 8. 最終主規劃器結論

```text
Mode C is complete.

Final label:
  SUCCESS_WITH_ADDITIONAL_SPEEDUP

Final accepted candidate:
  impl4_shape_specialized_large_reduce

Accepted additional-speedup claims:
  slice=784: 1.135540x vs impl=3
  slice=1024: 1.048740x vs impl=3

Profiler conclusion:
  LIMITED_PROFILER_EVIDENCE

Ablation conclusion:
  impl=5 BLOCKED; no valid attribution evidence

Next:
  Write final Mode C report and integrate into the Chinese full report.
```
