## Final Audit Update

你提供的 auditor / contradiction check 已補齊，且全部 PASS。  
因此 Mode C final confirmation 可從先前的「pending auditor evidence」升級為正式接受。

## Final Decision

```text
Mode C Final Confirmation verdict: ACCEPT
Mode C final label: SUCCESS_WITH_ADDITIONAL_SPEEDUP
Final confirmation status: CONFIRMED
Current accepted Mode C candidate: impl4_shape_specialized_large_reduce
Auditor status: PASS
Contradiction check: PASS
```

## Accepted Final Claims

可正式接受的 Mode C additional speedup claims 僅限：

| slice | speedup\_vs\_impl3 | claim status |
| ----: | -----------------: | ------------ |
|   784 |          1.135540x | ACCEPTED     |
|  1024 |          1.048740x | ACCEPTED     |

## Rejected / Not Accepted Claims

| slice | reason                                                                 |
| ----: | ---------------------------------------------------------------------- |
|   128 | small-slice guardrail；dispatch 到 unchanged `impl=1`；不宣稱 Mode C speedup |
|   256 | small-slice guardrail；dispatch 到 unchanged `impl=1`；不宣稱 Mode C speedup |
|  2048 | `speedup_vs_impl3=1.008239x < 1.01`，分類為 `MEASUREMENT_EQUIVALENT`       |

## Auditor Evidence

你貼出的 checks 顯示：

* official cases complete: PASS
* `impl=1/3/4` rows exist for every slice: PASS
* raw stdout/stderr paths exist: PASS
* correctness PASS before speedup claim: PASS
* `speedup_vs_impl3` exists for `impl=4`: PASS
* 128/256 not claimed: PASS
* `speedup_vs_impl1` not used as main Mode C metric: PASS
* `profiler_status=NOT_RUN`: PASS
* `official_timing_used=true`: PASS
* no `impl=5` promotion: PASS
* no causality overclaim: PASS
* no aggregate-only hiding: PASS

這滿足 final confirmation 的 evidence gate。

## Paper-Safe Final Interpretation

可直接放入報告：

```text
Mode C final confirmation accepted `impl4_shape_specialized_large_reduce` as the final Mode C candidate. The final confirmation compared `impl=4` against the paired `impl=1` baseline and the Mode B accepted `impl=3` baseline across all official softmax-cuda slices. All required implementation rows and raw stdout/stderr artifacts were present, correctness passed before any speedup claim, and both auditor and contradiction checks passed. The final accepted additional Mode C speedups over `impl=3` are limited to slice=784 and slice=1024, with speedup_vs_impl3 of 1.135540x and 1.048740x, respectively. Slices 128 and 256 remain guardrail rows with no Mode C speedup claim, and slice 2048 is classified as measurement-equivalent because its speedup_vs_impl3 is below the 1% threshold. The final Mode C result is therefore `SUCCESS_WITH_ADDITIONAL_SPEEDUP`, but it does not support universal kernel optimization or causal claims about reduction structure, shared-memory footprint, cached exponentials, or profiler-supported bottlenecks.
```

中文版本：

```text
Mode C final confirmation 接受 `impl4_shape_specialized_large_reduce` 作為最終 Mode C candidate。Final confirmation 在所有 official softmax-cuda slices 上，比較 `impl=4`、paired `impl=1` baseline，以及 Mode B accepted `impl=3` baseline。所有必要 implementation rows 與 raw stdout/stderr artifacts 均存在，且 correctness 在任何 speedup claim 前均已通過；auditor 與 contradiction checks 也全部 PASS。最終可接受的 additional Mode C speedup 僅限於 slice=784 與 slice=1024，其相對 `impl=3` 的 speedup_vs_impl3 分別為 1.135540x 與 1.048740x。slice=128 與 slice=256 為 guardrail rows，不宣稱 Mode C speedup；slice=2048 因 speedup_vs_impl3 低於 1% 門檻，分類為 MEASUREMENT_EQUIVALENT。因此，Mode C final result 可標為 `SUCCESS_WITH_ADDITIONAL_SPEEDUP`，但不支撐 universal kernel optimization，也不支撐 reduction structure、shared-memory footprint、cached exponentials 或 profiler-supported bottleneck 的因果宣稱。
```

## Do-Not-Claim

最終報告中仍不可寫：

1. `impl=4` 改善所有 official slices。
2. `impl=4` 是 universal kernel optimization。
3. `slice=128` 或 `slice=256` 有 Mode C speedup。
4. `slice=2048` 有 accepted Mode C speedup。
5. `speedup_vs_impl1` 是 Mode C 主指標。
6. profiler 證明 bottleneck。
7. reduction structure caused speedup。
8. shared-memory footprint caused speedup。
9. cached-exp contribution caused speedup。
10. `impl=5` 可 promotion。
11. Submission 2 ablation 成功證明 causal attribution。
12. aggregate speedup 可取代 per-slice result。

## Final Frozen State

```text
Benchmark: softmax-cuda
Mode B final candidate: impl3_shape_dispatch_impl1_small_impl2_large
Mode B result type: PARAM_TUNE / SHAPE_AWARE_DISPATCH

Mode C final candidate: impl4_shape_specialized_large_reduce
Mode C final label: SUCCESS_WITH_ADDITIONAL_SPEEDUP
Final confirmation status: CONFIRMED
Accepted additional-speedup slices: 784, 1024
No-claim slices: 128, 256, 2048
Profiler causality: NOT_SUPPORTED
Ablation causality: NOT_SUPPORTED
Current best candidate: impl=4
```

## 下一步建議

現在可以停止 softmax-cuda 實驗，進入報告整理。  
不建議再開新的 Submission 3，除非你的研究問題需要更強的 causal attribution。以目前資料，最穩健的論文主張已經成立：

```text
Mode C 在 Mode B accepted candidate 之上，進一步產生有限但有效的 per-slice additional speedup，且 final confirmation 與 auditor checks 均通過；有效改善僅限 784 與 1024，不能泛化到所有 slices，也不能作因果歸因。
```
