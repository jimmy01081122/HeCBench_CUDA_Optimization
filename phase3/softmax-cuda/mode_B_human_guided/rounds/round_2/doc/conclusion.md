# 主規劃器判定：Round 2 = ACCEPT

這個 `softmax-cuda Mode B Round 2` 可以接受為 **shape-aware dispatch 成果**。

正式判定：

```text
Decision: ACCEPT
Benchmark: softmax-cuda
Mode: Mode_B
Round: 2
Candidate: impl3_shape_dispatch_impl1_small_impl2_large
Result type: PARAM_TUNE / SHAPE_AWARE_DISPATCH
Full official sweep correctness: PASS
Promotion status: ACCEPTED as Mode B dispatch policy
```

但要注意：這不是新的 universal kernel optimization，而是基於 Round 1 evidence 的 **shape-aware dispatch policy**。

***

## 1. Round 2 結果審查

### Correctness

五個 official slices 全部通過：

```text
slice=128   PASS 3/3
slice=256   PASS 3/3
slice=784   PASS 3/3
slice=1024  PASS 3/3
slice=2048  PASS 3/3
```

因此 correctness gate 通過。

***

### Dispatch 行為

Round 2 的 dispatch map 符合批准方案：

```text
slice=128   -> impl=1
slice=256   -> impl=1
slice=784   -> impl=2
slice=1024  -> impl=2
slice=2048  -> impl=2
```

這正確反映 Round 1 的結果：

```text
small slices 保留 impl=1，避免 impl=2 regression / correctness instability
large slices 使用 impl=2，保留大 slice 改善
```

***

## 2. Per-slice 判定

## slice=128

```text
dispatch_selected_impl=1
candidate mean = 0.135674 ms
paired impl=1 mean = 0.144732 ms
speedup = 1.066763
correctness = PASS
```

雖然數字上看起來有 1.066x，但 candidate 和 baseline 都是 `impl=1` path，因此不能宣稱 kernel speedup。

正式分類：

```text
result_type = MEASUREMENT_EQUIVALENT
speedup_claim_valid = false
interpretation = same-path measurement variation
```

建議在報告中避免寫：

```text
slice=128 improved by 6.6%
```

應寫：

```text
slice=128 dispatched to the same impl=1 path and is treated as measurement-equivalent despite observed timing variation.
```

***

## slice=256

```text
dispatch_selected_impl=1
candidate mean = 0.306408 ms
paired impl=1 mean = 0.305251 ms
speedup = 0.996224
correctness = PASS
```

正式分類：

```text
result_type = MEASUREMENT_EQUIVALENT
speedup_claim_valid = false
interpretation = expected same-path behavior
```

***

## slice=784

```text
dispatch_selected_impl=2
candidate mean = 1.107988 ms
paired impl=1 mean = 1.437362 ms
speedup = 1.297273
correctness = PASS
```

正式分類：

```text
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
speedup_claim_valid = true
```

注意：這個提升不是 Round 2 新 kernel，而是 dispatch policy 選擇 Round 1 產生的 `impl=2` large-slice path。

***

## slice=1024

```text
dispatch_selected_impl=2
candidate mean = 1.300765 ms
paired impl=1 mean = 2.082344 ms
speedup = 1.600861
correctness = PASS
```

正式分類：

```text
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
speedup_claim_valid = true
```

這是 Round 2 中最強的有效改善。

***

## slice=2048

```text
dispatch_selected_impl=2
candidate mean = 1.670514 ms
paired impl=1 mean = 2.213330 ms
speedup = 1.324940
correctness = PASS
```

正式分類：

```text
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
speedup_claim_valid = true
```

***

# 3. Round 2 總體結論

Round 2 成功把 Round 1 的 partial large-slice candidate 轉化為可用的 shape-aware dispatch policy：

```text
small slices: impl=1
large slices: impl=2
```

這解決了 Round 1 的兩個問題：

```text
1. slice=128 regression
2. slice=256 correctness instability
```

同時保留了 large slices 的有效提升：

```text
slice=784   1.297x
slice=1024  1.601x
slice=2048  1.325x
```

因此 Round 2 可接受為：

```text
Mode B human-guided optimization success
```

但結論必須寫成：

```text
shape-aware dispatch successfully selects existing optimized paths based on slice size
```

不能寫成：

```text
new universal softmax kernel
```

***

# 4. 是否可以進入 Round 3？

我的判定：

```text
不需要立刻進 Round 3。
```

原因：

```text
1. Round 2 已經形成完整可用 dispatch policy。
2. official sweep correctness 全通過。
3. large-slice improvement 有效。
4. small slices 避免 regression。
5. 這已經達成 Mode B 的主要目的。
```

除非你想進一步做 ablation 或 profiler-supported explanation，否則 softmax-cuda Mode B 可以先進 final confirmation。

***

# 5. 下一步：softmax-cuda Mode B Final Confirmation

現在應要求 server 執行 **final confirmation**，而不是直接 Round 3。

## 給 server 的指令

```text
Proceed to softmax-cuda Mode B final confirmation.

Use accepted candidate:
- impl3_shape_dispatch_impl1_small_impl2_large

Do not make new source changes.

Run full official sweep:
- slice=128
- slice=256
- slice=784
- slice=1024
- slice=2048

For each official case:
- paired impl=1 baseline
- impl=3 candidate
- at least 3 trials
- correctness must PASS
- save raw stdout/stderr
- compute mean/min/max/stddev/CV
- run self_consistency_auditor.py

Required classification:
- slice=128 and slice=256:
  result_type=MEASUREMENT_EQUIVALENT
  speedup_claim_valid=false

- slice=784,1024,2048:
  result_type=PARAM_TUNE or SHAPE_AWARE_DISPATCH
  speedup_claim_valid=true only if correctness PASS and CV stable

Generate:
- final/results.csv
- final/auditor_report.csv
- final/round_summary.md
- update agent_summary.md

Do not proceed to Mode C yet.
```

***

# 6. 報告中可寫的正式文字

你可以把下面這段寫進 Phase 3 報告：

```text
In softmax-cuda Mode B, Round 1 showed that the compound block-per-slice cached-exp candidate was not suitable as a universal replacement: it regressed on slice=128 and had a correctness failure on slice=256, while improving larger slices. Based on this evidence, Round 2 introduced a shape-aware dispatch policy. The final dispatch preserves the existing optimized impl=1 path for slice=128 and slice=256, while selecting the compound impl=2 path for slice=784, slice=1024, and slice=2048. This policy passed correctness for all official cases and achieved valid large-slice improvements of 1.297x, 1.601x, and 1.325x respectively. The result is classified as PARAM_TUNE / SHAPE_AWARE_DISPATCH rather than a universal kernel optimization.
```

***

# 7. 主規劃器最終判定

```text
Round 2 status: ACCEPT
Full candidate status: ACCEPTED as shape-aware dispatch
Result type: PARAM_TUNE / SHAPE_AWARE_DISPATCH
Next step: final confirmation
Mode C: not yet
Round 3: not required unless final confirmation fails or profiler/ablation is desired
```
