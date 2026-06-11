# 主規劃器判定：Round 1 = PARTIAL\_SUCCESS，但不可 promoted 為 full optimization

你這個 Round 1 結果**有研究價值**，但不能作為 full official sweep optimization。正式判定如下：

```text
Decision: PARTIAL_SUCCESS
Full optimization claim: REJECTED
Candidate status: keep as partial large-slice improvement candidate
Next action: do not promote; plan Round 2 ablation/fix
```

原因：

```text
1. slice=256 有 1/3 correctness FAIL。
2. slice=128 明顯 regression。
3. slice=784/1024/2048 有有效改善。
4. auditor PASS，表示報告分類沒有矛盾。
5. 結果已正確保存 FAIL 與 regression，這是可接受的研究輸出。
```

***

# 1. Per-slice 判定

## slice=128

```text
impl=1: 0.135152 ms
impl=2: 0.554750 ms
result_type: REGRESSION
speedup_claim_valid: false
```

判定：

```text
嚴重退化。block-per-slice + shared-memory cached exp 對小 slice overhead 過高。
```

這符合原先預期：small slice 可能被 block-level synchronization / shared memory overhead 拖慢。

***

## slice=256

```text
impl=1: 0.323384 ms
impl=2: 0.594147 ms
correctness: PASS 2/3, FAIL 1/3
result_type: INVALID
speedup_claim_valid: false
```

判定：

```text
此 case 不可用於任何 speedup claim。
```

這是 blocking issue。只要 official slice 有 correctness failure，candidate 不得 promoted 為 full valid optimization。

***

## slice=784

```text
impl=1: 1.434026 ms
impl=2: 1.108087 ms
result_type: KERNEL_OPT
speedup: 約 1.29x
```

判定：

```text
有效 large-slice improvement。
```

***

## slice=1024

```text
impl=1: 2.068956 ms
impl=2: 1.300902 ms
result_type: KERNEL_OPT
speedup: 約 1.59x
```

判定：

```text
目前最強改善點。
```

***

## slice=2048

```text
impl=1: 2.212359 ms
impl=2: 1.680560 ms
result_type: KERNEL_OPT
speedup: 約 1.30x
```

判定：

```text
有效 large-slice improvement。
```

***

# 2. Round 1 總結

Round 1 證明：

```text
block-per-slice + shared-memory cached exp compound candidate 對中大型 slice 有明顯潛力。
```

但同時也證明：

```text
該 candidate 不適合作為全 shape 通用替代方案。
```

具體結論：

```text
1. impl=2 不適合 slice=128。
2. impl=2 在 slice=256 有 correctness instability，必須修正或排除。
3. impl=2 對 slice=784/1024/2048 有明確提升。
4. 這支持後續建立 shape-aware dispatch，而不是單一 impl 取代 impl=1。
```

***

# 3. 是否接受 Round 1 修改？

## 不接受為 final

```text
Reject as full replacement.
```

理由：

```text
slice=256 correctness FAIL
slice=128 regression
```

## 接受為 partial candidate

```text
Accept as partial large-slice candidate for further investigation.
```

用途：

```text
作為 Round 2 shape-aware dispatch 或 ablation 的基礎。
```

***

# 4. 下一步不能做什麼

```text
不得把 Round 1 寫成 softmax-cuda full SUCCESS。
不得計算全 sweep average speedup 來掩蓋 slice=128 regression 和 slice=256 FAIL。
不得將改善單獨歸因於 cached exp。
不得進入 Mode C。
不得跳過 slice=128/256。
```

***

# 5. Round 2 建議方向

Round 2 應該不是繼續盲目優化，而是解釋 Round 1 的結果並建立可用 dispatch。

我建議 Round 2 目標如下：

```text
建立 shape-aware dispatch:
- slice=128 使用 impl=1
- slice=256 暫時使用 impl=1，除非修正 correctness
- slice=784/1024/2048 使用 impl=2
```

這是最直接、最符合結果的下一步。

***

# 6. Round 2 Proposal 要求

請回傳給 server 執行端以下指令。

```text
Prepare Mode B Round 2 proposal for softmax-cuda.

Do not submit sbatch yet.

Round 1 decision:
- impl2_block_cached_exp_compound is rejected as full replacement.
- It is accepted only as a partial large-slice candidate.
- slice=128 regressed.
- slice=256 had correctness failure.
- slice=784/1024/2048 improved.

Round 2 goal:
Create a shape-aware dispatch candidate.

Required dispatch policy:
- slice=128 -> impl=1
- slice=256 -> impl=1 unless a specific correctness fix is proposed and approved
- slice=784 -> impl=2
- slice=1024 -> impl=2
- slice=2048 -> impl=2

Required proposal contents:
1. Explain why this dispatch follows Round 1 evidence.
2. Confirm impl=1 and impl=2 kernels remain unchanged unless explicitly proposing a small fix.
3. Confirm no correctness tolerance, CPU reference, input generation, official cases, numSlice, or repeat changes.
4. Define candidate label:
   variant=impl3_shape_dispatch_impl1_small_impl2_large
5. Validation plan:
   - paired impl=1 baseline
   - candidate dispatch
   - all official slices 128/256/784/1024/2048
   - at least 3 trials
   - raw stdout/stderr for every trial
   - same CSV schema
   - auditor required
6. Classification rules:
   - if all official slices PASS and large-slice improvements remain, candidate may be KERNEL_OPT or PARAM_TUNE depending on implementation
   - if any official slice FAIL, candidate is not full success
   - if 128/256 are measurement-equivalent because they dispatch to impl=1, that is acceptable
7. Stop after proposal and wait for human approval.
```

***

# 7. 為什麼 Round 2 不是修 slice=256 correctness 先？

可以修，但不建議優先。

原因：

```text
1. slice=256 已經較小，impl=1 只有 0.323 ms，baseline 很快。
2. impl=2 對 slice=256 不只 correctness unstable，效能也比 impl=1 慢。
3. 修 slice=256 的收益低，風險高。
4. shape-aware dispatch 更符合 Round 1 evidence。
```

因此 Round 2 的合理方向是：

```text
不要讓 impl=2 覆蓋所有 shape；
只在它有證據優勢的大 slice 上使用。
```

***

# 8. 報告中可寫的正式解釋

可以把 Round 1 寫成：

```text
Mode B Round 1 evaluated a compound block-per-slice cached-exp softmax candidate. The candidate was not accepted as a full replacement because slice=256 showed one correctness failure and slice=128 regressed substantially. However, the candidate produced consistent improvements on larger slices: approximately 1.29x for slice=784, 1.59x for slice=1024, and 1.30x for slice=2048. Therefore, Round 1 supports a shape-aware dispatch strategy rather than a universal kernel replacement. Subsequent Mode B rounds should retain the existing impl=1 path for small slices and use the compound candidate only for larger slices where it is validated.
```

***

# 9. 主規劃器最終判定

```text
Round 1 status: PARTIAL_SUCCESS
Candidate as full replacement: REJECT
Candidate as large-slice path: KEEP
Next step: request Round 2 shape-aware dispatch proposal
Mode C: not yet
```
