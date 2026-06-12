可以。建議不要把「全方面優化」混進原本 Mode C 主流程，否則會讓結果歸因混亂。正確做法是：

```text
Mode C 主流程照原版：
  Evidence-guided aggressive optimization，主要嘗試超越 Mode B impl=3，合理情況下聚焦 large slices。

新增一個獨立附加實驗：
  Mode C-Extra / Full-Slice Optimization Experiment
  專門檢查 small / medium / large slices 是否都還有可優化空間。
```

這樣可以同時滿足：

```text
1. 原本 Mode C 保持高成功率與研究嚴謹性。
2. 額外實驗回答「是否有全面優化可能」。
3. 不會把 large-slice 優化、small-slice 探索、ablation 解釋混在同一個結果裡。
```

***

# 一、建議新增的實驗定位

## 新增實驗名稱

```text
Mode C-Extra: Full-Slice Optimization Opportunity Experiment
```

中文：

```text
Mode C 附加實驗：全 slice 優化機會檢查
```

***

# 二、這個實驗要回答什麼

這個附加實驗不是直接取代原 Mode C，而是回答：

```text
RQ-C-Extra:
在 Mode B 已經建立 shape-aware dispatch 後，softmax-cuda 是否仍存在跨 small / medium / large slices 的全面優化空間？
```

具體問題：

```text
1. slice=128 / 256 是否仍有安全優化空間？
2. slice=784 / 1024 / 2048 是否還能超越 impl=3？
3. 若 small slices 不適合再優化，原因是什麼？
4. 若 large slices 還能再優化，是否能整合成新的 dispatch policy？
5. 全方面優化是否優於「只針對 large slices」的 Mode C 主策略？
```

***

# 三、結果詮釋規則

這個附加實驗的結論必須獨立於主 Mode C。

## 可以宣稱

```text
Full-slice experiment attempted to evaluate optimization opportunities across all official slices.
```

若成功：

```text
Full-slice experiment found additional improvements beyond Mode B for specific slices.
```

若失敗：

```text
Full-slice experiment showed that small slices should remain on impl=1 due to low expected benefit or high risk.
```

## 不可以宣稱

```text
1. small slices 有效加速，除非 speedup_vs_impl3 >= 1% 且 correctness PASS。
2. full-slice candidate 是 universal optimization，除非五個 official slices 全部無 regression。
3. cached exp 是唯一原因，除非 ablation 支持。
4. profiler-supported conclusion，除非 profiler 實際可用。
```

***

# 四、加入 CLI Prompt 1 的補充段落

把下面這段加到 Prompt 1 的 **Submission 2** 後、**Submission 3** 前。  
這是最小且清楚的修改方式。

```text
# Additional Experiment: Mode C-Extra Full-Slice Optimization Opportunity Experiment

In addition to the main Mode C workflow, run one separate optional experiment:

  Mode C-Extra: Full-Slice Optimization Opportunity Experiment

Purpose:
  Evaluate whether all official softmax slices, including small slices 128 and 256, still have safe optimization opportunities beyond the accepted Mode B impl=3 dispatch policy.

This experiment must be separated from the main Mode C candidate evaluation.

Do not mix Mode C-Extra results into the main Mode C speedup table unless explicitly accepted later.

# Motivation document

Before running Mode C-Extra, create:

  mode_C_literature_profiler/full_slice_experiment_reason.md

This document must explain:

1. Why this experiment is needed.
2. How it differs from the main Mode C optimization.
3. Why the main Mode C path may naturally focus on large slices.
4. Why small slices 128 and 256 still deserve explicit evaluation.
5. What risks exist when optimizing small slices.
6. What evidence would justify keeping small slices unchanged.
7. What evidence would justify introducing a new small-slice candidate.
8. How results will be classified.

The document must clearly state:

  Mode C-Extra is an exploratory full-slice opportunity test.
  It is not allowed to override Mode B impl=3 unless all official cases pass correctness and the result is superior to impl=3 without hidden regression.

# Full-slice experiment required analysis

Before implementing any full-slice candidate, produce:

  mode_C_literature_profiler/full_slice_opportunity_table.md

For every official slice:

  slice=128
  slice=256
  slice=784
  slice=1024
  slice=2048

Report:

1. Mode B selected implementation:
   - impl=1 or impl=2

2. Current Mode B role:
   - small-slice baseline path
   - large-slice optimized path

3. Remaining optimization opportunity:
   - HIGH
   - MEDIUM
   - LOW
   - UNKNOWN

4. Proposed action:
   - keep impl=1
   - tune impl=1
   - add small-slice candidate
   - keep impl=2
   - tune impl=2
   - add large-slice candidate
   - add full-slice dispatch candidate

5. Risk:
   - correctness risk
   - regression risk
   - measurement noise risk
   - implementation complexity risk

6. Evidence:
   - Mode B final result
   - Round 1 result
   - source-level reasoning
   - literature / documentation
   - profiler data if available

If the agent chooses not to optimize slice=128 or slice=256, it must explicitly justify this using evidence.

Do not silently ignore small slices.

# Full-slice candidate rule

If Mode C-Extra implements a candidate, prefer adding a new implementation ID:

  impl=5 or higher

Do not overwrite:
  impl=0
  impl=1
  impl=2
  impl=3
  any main Mode C candidate

The full-slice candidate may combine:
  - small-slice strategy
  - large-slice strategy
  - dispatch logic

But it must be labeled clearly as:

  variant=modeC_extra_full_slice_candidate

# Full-slice validation rule

Mode C-Extra must compare:

  impl=1 baseline
  impl=3 Mode B accepted candidate
  full-slice candidate

For all official cases:

  numSlice=100000, sliceSize=128, repeat=100
  numSlice=100000, sliceSize=256, repeat=100
  numSlice=100000, sliceSize=784, repeat=100
  numSlice=100000, sliceSize=1024, repeat=100
  numSlice=50000,  sliceSize=2048, repeat=100

At least 3 trials per official slice and implementation.

CSV must include:

  speedup_vs_impl1
  speedup_vs_impl3
  correctness_status
  measurement_validity
  speedup_claim_valid
  result_type
  raw_stdout_path
  raw_stderr_path
  profiler_status

# Full-slice classification

For every slice:

- correctness FAIL:
    result_type=INVALID
    speedup_claim_valid=false

- speedup_vs_impl3 < 1.01:
    no additional Mode C speedup

- speedup_vs_impl3 >= 1.01 and correctness PASS and CV stable:
    additional full-slice improvement may be claimed for that slice

- speedup_vs_impl3 < 0.99:
    result_type=REGRESSION

- slice=128 or slice=256:
    if candidate still dispatches to impl=1, classify as MEASUREMENT_EQUIVALENT
    if candidate changes implementation, require correctness PASS and stable timing before any speedup claim

# Full-slice final interpretation

Mode C-Extra final label must be one of:

  FULL_SLICE_SUCCESS
  FULL_SLICE_PARTIAL_SUCCESS
  FULL_SLICE_NO_ADDITIONAL_GAIN
  FULL_SLICE_INCONCLUSIVE
  FULL_SLICE_INVALID

Definitions:

FULL_SLICE_SUCCESS:
  All official slices correctness PASS.
  Candidate improves at least one slice over impl=3.
  No official slice regresses by >=1%.
  Auditor PASS.

FULL_SLICE_PARTIAL_SUCCESS:
  Some slices improve over impl=3, but others are measurement-equivalent or mildly regressed.
  No correctness failure.
  Must report per-slice result.

FULL_SLICE_NO_ADDITIONAL_GAIN:
  Candidate does not improve beyond impl=3.
  But experiment confirms impl=3 is already a reasonable dispatch.

FULL_SLICE_INCONCLUSIVE:
  Evidence insufficient due to high CV, profiler limitation, or incomplete artifacts.

FULL_SLICE_INVALID:
  Any official slice correctness FAIL, missing official case, or invalid baseline.

# Separation rule

Mode C-Extra must not overwrite the main Mode C final label.

The final Mode C report must have separate sections:

1. Main Mode C result
2. Mode C-Extra full-slice experiment
3. Comparison between main Mode C and full-slice experiment
4. Final accepted candidate
5. Do-not-claim list

Stop before running Mode C-Extra if the main Mode C candidate has not yet produced a valid baseline comparison.
```

***

# 五、加入 Prompt 1 的「最終報告要求」補充

把下面加到 Prompt 1 的 final summary section。

```text
# Additional final report requirement for Mode C-Extra

If Mode C-Extra is performed, write:

  mode_C_literature_profiler/final/full_slice_experiment_summary.md

It must include:

1. Experiment motivation
2. Why main Mode C may focus on large slices
3. Why full-slice evaluation was added
4. Per-slice opportunity table
5. Candidate description
6. Results vs impl=1
7. Results vs impl=3
8. Per-slice correctness
9. Per-slice speedup_claim_valid
10. Small-slice conclusion
11. Large-slice conclusion
12. Whether full-slice candidate should replace impl=3
13. Limitations
14. Final label
15. Do-not-claim list

The final Mode C report must explicitly state whether the final accepted candidate is:
  - Mode B impl=3
  - main Mode C candidate
  - Mode C-Extra full-slice candidate
  - no replacement due to invalid or inconclusive evidence
```

***

# 六、審查 session prompt 也要補充

把下面加到 Prompt 2 的 **Mode C result audit** 後面。

```text
# E. Mode C-Extra full-slice experiment review

If I paste a Mode C-Extra proposal or result, review it separately from the main Mode C workflow.

Mode C-Extra purpose:
  Evaluate whether all official slices, including 128 and 256, have additional safe optimization opportunities beyond Mode B impl=3.

Review checklist:

1. Does the proposal include full_slice_experiment_reason.md?
2. Does it explain why the full-slice experiment is needed?
3. Does it include a per-slice opportunity table?
4. Does it explicitly analyze slice=128 and slice=256?
5. If it keeps small slices on impl=1, does it justify that decision?
6. If it modifies small slices, does it preserve correctness and compare against impl=3?
7. Does it compare against both impl=1 and impl=3?
8. Does it preserve all official cases?
9. Does it avoid hiding regression?
10. Does it keep Mode C-Extra separate from the main Mode C result?
11. Does it avoid claiming universal optimization unless all slices support it?
12. Does it include auditor output?
13. Does it preserve raw stdout/stderr?
14. Does it define final label correctly?

Mode C-Extra labels:

- FULL_SLICE_SUCCESS
- FULL_SLICE_PARTIAL_SUCCESS
- FULL_SLICE_NO_ADDITIONAL_GAIN
- FULL_SLICE_INCONCLUSIVE
- FULL_SLICE_INVALID

Audit output format:

## Mode C-Extra Review Summary

- verdict: APPROVE / NEEDS_REVISION / REJECT
- final label if result is provided:
- blocking issues:
- non-blocking issues:
- required fixes:

## Full-Slice Per-Case Audit

Include:
- slice
- current Mode B impl
- full-slice candidate impl
- correctness
- speedup_vs_impl3
- measurement_validity
- speedup_claim_valid
- result_type
- notes

## Accepted Claims

List claims supported by evidence.

## Rejected Claims

List unsupported claims.

## Paper-Safe Interpretation

Provide wording that can be used in the paper.

## Do-Not-Claim

List claims that must not be made.
```

***

# 七、你給 CLI 的短版指令

如果你不想整段重貼，可以直接補這段給 CLI：

```text
Update the Mode C plan.

Keep the original main Mode C workflow.

Additionally, add a separate experiment:

  Mode C-Extra: Full-Slice Optimization Opportunity Experiment

This experiment must explicitly evaluate optimization opportunities for all official slices:
  128, 256, 784, 1024, 2048

Before running it, create:
  full_slice_experiment_reason.md
  full_slice_opportunity_table.md

The experiment must compare:
  impl=1 baseline
  impl=3 Mode B accepted candidate
  full-slice candidate

The main metric is speedup_vs_impl3.

Do not silently ignore small slices.
If 128/256 remain on impl=1, justify using Mode B and Round 1 evidence.
If attempting small-slice optimization, preserve correctness and report per-slice results.

Keep Mode C-Extra separate from the main Mode C result.
Do not let Mode C-Extra overwrite the main Mode C final label.
```

***

# 八、這樣設計後的研究邏輯

最後你的 Mode C 會變成：

```text
Main Mode C:
  積極嘗試超越 impl=3，通常聚焦 large slices。

Mode C-Extra:
  回答「是否所有 slices 都還有可優化空間？」
  特別檢查 128 / 256 是否應該保留 impl=1。
```

這樣結論會更完整：

```text
如果 Extra 成功：
  你可以說 Mode C 找到更全面的 dispatch/candidate。

如果 Extra 失敗：
  你可以說 full-slice exploration 支持 Mode B 的原始判斷：
  small slices 應保留 impl=1，large slices 才值得積極優化。
```

這不會削弱研究，反而會補強你的「為什麼不優化小 slice」的論證。
