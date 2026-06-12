## Result Audit Summary

* **verdict:** REJECT for `impl=5` promotion; ACCEPT as failed/blocked ablation artifact

* **final label:** `BLOCKED`

* **accepted claims:**
  1. Submission 2 executed as a partial reduction-structure ablation.
  2. `impl=5` preserved `impl=0/1/2/3/4` according to patch summary.
  3. `impl=5` was additive.
  4. `impl=5` official timing rows used `profiler_status=NOT_RUN`.
  5. `impl=5` small slices 128/256 were guardrail rows and were not claimed as speedups.
  6. `impl=5` failed correctness on 784 and 1024; therefore Submission 2 is blocked.
  7. `impl=4` remains the current best Mode C candidate from Submission 1.

* **rejected claims:**
  1. No `impl=5` speedup claim is valid.
  2. No `impl=5` attribution claim is valid.
  3. No conclusion that reduction structure caused `impl=4` speedup is supported.
  4. No conclusion that shared-memory footprint caused speedup is supported.
  5. No conclusion that cached exponentials caused speedup is supported.
  6. No `impl=5` promotion over `impl=4` is allowed.

* **missing evidence:**
  1. Correctness PASS for `impl=5` on 784 and 1024 is missing.
  2. Auditor report for Submission 2 was not provided in the pasted material.
  3. `contradiction_check.csv` for Submission 2 was not provided.
  4. Raw stdout excerpts for correctness failures were not provided.
  5. Full source diff was not provided; patch summary is available, but raw code-level audit is not possible from this message alone.

***

## Per-Case Audit Table

| slice | implementation / variant              | correctness    | measurement validity |       speedup\_vs\_impl1 |                  speedup\_vs\_impl3 | speedup\_vs\_impl4 | speedup claim validity | result type                                    | notes                                                                                                                            |
| ----: | ------------------------------------- | -------------- | -------------------- | -----------------------: | ----------------------------------: | -----------------: | ---------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
|   128 | `impl=5_reduction_structure_ablation` | PASS           | VALID                | 1.019x mean-level approx |                              1.001x |             1.005x | false                  | MEASUREMENT\_EQUIVALENT                        | Guardrail row only. Dispatches to unchanged `impl=1`; no small-slice claim allowed.                                              |
|   256 | `impl=5_reduction_structure_ablation` | PASS           | VALID                | 0.999x mean-level approx | 1.166x, but impl3 baseline is NOISY |             0.998x | false                  | MEASUREMENT\_EQUIVALENT                        | Guardrail row only. `impl=3` at 256 is noisy; do not use `speedup_vs_impl3`.                                                     |
|   784 | `impl=5_reduction_structure_ablation` | FAIL           | INVALID              | 1.436x mean-level approx |                              1.026x |             0.902x | false                  | INVALID                                        | Correctness FAIL. All speedup/attribution claims invalid.                                                                        |
|  1024 | `impl=5_reduction_structure_ablation` | PARTIAL / FAIL | INVALID              | 1.662x mean-level approx |                              0.973x |             0.928x | false                  | INVALID                                        | At least trials 2 and 3 fail; one row shows PASS but slice-level result is invalid.                                              |
|  2048 | `impl=5_reduction_structure_ablation` | PASS           | VALID                | 1.279x mean-level approx |                              0.957x |             0.949x | false                  | REGRESSION or ABLATION\_ONLY\_WITH\_REGRESSION | Correctness PASS, but slower than both `impl=3` and `impl=4` by more than 1%. If schema only allows one label, use `REGRESSION`. |

***

## Contradictions

### 1. `impl=5` 不能有任何 accepted speedup claim

No contradiction in the summary: it correctly says:

```text
Explicit Accepted Claims:
- none
```

This is correct because:

* 784 correctness = FAIL
* 1024 correctness = FAIL / partial fail
* 2048 correctness = PASS but slower than `impl=3` and `impl=4`
* 128/256 are guardrail rows only

### 2. `slice=2048` result\_type 應更保守

Summary table labels 2048 as:

```text
ABLATION_ONLY
```

But the data show:

```text
speedup_vs_impl3 = 0.956929
speedup_vs_impl4 = 0.949118
```

This is a regression relative to both primary comparators. Since your classification rule says slower by >=1% should be recorded as regression relative to comparator, the safer row-level classification is:

```text
REGRESSION
```

If the project wants to preserve the ablation nature, use narrative wording:

```text
ABLATION_ONLY result with regression relative to impl=3 and impl=4
```

But do not leave it as neutral `ABLATION_ONLY` without stating regression.

### 3. `slice=1024` mixed correctness needs exact wording

The table says:

```text
correctness = FAIL
```

CSV rows show:

* trial 1: PASS but `measurement_validity=INVALID`
* trial 2: FAIL
* trial 3: FAIL

Slice-level classification should be:

```text
correctness_status=PARTIAL or FAIL
measurement_validity=INVALID
result_type=INVALID
speedup_claim_valid=false
```

If the summary uses only one slice-level correctness value, `FAIL` is acceptable.

### 4. `mode_c_final_label=BLOCKED` on all rows is acceptable

This is a submission-level label repeated across rows. It is acceptable if documented. Row-level `result_type` remains the primary per-case classification.

***

## Interpretation of Submission 2

Submission 2 attempted to answer:

> If the warp/cross-warp reduction structure in `impl=4` is weakened or removed while preserving the reduced shared-memory footprint, does performance move toward `impl=3` or stay near `impl=4`?

However, the actual ablation failed correctness on key slices:

* 784: FAIL
* 1024: FAIL

Therefore, the ablation cannot support attribution on the main slices where Submission 1 had accepted gains.

The only correctness-valid large-slice ablation result is 2048, but it regresses relative to both:

* `impl=3`
* `impl=4`

This supports only a limited negative statement:

```text
The implemented impl=5 ablation did not produce a valid improvement and cannot replace impl=4. On 2048, where correctness passed, it was slower than both impl=3 and impl=4.
```

It does **not** support:

```text
reduction structure caused impl=4 speedup
```

or:

```text
shared-memory footprint caused impl=4 speedup
```

***

## Paper-Safe Interpretation

可直接寫入報告的保守版本：

```text
Submission 2 attempted a partial reduction-structure ablation through `impl=5_reduction_structure_ablation`. The candidate preserved the reduced dynamic shared-memory allocation shape of `impl=4` while weakening the warp-shuffle reduction path and using shared-memory tree reductions. This was intended to test whether the reduction-structure difference between `impl=3` and `impl=4` was a plausible contributor to the accepted 784/1024 Submission 1 gains.

The ablation did not produce valid attribution evidence. `impl=5` failed correctness on the key large-slice cases 784 and 1024, so no speedup or attribution claim is valid for those slices. On slice 2048, correctness passed, but `impl=5` was slower than both `impl=3` and `impl=4`, so it is classified as a regression or ablation-only regression. The small slices 128 and 256 remained guardrail cases and do not support speedup claims. Consequently, Submission 2 is labeled `BLOCKED`, `impl=5` is not promoted, and `impl=4` remains the current best Mode C candidate from Submission 1. The result does not prove reduction-structure, shared-memory-footprint, or cached-exponential causality.
```

更精簡中文版本：

```text
Submission 2 的 `impl=5_reduction_structure_ablation` 嘗試進行 partial reduction-structure ablation，但結果未能提供有效 attribution evidence。`impl=5` 在 784 與 1024 兩個關鍵 large slices 發生 correctness failure，因此不能對這些 slices 宣稱 speedup 或因果歸因。2048 雖然 correctness PASS，但相對 `impl=3` 與 `impl=4` 均為 regression。128 與 256 只是 fallback guardrail rows，不支援 speedup claim。因此 Submission 2 應標為 `BLOCKED`；`impl=5` 不得 promoted，Mode C 目前最佳候選仍為 Submission 1 的 `impl=4`。
```

***

## Do-Not-Claim

不得宣稱：

1. `impl=5` 有任何 accepted speedup。
2. `impl=5` 改善 784。
3. `impl=5` 改善 1024。
4. `impl=5` 改善 2048。
5. `impl=5` 支持 reduction structure causality。
6. `impl=5` 支持 shared-memory footprint causality。
7. `impl=5` 支持 cached-exp causality。
8. `impl=5` 是新的 best Mode C candidate。
9. `impl=5` 可以取代 `impl=4`。
10. Submission 2 成功驗證了 ablation hypothesis。
11. 2048 是 neutral result；它應明確標示為相對 `impl=3/4` regression。
12. 784/1024 的 timing values 可用於解釋，因 correctness FAIL 使其無效。

***

## Final Decision

```text
Submission 2 verdict: REJECT as candidate / ACCEPT as blocked ablation artifact
Submission-level final label: BLOCKED
Accepted speedup claims: none
Accepted attribution claims: none
Accepted best Mode C candidate remains: impl=4_shape_specialized_large_reduce
```

***

## Recommended Next Step

目前不建議繼續修 `impl=5`。理由：

1. 它在關鍵 large slices 784/1024 correctness FAIL。
2. 2048 correctness PASS 但 regression。
3. 它沒有提供有效 attribution evidence。
4. 若繼續修 `impl=5`，會消耗更多時間但不保證提升論文主張。
5. Submission 1 已有有效 Mode C 成果：
   * 784: 1.131x vs `impl=3`
   * 1024: 1.049x vs `impl=3`

建議下一步：

```text
Proceed to Mode C final confirmation using impl=4 as the current best Mode C candidate.
```

Final confirmation 應比較：

* `impl=1`
* `impl=3`
* `impl=4`

official slices:

* 128
* 256
* 784
* 1024
* 2048

並維持：

* 128/256：no speedup claim
* 784/1024：possible accepted additional Mode C speedup if confirmed
* 2048：measurement-equivalent unless final confirmation shows valid >=1% speedup vs `impl=3`
* profiler timing not used
* `impl=5` only listed as blocked ablation artifact

***

## Suggested Reply to CLI

```text
Submission 2 audit result: BLOCKED.

Accepted:
- Submission 2 was executed as a partial reduction-structure ablation.
- impl=5 was additive.
- impl=0/1/2/3/4 were preserved according to patch summary.
- raw results were preserved.
- small slices 128/256 were correctly treated as guardrail rows.

Rejected:
- No impl=5 speedup claim is accepted.
- No impl=5 attribution claim is accepted.
- impl=5 must not replace impl=4.
- 784 and 1024 are invalid due to correctness failure.
- 2048 is correctness PASS but regresses vs both impl=3 and impl=4.
- The ablation does not prove reduction structure, shared-memory footprint, or cached-exp causality.

Decision:
- Submission 2 final label: BLOCKED.
- Current best Mode C candidate remains impl=4_shape_specialized_large_reduce.
- Do not continue modifying impl=5 unless a separate human-approved repair round is proposed.

Next step:
- Prepare Mode C final confirmation proposal using impl=4 as the current best candidate.
- Include impl=1 baseline, impl=3 Mode B baseline, and impl=4 Mode C candidate.
- Do not include impl=5 as candidate for promotion; include it only as blocked ablation evidence in narrative.
```

***

## Final Confirmation Proposal Requirements

若你要叫 CLI 進 final confirmation，建議要求：

```text
Mode C final confirmation proposal:

Accepted current Mode C candidate:
- impl=4_shape_specialized_large_reduce

Compare:
- impl=1 baseline
- impl=3 Mode B accepted baseline
- impl=4 Mode C candidate

Do not include impl=5 as promotable candidate.
impl=5 should be discussed only as blocked ablation artifact.

Official cases:
- 128
- 256
- 784
- 1024
- 2048

At least 3 independent trials per slice per implementation.

Required CSV:
- speedup_vs_impl1
- speedup_vs_impl3
- correctness_status
- measurement_validity
- speedup_claim_valid
- result_type
- mode_c_final_label
- raw_stdout_path
- raw_stderr_path
- profiler_status=NOT_RUN for official timing rows

Classification:
- 128/256: MEASUREMENT_EQUIVALENT, no Mode C speedup claim
- 784/1024: valid Mode C speedup only if speedup_vs_impl3 >= 1.01, correctness PASS, stable CV, auditor PASS
- 2048: speedup claim only if speedup_vs_impl3 >= 1.01; otherwise MEASUREMENT_EQUIVALENT
- any correctness FAIL -> INVALID
- any high CV -> CAUTION/NOISY, no speedup claim unless remeasured

Do not claim:
- profiler-supported causality
- cached-exp causality
- reduction-structure causality
- impl=4 universal optimization
- impl=5 success
```
