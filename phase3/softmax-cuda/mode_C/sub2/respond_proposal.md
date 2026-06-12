## Review Summary

* **verdict: NEEDS\_REVISION**

* **blocking issues:**
  1. **`impl=5` 的 ablation design 還不夠可執行，且隔離性不足。**  
     Proposal 說「保留 `impl=4` reduced shared-memory footprint，改回 full-block shared-memory tree reduction over compact 32-float reduction area」。這裡有概念矛盾：如果 reduction buffer 只有 32 floats，已經不是 `impl=3` 原本 256-thread shared-memory tree reduction 的等價對照。這可能無法清楚隔離「warp/cross-warp reduction」與「reduced shared-memory footprint」。
  2. **`impl=5` 的比較目標不夠明確。**  
     目前同時想回答：
     * reduction structure 是否解釋 `impl=4` gains
     * reduced shared-memory footprint 是否解釋 `impl=4` gains  
       但單一 `impl=5` 可能無法乾淨分離兩者。需要更明確的 ablation matrix 或至少承認此為 partial ablation。
  3. **classification 缺少 `PARTIAL / INCONCLUSIVE` 的 submission-level final label criteria。**  
     目前只有 row-level result classification，沒有定義 Submission 2 整體要如何標示：
     * `SUCCESS_WITH_ADDITIONAL_SPEEDUP`
     * `PARTIAL_SUCCESS`
     * `INCONCLUSIVE`
     * `BLOCKED`
  4. **沒有明確規定 `impl=5` 若只是 ablation，不應自動取代 `impl=4`。**  
     Proposal 有 rollback plan，但還需明確：除非 `impl=5` 同時 validly improves over `impl=3` and `impl=4`，否則 accepted Mode C candidate 仍為 `impl=4`。
  5. **`SUBMISSION_1_REFERENCE` 不在你前面指定的 result\_type classification 清單中。**  
     目前允許的 result\_type 包含 `MODE_C_CANDIDATE`、`ABLATION_ONLY`、`BASELINE_COMPARISON` 等，但不含 `SUBMISSION_1_REFERENCE`。需改成已定義 label，或明確新增並定義。

* **non-blocking issues:**
  1. `impl=5` 的 source-level design 沒有 pseudocode 或同步點說明，實作端可能自由度過大。
  2. Proposal 說「no causal claim without stable timing and auditor PASS」，但因果 claim 仍應更嚴格：需要 ablation pattern 支持，且只能寫 `plausible contributor`，不能寫 `proven cause`。
  3. profiler rerun 的 limited evidence 有提到，但沒有要求在 `summary.md` 中區分：
     * timing evidence
     * profiler resource observation
     * ablation evidence  
       建議補上。
  4. Small slices 128/256 dispatch 到 `impl=1`，但 `speedup_vs_impl3` 可能受到 noisy `impl=3` baseline 影響。應明確禁止小 slice 以 `speedup_vs_impl3` 形成任何 claim。

* **required fixes:**
  1. 重寫 `impl=5` 的 ablation intent，承認它是 **partial reduction-structure ablation**，或改成更乾淨的 two-candidate ablation design。
  2. 補 submission-level final label criteria。
  3. 修正 result\_type vocabulary：不要使用未定義的 `SUBMISSION_1_REFERENCE`，或正式新增定義。
  4. 明確規定 accepted candidate promotion rule。
  5. 明確規定 `impl=5` 的結果最多支持「plausible contributor」，不得證明 causality。
  6. 補 `summary.md` 必須列出 per-slice `speedup_vs_impl3`、`speedup_vs_impl4`、classification 與 interpretation。

* **optional improvements:**
  1. 如果想要更乾淨的 attribution，Submission 2 可改為兩個 ablation candidates：
     * `impl=5a`: `impl=4` shared-memory footprint + `impl=3-like` reduction
     * `impl=5b`: `impl=3` shared-memory footprint + `impl=4-like` reduction  
       但這會增加實作與測試成本。
  2. 若預算有限，保留單一 `impl=5` 也可，但需將結論限定為 partial evidence。

***

## Detailed Review

### 1. 目標與定位

目前 proposal 的目標正確：

```text
Submission 2 = reduction-structure ablation
```

它不是 blind tuning，也不是 2048-specific optimization。這符合 profiler rerun 後的方向判定。

但目前設計還不夠乾淨。你想用 `impl=5` 測試：

```text
reduction structure 是否解釋 impl=4 在 784/1024 的改善
```

然而 proposal 同時保留 `impl=4` 的 reduced shared-memory footprint，並改成 compact 32-float reduction buffer 的 shared-memory tree reduction。這不等於回到 `impl=3` reduction，因為：

* `impl=3` 使用較大的 dynamic shared memory。
* `impl=3` 的 reduction 結構與同步點可能基於 256-thread / full-block buffer。
* `impl=5` 若只在 32-float buffer 上做 tree reduction，它測到的是「compact-buffer shared-memory reduction」而不是原始 `impl=3` reduction。

因此它可以做 partial ablation，但不能說完全 isolate reduction structure。

***

### 2. Source-change scope

通過。

Proposal 明確只修改：

```text
/home/r14525078/HeCBench/src/softmax-cuda/main.cu
```

並新增：

* `softMax5`
* `implementation 5`
* `kernel == 5`

也明確保留：

* `impl=0/1/2/3/4`
* CPU reference
* tolerance
* input generation
* official cases
* `numSlice`
* `repeat`

這符合 additive candidate policy。

***

### 3. Dispatch policy

基本通過。

`impl=5` dispatch：

* 128/256 -> unchanged `impl=1`
* 784/1024/2048 -> `softMax5`

合理。

但建議補一句：

```text
For 128/256, impl=5 rows are guardrail rows only and must not be interpreted using speedup_vs_impl3.
```

因為 small slice 的 `impl=3` 可能曾出現 noisy row，且 small slices 本來就是 fallback cases。

***

### 4. Ablation interpretability

目前是主要問題。

你的 interpretation table 有價值，但還不夠保守。建議改成：

| impl=5 outcome     | Paper-safe interpretation                                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| matches `impl=4`   | reduction structure alone is not isolated as the sole contributor; reduced shared-memory footprint or codegen remains plausible |
| matches `impl=3`   | reduction-structure difference becomes a plausible contributor to `impl=4`; not proven cause                                    |
| slower than both   | ablation is not interpretable as clean attribution; implementation overhead likely confounds result                             |
| improves over both | may be a new candidate, but only after official validation; ablation role becomes mixed with optimization                       |

尤其「matches impl=3」不能寫成「reduction structure caused speedup」。只能寫：

```text
consistent with reduction structure being a plausible contributor
```

***

### 5. Validation plan

通過。

要求包含：

* `impl=1`
* `impl=3`
* `impl=4`
* `impl=5`
* all official slices
* at least 3 trials
* interleaved order
* raw stdout/stderr
* auditor / contradiction check

這足夠。

***

### 6. CSV schema

大致通過，但有一個 label 問題。

你目前 result classification 使用：

```text
impl=4: SUBMISSION_1_REFERENCE
```

但這不是前面 Mode C classification 清單中的合法值。建議改成：

```text
impl=4: MODE_C_CANDIDATE
```

並在 notes 中寫：

```text
Submission 1 accepted candidate reference
```

或者新增：

```text
SUBMISSION_1_REFERENCE
```

但若要新增，必須在 classification vocabulary 中明確定義。為了減少 schema 分裂，建議用既有 `MODE_C_CANDIDATE`。

***

### 7. Result classification

需補 submission-level final label。

建議新增：

```text
Submission 2 final label criteria:

SUCCESS_WITH_ADDITIONAL_SPEEDUP:
- impl=5 correctness PASS for all official cases
- auditor PASS
- at least one large slice improves over both impl=3 and impl=4 by >=1%
- no large-slice regression vs impl=4
- small slices remain measurement-equivalent guardrails

PARTIAL_SUCCESS:
- correctness PASS for all official cases
- impl=5 provides useful ablation evidence
- but does not improve over both impl=3 and impl=4, or only improves some large slices with regression elsewhere

INCONCLUSIVE:
- correctness PASS but noisy measurement, unclear attribution, or impl=5 overhead prevents interpretation

BLOCKED:
- build failure, missing official cases, correctness FAIL, missing raw output, missing paired impl=3/4, or auditor failure
```

若 `impl=5` 是純 ablation，即使不改善，也可有研究價值，但不應標 `SUCCESS_WITH_ADDITIONAL_SPEEDUP`。

***

### 8. Auditor plan

通過，但需補兩條：

```text
- check speedup_vs_impl4 exists for impl=5 rows
- check impl=5 is not promoted unless it improves over both impl=3 and impl=4 or is explicitly labeled ABLATION_ONLY
```

你已有類似內容，但可以更明確。

***

### 9. Rollback plan

通過。

但建議加：

```text
If impl=5 is ABLATION_ONLY, the accepted best candidate remains impl=4 unless impl=5 validly improves over impl=4 and impl=3.
```

這可以防止 ablation 被誤 promoted。

***

## Revised Text for CLI

以下是可直接貼回 CLI 的修正版要求。不需要整份重寫，只需要求它修訂 proposal 中指定段落。

```text
Submission 2 proposal review: NEEDS_REVISION.

The proposal direction is accepted in principle:
- Submission 2 = reduction-structure ablation
- additive impl=5 candidate
- preserve impl=0/1/2/3/4
- compare impl=5 against impl=1, impl=3, and impl=4
- all official slices and raw/auditor requirements are correct

However, revise the proposal before execution.

Required revisions:

1. Clarify ablation scope.

The current impl=5 design is not a perfect isolation of reduction structure because it preserves impl=4's reduced shared-memory footprint and uses a compact 32-float reduction area. Therefore, label this as a partial reduction-structure ablation.

Add this statement:

"impl=5 is a partial reduction-structure ablation. It attempts to weaken or remove the warp-shuffle reduction path while preserving the reduced shared-memory footprint of impl=4. Because shared-memory footprint and code generation remain possible confounders, impl=5 can support only plausible attribution, not proof of causality."

2. Revise interpretation language.

Use "plausible contributor" instead of "cause" or "proof".

Allowed:
- consistent with reduction structure being a plausible contributor
- supports reduction-structure hypothesis
- weakens reduction-structure attribution
- inconclusive due to implementation overhead

Forbidden:
- proves reduction structure caused speedup
- proves shared-memory footprint caused speedup
- proves cached-exp contribution

3. Fix result_type vocabulary.

Do not use SUBMISSION_1_REFERENCE unless you define it explicitly.

Preferred:
- impl=1: BASELINE
- impl=3: MODE_B_BASELINE
- impl=4: MODE_C_CANDIDATE, notes="Submission 1 accepted candidate reference"
- impl=5: ABLATION_ONLY unless it validly improves over both impl=3 and impl=4

4. Add submission-level final label criteria.

Add:

SUCCESS_WITH_ADDITIONAL_SPEEDUP:
- impl=5 correctness PASS for all official cases
- auditor PASS
- at least one large slice improves over both impl=3 and impl=4 by >=1%
- no large-slice regression vs impl=4
- small slices remain measurement-equivalent guardrails

PARTIAL_SUCCESS:
- correctness PASS for all official cases
- impl=5 provides useful ablation evidence
- but it does not improve over both impl=3 and impl=4, or only improves some large slices while another large slice is measurement-equivalent or regresses

INCONCLUSIVE:
- correctness PASS but measurements are noisy, attribution remains unclear, or impl=5 overhead prevents clean interpretation

BLOCKED:
- build failure
- missing official cases
- correctness FAIL
- missing raw output
- missing paired impl=3 or impl=4
- auditor failure

5. Clarify promotion rule.

Add:

"impl=5 must not replace impl=4 as the accepted Mode C candidate unless it validly improves over both impl=3 and impl=4 on at least one large slice, has no large-slice regression vs impl=4, passes correctness on all official cases, and passes auditor checks. Otherwise, impl=5 remains ABLATION_ONLY and impl=4 remains the current best Mode C candidate."

6. Clarify small-slice rule.

Add:

"For slices 128 and 256, impl=5 dispatches to unchanged impl=1. These rows are guardrail rows only. Do not use speedup_vs_impl3 to claim any small-slice Mode C improvement."

7. Require summary.md to include:

- per-slice comparison table
- speedup_vs_impl3
- speedup_vs_impl4
- correctness
- CV
- row-level result_type
- submission-level final label
- explicit accepted claims
- explicit rejected claims
- do-not-claim list

After revising the proposal, stop and wait for human approval. Do not modify source and do not submit sbatch yet.
```

***

## Approval Decision

```text
NEEDS_REVISION
```

目前不批准 Submission 2 execution。  
可以批准的只有：

```text
Revise Submission 2 proposal according to the required fixes.
```

不批准：

```text
source modification
impl=5 implementation
optimization sbatch
official validation run
```

***

## 建議主規劃器目前回覆 CLI

可直接回覆：

```text
Submission 2 proposal direction is accepted in principle, but the proposal needs revision before execution.

Do not modify source.
Do not submit sbatch.

Revise the proposal to clarify that impl=5 is only a partial reduction-structure ablation, not a clean proof of causality. Fix result_type vocabulary, add submission-level final label criteria, clarify promotion rules, and explicitly state that small-slice rows are guardrails only.

After revision, stop for human approval.
```
