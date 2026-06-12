# Mode C Submission 2 Reduction-Structure Ablation Proposal

## Status

This is inspection and planning only.

No source modification has been made for Submission 2. No sbatch job has been
submitted for Submission 2. Human approval is required before any source change
or benchmark execution.

## Background

Submission 1 candidate:

- `impl=4_shape_specialized_large_reduce`

Accepted additional speedup claims:

- slice 784: 1.131x vs `impl=3`
- slice 1024: 1.049x vs `impl=3`

Not accepted as additional speedup:

- slice 128
- slice 256
- slice 2048

Profiler rerun status:

- `ACCEPT_WITH_LIMITATIONS`
- selected launch/resource metrics recovered
- `impl=4` uses less dynamic shared memory per block than `impl=3`
- registers/thread and waves/SM are unchanged
- no memory throughput, warp execution efficiency, instruction mix,
  math/special-function, or scheduler/stall breakdown recovered
- profiler timing is diagnostic only and must not be used for official speedup

## Candidate Design

Required candidate:

- `impl=5_reduction_structure_ablation`

Goal:

- test whether the reduction-structure difference between `impl=3` / `softMax3`
  and `impl=4` / `softMax4` is a plausible contributor to the accepted 784/1024
  additional speedups.

Scope statement:

> impl=5 is a partial reduction-structure ablation. It attempts to weaken or remove the warp-shuffle reduction path while preserving the reduced shared-memory footprint of impl=4. Because shared-memory footprint and code generation remain possible confounders, impl=5 can support only plausible attribution, not proof of causality.

Proposed design:

- Add a new large-slice kernel, tentatively named `softMax5`.
- Use the same block-per-slice work decomposition as `impl=3` and `impl=4`.
- Use the same cached-exp softmax math as `impl=3` and `impl=4`.
- Keep the reduced dynamic shared-memory footprint of `impl=4`:
  `sizeof(float) * (sliceSize + 32)`.
- Intentionally replace `impl=4`'s warp/cross-warp shuffle reduction with a
  full-block shared-memory tree reduction over the compact 32-float reduction
  area where feasible.

Interpretation intent:

- `impl=4` changed both reduction structure and dynamic shared-memory footprint
  relative to `impl=3`.
- `impl=5` should preserve the reduced shared-memory footprint while weakening
  or removing the warp-shuffle reduction advantage.
- This helps test whether the observed `impl=4` result is more consistent with
  reduction-structure changes or shared-memory footprint changes.
- This does not prove reduction structure, shared-memory footprint, or cached
  exponentials caused the speedup.

## Exact Source-Level Changes

If approved, modify only:

- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`

Planned source changes:

- Add a new `softMax5` kernel.
- Add `implementation 5` usage text.
- Add a new `kernel == 5` dispatch path.
- For official slices 128 and 256, dispatch to unchanged `impl=1` behavior.
- For official slices 784, 1024, and 2048, launch `softMax5`.
- Preserve all existing `impl=0/1/2/3/4` code paths.

Small-slice rule:

> For slices 128 and 256, impl=5 dispatches to unchanged impl=1. These rows are guardrail rows only. Do not use speedup_vs_impl3 to claim any small-slice Mode C improvement.

Do not modify:

- CPU reference
- correctness tolerance
- input generation
- official cases
- `numSlice`
- `repeat`

## Kept Identical To impl=4

The ablation should keep these features identical or as close as possible to
`impl=4`:

- large-slice dispatch policy for 784, 1024, and 2048
- small-slice fallback to unchanged `impl=1`
- `BLOCK_SIZE=256`
- one block per slice for large slices
- cached exponentials in shared memory
- reduced shared-memory allocation size: `sliceSize + 32`
- output write pattern
- softmax math and correctness tolerance

## Intentionally Changed Relative To impl=4

The ablation intentionally changes:

- reduction implementation
- number and placement of block-level synchronization points, if required by
  the non-shuffle shared-memory reduction
- how per-warp or per-thread partial values are combined

The intended change is to move away from `impl=4`'s warp-shuffle
`warp_reduce_max` / `warp_reduce_sum` path while retaining the reduced
shared-memory footprint.

## Kept Identical To impl=3 Or impl=2

The ablation should keep these features aligned with `impl=3` / `impl=2`:

- block-per-slice large-slice organization
- cached-exp strategy
- no approximate softmax
- exact official input generation and CPU reference
- full correctness validation after timing

The ablation may reuse the conceptual full-block shared-memory tree reduction
from `softMax3`, adapted to the compact 32-float reduction buffer if needed.

## Hypothesis Being Tested

Primary hypothesis:

- If `impl=5` regresses toward `impl=3` while preserving `impl=4`'s reduced
  dynamic shared-memory footprint, then the `impl=4` reduction structure is a
  plausible contributor to the 784/1024 gains.

Alternative hypotheses:

- If `impl=5` matches `impl=4`, reduced dynamic shared-memory footprint or
  another shared factor may be a stronger contributor than reduction structure.
- If `impl=5` matches `impl=3`, reduction structure alone is not supported as
  the contributor unless other confounders are identified.
- If `impl=5` is worse than both, the ablation introduced overhead or weakened
  the implementation in a way that prevents clean attribution.

No causal claim is allowed. With correctness PASS, stable timing, per-slice
comparison, and auditor PASS, the strongest allowed wording is plausible
attribution.

Allowed interpretation language:

- consistent with reduction structure being a plausible contributor
- supports reduction-structure hypothesis
- weakens reduction-structure attribution
- inconclusive due to implementation overhead

Forbidden interpretation language:

- proves reduction structure caused speedup
- proves shared-memory footprint caused speedup
- proves cached-exp contribution

## Expected Outcomes And Interpretation

| impl=5 outcome | Interpretation |
|---|---|
| matches `impl=4` on 784/1024 | reduction structure is not isolated as the main contributor; reduced shared memory or another shared factor remains plausible |
| matches `impl=3` on 784/1024 | reduction-structure difference becomes a plausible contributor to `impl=4` gains |
| slower than both `impl=3` and `impl=4` | ablation likely introduces overhead; attribution is weakened |
| improves over both `impl=3` and `impl=4` | may be a new optimization, but only if correctness PASS, stable CV, and auditor PASS; otherwise classify conservatively |
| 2048 remains measurement-equivalent | consistent with Submission 1; no additional 2048 speedup claim |
| 2048 improves over `impl=3` and `impl=4` | potential new optimization claim only if valid and audited |

Expected classification:

- 128/256: `MEASUREMENT_EQUIVALENT`; no speedup claim expected.
- 784/1024/2048: `ABLATION_ONLY` unless `impl=5` also validly improves over
  `impl=3` or `impl=4`.

## Submission-Level Final Label Criteria

`SUCCESS_WITH_ADDITIONAL_SPEEDUP`:

- `impl=5` correctness PASS for all official cases
- auditor PASS
- at least one large slice improves over both `impl=3` and `impl=4` by >=1%
- no large-slice regression vs `impl=4`
- small slices remain measurement-equivalent guardrails

`PARTIAL_SUCCESS`:

- correctness PASS for all official cases
- `impl=5` provides useful ablation evidence
- but it does not improve over both `impl=3` and `impl=4`, or only improves
  some large slices while another large slice is measurement-equivalent or
  regresses

`INCONCLUSIVE`:

- correctness PASS but measurements are noisy, attribution remains unclear, or
  `impl=5` overhead prevents clean interpretation

`BLOCKED`:

- build failure
- missing official cases
- correctness FAIL
- missing raw output
- missing paired `impl=3` or `impl=4`
- auditor failure

## Validation Plan

Run only after human approval.

Execution rules:

- use sbatch only
- do not run `./main` on login node
- do not skip official cases
- do not delete slow, failing, or regressing rows
- do not use profiler timing as official timing

Official cases:

- `numSlice=100000`, `sliceSize=128`, `repeat=100`
- `numSlice=100000`, `sliceSize=256`, `repeat=100`
- `numSlice=100000`, `sliceSize=784`, `repeat=100`
- `numSlice=100000`, `sliceSize=1024`, `repeat=100`
- `numSlice=50000`, `sliceSize=2048`, `repeat=100`

Required implementation set:

- `impl=1` baseline
- `impl=3` Mode B baseline
- `impl=4` Submission 1 candidate
- `impl=5` ablation candidate

Trial plan:

- at least 3 independent trials per slice per implementation
- interleaved order per slice/trial when practical:
  `impl=1`, `impl=3`, `impl=4`, `impl=5`

Artifacts:

- raw stdout/stderr for every trial
- build log
- Slurm stdout/stderr
- environment metadata
- `results.csv`
- `auditor_report.csv`
- `contradiction_check.csv`
- `summary.md`
- `patch_summary.md`

`summary.md` must include:

- per-slice comparison table
- `speedup_vs_impl3`
- `speedup_vs_impl4`
- correctness
- CV
- row-level `result_type`
- submission-level final label
- explicit accepted claims
- explicit rejected claims
- do-not-claim list

`patch_summary.md` must include:

- exact new kernel name
- exact `impl=5` dispatch map
- whether `impl=5` preserves `sliceSize + 32` dynamic shared-memory allocation
- reduction implementation difference vs `impl=4`
- reduction/shared-memory difference vs `impl=3`
- number and placement of added synchronization points if changed
- confirmation that `impl=0/1/2/3/4` were not modified
- confirmation that CPU reference, tolerance, input generation, official cases,
  `numSlice`, and `repeat` were not modified

## CSV Schema

`submission_2/results.csv` must include at least:

```text
benchmark
mode
stage
submission_id
variant
impl
baseline_impl
mode_b_impl
submission1_impl
ablation_impl
dispatch_selected_impl
numSlice
sliceSize
repeat
trial_id
time_ms
impl1_time_ms
impl3_time_ms
impl4_time_ms
speedup_vs_impl1
speedup_vs_impl3
speedup_vs_impl4
correctness_status
measurement_validity
speedup_claim_valid
result_type
mode_c_final_label
mean_ms
min_ms
max_ms
stddev_ms
cv
raw_stdout_path
raw_stderr_path
build_log_path
slurm_job_id
hostname
gpu_name
cuda_version
profiler_status
notes
```

Required values:

- `benchmark=softmax-cuda`
- `mode=Mode_C`
- `stage=submission_2`
- `submission_id=2`
- `baseline_impl=1`
- `mode_b_impl=3`
- `submission1_impl=4`
- `ablation_impl=5`
- `profiler_status=NOT_RUN` for official timing rows unless profiler is
  separately approved

## Result Classification

Baseline rows:

- `impl=1`: `BASELINE`
- `impl=3`: `MODE_B_BASELINE`
- `impl=4`: `MODE_C_CANDIDATE`, with
  `notes="Submission 1 accepted candidate reference"`

`impl=5` rows:

- correctness FAIL -> `INVALID`, `speedup_claim_valid=false`
- missing `impl=3` or `impl=4` paired row -> `INVALID`
- small slices 128/256 -> `MEASUREMENT_EQUIVALENT`, no speedup claim expected
- large slices without valid improvement -> `ABLATION_ONLY`
- speedup over both `impl=3` and `impl=4` by >=1%, correctness PASS, stable CV
  -> may be classified as candidate improvement after auditor/human review
- slower than `impl=3` or `impl=4` by >=1% -> record as regression relative to
  that comparator; do not hide with aggregate speedup

CV thresholds:

- CV <= 5%: `VALID`
- 5% < CV <= 15%: `CAUTION`
- CV > 15%: `NOISY`; no speedup claim unless remeasured

Promotion rule:

> impl=5 must not replace impl=4 as the accepted Mode C candidate unless it validly improves over both impl=3 and impl=4 on at least one large slice, has no large-slice regression vs impl=4, passes correctness on all official cases, and passes auditor checks. Otherwise, impl=5 remains ABLATION_ONLY and impl=4 remains the current best Mode C candidate.

## Auditor Plan

Auditor must check:

- all official cases present
- all required implementations present for every official case
- raw stdout/stderr paths exist
- correctness PASS before any speedup claim
- `speedup_vs_impl3` and `speedup_vs_impl4` exist for `impl=5` rows
- small slices are not claimed as Mode C speedups
- profiler timing is not used as official timing
- `speedup_vs_impl1` is not used as the main Mode C success metric
- `impl=5` is not called a new optimization unless it validly improves over
  both `impl=3` and `impl=4`
- `impl=5` does not replace `impl=4` unless the promotion rule is satisfied
- `summary.md` includes per-slice table, accepted claims, rejected claims, and
  do-not-claim list
- no aggregate-only success hides per-slice regression
- no causality language exceeds plausible attribution

## Risks

Correctness risk:

- medium. Changing reduction structure can alter accumulation order, though
  tolerance remains 1e-3.

Attribution risk:

- medium. A practical `impl=5` may not perfectly isolate reduction structure
  from shared-memory footprint or synchronization count.

Performance risk:

- medium to high. `impl=5` may be slower than both `impl=3` and `impl=4`.

Interpretation risk:

- profiler evidence is limited and does not include memory throughput, warp
  efficiency, instruction mix, or stall breakdown.
- even if `impl=5` matches `impl=3` or `impl=4`, the conclusion remains
  plausible attribution rather than proof.

Budget risk:

- implementation and official validation would consume Submission 2.

## Rollback Plan

Because `impl=5` is additive:

- keep `impl=0/1/2/3/4` unchanged
- if `impl=5` fails correctness, regresses, or produces noisy measurements,
  do not promote it
- keep `impl=4` as current best Mode C candidate from Submission 1
- keep `impl=5` as `ABLATION_ONLY` unless the promotion rule is satisfied
- preserve all raw outputs, including failures and regressions

## Human Approval Checkpoint

Stop here.

Do not modify source and do not submit sbatch until the human planner explicitly
approves Submission 2 execution.
