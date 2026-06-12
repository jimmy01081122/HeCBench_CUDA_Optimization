# Mode C Stage 0 Plan

## Status

Stage 0 inspection is complete. No source modification and no benchmark run have
been performed for Mode C.

## Baseline

Primary reference:
- Mode B accepted `impl=3`
- variant: `impl3_shape_dispatch_impl1_small_impl2_large`

Secondary reference:
- paired `impl=1`

Official slices:
- `numSlice=100000`, `sliceSize=128`, `repeat=100`
- `numSlice=100000`, `sliceSize=256`, `repeat=100`
- `numSlice=100000`, `sliceSize=784`, `repeat=100`
- `numSlice=100000`, `sliceSize=1024`, `repeat=100`
- `numSlice=50000`, `sliceSize=2048`, `repeat=100`

## Submission 1 Revised Proposal

Review status:
- Stage 0 review result: NEEDS_REVISION before Submission 1 execution.
- This plan incorporates the requested revision from
  `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C/respond_to_sub1.md`.
- No source modification may occur until explicit human approval is recorded.

Candidate:
- `impl=4_shape_specialized_large_reduce`

Source change:
- Add a new `impl=4` path in
  `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`.
- Preserve all existing `impl=0/1/2/3` behavior.
- For official slices 128 and 256, `impl=4` should dispatch to unchanged
  `impl=1` behavior.
- For official slices 784, 1024, and 2048, `impl=4` should use a new large-slice
  kernel derived from `softMax3` but with reduced shared-memory reduction and
  synchronization overhead.
- Non-official slices should fall back conservatively to unchanged `impl=3`
  behavior. No performance claims are allowed for non-official slices.

Hypothesis:
- Keep the same mathematical softmax semantics.
- Preserve large-slice block-level parallelism.
- Reduce shared-memory reduction and synchronization overhead by using per-warp
  reductions plus compact cross-warp reduction.
- This is a hypothesis only. No profiler-supported bottleneck claim is allowed
  without profiler evidence.

Expected per-slice effects:

| slice | expected effect |
|---:|---|
| 128 | measurement-equivalent to `impl=3`, via fallback to `impl=1` |
| 256 | measurement-equivalent to `impl=3`, via fallback to `impl=1` |
| 784 | possible improvement from lower reduction overhead |
| 1024 | possible improvement from lower reduction overhead |
| 2048 | possible smaller improvement or measurement-equivalent result |

Validation plan after approval:
- Build and benchmark through sbatch only.
- Do not run `./main` or any GPU benchmark binary on the login node.
- Run official cases only.
- For each official slice and trial, run interleaved `impl=1`, `impl=3`,
  `impl=4`.
- Use at least 3 trials per implementation and slice.
- Preserve raw stdout/stderr, build log, environment metadata, and Slurm job id.
- Produce `submission_1/results.csv`, `auditor_report.csv`,
  `contradiction_check.csv`, `summary.md`, and `patch_summary.md`.
- Required CSV schema is recorded in `submission_1/plan.md`.

Measurement validity:
- correctness FAIL -> INVALID and `speedup_claim_valid=false`.
- missing raw output -> LIMITED and `speedup_claim_valid=false`.
- missing `impl=3` baseline -> INVALID and no Mode C speedup claim.
- improvement <1% vs `impl=3` -> MEASUREMENT_EQUIVALENT.
- slower by >=1% vs `impl=3` -> REGRESSION.
- CV <= 5% -> VALID; 5% < CV <= 15% -> CAUTION; CV > 15% -> NOISY.

Auditor plan:
- Check all official cases are present.
- Check correctness PASS before any speedup claim.
- Check `speedup_vs_impl3 >= 1.01` before claiming additional Mode C speedup.
- Check small slices are not claimed as kernel improvements when they fallback
  to `impl=1`.
- Check profiler-supported claims are absent unless profiler data exists.
- Check no cached-exp causality claim is made without ablation.
- Check no aggregate-only success hides per-slice regression.

Final label criteria:
- `SUCCESS_WITH_ADDITIONAL_SPEEDUP`: all official slices correctness PASS, all
  cases present, auditor PASS, no hidden regression, at least one large slice has
  valid `speedup_vs_impl3 >= 1.01`, no large-slice regression, and small slices
  are measurement-equivalent or valid.
- `PARTIAL_SUCCESS`: correctness PASS for all official slices, some large slices
  improve vs `impl=3`, but one or more large slices are measurement-equivalent
  or regress.
- `INCONCLUSIVE`: correctness PASS, but measurements are unstable or no reliable
  `speedup_vs_impl3` claim exists.
- `BLOCKED`: build failure, runtime failure, missing official cases, missing
  `impl=3` baseline, missing raw output, or auditor failure.

Rollback plan:
- Because `impl=4` will be additive, rollback is to ignore `impl=4` and retain
  existing `impl=3` as the accepted Mode B reference.

Proceeding to Submission 1 proposal.

No source change yet.
