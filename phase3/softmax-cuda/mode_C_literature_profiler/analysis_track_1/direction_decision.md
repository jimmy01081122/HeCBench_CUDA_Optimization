# Direction Decision

## Recommendation

Recommended next direction:

- defer Submission 2 until profiler results are audited

This supersedes the earlier provisional recommendation. Analysis Track 1 is
accepted, but Submission 2 must not start yet. A profiler-only sbatch may be
proposed, but it must not be submitted until explicitly approved.

If the human planner approves one profiler-only sbatch run, the run itself
should be analysis-only and should not count as an optimization submission.

If profiler approval is not given, fallback recommendation:

- `Submission 2 = ablation`

## Why This Is Better Than Blind Tuning

Submission 1 already produced accepted additional speedup on 784 and 1024:

- 784: 1.131x vs `impl=3`
- 1024: 1.049x vs `impl=3`

But the result does not explain why:

- 2048 is only measurement-equivalent at 1.008x,
- warp/cross-warp reduction is the relevant contributor,
- shared-memory footprint is or is not limiting,
- block-size/resource tuning would help.

Blind tuning could find another timing win, but it would not improve the final
paper claims unless it also explains why the observed result happened. A
profiler-informed Submission 2 can target a concrete resource signal instead of
guessing, but only after the profiler artifacts have been audited.

## How It Builds On Submission 1

Submission 1 introduced `impl=4`, which keeps large-slice block-level
parallelism and changes the reduction structure. The accepted evidence is
per-slice and limited:

- improvement on 784 and 1024,
- no accepted improvement on 2048,
- no small-slice improvement claim.

Profiler evidence can compare `impl=3` and `impl=4` on exactly those three large
slices and identify whether the relevant difference appears in occupancy,
shared memory, memory throughput, warp execution behavior, instruction mix, or
math/special-function indicators. The profiler plan must avoid profiling all
`repeat=100` launches and must use kernel filtering or launch-count control.

## Evidence To Gain

Profiler before Submission 2 may answer:

- whether `impl=4` changes achieved occupancy or resource usage;
- whether shared-memory usage changes are visible and relevant;
- whether memory throughput differs between `impl=3` and `impl=4`;
- whether warp execution behavior differs;
- whether 2048 is limited by a different resource profile from 784 and 1024.

## Value For Final Claims

This helps final reporting because it separates:

- accepted speedup claims, which come from official timing;
- explanatory hypotheses, which require profiler or ablation;
- non-claims, including small slices and 2048.

The final report can then say either:

- profiler evidence supports a targeted Submission 2 direction, or
- profiler was unavailable/inconclusive and ablation was chosen instead.

No such statement should be made until profiler results are audited.

## Risk Introduced

Profiler risks:

- Nsight Compute may fail due to permissions or unsupported counters.
- Profiler runs may be slow.
- Metrics may be inconclusive.
- Profiler timing cannot be used for official speedup.
- Profiling all `repeat=100` launches would be wasteful and must be avoided.

Submission 2 risks if profiler-informed optimization is chosen:

- a targeted 2048-specific or resource-tuning candidate may improve one large
  slice but regress another.
- a profiler-informed change still consumes a submission budget slot.

## Decision Rule

Proceed in this order:

1. Ask human planner whether to approve an analysis-only profiler sbatch run.
2. If approved, run profiler with kernel filtering or launch-count control.
3. Generate and audit `profiler_summary.csv`.
4. After human audit of profiler results, choose Submission 2 based on accepted
   profiler evidence.
5. If profiler is unavailable, inconclusive, or not approved, consider
   Submission 2 for Option A reduction-structure ablation.
6. If human planner prefers minimizing risk over explanation, skip Submission 2
   and proceed to final confirmation with `impl=4` as the best candidate.

Current recommended choice:

- wait for explicit approval to run profiler-only analysis,
- defer Submission 2 until profiler results are audited.
