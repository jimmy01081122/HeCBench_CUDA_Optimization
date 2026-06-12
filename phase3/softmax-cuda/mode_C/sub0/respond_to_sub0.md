Mode C Stage 0 review result: NEEDS_REVISION before Submission 1 execution.

Revise the Submission 1 proposal as follows.

# Mode C Submission 1 Revised Proposal

## Goal

Attempt to improve beyond the accepted Mode B candidate:

- Mode B accepted candidate: impl=3
- variant: impl3_shape_dispatch_impl1_small_impl2_large

Primary comparison:
- Mode C candidate impl=4 vs Mode B impl=3

Secondary comparison:
- Mode C candidate impl=4 vs impl=1

A Mode C speedup claim is allowed only if speedup_vs_impl3 is valid.

## Candidate

Candidate:
- impl=4_shape_specialized_large_reduce

Source-change scope:
- Add a new impl=4 path in:
  /home/r14525078/HeCBench/src/softmax-cuda/main.cu

Preserve unchanged:
- impl=0
- impl=1
- impl=2
- impl=3
- CPU reference
- correctness tolerance
- input generation
- official cases
- numSlice
- repeat

Do not modify Mode B artifacts.

No source modification may occur until explicit human approval is recorded.

## Dispatch policy

For official slices:

- slice=128:
  impl=4 dispatches to unchanged impl=1 behavior.
  Expected result vs impl=3: MEASUREMENT_EQUIVALENT.
  No Mode C speedup claim is allowed.

- slice=256:
  impl=4 dispatches to unchanged impl=1 behavior.
  Expected result vs impl=3: MEASUREMENT_EQUIVALENT.
  No Mode C speedup claim is allowed.

- slice=784:
  impl=4 uses the new large-slice reduction candidate.

- slice=1024:
  impl=4 uses the new large-slice reduction candidate.

- slice=2048:
  impl=4 uses the new large-slice reduction candidate.

Non-official slices:
- fallback to unchanged impl=3 or impl=1 must be specified.
- No performance claims are allowed for non-official slices.

## Hypothesis

The current Mode B large-slice path uses impl=2 / softMax3.
It performs block-per-slice computation with shared-memory cached exponentials and block-level shared-memory reductions.

The impl=4 hypothesis is:
- keep the same mathematical softmax semantics,
- preserve large-slice block-level parallelism,
- reduce shared-memory reduction and synchronization overhead by using per-warp reductions plus compact cross-warp reduction.

This is a hypothesis only.
Do not claim this is a proven bottleneck without profiler or ablation evidence.

## Execution rules

After human approval only:

- Build must run through sbatch.
- Benchmark must run through sbatch.
- Do not run ./main or any GPU benchmark binary on the login node.
- Do not skip official cases.
- Do not delete slow, failing, or regressing cases.

## Validation plan

For every official slice:
- 128
- 256
- 784
- 1024
- 2048

Run paired implementations:
- impl=1
- impl=3
- impl=4

Use at least 3 independent trials per slice per implementation.

Use interleaved ordering per slice/trial when practical:
- impl=1
- impl=3
- impl=4

Save:
- raw stdout for every trial
- raw stderr for every trial
- build log
- Slurm stdout/stderr
- environment metadata
- Slurm job id

Generate:
- submission_1/results.csv
- submission_1/auditor_report.csv
- submission_1/contradiction_check.csv
- submission_1/summary.md
- submission_1/patch_summary.md

## Required CSV schema

submission_1/results.csv must include at least:

benchmark
mode
stage
submission_id
variant
impl
baseline_impl
mode_b_impl
dispatch_selected_impl
numSlice
sliceSize
repeat
trial_id
time_ms
impl1_time_ms
impl3_time_ms
speedup_vs_impl1
speedup_vs_impl3
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

Required values:
- benchmark=softmax-cuda
- mode=Mode_C
- submission_id=1
- baseline_impl=1
- mode_b_impl=3
- profiler_status=NOT_RUN unless profiler is explicitly run

## Measurement validity rules

- correctness FAIL -> INVALID and speedup_claim_valid=false.
- missing raw output -> LIMITED and speedup_claim_valid=false.
- missing impl=3 baseline -> INVALID and no Mode C speedup claim.
- improvement <1% vs impl=3 -> MEASUREMENT_EQUIVALENT.
- slower by >=1% vs impl=3 -> REGRESSION.
- high CV -> CAUTION or NOISY; do not claim speedup unless remeasured.

CV thresholds:
- CV <= 5%: VALID
- 5% < CV <= 15%: CAUTION
- CV > 15%: NOISY

## Mode C claim rules

Primary claim uses:
- speedup_vs_impl3

Secondary reporting may include:
- speedup_vs_impl1

Do not claim Mode C improvement if only speedup_vs_impl1 improves.
Mode C additional speedup requires valid speedup_vs_impl3.

Small slices:
- slice=128 and slice=256 must not be reported as Mode C optimization speedups if they dispatch to unchanged impl=1.
- If small slices deviate from impl=3 by >=1%, classify conservatively and investigate.

Large slices:
- slice=784, 1024, and 2048 may claim Mode C improvement only if:
  - correctness PASS,
  - speedup_vs_impl3 >= 1.01,
  - CV is stable,
  - raw output exists,
  - auditor passes.

Do not use aggregate speedup to hide per-slice regression.

## Profiler policy

Profiler is optional for Submission 1.

Official timing must be collected without profiler.

If profiler is not run:
- profiler_status=NOT_RUN
- no profiler-supported bottleneck conclusion is allowed

If profiler is unavailable:
- profiler_status=UNAVAILABLE
- this is a limitation, not a failure

If profiler is run:
- save ncu output
- distinguish profiler timing from official benchmark timing
- do not use profiler timing for official speedup

## Auditor rules

Run self_consistency_auditor.py after results.csv is generated.

Auditor must check:
- all official cases present
- correctness PASS before speedup claim
- speedup_vs_impl3 exists for impl=4 rows
- small slices not claimed as kernel improvements
- no profiler-supported claim without profiler data
- no cached-exp causality claim without ablation
- no aggregate-only success hiding per-slice regression

If auditor fails:
- do not promote impl=4
- keep impl=3 as accepted best candidate

## Final label criteria

SUCCESS_WITH_ADDITIONAL_SPEEDUP:
- all official slices correctness PASS,
- all official cases present,
- auditor PASS,
- no hidden regression,
- at least one large slice has valid speedup_vs_impl3 >= 1.01,
- no large-slice regression vs impl=3,
- small slices are measurement-equivalent or valid.

PARTIAL_SUCCESS:
- correctness PASS for all official slices,
- some large slices improve vs impl=3,
- but one or more large slices are measurement-equivalent or regress,
- no invalid correctness result.

INCONCLUSIVE:
- correctness PASS,
- measurements unstable or CV noisy,
- no reliable speedup_vs_impl3 claim.

BLOCKED:
- build failure,
- runtime failure,
- missing official cases,
- missing impl=3 baseline,
- missing raw output,
- auditor failure.

If impl=4 does not validly improve over impl=3:
- accepted best candidate remains Mode B impl=3.

## Reporting rules

Report per-slice results first.
Do not report only aggregate speedup.
Do not claim impl=4 is universal unless all official slices support it.
Do not claim cached-exp causality without ablation.
Do not claim profiler-supported bottleneck without profiler data.
Do not reinterpret Mode B speedups as Mode C speedups.

Stop after revising the proposal.
Do not modify source or submit sbatch until explicit human approval is given.