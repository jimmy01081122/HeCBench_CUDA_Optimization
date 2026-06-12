# Mode C Final Confirmation Proposal

## Objective

Confirm the current best Mode C candidate:

- `impl=4_shape_specialized_large_reduce`

This final confirmation uses no new optimization candidate, no source modification, no `impl=5` repair, and no profiler run.

## Candidate Scope

Final candidate:

- `impl=4_shape_specialized_large_reduce`

Compared implementations:

- `impl=1`: baseline
- `impl=3`: Mode B baseline
- `impl=4`: current best Mode C candidate

Explicit `impl=5` rule:

- Do not run `impl=5` in final confirmation.
- Do not promote `impl=5`.
- Mention `impl=5` only as a blocked ablation artifact in narrative.

## Execution Plan

For every official slice and every implementation `impl=1`, `impl=3`, and `impl=4`, run at least 3 independent trials.

Official slices:

- 128
- 256
- 784
- 1024
- 2048

Use interleaved order per slice/trial when practical:

```text
impl=1, impl=3, impl=4
```

Execution constraints:

- Use `sbatch` only.
- Do not run `./main` on the login node.
- Preserve raw stdout/stderr for every trial.
- Do not skip official cases.
- Do not use profiler timing.
- `official_timing_used=true` for official timing rows.
- `profiler_status=NOT_RUN` for official timing rows.

## Results Schema

`final_confirmation/results.csv` must include:

```text
benchmark
mode
stage
candidate
variant
impl
baseline_impl
mode_b_impl
mode_c_impl
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
correctness_status
measurement_validity
speedup_claim_valid
result_type
mode_c_final_label
final_confirmation_status
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
official_timing_used
notes
```

## Result Classification

Allowed `mode_c_final_label`:

- `SUCCESS_WITH_ADDITIONAL_SPEEDUP`
- `PARTIAL_SUCCESS`
- `INCONCLUSIVE`
- `BLOCKED`

Allowed `final_confirmation_status`:

- `CONFIRMED`
- `PARTIAL_CONFIRMATION`
- `INCONCLUSIVE`
- `BLOCKED`

Allowed `result_type`:

- `BASELINE` for `impl=1`
- `MODE_B_BASELINE` for `impl=3`
- `MODE_C_CANDIDATE` for `impl=4` rows with valid large-slice additional speedup claims
- `MEASUREMENT_EQUIVALENT` for rows where no valid additional Mode C speedup claim is allowed

## Claim Gates

Small slices:

- Do not claim small-slice Mode C speedup for 128 or 256.
- Under `impl=4`, these slices dispatch to unchanged `impl=1` and must be classified as `MEASUREMENT_EQUIVALENT` with `speedup_claim_valid=false`.

Large slices:

For slices 784, 1024, and 2048, final Mode C speedup claim is valid only if:

- `correctness_status=PASS`
- `measurement_validity=VALID`
- `speedup_vs_impl3 >= 1.01`
- raw stdout/stderr exist
- auditor PASS
- `speedup_claim_valid=true`

If 2048 remains below 1% improvement vs `impl=3`, classify it as `MEASUREMENT_EQUIVALENT` and do not claim additional speedup.

CV thresholds:

- `CV <= 5%`: `VALID`
- `5% < CV <= 15%`: `CAUTION`
- `CV > 15%`: `NOISY`; no speedup claim unless remeasured

## Auditor Rules

Auditor must check:

- all official cases are present
- `impl=1`, `impl=3`, and `impl=4` rows exist for every official slice
- raw stdout/stderr paths exist
- correctness PASS before any speedup claim
- `speedup_vs_impl3` exists for `impl=4` rows
- 128/256 are not claimed as speedup
- `speedup_vs_impl1` is not used as the main Mode C metric
- `profiler_status=NOT_RUN` for official timing rows
- `official_timing_used=true` for official timing rows
- no `impl=5` promotion
- no profiler-supported causality claim
- no cached-exp, shared-memory-footprint, or reduction-structure causality claim
- no aggregate-only success hiding per-slice regression

## Required Final Summary

`final_confirmation/summary.md` must include:

- per-slice table
- `speedup_vs_impl3`
- `speedup_vs_impl1`
- correctness
- CV
- `result_type`
- `mode_c_final_label`
- `final_confirmation_status`
- accepted claims
- rejected claims
- do-not-claim list

## Interpretation Limits

Accepted wording may confirm measured per-slice speedup only where claim gates pass.

Do not claim:

- reduction structure caused speedup
- shared-memory footprint caused speedup
- cached-exp contribution
- profiler-supported causality
- `impl=5` is promotable
- small-slice Mode C improvement for 128/256

## Approval

Approved for one final confirmation sbatch execution after saving this proposal document.
