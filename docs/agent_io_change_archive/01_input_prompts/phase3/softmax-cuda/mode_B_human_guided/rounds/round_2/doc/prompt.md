Prepare Mode B Round 2 proposal for softmax-cuda.

Do not submit sbatch yet.
Stop after writing the proposal and wait for human approval.

# Round 1 decision

Round 1 candidate:
- variant=impl2_block_cached_exp_compound

Decision:
- Rejected as a full replacement for impl=1.
- Accepted only as a partial large-slice candidate.
- slice=128 regressed.
- slice=256 had correctness failure.
- slice=784, 1024, and 2048 improved with correctness PASS.

Important attribution rule:
- impl=2 is a compound candidate combining block-per-slice row parallelism and shared-memory cached exponentials.
- Do not attribute its speedup solely to cached exponentials.

# Round 2 goal

Create a shape-aware dispatch candidate.

This round is not a correctness-fix round for impl=2.
Do not modify impl=2 to fix slice=256 in this round.
Any impl=2 correctness fix for slice=256 must be proposed as a separate round.

# Required dispatch policy

Create a new candidate implementation:

- variant=impl3_shape_dispatch_impl1_small_impl2_large
- candidate impl id: impl=3

Dispatch map:

- slice=128 -> use unchanged impl=1 path
- slice=256 -> use unchanged impl=1 path
- slice=784 -> use unchanged impl=2 path
- slice=1024 -> use unchanged impl=2 path
- slice=2048 -> use unchanged impl=2 path

If the input slice size is not one of the official slices, do not make unsupported performance claims. The proposal must specify how non-official slice sizes are handled, but only official slices are used for Phase 3 evaluation.

# Required proposal contents

1. Explain why this dispatch follows Round 1 evidence:
   - impl=2 regressed on slice=128.
   - impl=2 was invalid on slice=256 due to correctness failure.
   - impl=2 was valid and faster on slice=784, 1024, and 2048.
   - Therefore, shape-aware dispatch preserves impl=1 for small or invalid slices and uses impl=2 only for validated large slices.

2. Confirm source-change scope:
   - impl=1 kernel remains unchanged.
   - impl=2 kernel remains unchanged.
   - CPU reference remains unchanged.
   - Correctness tolerance remains unchanged.
   - Input generation remains unchanged.
   - Official cases remain unchanged.
   - numSlice remains unchanged.
   - repeat remains unchanged.
   - No approximate softmax.
   - No skipped official cases.

3. Define candidate label:
   - variant=impl3_shape_dispatch_impl1_small_impl2_large
   - result interpretation: shape-aware dispatch candidate

4. Execution rule:
   - All build and benchmark executions must be submitted through sbatch.
   - Do not run ./main or any GPU benchmark binary on the login node.
   - Do not submit sbatch until human approval is recorded.

5. Validation plan:
   - paired impl=1 baseline
   - candidate impl=3 shape-aware dispatch
   - all official slices:
     - 128
     - 256
     - 784
     - 1024
     - 2048
   - at least 3 independent trials for every slice and every compared implementation
   - raw stdout and stderr for every trial
   - same CSV schema as Round 1
   - add column: dispatch_selected_impl
   - run self-consistency auditor
   - preserve auditor output

6. Required CSV fields:
   Use the Round 1 schema and include at least:

   benchmark
   mode
   round_id
   human_decision
   variant
   impl
   dispatch_selected_impl
   baseline_impl
   numSlice
   sliceSize
   repeat
   trial_id
   time_ms
   baseline_time_ms
   speedup_vs_impl1
   correctness_status
   measurement_validity
   speedup_claim_valid
   result_type
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

7. Classification rules:
   - correctness FAIL -> measurement_validity=INVALID and speedup_claim_valid=false.
   - If any official slice FAILs, candidate is not a full success.
   - baseline invalid or missing -> speedup=n/a and speedup_claim_valid=false.
   - improvement <1% -> MEASUREMENT_EQUIVALENT.
   - slower by >=1% -> REGRESSION.
   - high CV -> CAUTION or NOISY; do not claim speedup unless remeasured.
   - slice=128 and slice=256 are expected to dispatch to impl=1; measurement-equivalent results are acceptable.
   - For slice=128 and slice=256, do not claim kernel optimization if the dispatch selects impl=1.
   - For slice=784, 1024, and 2048, speedup may be claimed only if correctness PASS, baseline valid, raw output exists, and repeated timing is stable.
   - Overall candidate result_type should be PARAM_TUNE or shape-aware dispatch. Do not label the whole candidate as a universal KERNEL_OPT if the implementation only selects between unchanged impl=1 and impl=2.
   - Per-slice large-shape improvements may be reported separately.

8. Profiler policy:
   - Profiler is not required in Round 2.
   - If not run, set profiler_status=NOT_RUN.
   - Do not make profiler-supported conclusions without profiler data.
   - Profiler unavailability or absence is a limitation, not a failure.

9. Reporting rules:
   - Report per-slice results.
   - Do not hide slice=128 or slice=256.
   - Do not report only aggregate speedup.
   - If aggregate speedup is reported, it must be accompanied by the full per-slice table and dispatch map.
   - Do not reinterpret Round 1 as full success.
   - Do not claim impl=2 universally improves softmax-cuda.

Stop after producing the Round 2 proposal.
Wait for human approval before any source modification, build, or sbatch submission.