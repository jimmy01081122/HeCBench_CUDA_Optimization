You are a CUDA performance engineer executing Phase 3 Mode C for the HeCBench softmax-cuda experiment.

Mode C is an evidence-guided aggressive optimization workflow.

It combines:
1. aggressive CUDA optimization,
2. literature/documentation-informed hypotheses,
3. profiler evidence when available,
4. ablation-based attribution when feasible,
5. correctness-gated and auditor-checked validation.

Mode C is not explanation-only.
Mode C should actively attempt to improve beyond the accepted Mode B candidate impl=3.

However, all optimization claims must be supported by:
- official cases,
- paired baseline,
- correctness PASS,
- stable repeated measurements,
- raw outputs,
- auditor checks,
- and, when making causal claims, profiler or ablation evidence.

The primary comparison is:
  Mode C candidate vs Mode B impl=3

The secondary comparison is:
  Mode C candidate vs impl=1 baseline

Do not claim Mode C improvement unless speedup_vs_impl3 is valid.

# Project context

Repository root:
  /home/a/PP

Runtime benchmark source root:
  /home/r14525078/HeCBench/src/softmax-cuda

Phase 3 artifact root:
  /home/r14525078/HeCBench/phase3/softmax-cuda

Mode C output path:
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler

Current accepted Mode B candidate:
  impl3_shape_dispatch_impl1_small_impl2_large

Mode B final dispatch:
  slice=128  -> impl=1
  slice=256  -> impl=1
  slice=784  -> impl=2
  slice=1024 -> impl=2
  slice=2048 -> impl=2

Known implementations:
  impl=0:
    naive reference

  impl=1:
    existing optimized warp-level implementation

  impl=2:
    Round 1 compound candidate:
      block-per-slice + shared-memory cached exponentials

  impl=3:
    Mode B accepted shape-aware dispatcher:
      slice=128/256 -> impl=1
      slice=784/1024/2048 -> impl=2

Mode B final valid large-slice improvements against paired impl=1:
  slice=784:
    1.392x

  slice=1024:
    1.699x

  slice=2048:
    1.337x

Mode B non-speedup slices:
  slice=128:
    measurement-equivalent, no valid optimization speedup claim

  slice=256:
    measurement-equivalent, no valid optimization speedup claim

Mode B result type:
  PARAM_TUNE / SHAPE_AWARE_DISPATCH

Important boundaries:
  - impl=3 is not a universal kernel optimization.
  - impl=2 is not a universal replacement.
  - impl=0 -> impl=1 must never be counted as Phase 3 speedup.
  - slice=128 and slice=256 under impl=3 are measurement-equivalent, not speedup claims.
  - large-slice gains cannot be attributed solely to cached exponentials unless ablation supports it.
  - profiler was NOT_RUN in Mode B final, so existing Mode B report has no profiler-supported bottleneck conclusion.

Authoritative Mode B files:
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/results.csv
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/round_summary.md
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/main.cu
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/auditor_report.csv
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/contradiction_check.csv
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/round_summary.md
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/round_summary.md
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/patch_summary.md
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/patch_summary.md

# Mode C objectives

Mode C has two simultaneous objectives.

Objective 1: Aggressive optimization
  Try to improve beyond Mode B accepted impl=3 within at most 3 optimization submissions.

Objective 2: Evidence-supported explanation
  Use literature, profiler when available, and ablation when feasible to explain why a candidate improves, regresses, or fails to exceed impl=3.

A valid Mode C outcome may be:
  - additional speedup beyond Mode B,
  - explanation-only success,
  - partial success,
  - inconclusive,
  - or blocked.

# Hard rules

1. All GPU benchmark runs must use sbatch.
2. Do not run ./main or any GPU benchmark binary on the login node.
3. Do not run unrelated benchmarks.
4. Do not touch topk-cuda or shmembench-cuda.
5. Do not use impl=0 -> impl=1 as Phase 3 speedup.
6. Do not remove or skip official cases.
7. Do not shrink numSlice or repeat.
8. Do not change input generation.
9. Do not change CPU reference.
10. Do not loosen correctness tolerance.
11. Do not implement approximate softmax.
12. Do not delete slow, failing, or regressing cases.
13. correctness FAIL -> result invalid.
14. baseline invalid -> speedup=n/a.
15. improvement <1% -> MEASUREMENT_EQUIVALENT.
16. high CV -> speedup_claim_valid=false unless remeasured.
17. profiler unavailable is a limitation, not failure.
18. Do not claim profiler-supported conclusions unless profiler data exists.
19. Do not claim cached exponentials alone caused improvement unless ablation proves it.
20. Do not claim universal KERNEL_OPT unless all official cases improve or remain measurement-equivalent without regression and correctness PASS.
21. Do not promote a candidate if any official case has correctness FAIL.
22. Do not modify Mode B artifacts.
23. Prefer adding new implementation IDs rather than overwriting impl=1, impl=2, or impl=3.
24. If source unexpectedly differs from Mode B final source, stop and report.

# Submission budget

Mode C has a maximum of 3 optimization submissions, excluding pure inspection.

Stage 0:
  Inspection and plan only.
  No source modification.
  No benchmark run unless explicitly described as read-only metadata inspection.

Submission 1:
  Aggressive optimization candidate 1.

Submission 2:
  Correction, ablation, or second candidate based on Submission 1 evidence.

Submission 3:
  Final candidate confirmation.

Do not exceed 3 optimization sbatch submissions.

# Official cases

All official Mode C comparisons must use exactly:

  numSlice=100000, sliceSize=128, repeat=100
  numSlice=100000, sliceSize=256, repeat=100
  numSlice=100000, sliceSize=784, repeat=100
  numSlice=100000, sliceSize=1024, repeat=100
  numSlice=50000,  sliceSize=2048, repeat=100

Official baseline:
  impl=1

Mode B accepted candidate:
  impl=3

Mode C candidate:
  impl=4 or higher, unless modifying impl=3 is explicitly justified and approved.
  Prefer adding new implementation IDs rather than overwriting impl=1/2/3.

# Success criteria

A Mode C candidate is considered SUCCESS_WITH_ADDITIONAL_SPEEDUP only if:

1. All official cases correctness PASS.
2. All official cases are present.
3. Paired baseline, Mode B candidate, and Mode C candidate measurements are valid.
4. At least one official case improves over impl=3 by >=1%.
5. No official case regresses by >=1%, unless explicitly handled by shape-aware dispatch fallback.
6. Auditor PASS.
7. speedup_claim_valid=true for improved cases.
8. raw stdout/stderr preserved.
9. result is not based on profiler overhead timing.

If Mode C only explains Mode B but does not improve over impl=3:
  final label = SUCCESS_EXPLANATION_ONLY

If Mode C improves some cases but regresses or invalidates others:
  final label = PARTIAL_SUCCESS

If correctness fails:
  candidate invalid, but preserve all evidence.

# Required directory structure

Create only under:

  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler

Required:

  mode_C_literature_profiler/
    README.md
    plan.md
    literature_notes.md
    source_analysis.md
    stage_0_inspection/
      source_summary.md
      baseline_summary.md
    submission_1/
      plan.md
      patch_summary.md
      run.slurm
      results.csv
      raw/
      auditor_report.csv
      summary.md
    submission_2/
      plan.md
      patch_summary.md
      run.slurm
      results.csv
      raw/
      auditor_report.csv
      summary.md
    final/
      run.slurm
      final_results.csv
      raw/
      auditor_report.csv
      contradiction_check.csv
      mode_C_summary.md
      limitations.md

Do not modify Mode B artifacts.

# Stage 0: source analysis and optimization plan

First inspect only:
  - Mode B final results.csv
  - Mode B final round_summary.md
  - Mode B final main.cu
  - Round 1 and Round 2 summaries
  - Round 1 and Round 2 patch summaries
  - current runtime source in /home/r14525078/HeCBench/src/softmax-cuda/main.cu

Produce:
  mode_C_literature_profiler/source_analysis.md
  mode_C_literature_profiler/plan.md
  mode_C_literature_profiler/literature_notes.md

Stage 0 must answer:

1. Does runtime source already include impl=0/1/2/3?
2. Does impl=3 dispatch match Mode B final?
3. What are the current bottleneck hypotheses?
4. What optimization opportunities remain beyond impl=3?
5. Which candidate will be attempted in Submission 1?
6. Why is it expected to beat impl=3?
7. What literature, CUDA documentation, or known optimization principle supports the hypothesis?
8. What are the risks?
9. What exact source files would be changed?
10. Confirm no source file has been modified yet.

Stage 0 must also summarize known safe optimization directions.

Potential Mode C optimization directions include, but are not limited to:

A. Refine impl=2 large-slice path:
   - improve block reduction
   - reduce synchronization
   - reduce shared memory footprint
   - tune block size
   - reduce register pressure

B. Add specialized medium/large dispatch:
   - separate path for 784
   - separate path for 1024
   - separate path for 2048
   - use shape-specific block size or shared memory layout

C. Add impl=4 ablation or candidate:
   - block-per-slice without shared-memory cached exponentials
   - cached-exp with different block size
   - two-warps-per-slice or multi-warp strategy
   - reduce duplicate memory traffic

D. Improve small-slice handling only if safe:
   - do not risk breaking 128/256
   - if optimizing small slices, candidate must preserve or improve impl=1 behavior

Stage 0 output must end with:
  Proceeding to Submission 1 proposal.
No source change yet.

Stop after Stage 0 and wait for human approval before source modification or sbatch.

# Submission 1: aggressive but targeted candidate

Do not start Submission 1 until human approval is given after Stage 0.

Submission 1 should attempt one high-value candidate.

Preferred strategy:
  Add impl=4 or impl=5 rather than overwrite impl=3.

Possible candidate examples:
  - impl=4: shape-specialized large-slice kernel for 784/1024/2048
  - impl=4: optimized variant of impl=2 with block-size tuning
  - impl=4: reduce synchronization or shared memory overhead in large-slice path
  - impl=4: block-per-slice without cached exp for ablation and performance comparison
  - impl=4: separate kernels for 784/1024 and 2048 if justified

Submission 1 plan must include:
  1. hypothesis
  2. exact source change
  3. why it may beat impl=3
  4. expected per-slice effects
  5. risks
  6. validation plan
  7. rollback plan

After source change, run via sbatch.

For every official slice:
  - impl=1 paired baseline
  - impl=3 Mode B accepted candidate
  - impl=4 or current Mode C candidate

At least 3 trials per official slice and implementation.

Use interleaved order:
  impl=1
  impl=3
  impl=4

This allows:
  - speedup_vs_impl1
  - speedup_vs_impl3

Required CSV columns:

benchmark,mode,stage,submission_id,variant,impl,baseline_impl,modeB_impl,numSlice,sliceSize,repeat,trial_id,time_ms,baseline_impl1_time_ms,modeB_impl3_time_ms,speedup_vs_impl1,speedup_vs_impl3,correctness_status,measurement_validity,speedup_claim_valid,result_type,mean_ms,min_ms,max_ms,stddev_ms,cv,profiler_status,raw_stdout_path,raw_stderr_path,build_log_path,slurm_job_id,hostname,gpu_name,cuda_version,notes

Classification:
  - impl=1 rows -> BASELINE
  - impl=3 rows -> MODE_B_BASELINE
  - impl=4 rows -> MODE_C_CANDIDATE
  - correctness FAIL -> INVALID
  - speedup_vs_impl3 < 1.01 -> no additional Mode C speedup
  - speedup_vs_impl3 >= 1.01 and correctness PASS and CV stable -> Mode C improvement claim valid
  - speedup_vs_impl3 < 1.0 by >=1% -> REGRESSION
  - small slices 128/256 must not regress

After Submission 1:
  - run auditor
  - produce submission_1/summary.md
  - decide:
      accept for final
      revise in Submission 2
      rollback
      stop

# Submission 2: correction, ablation, or second candidate

Use Submission 2 only if needed and after human approval.

Allowed reasons:
  1. Submission 1 improved large slices but regressed small slices.
  2. Submission 1 improved one large slice but regressed another.
  3. Submission 1 failed correctness and needs a targeted fix.
  4. Submission 1 suggests ablation is needed to explain contribution.
  5. Submission 1 produced no improvement and a second candidate is justified.

Submission 2 must be narrower than Submission 1.
Do not introduce multiple unrelated changes.
Do not hide Submission 1 failure.

Run the same official cases:
  impl=1
  impl=3
  new Mode C candidate

At least 3 trials.

# Submission 3: final confirmation

Use the best valid Mode C candidate.

If no Mode C candidate beats impl=3:
  final confirmation should rerun impl=1 and impl=3 and state:
    Mode C produced no additional speedup beyond Mode B.

If a Mode C candidate beats impl=3:
  final confirmation must run:
    impl=1
    impl=3
    best Mode C candidate

Official cases:
  all five slices

At least 3 trials.

Final report must include:
  - speedup_vs_impl1
  - speedup_vs_impl3
  - correctness
  - measurement validity
  - result_type
  - profiler status
  - whether Mode C improved beyond Mode B
  - whether Mode C only explained Mode B

# Profiler policy

Profiler is optional but should be attempted if practical.

Attempt Nsight Compute at most once for representative cases:
  - slice=784 impl=3 and best Mode C candidate
  - slice=1024 impl=3 and best Mode C candidate
  - slice=2048 impl=3 and best Mode C candidate

If ncu unavailable:
  profiler_status=UNAVAILABLE
  record exact error
  do not retry repeatedly
  do not make profiler-supported claims

If profiler runs:
  collect, if available:
    achieved occupancy
    register count
    shared memory usage
    memory throughput
    warp execution efficiency
    instruction mix
    math / special-function related indicators if available

Profiler output is for explanation only.
Official speedup must come from normal timing runs without profiler overhead.

# Auditor requirements

Run self-consistency auditor after every submission and final confirmation.

Auditor must check:
  - all official cases present
  - correctness PASS before speedup claim
  - speedup_vs_impl3 only claimed when valid
  - small slices not incorrectly claimed as improved if same path
  - no impl=0 -> impl=1 speedup claim
  - profiler-supported claim only when profiler data exists
  - raw output paths exist
  - no missing official cases
  - no hidden regression
  - result_type consistent with evidence

# Final Mode C summary

Write:

  mode_C_literature_profiler/final/mode_C_summary.md

It must include:

1. Objective
2. Baseline and Mode B reference
3. Submission history
4. Best candidate
5. Final official results
6. speedup_vs_impl1
7. speedup_vs_impl3
8. correctness table
9. measurement validity table
10. profiler status
11. ablation or attribution conclusion
12. limitations
13. do-not-claim list
14. final label

Final label must be one of:

  SUCCESS_WITH_ADDITIONAL_SPEEDUP
  SUCCESS_EXPLANATION_ONLY
  PARTIAL_SUCCESS
  INCONCLUSIVE
  BLOCKED

# Final claim rules

If best candidate improves over impl=3 on large slices and does not regress small slices:
  Mode C may claim additional speedup.

If best candidate only matches impl=3:
  Mode C must claim no additional speedup beyond Mode B.

If profiler unavailable:
  profiler-supported explanation = NOT_AVAILABLE.

If ablation not run:
  cached-exp causality = NOT_PROVEN.

If ablation run and supports attribution:
  state only the supported attribution.

# Stop conditions

Stop immediately and report if:
  - source unexpectedly differs from Mode B final source
  - impl=3 dispatch no longer exists
  - official cases are missing
  - correctness fails in official timing
  - auditor reports contradiction
  - profiler unavailable and no timing evidence can be collected
  - any source modification occurs before Stage 0 approval

# Start now

Begin with Stage 0 inspection and plan.

Do not modify source before Stage 0 plan is written.
Do not submit sbatch before Stage 0 plan is reviewed and approved.