Mode C Submission 1 has been reviewed by the human planner.

Decision:
  ACCEPT_WITH_LIMITATIONS

Submission 1 final label:
  SUCCESS_WITH_ADDITIONAL_SPEEDUP

Accepted additional-speedup claims:
  - slice=784: 1.131x vs impl=3
  - slice=1024: 1.049x vs impl=3

Not accepted as additional speedup:
  - slice=128
  - slice=256
  - slice=2048

Important constraints:
  - Do not claim impl=4 is universal kernel optimization.
  - Do not claim slice=128 has Mode C speedup.
  - Do not claim slice=256 has Mode C speedup.
  - Do not claim slice=2048 has Mode C speedup.
  - Do not claim profiler-supported bottleneck because profiler_status=NOT_RUN.
  - Do not claim cached-exp or warp-reduction causality without profiler or ablation evidence.
  - Do not use speedup_vs_impl1 as the main Mode C success metric.
  - Main Mode C metric remains speedup_vs_impl3.

Next step:
  Do not start Submission 2 yet.

Create an analysis-only track:

  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1

This analysis track does not count as an optimization submission as long as:
  - no source code is modified
  - no new optimization candidate is introduced
  - no official speedup is claimed from profiler timing
  - all profiler or diagnostic execution, if any, uses sbatch
  - no login-node benchmark execution occurs

Purpose:
  Use Submission 1 evidence to decide whether Submission 2 should be:
    A. ablation
    B. profiler-informed optimization
    C. 2048-specific optimization
    D. final confirmation without more optimization
    E. stop and report Submission 1 as final Mode C result

Required outputs:

1. Create:
   analysis_track_1/submission_1_result_review_zh.md

   This Chinese report must summarize:
   - candidate: impl4_shape_specialized_large_reduce
   - accepted additional speedup:
     - slice=784: 1.131x vs impl=3
     - slice=1024: 1.049x vs impl=3
   - not accepted:
     - slice=128: no Mode C speedup claim
     - slice=256: no Mode C speedup claim
     - slice=2048: measurement-equivalent
   - correctness status:
     all official slices PASS
   - auditor status:
     PASS
   - final label:
     SUCCESS_WITH_ADDITIONAL_SPEEDUP, with limitations
   - limitations:
     - profiler NOT_RUN
     - ablation NOT_RUN
     - no causal attribution
     - impl=4 source patch not independently explained in causal terms
   - do-not-claim list

2. Create:
   analysis_track_1/profiler_feasibility_check.md

   Inspect only. Do not run benchmark binaries on login node.

   Check whether profiling is feasible:
   - which ncu
   - ncu --version, if safe
   - module availability, if safe
   - whether prior logs show profiler permission issues

   Report:
   - profiler_status_candidate:
     AVAILABLE / UNAVAILABLE / UNKNOWN
   - whether profiler requires sbatch
   - whether profiler should be attempted before Submission 2
   - exact reason if profiler seems unavailable
   - confirmation that profiler timing will not be used for official speedup

3. Create:
   analysis_track_1/profiler_plan.md

   If profiler appears feasible, propose an sbatch-only profiler plan.

   Recommended profiler comparison:
   - slice=784:
     impl=3 vs impl=4
   - slice=1024:
     impl=3 vs impl=4
   - slice=2048:
     impl=3 vs impl=4

   Profiler purpose:
   - understand why impl=4 improves 784 and 1024
   - understand why 2048 is only measurement-equivalent
   - inform whether Submission 2 should target reduction overhead, shared memory footprint, block size, or 2048-specific behavior

   Profiler metrics to collect if available:
   - achieved occupancy
   - register usage
   - shared memory usage
   - memory throughput
   - warp execution efficiency
   - instruction mix
   - math or special-function indicators if available

   Required rule:
   - profiler run is for explanation only
   - official speedup remains based on normal timing results
   - if profiler unavailable, record as limitation and continue with ablation planning

4. Create:
   analysis_track_1/ablation_plan.md

   Propose ablation options for Submission 2.

   The plan must compare possible directions:

   Option A: reduction-structure ablation
     Purpose:
       determine whether impl=4 improvement is due to warp/cross-warp reduction changes compared with impl=2.

   Option B: cached-exp attribution ablation
     Purpose:
       determine whether cached exponentials independently contribute to large-slice improvement.

   Option C: 2048-specific optimization
     Purpose:
       investigate why slice=2048 only reaches speedup_vs_impl3=1.008x and whether a separate path can improve it.

   Option D: block-size / resource tuning
     Purpose:
       tune large-slice path without changing small-slice behavior.

   For each option, report:
   - required source change
   - what hypothesis it tests
   - target slices
   - expected benefit
   - correctness risk
   - regression risk
   - whether it should count as Submission 2
   - whether it is worth doing

5. Create:
   analysis_track_1/direction_decision.md

   Recommend one Submission 2 direction.

   Must choose one of:
   - Submission 2 = ablation
   - Submission 2 = profiler-informed optimization
   - Submission 2 = 2048-specific optimization
   - Submission 2 = full-slice experiment
   - no Submission 2; proceed to final confirmation

   The recommendation must justify:
   - why this is better than blind tuning
   - how it builds on Submission 1
   - what evidence will be gained
   - whether it helps final paper claims
   - what risk it introduces

6. Create:
   analysis_track_1/main_planner_questions.md

   If there are unclear choices requiring human decision, list them explicitly.

Rules:
  - Do not modify source.
  - Do not create a new optimization implementation.
  - Do not run ./main on login node.
  - Do not submit optimization sbatch.
  - Profiler sbatch may be proposed but not executed unless explicitly approved.
  - Stop after writing all analysis_track_1 documents.