Profiler rerun audit: ACCEPT_WITH_LIMITATIONS.

Accepted:
- Profiler rerun job 950686 was analysis-only.
- No source modification occurred.
- No new optimization candidate was added.
- No Submission 2 was started.
- ncu reports were generated and imported.
- Selected launch/resource metrics were recovered.
- impl=4 shows lower dynamic shared memory per block than impl=3 on 784, 1024, and 2048.
- registers/thread and waves/SM are unchanged.

Limitations:
- No memory throughput metrics were recovered.
- No warp execution efficiency metrics were recovered.
- No instruction mix metrics were recovered.
- No math / special-function metrics were recovered.
- No scheduler / stall breakdown was recovered.
- Profiler timing remains diagnostic only and must not be used for official speedup.
- Profiler evidence does not prove reduction-overhead causality.
- Profiler evidence does not prove shared-memory causality.
- Profiler evidence does not prove why 2048 is measurement-equivalent.

Decision:
- Profiler evidence status = LIMITED.
- Submission 2 is NOT approved yet.
- Do not modify source.
- Do not submit optimization sbatch.
- Do not start Submission 2.

Next task:
Prepare a Mode C Submission 2 Reduction-Structure Ablation Proposal.

The proposal must be inspection/planning only.
No source modification is allowed yet.
No sbatch submission is allowed yet.

Required candidate:
- impl=5_reduction_structure_ablation

Goal:
- Test whether the reduction-structure difference between impl=3/softMax3 and impl=4/softMax4 plausibly explains the accepted 784/1024 additional speedups.

Must preserve:
- impl=0
- impl=1
- impl=2
- impl=3
- impl=4
- CPU reference
- correctness tolerance
- input generation
- official cases
- numSlice
- repeat

Required comparison set:
- impl=1 baseline
- impl=3 Mode B baseline
- impl=4 Submission 1 candidate
- impl=5 ablation candidate

Required official cases:
- slice=128
- slice=256
- slice=784
- slice=1024
- slice=2048

Required metrics:
- speedup_vs_impl1
- speedup_vs_impl3
- speedup_vs_impl4
- correctness_status
- measurement_validity
- speedup_claim_valid
- result_type
- mean/min/max/stddev/CV
- raw stdout/stderr paths
- auditor output

Expected classification:
- 128/256:
  MEASUREMENT_EQUIVALENT unless a new safe small-slice path is explicitly introduced and validated.
  No speedup claim expected.

- 784/1024/2048:
  ABLATION_ONLY unless impl=5 also validly improves over impl=3 or impl=4.

Ablation interpretation rules:
- If impl=5 matches impl=4, reduction structure is a plausible contributor, not a proven sole cause.
- If impl=5 matches impl=3, reduction structure alone is not supported as the contributor.
- If impl=5 is worse than both, the reduction-structure hypothesis is weakened or the ablation introduces overhead.
- No causal claim is allowed without careful comparison, correctness PASS, stable timing, and auditor PASS.
- Do not call the ablation a new optimization unless it improves over impl=3 and impl=4 with correctness PASS and stable CV.
- Do not use profiler timing as official speedup.
- Do not use speedup_vs_impl1 as the main Mode C success metric.

Proposal must include:
1. Candidate design.
2. Exact source-level changes.
3. What is kept identical to impl=4.
4. What is intentionally changed relative to impl=4.
5. What is kept identical to impl=3 or impl=2.
6. Hypothesis being tested.
7. Expected outcomes and interpretation table.
8. Validation plan.
9. CSV schema.
10. Auditor plan.
11. Risks.
12. Rollback plan.
13. Human approval checkpoint.

Stop after producing the proposal.
Wait for human approval.