Approved for Submission 1 execution after human approval.

Execution conditions:
1. Do not modify impl=0/1/2/3.
2. Add impl=4 only as an additive candidate.
3. Use impl=3 as the primary Mode B baseline.
4. Compute and report speedup_vs_impl3 for every impl=4 official-case row.
5. Report per-slice results first.
6. Do not use aggregate speedup to hide any per-slice regression.
7. Do not claim profiler-supported bottleneck if profiler_status is NOT_RUN or UNAVAILABLE.
8. Do not claim cached-exp causality without ablation evidence.
9. If impl=4 does not validly improve over impl=3, keep impl=3 as the accepted best candidate.
10. Stop after Submission 1 results and wait for human audit before any further submission.