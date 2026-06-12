# Direction Decision After Profiler Rerun

Recommendation: `Submission 2 = reduction-structure ablation`

Submission 2 remains deferred until human audit of profiler artifacts.

## Evidence Status

- profiler rows: 6
- AVAILABLE rows: 6
- PARTIAL rows: 0
- FAILED/UNAVAILABLE rows: 0
- official_timing_used=false for all rows

## Previous Failure Recorded

- The prior feasibility test failed because the binary was missing on the compute node.
- The prior profiler job was PARTIAL because direct stdout did not expose metrics from the selected sections.

## Rationale

Corrected profiler rerun produced resource metrics for all six large-slice impl=3/4 cases. The next safest evidence-building step is a reduction-structure ablation, unless human audit identifies a more specific resource-driven target.

## Guardrails

- Do not use profiler timing as official timing.
- Do not overclaim profiler-supported causality before human audit.
- Do not start Submission 2 without explicit human approval.
