# Final Profiler Interpretation

Evidence label: `LIMITED_PROFILER_EVIDENCE`.

Final confirmation speedups are unchanged:

- slice 784: speedup_vs_impl3=1.135540
- slice 1024: speedup_vs_impl3=1.048740
- slice 2048: speedup_vs_impl3=1.008239, below the 1% claim gate

## What Profiler Supports

- For slices 784, 1024, and 2048, `impl=4` and `impl=3` show the same registers/thread in the collected launch-resource metrics.
- For slices 784 and 1024, `impl=4` and `impl=3` show the same waves/SM in the collected launch-resource metrics.
- For slice 2048, `impl=4` and `impl=3` also show the same waves/SM in the collected launch-resource metrics.
- For all profiled slices, `impl=4` shows lower dynamic shared-memory allocation than `impl=3`.

## What Profiler Does Not Support

- It does not support a causal claim that shared-memory footprint caused the speedup.
- It does not support reduction-structure causality.
- It does not support cached-exp causality.
- It does not provide memory-throughput, warp-efficiency, instruction-mix, math/special-function, or scheduler/stall evidence.
- It does not provide official timing evidence.

## Slice 784

Profiler evidence is consistent with `impl=4` having reduced dynamic shared-memory allocation while preserving the same collected registers/thread and waves/SM as `impl=3`. This can be described only as limited profiler evidence accompanying the accepted official speedup; it does not identify the mechanism.

## Slice 1024

Profiler evidence follows the same pattern as slice 784: lower dynamic shared-memory allocation for `impl=4`, with matching collected registers/thread and waves/SM. This is limited supporting context for the accepted official speedup, not a causal explanation.

## Slice 2048

Profiler evidence also shows lower dynamic shared-memory allocation for `impl=4`, but final confirmation classified 2048 as measurement-equivalent because speedup_vs_impl3=1.008239 is below the 1% claim gate. Because the same resource-allocation pattern appears without an accepted speedup, the profiler does not explain why 784/1024 improve while 2048 is measurement-equivalent.

## Further Ablation

Further ablation would be needed for mechanism attribution. The blocked `impl=5` ablation must not be promoted and does not provide accepted attribution evidence.

## Paper-Safe Wording

Nsight Compute launch-resource diagnostics show that `impl=4` reduces dynamic shared-memory allocation relative to `impl=3` for the profiled large-slice cases, while the collected register and waves-per-SM metrics remain unchanged. These diagnostics provide limited context for the confirmed 784 and 1024 speedups, but they do not establish the underlying cause.

## Do-Not-Claim List

- Do not claim profiler timing as official timing.
- Do not claim shared-memory footprint caused the speedup.
- Do not claim reduction structure caused the speedup.
- Do not claim cached-exp contribution.
- Do not claim profiler evidence explains the 2048 measurement-equivalent result.
- Do not promote `impl=5`.
