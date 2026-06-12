# Submission 1 Patch Summary

Candidate:
- `impl=4_shape_specialized_large_reduce`

Source file changed:
- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`

Changes made:
- Added `warp_reduce_max` and `warp_reduce_sum` device helpers using warp
  shuffle reductions.
- Added a new `softMax4` CUDA kernel for official large slices.
- Added `implementation 4` usage text.
- Added a separate `kernel == 4` dispatch path.

Dispatch behavior:
- `sliceSize == 784 || sliceSize == 1024 || sliceSize == 2048`: launch
  `softMax4`.
- Other slice sizes: dispatch to unchanged `impl=1` / `softMax2` behavior.

Preserved unchanged:
- `impl=0` kernel and dispatch behavior.
- `impl=1` kernel and dispatch behavior.
- `impl=2` kernel and dispatch behavior.
- `impl=3` dispatch behavior.
- CPU reference implementation.
- Correctness tolerance.
- Input generation.
- Official cases, `numSlice`, and `repeat`.

Candidate hypothesis:
- Preserve the `softMax3` large-slice block-per-slice and cached-exp structure,
  but reduce shared-memory tree reduction and synchronization overhead by using
  per-warp reductions plus compact cross-warp reduction.

Attribution limits:
- No profiler was run for this patch.
- Do not claim a profiler-supported bottleneck.
- Do not claim cached-exp causality without ablation evidence.
