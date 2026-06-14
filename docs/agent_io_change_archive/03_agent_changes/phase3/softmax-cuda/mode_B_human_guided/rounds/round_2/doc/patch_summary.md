# softmax-cuda Mode B Round 2 Patch Summary

Approved candidate:
- `impl3_shape_dispatch_impl1_small_impl2_large`

Source file changed:
- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`

Round 2 change:
- Added a separate `impl=3` dispatch path.
- `sliceSize == 128 || sliceSize == 256`: dispatches to unchanged `impl=1`/`softMax2` behavior.
- `sliceSize == 784 || sliceSize == 1024 || sliceSize == 2048`: dispatches to unchanged `impl=2`/`softMax3` behavior.
- Non-official slice sizes fall back to unchanged `impl=1`/`softMax2`.

Preserved unchanged:
- `impl=0`
- `impl=1`
- `impl=2`
- CPU reference
- correctness tolerance
- input generation
- official cases
- `numSlice`
- `repeat`

Interpretation:
- This is a shape-aware dispatch candidate, not a universal kernel optimization.
- Per-slice large-shape improvements may be reported separately if valid.
- `slice=128` and `slice=256` are expected to be measurement-equivalent because they intentionally select `impl=1`.

Profiler:
- Not run in Round 2 official timing.
- `profiler_status=NOT_RUN`.
