# Stage 0 Source Summary

Inspected files:
- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`
- `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/final/main.cu`
- `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C/main.cu`

Findings:
- Runtime source already includes `impl=0`, `impl=1`, `impl=2`, and `impl=3`.
- Runtime source is byte-identical to Mode B final source.
- The user-mentioned `mode_C/main.cu` is the original source with only the
  naive `impl=0` and warp-level `impl=1`; it is not the current runtime source
  and should not be used as the Mode C performance reference.
- `impl=3` matches Mode B final dispatch:
  - `sliceSize == 784 || 1024 || 2048`: dispatch to `impl=2` / `softMax3`.
  - other official slices 128 and 256: dispatch to unchanged `impl=1` /
    `softMax2`.

Relevant source structure:
- `softMax`: one thread per slice naive implementation.
- `softMax2`: warp-per-slice implementation using cooperative-groups warp
  reductions.
- `softMax3`: block-per-slice large-slice candidate using shared memory for
  exponentials plus a shared-memory reduction buffer.
- `kernel == 3`: shape-aware dispatch policy, not a new universal kernel.

No source file has been modified in Stage 0.
