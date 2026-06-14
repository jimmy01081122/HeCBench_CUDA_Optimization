# softmax-cuda Mode B Round 1 Patch Summary

Approved candidate:
- `impl2_block_cached_exp_compound`

Source file changed:
- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`

Changes made:
- Added a new `softMax3` CUDA kernel.
- Added `implementation 2` usage text.
- Added a separate `kernel == 2` dispatch path.

Preserved unchanged:
- `impl=0` naive kernel and dispatch behavior.
- `impl=1` optimized kernel and dispatch behavior.
- CPU reference implementation.
- Correctness tolerance.
- Input generation.
- Official cases, `numSlice`, and `repeat`.

Candidate attribution:
- This candidate changes row parallelism from warp-per-slice to block-per-slice and changes computation/memory strategy from recomputing exponentials to shared-memory cached exponentials.
- Any performance result must be attributed to the compound candidate as a whole, not solely to cached exponentials.

Validation script:
- `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/run.slurm`

Profiler:
- Not run in Round 1 official timing.
- `profiler_status=NOT_RUN`.
