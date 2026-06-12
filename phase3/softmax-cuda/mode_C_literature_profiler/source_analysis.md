# Mode C Stage 0 Source Analysis

## Answers Required by Stage 0

1. Does runtime source already include `impl=0/1/2/3`?

Yes. `/home/r14525078/HeCBench/src/softmax-cuda/main.cu` contains:
- `impl=0`: naive one-thread-per-slice kernel `softMax`.
- `impl=1`: warp-per-slice cooperative-groups kernel `softMax2`.
- `impl=2`: block-per-slice cached-exponential kernel `softMax3`.
- `impl=3`: shape-aware dispatch between unchanged `impl=1` and `impl=2`.

2. Does `impl=3` dispatch match Mode B final?

Yes. Runtime source is byte-identical to Mode B final source. `impl=3`
dispatches official small slices 128 and 256 to `impl=1`, and dispatches
official large slices 784, 1024, and 2048 to `impl=2`.

3. What are the current bottleneck hypotheses?

- Large slices likely spend time in global-memory reads/writes, `expf`, shared
  memory traffic, and two block-wide reductions.
- `softMax3` has several full-block `__syncthreads` barriers per row.
- `softMax3` caches all exponentials in shared memory, reducing duplicate
  `expf` work but adding shared-memory footprint and traffic.
- The fixed 256-thread block may not be ideal for every large slice.
- No profiler data exists yet, so these are hypotheses.

4. What optimization opportunities remain beyond `impl=3`?

- Add an `impl=4` large-slice candidate while preserving `impl=3`.
- Tune large-slice block size or introduce shape-specialized launches.
- Reduce block-reduction overhead by using warp-level reductions plus a smaller
  shared-memory cross-warp reduction.
- Test a no-cache or reduced-cache ablation to separate cached-exp contribution
  from block-per-slice row parallelism.
- Keep small slices on unchanged `impl=1` unless a separate low-risk improvement
  is justified later.

5. Which candidate will be attempted in Submission 1?

Proposed Submission 1 candidate: `impl=4_shape_specialized_large_reduce`.
It should keep official slices 128 and 256 on `impl=1`, and use a new
large-slice kernel for 784, 1024, and 2048. The first high-value change should
target the `softMax3` reduction path: use per-warp reductions and one compact
cross-warp reduction instead of full shared-memory reductions over 256 elements
at every stride.

6. Why is it expected to beat `impl=3`?

`impl=3` large slices are exactly `softMax3`. If `impl=4` performs the same
mathematical work but reduces synchronization and shared-memory reduction
traffic, it may lower per-row overhead while preserving the known large-slice
parallelism benefit. The expected benefit is most plausible for 784 and 1024,
where reduction overhead is a larger fraction of total row work than for 2048.

7. What literature, CUDA documentation, or known optimization principle supports
the hypothesis?

NVIDIA CUDA documentation identifies shared memory, synchronization, occupancy,
register pressure, and execution configuration as central performance factors.
The Best Practices Guide also recommends iterative optimization with validation.
This supports testing a targeted reduction/synchronization change, then using
paired timing and optional profiler evidence before making claims.

8. What are the risks?

- The candidate may increase register pressure enough to offset reduced
  synchronization.
- Warp-level reduction code may be less portable or may require careful
  cooperative-groups use.
- Removing or reducing cached exponentials may regress if `expf` recomputation
  dominates.
- Shape-specialized dispatch can improve one large slice while regressing
  another.
- Any correctness failure invalidates the candidate.

9. What exact source files would be changed?

If approved for Submission 1, only:
- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`

Mode B artifacts must not be modified.

10. Confirm no source file has been modified yet.

Confirmed. Stage 0 created only Mode C artifact markdown files under:
- `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler`

No `.cu` source file was modified for Mode C Stage 0.

## Source Consistency Note

The user-mentioned file
`/home/r14525078/HeCBench/phase3/softmax-cuda/mode_C/main.cu` is the original
kernel-only version and does not contain Mode B `impl=2/3`. The current runtime
source and Mode B final source do contain `impl=2/3` and match each other.

## Safe Optimization Directions

- Preserve `impl=1`, `impl=2`, and `impl=3`.
- Add `impl=4` for Mode C.
- Dispatch small official slices to unchanged `impl=1`.
- Focus on large-slice kernel internals and shape-specific policy.
- Preserve input generation, CPU reference, tolerance, official cases,
  `numSlice`, and `repeat`.
- Use sbatch only for all GPU benchmark runs.
