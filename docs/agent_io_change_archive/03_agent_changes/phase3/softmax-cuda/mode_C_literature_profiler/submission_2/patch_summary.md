# Submission 2 Patch Summary

Candidate:

- `impl=5_reduction_structure_ablation`

Exact new kernel name:

- `softMax5`

Source file changed:

- `/home/r14525078/HeCBench/src/softmax-cuda/main.cu`

Exact `impl=5` dispatch map:

| slice | dispatch |
|---:|---|
| 128 | unchanged `impl=1` / `softMax2` |
| 256 | unchanged `impl=1` / `softMax2` |
| 784 | `softMax5` |
| 1024 | `softMax5` |
| 2048 | `softMax5` |
| other non-official slices | unchanged `impl=1` / `softMax2` |

Dynamic shared memory:

- `impl=5` preserves the `impl=4` dynamic shared-memory allocation shape:
  `sizeof(float) * (sliceSize + 32)`.

Reduction implementation difference vs `impl=4`:

- `impl=4` uses warp shuffle reductions plus a compact cross-warp reduction.
- `impl=5` removes the warp-shuffle reduction path for large slices and uses
  shared-memory tree reductions over `exp_cache[0..255]`.

Reduction/shared-memory difference vs `impl=3`:

- `impl=3` uses `sizeof(float) * (sliceSize + BLOCK_SIZE)` and a separate
  `reduce` buffer of 256 floats.
- `impl=5` uses `sizeof(float) * (sliceSize + 32)` and reuses the first 256
  entries of `exp_cache` as temporary reduction scratch.
- Because the sum reduction overwrites `exp_cache[0..255]`, `impl=5`
  recomputes those first 256 cached exponentials before the output pass.

Synchronization differences:

- `impl=5` uses one synchronization after writing local maxima, then one
  synchronization after each full-block max-reduction stride.
- `impl=5` uses one synchronization after writing local sums, then one
  synchronization after each full-block sum-reduction stride.
- `impl=5` adds one synchronization after recomputing `exp_cache[0..255]`.
- This intentionally weakens the `impl=4` reduction path and may introduce
  overhead; it is a partial ablation, not a clean proof of causality.

Preserved unchanged:

- `impl=0`
- `impl=1`
- `impl=2`
- `impl=3`
- `impl=4`
- CPU reference
- correctness tolerance
- input generation
- official cases
- `numSlice`
- `repeat`

Interpretation limits:

- `impl=5` is a partial reduction-structure ablation.
- Strongest allowed attribution wording is "plausible contributor".
- Do not claim reduction structure, shared-memory footprint, or cached
  exponentials caused speedup.
