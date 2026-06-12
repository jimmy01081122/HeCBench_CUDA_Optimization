# Literature and Documentation Notes

Sources consulted:
- NVIDIA CUDA Programming Guide v13.3:
  https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
- NVIDIA CUDA C++ Best Practices Guide v13.3:
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

Relevant guidance:
- CUDA exposes thread blocks, warps, cooperative groups, shared memory, and
  synchronization primitives. `softMax2` already uses warp-level cooperative
  groups, while `softMax3` uses block-level shared memory and `__syncthreads`.
- The CUDA Best Practices Guide frames optimization as an iterative process:
  identify bottlenecks, apply a focused change, verify correctness and timing,
  then repeat. This supports the Mode C submission budget and ablation style.
- The guide's memory optimization sections emphasize global-memory access
  patterns, shared memory, registers, and occupancy as major CUDA performance
  factors.
- The execution-configuration sections identify occupancy, block size, shared
  memory use, and register pressure as factors that can change throughput.
- The instruction optimization sections warn that synchronization and arithmetic
  instruction choices matter; this is relevant because `softMax3` performs two
  block reductions, multiple barriers, and `expf` work.

Implications for this benchmark:
- The large-slice path has enough work per slice for block-level cooperation,
  but the current `softMax3` pays for repeated full-block reductions and stores
  every exponential to shared memory.
- The shared-memory footprint is `sizeof(float) * (sliceSize + BLOCK_SIZE)`;
  for `slice=2048`, this is 9216 bytes per block. That is not huge on V100, but
  it is large enough to be part of the occupancy/block-residency tradeoff.
- Because `expf` is relatively expensive, avoiding duplicated exponentials can
  help, but shared-memory caching can also add pressure and synchronization.
  Mode C should test changes rather than make causal claims without evidence.
- Shape-specific tuning is reasonable because official slices are fixed at
  128, 256, 784, 1024, and 2048, and Mode B already showed that one policy does
  not fit all slices.

Profiler note:
- Nsight Compute has not yet been run for this Mode C workflow. Until profiler
  data exists, bottleneck statements are hypotheses only.
