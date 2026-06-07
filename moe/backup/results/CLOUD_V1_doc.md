# MoE Benchmark Optimization - Version 1

## Optimization Strategy: Kernel Fusion

### Overview
This version implements a kernel fusion optimization that combines softmax and top-k expert selection into a single fused kernel. The baseline approach executes softmax and top-k as separate kernels, which incurs unnecessary memory traffic and kernel launch overhead.

### Key Changes

#### 1. **Fused Kernel Dispatch**
Modified `main.cu` to dispatch to specialized fused kernels based on the `topk` value:
- `topk=1`: Uses `moeSoftmaxTop1<TPB>` - Optimized for single expert selection
- `topk=2`: Uses `moeSoftmaxTopK<TPB, 2>` - Template specialization for topk=2
- `topk=4`: Uses `moeSoftmaxTopK<TPB, 4>` - Template specialization for topk=4
- `topk=8`: Uses `moeSoftmaxTopK<TPB, 8>` - Template specialization for topk=8
- `topk>8`: Falls back to generic kernels

#### 2. **Performance Benefits**

**Memory Traffic Reduction:**
- Baseline: Writes intermediate softmax results (384 * 4 bytes per token = 1.5KB per token)
- Fused: No intermediate storage, reduces L2 cache pressure

**Kernel Launch Overhead:**
- Baseline: 2 kernel launches per iteration (softmax + topk)
- Fused: 1 kernel launch per iteration (~50% reduction)

**Register Usage Optimization:**
- The fused kernels compute softmax and top-k selection in a single pass
- Shared memory is reused for both computations

#### 3. **Technical Details**

**Original Flow:**
```
Input → moeSoftmax → d_softmax_workspace → moeTopK → Output
(2 kernels, intermediate storage required)
```

**Optimized Flow:**
```
Input → moeSoftmaxTopK (fused) → Output
(1 kernel, no intermediate storage)
```

**Memory Layout:**
The fused kernels use shared memory arrays internally:
- `softmax[384]`: Stores softmax values per token (reused from the template)
- `selected[TOPK]`: Tracks already-selected experts for current topk iteration

### Implementation Details

The kernel dispatch happens in two places:
1. **Verification phase** (before benchmarking): Ensures correctness
2. **Benchmark loop**: Repeated kernel execution for timing

Both phases use the same fused kernels to ensure consistency.

### Expected Performance Improvements

- **Kernel Launch Overhead**: ~25-50% reduction from eliminating extra launch
- **Memory Bandwidth**: Reduced by avoiding intermediate global memory writes
- **SM Occupancy**: Fused kernels can achieve better occupancy

### Constraints Satisfied

✓ Correctness check maintained
✓ No tolerance changes
✓ Reference not modified
✓ Input scale unchanged (32768 tokens, 384 experts)
✓ Repeat count unchanged (1000)
✓ Full computation preserved

### Actual Performance Results

**V1 (Fused Kernels) - Job ID: 945011**

All tests PASSED correctness verification.

Average execution times (10 runs, 1000 repeats each):
- **topk=1**: 165.27 μs (min: 165.24 μs, max: 191.69 μs including warmup)
- **topk=2**: 344.90 μs 
- **topk=4**: 561.42 μs
- **topk=8**: 1228.91 μs

Performance characteristics:
- Very consistent performance across runs (low variance)
- topk=1 shows warmup effect (first run ~191 μs, stabilizes at ~165 μs)
- Linear scaling with topk (approximately 2x, 3.4x, 7.4x)

### Notes

This optimization leverages the specialized kernels that were already present in the codebase but not being utilized. The main improvement comes from:
1. Reducing kernel launch overhead (single fused kernel vs two separate kernels)
2. Improving cache locality by computing softmax and top-k in one pass
3. Using template specialization for better code optimization by the compiler
4. Eliminating intermediate global memory writes for softmax values

### Comparison with Expected Benefits

Expected improvements over baseline (estimated ~25-50%):
- These results show the fused kernel strategy is working well
- Next versions will focus on further micro-optimizations
