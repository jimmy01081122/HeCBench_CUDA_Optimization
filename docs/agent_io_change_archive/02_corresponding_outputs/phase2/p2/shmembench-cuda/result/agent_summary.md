# shmembench-cuda Agent Summary

## Baseline

- Job id: 948672
- Node: gn1222
- Correctness: PASS (no `checksum failed` message)
- Average kernel execution time: 6.555778 ms (6555.778 us)
- Shared-memory bandwidth: 13110.52 GB/s
- Raw files:
  - `result/shmembench-cuda-948672.out`
  - `result/shmembench-cuda-948672.err`
  - `result/shmembench_cuda_948672.txt`

## Optimization Submissions

| Attempt | Job id | Node | Change | Correctness | Avg time (ms) | Bandwidth (GB/s) | Speedup vs baseline | Decision |
|---:|---:|---|---|---|---:|---:|---:|---|
| 1 | 948674 | gn1222 | Replaced block barriers with `__threadfence_block()` | PASS | 7.276476 | 11811.99 | 0.9010x | Rejected: slower |
| 2 | 948675 | gn1222 | Removed hot-loop barriers | PASS | 6.659015 | 12907.27 | 0.9845x | Rejected: slower |
| 3 | 948676 | gn1222 | Swept `BLOCK_SIZE` from 256 to 128 | FAIL | 6.459945 | 13305.02 | 1.0148x | Rejected: checksum failed |
| 4 | 948677 | gn1222 | Forced inline helpers and direct `float4` assignment | PASS | 6.549555 | 13122.98 | 1.0010x | Accepted |
| 5 | 948678 | gn1222 | Attempt 4 plus `#pragma unroll 64` | PASS | 6.547344 | 13127.41 | 1.0013x | Accepted: best |

## Final Result

- Final correctness: PASS (no `checksum failed` message)
- Final job id: 948678
- Final node: gn1222
- Final average kernel execution time: 6.547344 ms (6547.344 us)
- Final min/max kernel time: not reported by this benchmark binary
- Final shared-memory bandwidth: 13127.41 GB/s
- Final speedup: 1.0013x by average time and bandwidth
- Final raw files:
  - `result/shmembench-cuda-948678.out`
  - `result/shmembench-cuda-948678.err`
  - `result/shmembench_cuda_948678.txt`

## Best Strategy

The best valid strategy kept the original checksum-sensitive launch shape and synchronization semantics, then reduced small device-helper overhead with `__forceinline__`, used a direct `float4` shared-memory assignment during initialization, and increased the loop unroll factor from 32 to 64. Barrier-removal variants did not improve measured performance on the V100 run, and the block-size sweep was rejected because it changed the checksum.


TOKEN : 35,524 used