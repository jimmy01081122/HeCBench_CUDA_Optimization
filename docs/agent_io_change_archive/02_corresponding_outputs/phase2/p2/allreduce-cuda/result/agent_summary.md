# allreduce-cuda Agent Summary

## Baseline

- Job id: 947992
- Node: gn1222
- Correctness: PASS, all 12 buffer sizes printed `Verified allreduce`
- Launcher: `UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main`
- Metric: allreduce us/iteration by buffer size
- Largest-buffer metric: 120294 us/iteration at size 536870912
- Raw files: `allreduce_cuda_947992.out`, `allreduce_cuda_947992.err`, `allreduce_cuda_947992.txt`

## Optimization Submissions

| Attempt | Job id | Node | Strategy | Correctness | Largest-buffer us | Largest speedup | Geomean speedup | Improved rows | Result |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 947994 | gn1222 | Replace ring with CUDA-aware `MPI_Allreduce` for all sizes | PASS | 3410270 | 0.035x | 0.474x | 5/12 | Rejected |
| 2 | 947996 | gn1222 | `MPI_Allreduce` for size <= 4096, original ring otherwise | PASS | 120007 | 1.002x | 1.418x | 7/12 | Accepted |
| 3 | 947997 | gn1222 | Tiny `MPI_Allreduce`, two-rank direct exchange for all larger sizes | PASS | 131252 | 0.917x | 2.787x | 11/12 | Rejected |
| 4 | 947999 | gn1222 | Tiny `MPI_Allreduce`, two-rank direct exchange through size 67108864, ring for largest | PASS | 121921 | 0.987x | 2.801x | 11/12 | Accepted for non-largest rows; rejected as final largest metric |
| 5 | 948000 | gn1222 | Same thresholded fast paths, with original ring ordering restored for fallback | PASS | 120423 | 0.999x | 2.728x | 11/12 | Accepted final |

## Final Result

- Final job id: 948000
- Node: gn1222
- Correctness: PASS
- Final largest-buffer metric: 120423 us/iteration
- Baseline largest-buffer metric: 120294 us/iteration
- Largest-buffer speedup: 0.999x
- Geomean speedup across all buffer sizes: 2.728x
- Improved rows: 11 of 12

| Buffer size | Baseline us/iter | Final us/iter | Speedup |
| ---: | ---: | ---: | ---: |
| 0 | 9.41797 | 0.0671339 | 140.29x |
| 32 | 59.2458 | 29.7308 | 1.99x |
| 256 | 58.3669 | 32.3674 | 1.80x |
| 1024 | 61.0389 | 36.094 | 1.69x |
| 4096 | 67.3695 | 54.9319 | 1.23x |
| 16384 | 56.0534 | 39.4592 | 1.42x |
| 65536 | 100.603 | 37.1926 | 2.71x |
| 262144 | 974.914 | 84.2798 | 11.57x |
| 1048576 | 1978.77 | 802.643 | 2.47x |
| 8388608 | 4138.33 | 2922.32 | 1.42x |
| 67108864 | 17888 | 17053.2 | 1.05x |
| 536870912 | 120294 | 120423 | 0.999x |

## Best Strategy

The final implementation keeps the original ring algorithm as the safe fallback for the largest buffer, where the full-buffer direct exchange became memory-bandwidth limited. For tiny buffers it uses CUDA-aware `MPI_Allreduce`, which avoids the ring setup overhead. For two-rank mid-sized buffers it uses one `MPI_Sendrecv` of the full peer buffer followed by a GPU reduction, removing the original ring's extra staging and improving 11 of 12 measured sizes.


TOKENS = 58,919 used