Implemented and benchmarked the optimization for `allreduce-cuda`.

Changed:
- [collectives.cu](/home/r14525078/p1/HeCBench/src/allreduce-cuda/collectives.cu:11): added NCCL support and initialized a rank-wide NCCL communicator after `MPI_Init`.
- [collectives.cu](/home/r14525078/p1/HeCBench/src/allreduce-cuda/collectives.cu:271): replaced the measured GPU allreduce path with `ncclAllReduce`, with MPI fallback for CPU mode and a fast return for zero-length buffers.
- [Makefile](/home/r14525078/p1/HeCBench/src/allreduce-cuda/Makefile:14): added NCCL include/link flags.
- [run_allreduce_cuda.slurm](/home/r14525078/p1/HeCBench/src/allreduce-cuda/run_allreduce_cuda.slurm:1): added the requested Slurm build/run script.

Verified through Slurm on `gn1224`, 2x Tesla V100, job `947975`.
Final result file: [allreduce_cuda_result_947975.txt](/home/r14525078/p1/HeCBench/src/allreduce-cuda/result/allreduce_cuda_result_947975.txt)

Baseline job `947973` vs final job `947975`, us/iteration:

```text
size 0:         9.3641 -> 0.0682
size 32:       59.0720 -> 18.7702   3.15x
size 256:      58.3852 -> 18.2919   3.19x
size 1024:     60.8001 -> 24.7039   2.46x
size 4096:     72.0809 -> 28.7480   2.51x
size 16384:    60.0712 -> 41.6286   1.44x
size 65536:   105.6400 -> 89.8118   1.18x
size 262144: 1235.2500 -> 653.9120  1.89x
size 1048576: 2762.0200 -> 740.8860 3.73x
size 8388608: 5084.7300 -> 2951.4200 1.72x
size 67108864: 23121.1 -> 17307.9   1.34x
size 536870912: 141747 -> 121825    1.16x
```

Correctness passed for every buffer size.

Token cost report: exact task token accounting was not exposed in this environment; `get_goal` returned no active goal or completion budget report.



approximate : 56,398 tokens