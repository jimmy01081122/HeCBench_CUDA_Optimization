Implemented and verified `pingpong-cuda`.

Changed:
- Added `all` target so `make` builds both MPI and NCCL binaries: [Makefile](/home/r14525078/p1/HeCBench/src/pingpong-cuda/Makefile:54)
- Added the missing Slurm runner: [run_pingpong_cuda.slurm](/home/r14525078/p1/HeCBench/src/pingpong-cuda/run_pingpong_cuda.slurm:1)
- Optimized NCCL timed transfer loop by grouping each send/recv pair with `ncclGroupStart/End`: [main-nccl.cu](/home/r14525078/p1/HeCBench/src/pingpong-cuda/main-nccl.cu:119)

Verification:
- Baseline: `result/pingpong_cuda_result_948478.txt`
- Final: `result/pingpong_cuda_result_948489.txt`
- Both ran on `gn1224.twcc.ai` with 2 V100 GPUs.
- No correctness errors were printed.

Result at 1 GiB:
- MPI stayed essentially unchanged: `24.252 GB/s`
- NCCL improved from `22.899 GB/s` to `45.776 GB/s`
- NCCL transfer time improved from `0.046889655 s` to `0.023456587 s`

The earlier missing-NCCL issue was just because the Makefile’s default target only built `main-mpi`; the new `all` target fixes that.

TOKENS : 50,166 used 
