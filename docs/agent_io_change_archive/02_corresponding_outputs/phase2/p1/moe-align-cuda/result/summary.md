Implemented and benchmarked the CUDA optimization.

Changed [main.cu](/home/r14525078/p1/HeCBench/src/moe-align-cuda/main.cu:352): the large-batch path now reuses its device `cumsum_buff` scratch allocation instead of doing `cudaMalloc`, redundant `cudaMemset`, and `cudaFree` on every measured call. The align kernel overwrites the needed cumsum entries before the sort kernel uses them, so correctness is preserved.

Added [run_moe_align_cuda.slurm](/home/r14525078/p1/HeCBench/src/moe-align-cuda/run_moe_align_cuda.slurm:1), including the required CUDA 12.8 module load and the cluster account `ACD115083`.

Benchmark run:
`sbatch run_moe_align_cuda.slurm` -> job `948012`

Result:
- 30/30 combinations passed correctness checks.
- Mean latency across all reported token/topk/expert/block_size combinations: `15.948151 us`
- Min/max reported case latency: `11.375971 us` / `31.278940 us`
- Output saved at [moe-align-cuda-948012.out](/home/r14525078/p1/HeCBench/src/moe-align-cuda/result/moe-align-cuda-948012.out:1)

Only stderr content was the CUDA warning about future removal of offline compilation support for `sm_70`.

33,975 used