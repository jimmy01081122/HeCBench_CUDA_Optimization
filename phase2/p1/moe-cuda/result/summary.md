Implemented and verified the `moe-cuda` optimization.

Changed:
- Added fused `moeSoftmaxTopK` in [kernels.h](/home/r14525078/p1/HeCBench/src/moe-cuda/kernels.h:145), combining row softmax normalization and top-k selection without materializing the full softmax workspace.
- Switched warmup and timed loops in [main.cu](/home/r14525078/p1/HeCBench/src/moe-cuda/main.cu:79) to launch the fused kernel.
- Added Slurm runner [run_moe_cuda.slurm](/home/r14525078/p1/HeCBench/src/moe-cuda/run_moe_cuda.slurm:1).

Final Slurm verification: job `948369`, result file [moe_cuda_result_948369.txt](/home/r14525078/p1/HeCBench/src/moe-cuda/result/moe_cuda_result_948369.txt:1)

Results on Tesla V100, CUDA 12.8, `32768 384 topk 1000`:
```text
topk=1  PASS  235.236664 us
topk=2  PASS  289.782898 us
topk=4  PASS  483.853271 us
topk=8  PASS  936.080322 us
```

Earlier baseline run `948365` was:
```text
topk=1  304.189636 us
topk=2  348.113281 us
topk=4  520.498657 us
topk=8  952.412170 us
```

So the final verified speedups are about `22.7%`, `16.8%`, `7.0%`, and `1.7%` for topk `1/2/4/8`. Note: `src/moe-cuda/Makefile` already had an `ARCH=sm_70` worktree change when I started, so I left that untouched.

tokens : 34,700 used