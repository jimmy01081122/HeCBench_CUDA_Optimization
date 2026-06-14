# Input Prompt

Benchmark: `prefetch-cuda`
Category: ML kernel / memory

## Source prompt files

### `BASIC/prefetch-cuda/GM/prompt.md`
```text
你是一位 CUDA Unified Memory / memory migration / prefetch performance engineer。
請針對 HeCBench 的 prefetch-cuda benchmark 進行 baseline 建立、除錯與可控優化。

硬體與環境：
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- CUDA module:
  module purge
  module load cuda/12.8
- 使用單 GPU
- 不得在 login node 直接執行 GPU benchmark
- 必須使用 sbatch

目前 benchmark 行為：
- 程式 usage:
  ./main <repeat>
- main.cu 固定：
  numElements = 64 * 1024 * 1024
- 使用 Unified Memory:
  cudaMallocManaged(&A, ...)
  cudaMallocManaged(&B, ...)
- prefetch mode:
  每次 iteration 執行 cudaMemPrefetchAsync(A)
  每次 iteration 執行 cudaMemPrefetchAsync(B)
  然後執行 add kernel
- naive mode:
  不做 cudaMemPrefetchAsync
  直接執行 add kernel
- add kernel:
  y[i] += x[i]
- correctness:
  初始化 A=1.0f, B=2.0f
  repeat 次後，預期 B[i] == repeat + 2
  maxError == 0 則 PASS
- 程式目前會執行：
  prefetch 10 次
  naive 10 次
- 每次會輸出 Average execution time 與 PASS/FAIL

目前已知 Makefile 問題：
- CFLAGS := (ARCH) 是錯的
- 應修成合法 nvcc flags，例如：
  CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)
```

### `phase2/p1/prefetch-cuda/prompt.md`
```text
# P1 Weak Prompt: prefetch-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/p1/HeCBench/src/prefetch-cuda`

Goal:
- Improve performance for `avg_ms for with_prefetch and without_prefetch modes`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/p1/HeCBench/src/prefetch-cuda
mkdir -p /home/r14525078/p1/HeCBench/src/prefetch-cuda/result
module purge
module load cuda/12.8
sbatch run_prefetch.slurm
```

If `run_prefetch.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
```

### `phase2/p2/prefetch-cuda/prompt.md`
```text
# P2 Medium Prompt: prefetch-cuda

You are a CUDA performance engineer.

## Benchmark

- benchmark: prefetch-cuda
- benchmark path: `/home/r14525078/p2/HeCBench/src/prefetch-cuda`
- result path: `/home/r14525078/p2/HeCBench/src/prefetch-cuda/result`
- category: unified_memory
- expected metric: avg_ms for with_prefetch and without_prefetch modes

## Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
```bash
module purge
module load cuda/12.8
```
- required GPUs: 1
- requires MPI: false
- requires NCCL: false

## Goal

1. Establish a real baseline before modifying source code.
2. Improve performance while keeping correctness PASS.
3. Save raw output and summarize the final valid result.

## Rules

1. Run baseline before source changes.
2. Baseline does not count as an optimization submission.
3. After baseline, at most 5 optimization sbatch submissions.
4. Do not remove or weaken correctness checks.
5. Do not shrink input size or skip cases to fake speedup.
6. Save `.out`, `.err`, and result `.txt` files under the result directory.
7. Report job id, node, correctness, metric, speedup, and whether the result is accepted or rejected.

## Benchmark-Specific Notes

- Preserve both prefetch and no-prefetch modes; do not only report the faster mode.
- Separate prefetch API overhead from demand-paging penalty.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p2/HeCBench/src/prefetch-cuda
mkdir -p /home/r14525078/p2/HeCBench/src/prefetch-cuda/result
module purge
module load cuda/12.8
sbatch run_prefetch.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p2/HeCBench/src/prefetch-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
./main 10
./main 100
```

If `run_prefetch.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
```
