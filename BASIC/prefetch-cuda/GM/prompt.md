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
