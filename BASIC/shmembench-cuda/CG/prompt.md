你是一位 CUDA shared-memory / memory bandwidth benchmark performance engineer。
請針對 HeCBench 的 shmembench-cuda benchmark 進行 baseline 建立、除錯與可控優化。

Benchmark path:
  /home/r14525078/HeCBench/src/shmembench-cuda

Result path:
  /home/r14525078/HeCBench/src/shmembench-cuda/result

硬體與環境：
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- CUDA module:
  module purge
  module load cuda/12.8
- 優先使用單 GPU 測試
- 不得在 login node 直接執行 GPU benchmark
- 必須使用 sbatch

目前已知 benchmark 資訊：
- CMakeLists.txt 顯示：
  add_hecbench_benchmark(
    NAME shmembench
    MODEL cuda
    SOURCES main.cu shmem_kernels.cu
    CATEGORIES algorithms
  )
- main.cu 顯示此程式會輸出：
  Shared memory bandwidth microbenchmark
- main.cu 需要一個參數：
  ./main <repeat>
- main.cu 固定 VECTOR_SIZE = 1024 * 1024
- datasize = VECTOR_SIZE * sizeof(double)，約 8 MiB
- main.cu 呼叫：
  shmembenchGPU(c, VECTOR_SIZE, repeat)
- 你必須檢查 shmem_kernels.cu 與 shmem_kernels.h，因為真正的 shared-memory benchmark kernel 不在 main.cu。
- 目前 Makefile 有明顯錯誤：
  - ARCH = sm_60，應改為 sm_70
  - CFLAGS := (ARCH) 是錯的
  - link command 中 $ 可能是錯誤殘留
  - run target 應為 ./$(program) 1000，而不是 (program) 1000
  - recipe 行必須是 Tab，不可用空白

任務目標：
1. 修復 Makefile，使 benchmark 可在 V100 sm_70 上編譯。
2. 建立有效 baseline。
3. 確認 benchmark 是否有 correctness check；若沒有，必須明確標記 correctness not provided。
4. 分析 shmembenchGPU 實際測的是 shared memory bandwidth、bank conflict、latency、還是其他 memory pattern。
5. 在不改變 benchmark 本意的前提下優化 timing 可信度與 performance。
6. 最多 5 次優化 sbatch 提交；baseline 不算。

硬性限制：
1. baseline job 不算入 5 次優化提交。
2. baseline 完成後，最多只能提交 5 次 sbatch job。
3. 每次優化提交前必須說明：
   - 修改內容
   - 假設的瓶頸或錯誤原因
   - 預期改善
   - 此次提交要驗證什麼
4. 每次提交後必須讀取：
   - result/shmembench_cuda_<jobid>.out
   - result/shmembench_cuda_<jobid>.err
   - result/shmembench_cuda_result_<jobid>.txt
5. 不得刪除 correctness verification。
6. 若原 benchmark 沒有 correctness check，不得偽造 correctness PASS；必須寫 correctness not provided。
7. 不得把 correctness FAIL 視為成功。
8. 不得把 Waiving test / skipped test 視為成功。
9. 不得使用 V100 sm_70 不支援的 CUDA feature。
10. 不得只回報最高單次數值；必須保留完整 raw output。
11. 若修改 Makefile / main.cu / shmem_kernels.cu / shmem_kernels.h / run script，必須先備份：
    - Makefile.bak_agent
    - main.cu.bak_agent
    - shmem_kernels.cu.bak_agent
    - shmem_kernels.h.bak_agent
    - run_shmembench_cuda.slurm.bak_agent

============================================================
Stage 0: Baseline，不計入 5 次提交
============================================================

先不要修改 main.cu 或 shmem_kernels.cu。請先確認目前 benchmark 的真實狀態。

執行：
  cd /home/r14525078/HeCBench/src/shmembench-cuda
  pwd
  ls -la
  cat Makefile
  sed -n '1,220p' main.cu
  sed -n '1,260p' shmem_kernels.cu
  sed -n '1,160p' shmem_kernels.h
  grep -Rni "printf\|bandwidth\|GB/s\|shared\|__shared__\|cudaEvent\|chrono\|clock\|repeat\|shmembenchGPU" .

請先回答：
1. 此 benchmark 實際測什麼？
   - shared memory bandwidth？
   - shared memory bank conflict？
   - global memory to shared memory？
   - shared memory latency？
   - 其他 memory access pattern？
2. 是否有 correctness check？
3. 是否有 timing？
4. 是否輸出 bandwidth / latency / throughput？
5. repeat 參數如何影響結果？
6. 是否需要額外輸入參數？
7. 是否只需要單 GPU？
8. Makefile 的第一個錯誤是什麼？

建立 sbatch 腳本：
  run_shmembench_cuda.slurm

Slurm 腳本要求：
  #SBATCH -J shmembench_cuda
  #SBATCH -A ACD115083
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gpus-per-node=1
  #SBATCH -t 00:10:00
  #SBATCH -o result/shmembench_cuda_%j.out
  #SBATCH -e result/shmembench_cuda_%j.err

腳本必須：
1. mkdir -p result
2. module purge
3. module load cuda/12.8
4. 印出 module list
5. 印出 which nvcc / nvcc --version
6. 印出 hostname
7. 印出 CUDA_VISIBLE_DEVICES
8. 印出 nvidia-smi -L
9. 印出 nvidia-smi
10. build:
    make clean || true
    make ARCH=sm_70
11. run:
    ./main 1000
12. benchmark output 必須 tee 到：
    result/shmembench_cuda_result_<jobid>.txt

Baseline 成功條件：
- build PASS
- job COMPLETED
- 至少 1 GPU visible
- benchmark 沒有 Waiving / skipped
- 若有 correctness check，必須 PASS
- 若沒有 correctness check，必須明確標記 correctness not provided
- 有可解析 performance metric；若沒有，必須標記 no performance metric

若 baseline 失敗，請分類：
- Makefile failure
- compile failure
- missing input argument
- unsupported environment
- runtime crash
- correctness failure
- no performance metric
- timeout

============================================================
Optimization Submission 1
============================================================

目標：修復 build / run，不做激進性能優化。

可修改：
- Makefile
- run_shmembench_cuda.slurm
- 只修復必要語法錯誤

Makefile 建議修成：
  CC        = nvcc
  OPTIMIZE  = yes
  DEBUG     = no
  ARCH      = sm_70
  LAUNCHER  =

  program = main
  source = main.cu shmem_kernels.cu
  obj = main.o shmem_kernels.o

  CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)
  LDFLAGS =

  ifeq ($(DEBUG),yes)
    CFLAGS += -g -DDEBUG
    LDFLAGS += -g
  endif

  ifeq ($(OPTIMIZE),yes)
    CFLAGS += -O3
  endif

  $(program): $(obj) Makefile
      $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

  main.o: main.cu shmem_kernels.h Makefile
      $(CC) $(CFLAGS) -c $< -o $@

  shmem_kernels.o: shmem_kernels.cu shmem_kernels.h Makefile
      $(CC) $(CFLAGS) -c $< -o $@

  clean:
      rm -rf $(program) $(obj)

  run: $(program)
      $(LAUNCHER) ./$(program) 1000

注意：recipe 行必須是 Tab。若擔心 Tab 問題，可以用 .RECIPEPREFIX := >，但需確保 make 可用。

提交後必須判斷：
- build 是否 PASS
- ./main 1000 是否可執行
- 是否有 performance metric
- 是否有 correctness check

============================================================
Optimization Submission 2
============================================================

目標：改善 timing 可信度。

可修改 main.cu / shmem_kernels.cu，但必須保留 benchmark 本意。

要求：
1. 若原本使用 CPU chrono timing，請確認 timing 範圍合理。
2. 若測 CUDA kernel 或 shared-memory operation，優先加入 CUDA event timing。
3. 加入 warmup，避免第一次 kernel launch / context init 干擾。
4. 重複測試多次，輸出 avg / min / max。
5. 不得將 cudaMalloc / cudaFree 包進 bandwidth timing，除非 benchmark 本來就是測 allocation overhead。
6. 若有多個 kernel / memory pattern，請分開輸出 timing。
7. 若測 shared memory，請清楚輸出：
   - blockDim
   - gridDim
   - shared memory bytes
   - elements per thread
   - repeat
   - access pattern
   - bank conflict pattern if applicable

建議輸出：
  RESULT,test=...,repeat=...,block=...,shared_bytes=...,avg_us=...,min_us=...,max_us=...,bandwidth_GBps=...,correctness=...,status=PASS

若 benchmark 沒有 correctness：
  RESULT,test=...,avg_us=...,bandwidth_GBps=...,correctness=NOT_PROVIDED,status=MEASURED

============================================================
Optimization Submission 3
============================================================

目標：參數 sweep，找出瓶頸。

根據 shmem_kernels.cu 的實際內容選擇合適 sweep。

若是 shared memory bandwidth：
- 測 block size:
  128, 256, 512, 1024
- 測每 thread elements:
  1, 2, 4, 8
- 測 stride / bank conflict:
  contiguous
  stride 2
  stride 4
  stride 8
  stride 16
  stride 32

若是 global memory to shared memory：
- 測 buffer size 或 data reuse pattern
- 區分 global load time 與 shared memory access time

若是 latency / pointer chasing：
- 測 working set size
- 測 dependent chain length

要求：
1. 不要讓 sweep 超時。
2. 優先選少量代表性組合。
3. 每個測點至少有 warmup 與 repeat。
4. 所有測點都要輸出 RESULT 行。
5. 若某組合 correctness FAIL，該結果 invalid。
6. 若 correctness not provided，所有結果必須標記 correctness=NOT_PROVIDED。

============================================================
Optimization Submission 4
============================================================

目標：在語意不變的前提下嘗試小幅性能優化。

可嘗試：
1. 調整 block size / grid size。
2. 減少不必要的 global memory access。
3. 將 repeated calculation 移出 timed loop。
4. 若是 shared memory benchmark：
   - 測試 padding 避免 bank conflict
   - 調整 shared memory layout
   - 避免 unnecessary __syncthreads()
5. 若是 global memory benchmark：
   - 改善 coalescing
   - 使用 vectorized load/store，如 double2 / double4，但必須確保 alignment
6. 若是 reduction / accumulation：
   - 使用 warp-level primitive
   - 減少 atomic contention
   - 但不得改變 benchmark 本意

禁止：
- 不得刪除 correctness。
- 不得偽造 correctness。
- 不得只挑最快 case 而忽略完整 sweep。
- 不得把測試改成與原 benchmark 不同的問題。
- 不得為了提升 bandwidth 而少算資料量或改變 bytes 計算公式。

============================================================
Optimization Submission 5
============================================================

目標：final confirmation。

要求：
1. 使用目前最合理且可重現的版本。
2. 跑完整 final test。
3. 保存 raw output、CSV、summary。
4. 確認 stderr 沒有 fatal error。
5. 若 stderr 只有 CUDA 12.8 對 sm_70 offline compilation 的 warning，可標記為 non-fatal。
6. 若提升小於 1%，標記為 measurement-equivalent，不可宣稱顯著加速。

請產生 CSV：
  result/shmembench_results_<jobid>.csv

CSV 欄位依實際 benchmark 調整，但至少包含：
  job_id,node,test_name,repeat,block_size,grid_size,shared_bytes,avg_us,min_us,max_us,bandwidth_GBps,correctness,status,notes

============================================================
Final report
============================================================

請產生：
  /home/r14525078/HeCBench/src/shmembench-cuda/result/agent_summary.md

內容必須包含：

1. Environment
   - GPU model
   - CUDA_VISIBLE_DEVICES
   - number of GPUs
   - nvcc version
   - CUDA arch
   - node
   - Slurm settings

2. Benchmark characterization
   - shmembench-cuda 實際在測什麼
   - shared memory / global memory / latency / bank conflict / other
   - 是否有 correctness check
   - performance metric 是什麼
   - timing 方法是什麼
   - bytes / bandwidth 計算公式

3. Baseline
   - baseline job id
   - build PASS/FAIL
   - run PASS/FAIL
   - correctness PASS/FAIL 或 NOT_PROVIDED
   - baseline performance
   - failure reason if any

4. Submission history
   - optimization jobs 1 to 5
   For each:
   - job id
   - modification
   - hypothesis
   - result
   - correctness
   - performance if valid

5. Performance table
   至少包含：
   - test_name
   - block / stride / pattern
   - avg_us
   - bandwidth_GBps or latency
   - correctness
   - status

6. Optimization analysis
   - 哪些修改有效
   - 哪些修改無效
   - 是否接近硬體限制
   - 主要瓶頸是 shared memory bank conflict、global memory bandwidth、latency、occupancy、synchronization、instruction overhead 哪一類

7. Limitations
   - 只測單節點
   - sbatch 次數限制
   - 是否只測單 GPU
   - 是否缺少 profiler
   - 是否缺少 Nsight Compute metrics
   - 若 correctness not provided，必須明確說明

8. Final conclusion
   Choose one:
   - SUCCESS: correctness PASS and performance measured
   - PARTIAL: run PASS but correctness missing or performance metric insufficient
   - ENVIRONMENT ISSUE: GPU allocation / runtime issue
   - CODE ISSUE: benchmark logic or correctness issue
   - INCONCLUSIVE: insufficient submissions

Important:
- 如果 correctness FAIL，不能寫 SUCCESS。
- 如果 benchmark 原本沒有 correctness check，必須寫 correctness NOT_PROVIDED，不得偽造 PASS。
- 如果 benchmark 只是 microbenchmark，不要過度解讀成整體 application 加速。
- 如果最佳提升小於 1%，請標記為 measurement-equivalent，不可宣稱顯著加速。
- 若主要工作是修復 build / timing / output，而非加速，最終報告必須明確標示。
