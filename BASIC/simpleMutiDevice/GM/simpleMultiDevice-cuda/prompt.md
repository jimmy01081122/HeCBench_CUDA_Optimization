你是一位 CUDA multi-GPU performance engineer。請針對 HeCBench 的 simpleMultiDevice-cuda benchmark 進行修復、baseline 建立與可控優化。

Benchmark path:
  /home/r14525078/HeCBench/src/simpleMultiDevice-cuda

Result path:
  /home/r14525078/HeCBench/src/simpleMultiDevice-cuda/result

硬體與環境：
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- 使用 CUDA module:
  module purge
  module load cuda/12.8
- 預期至少測 2 GPUs
- 若程式修復後支援，可嘗試 4 GPUs
- 不得在 login node 直接執行 GPU benchmark
- 必須使用 sbatch

目前已知問題：
1. simpleMultiDevice.h 目前只定義：
   - int dataN
   - float *h_Data
   但 main.cu 會使用：
   - h_Sum
   - stream
   - d_Data
   - d_Sum
   - h_Sum_from_device
   因此 header 與 main.cu 不匹配，需要修復。

2. Makefile 目前 CFLAGS 類似：
   CFLAGS := (MAX_GPU)
   這是錯的。應修成合法 nvcc flags，例如：
   CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -DMAX_GPU_COUNT=$(MAX_GPU) -Xcompiler -Wall -arch=$(ARCH)

3. main.cu 目前可能有格式損壞，例如：
   - stdchronosteady_clock
   - stdchrono::duration_cast(...)
   - __global__ kernel 宣告格式可能需要確認
   必須修成可編譯 C++17 / CUDA 程式。

4. 這題的原始目標應是多 GPU reduction：
   - 將 DATA_N = 1048576 * 32 的 host input data 分配到多張 GPU
   - 每張 GPU 對自己的 partition 做 reduction
   - 回傳每張 GPU partial sum
   - host 端加總 partial sums
   - 與 CPU reference sum 比較
   - diff < 1e-5 才能 PASS

限制：
1. baseline job 不算入 5 次優化提交。
2. baseline 完成後，最多只能提交 5 次 sbatch job。
3. 每次優化提交前必須說明：
   - 修改內容
   - 假設的錯誤原因或效能瓶頸
   - 預期改善
   - 此次提交要驗證什麼
4. 每次提交後必須讀取：
   - result/simpleMultiDevice_cuda_<jobid>.out
   - result/simpleMultiDevice_cuda_<jobid>.err
   - result/simpleMultiDevice_cuda_result_<jobid>.txt
5. 不得刪除 correctness verification。
6. 不得把 correctness FAIL 視為成功。
7. 不得把 Waiving test / skipped test 視為成功。
8. 不得把只跑到 1 GPU 的結果宣稱為 multi-GPU 成功。
9. 不得使用 V100 sm_70 不支援的 CUDA feature。
10. 不得只回報最佳單次數值；必須保留完整 raw output。
11. 若修改 Makefile / main.cu / simpleMultiDevice.h / run script，必須先備份：
    - Makefile.bak_agent
    - main.cu.bak_agent
    - simpleMultiDevice.h.bak_agent
    - run_simpleMultiDevice_cuda.slurm.bak_agent

============================================================
Stage 0: Baseline，不計入 5 次提交
============================================================

先不要直接大改程式。請先確認目前狀態。

執行：
  cd /home/r14525078/HeCBench/src/simpleMultiDevice-cuda
  pwd
  ls -la
  cat Makefile
  sed -n '1,260p' main.cu
  sed -n '1,160p' simpleMultiDevice.h

請記錄：
1. 是否能 make
2. 第一個 build error 是什麼
3. Makefile 是否格式錯誤
4. header 是否缺欄位
5. main.cu 是否格式損壞
6. 是否已有 correctness check
7. 是否已有 timing
8. 是否真的使用多 GPU

建立 sbatch 腳本：
  run_simpleMultiDevice_cuda.slurm

Slurm 腳本要求：
  #SBATCH -J simpleMultiDevice_cuda
  #SBATCH -A ACD115083
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gpus-per-node=2
  #SBATCH -t 00:10:00
  #SBATCH -o result/simpleMultiDevice_cuda_%j.out
  #SBATCH -e result/simpleMultiDevice_cuda_%j.err

腳本必須：
1. mkdir -p result
2. module purge
3. module load cuda/12.8
4. 印出 module list
5. 印出 which nvcc / nvcc --version
6. 印出 hostname
7. 印出 CUDA_VISIBLE_DEVICES
8. 印出 nvidia-smi -L
9. build:
   make clean || true
   make ARCH=sm_70 MAX_GPU=4
10. run:
   ./main 1000
11. benchmark output tee 到：
   result/simpleMultiDevice_cuda_result_<jobid>.txt

Baseline 成功條件：
- build PASS
- job COMPLETED
- 至少 2 GPUs visible
- benchmark 沒有 Waiving / skipped
- correctness PASS
- 有 timing / performance output

若 baseline 失敗，請分類：
- Makefile failure
- header/main mismatch
- compile failure
- missing input argument
- insufficient GPU count
- cudaSetDevice failure
- runtime crash
- correctness failure
- no performance metric
- timeout

============================================================
Optimization Submission 1
============================================================

目標：修復 build，不做性能優化。

允許修改：
- Makefile
- simpleMultiDevice.h
- main.cu 的明顯語法 / 格式錯誤

必須完成：
1. Makefile 可用 nvcc 編譯。
2. ARCH 預設或命令列可設為 sm_70。
3. MAX_GPU_COUNT 可由 Makefile 傳入，例如：
   -DMAX_GPU_COUNT=$(MAX_GPU)
4. simpleMultiDevice.h 必須定義 main.cu 所需欄位，例如：
   - int dataN
   - float *h_Data
   - float *h_Sum
   - float *d_Data
   - float *d_Sum
   - float *h_Sum_from_device
   - cudaStream_t stream
5. 必須加入必要 include，例如 cuda_runtime.h。
6. 不得刪除 CPU reference 或 GPU/CPU comparison。

提交後判斷：
- build 是否 PASS
- 是否能啟動
- 若 correctness FAIL，記錄 diff、GPU sum、CPU sum

============================================================
Optimization Submission 2
============================================================

目標：修復 correctness。

請檢查並修復多 GPU reduction flow：

合理流程應該是：
1. cudaGetDeviceCount
2. GPU_N = min(device_count, MAX_GPU_COUNT)
3. 將 DATA_N 平均切分給 GPU_N 張 GPU
4. 為每個 GPU 配置 pinned host memory 與 device memory
5. 初始化每個 GPU 的 h_Data
6. cudaMemcpyAsync h_Data -> d_Data
7. 啟動 reduceKernel<<<BLOCK_N, THREAD_N, 0, stream>>>
8. cudaMemcpyAsync d_Sum -> h_Sum_from_device
9. cudaStreamSynchronize
10. host 端加總每張 GPU 的 partial reduction 結果
11. 計算 CPU reference
12. 比較 diff

注意：
- reduceKernel 目前每個 thread 輸出一個 partial sum 到 d_Result[tid]。
- host 端必須加總 ACCUM_N 個 partial sums。
- 不得只讀 d_Result[0]。
- 每張 GPU 的 h_Data 必須初始化，否則 CPU sum 與 GPU sum 都無意義。
- 必須正確處理 GPU_N 不整除 DATA_N 的 case。
- 如果使用 float reduction，誤差容忍要合理，但不要放太寬。原本 diff < 1e-5 可先保留；若浮點累加順序造成小誤差，請明確說明並使用合理 tolerance，例如 1e-4，但不得掩蓋錯誤。

提交後判斷：
- 2 GPU correctness 是否 PASS
- GPU sum / CPU sum / relative diff
- 是否使用所有 visible GPUs

============================================================
Optimization Submission 3
============================================================

目標：改善 timing 可信度。

可修改 main.cu，但必須保留 correctness。

要求：
1. 分離以下時間：
   - init/allocation time
   - H2D copy time
   - kernel time
   - D2H partial sum copy time
   - total GPU processing time
2. 使用 CUDA event timing 或 per-device stream synchronization。
3. 多 GPU timing 應量測所有 GPU 並行工作區間，即：
   - start before launching async work on all GPUs
   - stop after all GPU streams synchronize
4. 加入 warmup iteration，避免第一次 context/init 干擾 timing。
5. repeat=1000 時輸出 average time per iteration。
6. 不得把 malloc/free 包進 kernel time，除非另外標示 total time。

建議輸出格式：
  RESULT,num_gpus=2,repeat=1000,total_us=...,h2d_us=...,kernel_us=...,d2h_us=...,diff=...,status=PASS

============================================================
Optimization Submission 4
============================================================

目標：測試 GPU scaling。

若修復後 benchmark 支援多 GPU，請比較：
- 1 GPU
- 2 GPUs
- 若資源允許，4 GPUs

注意：
- 1 GPU 只作 scaling baseline。
- 2 GPUs 是最低正式 multi-GPU 條件。
- 4 GPUs 若排隊或資源不足，可以跳過，但必須記錄原因。
- 每次 sbatch 都算一次提交，請謹慎安排。

輸出：
  RESULT,num_gpus=1,...
  RESULT,num_gpus=2,...
  RESULT,num_gpus=4,...

分析：
- speedup_2gpu_vs_1gpu
- speedup_4gpu_vs_1gpu
- scaling efficiency
- 是否受 host-device copy、kernel reduction 或同步 overhead 限制

============================================================
Optimization Submission 5
============================================================

目標：final confirmation。

要求：
1. 使用目前 correctness PASS 的最佳版本。
2. 至少跑 2 GPU final。
3. 若 4 GPU 已成功，也跑 4 GPU final。
4. 保存 raw output、CSV、summary。
5. 確認 stderr 沒有 fatal error。
6. 若 stderr 只有 CUDA 12.8 對 sm_70 offline compilation 的 warning，可標記為 non-fatal。

請產生 CSV：
  result/simpleMultiDevice_results_<jobid>.csv

CSV 欄位至少包含：
  job_id,node,num_gpus,repeat,total_us,h2d_us,kernel_us,d2h_us,gpu_sum,cpu_sum,relative_diff,status

============================================================
Final report
============================================================

請產生：
  /home/r14525078/HeCBench/src/simpleMultiDevice-cuda/result/agent_summary.md

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
   - simpleMultiDevice-cuda 實際在測什麼
   - 是否 multi-GPU reduction
   - 資料如何切分
   - correctness check 是什麼
   - performance metric 是什麼

3. Baseline
   - baseline job id
   - build PASS/FAIL
   - run PASS/FAIL
   - correctness PASS/FAIL
   - 第一個錯誤原因
   - 是否原始檔案不完整或格式損壞

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
   - job_id
   - num_gpus
   - repeat
   - total_us
   - h2d_us
   - kernel_us
   - d2h_us
   - relative_diff
   - status

6. Scaling analysis
   如有 1/2/4 GPU：
   - 1 GPU time
   - 2 GPU time
   - 4 GPU time
   - speedup
   - scaling efficiency
   若沒有，請說明原因。

7. Optimization analysis
   - 哪些修改只是修 correctness
   - 哪些修改改善 timing
   - 哪些修改改善 performance
   - 目前瓶頸是 kernel、H2D、D2H、同步、allocation、host reduction 哪一類

8. Limitations
   - 只測單節點
   - sbatch 次數限制
   - 是否只測到 2 GPUs
   - 是否缺少 profiler
   - 是否缺少不同資料大小 sweep

9. Final conclusion
   Choose one:
   - SUCCESS: correctness PASS and performance measured
   - PARTIAL: correctness PASS but performance metric insufficient
   - ENVIRONMENT ISSUE: GPU allocation / runtime issue
   - CODE ISSUE: benchmark logic or source corruption issue
   - INCONCLUSIVE: insufficient submissions

Important:
- 如果 correctness FAIL，不能寫 SUCCESS。
- 如果只有 1 GPU 跑通，不能寫 multi-GPU SUCCESS。
- 如果 benchmark 沒有 performance metric，請先建立合理 timing，再報告。
- 如果最佳提升小於 1%，請標記為 measurement-equivalent，不可宣稱顯著加速。
- 若主要工作是修復損壞原始碼，而非加速，最終報告必須明確標示為 source repair / correctness recovery，而不是 performance optimization。