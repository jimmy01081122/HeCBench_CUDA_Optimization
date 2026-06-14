你是一位 CUDA GPU performance engineer，專長是 softmax、row-wise reduction、warp-level primitive、memory bandwidth、cooperative groups。

請針對 HeCBench 的 softmax-cuda benchmark 進行 baseline 建立、除錯與可控優化。


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
- usage:
  ./main <number of slices> <slice size> <implementation> <repeat>
- implementation:
  0 = naive softMax
  1 = optimized softMax2
- default run target 原意看起來是：
  ./main 100000 784 0 100
  ./main 100000 784 1 100
- softMax_cpu:
  對每個 slice 做 max reduction、exp、sum reduction、normalize
- softMax kernel:
  每個 CUDA thread 處理一個 slice，serial loop over sliceSize
- softMax2 kernel:
  使用 cooperative_groups tiled_partition<32>
  每個 warp 處理一個 slice
  使用 warp-level reduce 找 max 與 sum
- correctness:
  CPU softMax_cpu 作為 reference
  fabsf(output_cpu[i] - output_gpu[i]) > 1e-3 則 FAIL
- timing:
  目前用 std::chrono 包住 repeat 次 kernel launch，輸出：
  Average kernel execution time: ... (ms)

目前已知 Makefile 問題：
- CFLAGS := (ARCH) 是錯的
- 應修成合法 nvcc flags：
  CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)
- OPTIMIZE 可保留：
  -O3 --use_fast_math
  但必須注意 --use_fast_math 可能影響 softmax correctness；若導致 FAIL，必須移除或改為可控選項
- run target 應為：
  ./$(program) 100000 784 0 100
  ./$(program) 100000 784 1 100
- Makefile recipe 行必須是 Tab，不得是空白

硬性限制：
1. baseline job 不算入 5 次優化提交。
2. baseline 完成後，最多只能提交 5 次 sbatch job。
3. 每次優化提交前必須說明：
   - 修改內容
   - 假設的瓶頸或錯誤原因
   - 預期改善
   - 此次提交要驗證什麼
4. 每次提交後必須讀取：
   - result/softmax_cuda_<jobid>.out
   - result/softmax_cuda_<jobid>.err
   - result/softmax_cuda_result_<jobid>.txt
5. 不得刪除 correctness verification。
6. 不得把 correctness FAIL 視為成功。
7. 不得只測 implementation 1 而跳過 implementation 0。
8. 不得縮小 number of slices 或 slice size 來偽造加速。
9. 不得使用 V100 sm_70 不支援的 CUDA feature，例如 cp.async。
10. 不得只回報最快單次數值；必須保留完整 raw output。
11. 若修改 Makefile / main.cu / run script，必須先備份：
    - Makefile.bak_agent
    - main.cu.bak_agent
    - run_softmax_cuda.slurm.bak_agent

============================================================
Stage 0: Baseline，不計入 5 次提交
============================================================

先不要修改 main.cu。請先確認目前 benchmark 真實狀態。

執行：
  cd /home/r14525078/HeCBench/src/softmax-cuda
  pwd
  ls -la
  cat Makefile
  sed -n '1,260p' main.cu
  grep -n "softMax_cpu\|softMax2\|cooperative_groups\|Average kernel execution time\|PASS\|FAIL\|use_fast_math\|BLOCK_SIZE" main.cu Makefile

請先回答：
1. 目前能否 make？
2. 第一個 build error 是什麼？
3. Makefile 是否損壞？
4. implementation 0 是否 PASS？
5. implementation 1 是否 PASS？
6. --use_fast_math 是否造成 correctness 問題？
7. timing 是否可信？
8. 是否需要 CUDA event timing？
9. optimized softMax2 對 sliceSize=784 是否合理？

建立 sbatch 腳本：
  run_softmax_cuda.slurm

Slurm 腳本要求：
  #SBATCH -J softmax_cuda
  #SBATCH -A ACD115083
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gpus-per-node=1
  #SBATCH -t 00:10:00
  #SBATCH -o result/softmax_cuda_%j.out
  #SBATCH -e result/softmax_cuda_%j.err

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
11. run baseline:
    ./main 100000 784 0 100
    ./main 100000 784 1 100
12. benchmark output 必須 tee 到：
    result/softmax_cuda_result_<jobid>.txt

Baseline 成功條件：
- build PASS
- job COMPLETED
- 至少 1 GPU visible
- implementation 0 PASS
- implementation 1 PASS
- 有 Average kernel execution time
- raw output 完整保存

若 baseline 失敗，請分類：
- Makefile failure
- compile failure
- cooperative_groups compile issue
- fast math correctness issue
- runtime crash
- correctness failure
- no performance metric
- timeout

============================================================
Optimization Submission 1
============================================================

目標：修復 build / run / output reproducibility，不做激進性能優化。

可修改：
- Makefile
- run_softmax_cuda.slurm
- 輸出格式

Makefile 建議修成：
  CC        = nvcc
  OPTIMIZE  = yes
  DEBUG     = no
  ARCH      = sm_70
  FAST_MATH ?= yes
  LAUNCHER  =

  program = main
  source = main.cu
  obj = $(source:.cu=.o)

  CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)
  LDFLAGS =

  ifeq ($(DEBUG),yes)
    CFLAGS += -g -DDEBUG
    LDFLAGS += -g
  endif

  ifeq ($(OPTIMIZE),yes)
    CFLAGS += -O3
  endif

  ifeq ($(FAST_MATH),yes)
    CFLAGS += --use_fast_math
  endif

  $(program): $(obj) Makefile
      $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

  %.o: %.cu Makefile
      $(CC) $(CFLAGS) -c $< -o $@

  clean:
      rm -rf $(program) $(obj)

  run: $(program)
      $(LAUNCHER) ./$(program) 100000 784 0 100
      $(LAUNCHER) ./$(program) 100000 784 1 100

注意：recipe 行必須是 Tab。若擔心 Tab 問題，可以使用 .RECIPEPREFIX := >，但必須確保 make 可用。

提交後必須判斷：
- build 是否 PASS
- implementation 0 是否 PASS
- implementation 1 是否 PASS
- 是否有完整 timing
- FAST_MATH 是否安全

============================================================
Optimization Submission 2
============================================================

目標：改善 timing 可信度與輸出可解析性。

可修改 main.cu，但必須保留 correctness。

要求：
1. 使用 CUDA event timing 量測 kernel execution time。
2. 保留或輔助輸出 CPU chrono，但 final ranking 以 CUDA event timing 為主。
3. 加入 warmup，例如 10 次 kernel launch。
4. repeat 必須輸出。
5. 每個 implementation 都輸出可解析 RESULT 行：
   RESULT,impl=0,batch=100000,slice=784,repeat=100,avg_ms=...,status=PASS
   RESULT,impl=1,batch=100000,slice=784,repeat=100,avg_ms=...,status=PASS
6. 如果 correctness FAIL：
   RESULT,...,status=FAIL
   且該數據不得納入性能比較。
7. 不得把 H2D/D2H copy 或 CPU softMax_cpu time 包進 GPU kernel timing。
8. 必須在 timing 結束前 cudaEventSynchronize 或 cudaDeviceSynchronize。

============================================================
Optimization Submission 3
============================================================

目標：測試 shape sensitivity。

在不超時前提下做代表性 shape sweep。

保留官方 baseline shape：
  batch=100000, slice=784

額外測試少量代表性 slice sizes：
  slice=128
  slice=256
  slice=784
  slice=1024
  slice=2048

batch 可根據總元素控制，避免超時，例如：
  batch=100000 for slice <= 1024
  batch=50000 for slice=2048

要求：
1. implementation 0 與 implementation 1 都測。
2. 每個 shape 都要 correctness PASS 才能納入。
3. 每組輸出 RESULT 行。
4. 不得只挑對 optimized 有利的 shape。
5. 若某 shape 太慢或 OOM，必須記錄原因。

============================================================
Optimization Submission 4
============================================================

目標：在 correctness PASS 前提下做小幅性能優化。

可嘗試：
1. 優化 softMax2 的 warp-level implementation。
2. 減少 expf 重複計算：
   現在 softMax2 對每個元素可能在 sum pass 與 output pass 各算一次 expf。
   可考慮 cache 或重新設計，但需注意 sliceSize=784，per row 存全部 exp 可能不划算。
3. 針對 small slice 使用 one-warp-per-row。
4. 針對 larger slice 測 one block per row。
5. 調整 BLOCK_SIZE：
   128, 256, 512
6. 避免不必要的 cooperative_groups overhead，若改用 warp shuffle reduction，必須保持 correctness。
7. 若使用 --use_fast_math 導致精度問題，提供 FAST_MATH=no fallback。
8. 若新增 implementation 2，必須同時保留 implementation 0/1 作比較。

禁止：
- 不得移除 CPU reference。
- 不得放寬 tolerance 超過合理範圍；若要改 tolerance，必須說明原因。
- 不得改成 approximate softmax。
- 不得跳過 normalize。
- 不得把 sliceSize 固定寫死只支援 784，除非作為特化 implementation 並保留 general implementation。

============================================================
Optimization Submission 5
============================================================

目標：final confirmation。

要求：
1. 使用目前 correctness PASS 的最佳版本。
2. 跑完整 final test：
   至少包含 batch=100000, slice=784, impl 0/1/最佳新增 impl。
3. 保存 raw output、CSV、summary。
4. 確認 stderr 沒有 fatal error。
5. 若 stderr 只有 CUDA 12.8 對 sm_70 offline compilation warning，可標記為 non-fatal。
6. 若提升小於 1%，標記為 measurement-equivalent，不可宣稱顯著加速。

請產生 CSV：
  result/softmax_results_<jobid>.csv

CSV 欄位至少包含：
  job_id,node,implementation,batch_size,slice_size,repeat,avg_ms,correctness,status,notes

============================================================
Final report
============================================================

請產生：
  /home/r14525078/HeCBench/src/softmax-cuda/result/agent_summary.md

內容必須包含：

1. Environment
   - GPU model
   - CUDA_VISIBLE_DEVICES
   - number of GPUs
   - nvcc version
   - CUDA arch
   - node
   - Slurm settings
   - FAST_MATH setting

2. Benchmark characterization
   - softmax-cuda 實際在測什麼
   - numSlice / sliceSize / implementation
   - naive kernel vs optimized kernel 差異
   - correctness check 是什麼
   - timing 方法是什麼

3. Baseline
   - baseline job id
   - build PASS/FAIL
   - run PASS/FAIL
   - implementation 0 PASS/FAIL and timing
   - implementation 1 PASS/FAIL and timing
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
   - implementation
   - batch_size
   - slice_size
   - avg_ms
   - speedup vs naive
   - speedup vs original optimized
   - correctness
   - status

6. Optimization analysis
   - 哪些修改有效
   - 哪些修改無效
   - 是否只是修 timing / output
   - 是否有實質 kernel 加速
   - 主要瓶頸可能是 expf、global memory bandwidth、reduction、warp occupancy、cooperative_groups overhead、launch overhead 哪一類

7. Limitations
   - 只測單 GPU
   - sbatch 次數限制
   - 未使用 Nsight Compute
   - shape sweep 有限
   - CPU reference 不在 GPU timing 內，但會影響 wall time

8. Final conclusion
   Choose one:
   - SUCCESS: correctness PASS and performance improved or valid baseline established
   - PARTIAL: correctness PASS but performance improvement not significant
   - ENVIRONMENT ISSUE: GPU allocation / runtime issue
   - CODE ISSUE: benchmark logic or source corruption issue
   - INCONCLUSIVE: insufficient submissions

Important:
- 如果 correctness FAIL，不能寫 SUCCESS。
- 如果只通過 implementation 0，不可宣稱 optimized 成功。
- 如果 benchmark 原始檔案損壞，請先修復並明確標示為 source repair。
- 如果最佳提升小於 1%，請標記為 measurement-equivalent，不可宣稱顯著加速。
- 如果主要工作只是修復 Makefile 或 timing，最終報告必須明確標示。