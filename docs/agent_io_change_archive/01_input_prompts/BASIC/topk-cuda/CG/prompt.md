你是一位 CUDA GPU performance engineer，專長是 Top-K、radix selection、memory-bound kernel、CUB、warp/block-level optimization。

請針對 HeCBench 的 topk CUDA benchmark 進行 baseline 建立、除錯與可控優化。

Benchmark path:
  /home/r14525078/HeCBench/src/topk

Result path:
  /home/r14525078/HeCBench/src/topk/result

硬體與環境：
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- CUDA module:
  module purge
  module load cuda/12.8
- 優先使用單 GPU
- 不得在 login node 直接執行 GPU benchmark
- 必須使用 sbatch

目前已知程式行為：
- main.cu usage:
  ./main <batch_size> <repeat>
- main.cu 會測試 hidden sizes:
  3072, 4096, 8192, 16384, 32768, 65536, 131072
- main.cu 會測試 topk:
  2048, 1024
- input 先由 init_x 產生每 row 的 0..hidden_size-1，再由 shuffle_rows 打亂。
- GPU 呼叫:
  topk_radix<float, int32_t>(...)
- correctness:
  host 端對每 row 使用 std::partial_sort 取得 top-k reference。
  將 GPU output sort 後與 reference 比較。
  每組測試輸出 PASS 或 FAIL。
- timing:
  原始 main.cu 使用 std::chrono 包住 repeat 次 topk_radix 呼叫，輸出:
  Average execution time of topk : ... (us)

目前已知風險：
1. Makefile 可能損壞：
   - CFLAGS := (ARCH) 是錯的。
   - 必須修成合法 nvcc flags。
   - run target 應為 ./$(program) 3072 1000。
   - recipe 行必須是 Tab。
2. topk_per_row_kernels.h 可能經過格式轉換或不完整，必須以實際檔案能否編譯為準。
3. topk_per_row_kernels.h 內有 CUB dependency，需要確認 include 與 CUDA toolkit 是否可支援。
4. 不得為了通過 correctness 而縮小 hidden sizes 或跳過難測組合，除非明確作為 diagnostic，final 必須恢復完整官方測試範圍。
5. 若某 hidden_size/topk 組合因 OOM 或 timeout 失敗，必須明確記錄，不能偽造成 PASS。

硬性限制：
1. baseline job 不算入 5 次優化提交。
2. baseline 完成後，最多只能提交 5 次 sbatch job。
3. 每次優化提交前必須說明：
   - 修改內容
   - 假設的瓶頸或錯誤原因
   - 預期改善
   - 此次提交要驗證什麼
4. 每次提交後必須讀取：
   - result/topk_cuda_<jobid>.out
   - result/topk_cuda_<jobid>.err
   - result/topk_cuda_result_<jobid>.txt
5. 不得刪除 correctness verification。
6. 不得把 correctness FAIL 視為成功。
7. 不得把只通過部分 hidden_size/topk 視為完整成功。
8. 不得使用 V100 sm_70 不支援的 CUDA feature，例如 Ampere/Hopper 專屬 cp.async。
9. 不得只回報最快單次數值；必須保留完整 raw output。
10. 若修改 Makefile / main.cu / topk_per_row_kernels.h / utils.h / run script，必須先備份：
    - Makefile.bak_agent
    - main.cu.bak_agent
    - topk_per_row_kernels.h.bak_agent
    - utils.h.bak_agent
    - run_topk_cuda.slurm.bak_agent

============================================================
Stage 0: Baseline，不計入 5 次提交
============================================================

先不要修改 topk_per_row_kernels.h 或 main.cu。請先確認目前 benchmark 的真實狀態。

執行：
  cd /home/r14525078/HeCBench/src/topk
  pwd
  ls -la
  cat Makefile
  sed -n '1,260p' main.cu
  sed -n '1,260p' topk_per_row_kernels.h
  sed -n '1,160p' utils.h
  grep -Rni "topk_radix\|standalone_stable_radix\|cub\|CUB\|cudaEvent\|chrono\|Average execution time\|PASS\|FAIL" .

請先回答：
1. 目前能否 make？
2. 第一個 build error 是什麼？
3. Makefile 是否損壞？
4. topk_per_row_kernels.h 是否完整？
5. 是否需要額外 include path？
6. correctness 是否存在？
7. timing 是否可信？
8. benchmark 是否會測完整 hidden sizes/topk 組合？

建立 sbatch 腳本：
  run_topk_cuda.slurm

Slurm 腳本要求：
  #SBATCH -J topk_cuda
  #SBATCH -A ACD115083
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=1
  #SBATCH --gpus-per-node=1
  #SBATCH -t 00:20:00
  #SBATCH -o result/topk_cuda_%j.out
  #SBATCH -e result/topk_cuda_%j.err

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
    ./main 3072 100
    若 100 repeat 過慢，再用 diagnostic repeat=10，但 final 必須至少使用合理 repeat 並說明。
12. benchmark output 必須 tee 到：
    result/topk_cuda_result_<jobid>.txt

Baseline 成功條件：
- build PASS
- job COMPLETED
- 至少 1 GPU visible
- 每個 hidden_size/topk 組合 correctness PASS
- 有 Average execution time of topk
- raw output 完整保存

若 baseline 失敗，請分類：
- Makefile failure
- compile failure
- missing include / CUB issue
- template instantiation issue
- runtime crash
- OOM
- timeout
- correctness failure
- no performance metric

============================================================
Optimization Submission 1
============================================================

目標：修復 build / run，不做激進性能優化。

可修改：
- Makefile
- run_topk_cuda.slurm
- 只修復必要語法錯誤

Makefile 建議修成：
  CC        = nvcc
  OPTIMIZE  = yes
  DEBUG     = no
  ARCH      = sm_70
  LAUNCHER  =

  program = main
  source = main.cu
  obj = $(source:.cu=.o)

  CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH) --expt-relaxed-constexpr
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

  %.o: %.cu topk_per_row_kernels.h utils.h Makefile
      $(CC) $(CFLAGS) -c $< -o $@

  clean:
      rm -rf $(program) $(obj)

  run: $(program)
      $(LAUNCHER) ./$(program) 3072 100

注意：recipe 行必須是 Tab。若擔心 Tab 問題，可以用 .RECIPEPREFIX := >，但必須確保 make 可用。

提交後必須判斷：
- build 是否 PASS
- ./main 3072 100 是否可執行
- correctness 是否全部 PASS
- 是否有完整 timing

============================================================
Optimization Submission 2
============================================================

目標：改善 timing 可信度與輸出可解析性。

可修改 main.cu，但必須保留 correctness。

要求：
1. 使用 CUDA event timing 量測 topk_radix。
2. 保留或輔助輸出 CPU chrono，但 final ranking 以 CUDA event timing 為主。
3. warmup 次數合理化：
   - 原始 warmup=100 可能過高。
   - 可改為 warmup=10 或可設定，但必須記錄。
4. repeat 必須輸出。
5. 每個 hidden_size/topk 組合都輸出可解析 RESULT 行：
   RESULT,batch_size=...,hidden_size=...,topk=...,repeat=...,avg_us=...,status=PASS
6. 如果 correctness FAIL：
   RESULT,...,status=FAIL
   且該數據不得納入性能比較。
7. 不得把 CPU partial_sort time 包進 GPU timing。
8. GPU timing 必須包含 topk_radix 實際完成，可用 cudaEventRecord + cudaEventSynchronize。

============================================================
Optimization Submission 3
============================================================

目標：找出瓶頸與參數敏感性。

在不超時前提下做小範圍 sweep。

建議：
1. batch_size 固定為 3072。
2. hidden_size/topk 保留原官方組合。
3. 可額外測 batch_size:
   - 512
   - 1024
   - 3072
   但若時間不足，優先保留 3072 官方設定。
4. 針對 topk_per_row_kernels.h 內部策略，先不要大改 radix logic。
5. 若存在可調 block size / BitsPerPass / threshold，嘗試少量候選：
   - BitsPerPass 不得盲改，必須確認 correctness。
   - BlockSize 不得超過 V100 支援範圍。
6. 每個變體都必須保留 correctness PASS。

輸出：
  RESULT,variant=baseline_or_candidate,batch_size=...,hidden_size=...,topk=...,avg_us=...,status=PASS

============================================================
Optimization Submission 4
============================================================

目標：在 correctness PASS 前提下嘗試性能優化。

可根據程式實際內容選擇：

可能方向：
1. 減少不必要的 host-device copy。
   注意：correctness 需要 h_x / h_out / h_ids，但 timing 不能包含這些 copy。
2. 確保 input initialization 不在 timed loop 中。
3. 若 topk_radix 內有 workspace allocation，嘗試將 workspace allocation 移出 repeat loop。
4. 若 topk_radix 支援外部 workspace，重用 workspace。
5. 改善 launch configuration，但必須保留 correctness。
6. 若檔案中 WARP_SIZE=64，需確認這是否是移植殘留。CUDA warp size 是 32；但不得盲改。若嘗試改為 32，必須用一次提交驗證完整 correctness 與性能。
7. 若 kernel 使用 CUB block primitives，確認 block size / shared memory 使用合理。
8. 若 hidden_size <= 32768 與更大 hidden_size 有不同最佳策略，可分別處理，但不得只優化某一組。

禁止：
- 不得移除 CPU reference partial_sort。
- 不得只驗證 topk value 不驗證實際 topk correctness。
- 不得降低 topk 或 hidden size 來取得速度。
- 不得跳過最慢組合。
- 不得用 approximate top-k 取代 exact top-k。
- 不得改變排序語意；largest=true 時必須回傳最大 top-k。

============================================================
Optimization Submission 5
============================================================

目標：final confirmation。

要求：
1. 使用目前 correctness PASS 的最佳版本。
2. 跑完整官方測試：
   ./main 3072 <repeat>
   repeat 可依時間設定，但必須和前面結果可比較。
3. 保存 raw output、CSV、summary。
4. 確認 stderr 沒有 fatal error。
5. 若 stderr 只有 CUDA 12.8 對 sm_70 offline compilation warning，可標記為 non-fatal。
6. 若提升小於 1%，標記為 measurement-equivalent，不可宣稱顯著加速。

請產生 CSV：
  result/topk_results_<jobid>.csv

CSV 欄位至少包含：
  job_id,node,batch_size,hidden_size,topk,repeat,variant,avg_us,correctness,status,notes

============================================================
Final report
============================================================

請產生：
  /home/r14525078/HeCBench/src/topk/result/agent_summary.md

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
   - topk 實際在測什麼
   - batch_size / hidden_size / topk 組合
   - topk_radix 的大致策略
   - correctness check 是什麼
   - timing 方法是什麼

3. Baseline
   - baseline job id
   - build PASS/FAIL
   - run PASS/FAIL
   - correctness PASS/FAIL
   - 每組 hidden_size/topk 的 baseline avg_us
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
   - hidden_size
   - topk
   - baseline avg_us
   - final avg_us
   - speedup
   - correctness
   - status

6. Optimization analysis
   - 哪些修改有效
   - 哪些修改無效
   - 是否只是修 timing / output
   - 是否有實質 kernel 加速
   - 主要瓶頸可能是 radix passes、histogram、workspace traffic、global memory bandwidth、shared memory、synchronization、launch overhead 哪一類

7. Limitations
   - 只測單 GPU
   - sbatch 次數限制
   - 未使用 Nsight Compute
   - 未完整探索所有 radix parameters
   - CPU partial_sort correctness overhead 不在 GPU timing 內，但會影響整體 wall time

8. Final conclusion
   Choose one:
   - SUCCESS: correctness PASS and performance improved or valid baseline established
   - PARTIAL: correctness PASS but performance improvement not significant
   - ENVIRONMENT ISSUE: GPU allocation / runtime issue
   - CODE ISSUE: benchmark logic or source corruption issue
   - INCONCLUSIVE: insufficient submissions

Important:
- 如果 correctness FAIL，不能寫 SUCCESS。
- 如果只通過部分 hidden_size/topk，不能寫完整 SUCCESS。
- 如果 benchmark 原始檔案損壞，請先修復並明確標示為 source repair。
- 如果最佳提升小於 1%，請標記為 measurement-equivalent，不可宣稱顯著加速。
- 如果主要工作只是修復 Makefile 或 timing，最終報告必須明確標示。