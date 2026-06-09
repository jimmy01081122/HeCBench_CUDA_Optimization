你是一位 CUDA + MPI + NCCL performance engineer。請針對 HeCBench 的 pingpong-cuda benchmark 進行除錯、baseline 建立與可控優化。

目標：
在 NVIDIA Tesla V100-SXM2-32GB、單節點 2 GPU、2 MPI ranks 環境下，讓 pingpong-cuda 的 MPI 與 NCCL ping-pong 測試都能 correctness PASS，並在 correctness 保持 PASS 的前提下優化 bandwidth / latency。

Benchmark path:
  /home/r14525078/HeCBench/src/pingpong-cuda

Result path:
  /home/r14525078/HeCBench/src/pingpong-cuda/result

硬體與環境：
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Required config:
  - 1 node
  - 2 MPI ranks
  - 2 GPUs
- CUDA_VISIBLE_DEVICES 必須顯示 2 GPUs
- 使用 sbatch，不要在 login node 直接跑 benchmark
- OpenMPI / NCCL 建議使用 NVIDIA HPC SDK / HPC-X module
- 若 default UCX/GDRCopy path 出現 gdr_get_info_v2 問題，必須避開 broken GDRCopy path

目前程式內容：
1. main-mpi.cu:
   - 使用 MPI_Init。
   - 要求 MPI world size 正好為 2。
   - 使用 rank % num_devices 綁定 GPU。
   - 對每個 size 配置 CUDA device buffer d_A。
   - warmup 5 次：
     rank 0: MPI_Send device buffer 到 rank 1，再 MPI_Recv 回來。
     rank 1: MPI_Recv device buffer，執行 CUDA kernel test(d_A, N)，再 MPI_Send 回 rank 0。
   - rank 0 將 d_A copy 回 host，檢查每個元素是否為 5。
   - 正式 timing loop_count=50，量測雙向 ping-pong 平均單次 transfer time。
   - 輸出格式：
     MPI : Transfer size (B): ..., Transfer Time (s): ..., Bandwidth (GB/s): ...

2. main-nccl.cu:
   - 使用 MPI_Init。
   - 要求 MPI world size 正好為 2。
   - 使用 rank % num_devices 綁定 GPU。
   - rank 0 產生 ncclUniqueId，MPI_Bcast 給其他 rank。
   - 使用 ncclCommInitRank 建立 NCCL communicator。
   - 使用 ncclSend / ncclRecv 做 ping-pong。
   - warmup 5 次，rank 1 收到後執行 CUDA kernel test(d_A, N)，再送回 rank 0。
   - rank 0 檢查結果是否為 5。
   - timing loop_count=50。
   - 輸出格式：
     NCCL: Transfer size (B): ..., Transfer Time (s): ..., Bandwidth (GB/s): ...

Baseline 不算在 5 次優化提交內。

============================================================
硬性限制
============================================================

1. Baseline job 不算入 5 次優化提交。
2. Baseline 完成後，最多只能提交 5 次 sbatch job。
3. 每次提交前必須說明：
   - 修改內容
   - 假設的錯誤原因或效能瓶頸
   - 預期改善
   - 此次提交要驗證什麼
4. 每次提交後必須讀取：
   - result/pingpong_cuda_<jobid>.out
   - result/pingpong_cuda_<jobid>.err
   - result/pingpong_cuda_result_<jobid>.txt
5. 不得刪除 MPI 或 NCCL correctness verification。
6. 不得把 correctness FAIL 視為成功。
7. 不得把 Waiving test 視為成功。
8. 不得改成只用 1 rank 或 1 GPU。
9. 不得跳過任何非零 transfer size 來偽造成功。
10. 不得使用 V100 sm_70 不支援的 CUDA feature。
11. 不得只報告最高 bandwidth；必須保留完整 size sweep。
12. 若 baseline MPI 或 NCCL 有一方失敗，必須明確標示是哪一方失敗。
13. 如果 MPI 版本只能在 tuned UCX launcher 下通過，必須記錄 launcher。
14. 如果 NCCL 版本通過但 MPI 版本失敗，不能宣稱整個 benchmark 全部成功。
15. 如果修改 Makefile / main-mpi.cu / main-nccl.cu / Slurm script，必須先備份：
    - Makefile.bak_agent
    - main-mpi.cu.bak_agent
    - main-nccl.cu.bak_agent
    - run_pingpong_cuda.slurm.bak_agent

============================================================
Stage 0: Baseline，不計入 5 次提交
============================================================

請先不要修改 main-mpi.cu 或 main-nccl.cu。先建立可重現 baseline。

步驟：
1. 進入：
   /home/r14525078/HeCBench/src/pingpong-cuda

2. 檢查：
   - pwd
   - ls -la
   - cat Makefile
   - grep -n "CC\|ARCH\|MPI_ROOT\|NVHPC\|CFLAGS\|LDFLAGS\|LAUNCHER\|main-mpi\|main-nccl" Makefile
   - ls -lh main-mpi.cu main-nccl.cu

3. 修正 build infrastructure 只限於必要項目：
   - 若 Makefile 的 MPI_ROOT 指向不存在路徑，改成由 mpicc 自動推導：
     MPI_ROOT := $(shell dirname $$(dirname $$(which mpicc)))
   - CFLAGS 必須包含：
     -std=c++17
     -I$(MPI_ROOT)/include
     -DOMPI_SKIP_MPICXX=
     -Xcompiler -Wall
     -arch=$(ARCH)
   - LDFLAGS 必須包含：
     -L$(MPI_ROOT)/lib
     -lmpi
     -lnccl
   - ARCH 必須是 sm_70。
   - Makefile recipe 行必須使用 Tab，不得使用空白造成 missing separator。

4. 建立 sbatch 腳本：
   run_pingpong_cuda.slurm

腳本要求：
- 建立 result/
- Slurm output:
  #SBATCH -o result/pingpong_cuda_%j.out
  #SBATCH -e result/pingpong_cuda_%j.err
- 使用：
  #SBATCH -A ACD115083
  #SBATCH -N 1
  #SBATCH --ntasks-per-node=2
  #SBATCH --gpus-per-node=2
  #SBATCH -t 00:15:00
- module:
  module purge
  module load nvhpc-24.11_hpcx-2.20_cuda-12.6
- 印出：
  module list
  which nvcc / nvcc --version
  which mpicc
  which mpirun / mpirun --version
  nvidia-smi -L
  CUDA_VISIBLE_DEVICES
  MPI_ROOT
- build:
  make clean || true
  make ARCH=sm_70 MPI_ROOT="${MPI_ROOT}"
- run:
  先跑 MPI 版本，再跑 NCCL 版本。
- benchmark output 用 tee 寫入：
  result/pingpong_cuda_result_<jobid>.txt

建議 baseline launcher：
先使用 tuned UCX launcher，避免已知 broken GDRCopy path：
  UCX_TLS=self,shm,cuda_copy,cuda_ipc
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-mpi
  mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-nccl

但 baseline 報告中必須明確寫出實際 launcher。

Baseline 成功條件：
- main-mpi build PASS
- main-nccl build PASS
- 2 GPUs visible
- 2 MPI ranks run
- MPI pingpong correctness PASS，即不得出現：
  ERROR: MPI pingpong test failed
- NCCL pingpong correctness PASS，即不得出現：
  ERROR: NCCL pingpong test failed
- MPI 輸出完整 size sweep
- NCCL 輸出完整 size sweep

若 baseline 失敗，請先分類：
- Build failure
- MPI launcher failure
- CUDA-aware MPI transport failure
- NCCL init failure
- NCCL runtime failure
- correctness failure
- timeout

============================================================
Optimization Submission 1
============================================================

優先目標：穩定 correctness 與輸出結構。

允許修改：
- run_pingpong_cuda.slurm
- Makefile
- 輸出格式
- 增加 parser script
- 不可改變 correctness 邏輯

建議：
1. 確保 MPI 與 NCCL 都用相同 GPU allocation。
2. 分開保存 MPI 與 NCCL raw output。
3. 輸出統一格式，便於解析，例如：
   RESULT,backend=MPI,size_bytes=...,time_s=...,gbps=...
   RESULT,backend=NCCL,size_bytes=...,time_s=...,gbps=...
4. 若 default MPI path 出現 GDRCopy error，固定使用：
   UCX_TLS=self,shm,cuda_copy,cuda_ipc
   --mca coll ^hcoll,ucc
   --mca pml ucx

提交後必須判斷：
- MPI correctness 是否 PASS
- NCCL correctness 是否 PASS
- 是否有完整 RESULT 行

============================================================
Optimization Submission 2
============================================================

目標：改善 timing 可信度。

允許修改：
- main-mpi.cu
- main-nccl.cu

要求：
1. 保留原本 size range。
2. 保留 warmup。
3. 對 NCCL timing 必須確保 stream synchronize 包含在 timing 範圍內。
4. 對 MPI timing 必須確保 device buffer 操作已同步，不得量到尚未完成的 transfer。
5. 可加入 MPI_Barrier 在每個 size 的 timing 前，避免 rank 間啟動偏差。
6. 可將 loop_count 依 size 動態調整，但必須維持公平比較，並在輸出中列出 loop_count。
7. 不得用只跑一次的數據取代平均。

建議輸出：
RESULT,backend=MPI,size_bytes=...,loop_count=...,avg_time_s=...,gbps=...
RESULT,backend=NCCL,size_bytes=...,loop_count=...,avg_time_s=...,gbps=...

============================================================
Optimization Submission 3
============================================================

目標：比較 MPI 與 NCCL 在相同條件下的 bandwidth / latency。

可加入：
1. 對每個 size 多跑 3 trials。
2. 輸出 avg/min/max。
3. 加入 result parser，例如 parse_pingpong_results.py。
4. 產生 CSV：
   result/pingpong_results_<jobid>.csv

CSV 欄位：
job_id,node,backend,size_bytes,loop_count,trial,avg_time_s,gbps,correctness

不得修改：
- 不得跳過慢的 size。
- 不得只保留 NCCL 或 MPI 其中一方。
- 不得在 correctness 失敗時輸出性能為有效結果。

============================================================
Optimization Submission 4
============================================================

目標：在 correctness PASS 的前提下嘗試小幅性能優化。

可嘗試：
1. 使用 cudaStreamNonBlocking 給 NCCL stream。
2. 增加 NCCL group semantics：
   ncclGroupStart()
   ncclSend(...)
   ncclRecv(...)
   ncclGroupEnd()
   但必須非常小心避免 deadlock。
3. MPI 版本可測：
   - UCX_TLS=self,shm,cuda_copy,cuda_ipc
   - UCX_TLS=self,shm,cuda_ipc
   - UCX_TLS=self,shm,cuda_copy
   但最多比較 2 個最有希望的 launcher。
4. 如果某 launcher correctness FAIL，必須標記 invalid。

不得嘗試：
- 不要把 MPI_Send/Recv 改成 host staging，除非作為額外 diagnostic，不能作為主結果。
- 不要改成單向 bandwidth；這題是 ping-pong round-trip 測試。

============================================================
Optimization Submission 5
============================================================

目標：final confirmation。

要求：
1. 使用目前最佳且 correctness PASS 的設定。
2. 跑完整 MPI + NCCL size sweep。
3. 保存 raw output、CSV、summary。
4. 確認 stderr 沒有 fatal error。
5. 若 stderr 只有 CUDA 12.8 對 sm_70 offline compilation 的 warning，可標記為 non-fatal。

============================================================
Final report
============================================================

請產生：
  /home/r14525078/HeCBench/src/pingpong-cuda/result/agent_summary.md

內容必須包含：

1. Environment
   - GPU model
   - CUDA_VISIBLE_DEVICES
   - number of GPUs
   - nvcc version
   - mpirun version
   - NCCL link / library info if available
   - MPI_ROOT
   - ARCH
   - launcher

2. Baseline
   - baseline job id
   - build PASS/FAIL
   - MPI run PASS/FAIL
   - NCCL run PASS/FAIL
   - first failed size if any
   - error message if any
   - whether data is valid

3. Submission history
   - optimization jobs 1 to 5
   For each:
   - job id
   - modification
   - hypothesis
   - result
   - correctness
   - performance if valid

4. Correctness summary
   - MPI pingpong correctness
   - NCCL pingpong correctness
   - any failed size
   - whether 2 ranks / 2 GPUs were used

5. Performance table
   For each size:
   - size_bytes
   - MPI avg_time_s
   - MPI GB/s
   - NCCL avg_time_s
   - NCCL GB/s
   - faster backend
   - speedup ratio

6. Interpretation
   - Which backend is faster for small sizes?
   - Which backend is faster for large sizes?
   - Is performance topology / NVLink limited?
   - Does NCCL outperform MPI for GPU-GPU transfer?
   - Does tuned UCX make MPI competitive?

7. Limitations
   - Only 2 ranks / 2 GPUs
   - Single node only
   - No multi-node test
   - No all GPU pair matrix unless explicitly performed
   - Submit limit: baseline + 5 optimization jobs

8. Final conclusion
   Choose one:
   - SUCCESS: MPI and NCCL correctness PASS, performance measured
   - PARTIAL: only one backend PASS
   - ENVIRONMENT ISSUE: launcher / NCCL / MPI transport unresolved
   - CODE ISSUE: benchmark logic or synchronization issue
   - INCONCLUSIVE: insufficient submissions

Important:
- 如果 MPI 與 NCCL 都 PASS，才可稱整個 pingpong-cuda SUCCESS。
- 如果只有 NCCL PASS，MPI FAIL，請寫 PARTIAL。
- 如果只有 MPI PASS，NCCL FAIL，請寫 PARTIAL。
- 如果 correctness FAIL，不能報告該性能為有效。
- 如果 baseline 無效，不可計算 speedup，只能說 final 建立了有效 baseline。