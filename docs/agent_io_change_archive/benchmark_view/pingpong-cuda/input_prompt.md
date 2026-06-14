# Input Prompt

Benchmark: `pingpong-cuda`
Category: Memory / communication

## Source prompt files

### `BASIC/pingppong/pingpong-cudaCODEX/prompt.md`
```text
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
```

### `phase2/p1/pingpong-cuda/prompt.md`
```text
# P1 Weak Prompt: pingpong-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/p1/HeCBench/src/pingpong-cuda`

Goal:
- Improve performance for `MPI and NCCL one-way transfer time and GB/s by size`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/p1/HeCBench/src/pingpong-cuda
mkdir -p /home/r14525078/p1/HeCBench/src/pingpong-cuda/result
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
sbatch run_pingpong_cuda.slurm
```

If `run_pingpong_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
```

### `phase2/p2/pingpong-cuda/prompt.md`
```text
# P2 Medium Prompt: pingpong-cuda

You are a CUDA performance engineer.

## Benchmark

- benchmark: pingpong-cuda
- benchmark path: `/home/r14525078/p2/HeCBench/src/pingpong-cuda`
- result path: `/home/r14525078/p2/HeCBench/src/pingpong-cuda`
- category: mpi_nccl_pingpong
- expected metric: MPI and NCCL one-way transfer time and GB/s by size

## Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
```bash
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
```
- required GPUs: 2
- requires MPI: true
- requires NCCL: true

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

- Use 2 MPI ranks / 2 GPUs. Compare MPI and NCCL without claiming general collective superiority.
- Record UCX_TLS and NCCL environment. Full size sweep required.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p2/HeCBench/src/pingpong-cuda
mkdir -p /home/r14525078/p2/HeCBench/src/pingpong-cuda/result
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
sbatch run_pingpong_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p2/HeCBench/src/pingpong-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-mpi
UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-nccl
```

If `run_pingpong_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`2`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
```

### `phase2/p3/pingpong-cuda/prompt.md`
```text
# P3 Strong Prompt: pingpong-cuda

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as an experimental protocol, not a casual request.

## Prompt Metadata

- benchmark: pingpong-cuda
- canonical_name: pingpong-cuda
- benchmark_category: mpi_nccl_pingpong
- prompt_level: P3
- target_agent: server-side coding agent
- submission_limit: 5
- baseline_counts_as_submission: false
- required_gpus: 2
- requires_mpi: true
- requires_nccl: true
- expected_metric: MPI and NCCL one-way transfer time and GB/s by size
- correctness_required: true

## Environment

- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA arch: sm_70
- Scheduler: Slurm
- Account: ACD115083
- Module:
```bash
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
```

## Paths

- benchmark_path: `/home/r14525078/p3/HeCBench/src/pingpong-cuda`
- result_path: `/home/r14525078/p3/HeCBench/src/pingpong-cuda/result`

## Benchmark-Specific Requirements

- Use 2 MPI ranks / 2 GPUs. Compare MPI and NCCL without claiming general collective superiority.
- Record UCX_TLS and NCCL environment. Full size sweep required.
- profiler requirement: Profiler optional; transport metadata and topology are required.
- expected primary result type: MEASURE_FIX

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p3/HeCBench/src/pingpong-cuda
mkdir -p /home/r14525078/p3/HeCBench/src/pingpong-cuda/result
module purge
module load nvhpc-24.11_hpcx-2.20_cuda-12.6
sbatch run_pingpong_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p3/HeCBench/src/pingpong-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-mpi
UCX_TLS=self,shm,cuda_copy,cuda_ipc mpirun -x UCX_TLS --mca coll ^hcoll,ucc --mca pml ucx -n 2 ./main-nccl
```

If `run_pingpong_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`2`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Hard Rules

1. Baseline does not count toward submission limit.
2. After baseline, at most 5 optimization sbatch submissions.
3. Before each submission, state:
   - modification
   - hypothesis
   - expected improvement
   - validation target
4. After each submission, read and summarize:
   - result `.out`
   - result `.err`
   - result `.txt`
   - generated CSV, if any
5. Do not delete correctness checks.
6. Do not loosen tolerance.
7. Do not modify CPU/reference validation to match GPU output.
8. Do not shrink input or skip official cases to fake speedup.
9. If correctness FAIL, the metric is invalid.
10. If only size 0 PASS, the result is invalid.
11. If output is missing, stderr has fatal error, or a case is waived/skipped, the result is invalid.
12. If improvement is below 1%, mark it as `MEASUREMENT_EQUIVALENT`, not a real speedup.
13. Do not claim all tests PASS if any case failed.
14. Preserve full raw output.

## Baseline Requirements

Run baseline before source modification.

Save:

- `result/pingpong-cuda_<jobid>.out`
- `result/pingpong-cuda_<jobid>.err`
- `result/pingpong-cuda_result_<jobid>.txt`

Record:

- job_id
- node
- CUDA_VISIBLE_DEVICES
- `nvcc --version`
- loaded modules
- benchmark command
- correctness
- primary metric
- full case list tested

If baseline fails, classify the failure:

- BUILD_FAIL
- COMPILE_FAIL
- RUNTIME_FAIL
- ENV_FAIL
- CORRECTNESS_FAIL
- NO_VALID_NONZERO_RESULT
- NO_PERFORMANCE_METRIC
- TIMEOUT

If baseline has no valid metric, do not compute speedup. Optimize toward correctness or measurement recovery first.

## Optimization Submission Rules

For each optimization submission:

1. Create backups before editing changed files using `.bak_agent`.
2. State the hypothesis before `sbatch`.
3. Submit exactly one sbatch job for that attempt.
4. Read out/err/result immediately after completion.
5. Classify result as accepted or rejected.
6. If rejected, preserve the reason and do not use its metric in final speedup.

## Correctness Gate

A result is valid only when correctness is PASS for every required case.

Invalid cases:

- correctness FAIL
- only size 0 PASS
- skipped or waived tests
- output missing
- stderr fatal error
- benchmark semantics changed
- input size/repeat reduced for final result

## Required Result Types

Classify every attempt using one of:

- KERNEL_OPT
- PARAM_TUNE
- MEASURE_FIX
- BUILD_FIX
- ENV_FIX
- CORRECT_FIX
- TOPOLOGY_MEASURE
- NO_EFFECT
- REGRESSION
- MEASUREMENT_EQUIVALENT

## CSV Result Schema

Generate or maintain a CSV under result path with at least:

```csv
benchmark,job_id,node,prompt_level,submission_index,variant,case,metric_name,metric_value,metric_unit,correctness,status,result_type,accepted,reject_reason,notes
```

If the benchmark naturally has multiple dimensions, encode them in `case`, for example `size_bytes=...`, `topk=...`, `slice=...`, or `num_gpus=...`.

## Variance / Repeated Trials

Final accepted candidate must include at least 3 trials when feasible. Report:

- mean
- min
- max
- stddev or coefficient of variation

If trial count cannot be increased due to submission or queue constraints, explain why in `agent_summary.md`.

## Profiler / Measurement Notes

For the final accepted candidate, collect profiler data if available without exceeding practical limits. If profiler cannot be run, include a measurement note explaining why. Required or preferred profiler focus:

Profiler optional; transport metadata and topology are required.

## Contradiction Check

Before writing the final conclusion:

1. Count PASS and FAIL cases from raw output.
2. Check that summary text matches those counts.
3. Check that speedup uses real measured baseline, not estimated baseline.
4. Check that rejected attempts are not used as final results.
5. If any case failed, do not write "all tests PASS".

## Final Output

Write `agent_summary.md` in the result path with:

- environment
- prompt level and submission limit
- baseline result
- submission history
- accepted/rejected attempts
- correctness table
- performance table
- variance statistics
- profiler or measurement notes
- result type classification
- final conclusion label:
  - SUCCESS
  - PARTIAL_SUCCESS
  - INVALID
  - BLOCKED
- next optimization recommendations
```
