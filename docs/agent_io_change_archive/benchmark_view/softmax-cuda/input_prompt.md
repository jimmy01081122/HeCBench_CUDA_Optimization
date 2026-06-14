# Input Prompt

Benchmark: `softmax-cuda`
Category: ML kernel

## Source prompt files

### `BASIC/softmax-cuda/GM/prompt.md`
```text
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
```

### `phase2/p1/softmax-cuda/prompt.md`
```text
# P1 Weak Prompt: softmax-cuda

Please optimize the CUDA benchmark at:

`/home/r14525078/p1/HeCBench/src/softmax-cuda`

Goal:
- Improve performance for `avg_ms by slice size and implementation`.
- Keep correctness.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

```bash
cd /home/r14525078/p1/HeCBench/src/softmax-cuda
mkdir -p /home/r14525078/p1/HeCBench/src/softmax-cuda/result
module purge
module load cuda/12.8
sbatch run_softmax_cuda.slurm
```

If `run_softmax_cuda.slurm` does not exist yet, create a Slurm script that builds and runs the benchmark command required for this benchmark.


Please inspect the code, make changes, run the benchmark on the server, and report what changed and what result you got.
```

### `phase2/p2/softmax-cuda/prompt.md`
```text
# P2 Medium Prompt: softmax-cuda

You are a CUDA performance engineer.

## Benchmark

- benchmark: softmax-cuda
- benchmark path: `/home/r14525078/p2/HeCBench/src/softmax-cuda`
- result path: `/home/r14525078/p2/HeCBench/src/softmax-cuda/result`
- category: softmax_kernel
- expected metric: avg_ms by slice size and implementation

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

- Benchmark naive and optimized implementations; preserve full slice-size sweep.
- Dispatch policy may depend on slice size; do not assume one kernel dominates all shapes.

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p2/HeCBench/src/softmax-cuda
mkdir -p /home/r14525078/p2/HeCBench/src/softmax-cuda/result
module purge
module load cuda/12.8
sbatch run_softmax_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p2/HeCBench/src/softmax-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
./main 100000 784 0 100
./main 100000 784 1 100
```

If `run_softmax_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


## Final Output

Write `agent_summary.md` in the result directory with:

- baseline status and metric
- all optimization submissions
- accepted/rejected result table
- final correctness
- final metric and speedup
- short explanation of the best strategy
```

### `phase2/p3/softmax-cuda/prompt.md`
```text
# P3 Strong Prompt: softmax-cuda

You are a CUDA performance engineer conducting a reproducible optimization experiment. Treat this prompt as an experimental protocol, not a casual request.

## Prompt Metadata

- benchmark: softmax-cuda
- canonical_name: softmax-cuda
- benchmark_category: softmax_kernel
- prompt_level: P3
- target_agent: server-side coding agent
- submission_limit: 5
- baseline_counts_as_submission: false
- required_gpus: 1
- requires_mpi: false
- requires_nccl: false
- expected_metric: avg_ms by slice size and implementation
- correctness_required: true

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

## Paths

- benchmark_path: `/home/r14525078/p3/HeCBench/src/softmax-cuda`
- result_path: `/home/r14525078/p3/HeCBench/src/softmax-cuda/result`

## Benchmark-Specific Requirements

- Benchmark naive and optimized implementations; preserve full slice-size sweep.
- Dispatch policy may depend on slice size; do not assume one kernel dominates all shapes.
- profiler requirement: Collect occupancy, expf instruction reduction, shared memory, and memory throughput notes.
- expected primary result type: KERNEL_OPT

## Server Run Instructions

Run on the server with Slurm only. Do not run GPU benchmarks directly on the login node.

Recommended workflow:

```bash
cd /home/r14525078/p3/HeCBench/src/softmax-cuda
mkdir -p /home/r14525078/p3/HeCBench/src/softmax-cuda/result
module purge
module load cuda/12.8
sbatch run_softmax_cuda.slurm
```

The Slurm script should build and run the baseline command below, saving stdout/stderr and benchmark output under `/home/r14525078/p3/HeCBench/src/softmax-cuda/result`:

```bash
make clean || true && make ARCH=sm_70
./main 100000 784 0 100
./main 100000 784 1 100
```

If `run_softmax_cuda.slurm` does not exist, create it before the baseline run. The script must include account `ACD115083`, request the required GPU count (`1`), print environment metadata, build the benchmark, run the command above, and tee benchmark output to a result `.txt` file.


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

- `result/softmax-cuda_<jobid>.out`
- `result/softmax-cuda_<jobid>.err`
- `result/softmax-cuda_result_<jobid>.txt`

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

Collect occupancy, expf instruction reduction, shared memory, and memory throughput notes.

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

### `phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/docs/prompt.md`
```text
Proceed to Mode B Round 1 for softmax-cuda ONLY.

Do not submit any sbatch job yet.

First produce a Round 1 proposal containing:

1. Robust baseline summary
   - official cases
   - impl=1 baseline metrics
   - correctness status
   - CV / measurement_validity

2. Bottleneck hypothesis
   - identify exactly one suspected bottleneck
   - explain why this bottleneck is plausible from the baseline/source

3. Proposed change
   - propose exactly one minimal source-level modification
   - do not modify correctness tolerance
   - do not remove any official case
   - do not compare impl=0 -> impl=1 as speedup
   - candidate must be compared against impl=1 robust baseline

4. Expected improvement
   - specify which slice sizes may improve
   - specify expected risk/regression

5. Validation plan
   - run all official softmax cases:
     slice=128,256,784,1024,2048
   - correctness must PASS for all cases
   - at least 3 trials
   - report mean/min/max/stddev/CV
   - run self_consistency_auditor.py
   - set human_decision=Approved only after I approve

Note : Make sure you consider the hardware condition and enviroment here. 
Stop after producing the proposal and wait for human approval.
```

### `phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/doc/prompt.md`
```text
Prepare Mode B Round 2 proposal for softmax-cuda.

Do not submit sbatch yet.
Stop after writing the proposal and wait for human approval.

# Round 1 decision

Round 1 candidate:
- variant=impl2_block_cached_exp_compound

Decision:
- Rejected as a full replacement for impl=1.
- Accepted only as a partial large-slice candidate.
- slice=128 regressed.
- slice=256 had correctness failure.
- slice=784, 1024, and 2048 improved with correctness PASS.

Important attribution rule:
- impl=2 is a compound candidate combining block-per-slice row parallelism and shared-memory cached exponentials.
- Do not attribute its speedup solely to cached exponentials.

# Round 2 goal

Create a shape-aware dispatch candidate.

This round is not a correctness-fix round for impl=2.
Do not modify impl=2 to fix slice=256 in this round.
Any impl=2 correctness fix for slice=256 must be proposed as a separate round.

# Required dispatch policy

Create a new candidate implementation:

- variant=impl3_shape_dispatch_impl1_small_impl2_large
- candidate impl id: impl=3

Dispatch map:

- slice=128 -> use unchanged impl=1 path
- slice=256 -> use unchanged impl=1 path
- slice=784 -> use unchanged impl=2 path
- slice=1024 -> use unchanged impl=2 path
- slice=2048 -> use unchanged impl=2 path

If the input slice size is not one of the official slices, do not make unsupported performance claims. The proposal must specify how non-official slice sizes are handled, but only official slices are used for Phase 3 evaluation.

# Required proposal contents

1. Explain why this dispatch follows Round 1 evidence:
   - impl=2 regressed on slice=128.
   - impl=2 was invalid on slice=256 due to correctness failure.
   - impl=2 was valid and faster on slice=784, 1024, and 2048.
   - Therefore, shape-aware dispatch preserves impl=1 for small or invalid slices and uses impl=2 only for validated large slices.

2. Confirm source-change scope:
   - impl=1 kernel remains unchanged.
   - impl=2 kernel remains unchanged.
   - CPU reference remains unchanged.
   - Correctness tolerance remains unchanged.
   - Input generation remains unchanged.
   - Official cases remain unchanged.
   - numSlice remains unchanged.
   - repeat remains unchanged.
   - No approximate softmax.
   - No skipped official cases.

3. Define candidate label:
   - variant=impl3_shape_dispatch_impl1_small_impl2_large
   - result interpretation: shape-aware dispatch candidate

4. Execution rule:
   - All build and benchmark executions must be submitted through sbatch.
   - Do not run ./main or any GPU benchmark binary on the login node.
   - Do not submit sbatch until human approval is recorded.

5. Validation plan:
   - paired impl=1 baseline
   - candidate impl=3 shape-aware dispatch
   - all official slices:
     - 128
     - 256
     - 784
     - 1024
     - 2048
   - at least 3 independent trials for every slice and every compared implementation
   - raw stdout and stderr for every trial
   - same CSV schema as Round 1
   - add column: dispatch_selected_impl
   - run self-consistency auditor
   - preserve auditor output

6. Required CSV fields:
   Use the Round 1 schema and include at least:

   benchmark
   mode
   round_id
   human_decision
   variant
   impl
   dispatch_selected_impl
   baseline_impl
   numSlice
   sliceSize
   repeat
   trial_id
   time_ms
   baseline_time_ms
   speedup_vs_impl1
   correctness_status
   measurement_validity
   speedup_claim_valid
   result_type
   mean_ms
   min_ms
   max_ms
   stddev_ms
   cv
   raw_stdout_path
   raw_stderr_path
   build_log_path
   slurm_job_id
   hostname
   gpu_name
   cuda_version
   profiler_status
   notes

7. Classification rules:
   - correctness FAIL -> measurement_validity=INVALID and speedup_claim_valid=false.
   - If any official slice FAILs, candidate is not a full success.
   - baseline invalid or missing -> speedup=n/a and speedup_claim_valid=false.
   - improvement <1% -> MEASUREMENT_EQUIVALENT.
   - slower by >=1% -> REGRESSION.
   - high CV -> CAUTION or NOISY; do not claim speedup unless remeasured.
   - slice=128 and slice=256 are expected to dispatch to impl=1; measurement-equivalent results are acceptable.
   - For slice=128 and slice=256, do not claim kernel optimization if the dispatch selects impl=1.
   - For slice=784, 1024, and 2048, speedup may be claimed only if correctness PASS, baseline valid, raw output exists, and repeated timing is stable.
   - Overall candidate result_type should be PARAM_TUNE or shape-aware dispatch. Do not label the whole candidate as a universal KERNEL_OPT if the implementation only selects between unchanged impl=1 and impl=2.
   - Per-slice large-shape improvements may be reported separately.

8. Profiler policy:
   - Profiler is not required in Round 2.
   - If not run, set profiler_status=NOT_RUN.
   - Do not make profiler-supported conclusions without profiler data.
   - Profiler unavailability or absence is a limitation, not a failure.

9. Reporting rules:
   - Report per-slice results.
   - Do not hide slice=128 or slice=256.
   - Do not report only aggregate speedup.
   - If aggregate speedup is reported, it must be accompanied by the full per-slice table and dispatch map.
   - Do not reinterpret Round 1 as full success.
   - Do not claim impl=2 universally improves softmax-cuda.

Stop after producing the Round 2 proposal.
Wait for human approval before any source modification, build, or sbatch submission.
```

### `phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_final/prompt.md`
```text
Perform Mode C final profiler analysis for the final confirmation result.

This is analysis-only.
Do not modify source.
Do not add a new candidate.
Do not run Submission 4.
Do not compute official speedup from profiler timing.
Do not run ./main on the login node.
All profiler execution must use sbatch.

Purpose:
Analyze the final confirmation result for impl4_shape_specialized_large_reduce.

Final confirmation accepted claims:
- slice=784:  speedup_vs_impl3=1.135540
- slice=1024: speedup_vs_impl3=1.048740

Not accepted as Mode C speedup:
- slice=128
- slice=256
- slice=2048, because speedup_vs_impl3=1.008239 < 1.01

Profiler analysis questions:
1. What resource differences exist between impl=3 and impl=4 for 784?
2. What resource differences exist between impl=3 and impl=4 for 1024?
3. What resource differences exist between impl=3 and impl=4 for 2048?
4. Can profiler explain why 784/1024 improve but 2048 is measurement-equivalent?
5. What profiler evidence is missing?

Profile only:
- slice=784: impl=3 and impl=4
- slice=1024: impl=3 and impl=4
- slice=2048: impl=3 and impl=4

Do not profile:
- slice=128
- slice=256

Use profiler diagnostic repeat, not official repeat:
- repeat_for_profiler=10
- launch_skip=2 or 3
- launch_count=1

Use kernel filtering if possible:
- softMax3 for impl=3
- softMax4 for impl=4

If kernel filtering fails:
- use launch skip/count fallback
- record actual kernel names observed
- do not profile all repeat=100 launches

Profiler output directory:
  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/final_profiler_analysis

Required outputs:
1. final_profiler_analysis/run.slurm
2. final_profiler_analysis/profiler_summary.csv
3. final_profiler_analysis/profiler_summary.md
4. final_profiler_analysis/raw/
5. final_profiler_analysis/ncu_reports/
6. final_profiler_analysis/final_profiler_interpretation.md

profiler_summary.csv columns:
benchmark,mode,stage,profiler_job_id,sliceSize,numSlice,repeat_for_profiler,impl,kernel_filter,launch_skip,launch_count,profiler_status,ncu_version,hostname,gpu_name,cuda_version,report_path,stdout_path,stderr_path,achieved_occupancy,registers_per_thread,static_shared_memory_bytes,dynamic_shared_memory_bytes,waves_per_sm,memory_throughput,warp_execution_efficiency,instruction_mix_summary,math_special_function_summary,stall_or_scheduler_summary,profiler_timing_ms,official_timing_used,notes

Required fixed values:
- benchmark=softmax-cuda
- mode=Mode_C
- stage=final_profiler_analysis
- official_timing_used=false

Allowed profiler_status:
- AVAILABLE
- PARTIAL
- UNAVAILABLE
- FAILED
- NOT_RUN

Interpretation rules:
- Do not make profiler-supported claims unless the relevant metric exists.
- If only resource allocation metrics are available, label evidence as LIMITED_PROFILER_EVIDENCE.
- Do not claim shared-memory footprint caused speedup unless supported by stronger evidence or ablation.
- Do not claim reduction-structure causality.
- Do not claim cached-exp causality.
- Do not use profiler timing for official speedup.
- Do not change final confirmation speedup values.

Final profiler interpretation must include:
1. What profiler supports.
2. What profiler does not support.
3. Whether profiler helps explain 784.
4. Whether profiler helps explain 1024.
5. Whether profiler helps explain 2048.
6. Whether further ablation would be needed.
7. Paper-safe wording.
8. Do-not-claim list.

Stop after producing profiler analysis artifacts.
```

### `phase3/softmax-cuda/mode_C_literature_profiler/submission_1/prompt.md`
```text
Approved for Submission 1 execution after human approval.

Execution conditions:
1. Do not modify impl=0/1/2/3.
2. Add impl=4 only as an additive candidate.
3. Use impl=3 as the primary Mode B baseline.
4. Compute and report speedup_vs_impl3 for every impl=4 official-case row.
5. Report per-slice results first.
6. Do not use aggregate speedup to hide any per-slice regression.
7. Do not claim profiler-supported bottleneck if profiler_status is NOT_RUN or UNAVAILABLE.
8. Do not claim cached-exp causality without ablation evidence.
9. If impl=4 does not validly improve over impl=3, keep impl=3 as the accepted best candidate.
10. Stop after Submission 1 results and wait for human audit before any further submission.
```
