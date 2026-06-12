## Review Summary

* **verdict: NEEDS\_REVISION**

* **blocking issues:**
  1. **Submission 1 尚未明確要求 human approval 後才可修改 source。**  
     目前文字最後寫「Proceeding to Submission 1 proposal」，但沒有明確寫「停止並等待 human approval」。Mode C Submission 1 涉及新增 `impl=4`，必須先經批准。
  2. **CSV schema 不完整。**  
     目前 validation plan 只說產生 `submission_1/results.csv`，但沒有明確要求欄位包含：
     * `speedup_vs_impl1`
     * `speedup_vs_impl3`
     * `baseline_impl`
     * `mode_b_impl`
     * `profiler_status`
     * `raw_stdout_path`
     * `raw_stderr_path`
     * `result_type`
     * `mode_c_final_label`
  3. **final label criteria 不完整。**  
     有提到 `speedup_vs_impl3 >= 1.01` 才能 claim additional Mode C speedup，但沒有完整定義：
     * `SUCCESS_WITH_ADDITIONAL_SPEEDUP`
     * `PARTIAL_SUCCESS`
     * `INCONCLUSIVE`
     * `BLOCKED`
  4. **沒有明確規定 impl=4 必須逐 slice 超越 impl=3，不能只靠 aggregate。**  
     Mode C primary comparison 是 `impl=4 vs impl=3`。若 784 改善但 1024/2048 regression，不可用 aggregate 掩蓋。
  5. **profiler fallback 還不夠具體。**  
     有說 profiler 未跑、不得做 profiler-supported claims，但 Submission 1 plan 應明確記錄：
     * official timing 不使用 profiler
     * profiler 若未跑，`profiler_status=NOT_RUN`
     * 若 ncu unavailable，`profiler_status=UNAVAILABLE`
     * 無 profiler 時只能稱 hypothesis，不可稱 bottleneck conclusion

* **non-blocking issues:**
  1. 文獻與文件依據目前偏弱，只引用 CUDA Programming Guide / Best Practices。對 Stage 0 可接受，但若 Mode C 要寫「literature/documentation-informed」，後續應補 Nsight Compute、CUB reduction、Online Softmax 或 softmax kernel 相關來源。
  2. `impl=4_shape_specialized_large_reduce` 名稱可以接受，但建議更精確標示它的作用是「large-slice reduction-overhead candidate」，避免被誤解為全域 shape-specialization。
  3. Proposal 說「large-slice gains are valid Mode B dispatch-policy gains」，這正確；但 Submission 1 應明確禁止把 Mode B 的 speedup 當 Mode C speedup。
  4. `mode_C/main.cu` 是舊 source 的說明很重要，建議保留在 plan 中，避免 CLI 用錯檔案。

* **required fixes:**
  1. 加入「停止等待 human approval」。
  2. 補完整 CSV schema，尤其是 `speedup_vs_impl3`。
  3. 補 Mode C final label criteria。
  4. 補 per-slice comparison rule，不允許 aggregate hide regression。
  5. 補 profiler fallback 欄位與解釋規則。
  6. 補 small-slice classification：
     * 128/256 fallback 到 `impl=1`
     * 與 `impl=3` 應為 measurement-equivalent
     * 不得 claim Mode C speedup
  7. 補 rollback / acceptance rule：
     * 若 `impl=4` 未有效超越 `impl=3`，accepted candidate 仍為 Mode B `impl=3`

* **optional improvements:**
  1. Submission 1 先只做 reduction-overhead candidate，不同時做 block-size tuning，避免 attribution 混亂。
  2. 將 block-size tuning 留到 Submission 2。
  3. 若 Submission 1 成功，再用 profiler 或 ablation 做 causal explanation。
  4. 若 Submission 1 失敗，仍可記為 Mode C attempted optimization with `INCONCLUSIVE` 或 `REGRESSION`，不要丟棄結果。

***

## Detailed Review

### 1. 是否保留 Mode B accepted candidate

**通過。**

Stage 0 正確確認：

* runtime source 已含 `impl=0/1/2/3`
* runtime source 與 Mode B final source byte-identical
* `impl=3` dispatch 符合 final Mode B
* `mode_C/main.cu` 是舊 source，不應作 performance reference

這點非常重要，因為 Mode C 必須以 accepted Mode B `impl=3` 為 primary baseline。

***

### 2. 是否避免修改 `impl=0/1/2/3`

**部分通過。**

Proposal 說：

> Preserve all existing `impl=0/1/2/3` behavior.

這正確。  
但因為 Submission 1 要新增 `impl=4`，必須補上：

```text
No source modification may occur until human approval is recorded.
```

目前「No source change yet」不等於「不得在 approval 前修改」。需要明確寫。

***

### 3. 是否保留 CPU reference、tolerance、official cases、numSlice、repeat

**通過。**

Stage 0 明確要求保留：

* input generation
* CPU reference
* tolerance
* official cases
* `numSlice`
* `repeat`

***

### 4. 是否使用 sbatch only / 避免 login-node benchmark

**部分通過。**

Validation plan 說：

> Build inside a Slurm job.

但還應明確寫：

```text
All build and benchmark commands must be executed through sbatch.
Do not run ./main or any GPU benchmark binary on the login node.
```

目前沒有明確禁止 login-node execution。

***

### 5. 是否包含 paired `impl=1` 與 paired `impl=3`

**通過。**

Validation plan 要求 interleaved：

```text
impl=1
impl=3
impl=4
```

這符合：

* secondary comparison: `impl=4 vs impl=1`
* primary comparison: `impl=4 vs impl=3`

***

### 6. 是否計算 `speedup_vs_impl3`

**部分通過。**

Auditor plan 有提到：

```text
speedup_vs_impl3 >= 1.01
```

但 CSV schema 沒有定義 `speedup_vs_impl3` 欄位。  
這是 blocking issue。

Mode C claim 必須用 `speedup_vs_impl3`，不是只看 `speedup_vs_impl1`。

***

### 7. 是否報 raw stdout/stderr

**通過。**

Validation plan 有要求保存：

* raw stdout/stderr
* build log
* environment metadata
* Slurm job id

建議補上 raw path 欄位進 CSV。

***

### 8. 是否包含 profiler fallback

**部分通過。**

Stage 0 有正確說：

> Nsight Compute has not yet been run. Until profiler data exists, bottleneck statements are hypotheses only.

但 Submission 1 應明確規定：

```text
profiler_status=NOT_RUN if no profiler is executed.
No profiler-supported claim may be made without ncu output.
Official timing must be collected without profiler.
```

***

### 9. 是否區分 explanation 與 new optimization

**通過。**

Proposal 明確提出新增 `impl=4`，目標是超越 `impl=3`。這符合更新後 Mode C 定義：Mode C 不是 explanation-only。

***

### 10. 是否避免 cached-exp causality overclaim

**通過。**

Stage 0 明確寫：

* cached exp alone 未被證明
* 需要 ablation 才能做 causality claim

這符合要求。

***

### 11. 是否包含 proper CSV schema

**不通過。Blocking。**

目前只說：

```text
Produce submission_1/results.csv
```

但沒有 schema。

Mode C 至少需要：

```text
benchmark
mode
stage
submission_id
variant
impl
baseline_impl
mode_b_impl
dispatch_selected_impl
numSlice
sliceSize
repeat
trial_id
time_ms
impl1_time_ms
impl3_time_ms
speedup_vs_impl1
speedup_vs_impl3
correctness_status
measurement_validity
speedup_claim_valid
result_type
mode_c_final_label
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
```

***

### 12. 是否 run auditor

**部分通過。**

有 auditor plan，但應明確要求：

```text
Run self_consistency_auditor.py after results.csv is generated.
Preserve auditor_report.csv and contradiction_check.csv.
Do not promote candidate if auditor fails.
```

***

### 13. 是否 preserving all official slices / preventing hidden regression

**部分通過。**

有 official slices，但需補明確規則：

```text
Report all five slices.
Do not report only aggregate speedup.
If any large slice regresses vs impl=3, do not claim full Mode C success.
If small slices deviate from impl=3 by >=1%, classify as REGRESSION or CAUTION.
```

***

### 14. 是否定義 final label criteria

**不通過。Blocking。**

需要明確定義：

* `SUCCESS_WITH_ADDITIONAL_SPEEDUP`
* `PARTIAL_SUCCESS`
* `INCONCLUSIVE`
* `BLOCKED`

見下方修正版。

***

## Revised Text for CLI

以下可直接貼回 CLI，作為 Submission 1 proposal 的修正版。

```text
Mode C Stage 0 review result: NEEDS_REVISION before Submission 1 execution.

Revise the Submission 1 proposal as follows.

# Mode C Submission 1 Revised Proposal

## Goal

Attempt to improve beyond the accepted Mode B candidate:

- Mode B accepted candidate: impl=3
- variant: impl3_shape_dispatch_impl1_small_impl2_large

Primary comparison:
- Mode C candidate impl=4 vs Mode B impl=3

Secondary comparison:
- Mode C candidate impl=4 vs impl=1

A Mode C speedup claim is allowed only if speedup_vs_impl3 is valid.

## Candidate

Candidate:
- impl=4_shape_specialized_large_reduce

Source-change scope:
- Add a new impl=4 path in:
  /home/r14525078/HeCBench/src/softmax-cuda/main.cu

Preserve unchanged:
- impl=0
- impl=1
- impl=2
- impl=3
- CPU reference
- correctness tolerance
- input generation
- official cases
- numSlice
- repeat

Do not modify Mode B artifacts.

No source modification may occur until explicit human approval is recorded.

## Dispatch policy

For official slices:

- slice=128:
  impl=4 dispatches to unchanged impl=1 behavior.
  Expected result vs impl=3: MEASUREMENT_EQUIVALENT.
  No Mode C speedup claim is allowed.

- slice=256:
  impl=4 dispatches to unchanged impl=1 behavior.
  Expected result vs impl=3: MEASUREMENT_EQUIVALENT.
  No Mode C speedup claim is allowed.

- slice=784:
  impl=4 uses the new large-slice reduction candidate.

- slice=1024:
  impl=4 uses the new large-slice reduction candidate.

- slice=2048:
  impl=4 uses the new large-slice reduction candidate.

Non-official slices:
- fallback to unchanged impl=3 or impl=1 must be specified.
- No performance claims are allowed for non-official slices.

## Hypothesis

The current Mode B large-slice path uses impl=2 / softMax3.
It performs block-per-slice computation with shared-memory cached exponentials and block-level shared-memory reductions.

The impl=4 hypothesis is:
- keep the same mathematical softmax semantics,
- preserve large-slice block-level parallelism,
- reduce shared-memory reduction and synchronization overhead by using per-warp reductions plus compact cross-warp reduction.

This is a hypothesis only.
Do not claim this is a proven bottleneck without profiler or ablation evidence.

## Execution rules

After human approval only:

- Build must run through sbatch.
- Benchmark must run through sbatch.
- Do not run ./main or any GPU benchmark binary on the login node.
- Do not skip official cases.
- Do not delete slow, failing, or regressing cases.

## Validation plan

For every official slice:
- 128
- 256
- 784
- 1024
- 2048

Run paired implementations:
- impl=1
- impl=3
- impl=4

Use at least 3 independent trials per slice per implementation.

Use interleaved ordering per slice/trial when practical:
- impl=1
- impl=3
- impl=4

Save:
- raw stdout for every trial
- raw stderr for every trial
- build log
- Slurm stdout/stderr
- environment metadata
- Slurm job id

Generate:
- submission_1/results.csv
- submission_1/auditor_report.csv
- submission_1/contradiction_check.csv
- submission_1/summary.md
- submission_1/patch_summary.md

## Required CSV schema

submission_1/results.csv must include at least:

benchmark
mode
stage
submission_id
variant
impl
baseline_impl
mode_b_impl
dispatch_selected_impl
numSlice
sliceSize
repeat
trial_id
time_ms
impl1_time_ms
impl3_time_ms
speedup_vs_impl1
speedup_vs_impl3
correctness_status
measurement_validity
speedup_claim_valid
result_type
mode_c_final_label
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

Required values:
- benchmark=softmax-cuda
- mode=Mode_C
- submission_id=1
- baseline_impl=1
- mode_b_impl=3
- profiler_status=NOT_RUN unless profiler is explicitly run

## Measurement validity rules

- correctness FAIL -> INVALID and speedup_claim_valid=false.
- missing raw output -> LIMITED and speedup_claim_valid=false.
- missing impl=3 baseline -> INVALID and no Mode C speedup claim.
- improvement <1% vs impl=3 -> MEASUREMENT_EQUIVALENT.
- slower by >=1% vs impl=3 -> REGRESSION.
- high CV -> CAUTION or NOISY; do not claim speedup unless remeasured.

CV thresholds:
- CV <= 5%: VALID
- 5% < CV <= 15%: CAUTION
- CV > 15%: NOISY

## Mode C claim rules

Primary claim uses:
- speedup_vs_impl3

Secondary reporting may include:
- speedup_vs_impl1

Do not claim Mode C improvement if only speedup_vs_impl1 improves.
Mode C additional speedup requires valid speedup_vs_impl3.

Small slices:
- slice=128 and slice=256 must not be reported as Mode C optimization speedups if they dispatch to unchanged impl=1.
- If small slices deviate from impl=3 by >=1%, classify conservatively and investigate.

Large slices:
- slice=784, 1024, and 2048 may claim Mode C improvement only if:
  - correctness PASS,
  - speedup_vs_impl3 >= 1.01,
  - CV is stable,
  - raw output exists,
  - auditor passes.

Do not use aggregate speedup to hide per-slice regression.

## Profiler policy

Profiler is optional for Submission 1.

Official timing must be collected without profiler.

If profiler is not run:
- profiler_status=NOT_RUN
- no profiler-supported bottleneck conclusion is allowed

If profiler is unavailable:
- profiler_status=UNAVAILABLE
- this is a limitation, not a failure

If profiler is run:
- save ncu output
- distinguish profiler timing from official benchmark timing
- do not use profiler timing for official speedup

## Auditor rules

Run self_consistency_auditor.py after results.csv is generated.

Auditor must check:
- all official cases present
- correctness PASS before speedup claim
- speedup_vs_impl3 exists for impl=4 rows
- small slices not claimed as kernel improvements
- no profiler-supported claim without profiler data
- no cached-exp causality claim without ablation
- no aggregate-only success hiding per-slice regression

If auditor fails:
- do not promote impl=4
- keep impl=3 as accepted best candidate

## Final label criteria

SUCCESS_WITH_ADDITIONAL_SPEEDUP:
- all official slices correctness PASS,
- all official cases present,
- auditor PASS,
- no hidden regression,
- at least one large slice has valid speedup_vs_impl3 >= 1.01,
- no large-slice regression vs impl=3,
- small slices are measurement-equivalent or valid.

PARTIAL_SUCCESS:
- correctness PASS for all official slices,
- some large slices improve vs impl=3,
- but one or more large slices are measurement-equivalent or regress,
- no invalid correctness result.

INCONCLUSIVE:
- correctness PASS,
- measurements unstable or CV noisy,
- no reliable speedup_vs_impl3 claim.

BLOCKED:
- build failure,
- runtime failure,
- missing official cases,
- missing impl=3 baseline,
- missing raw output,
- auditor failure.

If impl=4 does not validly improve over impl=3:
- accepted best candidate remains Mode B impl=3.

## Reporting rules

Report per-slice results first.
Do not report only aggregate speedup.
Do not claim impl=4 is universal unless all official slices support it.
Do not claim cached-exp causality without ablation.
Do not claim profiler-supported bottleneck without profiler data.
Do not reinterpret Mode B speedups as Mode C speedups.

Stop after revising the proposal.
Do not modify source or submit sbatch until explicit human approval is given.
```

***

## Approval Decision

**CLI may not execute Submission 1 yet.**

目前只能批准 Stage 0 inspection 作為完成。  
Submission 1 需要先依上方修正版補齊：

1. CSV schema
2. `speedup_vs_impl3` rule
3. final label criteria
4. profiler fallback
5. no aggregate hiding regression rule
6. explicit human approval gate

修正後再貼回，我再判定是否 **APPROVE**。
