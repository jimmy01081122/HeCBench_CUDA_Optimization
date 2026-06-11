# softmax-cuda Mode B Round 1 Proposal

## 1. Robust Baseline Summary

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA: 12.8 module, `sm_70`
- Scheduler: Slurm
- Robust baseline job: 949640 on `gn1221.twcc.ai`

Baseline definition:
- `impl=1`, the existing optimized warp-level implementation.
- No `impl=0 -> impl=1` speedup is used or claimed.

Official cases:

| slice | batch | impl | mean ms | min ms | max ms | stddev | CV | correctness | measurement_validity |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 128 | 100000 | 1 | 0.137130 | 0.135259 | 0.140764 | 0.003147 | 0.022951 | PASS | VALID |
| 256 | 100000 | 1 | 0.304849 | 0.304549 | 0.305316 | 0.000410 | 0.001344 | PASS | VALID |
| 784 | 100000 | 1 | 1.450886 | 1.444919 | 1.454336 | 0.005189 | 0.003576 | PASS | VALID |
| 1024 | 100000 | 1 | 2.108163 | 2.107284 | 2.108716 | 0.000770 | 0.000365 | PASS | VALID |
| 2048 | 50000 | 1 | 2.237133 | 2.236458 | 2.238355 | 0.001060 | 0.000474 | PASS | VALID |

Auditor status:
- `contradiction_check.csv`: all PASS.

## 2. Bottleneck Hypothesis

Suspected bottleneck: redundant exponential evaluation in `softMax2`.

Why this is plausible:
- The optimized baseline kernel `softMax2` is one warp per slice. Each warp first scans the row for `max_`, then computes the denominator with:
  - `sum += expf(src[i * sliceSize + j] - max_)`
- It then makes a second full pass over the same elements and recomputes the same exponential:
  - `dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum`
- On V100, special-function throughput for `expf` is much lower than ordinary FP32 arithmetic, and the baseline performs two `expf` calls per element. For large slices this dominates more strongly because total elements per launch are high:
  - slice 784 and 1024 process 78.4M and 102.4M elements per trial batch.
  - slice 2048 processes 102.4M elements per trial batch.
- The robust baseline is stable, so a source-level change can be compared against `impl=1` without the high-CV ambiguity seen in topk.

## 3. Proposed Change

Make exactly one minimal source-level modification:

Add a new candidate implementation, `impl=2`, that computes softmax with one block per slice and caches the per-element exponential values in shared memory for supported official slice sizes.

Implementation shape:
- Keep `impl=1` unchanged as the robust baseline.
- Add `softMax3` or equivalent block-level kernel.
- Use one CUDA block per slice.
- Use `BLOCK_SIZE=256` threads.
- Shared memory buffer stores `exp(src - max)` for one slice.
- Compute row max with a block reduction.
- Compute `exp(src - max)` once per element, store it in shared memory, and reduce the sum.
- Write `dest = cached_exp / sum`.
- Dispatch `impl=2` only when explicitly requested by argv kernel id `2`; do not replace `impl=1`.
- Do not change CPU reference code.
- Do not change correctness tolerance.
- Do not remove or skip any official case.

Scope control:
- This is one kernel/dispatch addition only.
- It does not change the benchmark input generation, output verification, timing repeat, official shapes, or baseline rows.

## 4. Expected Improvement

Expected improvement:
- slice 784, 1024, and 2048: likely improvement if the saved `expf` pass outweighs shared-memory traffic and block-level reduction overhead.
- slice 256: possible small improvement or measurement-equivalent result.
- slice 128: higher regression risk because baseline `impl=1` is already very fast at 0.137 ms, and block-level/shared-memory overhead may dominate.

Expected risk:
- Shared memory for slice 2048 is 2048 floats, about 8 KiB, which is acceptable on V100.
- One block per slice means many blocks for the official batch sizes, so occupancy should remain sufficient, but per-slice work may have more synchronization than the existing warp-only implementation.
- The candidate may regress small slices, especially 128.
- If any official slice fails correctness, the round is invalid.

Success criterion:
- Candidate `impl=2` must be compared only against the `impl=1` robust baseline.
- Any speedup claim must be per-case and auditor-clean.
- If improvement is below 1%, classify as `MEASUREMENT_EQUIVALENT`, not a meaningful optimization.

## 5. Validation Plan

No sbatch job will be submitted until human approval is recorded.

If approved:
1. Apply the minimal source change adding `impl=2`.
2. Build only inside Slurm:
   - `module purge`
   - `module load cuda/12.8`
   - `make clean || true`
   - `make ARCH=sm_70`
3. Run all official cases with at least 3 trials:
   - `./main 100000 128 2 100`
   - `./main 100000 256 2 100`
   - `./main 100000 784 2 100`
   - `./main 100000 1024 2 100`
   - `./main 50000 2048 2 100`
4. Correctness must be PASS for every official case.
5. Report mean, min, max, stddev, and CV for each case.
6. Generate `rounds/round_1/results.csv` with:
   - `mode=Mode_B`
   - `round=1`
   - `variant=impl2_cached_exp`
   - `baseline_metric` from robust `impl=1`
   - `human_decision=Approved` only after explicit approval
7. Run `/home/r14525078/HeCBench/phase3/tools/self_consistency_auditor.py`.
8. Do not promote to final if auditor fails, correctness fails, or CV invalidates the speedup claim.

## Human Approval State

Current decision: pending human review.

No source code has been modified for this round. No sbatch job has been submitted.


---
# softmax-cuda Mode B Round 1 Revised Proposal

## Approval State

Human decision: `NEEDS_REVISION`.

This revised proposal supersedes the initial Round 1 proposal. No source code has been modified and no `sbatch` job has been submitted. Execution remains blocked until explicit human approval is provided.

## 1. Robust Baseline Summary

Environment:
- GPU: NVIDIA Tesla V100-SXM2-32GB
- CUDA: 12.8 module, `sm_70`
- Scheduler: Slurm
- Robust baseline job: 949640 on `gn1221.twcc.ai`

Baseline definition:
- `impl=1`, the existing optimized implementation.
- `impl=0 -> impl=1` is not used as a speedup claim.

Official robust baseline cases:

| slice | batch | impl | repeat | mean ms | min ms | max ms | stddev | CV | correctness | measurement_validity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 128 | 100000 | 1 | 100 | 0.137130 | 0.135259 | 0.140764 | 0.003147 | 0.022951 | PASS | VALID |
| 256 | 100000 | 1 | 100 | 0.304849 | 0.304549 | 0.305316 | 0.000410 | 0.001344 | PASS | VALID |
| 784 | 100000 | 1 | 100 | 1.450886 | 1.444919 | 1.454336 | 0.005189 | 0.003576 | PASS | VALID |
| 1024 | 100000 | 1 | 100 | 2.108163 | 2.107284 | 2.108716 | 0.000770 | 0.000365 | PASS | VALID |
| 2048 | 50000 | 1 | 100 | 2.237133 | 2.236458 | 2.238355 | 0.001060 | 0.000474 | PASS | VALID |

Auditor status:
- `/home/r14525078/HeCBench/phase3/softmax-cuda/mode_B_human_guided/contradiction_check.csv`: all PASS.

## 2. Bottleneck Hypothesis

Suspected bottleneck: the existing `impl=1` kernel performs two full passes that call `expf` for each output element.

Why this is plausible from source and baseline:
- `softMax2` uses one warp per slice.
- It first computes the denominator:
  - `sum += expf(src[i * sliceSize + j] - max_)`
- It then writes outputs with a second exponential evaluation:
  - `dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum`
- On V100, `expf` is expensive relative to basic FP32 arithmetic. Avoiding duplicate exponentials may help for larger slices, where the number of elements processed per benchmark trial is large.

Important attribution constraint:
- The proposed `impl=2` must be labeled as a compound candidate.
- Any measured result must be attributed to the compound candidate as a whole, not solely to cached exponentials.

## 3. Proposed Compound Candidate

Candidate label: `impl2_block_cached_exp_compound`.

This is not exactly one minimal optimization. It intentionally changes two things at once:
- Row parallelism: `impl=1` uses warp-per-slice; `impl=2` would use block-per-slice.
- Computation/memory strategy: `impl=1` recomputes `expf`; `impl=2` would cache `exp(src - max)` in shared memory.

Proposed source-level change after approval:
- Keep `impl=0` unchanged.
- Keep `impl=1` unchanged.
- Keep CPU reference unchanged.
- Keep correctness tolerance unchanged.
- Keep input generation unchanged.
- Keep official cases unchanged.
- Keep `numSlice` and `repeat` unchanged.
- Add a separate `impl=2` dispatch path.
- Add a block-level softmax kernel using one CUDA block per slice, `BLOCK_SIZE=256`, and shared memory to store per-element exponentials for the current row.

Attribution rule:
- If `impl=2` improves any case, the improvement is attributed to `impl2_block_cached_exp_compound`.
- The result cannot be described as proof that cached exponentials alone caused the improvement because row parallelism also changes.

## 4. Expected Improvement And Risk

Expected possible improvement:
- slice 784, 1024, 2048: most likely to benefit if avoiding duplicate `expf` outweighs block-level synchronization and shared-memory traffic.
- slice 256: possible small improvement or measurement-equivalent result.
- slice 128: highest regression risk because the existing warp-per-slice implementation is already very fast and may have lower overhead.

Expected risks:
- Block-per-slice parallelism may add synchronization overhead compared with warp-per-slice.
- Shared-memory caching may increase memory traffic and pressure.
- Small slices may regress even if large slices improve.
- Partial improvement is possible: only some official slices may improve.

Result interpretation:
- Per-slice results must be reported.
- Failures and regressions must remain visible.
- Do not average away failures or regressions.
- If only some official slices improve, the result is labeled as partial improvement, not full valid optimization.

## 5. Paired Validation Plan

No benchmark execution may occur until explicit human approval is recorded.

If approved, all benchmark execution must happen through `sbatch`; `./main` or any GPU benchmark binary must not be run on the login node.

Round 1 Slurm job must run paired measurements for every official slice:
- paired baseline: `impl=1`
- candidate: `impl=2`

Official cases:
- `numSlice=100000`, `sliceSize=128`, `repeat=100`
- `numSlice=100000`, `sliceSize=256`, `repeat=100`
- `numSlice=100000`, `sliceSize=784`, `repeat=100`
- `numSlice=100000`, `sliceSize=1024`, `repeat=100`
- `numSlice=50000`, `sliceSize=2048`, `repeat=100`

Trials:
- At least 3 paired trials per official slice.
- For each trial and slice, run both `impl=1` and `impl=2` in the same Slurm job.

Speedup baseline rule:
- If the paired `impl=1` baseline for a slice is PASS and stable, use paired `impl=1` mean time for speedup.
- If paired `impl=1` is invalid, missing, or unstable, use robust baseline job 949640 for that slice and mark this as a limitation in `notes`.
- Never use `impl=0 -> impl=1` as a speedup.

Raw artifact preservation:
- Preserve raw stdout and stderr for every trial.
- Every trial row must include:
  - `raw_stdout_path`
  - `raw_stderr_path`
  - `build_log_path`
  - `slurm_job_id`
  - `hostname`
  - `gpu_name`
  - `cuda_version`

Profiler:
- Profiler is optional for Round 1.
- Official timing must be collected without profiler.
- If profiler is not run, set `profiler_status=NOT_RUN`.
- If Nsight Compute is unavailable, set `profiler_status=UNAVAILABLE`.
- Do not make profiler-supported conclusions unless profiler data exists.

## 6. Round 1 CSV Schema

Round 1 result rows must use this schema:

```csv
benchmark,mode,round_id,human_decision,variant,impl,baseline_impl,numSlice,sliceSize,repeat,trial_id,time_ms,baseline_time_ms,speedup_vs_impl1,correctness_status,measurement_validity,speedup_claim_valid,result_type,mean_ms,min_ms,max_ms,stddev_ms,cv,raw_stdout_path,raw_stderr_path,build_log_path,slurm_job_id,hostname,gpu_name,cuda_version,profiler_status,notes
```

Required values:
- `benchmark=softmax-cuda`
- `mode=Mode_B`
- `round_id=1`
- `human_decision=Approved` only after explicit approval
- `variant=impl2_block_cached_exp_compound` for candidate rows
- `impl=2` for candidate rows
- `baseline_impl=1`

For paired baseline rows, use:
- `variant=paired_impl1_baseline`
- `impl=1`
- `baseline_impl=1`
- `speedup_vs_impl1=n/a`
- `speedup_claim_valid=false`
- `result_type=BASELINE`

## 7. Classification Rules

Apply these rules per slice; do not collapse per-slice failures into an average:

- correctness FAIL:
  - `result_type=INVALID`
  - `measurement_validity=INVALID`
  - `speedup_claim_valid=false`
- baseline missing or invalid:
  - `speedup_vs_impl1=n/a`
  - `speedup_claim_valid=false`
  - note whether robust baseline job 949640 was used as fallback
- improvement < 1%:
  - `result_type=MEASUREMENT_EQUIVALENT`
  - `speedup_claim_valid=false`
- slower by >= 1%:
  - `result_type=REGRESSION`
  - `speedup_claim_valid=false`
- high CV:
  - set `measurement_validity=CAUTION` or `NOISY`
  - do not claim speedup unless remeasured and validated
- source-level kernel change with PASS and valid speedup >= 1%:
  - `result_type=KERNEL_OPT`
  - `speedup_claim_valid=true`

Measurement validity guidance:
- `VALID`: correctness PASS and CV acceptable.
- `CAUTION`: correctness PASS but CV is elevated enough to qualify claims cautiously.
- `NOISY`: correctness PASS but CV is too high for speedup claims without remeasurement.
- `INVALID`: correctness FAIL, missing output, invalid baseline, or unusable measurement.

## 8. Correctness Failure Handling

Do not discard failed cases.

If any official slice fails:
- Keep the failed case in CSV and report.
- Set `correctness_status=FAIL`.
- Set `measurement_validity=INVALID`.
- Set `speedup_claim_valid=false`.
- Do not promote the candidate as a full valid optimization.

If some slices PASS and improve while other slices FAIL or regress:
- Report per-slice outcomes.
- Label the round as partial improvement at most.
- Do not claim a full official-sweep optimization.

## 9. Required Round Artifacts After Approval

If approved and executed later, Round 1 must produce:

- `rounds/round_1/plan.md`
- `rounds/round_1/patch_summary.md`
- `rounds/round_1/run.slurm`
- raw stdout and stderr files for every trial and implementation
- `rounds/round_1/results.csv` using the Round 1 schema above
- `rounds/round_1/auditor_report.csv`
- `rounds/round_1/round_summary.md`

## Current Stop Point

This proposal is revised and ready for human review.

No source code has been modified.
No `sbatch` job has been submitted.
