# softmax-cuda Mode B Round 2 Proposal

## Approval State

Human decision: `Approved`.

The human reviewer approved execution of `impl3_shape_dispatch_impl1_small_impl2_large`.

## 1. Round 1 Evidence

Round 1 candidate:
- `variant=impl2_block_cached_exp_compound`
- `impl=2`
- Interpretation: compound candidate combining block-per-slice row parallelism and shared-memory cached exponentials.
- Attribution constraint: Round 1 speedups cannot be attributed solely to cached exponentials.

Round 1 decision:
- Rejected as a full replacement for `impl=1`.
- Accepted only as a partial large-slice candidate.

Per-slice evidence from Round 1 paired measurements:

| slice | paired impl=1 mean ms | impl=2 mean ms | impl=2 correctness | impl=2 result |
|---:|---:|---:|---|---|
| 128 | 0.135152 | 0.554750 | PASS 3/3 | REGRESSION |
| 256 | 0.323384 | 0.594147 | PASS 2/3, FAIL 1/3 | INVALID |
| 784 | 1.434026 | 1.108087 | PASS 3/3 | KERNEL_OPT per-slice |
| 1024 | 2.068956 | 1.300902 | PASS 3/3 | KERNEL_OPT per-slice |
| 2048 | 2.212359 | 1.680560 | PASS 3/3 | KERNEL_OPT per-slice |

Round 2 follows this evidence:
- `slice=128`: `impl=2` regressed, so dispatch should preserve unchanged `impl=1`.
- `slice=256`: `impl=2` was invalid due to a correctness failure, so dispatch should preserve unchanged `impl=1`.
- `slice=784`, `1024`, `2048`: `impl=2` was valid and faster, so dispatch may select unchanged `impl=2`.

This is not a correctness-fix round for `impl=2`. The `slice=256` correctness issue in `impl=2` is not modified here and must be proposed separately if pursued.

## 2. Candidate Definition

Candidate label:
- `variant=impl3_shape_dispatch_impl1_small_impl2_large`
- candidate implementation id: `impl=3`

Result interpretation:
- Shape-aware dispatch candidate.
- Overall candidate result type should be `PARAM_TUNE` or shape-aware dispatch, not universal `KERNEL_OPT`.
- Per-slice large-shape improvements may be reported separately where valid.

Required dispatch map for official slices:

| slice | dispatch_selected_impl | reason |
|---:|---:|---|
| 128 | 1 | Round 1 `impl=2` regressed |
| 256 | 1 | Round 1 `impl=2` had correctness failure |
| 784 | 2 | Round 1 `impl=2` was PASS and faster |
| 1024 | 2 | Round 1 `impl=2` was PASS and faster |
| 2048 | 2 | Round 1 `impl=2` was PASS and faster |

Non-official slice handling:
- If `impl=3` receives a non-official `sliceSize`, dispatch to unchanged `impl=1` by default.
- No Phase 3 performance claim will be made for non-official slice sizes.
- Only official slices 128, 256, 784, 1024, and 2048 are used for Round 2 evaluation.

## 3. Source-Change Scope

Round 2 makes only the dispatch addition for `impl=3`.

Preserved unchanged:
- `impl=0`
- `impl=1`
- `impl=2`
- CPU reference
- correctness tolerance
- input generation
- official cases
- `numSlice`
- `repeat`

Explicit prohibitions:
- No approximate softmax.
- No skipped official cases.
- No modification to `impl=2` to fix `slice=256`.
- No `impl=0 -> impl=1` speedup claim.

Expected source-level implementation:
- Add a separate `kernel == 3` dispatch path.
- For official slice sizes:
  - `sliceSize == 128 || sliceSize == 256`: launch the unchanged `softMax2` path used by `impl=1`.
  - `sliceSize == 784 || sliceSize == 1024 || sliceSize == 2048`: launch the unchanged `softMax3` path used by `impl=2`.
- For non-official slice sizes: launch unchanged `softMax2` as conservative fallback.

## 4. Execution Rule

All build and benchmark execution must be submitted through Slurm with `sbatch`.

Do not run `./main` or any GPU benchmark binary on the login node.

Do not submit `sbatch` until human approval is recorded.

## 5. Validation Plan

Run paired same-job measurements:
- paired baseline: `impl=1`
- candidate: `impl=3`

Official cases:
- `numSlice=100000`, `sliceSize=128`, `repeat=100`
- `numSlice=100000`, `sliceSize=256`, `repeat=100`
- `numSlice=100000`, `sliceSize=784`, `repeat=100`
- `numSlice=100000`, `sliceSize=1024`, `repeat=100`
- `numSlice=50000`, `sliceSize=2048`, `repeat=100`

Trials:
- At least 3 independent paired trials for every official slice and every compared implementation.
- Prefer interleaved ordering per slice/trial:
  - `impl=1`
  - `impl=3`

Raw artifact preservation:
- Save raw stdout and stderr for every trial and implementation.
- Preserve build log.
- Preserve Slurm stdout/stderr.
- Record `slurm_job_id`, `hostname`, `gpu_name`, and `cuda_version`.

Auditor:
- Run `/home/r14525078/HeCBench/phase3/tools/self_consistency_auditor.py` after the job.
- Preserve auditor output.

Profiler:
- Profiler is not required in Round 2.
- If not run, set `profiler_status=NOT_RUN`.
- If Nsight Compute is unavailable, set `profiler_status=UNAVAILABLE`.
- Do not make profiler-supported conclusions without profiler data.
- Profiler absence is a limitation, not a failure.

## 6. Required CSV Schema

Use the Round 1 schema plus `dispatch_selected_impl`:

```csv
benchmark,mode,round_id,human_decision,variant,impl,dispatch_selected_impl,baseline_impl,numSlice,sliceSize,repeat,trial_id,time_ms,baseline_time_ms,speedup_vs_impl1,correctness_status,measurement_validity,speedup_claim_valid,result_type,mean_ms,min_ms,max_ms,stddev_ms,cv,raw_stdout_path,raw_stderr_path,build_log_path,slurm_job_id,hostname,gpu_name,cuda_version,profiler_status,notes
```

Required values:
- `benchmark=softmax-cuda`
- `mode=Mode_B`
- `round_id=2`
- `human_decision=Approved`
- candidate `variant=impl3_shape_dispatch_impl1_small_impl2_large`
- candidate `impl=3`
- `baseline_impl=1`

Candidate `dispatch_selected_impl` values:
- `1` for `sliceSize=128`
- `1` for `sliceSize=256`
- `2` for `sliceSize=784`
- `2` for `sliceSize=1024`
- `2` for `sliceSize=2048`

For paired baseline rows:
- `variant=paired_impl1_baseline`
- `impl=1`
- `dispatch_selected_impl=1`
- `baseline_impl=1`
- `speedup_vs_impl1=n/a`
- `speedup_claim_valid=false`
- `result_type=BASELINE`

## 7. Classification Rules

Apply rules per slice and preserve all cases:

- correctness FAIL:
  - `measurement_validity=INVALID`
  - `speedup_claim_valid=false`
  - candidate is not a full success if any official slice fails
- baseline invalid or missing:
  - `speedup_vs_impl1=n/a`
  - `speedup_claim_valid=false`
- improvement < 1%:
  - `result_type=MEASUREMENT_EQUIVALENT`
  - `speedup_claim_valid=false`
- slower by >= 1%:
  - `result_type=REGRESSION`
  - `speedup_claim_valid=false`
- high CV:
  - `measurement_validity=CAUTION` or `NOISY`
  - do not claim speedup unless remeasured
- `slice=128` and `slice=256`:
  - expected to dispatch to `impl=1`
  - measurement-equivalent results are acceptable
  - do not claim kernel optimization if dispatch selects `impl=1`
- `slice=784`, `1024`, and `2048`:
  - speedup may be claimed only if correctness PASS, paired baseline is valid, raw output exists, and repeated timing is stable
- Overall candidate:
  - classify as shape-aware dispatch / `PARAM_TUNE`
  - do not label the whole candidate as universal `KERNEL_OPT`
  - per-slice large-shape improvements may be reported separately

## 8. Reporting Rules

Round 2 report must:
- Report per-slice results.
- Include the dispatch map.
- Include `slice=128` and `slice=256`.
- Not hide regressions, failures, or measurement-equivalent slices.
- Not report only aggregate speedup.
- If aggregate speedup is reported, include the full per-slice table and dispatch map beside it.
- Not reinterpret Round 1 as a full success.
- Not claim `impl=2` universally improves `softmax-cuda`.

## 9. Required Artifacts After Approval

Round 2 must produce:
- `rounds/round_2/plan.md`
- `rounds/round_2/patch_summary.md`
- `rounds/round_2/run.slurm`
- raw stdout and stderr files for every trial and implementation
- `rounds/round_2/results.csv`
- `rounds/round_2/auditor_report.csv`
- `rounds/round_2/round_summary.md`

## Current Execution State

This proposal is approved for execution.
