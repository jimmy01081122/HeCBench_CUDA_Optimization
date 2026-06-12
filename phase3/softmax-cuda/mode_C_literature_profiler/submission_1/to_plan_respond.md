## Review Summary

* **verdict: APPROVE**

* **blocking issues:**
  * 無 blocking issue。修正版已補齊先前要求的 human approval gate、`speedup_vs_impl3`、paired `impl=3` baseline、CSV schema、profiler fallback、auditor rules、final label criteria 與 anti-aggregate-hiding rules。

* **non-blocking issues:**
  1. `PARTIAL_SUCCESS` 定義中寫「one or more large slices are measurement-equivalent or regress」稍微寬。若有 large-slice regression，但其他 slice 改善，是否仍標 `PARTIAL_SUCCESS` 要看 regression 幅度與主規劃器偏好。建議補一句：large-slice regression 必須逐 case 保留，且不得用 aggregate 掩蓋。
  2. `impl=4` 的「per-warp reductions plus compact cross-warp reduction」仍屬實作方向，尚未具體到 pseudocode 或 resource estimate；但 proposal 階段可接受。
  3. 若 profiler 不跑，Submission 1 只能做 timing-based optimization claim，不能做 bottleneck explanation。proposal 已寫，但 summary 也應重申。

* **required fixes:**
  * 無必修修正。可直接批准 Submission 1 進入「human approval 後實作與 sbatch」。

* **optional improvements:**
  1. 在 `patch_summary.md` 中要求明確列出：
     * 新增 kernel 名稱
     * `impl=4` dispatch map
     * reduction implementation difference from `softMax3`
     * 是否改變 shared-memory footprint
     * 是否改變 block size
  2. 在 `summary.md` 中要求列出：
     * per-slice `speedup_vs_impl3`
     * per-slice `speedup_vs_impl1`
     * per-slice CV
     * per-slice result\_type
     * final label
  3. 若 `impl=4` 改善但幅度接近 1%，建議追加 remeasurement，不要直接放入論文主張。

***

## Detailed Review

### 1. 是否保留 Mode B accepted candidate

通過。

proposal 明確以 Mode B accepted `impl=3` 作為 primary reference，並規定：

* primary comparison: `impl=4` vs `impl=3`
* secondary comparison: `impl=4` vs `impl=1`
* 若 `impl=4` 未有效改善，accepted best candidate 保持 Mode B `impl=3`

這符合 Mode C 定義。

***

### 2. 是否避免未批准修改 source

通過。

proposal 明確寫：

> No source modification may occur until explicit human approval is recorded.

並且 source-change scope 只允許 approval 後新增 `impl=4` path。這符合 human approval gate。

***

### 3. 是否保留 `impl=0/1/2/3`

通過。

proposal 明確要求保留：

* `impl=0`
* `impl=1`
* `impl=2`
* `impl=3`

且 Mode B artifacts 不得修改。

***

### 4. 是否保留 CPU reference、correctness tolerance、input generation、official cases、numSlice、repeat

通過。

proposal 明確列出全部 preserved unchanged。

***

### 5. 是否使用 sbatch only 並禁止 login node benchmark

通過。

proposal 明確寫：

* Build must run through sbatch.
* Benchmark must run through sbatch.
* Do not run `./main` or any GPU benchmark binary on login node.

***

### 6. 是否包含 paired `impl=1` 與 paired `impl=3`

通過。

Validation plan 要求每個 official slice 跑：

* `impl=1`
* `impl=3`
* `impl=4`

並至少 3 trials，且建議 interleaved ordering。這符合 Mode C primary / secondary comparison requirement。

***

### 7. 是否計算 `speedup_vs_impl3`

通過。

proposal 明確規定：

* Mode C primary claim uses `speedup_vs_impl3`
* CSV schema 包含 `speedup_vs_impl3`
* `impl=4` rows 必須有 `speedup_vs_impl3`
* 額外 Mode C speedup claim 需要 `speedup_vs_impl3 >= 1.01`

這是本次修正的核心，已符合要求。

***

### 8. 是否保存 raw stdout/stderr 與 artifact

通過。

proposal 要求保存：

* raw stdout
* raw stderr
* build log
* Slurm stdout/stderr
* environment metadata
* Slurm job id

並產生：

* `submission_1/results.csv`
* `submission_1/auditor_report.csv`
* `submission_1/contradiction_check.csv`
* `submission_1/summary.md`
* `submission_1/patch_summary.md`

足夠。

***

### 9. CSV schema 是否足夠

通過。

schema 包含 Mode C 所需關鍵欄位：

```text
speedup_vs_impl1
speedup_vs_impl3
baseline_impl
mode_b_impl
dispatch_selected_impl
correctness_status
measurement_validity
speedup_claim_valid
result_type
mode_c_final_label
profiler_status
raw_stdout_path
raw_stderr_path
```

可接受。

***

### 10. profiler fallback 是否完整

通過。

proposal 明確區分：

* profiler optional
* official timing 不使用 profiler
* profiler 未跑：`profiler_status=NOT_RUN`
* profiler unavailable：`profiler_status=UNAVAILABLE`
* 若 profiler 跑了，ncu output 必須保存，且 profiler timing 不可用於 official speedup

這符合要求。

***

### 11. 是否避免 profiler-supported overclaim

通過。

proposal 多處明確寫：

* hypothesis only
* no profiler-supported bottleneck conclusion without profiler evidence
* no profiler-supported claim without profiler data

可接受。

***

### 12. 是否避免 cached-exp causality overclaim

通過。

proposal 明確要求：

* 不可 claim cached-exp causality without ablation
* auditor 檢查 no cached-exp causality claim without ablation

可接受。

***

### 13. 是否避免 hidden regression

通過。

proposal 明確寫：

* report per-slice first
* do not report only aggregate speedup
* no aggregate-only success hiding per-slice regression
* large slices 需要 per-case 判定
* small slices deviation >=1% 要 conservative classify and investigate

可接受。

***

### 14. final label criteria 是否完整

通過。

proposal 定義：

* `SUCCESS_WITH_ADDITIONAL_SPEEDUP`
* `PARTIAL_SUCCESS`
* `INCONCLUSIVE`
* `BLOCKED`

且 Mode C 成功條件以 `speedup_vs_impl3` 為核心。

唯一非阻塞建議：如果 large-slice regression 出現，`PARTIAL_SUCCESS` 要非常保守，不能用其他改善掩蓋 regression。你目前已有「Do not use aggregate speedup to hide per-slice regression」，足夠。

***

## Revised Text for CLI

不需要完整重寫。可在批准回覆中追加以下補充文字，作為 execution condition：

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

***

## Approval Decision

**CLI may execute Submission 1 after explicit human approval.**

批准範圍：

```text
APPROVE: Submission 1 implementation and sbatch validation
```

限制：

```text
- Only additive impl=4 change is allowed.
- No modification to impl=0/1/2/3.
- No Mode C success claim until results are audited.
- No profiler-supported or cached-exp causality claim without evidence.
- Mode C improvement must be based on speedup_vs_impl3, not speedup_vs_impl1.
```
