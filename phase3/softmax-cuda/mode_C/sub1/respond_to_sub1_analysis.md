確認。依你的回答，Mode C 後續流程應調整為：

```text
Submission 1 已接受，但帶限制。
接下來先做 Analysis Track 1。
Analysis Track 1 目標：用 profiler plan / ablation plan 判斷 Submission 2 方向。
Submission 2 完成後，再做 Analysis Track 2。
Analysis Track 2 目標：重新判斷是否 final confirmation、是否還需要修正、或是否停止。
Profiler run 不計入三次 optimization submission，但必須只作解釋用途，不得用 profiler timing 算 speedup。
中文分析文件全部納入最終 artifact。
```

下面是你可以直接交給 CLI 的完整指令。

***

# 給 CLI 的下一步指令：Mode C Analysis Track 1

```text
Mode C Submission 1 has been reviewed by the human planner.

Decision:
  ACCEPT_WITH_LIMITATIONS

Submission 1 final label:
  SUCCESS_WITH_ADDITIONAL_SPEEDUP

Accepted additional-speedup claims:
  - slice=784: 1.131x vs impl=3
  - slice=1024: 1.049x vs impl=3

Not accepted as additional speedup:
  - slice=128
  - slice=256
  - slice=2048

Important constraints:
  - Do not claim impl=4 is universal kernel optimization.
  - Do not claim slice=128 has Mode C speedup.
  - Do not claim slice=256 has Mode C speedup.
  - Do not claim slice=2048 has Mode C speedup.
  - Do not claim profiler-supported bottleneck because profiler_status=NOT_RUN.
  - Do not claim cached-exp or warp-reduction causality without profiler or ablation evidence.
  - Do not use speedup_vs_impl1 as the main Mode C success metric.
  - Main Mode C metric remains speedup_vs_impl3.

Next step:
  Do not start Submission 2 yet.

Create an analysis-only track:

  /home/r14525078/HeCBench/phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1

This analysis track does not count as an optimization submission as long as:
  - no source code is modified
  - no new optimization candidate is introduced
  - no official speedup is claimed from profiler timing
  - all profiler or diagnostic execution, if any, uses sbatch
  - no login-node benchmark execution occurs

Purpose:
  Use Submission 1 evidence to decide whether Submission 2 should be:
    A. ablation
    B. profiler-informed optimization
    C. 2048-specific optimization
    D. final confirmation without more optimization
    E. stop and report Submission 1 as final Mode C result

Required outputs:

1. Create:
   analysis_track_1/submission_1_result_review_zh.md

   This Chinese report must summarize:
   - candidate: impl4_shape_specialized_large_reduce
   - accepted additional speedup:
     - slice=784: 1.131x vs impl=3
     - slice=1024: 1.049x vs impl=3
   - not accepted:
     - slice=128: no Mode C speedup claim
     - slice=256: no Mode C speedup claim
     - slice=2048: measurement-equivalent
   - correctness status:
     all official slices PASS
   - auditor status:
     PASS
   - final label:
     SUCCESS_WITH_ADDITIONAL_SPEEDUP, with limitations
   - limitations:
     - profiler NOT_RUN
     - ablation NOT_RUN
     - no causal attribution
     - impl=4 source patch not independently explained in causal terms
   - do-not-claim list

2. Create:
   analysis_track_1/profiler_feasibility_check.md

   Inspect only. Do not run benchmark binaries on login node.

   Check whether profiling is feasible:
   - which ncu
   - ncu --version, if safe
   - module availability, if safe
   - whether prior logs show profiler permission issues

   Report:
   - profiler_status_candidate:
     AVAILABLE / UNAVAILABLE / UNKNOWN
   - whether profiler requires sbatch
   - whether profiler should be attempted before Submission 2
   - exact reason if profiler seems unavailable
   - confirmation that profiler timing will not be used for official speedup

3. Create:
   analysis_track_1/profiler_plan.md

   If profiler appears feasible, propose an sbatch-only profiler plan.

   Recommended profiler comparison:
   - slice=784:
     impl=3 vs impl=4
   - slice=1024:
     impl=3 vs impl=4
   - slice=2048:
     impl=3 vs impl=4

   Profiler purpose:
   - understand why impl=4 improves 784 and 1024
   - understand why 2048 is only measurement-equivalent
   - inform whether Submission 2 should target reduction overhead, shared memory footprint, block size, or 2048-specific behavior

   Profiler metrics to collect if available:
   - achieved occupancy
   - register usage
   - shared memory usage
   - memory throughput
   - warp execution efficiency
   - instruction mix
   - math or special-function indicators if available

   Required rule:
   - profiler run is for explanation only
   - official speedup remains based on normal timing results
   - if profiler unavailable, record as limitation and continue with ablation planning

4. Create:
   analysis_track_1/ablation_plan.md

   Propose ablation options for Submission 2.

   The plan must compare possible directions:

   Option A: reduction-structure ablation
     Purpose:
       determine whether impl=4 improvement is due to warp/cross-warp reduction changes compared with impl=2.

   Option B: cached-exp attribution ablation
     Purpose:
       determine whether cached exponentials independently contribute to large-slice improvement.

   Option C: 2048-specific optimization
     Purpose:
       investigate why slice=2048 only reaches speedup_vs_impl3=1.008x and whether a separate path can improve it.

   Option D: block-size / resource tuning
     Purpose:
       tune large-slice path without changing small-slice behavior.

   For each option, report:
   - required source change
   - what hypothesis it tests
   - target slices
   - expected benefit
   - correctness risk
   - regression risk
   - whether it should count as Submission 2
   - whether it is worth doing

5. Create:
   analysis_track_1/direction_decision.md

   Recommend one Submission 2 direction.

   Must choose one of:
   - Submission 2 = ablation
   - Submission 2 = profiler-informed optimization
   - Submission 2 = 2048-specific optimization
   - Submission 2 = full-slice experiment
   - no Submission 2; proceed to final confirmation

   The recommendation must justify:
   - why this is better than blind tuning
   - how it builds on Submission 1
   - what evidence will be gained
   - whether it helps final paper claims
   - what risk it introduces

6. Create:
   analysis_track_1/main_planner_questions.md

   If there are unclear choices requiring human decision, list them explicitly.

Rules:
  - Do not modify source.
  - Do not create a new optimization implementation.
  - Do not run ./main on login node.
  - Do not submit optimization sbatch.
  - Profiler sbatch may be proposed but not executed unless explicitly approved.
  - Stop after writing all analysis_track_1 documents.
```

***

# 你現在的主規劃器決策狀態

目前應該標記為：

```text
Mode C Submission 1:
  ACCEPT_WITH_LIMITATIONS

Mode C current phase:
  Analysis Track 1 before Submission 2

Next decision needed:
  Whether Submission 2 should be ablation or 2048-specific / profiler-informed optimization
```

***

# Submission 2 前的決策規則

等 CLI 回傳 `analysis_track_1` 後，你再決定 Submission 2。不要現在直接決定。

我建議用下面規則判斷。

## 若 profiler 可用

優先做：

```text
Profiler run via sbatch
```

目的：

```text
理解 impl=4 在 784/1024 有效、2048 不顯著的原因。
```

接著再決定 Submission 2 是：

```text
ablation
```

還是：

```text
2048-specific optimization
```

***

## 若 profiler 不可用

優先做：

```text
ablation plan
```

Submission 2 建議偏向：

```text
A. reduction-structure ablation
```

或：

```text
C. 2048-specific optimization
```

不要盲目調所有參數。

***

## 若 analysis 顯示 Submission 1 已足夠

可以直接進：

```text
Submission 3 final confirmation
```

但這會留下限制：

```text
無 profiler
無 ablation
因果解釋較弱
```

如果你的論文想強調「為什麼有效」，不建議太早 final。

***

# Submission 2 後的 Analysis Track 2

你已決定：

```text
sub2 後再做一次 track，確定改進方向
```

這是合理的。先定義如下，等 Submission 2 完成後使用。

```text
Analysis Track 2:
  Post-Submission-2 Direction Review
```

目標：

```text
1. 比較 Submission 1 與 Submission 2。
2. 判斷哪個 candidate 是 final candidate。
3. 判斷是否需要 final confirmation。
4. 判斷是否還需要 Mode C-Extra full-slice experiment。
5. 判斷最終 label 是：
   - SUCCESS_WITH_ADDITIONAL_SPEEDUP
   - SUCCESS_EXPLANATION_ONLY
   - PARTIAL_SUCCESS
   - INCONCLUSIVE
   - BLOCKED
```

Analysis Track 2 應產出：

```text
analysis_track_2/submission_2_result_review_zh.md
analysis_track_2/comparison_sub1_sub2.md
analysis_track_2/final_candidate_decision.md
analysis_track_2/final_confirmation_plan.md
analysis_track_2/report_claims.md
```

***

# 中文分析文件納入最終 artifact 的建議

你回答 Q4 是「納入」。  
因此 final report 應列這些文件為 qualitative evidence：

```text
analysis_track_1/submission_1_result_review_zh.md
analysis_track_1/profiler_feasibility_check.md
analysis_track_1/profiler_plan.md
analysis_track_1/ablation_plan.md
analysis_track_1/direction_decision.md
analysis_track_2/submission_2_result_review_zh.md
analysis_track_2/comparison_sub1_sub2.md
analysis_track_2/final_candidate_decision.md
analysis_track_2/final_confirmation_plan.md
analysis_track_2/report_claims.md
```

這些文件的研究價值是：

```text
它們記錄 human planner 如何根據 evidence 決定是否繼續優化、做 ablation、做 profiler 或停止。
```

這正好支撐你的核心主題：

```text
AI 產生候選，human planner 根據 evidence 收斂策略。
```

***

# 目前不要做的事

```text
不要直接進 Submission 2。
不要直接 final confirmation。
不要把 Submission 1 寫成 universal success。
不要讓 profiler timing 算 speedup。
不要在沒有 ablation 前寫 cached-exp 或 warp-reduction 是單獨原因。
不要讓 analysis track 修改 source。
```

***

# 最終結論

依你的回答，目前 Mode C workflow 應調整為：

```text
Submission 1:
  已完成，ACCEPT_WITH_LIMITATIONS。

Analysis Track 1:
  立即開始，目的為 profiler / ablation / 2048-specific 方向判定。
  不算 optimization submission。

Submission 2:
  等 Analysis Track 1 完成後再決定。
  可能是 ablation 或 2048-specific / profiler-informed optimization。

Analysis Track 2:
  Submission 2 後執行，用來決定 final candidate 與 final confirmation。

Final confirmation:
  等 Analysis Track 2 完成後再做。
```

這個流程符合 Mode C，且比直接盲目進 Submission 2 更嚴謹。
