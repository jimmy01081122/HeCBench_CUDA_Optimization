# Phase 3 Softmax-focused Human-AI Collaborative Optimization Report

## AI 輔助 CUDA 程式優化中的人機協作、Prompt 約束與可審核工作流

***

## 摘要

本報告整理 Phase 3 目前已完成的 HeCBench CUDA 人機協作優化實驗。Phase 3 原始設計涵蓋三個代表性 benchmark：`softmax-cuda`、`topk-cuda` 與 `shmembench-cuda`。其中，`softmax-cuda` 作為主要實驗對象，已完成 Mode B（Human-in-the-loop Guided Optimization）完整流程，包括 robust baseline、Round 1、Round 2 與 final confirmation；`topk-cuda` 與 `shmembench-cuda` 則完成 robust baseline，但尚未進入 Mode B optimization，因此在本報告中僅作為 optional supporting evidence。

本階段最重要的成果不是單純取得某個最高 speedup，而是證明：在人類審查、paired baseline、correctness gate、per-case analysis 與 artifact audit 的共同作用下，AI agent 產生的 partial candidate 可以被收斂為可驗證、可解釋、可審核的工程策略。

`softmax-cuda` 的 Mode B 流程中，Agent 在 Round 1 提出 `impl=2` compound candidate，即 block-per-slice + shared-memory cached exponentials。該候選版本在 large slices 上有效提升效能，但在 small slices 上出現 regression 或 correctness failure，因此被人類審查拒絕作為 universal replacement。隨後 Round 2 根據 Round 1 evidence 建立 `impl=3` shape-aware dispatch policy：對 `slice=128` 與 `slice=256` 保留既有 `impl=1` optimized baseline；對 `slice=784`、`slice=1024` 與 `slice=2048` 選擇 `impl=2` path。Final confirmation 顯示所有 official cases 均 3/3 correctness PASS，large slices 分別取得 1.392x、1.699x 與 1.337x 的有效改善。該結果應分類為 `PARAM_TUNE / SHAPE_AWARE_DISPATCH`，而不是 universal `KERNEL_OPT`。

本報告同時明確保留限制：Mode C 尚未執行；`topk-cuda` 與 `shmembench-cuda` 尚未完成 Mode B optimization；Profiler 在 softmax Mode B 中未執行，因此不得宣稱 profiler-supported bottleneck conclusion；`impl=2` 是 compound candidate，因此不能把 large-slice 改善單獨歸因於 cached exponentials。

***

## 1. 報告目的與範圍

本報告目的有三個：

1. 整理目前 Phase 3 專案狀態與 artifact 狀況。
2. 以 `softmax-cuda` Mode B 作為主要成果，分析人機協作如何將 AI agent 的 partial candidate 收斂為可用策略。
3. 將 `topk-cuda` 與 `shmembench-cuda` 作為 optional supporting evidence，說明 robust baseline、measurement noise、diagnostic failure 與 official comparison 收斂的重要性。

本報告不包含：

```text
1. Mode C 結果分析
2. Mode C 最佳協作方式結論
3. Mode C 加速程度結論
4. topk-cuda Mode B optimization 成果
5. shmembench-cuda Mode B optimization 成果
```

目前 Mode C artifact 狀態為 `NOT_STARTED / NOT_FOUND`。因此，本報告中的 Mode C 章節僅保留結構與待填項目，不作任何效能或協作方式結論。

***

## 2. 研究背景

本研究探討 AI 輔助 CUDA 程式優化時，如何透過 prompt 約束、人類審查、agent 執行、correctness gate、baseline 管理、variance 檢查與 artifact auditing，建立可重現、可審核、可放入論文的實驗流程。

先前 Phase 2 已整理 10 個 HeCBench benchmark 的 AI-assisted optimization 結果，並指出不同 benchmark 的成果本質差異明顯。例如，`softmax-cuda` 屬於較明確的 kernel / primitive optimization 類案例；`topk-cuda` 的有效策略主要與 workspace reuse、block size tuning 或 radix selection 有關；`p2p-cuda` 更接近 topology-aware measurement；`allreduce-cuda` 則是 environment / launcher repair，而非 kernel optimization。這些結果顯示，若不區分 `KERNEL_OPT`、`PARAM_TUNE`、`MEASURE_FIX`、`ENV_FIX` 與 `TOPOLOGY_MEASURE`，AI agent 的成果容易被過度宣稱或錯誤分類。 [\[arxiv.org\]](https://arxiv.org/abs/2510.00555), [\[arxiv.org\]](https://arxiv.org/abs/2403.17134)

Phase 2 也顯示，高約束 prompt 通常包含 baseline、correctness、sbatch、submission limit 與 raw output preservation 等條款；這些約束能降低偽加速與不可審核結果的風險。不過，prompt inventory 也指出多數 prompt 仍缺少 profiler 指標與變異統計要求，這成為 Phase 3 強化的核心方向。 [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1)

因此 Phase 3 的定位不是單純「讓 AI 再優化更多 benchmark」，而是以少量代表案例研究：

```text
AI 產生候選方向；
人類判斷有效邊界；
robust baseline 與 paired measurement 驗證差異；
correctness gate 排除錯誤結果；
最終將 partial improvement 收斂為可審核策略。
```

***

## 3. 專案與 Artifact 狀態

### 3.1 目前專案狀態

依據目前 artifact scan 與 status-freeze 結果，Phase 3 狀態如下：

| Benchmark         | Mode A                        | Mode B         | Mode C                    | 目前角色                         |
| ----------------- | ----------------------------- | -------------- | ------------------------- | ---------------------------- |
| `softmax-cuda`    | SUCCESS                       | SUCCESS        | NOT\_STARTED / NOT\_FOUND | 主成果                          |
| `topk-cuda`       | SUCCESS                       | BASELINE\_ONLY | NOT\_STARTED / NOT\_FOUND | optional supporting evidence |
| `shmembench-cuda` | INVALID / diagnostic findings | BASELINE\_ONLY | NOT\_STARTED / NOT\_FOUND | optional supporting evidence |

### 3.2 Authoritative files

本報告應以以下檔案作為主要證據：

```text
phase3/softmax-cuda/mode_B_human_guided/final/results.csv
phase3/softmax-cuda/mode_B_human_guided/final/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/final/main.cu
phase3/softmax-cuda/mode_B_human_guided/final/auditor_report.csv
phase3/softmax-cuda/mode_B_human_guided/final/contradiction_check.csv
phase3/softmax-cuda/mode_B_human_guided/final/state_report.md
phase3/softmax-cuda/mode_B_human_guided/final/temp_commition.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_1/patch_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/round_summary.md
phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/patch_summary.md
phase3/reports/mode_A_report.md
```

### 3.3 Stale / superseded / do-not-cite files

以下檔案目前不應作為正式報告證據，除非後續更新或明確標記 superseded：

```text
phase3/softmax-cuda/mode_B_human_guided/agent_summary.md
phase3/softmax-cuda/mode_B_human_guided/results.csv
phase3/reports/mode_B_report.md
phase3/reports/MODE_B_REPORT.md
phase3/mode_A_agent_only
```

原因：

```text
1. parent-level agent_summary.md 與 results.csv 仍停留在 robust baseline 或舊 schema。
2. mode_B_report.md / MODE_B_REPORT.md 未反映 softmax Round 1、Round 2 與 final confirmation。
3. phase3/mode_A_agent_only 是 topk Mode A 的重複目錄。
```

### 3.4 Git 狀態待人工決策項目

目前有以下待人工處理項目：

```text
deleted:
  phase3/softmax-cuda/mode_B_human_guided/rounds/round_2/advice

untracked:
  phase3/softmax-cuda/mode_B_human_guided/final/state_report.md
  phase3/softmax-cuda/mode_B_human_guided/final/temp_commition.md
```

建議：

```text
state_report.md:
  應納入追蹤，作為 qualitative evidence。

temp_commition.md:
  內容有 timeline / decision log 價值，但檔名像臨時檔。
  建議後續改名為 decision_timeline.md 或 collaboration_timeline.md。
```

目前不建議自動刪除或改名；應先保留原始狀態。

***

## 4. 實驗規則與審查標準

本研究所有 Phase 3 結果均依照以下規則審查：

```text
1. 所有 GPU benchmark execution 必須透過 sbatch。
2. 禁止 login node 直接執行 ./main 或任何 GPU benchmark binary。
3. correctness FAIL → result invalid。
4. baseline invalid → speedup=n/a。
5. improvement <1% → MEASUREMENT_EQUIVALENT，不得宣稱顯著加速。
6. high CV → speedup claim invalid unless remeasured。
7. ENV_FIX / MEASURE_FIX / TOPOLOGY_MEASURE 不得報告為 KERNEL_OPT。
8. profiler unavailable 是 limitation，不是 experiment failure。
9. P1/P2/P3 或 Mode A/B/C 比較必須使用相同 official cases。
10. 不得刪除 slow、failing 或 regressing cases。
```

### 4.1 Result type 定義

本報告使用以下主要分類：

```text
BASELINE
BASELINE_COMPARISON
KERNEL_OPT
PARAM_TUNE
SHAPE_AWARE_DISPATCH
MEASURE_FIX
ENV_FIX
TOPOLOGY_MEASURE
MEASUREMENT_EQUIVALENT
REGRESSION
INVALID
INCONCLUSIVE
```

### 4.2 Measurement validity

結果還需分離 correctness 與 measurement validity：

```text
correctness_status:
  PASS / FAIL / PARTIAL / NOT_PROVIDED

measurement_validity:
  VALID / CAUTION / NOISY / INVALID / LIMITED / DIAGNOSTIC_FAIL

speedup_claim_valid:
  true / false
```

這樣可以避免「correctness PASS」被錯誤解讀為「speedup claim 有效」。

***

## 5. Phase 1 / Phase 2 摘要

### 5.1 Benchmark taxonomy

Phase 1 將實驗 benchmark 分類為：

```text
AI Primitive / Kernel Optimization:
  softmax-cuda
  topk-cuda
  moe-cuda
  moe-align-cuda

Memory-System / Measurement Benchmark:
  prefetch-cuda
  shmembench-cuda
  p2p-cuda

Multi-GPU / Communication / Environment:
  allreduce-cuda
  pingpong-cuda
  simpleMultiDevice-cuda
```

其中 `softmax-cuda` 適合作為高優化空間案例，`topk-cuda` 適合作為中等優化空間案例，`shmembench-cuda` 適合作為低優化空間與 measurement boundary 案例。

### 5.2 P1 / P2 / P3 prompt comparison

Phase 2 對 prompt 強度進行分層：

```text
P1: 弱約束
P2: 中約束
P3: 強約束
```

Phase 2 的核心發現是：P3 不一定產生最高 speedup，但能顯著提升結果的可審核性，特別是在 baseline validity、correctness gate、raw output preservation、submission limit、CSV schema 與 contradiction check 方面。既有 prompt inventory 顯示，較完整的 prompt 多數已包含 correctness、baseline、sbatch、submission limit 與 raw output 等關鍵條款；而較短 prompt 則缺少 profiler 與變異統計要求。 [\[arxiv.org\]](https://arxiv.org/html/2603.07169v1)

### 5.3 對 Phase 3 的啟示

Phase 2 的啟示如下：

```text
1. 需要 robust baseline。
2. 需要 paired measurement。
3. 需要 per-case result，不得只看 aggregate。
4. 需要清楚區分 result_type。
5. 需要人類審查 AI partial candidate 的有效邊界。
```

這些原則直接被用於 Phase 3 Mode B。

***

## 6. Phase 3 Mode A 摘要

Mode A 是 agent-only baseline，其主要目的不是產生優化，而是揭露 measurement risk 與 baseline stability。

### 6.1 softmax-cuda Mode A

Mode A 建立了 `impl=0` naive reference 與 `impl=1` existing optimized baseline 的差異。`impl=1` 明顯優於 `impl=0`，但這是既有 baseline 差異，不是 Phase 3 agent speedup。

因此 Phase 3 後續規則明確規定：

```text
impl=0 = naive reference
impl=1 = existing optimized baseline
Mode B/C 的 baseline 必須是 impl=1
不得將 impl=0 → impl=1 計為 Phase 3 speedup
```

### 6.2 topk-cuda Mode A

Mode A 暴露了 topk-cuda 的 high-CV pseudo-speedup 風險。部分 cases 在沒有程式修改的情況下出現約 1.4x 表面加速，但原因是 baseline CV 過高，而非有效優化。這證明 topk-cuda 進入 Mode B 前必須先重建 robust baseline。

### 6.3 shmembench-cuda Mode A

Mode A 顯示 shmembench-cuda 的原始 block-size sweep 不完全可用。僅 `block_size=256, variant=original` 通過 correctness，`128/512/1024` 應保留為 diagnostic failures，而不是納入 official speedup。

***

## 7. Mode B Robust Baseline

Mode B 開始前，三個 benchmark 均完成 robust baseline。

### 7.1 softmax-cuda robust baseline

```text
official cases: slice=128,256,784,1024,2048
baseline implementation: impl=1
correctness: all PASS
measurement_validity: all VALID
```

這是後續 Round 1、Round 2 與 final confirmation 的比較基準。

### 7.2 topk-cuda robust baseline

```text
official cases: 14 cases
trials: 7
correctness: all PASS
measurement_validity: 12 VALID, 2 CAUTION, 0 NOISY
```

topk-cuda 尚未開始 Mode B optimization。  
因此它在本報告中只作為 robust baseline 與 measurement noise supporting evidence。

### 7.3 shmembench-cuda robust baseline

```text
official validated comparison:
  block_size=256, variant=original

diagnostic sweep:
  block_size=128,512,1024
```

shmembench-cuda 尚未開始 Mode B optimization。  
因此它在本報告中只作為 low-optimization-space 與 diagnostic-boundary supporting evidence。

***

## 8. softmax-cuda Mode B Round 1

### 8.1 Candidate

Round 1 candidate：

```text
impl2_block_cached_exp_compound
```

該 candidate 是 compound candidate，同時改變：

```text
1. row parallelism:
   warp-per-slice → block-per-slice

2. computation / memory strategy:
   recompute exp → shared-memory cached exp
```

因此，任何改善都不能單獨歸因於 cached exponentials。

### 8.2 Round 1 結果

| slice | paired impl=1 mean ms | impl=2 mean ms | correctness        | result                        |
| ----: | --------------------: | -------------: | ------------------ | ----------------------------- |
|   128 |              0.135152 |       0.554750 | PASS 3/3           | REGRESSION                    |
|   256 |              0.323384 |       0.594147 | PASS 2/3, FAIL 1/3 | INVALID                       |
|   784 |              1.434026 |       1.108087 | PASS 3/3           | valid large-slice improvement |
|  1024 |              2.068956 |       1.300902 | PASS 3/3           | valid large-slice improvement |
|  2048 |              2.212359 |       1.680560 | PASS 3/3           | valid large-slice improvement |

### 8.3 Round 1 判定

```text
Round 1 = PARTIAL_SUCCESS
Full replacement = REJECTED
Large-slice candidate = KEEP
```

原因：

```text
slice=128 regression
slice=256 correctness failure
slice=784/1024/2048 有效改善
```

這一步證明了 human review 的必要性。若只看 large-slice speedup，AI agent 可能把 candidate 錯誤包裝為 full success。但人類審查拒絕了 universal replacement 的說法，並將方向轉為 shape-aware dispatch。

***

## 9. softmax-cuda Mode B Round 2

### 9.1 Human-guided correction

Round 2 根據 Round 1 evidence 建立 dispatch policy：

| slice | selected impl | reason                             |
| ----: | ------------: | ---------------------------------- |
|   128 |        impl=1 | Round 1 impl=2 regression          |
|   256 |        impl=1 | Round 1 impl=2 correctness failure |
|   784 |        impl=2 | Round 1 impl=2 PASS and faster     |
|  1024 |        impl=2 | Round 1 impl=2 PASS and faster     |
|  2048 |        impl=2 | Round 1 impl=2 PASS and faster     |

Candidate：

```text
impl3_shape_dispatch_impl1_small_impl2_large
```

### 9.2 Round 2 結果

| slice | selected impl | candidate mean ms | paired impl=1 mean ms | correctness | result\_type            |  speedup |
| ----: | ------------: | ----------------: | --------------------: | ----------- | ----------------------- | -------: |
|   128 |             1 |          0.135674 |              0.144732 | PASS        | MEASUREMENT\_EQUIVALENT | 1.066763 |
|   256 |             1 |          0.306408 |              0.305251 | PASS        | MEASUREMENT\_EQUIVALENT | 0.996224 |
|   784 |             2 |          1.107988 |              1.437362 | PASS        | PARAM\_TUNE             | 1.297273 |
|  1024 |             2 |          1.300765 |              2.082344 | PASS        | PARAM\_TUNE             | 1.600861 |
|  2048 |             2 |          1.670514 |              2.213330 | PASS        | PARAM\_TUNE             | 1.324940 |

### 9.3 Round 2 判定

```text
Round 2 = ACCEPT
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

Round 2 解決了 Round 1 的問題：

```text
1. small slices 不再使用 impl=2，避免 regression / invalid。
2. large slices 保留 impl=2 的有效改善。
3. 所有 official cases correctness PASS。
```

***

## 10. softmax-cuda Final Confirmation

### 10.1 Final confirmation 設定

```text
Slurm job: 949717
Node: gn1228.twcc.ai
Candidate: impl3_shape_dispatch_impl1_small_impl2_large
Profiler status: NOT_RUN
```

### 10.2 Final result table

| slice | dispatch | baseline mean ms | candidate mean ms |   speedup | correctness | result\_type            | speedup\_claim\_valid |
| ----: | -------: | ---------------: | ----------------: | --------: | ----------- | ----------------------- | --------------------- |
|   128 |   impl=1 |         0.134869 |          0.134574 | 1.002197x | PASS        | MEASUREMENT\_EQUIVALENT | false                 |
|   256 |   impl=1 |         0.321793 |          0.321505 | 1.000895x | PASS        | MEASUREMENT\_EQUIVALENT | false                 |
|   784 |   impl=2 |         1.442716 |          1.036402 | 1.392043x | PASS        | PARAM\_TUNE             | true                  |
|  1024 |   impl=2 |         2.104045 |          1.238443 | 1.698944x | PASS        | PARAM\_TUNE             | true                  |
|  2048 |   impl=2 |         2.237452 |          1.672904 | 1.337466x | PASS        | PARAM\_TUNE             | true                  |

### 10.3 Final interpretation

Final confirmation 顯示：

```text
1. 五個 official slices 均 3/3 correctness PASS。
2. slice=128/256 dispatch 到 impl=1，結果為 measurement-equivalent。
3. slice=784/1024/2048 dispatch 到 impl=2，保留有效 large-slice improvement。
4. auditor / contradiction check 通過。
```

因此：

```text
softmax-cuda Mode B = SUCCESS
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

### 10.4 正式可宣稱結果

可以宣稱：

```text
softmax-cuda Mode B 建立了 shape-aware dispatch policy。
large slices 取得有效改善：
slice=784: 1.392x
slice=1024: 1.699x
slice=2048: 1.337x
```

不可宣稱：

```text
1. impl=3 是 universal kernel optimization。
2. impl=2 是 universal replacement。
3. slice=128/256 有有效 speedup。
4. profiler 支持 bottleneck 結論。
5. cached exp 單獨造成提升。
```

***

## 11. topk-cuda Supporting Evidence

`topk-cuda` 在本報告中不是 Mode B optimization 成果，而是 optional supporting evidence。

目前狀態：

```text
Mode B = BASELINE_ONLY
robust baseline completed
14 cases PASS
7 trials
12 VALID
2 CAUTION
0 NOISY
optimization rounds not started
```

其研究價值在於：

```text
1. Mode A 曾揭露 high-CV pseudo-speedup 風險。
2. Mode B robust baseline 以 7 trials 改善測量穩定性。
3. 這支持本研究對 variance / CV / paired baseline 的重視。
```

不可宣稱：

```text
topk-cuda Mode B 已完成 optimization
topk-cuda Mode B 有 speedup
```

***

## 12. shmembench-cuda Supporting Evidence

`shmembench-cuda` 同樣不是 Mode B optimization 成果，而是 optional supporting evidence。

目前狀態：

```text
Mode B = BASELINE_ONLY
official validated comparison = block_size=256, variant=original
diagnostic failures = block_size=128,512,1024
optimization rounds not started
```

其研究價值在於：

```text
1. low-optimization-space benchmark 需要先確認 correctness-valid comparison。
2. diagnostic failures 不應被刪除，也不應納入 official speedup。
3. 這證明 official sweep 必須依 correctness 收斂，而不能盲目追求完整 sweep。
```

不可宣稱：

```text
shmembench-cuda Mode B 已完成 optimization
diagnostic failures 代表整個 benchmark official failure
```

***

## 13. Mode C

### 13.1 Artifact status

```text
Mode C: NOT_STARTED / NOT_FOUND
```

目前沒有 Mode C artifact。

### 13.2 最佳協作方式

留白。

```text
尚未執行 Mode C，因此不對 Mode C 的最佳協作方式下結論。
```

### 13.3 加速程度

留白。

```text
尚未執行 Mode C，因此不對 Mode C 的加速程度下結論。
```

***

## 14. 風險與限制

### 14.1 Stale summaries

以下檔案過時，不應引用：

```text
phase3/softmax-cuda/mode_B_human_guided/agent_summary.md
phase3/softmax-cuda/mode_B_human_guided/results.csv
phase3/reports/mode_B_report.md
phase3/reports/MODE_B_REPORT.md
```

### 14.2 Profiler 未執行

softmax Mode B final confirmation 中：

```text
profiler_status = NOT_RUN
```

因此不得宣稱 profiler-supported bottleneck conclusion。

### 14.3 `impl=2` 是 compound candidate

`impl=2` 同時改變：

```text
warp-per-slice → block-per-slice
recompute exp → shared-memory cached exp
```

因此不能把 large-slice improvement 單獨歸因於 cached exponentials。

### 14.4 topk/shmembench 未完成 Mode B optimization

它們只能作為 supporting evidence，不能納入 Mode B speedup 主結果。

### 14.5 Mode C 尚未開始

不得寫 Mode C 有效或無效。

***

## 15. Do-Not-Claim List

正式報告不得宣稱：

```text
1. Mode C has started.
2. Mode C is effective.
3. topk-cuda Mode B optimization has begun or succeeded.
4. shmembench-cuda Mode B optimization has begun or succeeded.
5. impl=3 is a universal kernel optimization.
6. impl=2 is a universal replacement for all softmax shape sizes.
7. slice=128 or slice=256 has valid optimization speedup under impl=3.
8. profiler-supported bottleneck conclusions for Mode B.
9. shared-memory cached exponentials alone caused the large-slice speedup.
10. impl=0 → impl=1 is Phase 3 speedup.
```

***

## 16. 結論

本階段實驗證明，在 `softmax-cuda` 這個高優化空間 benchmark 上，Mode B human-in-the-loop workflow 能有效將 AI agent 產生的 partial candidate 收斂為可驗證、可審核、可解釋的工程策略。

具體而言，Agent 在 Round 1 提出 `impl=2` compound block-level cached-exp candidate。該 candidate 在 large slices 上有有效改善，但在 small slices 上 regression 或 correctness failure。人類審查拒絕其作為 universal replacement，並要求建立 shape-aware dispatch。Round 2 因此產生 `impl=3` dispatcher，對 `slice=128/256` 保留既有 `impl=1` optimized path，對 `slice=784/1024/2048` 選擇 `impl=2` path。Final confirmation 中所有 official cases 均通過 correctness，且大 slice 分別取得 1.392x、1.699x 與 1.337x 的有效改善。

因此，本研究目前最穩定的結論是：

```text
人機協作的價值不是讓 AI 一次產生完美 kernel，
而是讓 AI 產生的 partial improvement 經由人類審查、
baseline 對照、correctness gate 與 per-case 分析，
收斂成可驗證、可解釋、可放入論文的優化策略。
```

`softmax-cuda Mode B` 可標記為：

```text
SUCCESS
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

`topk-cuda` 與 `shmembench-cuda` 則暫列為 optional supporting evidence。Mode C 尚未執行，相關最佳協作方式與加速程度留白。
