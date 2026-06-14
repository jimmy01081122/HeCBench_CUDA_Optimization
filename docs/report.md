
# AI 輔助 HeCBench Benchmark 優化與人機協作工作流研究報告

## Phase 3 Softmax-focused Human-AI Collaborative Optimization Report

***

## 摘要

本研究以 HeCBench CUDA benchmark 為實驗平台，探討 AI agent 在既有 GPU benchmark 上進行程式優化時的能力、限制與可審核性。研究核心不是單純追求最高 speedup，而是分析 AI 在不同 benchmark 類型中何時能成功優化、何時失敗，以及失敗原因是否來自硬體瓶頸、測量噪音、correctness violation、baseline 無效、prompt 約束不足或結果分類錯誤。

本研究前期以多個 HeCBench CUDA benchmark 建立 benchmark taxonomy 與 prompt 約束分層。Phase 2 結果顯示，不同 benchmark 的 AI 輔助成果性質差異很大：softmax-cuda 屬於較明確的 kernel / AI primitive 優化案例；topk-cuda 的有效策略主要與 workspace reuse、block-size tuning 及 selection primitive 有關；p2p-cuda 屬 topology-aware measurement；allreduce-cuda 則主要是 launcher / UCX / environment fix，而非 kernel optimization。這說明若不區分 KERNEL\_OPT、PARAM\_TUNE、ENV\_FIX、MEASURE\_FIX 與 TOPOLOGY\_MEASURE，AI 結果容易被過度宣稱或錯誤分類。

Phase 2 也顯示，較完整的 prompt 通常包含 baseline、correctness、sbatch、submission limit、raw output preservation 等條款；這些條款能降低偽加速與不可審核結果的風險。不過，prompt inventory 也指出，多數 prompt 仍缺少 profiler 指標與重複統計要求，這成為 Phase 3 強化的核心方向。

Phase 3 原始設計選擇三個代表性 benchmark：softmax-cuda、topk-cuda 與 shmembench-cuda。其中 softmax-cuda 已完成 Mode B human-in-the-loop guided optimization 與 Mode C evidence-guided aggressive optimization；topk-cuda 與 shmembench-cuda 則完成 robust baseline，但尚未進入 Mode B optimization 或 Mode C optimization。因此，本報告將 softmax-cuda 作為主要成果，topk-cuda 與 shmembench-cuda 僅作為 optional supporting evidence。

本報告主要成果有兩項。

第一，softmax-cuda Mode B 證明，人類審查能將 AI agent 產生的 partial candidate 收斂為可驗證、可審核、可解釋的 shape-aware dispatch policy。Agent 在 Round 1 提出 impl=2 compound block-level cached-exp candidate，該候選版本在 large slices 上有效，但在 small slices 上出現 regression 或 correctness failure。人類審查拒絕將其作為 universal replacement，並要求建立 shape-aware dispatch。Round 2 產生 impl=3 dispatcher，對 slice=128/256 保留既有 impl=1 optimized path，對 slice=784/1024/2048 選擇 impl=2 path。Final confirmation 中所有 official cases 均 3/3 correctness PASS，大 slice 分別取得 1.392x、1.699x 與 1.337x 的有效改善。本結果應分類為 PARAM\_TUNE / SHAPE\_AWARE\_DISPATCH，而非 universal KERNEL\_OPT。

第二，softmax-cuda Mode C 在 Mode B accepted candidate impl=3 之上，進一步測試 evidence-guided aggressive optimization。Mode C final accepted candidate 為 impl4\_shape\_specialized\_large\_reduce。Final confirmation 顯示，impl=4 在 slice=784 與 slice=1024 上分別取得 1.135540x 與 1.048740x 的 additional speedup\_vs\_impl3；slice=128、slice=256 與 slice=2048 不接受為 Mode C additional speedup claim。Mode C final label 為 SUCCESS\_WITH\_ADDITIONAL\_SPEEDUP，final confirmation status 為 CONFIRMED。Profiler evidence 僅支持有限的 resource observation，不支持 reduction structure、shared-memory footprint 或 cached-exp 的因果宣稱。Submission 2 的 impl=5 ablation 被判定為 BLOCKED，不得 promotion，也不支持 attribution claim。

***

## 1. 研究動機與核心問題

現有 GPU benchmark 通常涵蓋多種類型的 CUDA kernel，例如 memory bandwidth、shared memory、reduction、Top-K selection、softmax、multi-GPU communication 等。這些 benchmark 原本用於測量系統或程式效能，但也可以作為 AI 輔助程式優化的實驗平台。

本研究的起點是：若讓 AI agent 嘗試優化一組既有 benchmark，例如從 HeCBench 中選取多個 CUDA benchmark，AI 可能會在部分題目取得加速，也可能在部分題目失敗。然而，單純統計「幾題成功、幾題失敗」不足以形成有價值的研究。更重要的是回答：

1. AI 成功優化的 benchmark 有什麼共同特徵？
2. AI 失敗的 benchmark 是因為硬體瓶頸、測量不穩定、correctness 限制，還是 prompt 約束不足？
3. AI 是否會產生偽加速，例如刪除慢 case、使用 invalid baseline、忽略 correctness failure？
4. 人類操作者能否透過 prompt、baseline、correctness gate、variance check 與審查流程，將 AI 的 partial result 收斂為可用策略？

因此，本研究不只關心 AI 能否產生 speedup，更關心 AI 產生的結果是否可驗證、可重現、可審核。

***

## 2. 研究問題

本研究圍繞以下問題展開：

```text
RQ1: AI agent 在既有 CUDA benchmark 上能否產生有效優化？

RQ2: 成功與失敗的 benchmark 有什麼差異？

RQ3: AI 產生的 speedup 是否一定可信？

RQ4: Prompt 約束是否能降低偽加速與不可審核結果？

RQ5: 人類操作者能否將 AI 的 partial optimization 轉化為可用策略？

RQ6: 哪些成果應分類為 kernel optimization，哪些只是 measurement fix、environment fix 或 parameter tuning？

RQ7: Evidence-guided aggressive optimization 是否能在 Mode B accepted candidate 之上取得額外改善？
```

本報告目前主要回答兩個具體子問題：

1. 在 softmax-cuda 這個高優化空間 benchmark 上，人機協作是否能將 AI agent 的 partial candidate 收斂成可驗證的優化策略？
2. 在 Mode B accepted candidate 之上，Mode C 是否能透過 evidence-guided aggressive optimization 取得額外 per-slice speedup？

目前結果支持肯定答案，但限定條件是：

1. 使用 robust / paired baseline。
2. 所有 official cases 保留。
3. correctness gate 嚴格執行。
4. regression、measurement-equivalent 與 failure 不被隱藏。
5. 結果類型正確分類，不將 shape-aware dispatch 或 parameter tuning 誤稱為 universal kernel optimization。
6. Mode C additional speedup 必須以 speedup\_vs\_impl3 為主，不以 speedup\_vs\_impl1 取代。
7. profiler / ablation 只在證據足夠時支持機制解釋，不能補不存在的 causality。

***

## 3. 研究流程總覽

本研究分為三個階段。

### 3.1 Phase 1：Benchmark taxonomy

Phase 1 將 benchmark 分類為：

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

### 3.2 Phase 2：Prompt 約束分層

Phase 2 比較 P1 / P2 / P3 三種 prompt 約束強度：

```text
P1: 弱約束
P2: 中約束
P3: 強約束
```

Phase 2 的核心發現是：P3 不一定帶來最高 speedup，但能顯著提升結果的可審核性，尤其是在 baseline validity、correctness gate、raw output preservation、submission limit、CSV schema 與 contradiction check 方面。

### 3.3 Phase 3：人機協作式優化

Phase 3 原始設計選擇三個代表性 benchmark：

```text
softmax-cuda:
  高優化空間

topk-cuda:
  中優化空間

shmembench-cuda:
  低優化空間 / shared memory microbenchmark
```

目前 Phase 3 已完成：

1. 三題 robust baseline。
2. softmax-cuda Mode A。
3. softmax-cuda Mode B Round 1。
4. softmax-cuda Mode B Round 2。
5. softmax-cuda Mode B final confirmation。
6. softmax-cuda Mode C Submission 1。
7. softmax-cuda Mode C profiler analysis。
8. softmax-cuda Mode C Submission 2 ablation。
9. softmax-cuda Mode C final confirmation。

目前尚未完成：

1. topk-cuda Mode B optimization。
2. shmembench-cuda Mode B optimization。
3. topk-cuda Mode C。
4. shmembench-cuda Mode C。

因此，本報告收斂為 softmax-focused report；topk-cuda 與 shmembench-cuda 僅作為 supporting evidence。

***

## 4. 專案與 Artifact 狀態

### 4.1 目前專案狀態

| Benchmark       | Mode A                        | Mode B         | Mode C                             | 目前角色                         |
| --------------- | ----------------------------- | -------------- | ---------------------------------- | ---------------------------- |
| softmax-cuda    | SUCCESS                       | SUCCESS        | SUCCESS\_WITH\_ADDITIONAL\_SPEEDUP | 主成果                          |
| topk-cuda       | SUCCESS                       | BASELINE\_ONLY | NOT\_STARTED / NOT\_FOUND          | optional supporting evidence |
| shmembench-cuda | INVALID / diagnostic findings | BASELINE\_ONLY | NOT\_STARTED / NOT\_FOUND          | optional supporting evidence |

### 4.2 Authoritative files

本報告應以以下檔案作為主要證據來源。

Mode B authoritative files：

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

Mode C authoritative files：

```text
phase3/softmax-cuda/mode_C_literature_profiler/submission_1/results.csv
phase3/softmax-cuda/mode_C_literature_profiler/submission_1/summary.md
phase3/softmax-cuda/mode_C_literature_profiler/submission_1/auditor_report.csv
phase3/softmax-cuda/mode_C_literature_profiler/submission_1/contradiction_check.csv

phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_summary.csv
phase3/softmax-cuda/mode_C_literature_profiler/analysis_track_1/profiler_results/profiler_summary.md

phase3/softmax-cuda/mode_C_literature_profiler/submission_2/results.csv
phase3/softmax-cuda/mode_C_literature_profiler/submission_2/summary.md
phase3/softmax-cuda/mode_C_literature_profiler/submission_2/patch_summary.md

phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation/results.csv
phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation/summary.md
phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation/auditor_report.csv
phase3/softmax-cuda/mode_C_literature_profiler/final_confirmation/contradiction_check.csv
```

若 repository 中實際路徑略有不同，正式引用應以實際檔案路徑為準。

### 4.3 Stale / superseded / do-not-cite files

以下檔案目前不應作為正式報告證據：

```text
phase3/softmax-cuda/mode_B_human_guided/agent_summary.md
phase3/softmax-cuda/mode_B_human_guided/results.csv
phase3/reports/mode_B_report.md
phase3/reports/MODE_B_REPORT.md
phase3/mode_A_agent_only
```

原因：

1. parent-level agent\_summary.md 與 results.csv 仍停留在 robust baseline 或舊 schema。
2. mode\_B\_report.md / MODE\_B\_REPORT.md 未反映 softmax Round 1、Round 2 與 final confirmation。
3. phase3/mode\_A\_agent\_only 是 topk Mode A 的重複目錄。

### 4.4 Hostname conflict

先前摘要曾出現 gn1288.twcc.ai，經 artifact sync 檢查後確認這是文字摘要 typo。正式報告應採用：

```text
Round 1 / Round 2:
  gn1221.twcc.ai

Mode B final confirmation:
  gn1228.twcc.ai

Mode C final confirmation:
  gn1224.twcc.ai
```

***

## 5. 實驗規則與可審核性要求

Phase 3 結果依下列規則審查：

1. 所有 GPU benchmark execution 必須透過 sbatch。
2. 禁止 login node 直接執行 ./main 或任何 GPU benchmark binary。
3. correctness FAIL → result invalid。
4. baseline invalid → speedup=n/a。
5. improvement < 1% → MEASUREMENT\_EQUIVALENT，不得宣稱顯著加速。
6. high CV → speedup claim invalid unless remeasured。
7. ENV\_FIX / MEASURE\_FIX / TOPOLOGY\_MEASURE 不得報告為 KERNEL\_OPT。
8. profiler unavailable 是 limitation，不是 experiment failure。
9. P1/P2/P3 或 Mode A/B/C 比較必須使用相同 official cases。
10. 不得刪除 slow、failing 或 regressing cases。
11. Mode C additional speedup 必須以 speedup\_vs\_impl3 為 primary metric。
12. Profiler timing 不得用於 official speedup。

### 5.1 Result type classification

本報告使用以下分類：

```text
BASELINE
BASELINE_COMPARISON
MODE_B_BASELINE
MODE_C_CANDIDATE
KERNEL_OPT
PARAM_TUNE
SHAPE_AWARE_DISPATCH
MEASURE_FIX
ENV_FIX
TOPOLOGY_MEASURE
DIAGNOSTIC_ONLY
ABLATION_ONLY
MEASUREMENT_EQUIVALENT
REGRESSION
INVALID
INCONCLUSIVE
```

### 5.2 Correctness 與 measurement validity 解耦

```text
correctness_status:
  PASS / FAIL / PARTIAL / NOT_PROVIDED

measurement_validity:
  VALID / CAUTION / NOISY / INVALID / LIMITED / DIAGNOSTIC_FAIL

speedup_claim_valid:
  true / false
```

此設計避免把 correctness PASS 直接等同於 speedup 有效。

***

## 6. 成功、失敗與 Partial Success 的分類邏輯

AI 輔助 benchmark 優化不能只用「有沒有變快」判斷成功。本研究將結果分為：

1. 完整成功：所有 official cases correctness PASS，且相對有效 baseline 有穩定提升。
2. Partial success：部分 cases 有效改善，但另一些 cases regression 或 invalid。例如 softmax Mode B Round 1 的 impl=2。
3. Measurement-equivalent：效能差異小於 1%，不可宣稱顯著加速。例如 softmax Mode C final 中 slice=2048。
4. Invalid：correctness FAIL、baseline invalid、raw output 缺失或 official cases 不完整。
5. Environment / measurement fix：問題解決來自 launcher、UCX、NCCL、Slurm、timer 或測量方式修正，不應寫成 kernel optimization。
6. Diagnostic failure：該 case 失敗本身具有研究價值，但不參與 official speedup。例如 shmembench-cuda 中 block\_size=128/512/1024。
7. Blocked ablation：ablation candidate 由於 correctness failure、regression 或無法隔離變因，不支援 speedup 或 attribution claim。例如 Mode C Submission 2 的 impl=5。

這個分類使研究能分析：

```text
為什麼 AI 在某些 benchmark 成功？
為什麼 AI 在某些 benchmark 失敗？
失敗是否揭露了 benchmark 的硬體限制、correctness 限制或測量問題？
```

***

## 7. Phase 2 結果與啟示

Phase 2 結果顯示，AI agent 的優化成果必須依 benchmark 類型與結果性質分類。例如：

```text
softmax-cuda:
  明確 AI primitive / kernel-oriented optimization case。

topk-cuda:
  workspace reuse、block size tuning、radix selection 類問題。

p2p-cuda:
  topology-aware measurement，不應宣稱為顯著 kernel speedup。

allreduce-cuda:
  launcher / UCX / environment fix，不應寫成 kernel optimization。
```

這些分類說明，AI-assisted optimization 研究不能只比較 speedup 數值，還必須區分：

1. 修改了什麼？
2. 是否保持 correctness？
3. 是否使用有效 baseline？
4. 是否只是 environment / measurement 修復？
5. 是否可能是 measurement noise？

Phase 3 的 Mode B 與 Mode C 設計即是將 Phase 2 的防偽加速規則帶入人機協作與 evidence-guided optimization 流程。

***

## 8. Phase 3 Mode A：Agent-only Baseline 與風險揭露

Mode A 的主要價值不是優化，而是測量基準與風險揭露。

### 8.1 softmax-cuda Mode A

Mode A 建立 impl=0 naive reference 與 impl=1 existing optimized baseline 的差異。impl=1 明顯優於 impl=0，但這是既有 implementation 差異，不是 Phase 3 agent speedup。

因此後續規則明確定義：

```text
impl=0 = naive reference
impl=1 = existing optimized baseline
Mode B/C 的 baseline 必須是 impl=1
不得將 impl=0 → impl=1 計為 Phase 3 speedup
```

### 8.2 topk-cuda Mode A

Mode A 顯示 topk-cuda 可能在沒有程式修改的情況下，因 baseline CV 過高產生約 1.4x 表面加速。這是 pseudo-speedup，不是有效優化。

因此 topk 進入 Mode B 前必須先做 robust baseline remeasurement。

### 8.3 shmembench-cuda Mode A

Mode A 顯示 shmembench-cuda 的原始 block-size sweep 不完全可用。block\_size=256, variant=original 是 official validated comparison；128/512/1024 應保留為 diagnostic failures，不納入 official speedup。

***

## 9. Mode B Robust Baseline

### 9.1 softmax-cuda robust baseline

```text
official cases:
  slice=128, 256, 784, 1024, 2048

baseline implementation:
  impl=1

correctness:
  all PASS

measurement_validity:
  all VALID
```

### 9.2 topk-cuda robust baseline

```text
official cases:
  14 cases

trials:
  7

correctness:
  all PASS

measurement_validity:
  12 VALID, 2 CAUTION, 0 NOISY
```

topk-cuda 尚未進入 Mode B optimization，因此不得宣稱 topk Mode B speedup。

### 9.3 shmembench-cuda robust baseline

```text
official validated comparison:
  block_size=256, variant=original

diagnostic sweep:
  block_size=128, 512, 1024
```

shmembench-cuda 尚未進入 Mode B optimization，因此不得宣稱 shmembench Mode B speedup。

***

## 10. softmax-cuda Mode B Round 1：Partial Candidate

### 10.1 Candidate

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

因此，任何效能改善不能單獨歸因於 cached exponentials。

### 10.2 Round 1 結果

| slice | paired impl=1 mean ms | impl=2 mean ms | correctness        | result                        |
| ----: | --------------------: | -------------: | ------------------ | ----------------------------- |
|   128 |              0.135152 |       0.554750 | PASS 3/3           | REGRESSION                    |
|   256 |              0.323384 |       0.594147 | PASS 2/3, FAIL 1/3 | INVALID                       |
|   784 |              1.434026 |       1.108087 | PASS 3/3           | valid large-slice improvement |
|  1024 |              2.068956 |       1.300902 | PASS 3/3           | valid large-slice improvement |
|  2048 |              2.212359 |       1.680560 | PASS 3/3           | valid large-slice improvement |

### 10.3 Round 1 判定

```text
Round 1 = PARTIAL_SUCCESS
Full replacement = REJECTED
Large-slice candidate = KEEP
```

Round 1 證明：

```text
impl=2 不適合作為 universal replacement。
impl=2 對 large slices 有價值。
```

這一步也證明 human review 的重要性。若只看 large-slice speedup，agent 可能把 candidate 錯誤包裝為 full success。但人類審查拒絕 universal replacement，並要求建立 shape-aware dispatch。

***

## 11. softmax-cuda Mode B Round 2：Shape-aware Dispatch

### 11.1 Human-guided correction

Round 2 根據 Round 1 evidence 建立 dispatch policy：

| slice | selected impl | reason                             |
| ----: | ------------- | ---------------------------------- |
|   128 | impl=1        | Round 1 impl=2 regression          |
|   256 | impl=1        | Round 1 impl=2 correctness failure |
|   784 | impl=2        | Round 1 impl=2 PASS and faster     |
|  1024 | impl=2        | Round 1 impl=2 PASS and faster     |
|  2048 | impl=2        | Round 1 impl=2 PASS and faster     |

Candidate：

```text
impl3_shape_dispatch_impl1_small_impl2_large
```

### 11.2 Round 2 結果

| slice | selected impl | candidate mean ms | paired impl=1 mean ms | correctness | result\_type            |  speedup |
| ----: | ------------: | ----------------: | --------------------: | ----------- | ----------------------- | -------: |
|   128 |             1 |          0.135674 |              0.144732 | PASS        | MEASUREMENT\_EQUIVALENT | 1.066763 |
|   256 |             1 |          0.306408 |              0.305251 | PASS        | MEASUREMENT\_EQUIVALENT | 0.996224 |
|   784 |             2 |          1.107988 |              1.437362 | PASS        | PARAM\_TUNE             | 1.297273 |
|  1024 |             2 |          1.300765 |              2.082344 | PASS        | PARAM\_TUNE             | 1.600861 |
|  2048 |             2 |          1.670514 |              2.213330 | PASS        | PARAM\_TUNE             | 1.324940 |

### 11.3 Round 2 判定

```text
Round 2 = ACCEPT
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

Round 2 解決了 Round 1 的問題：

1. small slices 不再使用 impl=2，避免 regression / invalid。
2. large slices 保留 impl=2 的有效改善。
3. 所有 official cases correctness PASS。

***

## 12. softmax-cuda Mode B Final Confirmation

### 12.1 Final 設定

```text
Slurm job:
  949717

Node:
  gn1228.twcc.ai

Candidate:
  impl3_shape_dispatch_impl1_small_impl2_large

Profiler status:
  NOT_RUN
```

### 12.2 Final result table

| slice | dispatch | baseline mean ms | candidate mean ms |   speedup | correctness | result\_type            | speedup\_claim\_valid |
| ----: | -------- | ---------------: | ----------------: | --------: | ----------- | ----------------------- | --------------------- |
|   128 | impl=1   |         0.134869 |          0.134574 | 1.002197x | PASS        | MEASUREMENT\_EQUIVALENT | false                 |
|   256 | impl=1   |         0.321793 |          0.321505 | 1.000895x | PASS        | MEASUREMENT\_EQUIVALENT | false                 |
|   784 | impl=2   |         1.442716 |          1.036402 | 1.392043x | PASS        | PARAM\_TUNE             | true                  |
|  1024 | impl=2   |         2.104045 |          1.238443 | 1.698944x | PASS        | PARAM\_TUNE             | true                  |
|  2048 | impl=2   |         2.237452 |          1.672904 | 1.337466x | PASS        | PARAM\_TUNE             | true                  |

### 12.3 Final interpretation

Final confirmation 顯示：

1. 五個 official slices 均 3/3 correctness PASS。
2. slice=128/256 dispatch 到 impl=1，結果為 measurement-equivalent。
3. slice=784/1024/2048 dispatch 到 impl=2，且均為有效 large-slice improvement。

因此：

```text
softmax-cuda Mode B = SUCCESS
result_type = PARAM_TUNE / SHAPE_AWARE_DISPATCH
```

***

## 13. softmax-cuda Mode C：Evidence-guided Aggressive Optimization

### 13.1 Mode C 目標

Mode C 的定位是：

```text
Evidence-guided aggressive optimization
```

其目標是在 Mode B accepted candidate impl=3 之上，嘗試進一步取得 additional speedup，同時維持：

1. official cases 完整。
2. paired baseline。
3. correctness PASS。
4. repeated measurements。
5. raw stdout/stderr preservation。
6. auditor / contradiction checks。
7. per-slice analysis。
8. 不以 aggregate speedup 掩蓋 regression 或 measurement-equivalent case。

Mode C 的 primary comparison 是：

```text
impl=4 vs impl=3
```

secondary comparison 是：

```text
impl=4 vs impl=1
```

因此，Mode C 的成功與否以 speedup\_vs\_impl3 為主；speedup\_vs\_impl1 只能作 supporting context，不可取代 Mode C additional speedup 判定。

***

### 13.2 Submission 1：impl4\_shape\_specialized\_large\_reduce

Submission 1 candidate：

```text
impl4_shape_specialized_large_reduce
```

Submission 1 被接受為：

```text
ACCEPT_WITH_LIMITATIONS
```

其有效 additional speedup claim 僅限於：

```text
slice=784
slice=1024
```

Submission 1 後續經 final confirmation 固定為：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

不接受為 additional Mode C speedup 的 slices：

```text
slice=128
slice=256
slice=2048
```

其中 128 與 256 是 guardrail / measurement-equivalent rows；2048 雖 correctness PASS，但 final confirmation 的 speedup\_vs\_impl3=1.008239，低於 1% claim gate，因此分類為 MEASUREMENT\_EQUIVALENT。

***

### 13.3 Analysis Track：Profiler Evidence

Mode C 中進行了 analysis-only profiler run。Profiler timing 不作為 official timing，也不參與 speedup\_vs\_impl3 的計算。

Final profiler analysis 覆蓋：

```text
slice=784:  impl=3 vs impl=4
slice=1024: impl=3 vs impl=4
slice=2048: impl=3 vs impl=4
```

未 profile：

```text
slice=128
slice=256
```

Profiler 支持的有限觀察是：

```text
impl=4 相較 impl=3，在 784、1024、2048 三個 large slices 上均降低約 0.90 KB/block 的 dynamic shared memory 使用量，而 collected registers/thread 與 waves/SM 在相同 slice 內維持不變。
```

Profiler resource table：

| slice | impl | registers/thread | dynamic shared memory | static shared memory | waves/SM | profiler timing |
| ----: | ---: | ---------------: | --------------------: | -------------------: | -------: | --------------: |
|   784 |    3 |               18 |         4.16 KB/block |                    0 |   156.25 |         1.10 ms |
|   784 |    4 |               18 |         3.26 KB/block |                    0 |   156.25 |       938.11 us |
|  1024 |    3 |               18 |         5.12 KB/block |                    0 |   156.25 |         1.30 ms |
|  1024 |    4 |               18 |         4.22 KB/block |                    0 |   156.25 |         1.20 ms |
|  2048 |    3 |               18 |         9.22 KB/block |                    0 |    78.12 |         1.66 ms |
|  2048 |    4 |               18 |         8.32 KB/block |                    0 |    78.12 |         1.64 ms |

Profiler timing in this table is diagnostic only and is not official timing.

Profiler evidence classification：

```text
LIMITED_PROFILER_EVIDENCE
```

原因是 profiler 缺少下列指標：

1. memory throughput
2. warp execution efficiency
3. instruction mix
4. math / special-function indicators
5. stall / scheduler breakdown

因此 profiler 不能支持以下因果結論：

```text
dynamic shared memory reduction caused speedup
reduction structure caused speedup
cached-exp contribution
profiler-supported bottleneck conclusion
```

特別是，slice=2048 同樣出現 dynamic shared memory reduction，但 final speedup\_vs\_impl3=1.008239，未達 1% 門檻，因此不能宣稱 dynamic shared memory reduction alone 是 Mode C improvement 的原因。

***

### 13.4 Submission 2：impl5\_reduction\_structure\_ablation

Submission 2 candidate：

```text
impl5_reduction_structure_ablation
```

其目的為 partial reduction-structure ablation。它不是 final candidate，也不得作為 replacement。

Submission 2 final label：

```text
BLOCKED
```

結果摘要：

```text
slice=128:
  correctness PASS
  guardrail row
  MEASUREMENT_EQUIVALENT

slice=256:
  correctness PASS
  guardrail row
  MEASUREMENT_EQUIVALENT

slice=784:
  correctness FAIL
  INVALID

slice=1024:
  correctness failure observed
  INVALID

slice=2048:
  correctness PASS
  slower than impl=3 and impl=4
  REGRESSION / ABLATION_ONLY_WITH_REGRESSION
```

因此：

```text
impl=5 不得 promoted。
impl=5 不支援任何 speedup claim。
impl=5 不支援 reduction-structure causality。
impl=5 只能作為 blocked ablation artifact。
```

***

### 13.5 Mode C Final Confirmation

Final confirmation job：

```text
Slurm job:
  950691

Node:
  gn1224.twcc.ai

Candidate:
  impl4_shape_specialized_large_reduce

Profiler status for official timing rows:
  NOT_RUN

Final confirmation status:
  CONFIRMED
```

Final confirmation 比較：

```text
impl=1: baseline
impl=3: Mode B baseline
impl=4: Mode C candidate
```

Final confirmation 未執行：

```text
impl=5
```

Final result table：

| slice | impl=1 mean | impl=3 mean | impl=4 mean | speedup\_vs\_impl3 | speedup\_vs\_impl1 | correctness |       CV | result\_type            |
| ----: | ----------: | ----------: | ----------: | -----------------: | -----------------: | ----------- | -------: | ----------------------- |
|   128 |    0.137974 |    0.136672 |    0.136548 |           1.000906 |           1.010441 | PASS        | 0.005285 | MEASUREMENT\_EQUIVALENT |
|   256 |    0.305982 |    0.306187 |    0.305269 |           1.003007 |           1.002335 | PASS        | 0.002846 | MEASUREMENT\_EQUIVALENT |
|   784 |    1.448263 |    1.031688 |    0.908544 |           1.135540 |           1.594048 | PASS        | 0.000143 | MODE\_C\_CANDIDATE      |
|  1024 |    2.107634 |    1.240982 |    1.183307 |           1.048740 |           1.781139 | PASS        | 0.000211 | MODE\_C\_CANDIDATE      |
|  2048 |    2.238517 |    1.674963 |    1.661275 |           1.008239 |           1.347470 | PASS        | 0.000069 | MEASUREMENT\_EQUIVALENT |

Final confirmation auditor 與 contradiction checks 均為 PASS。Auditor 確認：

1. official cases 完整。
2. impl=1/3/4 rows 齊全。
3. raw stdout/stderr paths 存在。
4. correctness 在 speedup claim 前通過。
5. speedup\_vs\_impl3 存在於 impl=4 rows。
6. 128/256 未被宣稱為 speedup。
7. speedup\_vs\_impl1 未作為 Mode C 主指標。
8. profiler\_status=NOT\_RUN。
9. official\_timing\_used=true。
10. no impl5 promotion。
11. no profiler 或 mechanism causality overclaim。
12. no aggregate-only success hiding per-slice regression。

Final accepted claims：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

Rejected / not accepted claims：

```text
slice=128:
  no Mode C additional speedup claim

slice=256:
  no Mode C additional speedup claim

slice=2048:
  speedup_vs_impl3=1.008239 < 1.01
  MEASUREMENT_EQUIVALENT
```

Final Mode C label：

```text
SUCCESS_WITH_ADDITIONAL_SPEEDUP
```

Final confirmation status：

```text
CONFIRMED
```

***

### 13.6 Mode C Final Interpretation

Mode C 在 softmax-cuda 上完成 final confirmation。最終 accepted candidate 為：

```text
impl4_shape_specialized_large_reduce
```

Mode C 的有效 additional speedup claims 僅限於：

```text
slice=784:  1.135540x vs impl=3
slice=1024: 1.048740x vs impl=3
```

不接受為 Mode C additional speedup：

```text
slice=128
slice=256
slice=2048
```

Profiler evidence 為：

```text
LIMITED_PROFILER_EVIDENCE
```

Ablation evidence：

```text
impl=5 BLOCKED
```

Causal attribution：

```text
NOT_PROVEN
```

因此，Mode C 的最終結論是：

```text
Mode C 在 Mode B accepted candidate impl=3 之上，於 784 與 1024 兩個 large slices 取得可驗證的 additional speedup。但此結果不是 universal kernel optimization，也不支持 dynamic shared memory、reduction structure 或 cached-exp 的因果宣稱。
```

***

## 14. topk-cuda Optional Supporting Evidence

topk-cuda 在本報告中不是 Mode B 或 Mode C optimization 成果，而是 optional supporting evidence。

目前狀態：

```text
Mode B = BASELINE_ONLY
Mode C = NOT_STARTED / NOT_FOUND
robust baseline completed
14 cases PASS
7 trials
12 VALID
2 CAUTION
0 NOISY
optimization rounds not started
```

其研究價值在於：

1. Mode A 曾揭露 high-CV pseudo-speedup 風險。
2. Mode B robust baseline 以 7 trials 改善測量穩定性。
3. 這支持本研究對 variance / CV / paired baseline 的重視。

不可宣稱：

```text
topk-cuda Mode B 已完成 optimization
topk-cuda Mode B 有 speedup
topk-cuda Mode C 已完成 optimization
```

***

## 15. shmembench-cuda Optional Supporting Evidence

shmembench-cuda 同樣不是 Mode B 或 Mode C optimization 成果，而是 optional supporting evidence。

目前狀態：

```text
Mode B = BASELINE_ONLY
Mode C = NOT_STARTED / NOT_FOUND
official validated comparison = block_size=256, variant=original
diagnostic failures = block_size=128,512,1024
optimization rounds not started
```

其研究價值在於：

1. low-optimization-space benchmark 需要先確認 correctness-valid comparison。
2. diagnostic failures 不應被刪除，也不應納入 official speedup。
3. 這證明 official sweep 必須依 correctness 收斂，而不能盲目追求完整 sweep。

不可宣稱：

```text
shmembench-cuda Mode B 已完成 optimization
shmembench-cuda Mode C 已完成 optimization
diagnostic failures 代表整個 benchmark official failure
```

***

## 16. 成功與失敗原因討論

### 16.1 softmax Mode B 成功原因

softmax-cuda Mode B 成功不是因為 AI 一次產生完美 kernel，而是因為人機協作流程能處理 partial success：

1. AI agent 產生 impl=2 compound candidate。
2. impl=2 在 large slices 有效，但在 small slices 失敗。
3. human review 拒絕 universal replacement。
4. human review 要求 shape-aware dispatch。
5. impl=3 將 partial success 轉成全 official sweep correctness PASS 的策略。

### 16.2 softmax Mode C 成功與限制

Mode C 在 Mode B accepted candidate impl=3 之上，進一步提出 impl=4，並在 784 與 1024 上取得 additional speedup。這證明 evidence-guided aggressive optimization 在 softmax-cuda 上仍有進一步空間。

但 Mode C 也呈現明確限制：

1. improvement 是 per-slice 的，不是 universal。
2. 128/256 沒有 additional speedup claim。
3. 2048 未達 1% additional-speedup gate。
4. profiler 只支持 limited resource observation。
5. impl=5 ablation 被 BLOCKED，無法支持 causal attribution。

### 16.3 topk 暫緩原因

topk-cuda 暫未進入 optimization，原因是 Mode A 曾揭露 high-CV pseudo-speedup 風險。它需要 robust baseline 與 variance filter，不能直接從表面 speedup 判斷 AI 是否成功。

### 16.4 shmembench 暫緩原因

shmembench-cuda 暫未進入 optimization，原因是 official comparison 需先依 correctness 收斂到 block\_size=256。其他 block sizes 應作 diagnostic failures，而非納入 official speedup。

### 16.5 人類審查角色

本研究顯示，人類操作者不是只負責看結果，而是負責：

1. 判斷 partial result 的有效邊界。
2. 防止錯誤歸因。
3. 防止偽加速。
4. 阻止 universal replacement 的過度宣稱。
5. 將 regression / invalid case 轉化為下一輪策略。
6. 控制 profiler 與 ablation 的解釋邊界。
7. 區分 timing-supported claim、resource observation 與 unsupported causality。

***

## 17. Threats to Validity

### 17.1 單一主案例限制

目前完整 Mode B 與 Mode C optimization 僅完成 softmax-cuda。因此不能外推為：

```text
Mode B / Mode C 對所有 benchmark 都有效。
```

更精確的結論是：

```text
在 softmax-cuda 這個高優化空間 benchmark 上，Mode B human-in-the-loop workflow 與 Mode C evidence-guided aggressive optimization 有效。
```

### 17.2 Mode B profiler 未執行

softmax Mode B final confirmation 中：

```text
profiler_status = NOT_RUN
```

因此不得宣稱 Mode B 有 profiler-supported bottleneck conclusion。

### 17.3 Mode C profiler evidence limited

Mode C profiler 提供有限 resource observation，但缺少：

```text
memory throughput
warp execution efficiency
instruction mix
math / special-function indicators
stall / scheduler breakdown
```

因此不能宣稱 profiler-supported causality。

### 17.4 impl=2 是 compound candidate

impl=2 同時改變 row parallelism 與 exp caching，因此不能把 large-slice improvement 單獨歸因於 cached exponentials。

### 17.5 impl=4 的 causality 未證明

impl=4 在 784 與 1024 上取得 additional speedup，但 profiler 與 ablation 都不足以證明原因。Final profiler 只顯示 dynamic shared memory reduction；但 2048 同樣有 dynamic shared memory reduction，卻未達有效 additional speedup。因此不能宣稱 dynamic shared memory reduction alone caused speedup。

### 17.6 Submission 2 ablation blocked

impl5\_reduction\_structure\_ablation 因 784/1024 correctness failure 與 2048 regression，被判定為 BLOCKED。因此它不能支持 reduction-structure attribution。

### 17.7 Optional benchmarks 未完成 optimization

topk-cuda 與 shmembench-cuda 目前只完成 robust baseline，因此不能作為 Mode B / Mode C speedup 主結果。

### 17.8 Stale summaries

部分 parent-level summary 檔案過時，不應引用。

***

## 18. Do-Not-Claim List

正式報告不得宣稱：

1. Mode B / Mode C 對所有 benchmarks 有效。
2. topk-cuda Mode B optimization has begun or succeeded。
3. shmembench-cuda Mode B optimization has begun or succeeded。
4. topk-cuda Mode C optimization has begun or succeeded。
5. shmembench-cuda Mode C optimization has begun or succeeded。
6. impl=3 is a universal kernel optimization。
7. impl=2 is a universal replacement for all softmax shape sizes。
8. impl=4 is a universal kernel optimization。
9. impl=4 improves all official slices。
10. slice=128 has valid optimization speedup under impl=3 or impl=4。
11. slice=256 has valid optimization speedup under impl=3 or impl=4。
12. slice=2048 has valid Mode C additional speedup。
13. profiler timing is official timing。
14. profiler-supported bottleneck conclusions for Mode B。
15. profiler proves dynamic shared memory reduction caused speedup。
16. profiler proves reduction structure caused speedup。
17. profiler proves cached-exp contribution。
18. shared-memory cached exponentials alone caused the large-slice speedup。
19. impl=5 ablation succeeded。
20. impl=5 supports attribution。
21. speedup\_vs\_impl1 is the main Mode C success metric。
22. impl=0 → impl=1 is Phase 3 speedup。
23. aggregate speedup can replace per-slice analysis。

***

## 19. 論文可用核心結論

可寫入論文或報告的核心結論如下：

```text
softmax-cuda 的 Mode B 實驗證明，人機協作流程能將 AI agent 產生的 partial optimization 轉化為可驗證的 shape-aware dispatch policy。Agent 在 Round 1 中提出的 compound block-level cached-exp candidate 對 large slices 有效，但在 small slices 上 regression 或 correctness failure。人類審查拒絕其作為 universal replacement，並引導 agent 在 Round 2 中建立 shape-aware dispatcher。Final confirmation 顯示，該 dispatcher 保留 impl=1 給 slice=128/256，並選擇 impl=2 給 slice=784/1024/2048，所有 official cases correctness PASS，large slices 分別取得 1.392x、1.699x 與 1.337x 有效改善。此結果證明，Mode B 的主要價值在於將 trial-and-error 的 AI 候選優化轉化為 correctness-gated、shape-aware、可審核的工程策略。

在此基礎上，Mode C 進一步測試 evidence-guided aggressive optimization 是否能超越 Mode B accepted candidate impl=3。Final confirmation 顯示，Mode C candidate impl4_shape_specialized_large_reduce 在 slice=784 與 slice=1024 上分別取得 1.135540x 與 1.048740x 的 additional speedup_vs_impl3。slice=128、slice=256 與 slice=2048 不接受為 Mode C additional speedup。Profiler 僅提供 limited resource observation，Submission 2 的 impl=5 ablation 被判定為 BLOCKED，因此本研究不對 reduction structure、shared-memory footprint 或 cached-exp 做因果宣稱。Mode C 的可接受結論是：在 softmax-cuda 上，evidence-guided aggressive optimization 能在部分 large slices 上進一步超越 Mode B candidate，但其有效範圍是 per-slice 且有限的。
```

***

## 20. 總結

本研究整體目標是利用 AI 協助優化既有 benchmark，並分析 AI 成功與失敗的原因。Phase 3 目前已完成 softmax-cuda 這一個代表性高優化空間 benchmark 的 Mode B 人機協作實驗與 Mode C evidence-guided aggressive optimization。

本階段最重要的結論不是「AI 產生了一個完美 kernel」，而是：

```text
AI 產生的 partial candidate 需要人類審查、baseline 對照、correctness gate、per-case 分析、profiler / ablation 邊界控管與 artifact auditing，才能收斂成可驗證、可解釋、可放入論文的優化策略。
```

在 softmax-cuda 中，這個過程具體表現為：

```text
Mode B Round 1:
  AI 提出 impl=2 compound candidate。
  large slices 有效，但 small slices 失敗。

Human review:
  拒絕 universal replacement。
  要求 shape-aware dispatch。

Mode B Round 2:
  建立 impl=3 dispatcher。
  small slices 使用 impl=1。
  large slices 使用 impl=2。

Mode B Final:
  全 official cases correctness PASS。
  large slices 取得 1.392x、1.699x 與 1.337x 有效改善。

Mode C Submission 1:
  建立 impl=4 candidate。
  在 784 與 1024 上超越 impl=3。

Mode C Analysis:
  profiler 提供 limited resource observation。
  dynamic shared memory 減少被觀察到，但 causality 未證明。

Mode C Submission 2:
  impl=5 ablation BLOCKED。
  不支持 speedup 或 attribution。

Mode C Final:
  impl=4 confirmed。
  784 與 1024 取得 1.135540x 與 1.048740x additional speedup_vs_impl3。
  128、256、2048 不接受為 Mode C additional speedup。
```

因此，目前可正式標記：

```text
softmax-cuda Mode B:
  SUCCESS

Mode B result_type:
  PARAM_TUNE / SHAPE_AWARE_DISPATCH

Mode B accepted candidate:
  impl3_shape_dispatch_impl1_small_impl2_large

softmax-cuda Mode C:
  COMPLETE

Mode C final label:
  SUCCESS_WITH_ADDITIONAL_SPEEDUP

Mode C accepted candidate:
  impl4_shape_specialized_large_reduce

Mode C accepted additional speedup:
  slice=784:  1.135540x vs impl=3
  slice=1024: 1.048740x vs impl=3

No Mode C speedup claim:
  slice=128
  slice=256
  slice=2048

Profiler:
  LIMITED_PROFILER_EVIDENCE

Ablation:
  impl=5 BLOCKED

Causal attribution:
  NOT_PROVEN

topk-cuda:
  optional supporting evidence only

shmembench-cuda:
  optional supporting evidence only
```

本報告的中心主張是：

```text
人機協作的價值不在於讓 AI 一次產生完美解，而在於讓 AI 產生的候選方向，經由人類審查、paired baseline、correctness gate、per-case analysis、profiler / ablation 邊界控管與 artifact auditing，轉化為穩定、可驗證、可放入論文的工程策略。
```

在 softmax-cuda 中，Mode B 將 AI agent 的 partial candidate 收斂為 shape-aware dispatch policy；Mode C 則在 Mode B accepted candidate 之上，於 784 與 1024 兩個 large slices 取得額外且經 final confirmation 的 speedup。此結果支持人機協作與 evidence-guided optimization 的價值，但也顯示有效改善具有 shape-specific 範圍，且沒有 profiler 或 ablation 證據可支持機制層級的因果歸因。
